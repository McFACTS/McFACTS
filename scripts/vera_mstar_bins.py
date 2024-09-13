#!/usr/bin/env python3
######## Imports ########
import numpy as np
import os
import sys
from os.path import expanduser, join, isfile, isdir
from basil.relations import Neumayer_early_NSC_mass, Neumayer_late_NSC_mass
from basil.relations import SchrammSilvermanSMBH_mass_of_GSM as SMBH_mass_of_GSM

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mstar-min", default=1e9, type=float,
        help="Minimum galactic stellar mass")
    parser.add_argument("--mstar-max", default=1e13, type=float,
        help="Maximum galactic stellar mass")
    parser.add_argument("--nbins", default=9, type=int, help="Number of stellar mass bins")
    parser.add_argument("--bin_num_max", default=1000, help="Number of binaries allowed at once")
    parser.add_argument("--wkdir", default='./run_many', help="top level working directory")
    parser.add_argument("--mcfacts-exe", default="./scripts/mcfacts_sim.py", help="Path to mcfacts exe")
    parser.add_argument("--fname-ini", required=True, help="Path to mcfacts inifile")
    #parser.add_argument("--vera-plots-exe", default="./scripts/vera_plots.py", help="Path to Vera plots script")
    parser.add_argument("--fname-nal", default=join(expanduser("~"), "Repos", "nal-data", "GWTC-2.nal.hdf5" ),
        help="Path to Vera's data from https://gitlab.com/xevra/nal-data")
    parser.add_argument("--max-nsc-mass", default=1.e8, type=float,
        help="Maximum NSC mass (solar mass)")
    parser.add_argument("--timestep_num", default=100, type=int,
        help="Number of timesteps (10,000 yrs by default)")
    parser.add_argument("--galaxy_num", default=2, type=int,
        help="Number of iterations per mass bin")
    parser.add_argument("--scrub", action='store_true',
        help="Remove timestep data for individual runs as we go to conserve disk space.")
    parser.add_argument("--force", action="store_true",
        help="Force overwrite and rerun everything?")
    parser.add_argument("--print-only", action="store_true",
        help="Don't run anything. Just print the commands.")
    # Handle top level working directory
    opts = parser.parse_args()
    if not isdir(opts.wkdir):
        os.mkdir(opts.wkdir)
    assert isdir(opts.wkdir)
    opts.wkdir=os.path.abspath(opts.wkdir)
    # Check exe
    assert isfile(opts.mcfacts_exe)
    return opts

######## Batch ########
def make_batch(opts, wkdir, mcfacts_args, mass_smbh, mass_nsc):
    ## Early-type ##
    # identify output_mergers_population.dat
    outfile = join(wkdir, "output_mergers_population.dat")
    # Check if outfile exists
    outfile_exists = isfile(outfile)
    # Check for runs
    all_runs = []
    for item in os.listdir(wkdir):
        if isdir(join(wkdir, item)) and item.startswith("run"):
            all_runs.append(item)
    any_runs = len(all_runs) > 0

    # Check force
    if opts.force:
        # remove whole wkdir
        cmd = "rm -rf %s"%wkdir
        # Print the command
        print(cmd)
        # Check print_only
        if not opts.print_only:
            # Execute rm command
            os.system(cmd)
    elif outfile_exists:
        # The outfile already exists.
        # We can move on
        print("%s already exists! skipping..."%(outfile),file=sys.stderr)
        return
    elif any_runs:
        # Some runs exist, but not an outfile. We can start these over
        # remove whole wkdir
        cmd = "rm -rf %s"%wkdir
        # Print the command
        print(cmd)
        # Check print_only
        if not opts.print_only:
            # Execute rm command
            os.system(cmd)
    else:
        # Nothing exists, and nothing needs to be forced
        pass

    # Make all iterations
    cmd = "python3 %s --fname-ini %s --smbh_mass %f --nsc_mass %f --work-directory %s %s"%(
        opts.mcfacts_exe, opts.fname_ini, mass_smbh, mass_nsc, wkdir, mcfacts_args)
    print(cmd)
    if not opts.print_only:
        os.system(cmd)
    # Make plots for all iterations
    #cmd = "python3 %s --fname-mergers %s/output_mergers_population.dat --fname-nal %s --cdf chi_eff chi_p M gen1 gen2 t_merge"%(
    #    opts.vera_plots_exe, wkdir, opts.fname_nal)
    #print(cmd)
    #if not opts.print_only:
    #    os.system(cmd)

    # Scrub runs
    if opts.scrub:
        cmd = "rm -rf %s/run*"%wkdir
        print(cmd)
        os.system(cmd)

######## Main ########
def main():
    # Load arguments
    opts = arg()
    # Get mstar array
    mstar_arr = np.logspace(np.log10(opts.mstar_min),np.log10(opts.mstar_max), opts.nbins)
    # Calculate SMBH and NSC mass 
    SMBH_arr = SMBH_mass_of_GSM(mstar_arr)
    NSC_early_arr = Neumayer_early_NSC_mass(mstar_arr)
    NSC_late_arr = Neumayer_late_NSC_mass(mstar_arr)
    # Limit NSC mass to maximum value
    NSC_early_arr[NSC_early_arr > opts.max_nsc_mass] = opts.max_nsc_mass
    NSC_late_arr[NSC_late_arr > opts.max_nsc_mass] = opts.max_nsc_mass
    # Create directories for early and late-type runs
    if not isdir(join(opts.wkdir, 'early')):
        os.mkdir(join(opts.wkdir, 'early'))
    if not isdir(join(opts.wkdir, 'late')):
        os.mkdir(join(opts.wkdir, 'late'))

    # Initialize mcfacts arguments dictionary
    mcfacts_arg_dict = {
                        "--timestep_num"    : opts.timestep_num,
                        "--galaxy_num"      : opts.galaxy_num,
                        "--fname-log"       : "out.log",
                        "--bin_num_max"     : opts.bin_num_max,
                       }
    mcfacts_args = ""
    for item in mcfacts_arg_dict:
        mcfacts_args = mcfacts_args + "%s %s "%(item, str(mcfacts_arg_dict[item]))

    ## Loop early-type galaxies
    for i in range(opts.nbins):
        # Extract values for this set of galaxies
        mstar = mstar_arr[i]
        mass_smbh = SMBH_arr[i]
        early_mass = NSC_early_arr[i]
        late_mass = NSC_late_arr[i]
        # Generate label for this mass bin
        mstar_str = "%.8f"%np.log10(mstar)
        # Generate directories
        early_dir = join(opts.wkdir, 'early', mstar_str)
        if not isdir(early_dir):
            os.mkdir(early_dir)
        late_dir = join(opts.wkdir, 'late', mstar_str)
        if not isdir(late_dir):
            os.mkdir(late_dir)

        make_batch(opts, early_dir, mcfacts_args, mass_smbh, early_mass)
        make_batch(opts, late_dir,  mcfacts_args, mass_smbh, late_mass)

    return

######## Execution ########
if __name__ == "__main__":
    main()
