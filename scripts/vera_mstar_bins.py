#!/usr/bin/env python3
######## Imports ########
import numpy as np
import os
import sys
from os.path import expanduser, join, isfile, isdir, basename
from astropy import units
from basil_core.astro.relations import Neumayer_early_NSC_mass, Neumayer_late_NSC_mass
from basil_core.astro.relations import SchrammSilvermanSMBH_mass_of_GSM as SMBH_mass_of_GSM
from mcfacts.physics.point_masses import time_of_orbital_shrinkage
from mcfacts.physics.point_masses import orbital_separation_evolve_reverse
from mcfacts.physics.point_masses import si_from_r_g, r_g_from_units
from mcfacts.inputs.ReadInputs import ReadInputs_ini
from mcfacts.inputs.ReadInputs import construct_disk_pAGN

######## Constants ########
smbh_mass_fiducial = 1e8 * units.solMass
test_mass = 10 * units.solMass
inner_disk_outer_radius_fiducial = si_from_r_g(smbh_mass_fiducial,50.) #50 r_g

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
    parser.add_argument("--vera-plots-exe", default="./scripts/vera_plots.py", help="Path to Vera plots script")
    parser.add_argument("--plot-disk-exe", default="./scripts/plot_disk_properties.py")
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
    parser.add_argument("--truncate-opacity", action="store_true",
        help="Truncate disk at opacity cliff")
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
def make_batch(opts, wkdir, smbh_mass, nsc_mass):
    ## Early-type ##
    # identify output_mergers_population.dat
    outfile = join(wkdir, "output_mergers_population.dat")
    # Check if outfile exists
    outfile_exists = isfile(outfile)
    # Check for runs
    all_runs = []
    for item in os.listdir(wkdir):
        if isdir(join(wkdir, item)) and item.startswith("gal"):
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
        cmd = "rm -rf %s/*"%wkdir
        # Print the command
        print(cmd)
        # Check print_only
        if not opts.print_only:
            # Execute rm command
            os.system(cmd)
    else:
        # Nothing exists, and nothing needs to be forced
        pass
    
    ## Copy inifile
    # Identify local inifile
    fname_ini_local = join(wkdir, basename(opts.fname_ini))
    # Load command
    cmd = f"cp {opts.fname_ini} {fname_ini_local}"
    print(cmd)
    # Execute copy
    if not opts.print_only:
        os.system(cmd)
        # Reality check
        assert isfile(fname_ini_local)

    ## Regex for known assumptions ##
    # SMBH mass
    cmd=f"sed --in-place 's/smbh_mass =.*/smbh_mass = {smbh_mass}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only: os.system(cmd)
    # NSC mass
    cmd=f"sed --in-place 's/nsc_mass =.*/nsc_mass = {nsc_mass}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only: os.system(cmd)
    # timestep_num mass
    cmd=f"sed --in-place 's/timestep_num =.*/timestep_num = {opts.timestep_num}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only: os.system(cmd)
    # galaxy_num mass
    cmd=f"sed --in-place 's/galaxy_num =.*/galaxy_num = {opts.galaxy_num}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only: os.system(cmd)
    # bin_num_max mass
    cmd=f"sed --in-place 's/bin_num_max =.*/bin_num_max = {opts.bin_num_max}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only: os.system(cmd)

    # Read the inifile
    if not opts.print_only: 
        mcfacts_input_variables = ReadInputs_ini(fname_ini_local)
    else:
        mcfacts_input_variables = ReadInputs_ini(opts.fname_ini)
    
    # Check truncate opacity flag
    if opts.truncate_opacity and not opts.print_only:
        # Make sure pAGN is enabled
        if not mcfacts_input_variables["flag_use_pagn"]:
            raise NotImplementedError
        # Load pAGN disk model
        pagn_surf_dens_func, pagn_aspect_ratio_func, pagn_opacity_func, pagn_model, bonus_structures =\
            construct_disk_pAGN(
                mcfacts_input_variables["disk_model_name"],
                smbh_mass,
                mcfacts_input_variables["disk_radius_outer"],
                mcfacts_input_variables["disk_alpha_viscosity"],
                mcfacts_input_variables["disk_bh_eddington_ratio"],
            )
        # Load R and tauV
        pagn_R = bonus_structures["R"]
        pagn_tauV = bonus_structures["tauV"]
        # Find where tauV is greater than its initial value
        tau_drop_mask = (pagn_tauV < pagn_tauV[0]) & (np.log10(pagn_R) > 3)
        # Find the drop index
        tau_drop_index = np.argmax(tau_drop_mask)
        # Find the drop radius
        tau_drop_radius = pagn_R[tau_drop_index]

        # Modify the inifile once again
        # outer disk radius
        cmd=f"sed --in-place 's/disk_radius_outer =.*/disk_radius_outer = {tau_drop_radius}/' {fname_ini_local}"
        print(cmd)
        if not opts.print_only: os.system(cmd)
        # Print radius
        #print("np.log10(smbh_mass):", np.log10(mcfacts_input_variables["smbh_mass"]))
        #print("tau_drop_radius:", tau_drop_radius)
        #print("tau_drop_radius:", si_from_r_g(mcfacts_input_variables["smbh_mass"],tau_drop_radius))
        #print("tau_drop_radius:", si_from_r_g(mcfacts_input_variables["smbh_mass"],tau_drop_radius).to('pc'))
        #print("tau_drop_radius:", si_from_r_g(1409937948.5103269,12813.45465546737).to('pc'))
        #raise Exception


    # Rescale inner_disk_outer_radius
    # rescale 
    t_gw_inner_disk = time_of_orbital_shrinkage(
        smbh_mass_fiducial,
        test_mass,
        inner_disk_outer_radius_fiducial,
        0. * units.m,
    )
    # Find the new inner_disk_outer_radius
    new_inner_disk_outer_radius = orbital_separation_evolve_reverse(
        mcfacts_input_variables["smbh_mass"] * units.solMass,
        test_mass,
        0 * units.m,
        t_gw_inner_disk,
    )
    # Estimate in r_g
    new_inner_disk_outer_radius_r_g = r_g_from_units(
        mcfacts_input_variables["smbh_mass"] * units.solMass,
        new_inner_disk_outer_radius,
    )
    cmd=f"sed --in-place 's/inner_disk_outer_radius =.*/inner_disk_outer_radius = {new_inner_disk_outer_radius_r_g}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)

    # Estimate new trap radius
    new_trap_radius = mcfacts_input_variables["disk_radius_trap"] * np.sqrt(
        smbh_mass_fiducial /
        (mcfacts_input_variables["smbh_mass"] * units.solMass)
    ) 
    cmd=f"sed --in-place 's/disk_radius_trap =.*/disk_radius_trap = {new_trap_radius}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)

    # Make all iterations
    cmd = "python3 %s --fname-ini %s --work-directory %s"%(
        opts.mcfacts_exe, fname_ini_local, wkdir)
    print(cmd)
    if not opts.print_only:
        os.system(cmd)
    # Make plots for all iterations
    cmd = "python3 %s --fname-mergers %s/output_mergers_population.dat --fname-nal %s --cdf bin_com chi_eff final_mass time_merge"%(
        opts.vera_plots_exe, wkdir, opts.fname_nal)
    print(cmd)
    if not opts.print_only:
        os.system(cmd)
    # Make disk plots
    cmd = f"python3 {opts.plot_disk_exe} --fname-ini {fname_ini_local} --outdir {wkdir}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)

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

    ## Loop early-type galaxies
    for i in range(opts.nbins):
        # Extract values for this set of galaxies
        mstar = mstar_arr[i]
        smbh_mass = SMBH_arr[i]
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

        make_batch(opts, early_dir, smbh_mass, early_mass)
        make_batch(opts, late_dir,  smbh_mass, late_mass)

    return

######## Execution ########
if __name__ == "__main__":
    main()
