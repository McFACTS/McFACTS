#!/usr/bin/env python3
######## Imports ########
import numpy as np
import os
import sys
from os.path import expanduser, join, isfile, isdir, basename
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from basil_core.astro.relations import Neumayer_early_NSC_mass, Neumayer_late_NSC_mass
from basil_core.astro.relations import SchrammSilvermanSMBH_mass_of_GSM as SMBH_mass_of_GSM
from mcfacts.physics.point_masses import time_of_orbital_shrinkage
from mcfacts.physics.point_masses import orbital_separation_evolve_reverse
from mcfacts.physics.point_masses import si_from_r_g, r_g_from_units
from mcfacts.inputs.ReadInputs import ReadInputs_ini
from mcfacts.inputs.ReadInputs import construct_disk_pAGN, construct_disk_direct

######## Constants ########
smbh_mass_fiducial = 1e8 * u.solMass
test_mass = 10 * u.solMass
inner_disk_outer_radius_fiducial = si_from_r_g(smbh_mass_fiducial,50.) #50 r_g

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mstar-min", default=1e9, type=float,
        help="Minimum galactic stellar mass")
    parser.add_argument("--mstar-max", default=1e13, type=float,
        help="Maximum galactic stellar mass")
    parser.add_argument("--mbins", nargs="+",help="Stellar mass bin labels")
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
    opts.plot_disk_exe=os.path.abspath(opts.plot_disk_exe)
    opts.vera_plots_exe=os.path.abspath(opts.vera_plots_exe)
    # Check exe
    assert isfile(opts.mcfacts_exe)
    # Check nbins
    opts.nbins = np.size(opts.mbins)
    return opts

######## Physics ########
def capture_time(
        smbh_mass,
        nsc_mass,
        agn_lifetime,
        disk_surf_dens_func,
        nsc_ratio_bh_num_star_num,
        nsc_ratio_bh_mass_star_mass,
        m_bh = 10 * u.solMass,
    ):
    print(f"smbh_mass: {smbh_mass}")
    print(f"nsc_mass: {nsc_mass}")
    print(f"agn_lifetime: {agn_lifetime}")
    print(f"nsc_ratio_bh_num_star_num: {nsc_ratio_bh_num_star_num}")
    print(f"nsc_ratio_bh_mass_star_mass: {nsc_ratio_bh_mass_star_mass}")
    print(f"m_bh: {m_bh}")
    # velocity dispersion of nsc
    sig_nsc = (2.3*(u.km/u.s)) * \
        (smbh_mass / (1 * u.solMass))**(1./4.38)
    print(f"sig_nsc: {sig_nsc}")
    # Calculate radius of influence
    r_infl = const.G * smbh_mass / sig_nsc**2
    r_infl = r_infl.to("pc")
    print(f"r_infl: {r_infl}")
    # Calculate radius_of_influence in r_g
    r_infl_g = r_g_from_units(
        smbh_mass,
        r_infl
    )
    print(f"r_infl_g: {r_infl_g}")
    # Calculate orbital time at radius of influence
    p_orb_r_infl = 2 * np.pi * np.sqrt(r_infl**3 / (const.G * smbh_mass))
    p_orb_r_infl = p_orb_r_infl.si
    print(f"p_orb_r_infl: {p_orb_r_infl}")
    # Calculate the surface density at the radius of influence
    Sigma_m = disk_surf_dens_func(r_infl_g) * u.kg / u.m**2
    print(f"Sigma_m: {Sigma_m}")
    if not np.isfinite(Sigma_m):
        return cosmo.hubble_time
    # Total mass of BH in NSC
    total_mass_bh_in_nsc = nsc_mass * nsc_ratio_bh_num_star_num * nsc_ratio_bh_mass_star_mass
    print(f"total_mass_bh_in_nsc: {total_mass_bh_in_nsc}")
    # Mass fraction of BH in NSC
    f_bh = total_mass_bh_in_nsc / nsc_mass
    print(f"f_bh: {f_bh}")
    # Capture mass
    captured_mass = (2. * smbh_mass * f_bh * \
        (m_bh / smbh_mass) * \
        (Sigma_m *np.pi * r_infl**2 / smbh_mass) *\
        (agn_lifetime / p_orb_r_infl) \
    ).to("Msun")
    print(f"captured_mass: {captured_mass}")
    # Calculate number of bh in agn lifetime
    n_bh_capture = (captured_mass / m_bh)
    print(f"n_bh_capture: {n_bh_capture}")
    # Capture time
    t_capture = agn_lifetime / n_bh_capture
    print(f"t_capture: {t_capture}")
    print(f"log_10(t_capture [yr]): {np.log10(t_capture.to('yr').value)}")
    return t_capture


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

    # Load disk model:
    if mcfacts_input_variables["flag_use_pagn"]:
        disk_surf_dens_func, \
            disk_aspect_ratio_func, \
            disk_opacity_func, \
            sound_speed_func, \
            disk_density_func, \
            disk_pressure_grad_func, \
            disk_omega_func, \
            disk_surf_dens_func_log, \
            temp_func, \
            surf_dens_log10_derivative_func, \
            temp_log10_derivative_func, \
            pressure_log10_derivative_func, \
            disk_model_properties, \
        bonus_structures = construct_disk_pAGN(
                mcfacts_input_variables["disk_model_name"],
                smbh_mass,
                mcfacts_input_variables["disk_radius_outer"],
                mcfacts_input_variables["disk_alpha_viscosity"],
                mcfacts_input_variables["disk_bh_eddington_ratio"],
            )
    else:
        disk_surf_dens_func, \
            disk_aspect_ratio_func, \
            disk_opacity_func, \
            sound_speed_func, \
            disk_density_func, \
            disk_pressure_grad_func, \
            disk_omega_func, \
            disk_surf_dens_func_log, \
            temp_func, \
            surf_dens_log10_derivative_func, \
            temp_log10_derivative_func, \
            pressure_log10_derivative_func, \
            disk_model_properties = \
        construct_disk_direct(
                mcfacts_input_variables["disk_model_name"],
                mcfacts_input_variables["disk_radius_outer"],
            )

    
    # Check truncate opacity flag
    if opts.truncate_opacity and not opts.print_only:
        # Make sure pAGN is enabled
        if not mcfacts_input_variables["flag_use_pagn"]:
            raise NotImplementedError
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
        0. * u.m,
    )
    # Find the new inner_disk_outer_radius
    new_inner_disk_outer_radius = orbital_separation_evolve_reverse(
        mcfacts_input_variables["smbh_mass"] * u.solMass,
        test_mass,
        0 * u.m,
        t_gw_inner_disk,
    )
    # Estimate in r_g
    new_inner_disk_outer_radius_r_g = r_g_from_units(
        mcfacts_input_variables["smbh_mass"] * u.solMass,
        new_inner_disk_outer_radius,
    )
    cmd=f"sed --in-place 's/inner_disk_outer_radius =.*/inner_disk_outer_radius = {new_inner_disk_outer_radius_r_g}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)

    # Estimate new trap radius
    new_trap_radius = mcfacts_input_variables["disk_radius_trap"] * np.sqrt(
        smbh_mass_fiducial /
        (mcfacts_input_variables["smbh_mass"] * u.solMass)
    ) 
    cmd=f"sed --in-place 's/disk_radius_trap =.*/disk_radius_trap = {new_trap_radius}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)
    
    # Estimate a new capture radius
    new_capture_radius = mcfacts_input_variables["disk_radius_capture_outer"] * np.sqrt(
        smbh_mass_fiducial /
        (mcfacts_input_variables["smbh_mass"] * u.solMass)
    )
    cmd=f"sed --in-place 's/disk_radius_capture_outer =.*/disk_radius_capture_outer = {new_capture_radius}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)
    # Estimate the agn lifetime
    agn_lifetime = mcfacts_input_variables["timestep_duration_yr"] * u.yr * \
        mcfacts_input_variables["timestep_num"]
    # Estimate the capture time
    t_capture = capture_time(
        mcfacts_input_variables["smbh_mass"] * u.solMass,
        mcfacts_input_variables["nsc_mass"] * u.solMass,
        agn_lifetime,
        disk_surf_dens_func,
        mcfacts_input_variables["nsc_ratio_bh_num_star_num"],
        mcfacts_input_variables["nsc_ratio_bh_mass_star_mass"],
    )
    print(f"smbh_mass: {mcfacts_input_variables['smbh_mass'] * u.solMass}")
    print(f"smbh_mass: {smbh_mass}")
    print(f"capture time: {t_capture}")
    # velocity dispersion of nsc
    cmd=f"sed --in-place 's/capture_time_yr =.*/capture_time_yr = {t_capture.to('yr').value}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)

    # Open script
    mcfacts_script = fname_ini_local.rstrip("ini") + "sh"
    # Identify output file
    mcfacts_out = fname_ini_local.rstrip("ini") + "out"
    with open(mcfacts_script, 'w') as F:
        # Make all iterations
        cmd = "python3 %s --fname-ini %s --work-directory %s > %s\n"%(
            os.path.abspath(opts.mcfacts_exe), fname_ini_local, wkdir, mcfacts_out)
        print(cmd)
        F.writelines(cmd)
        # Make plots for all iterations
        cmd = "python3 %s --fname-mergers %s/output_mergers_population.dat --fname-nal %s --cdf bin_com chi_eff final_mass time_merge\n"%(
            opts.vera_plots_exe, wkdir, opts.fname_nal)
        print(cmd)
        F.writelines(cmd)
        # Make disk plots
        cmd = f"python3 {opts.plot_disk_exe} --fname-ini {fname_ini_local} --outdir {wkdir}\n"
        print(cmd)
        F.writelines(cmd)

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
        # Generate directories
        early_dir = join(opts.wkdir, 'early', opts.mbins[i])
        if not isdir(early_dir):
            os.mkdir(early_dir)
        late_dir = join(opts.wkdir, 'late', opts.mbins[i])
        if not isdir(late_dir):
            os.mkdir(late_dir)

        make_batch(opts, early_dir, smbh_mass, early_mass)
        make_batch(opts, late_dir,  smbh_mass, late_mass)

    return

######## Execution ########
if __name__ == "__main__":
    main()
