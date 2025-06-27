#!/usr/bin/env python3
import os
import warnings
from importlib import resources as impresources
from os.path import isfile, isdir
from pathlib import Path
import time

import numpy as np
from astropy import units as u

from mcfacts.physics.binary import evolve
from mcfacts.physics.binary import formation
from mcfacts.physics.binary import merge

from mcfacts.physics import accretion
from mcfacts.physics import disk_capture
from mcfacts.physics import dynamics
from mcfacts.physics import eccentricity
from mcfacts.physics import emri
from mcfacts.physics import tde
from mcfacts.physics import feedback
from mcfacts.physics import gw
from mcfacts.physics import migration
from mcfacts.physics import stellar_interpolation
#from mcfacts.physics import star_interactions
from mcfacts.physics import point_masses
from mcfacts.physics import lum
from mcfacts.physics import analytical_velo

from mcfacts.inputs import ReadInputs
from mcfacts.inputs import data as input_data
from mcfacts.mcfacts_random_state import reset_random
from mcfacts.objects.agnobject import AGNBlackHole, AGNBinaryBlackHole, AGNMergedBlackHole, AGNStar, AGNMergedStar, AGNExplodedStar, AGNFilingCabinet
from mcfacts.setup import setupdiskblackholes, setupdiskstars, initializediskstars
from mcfacts.outputs import merger_cols, binary_cols
from mcfacts.outputs import emri_cols, bh_surviving_cols, \
    population_cols, binary_gw_cols, stars_cols, stars_explode_cols, \
    tde_cols, stars_merge_cols

#binary_field_names = "bin_orb_a1 bin_orb_a2 mass1 mass2 spin1 spin2 theta1 theta2 sep bin_com time_gw merger_flag time_mgr  gen_1 gen_2  bin_ang_mom bin_ecc bin_incl bin_orb_ecc nu_gw h_bin"
# This one isn't used anywhere

# Do not change this line EVER
DEFAULT_INI = impresources.files(input_data) / "model_choice.ini"
# Feature in testing do not use unless you know what you're doing.

assert DEFAULT_INI.is_file()

FORBIDDEN_ARGS = [
    "disk_radius_outer",
    "disk_radius_max_pc",
    "disk_radius_inner",
    ]


def arg():
    import argparse
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--bin_num_max", default=1000, type=int)
    parser.add_argument("--fname-ini", help="Filename of configuration file",
                        default=DEFAULT_INI, type=str)
    parser.add_argument("--fname-output-mergers", default="output_mergers.dat",
                        help="output merger file (if any)", type=str)
    parser.add_argument("--fname-output", default="output.dat",
                        help="output file (if any)", type=str)
    parser.add_argument("--fname-snapshots-bh",
                        default="output_bh_[single|binary]_pro_$(timestep_current_num).dat",
                        help="output of BH snapshot file ")
    parser.add_argument("--save-snapshots", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-w", "--work-directory",
                        default=Path().parent.resolve(),
                        help="Set the working directory for saving output. Default: current working directory",
                        type=str
                        )
    parser.add_argument("--seed", type=int, default=None,
                        help="Set the random seed. Randomly sets one if not passed. Default: None")
    parser.add_argument("--fname-log", default="mcfacts.log", type=str,
                        help="Specify a file in which to save the arguments and some runtime information. Default: mcfacts.log")

    #### Begin Argparse Nonsense
    #### Please do not modify between this line and the END Argparse Nonsense
    ####  line unless you have a good reason.
    # Add inifile arguments
    # Read default inifile
    _variable_inputs = ReadInputs.ReadInputs_ini(DEFAULT_INI, False)
    # Loop the arguments
    for name in _variable_inputs:
        # Skip CL read of forbidden arguments
        if name in FORBIDDEN_ARGS:
            continue
        _metavar = name
        _opt = "--%s" % (name)
        _default = _variable_inputs[name]
        _dtype = type(_variable_inputs[name])
        parser.add_argument(_opt,
                            default=None,
                            type=_dtype,
                            metavar=_metavar,
                            )


    # Parse arguments, using the opts namespace
    # This causes the default inifile values to be overwritten
    # by CLI arguments
    opts = parser.parse_args()
    # Check that the inifile exists
    assert isfile(opts.fname_ini)
    # Convert to path objects
    opts.fname_ini = Path(opts.fname_ini)
    assert opts.fname_ini.is_file()

    # Parse inifile
    print("opts.fname_ini", opts.fname_ini)
    # Read inifile
    variable_inputs = ReadInputs.ReadInputs_ini(opts.fname_ini, opts.verbose)
    print("variable_inputs", variable_inputs)
    # Hidden variable inputs
    print("_variable_inputs", _variable_inputs)
    # Okay, this is important. The priority of input arguments is:
    # command line > specified inifile > default inifile
    for name in variable_inputs:
        # Check for args not in parser. These were generated or changed in ReadInputs.py
        if not hasattr(opts, name):
            setattr(opts, name, variable_inputs[name])
            continue
        # Check for args not in the default_ini file
        if getattr(opts, name) is not None:
            # This is the case where the user has set the value of an argument
            # from the command line. We don't want to argue with the user.
            pass
        else:
            # This is the case where the user has not set the value of an
            # argument from the command line.
            # We can overwrite the default value with the inifile value
            setattr(opts, name, variable_inputs[name])
    # Case 3: if an attribute is in the default infile,
    #   and not the specified inifile,
    #   it remains unaltered.
    #### End Argparse Nonsense ####

    # Update paths of arguments to Path type
    opts.fname_snapshots_bh = Path(opts.fname_snapshots_bh)
    opts.fname_output_mergers = Path(opts.fname_output_mergers)
    opts.fname_output = Path(opts.fname_output)
    if opts.verbose:
        for item in opts.__dict__:
            print(item, getattr(opts, item))
    print("variable_inputs", variable_inputs)
    # Get the user-defined or default working directory / output location
    opts.work_directory = Path(opts.work_directory).resolve()
    if not isdir(opts.work_directory):
        os.mkdir(opts.work_directory)
    assert opts.work_directory.is_dir()
    try:  # check if working directory for output exists
        os.stat(opts.work_directory)
    except FileNotFoundError as e:
        raise e
    print(f"Output will be saved to {opts.work_directory}")

    # Get the parent path to this file and cd to that location for runtime
    opts.runtime_directory = Path(__file__).parent.resolve()
    assert opts.runtime_directory.is_dir()
    os.chdir(opts.runtime_directory)

    # set the seed for random number generation and reproducibility if not user-defined
    if opts.seed is None:
        opts.seed = np.random.randint(low=0, high=int(1e9), dtype=np.int_)
        print(f'Random number generator seed set to: {opts.seed}')

    # Check ISCO
    if opts.inner_disk_outer_radius < opts.disk_inner_stable_circ_orb:
        warnings.warn(
            "Warning: inner_disk_outer_radius < disk_inner_stable_circ_orb;\n" +\
            "Setting opts.inner_disk_outer_radius = disk_inner_stable_circ_orb"
        )
        opts.inner_disk_outer_radius = opts.disk_inner_stable_circ_orb

    # Write parameters to log file
    with open(opts.work_directory / opts.fname_log, 'w') as F:
        for item in opts.__dict__:
            # Convert booleans to integers
            if opts.__dict__[item] == False:
                line = "%s = %s\n" % (item, 0)
            elif opts.__dict__[item] == True:
                line = "%s = %s\n" % (item, 1)
            else: # everything else
                line = "%s = %s\n" % (item, str(opts.__dict__[item]))
            F.write(line)
    return opts

def main():
    """
    """
    tic_perf = time.perf_counter()
    # Setting up automated input parameters
    # see IOdocumentation.txt for documentation of variable names/types/etc.
    opts = arg()
    # Disk surface density (in kg/m^2) is a function of radius, where radius is in r_g
    # Disk aspect ratio is a function of radius, where radius is in r_g
    # Disk opacity ...
    # Disk sound speed [m/s] is a function of radius, where radius is in r_g
    # Disk density [kg/m^3] is a function of radius, where radius is in r_g
    # Return disk log of disk surface density as a function of log (R)
    disk_surface_density, disk_aspect_ratio, disk_opacity, disk_sound_speed, disk_density, disk_pressure_grad, disk_omega, disk_surface_density_log, temp_func, disk_dlog10surfdens_dlog10R_func, disk_dlog10temp_dlog10R_func, disk_dlog10pressure_dlog10R_func = \
        ReadInputs.construct_disk_interp(opts.smbh_mass,
                                         opts.disk_radius_outer,
                                         opts.disk_model_name,
                                         opts.disk_alpha_viscosity,
                                         opts.disk_bh_eddington_ratio, 
                                         disk_radius_max_pc=opts.disk_radius_max_pc,
                                         flag_use_pagn=opts.flag_use_pagn,
                                         verbose=opts.verbose
                                         )

    blackholes_merged_pop = AGNMergedBlackHole()
    emris_pop = AGNBlackHole()
    blackholes_binary_gw_pop = AGNBinaryBlackHole()
    stars_pop = AGNStar()
    tdes_pop = AGNStar()
    stars_plunge_pop = AGNStar()
    stars_explode_pop = AGNExplodedStar()
    stars_merge_pop = AGNMergedStar()

    # Setting up arrays to keep track of how much mass is cycled through stars
    disk_arr_galaxy = []
    disk_arr_timestep_pop = np.array([])
    disk_arr_mass_lost_pop = np.array([])
    disk_arr_mass_gained_pop = np.array([])

    # tdes_pop = AGNStar()

    print("opts.__dict__", opts.__dict__)
    print("opts.smbh_mass", opts.smbh_mass)
    print("opts.fraction_bin_retro", opts.fraction_bin_retro)

    for galaxy in range(opts.galaxy_num):
        print("Galaxy", galaxy)
        # Set random number generator for this run with incremented seed
        rng = reset_random(opts.seed+galaxy)

        # Make subdirectories for each galaxy
        # Fills run number with leading zeros to stay sequential
        galaxy_zfilled_str = f"{galaxy:>0{int(np.log10(opts.galaxy_num))+1}}"
        try:  # Make subdir, exit if it exists to avoid clobbering.
            os.makedirs(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}"), exist_ok=False)
        except FileExistsError:
            raise FileExistsError(f"Directory \'gal{galaxy_zfilled_str}\' exists. Exiting so I don't delete your data.")

        # Housekeeping for array initialization
        blackholes_binary = AGNBinaryBlackHole()
        blackholes_binary_gw = AGNBinaryBlackHole()
        blackholes_merged = AGNMergedBlackHole()

        # Fractional rate of mass growth per year set to
        # the Eddington rate(2.3e-8/yr)
        disk_bh_eddington_mass_growth_rate = 2.3e-8
        # minimum spin angle resolution
        # (ie less than this value gets fixed to zero)
        # e.g 0.02 rad=1deg
        disk_bh_spin_resolution_min = 0.02
        agn_redshift = 0.1
        #------------------       HARDCODING agn_redshift = 0.1 HERE       -----------------------------------
        # This is for computing the gw strain for sources and NOTHING else if you are 
        #   not using our strain this parameter will do nothing. If you are using our strain and you want to put 
        #   your sources at a different distance, scale them to the value here DO NOT CHANGE 

        # Set up number of BH in disk
        disk_bh_num = setupdiskblackholes.setup_disk_nbh(
            opts.nsc_mass,
            opts.nsc_ratio_bh_num_star_num,
            opts.nsc_ratio_bh_mass_star_mass,
            opts.nsc_radius_outer,
            opts.nsc_density_index_outer,
            opts.smbh_mass,
            opts.disk_radius_outer,
            opts.disk_aspect_ratio_avg,
            opts.nsc_radius_crit,
            opts.nsc_density_index_inner,
        )

        '''
        # Skip the whole galaxy if there are no black holes
        if disk_bh_num < 1:
            # Warn the user once, even if verbose is off
            warnings.warn("No black holes in the disk. Skipping galaxy %d."%(galaxy))
            # Warn the user more often if verbose is on
            if opts.verbose:
                print("No black holes in the disk. Skipping galaxy %d."%(galaxy))
            # Set total emris to zero
            if not "total_emris" in locals():
                total_emris = 0
            continue
        '''

        # generate initial BH parameter arrays
        print("Generate initial BH parameter arrays")
        bh_orb_a_initial = setupdiskblackholes.setup_disk_blackholes_location_NSC_powerlaw(
                disk_bh_num, opts.disk_radius_outer, opts.disk_inner_stable_circ_orb,
                opts.smbh_mass, opts.nsc_radius_crit, opts.nsc_density_index_inner,
                opts.nsc_density_index_outer, volume_scaling=True)
        bh_mass_initial = setupdiskblackholes.setup_disk_blackholes_masses(
                disk_bh_num,
                opts.nsc_imf_bh_mode, opts.nsc_imf_bh_mass_max, opts.nsc_imf_bh_powerlaw_index, opts.mass_pile_up)
        bh_spin_initial = setupdiskblackholes.setup_disk_blackholes_spins(
                disk_bh_num,
                opts.nsc_bh_spin_dist_mu, opts.nsc_bh_spin_dist_sigma)
        bh_spin_angle_initial = setupdiskblackholes.setup_disk_blackholes_spin_angles(
                disk_bh_num,
                bh_spin_initial)
        bh_orb_ang_mom_initial = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(
                disk_bh_num)
        if opts.flag_orb_ecc_damping == 1:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(disk_bh_num, opts.disk_bh_orb_ecc_max_init)
        else:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_circularized(disk_bh_num, opts.disk_bh_pro_orb_ecc_crit)

        bh_orb_inc_initial = setupdiskblackholes.setup_disk_blackholes_incl(disk_bh_num, bh_orb_a_initial, bh_orb_ang_mom_initial, disk_aspect_ratio)
        bh_orb_arg_periapse_initial = setupdiskblackholes.setup_disk_blackholes_arg_periapse(disk_bh_num)

        # Initialize black holes
        blackholes = AGNBlackHole(mass=bh_mass_initial,
                                  spin=bh_spin_initial,
                                  spin_angle=bh_spin_angle_initial,
                                  orb_ang_mom=bh_orb_ang_mom_initial,
                                  orb_a=bh_orb_a_initial,
                                  orb_inc=bh_orb_inc_initial,
                                  orb_ecc=bh_orb_ecc_initial,
                                  orb_arg_periapse=bh_orb_arg_periapse_initial,
                                  bh_num=disk_bh_num,
                                  galaxy=np.full(disk_bh_num, galaxy),
                                  time_passed=np.zeros(disk_bh_num))

        # Initialize filing_cabinet
        filing_cabinet = AGNFilingCabinet(id_num=blackholes.id_num,
                                          category=np.full(blackholes.num, 0),
                                          orb_a=blackholes.orb_a,
                                          mass=blackholes.mass,
                                          orb_ecc=blackholes.orb_ecc,
                                          size=np.full(blackholes.num, -1.5),
                                          )

        # Initialize stars
        if opts.flag_add_stars:
            stars, disk_star_num = initializediskstars.init_single_stars(opts, disk_aspect_ratio, galaxy, id_start_val=filing_cabinet.id_max+1)
        else:
            stars, disk_star_num = AGNStar(), 0

        print(f"{disk_bh_num} black holes, {disk_star_num} stars")
        filing_cabinet.add_objects(new_id_num=stars.id_num,
                                   new_category=np.full(stars.num, 1),
                                   new_orb_a=stars.orb_a,
                                   new_mass=stars.mass,
                                   new_orb_ecc=stars.orb_ecc,
                                   new_size=point_masses.r_g_from_units(opts.smbh_mass, (10 ** stars.log_radius) * u.Rsun).value,
                                   new_direction=np.zeros(stars.num),
                                   new_disk_inner_outer=np.zeros(stars.num))

        # Writing initial parameters to file
        if opts.flag_add_stars:
            stars.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/initial_params_star.dat"))
        blackholes.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/initial_params_bh.dat"))

        # Write torques stuff to file
        star_torque_array_radius_outer = stellar_interpolation.ratio_star_torques(disk_density, disk_pressure_grad, disk_aspect_ratio, disk_surface_density, disk_omega, opts.disk_radius_outer, opts.smbh_mass)
        np.savetxt(os.path.join(opts.work_directory) + "/star_torques_disk_radius_outer.dat", star_torque_array_radius_outer, header="drag_torque mig_torque ratio_torque v_phi v_kep v_rel")
        star_torque_array_radius_trap = stellar_interpolation.ratio_star_torques(disk_density, disk_pressure_grad, disk_aspect_ratio, disk_surface_density, disk_omega, opts.disk_radius_trap, opts.smbh_mass)
        np.savetxt(os.path.join(opts.work_directory) + "/star_torques_disk_radius_trap.dat", star_torque_array_radius_trap, header="drag_torque mig_torque ratio_torque v_phi v_kep v_rel")

        if (opts.flag_initial_stars_BH_immortal == 1):
            # Stars over disk_star_initial_mass_cutoff will explode and turn into BH

            star_to_bh_id_num = stars.id_num[stars.mass > opts.disk_star_initial_mass_cutoff]
            star_to_bh_spin = setupdiskblackholes.setup_disk_blackholes_spins(len(star_to_bh_id_num),
                                                                              opts.nsc_bh_spin_dist_mu, opts.nsc_bh_spin_dist_sigma)
            star_to_bh_spin_angle = setupdiskblackholes.setup_disk_blackholes_spin_angles(len(star_to_bh_id_num), star_to_bh_spin)
            star_to_bh_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(len(star_to_bh_id_num))
            star_to_bh_inc = setupdiskblackholes.setup_disk_blackholes_incl(len(star_to_bh_id_num), stars.at_id_num(star_to_bh_id_num, "orb_a"), star_to_bh_orb_ang_mom, disk_aspect_ratio)

            blackholes.add_blackholes(new_mass=stars.at_id_num(star_to_bh_id_num, "mass"),
                                      new_id_num=star_to_bh_id_num,
                                      new_orb_ang_mom=star_to_bh_orb_ang_mom,
                                      new_spin=star_to_bh_spin,
                                      new_spin_angle=star_to_bh_spin_angle,
                                      new_orb_a=stars.at_id_num(star_to_bh_id_num, "orb_a"),
                                      new_orb_inc=star_to_bh_inc,
                                      new_orb_ecc=stars.at_id_num(star_to_bh_id_num, "orb_ecc"),
                                      new_orb_arg_periapse=stars.at_id_num(star_to_bh_id_num, "orb_arg_periapse"),
                                      new_galaxy=stars.at_id_num(star_to_bh_id_num, "galaxy"),
                                      new_gen=stars.at_id_num(star_to_bh_id_num, "gen"),
                                      new_time_passed=stars.at_id_num(star_to_bh_id_num, "time_passed"))
            # Remove from stars array
            stars.remove_id_num(star_to_bh_id_num)
            # Update filing cabinet
            filing_cabinet.update(star_to_bh_id_num,
                                  "category",
                                  np.full(len(star_to_bh_id_num), 0))
            filing_cabinet.update(star_to_bh_id_num,
                                  "size",
                                  np.full(len(star_to_bh_id_num), -1.5))

        # Generate initial inner disk arrays for objects that end up in the inner disk. 
        # This is to track possible EMRIs--we're tossing things in these arrays
        #  that end up with semi-major axis < 50rg
        # Assume all drawn from prograde population for now.
        #   SF: Is this assumption important here? Where does it come up?

        # Test if any BH or BBH are in the danger-zone (<inner_disk_outer_radius, default =50r_g) from SMBH.
        # Potential EMRI/BBH EMRIs.
        # Find prograde BH in inner disk. Define inner disk as <=50r_g. 
        # Since a 10Msun BH will decay into a 10^8Msun SMBH at 50R_g in ~38Myr and decay time propto a^4.
        # e.g at 25R_g, decay time is only 2.3Myr.

        # Create empty EMRIs object
        blackholes_emris = AGNBlackHole()

        # Create empty TDEs object
        stars_tdes = AGNStar()

        # Create empty plunging stars object
        stars_plunge = AGNStar()

        # Create empty exploded stars object
        stars_explode = AGNExplodedStar()

        # Create empty merged stars object
        stars_merge = AGNMergedStar()

        # Find inner disk BH (potential EMRI)
        bh_id_num_inner_disk = blackholes.id_num[blackholes.orb_a < opts.inner_disk_outer_radius]
        blackholes_inner_disk = blackholes.copy()
        blackholes_inner_disk.keep_id_num(bh_id_num_inner_disk)

        # Remove inner disk BH from blackholes
        blackholes.remove_id_num(bh_id_num_inner_disk)

        # Update filing cabinet for inner disk BHs
        filing_cabinet.update(id_num=bh_id_num_inner_disk,
                              attr="disk_inner_outer",
                              new_info=np.full(bh_id_num_inner_disk.size, -1))

        # Update filing cabinet for outer disk BHs
        filing_cabinet.update(id_num=blackholes.id_num,
                              attr="disk_inner_outer",
                              new_info=np.ones(blackholes.id_num.size))

        # Find inner disk stars (potential TDEs)
        star_id_num_inner_disk = stars.id_num[stars.orb_a < opts.inner_disk_outer_radius]
        stars_inner_disk = stars.copy()
        stars_inner_disk.keep_id_num(star_id_num_inner_disk)

        # Remove inner disk stars from stars
        stars.remove_id_num(star_id_num_inner_disk)

        # Update filing cabinet for inner disk stars
        filing_cabinet.update(id_num=star_id_num_inner_disk,
                              attr="disk_inner_outer",
                              new_info=np.full(star_id_num_inner_disk.size, -1))

        # Update filing cabinet for outer disk stars
        filing_cabinet.update(id_num=stars.id_num,
                              attr="disk_inner_outer",
                              new_info=np.ones(stars.num))

        # Housekeeping: Set up time
        time_init = 0.0
        time_final = opts.timestep_duration_yr*opts.timestep_num

        # Find prograde BH orbiters. Identify BH with orb. ang mom > 0 (orb_ang_mom is only ever +1 or -1)
        bh_id_num_pro = blackholes.id_num[blackholes.orb_ang_mom > 0]
        blackholes_pro = blackholes.copy()
        blackholes_pro.keep_id_num(bh_id_num_pro)

        # Update filing cabinet and remove from blackholes
        blackholes.remove_id_num(blackholes_pro.id_num)
        filing_cabinet.update(id_num=blackholes_pro.id_num,
                              attr="direction",
                              new_info=np.ones(blackholes_pro.num))

        # Find prograde star orbiters.
        star_id_num_pro = stars.id_num[stars.orb_ang_mom > 0]
        stars_pro = stars.copy()
        stars_pro.keep_id_num(star_id_num_pro)

        # Update filing cabinet and remove from stars
        stars.remove_id_num(stars_pro.id_num)
        filing_cabinet.update(id_num=stars_pro.id_num,
                              attr="direction",
                              new_info=np.ones(stars_pro.num))

        # Find retrograde black holes
        bh_id_num_retro = blackholes.id_num[blackholes.orb_ang_mom < 0]
        blackholes_retro = blackholes.copy()
        blackholes_retro.keep_id_num(bh_id_num_retro)

        # Update filing cabinet and remove from blackholes
        blackholes.remove_id_num(blackholes_retro.id_num)
        filing_cabinet.update(id_num=blackholes_retro.id_num,
                              attr="direction",
                              new_info=np.full(blackholes_retro.num, -1))

        # Find retrograde stars
        star_id_num_retro = stars.id_num[stars.orb_ang_mom < 0]
        stars_retro = stars.copy()
        stars_retro.keep_id_num(star_id_num_retro)

        # Update filing cabinet and remove from stars
        stars.remove_id_num(stars_retro.id_num)
        filing_cabinet.update(id_num=stars_retro.id_num,
                              attr="direction",
                              new_info=np.full(stars_retro.num, -1))

        # Tracker for all binaries ever formed in this galaxy
        num_bbh_gw_tracked = 0

        # Set up normalization for t_gw (SF: I do not like this way of handling, flag for update)
        time_gw_normalization = merge.normalize_tgw(opts.smbh_mass, opts.inner_disk_outer_radius)
        print("Scale of t_gw (yrs)=", time_gw_normalization)

        # Multiple AGN episodes:
        # If you want to use the output of a previous AGN simulation as an input to another AGN phase
        # Make sure you have a file 'recipes/prior_model_name_population.dat' so that ReadInputs can take it in
        # and in your .ini file set switch prior_agn = 1.0.
        # Initial orb ecc is modified uniform using setup_disk_bh_orb_ecc_uniform(bh_pro_num,opts.disk_bh_orb_ecc_max_init)
        # SF: No promises this handles retrograde orbiters correctly yet
        '''
        if opts.flag_prior_agn == 1.0:

            prior_radii, prior_masses, prior_spins, prior_spin_angles, prior_gens \
                = ReadInputs.ReadInputs_prior_mergers()

            bh_pro_num = blackholes_pro.num

            prior_indices = setupdiskblackholes.setup_prior_blackholes_indices(bh_pro_num, prior_radii)
            prior_indices = prior_indices.astype('int32')
            blackholes_pro.keep_index(prior_indices)

            print("prior indices", prior_indices)
            print("prior locations", blackholes_pro.orb_a)
            print("prior gens", blackholes_pro.gen)
            blackholes_pro.orb_ecc = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(bh_pro_num, opts.disk_bh_orb_ecc_max_init)
            print("prior ecc", blackholes_pro.orb_ecc)
        '''

        # Set up arrays to keep track of mass cycled through disk
        disk_arr_timestep = []
        disk_arr_mass_lost = []
        disk_arr_mass_gained = []

        # Start Loop of Timesteps
        print("Start Loop!")
        time_passed = time_init
        print("Initial Time(yrs) = ", time_passed)

        timestep_current_num = 0

        while time_passed < time_final:
            # Record snapshots if user wishes
            if opts.save_snapshots == 1:

                blackholes_pro.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/output_bh_single_pro_{timestep_current_num}.dat"))
                blackholes_retro.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/output_bh_single_retro_{timestep_current_num}.dat"))
                if opts.flag_add_stars:
                    stars_pro.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/output_stars_single_pro_{timestep_current_num}.dat"))
                    stars_retro.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/output_stars_single_retro_{timestep_current_num}.dat"))
                blackholes_binary.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/output_bh_binary_{timestep_current_num}.dat"),
                                         cols=binary_cols)
                timestep_current_num += 1

            # Set up array to keep track of mass cycled in this timestep
            disk_mass_gained = []
            disk_mass_lost = []

            # Order of operations:
            # No migration until orbital eccentricity damped to e_crit
            # 1. check orb. eccentricity to see if any prograde_bh_location BH have orb. ecc. <e_crit.
            #    Create array prograde_bh_location_ecrit for those (mask prograde_bh_locations?)
            #       If yes, migrate those BH.
            #       All other BH, damp ecc and spin *down* BH (retrograde accretion), accrete mass.
            # 2. Run close encounters only on those prograde_bh_location_ecrit members.

            # Migrate
            # First if feedback present, find ratio of feedback heating torque to migration torque
            if opts.flag_thermal_feedback > 0:
                ratio_heat_mig_torques = feedback.feedback_bh_hankla(
                    blackholes_pro.orb_a,
                    disk_surface_density,
                    disk_opacity,
                    opts.disk_bh_eddington_ratio,
                    opts.disk_alpha_viscosity,
                    opts.disk_radius_outer)

                ratio_heat_mig_stars_torques = feedback.feedback_stars_hankla(
                    stars_pro.orb_a,
                    disk_surface_density,
                    disk_opacity,
                    opts.disk_star_eddington_ratio,
                    opts.disk_alpha_viscosity,
                    opts.disk_radius_outer,)
            else:
                ratio_heat_mig_torques = np.ones(blackholes_pro.num)
                ratio_heat_mig_stars_torques = np.ones(stars_pro.num)

            # Migration, choose your torque_prescription
            new_orb_a_bh = None  # Set empty variable, we'll fill it based on torque_prescription
            new_orb_a_star = None

            # Old is the original approximation used in v.0.1.0, based off (but not identical to Paardekooper 2010)-usually within factor [0.5-2]
            if opts.torque_prescription == 'old':
                # Old migration prescription
                new_orb_a_bh = migration.type1_migration_single(
                    opts.smbh_mass,
                    blackholes_pro.orb_a,
                    blackholes_pro.mass,
                    blackholes_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_surface_density,
                    disk_aspect_ratio,
                    ratio_heat_mig_torques,
                    opts.disk_radius_trap,
                    opts.disk_radius_outer,
                    opts.timestep_duration_yr
                )

                new_orb_a_star = migration.type1_migration_single(
                    opts.smbh_mass,
                    stars_pro.orb_a,
                    stars_pro.mass,
                    stars_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_surface_density,
                    disk_aspect_ratio,
                    ratio_heat_mig_stars_torques,
                    opts.disk_radius_trap,
                    opts.disk_radius_outer,
                    opts.timestep_duration_yr
                )

            # Alternatively, calculate actual torques from disk profiles.
            # Paardekooper torque coeff (default)
            if opts.torque_prescription == 'paardekooper':
                paardekooper_torque_coeff_bh = migration.paardekooper10_torque(
                    blackholes_pro.orb_a,
                    blackholes_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_dlog10surfdens_dlog10R_func,
                    disk_dlog10temp_dlog10R_func
                )

                paardekooper_torque_coeff_star = migration.paardekooper10_torque(
                    stars_pro.orb_a,
                    stars_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_dlog10surfdens_dlog10R_func,
                    disk_dlog10temp_dlog10R_func
                )

            # Jimenez-Masset torque coeff (from Grishin+24)
            if opts.torque_prescription == 'jimenez_masset':
                jimenez_masset_torque_coeff_bh = migration.jimenezmasset17_torque(
                    opts.smbh_mass,
                    disk_surface_density,
                    disk_opacity,
                    disk_aspect_ratio,
                    temp_func,
                    blackholes_pro.orb_a,
                    blackholes_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_dlog10surfdens_dlog10R_func,
                    disk_dlog10temp_dlog10R_func
                )

                jimenez_masset_torque_coeff_star = migration.jimenezmasset17_torque(
                    opts.smbh_mass,
                    disk_surface_density,
                    disk_opacity,
                    disk_aspect_ratio,
                    temp_func,
                    stars_pro.orb_a,
                    stars_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_dlog10surfdens_dlog10R_func,
                    disk_dlog10temp_dlog10R_func
                )

                # Thermal torque from JM17 (if flag_thermal_feedback off, this component is 0.)
                jimenez_masset_thermal_torque_coeff_bh = migration.jimenezmasset17_thermal_torque_coeff(
                    opts.smbh_mass,
                    disk_surface_density,
                    disk_opacity,
                    disk_aspect_ratio,
                    temp_func,
                    opts.disk_bh_eddington_ratio,
                    blackholes_pro.orb_a,
                    blackholes_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    blackholes_pro.mass,
                    opts.flag_thermal_feedback,
                    disk_dlog10pressure_dlog10R_func
                )

                jimenez_masset_thermal_torque_coeff_star = migration.jimenezmasset17_thermal_torque_coeff(
                    opts.smbh_mass,
                    disk_surface_density,
                    disk_opacity,
                    disk_aspect_ratio,
                    temp_func,
                    opts.disk_bh_eddington_ratio,
                    stars_pro.orb_a,
                    stars_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    stars_pro.mass,
                    opts.flag_thermal_feedback,
                    disk_dlog10pressure_dlog10R_func
                )

                if opts.flag_thermal_feedback == 1:
                    total_jimenez_masset_torque_bh = jimenez_masset_torque_coeff_bh + jimenez_masset_thermal_torque_coeff_bh
                    total_jimenez_masset_torque_star = jimenez_masset_torque_coeff_star + jimenez_masset_thermal_torque_coeff_star
                else:
                    total_jimenez_masset_torque_bh = jimenez_masset_torque_coeff_bh
                    total_jimenez_masset_torque_star = jimenez_masset_torque_coeff_star

            # Normalized torque (multiplies torque coeff)
            if opts.torque_prescription == 'paardekooper' or opts.torque_prescription == 'jimenez_masset':
                normalized_torque_bh = migration.normalized_torque(
                    opts.smbh_mass,
                    blackholes_pro.orb_a,
                    blackholes_pro.mass,
                    blackholes_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_surface_density,
                    disk_aspect_ratio
                )

                normalized_torque_star = migration.normalized_torque(
                    opts.smbh_mass,
                    stars_pro.orb_a,
                    stars_pro.mass,
                    stars_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_surface_density,
                    disk_aspect_ratio
                )

                if opts.torque_prescription == 'paardekooper':
                    torque_bh = paardekooper_torque_coeff_bh * normalized_torque_bh
                    torque_star = paardekooper_torque_coeff_star * normalized_torque_star
                    disk_trap_radius = opts.disk_radius_trap
                    disk_anti_trap_radius = opts.disk_radius_trap
                if opts.torque_prescription == 'jimenez_masset':
                    torque_bh = total_jimenez_masset_torque_bh * normalized_torque_bh
                    torque_star = total_jimenez_masset_torque_star * normalized_torque_star
                    # Set up trap scaling as a function of mass for Jimenez-Masset (for SG-like disk)
                    # No traps if M_smbh >10^8Msun (approx.)
                    if opts.smbh_mass > 1.e8:
                        disk_trap_radius = opts.disk_inner_stable_circ_orb
                        disk_anti_trap_radius = opts.disk_inner_stable_circ_orb
                    if opts.smbh_mass == 1.e8:
                        disk_trap_radius = opts.disk_radius_trap
                        disk_anti_trap_radius = opts.disk_radius_trap
                    # Trap changes as a function of r_g if M_smbh <10^8Msun (default trap radius ~700r_g). Grishin+24
                    if opts.smbh_mass < 1.e8 and opts.smbh_mass > 1.e6:
                        disk_trap_radius = opts.disk_radius_trap * (opts.smbh_mass / 1.e8) ** (-1.225)
                        disk_anti_trap_radius = opts.disk_radius_trap * (opts.smbh_mass / 1.e8) ** (0.099)
                    # Trap location changes again at low SMBH mass (Grishin+24)
                    if opts.smbh_mass < 1.e6:
                        disk_trap_radius = opts.disk_radius_trap * (opts.smbh_mass / 1.e8) ** (-0.97)
                        disk_anti_trap_radius = opts.disk_radius_trap * (opts.smbh_mass / 1.e8) ** (0.099)
                # Timescale on which migration happens based on overall torque
                torque_mig_timescales_bh = migration.torque_mig_timescale(
                    opts.smbh_mass,
                    blackholes_pro.orb_a,
                    blackholes_pro.mass,
                    blackholes_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    torque_bh
                )

                torque_mig_timescales_star = migration.torque_mig_timescale(
                    opts.smbh_mass,
                    stars_pro.orb_a,
                    stars_pro.mass,
                    stars_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    torque_star
                )
                # Calculate new bh_orbs_a using torque (here including details from Jimenez & Masset '17 & Grishin+'24)
                new_orb_a_bh = migration.type1_migration_distance(
                    opts.smbh_mass,
                    blackholes_pro.orb_a,
                    blackholes_pro.mass,
                    blackholes_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    torque_mig_timescales_bh,
                    ratio_heat_mig_torques,
                    disk_trap_radius,
                    disk_anti_trap_radius,
                    opts.disk_radius_outer,
                    opts.timestep_duration_yr,
                    opts.flag_phenom_turb,
                    opts.phenom_turb_centroid,
                    opts.phenom_turb_std_dev,
                    opts.nsc_imf_bh_mode,
                    opts.torque_prescription
                )

                new_orb_a_star = migration.type1_migration_distance(
                    opts.smbh_mass,
                    stars_pro.orb_a,
                    stars_pro.mass,
                    stars_pro.orb_ecc,
                    opts.disk_bh_pro_orb_ecc_crit,
                    torque_mig_timescales_star,
                    ratio_heat_mig_stars_torques,
                    disk_trap_radius,
                    disk_anti_trap_radius,
                    opts.disk_radius_outer,
                    opts.timestep_duration_yr,
                    opts.flag_phenom_turb,
                    opts.phenom_turb_centroid,
                    opts.phenom_turb_std_dev,
                    opts.disk_star_mass_min_init,
                    opts.torque_prescription
                )
            #Make sure no zeros in orb_a. Get indices of orbs_a that are less than disk_inner_stable_circ_orb
            # Get indices of objects with orb_ecc <= opts.disk_inner_stable_circ_orb so we can remove them.
            #plunging_indices = np.asarray(blackholes_pro.orb_a) <= opts.disk_inner_stable_circ_orb).nonzero()[0]
            #blackholes_pro.orb_a = blackholes_pro.orb_a[~plunging_indices]
            #blackholes_pro.orb_ecc = blackholes_pro.orb_ecc[~plunging_indices]
            #blackholes_pro.mass = blackholes_pro.mass[~plunging_indices]
            #blackholes_pro.spin = blackholes_pro.spin[~plunging_indices]
            #blackholes_pro.spin_angle = blackholes_pro.spin_angle[~plunging_indices]
            #blackholes_pro. = blackholes_pro.orb_ecc[~plunging_indices]

            blackholes_pro.orb_a = np.where(blackholes_pro.orb_a > opts.disk_inner_stable_circ_orb, blackholes_pro.orb_a, 3*opts.disk_inner_stable_circ_orb)
            if new_orb_a_bh is not None:
                blackholes_pro.orb_a = new_orb_a_bh

            stars_pro.orb_a = np.where(stars_pro.orb_a > opts.disk_inner_stable_circ_orb, stars_pro.orb_a, 3*opts.disk_inner_stable_circ_orb)
            if new_orb_a_star is not None:
                stars_pro.orb_a = new_orb_a_star

            # Update filing cabinet
            filing_cabinet.update(id_num=blackholes_pro.id_num,
                                  attr="orb_a",
                                  new_info=blackholes_pro.orb_a)
            filing_cabinet.update(id_num=stars_pro.id_num,
                                  attr="orb_a",
                                  new_info=stars_pro.orb_a)

            # Check for eccentricity > 1 (hyperbolic orbit, ejected from disk)
            bh_pro_id_num_ecc_hyperbolic = blackholes_pro.id_num[blackholes_pro.orb_ecc >= 1.]
            if bh_pro_id_num_ecc_hyperbolic.size > 0:
                blackholes_pro.remove_id_num(bh_pro_id_num_ecc_hyperbolic)
                filing_cabinet.remove_id_num(bh_pro_id_num_ecc_hyperbolic)

            # Stars lose mass via stellar winds
            stars_pro.mass, star_mass_lost = accretion.star_wind_mass_loss(
                stars_pro.mass,
                stars_pro.log_radius,
                stars_pro.log_luminosity,
                stars_pro.orb_a,
                disk_opacity,
                opts.timestep_duration_yr
            )

            # Mass lost from stars is gained by the disk
            disk_mass_gained.append(np.abs(star_mass_lost))

            # Accrete
            blackholes_pro.mass = accretion.change_bh_mass(
                blackholes_pro.mass,
                opts.disk_bh_eddington_ratio,
                disk_bh_eddington_mass_growth_rate,
                opts.timestep_duration_yr
            )

            disk_star_luminosity_factor = 4.  # Hardcoded from Cantiello+2021 and Fabj+2024
            stars_pro.mass, star_mass_gained, star_immortal_mass_lost = accretion.accrete_star_mass(
                stars_pro.mass,
                stars_pro.orb_a,
                stars_pro.orb_ecc,
                disk_star_luminosity_factor,
                opts.disk_star_initial_mass_cutoff,
                opts.smbh_mass,
                disk_sound_speed,
                disk_density,
                opts.timestep_duration_yr
            )

            # Mass gained by stars is lost from disk
            disk_mass_lost.append(star_mass_gained)
            # Mass gained over opts.disk_star_initial_mass_cutoff is immediately blown back into the disk
            disk_mass_gained.append(star_immortal_mass_lost)

            # Change stars' radii, luminosity, and temp
            stars_pro.log_radius, stars_pro.log_luminosity, stars_pro.log_teff = stellar_interpolation.interp_star_params(stars_pro.mass)

            # Update filing cabinet
            filing_cabinet.update(id_num=blackholes_pro.id_num,
                                  attr="mass",
                                  new_info=blackholes_pro.mass)
            filing_cabinet.update(id_num=stars_pro.id_num,
                                  attr="mass",
                                  new_info=stars_pro.mass)
            filing_cabinet.update(id_num=stars_pro.id_num,
                                  attr="size",
                                  new_info=point_masses.r_g_from_units(opts.smbh_mass, (10 ** stars_pro.log_radius) * u.Rsun).value)

            # Spin up
            blackholes_pro.spin = accretion.change_bh_spin_magnitudes(
                blackholes_pro.spin,
                opts.disk_bh_eddington_ratio,
                opts.disk_bh_torque_condition,
                opts.timestep_duration_yr,
                blackholes_pro.orb_ecc,
                opts.disk_bh_pro_orb_ecc_crit,
            )

            # Torque spin angle
            blackholes_pro.spin_angle = accretion.change_bh_spin_angles(
                blackholes_pro.spin_angle,
                opts.disk_bh_eddington_ratio,
                opts.disk_bh_torque_condition,
                disk_bh_spin_resolution_min,
                opts.timestep_duration_yr,
                blackholes_pro.orb_ecc,
                opts.disk_bh_pro_orb_ecc_crit
            )

            # Damp orbital eccentricity
            blackholes_pro.orb_ecc = eccentricity.orbital_ecc_damping(
                opts.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                disk_surface_density,
                disk_aspect_ratio,
                blackholes_pro.orb_ecc,
                opts.timestep_duration_yr,
                opts.disk_bh_pro_orb_ecc_crit,
            )

            stars_pro.orb_ecc = eccentricity.orbital_ecc_damping(
                opts.smbh_mass,
                stars_pro.orb_a,
                stars_pro.mass,
                disk_surface_density,
                disk_aspect_ratio,
                stars_pro.orb_ecc,
                opts.timestep_duration_yr,
                opts.disk_bh_pro_orb_ecc_crit,
            )

            # Update filing cabinet
            filing_cabinet.update(id_num=blackholes_pro.id_num,
                                  attr="orb_ecc",
                                  new_info=blackholes_pro.orb_ecc)
            filing_cabinet.update(id_num=stars_pro.id_num,
                                  attr="orb_ecc",
                                  new_info=stars_pro.orb_ecc)

            # Now do retrograde singles--change semi-major axis
            #   note this is dyn friction only, not true 'migration'
            # change retrograde eccentricity (some damping, some pumping)
            # damp orbital inclination
            blackholes_retro.orb_ecc, blackholes_retro.orb_a, blackholes_retro.orb_inc = disk_capture.retro_bh_orb_disk_evolve(
                opts.smbh_mass,
                blackholes_retro.mass,
                blackholes_retro.orb_a,
                blackholes_retro.orb_ecc,
                blackholes_retro.orb_inc,
                blackholes_retro.orb_arg_periapse,
                opts.disk_inner_stable_circ_orb,
                disk_surface_density,
                opts.timestep_duration_yr,
                opts.disk_radius_outer
            )
            # KN: Does this function apply to all disk objects and if so should we rename it?
            stars_retro.orb_ecc, stars_retro.orb_a, stars_retro.orb_inc = disk_capture.retro_bh_orb_disk_evolve(
                opts.smbh_mass,
                stars_retro.mass,
                stars_retro.orb_a,
                stars_retro.orb_ecc,
                stars_retro.orb_inc,
                stars_retro.orb_arg_periapse,
                opts.disk_inner_stable_circ_orb,
                disk_surface_density,
                opts.timestep_duration_yr,
                opts.disk_radius_outer
            )

            # Update filing cabinet
            filing_cabinet.update(id_num=blackholes_retro.id_num,
                                  attr="orb_ecc",
                                  new_info=blackholes_retro.orb_ecc)
            filing_cabinet.update(id_num=stars_retro.id_num,
                                  attr="orb_ecc",
                                  new_info=stars_retro.orb_ecc)

            filing_cabinet.update(id_num=blackholes_retro.id_num,
                                  attr="orb_a",
                                  new_info=blackholes_retro.orb_a)
            filing_cabinet.update(id_num=stars_retro.id_num,
                                  attr="orb_a",
                                  new_info=stars_retro.orb_a)

            # Check for hyperbolic eccentricity (ejected from disk)
            bh_retro_id_num_ecc_hyperbolic = blackholes_retro.id_num[blackholes_retro.orb_ecc >= 1.]
            if bh_retro_id_num_ecc_hyperbolic.size > 0:
                blackholes_retro.remove_id_num(bh_retro_id_num_ecc_hyperbolic)
                filing_cabinet.remove_id_num(bh_retro_id_num_ecc_hyperbolic)

            star_retro_id_num_ecc_hyperbolic = stars_retro.id_num[stars_retro.orb_ecc >= 1.]
            if star_retro_id_num_ecc_hyperbolic.size > 0:
                stars_retro.remove_id_num(star_retro_id_num_ecc_hyperbolic)
                filing_cabinet.remove_id_num(star_retro_id_num_ecc_hyperbolic)

            # Perturb eccentricity via dynamical encounters
            if opts.flag_dynamic_enc > 0:

                # BH-BH encounters
                blackholes_pro.orb_a, blackholes_pro.orb_ecc = dynamics.circular_singles_encounters_prograde(
                    opts.smbh_mass,
                    blackholes_pro.orb_a,
                    blackholes_pro.mass,
                    blackholes_pro.orb_ecc,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit,
                    opts.delta_energy_strong,
                    opts.disk_radius_outer
                )

                # Star-star encounters
                rstar_rhill_exponent = 2.0
                stars_pro.orb_a, stars_pro.orb_ecc, star_touch_id_nums = dynamics.circular_singles_encounters_prograde_stars(
                    opts.smbh_mass,
                    stars_pro.orb_a,
                    stars_pro.mass,
                    stars_pro.log_radius,
                    stars_pro.orb_ecc,
                    stars_pro.id_num,
                    rstar_rhill_exponent,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit,
                    opts.delta_energy_strong,
                    opts.disk_radius_outer
                )

                if (star_touch_id_nums.size > 0):
                    # Star and star touch each other: stellar merger
                    # Generate new ID numbers
                    star_merged_id_num_new = np.arange(filing_cabinet.id_max + 1, filing_cabinet.id_max + 1 + star_touch_id_nums.shape[1], 1)
                    # Merged mass is just masses added together. Any over disk_star_initial_mass_cutoff get set to disk_star_initial_mass_cutoff
                    star_merged_mass = stars_pro.at_id_num(star_touch_id_nums[0], "mass") + stars_pro.at_id_num(star_touch_id_nums[1], "mass")
                    # New orb_a is the center of mass of the two stars
                    star_merged_orbs_a = ((stars_pro.at_id_num(star_touch_id_nums[0], "mass") * stars_pro.at_id_num(star_touch_id_nums[0], "orb_a")) +
                                          (stars_pro.at_id_num(star_touch_id_nums[1], "mass") * stars_pro.at_id_num(star_touch_id_nums[1], "orb_a"))) / star_merged_mass
                    # After doing the weighted average for orb_a we then cut off stars with mass > disk_star_initial_mass_cutoff
                    star_merged_mass[star_merged_mass > opts.disk_star_initial_mass_cutoff] = opts.disk_star_initial_mass_cutoff
                    # Radius, luminosity, Teff are all interpolated based on the new mass
                    star_merged_logR, star_merged_logL, star_merged_logTeff = stellar_interpolation.interp_star_params(star_merged_mass)
                    # New gen is the maximum between the pair's gen plus one
                    star_merged_gen = np.maximum(stars_pro.at_id_num(star_touch_id_nums[0], "gen"), stars_pro.at_id_num(star_touch_id_nums[1], "gen")) + 1.0

                    # Add to stars object
                    stars_pro.add_stars(new_id_num=star_merged_id_num_new,
                                        new_mass=star_merged_mass,
                                        new_orb_a=star_merged_orbs_a,
                                        new_log_radius=star_merged_logR,
                                        new_log_teff=star_merged_logTeff,
                                        new_log_luminosity=star_merged_logL,
                                        new_X=stars_pro.at_id_num(star_touch_id_nums[0], "star_X"),  # no metallicity evolution
                                        new_Y=stars_pro.at_id_num(star_touch_id_nums[0], "star_Y"),
                                        new_Z=stars_pro.at_id_num(star_touch_id_nums[0], "star_Z"),
                                        new_orb_ang_mom=np.ones(star_touch_id_nums.shape[1]),  # orb_ang_mom is +1 because two prograde stars merge
                                        new_orb_ecc=np.full(star_touch_id_nums.shape[1], opts.disk_bh_pro_orb_ecc_crit),  # orb_ecc is initially very small
                                        new_orb_inc=np.zeros(star_touch_id_nums.shape[1]),  # orb_inc is zero
                                        new_orb_arg_periapse=stars_pro.at_id_num(star_touch_id_nums[0], "orb_arg_periapse"),  # Assume orb_arg_periapse is same as before
                                        new_gen=star_merged_gen,
                                        new_galaxy=stars_pro.at_id_num(star_touch_id_nums[0], "galaxy"),
                                        new_time_passed=stars_pro.at_id_num(star_touch_id_nums[0], "time_passed"))

                    # Add new merged stars to merged stars object
                    stars_merge.add_stars(new_id_num=star_merged_id_num_new,
                                          new_galaxy=stars_pro.at_id_num(star_touch_id_nums[0], "galaxy"),
                                          new_orb_a_final=star_merged_orbs_a,
                                          new_gen_final=star_merged_gen,
                                          new_mass_final=star_merged_mass,
                                          new_mass_1=stars_pro.at_id_num(star_touch_id_nums[0], "mass"),
                                          new_mass_2=stars_pro.at_id_num(star_touch_id_nums[1], "mass"),
                                          new_gen_1=stars_pro.at_id_num(star_touch_id_nums[0], "gen"),
                                          new_gen_2=stars_pro.at_id_num(star_touch_id_nums[1], "gen"),
                                          new_log_radius_final=star_merged_logR,
                                          new_orb_ecc=np.full(star_touch_id_nums.shape[1], opts.disk_bh_pro_orb_ecc_crit),
                                          new_time_merged=np.full(star_touch_id_nums.shape[1], time_passed))

                    # Add new merged stars to filing cabinet and delete previous stars
                    filing_cabinet.add_objects(new_id_num=star_merged_id_num_new,
                                               new_category=np.ones(star_merged_id_num_new.size),
                                               new_orb_a=stars_pro.at_id_num(star_merged_id_num_new, "orb_a"),
                                               new_mass=stars_pro.at_id_num(star_merged_id_num_new, "mass"),
                                               new_orb_ecc=stars_pro.at_id_num(star_merged_id_num_new, "orb_ecc"),
                                               new_size=point_masses.r_g_from_units(opts.smbh_mass, (10 ** stars_pro.at_id_num(star_merged_id_num_new, "log_radius")) * u.Rsun).value,
                                               new_direction=np.ones(star_merged_id_num_new.size),
                                               new_disk_inner_outer=np.ones(star_merged_id_num_new.size))
                    filing_cabinet.remove_id_num(star_touch_id_nums.flatten())
                    stars_pro.remove_id_num(star_touch_id_nums.flatten())

                # Star-BH encounters (circular stars and eccentric BH)
                stars_pro.orb_a, stars_pro.orb_ecc, blackholes_pro.orb_a, blackholes_pro.orb_ecc, bh_star_touch_id_nums = dynamics.circular_singles_encounters_prograde_star_bh(
                    opts.smbh_mass,
                    stars_pro.orb_a,
                    stars_pro.mass,
                    stars_pro.log_radius,
                    stars_pro.orb_ecc,
                    stars_pro.id_num,
                    rstar_rhill_exponent,
                    blackholes_pro.orb_a,
                    blackholes_pro.mass,
                    blackholes_pro.orb_ecc,
                    blackholes_pro.id_num,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit,
                    opts.delta_energy_strong,
                    opts.disk_radius_outer
                )

                if (bh_star_touch_id_nums.size > 0):
                    # BH and star encounter: star blows up, BH accretes mass
                    # Separate out into BH ID and star ID
                    bh_star_touch_id_nums = bh_star_touch_id_nums.flatten()
                    star_id_nums = bh_star_touch_id_nums[np.nonzero(filing_cabinet.at_id_num(bh_star_touch_id_nums, "category") == 1)]
                    bh_id_nums = bh_star_touch_id_nums[np.nonzero(filing_cabinet.at_id_num(bh_star_touch_id_nums, "category") == 0)]
                    stars_explode.add_stars(new_id_num_star=star_id_nums,
                                            new_id_num_bh=bh_id_nums,
                                            new_mass_star=stars_pro.at_id_num(star_id_nums, "mass"),
                                            new_mass_bh=blackholes_pro.at_id_num(bh_id_nums, "mass"),
                                            new_orb_a_star=stars_pro.at_id_num(star_id_nums, "orb_a"),
                                            new_orb_a_bh=blackholes_pro.at_id_num(bh_id_nums, "orb_a"),
                                            new_star_log_radius=stars_pro.at_id_num(star_id_nums, "log_radius"),
                                            new_orb_inc_star=stars_pro.at_id_num(star_id_nums, "orb_inc"),
                                            new_orb_inc_bh=blackholes_pro.at_id_num(bh_id_nums, "orb_inc"),
                                            new_orb_ecc_star=stars_pro.at_id_num(star_id_nums, "orb_ecc"),
                                            new_orb_ecc_bh=blackholes_pro.at_id_num(bh_id_nums, "orb_ecc"),
                                            new_gen_star=stars_pro.at_id_num(star_id_nums, "gen"),
                                            new_gen_bh=blackholes_pro.at_id_num(bh_id_nums, "gen"),
                                            new_galaxy=stars_pro.at_id_num(star_id_nums, "galaxy"),
                                            new_time_sn=np.full(star_id_nums.size, time_passed),
                                            )
                    # Add exploded star mass to mass gained by disk
                    disk_mass_gained.append(stars_pro.at_id_num(star_id_nums, "mass").sum())
                    # Delete exploded stars from regular array and filing cabinet
                    stars_pro.remove_id_num(star_id_nums)
                    filing_cabinet.remove_id_num(star_id_nums)

                    # BHs accrete mass and spin up
                    _, bh_id_mask = np.where(blackholes_pro.id_num == bh_id_nums[:, None])
                    blackholes_pro.mass[bh_id_mask] = accretion.change_bh_mass(
                        blackholes_pro.mass[bh_id_mask],
                        opts.disk_bh_eddington_ratio,
                        disk_bh_eddington_mass_growth_rate,
                        opts.timestep_duration_yr)

                    blackholes_pro.spin[bh_id_mask] = accretion.change_bh_spin_magnitudes(
                        blackholes_pro.spin[bh_id_mask],
                        opts.disk_bh_eddington_ratio,
                        opts.disk_bh_torque_condition,
                        opts.timestep_duration_yr,
                        blackholes_pro.orb_ecc[bh_id_mask],
                        opts.disk_bh_pro_orb_ecc_crit,
                    )

                    blackholes_pro.spin_angle[bh_id_mask] = accretion.change_bh_spin_angles(
                        blackholes_pro.spin_angle[bh_id_mask],
                        opts.disk_bh_eddington_ratio,
                        opts.disk_bh_torque_condition,
                        disk_bh_spin_resolution_min,
                        opts.timestep_duration_yr,
                        blackholes_pro.orb_ecc[bh_id_mask],
                        opts.disk_bh_pro_orb_ecc_crit
                    )

                    # Update filing cabinet
                    filing_cabinet.update(id_num=bh_id_nums,
                                          attr="mass",
                                          new_info=blackholes_pro.mass[bh_id_mask])

                # Update filing cabinet
                filing_cabinet.update(id_num=blackholes_pro.id_num,
                                      attr="orb_ecc",
                                      new_info=blackholes_pro.orb_ecc)
                filing_cabinet.update(id_num=stars_pro.id_num,
                                      attr="orb_ecc",
                                      new_info=stars_pro.orb_ecc)

                filing_cabinet.update(id_num=blackholes_pro.id_num,
                                      attr="orb_a",
                                      new_info=blackholes_pro.orb_a)
                filing_cabinet.update(id_num=stars_pro.id_num,
                                      attr="orb_a",
                                      new_info=stars_pro.orb_a)

            # Do things to the binaries--first check if there are any:
            if blackholes_binary.num > 0:

                # First check that binaries are real (mass and location are not zero)
                bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary.mass_1,
                                                                       blackholes_binary.mass_2,
                                                                       blackholes_binary.orb_a_1,
                                                                       blackholes_binary.orb_a_2,
                                                                       blackholes_binary.bin_ecc,
                                                                       blackholes_binary.id_num)
                if bh_binary_id_num_unphysical.size > 0:
                    blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                    filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)

                # Check for binaries with hyperbolic eccentricity (ejected from disk)
                bh_binary_id_num_ecc_hyperbolic = blackholes_binary.id_num[blackholes_binary.bin_orb_ecc >= 1.]
                if bh_binary_id_num_ecc_hyperbolic.size > 0:
                    blackholes_binary.remove_id_num(bh_binary_id_num_ecc_hyperbolic)
                    filing_cabinet.remove_id_num(bh_binary_id_num_ecc_hyperbolic)

                # If there are binaries, evolve them
                # Damp binary orbital eccentricity
                blackholes_binary.bin_orb_ecc = eccentricity.orbital_bin_ecc_damping(
                    opts.smbh_mass,
                    blackholes_binary.mass_1,
                    blackholes_binary.mass_2,
                    blackholes_binary.bin_orb_a,
                    blackholes_binary.bin_ecc,
                    blackholes_binary.bin_orb_ecc,
                    disk_surface_density,
                    disk_aspect_ratio,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit
                )

                # Update filing cabinet
                filing_cabinet.update(id_num=blackholes_binary.id_num,
                                      attr="orb_ecc",
                                      new_info=blackholes_binary.bin_orb_ecc)

                if (opts.flag_dynamic_enc > 0):
                    # Harden/soften binaries via dynamical encounters
                    # Harden binaries due to encounters with circular singletons (e.g. Leigh et al. 2018)
                    blackholes_binary.bin_sep, blackholes_binary.bin_ecc, blackholes_binary.bin_orb_ecc, blackholes_pro.orb_a, blackholes_pro.orb_ecc = dynamics.circular_binaries_encounters_circ_prograde(
                        opts.smbh_mass,
                        blackholes_pro.orb_a,
                        blackholes_pro.mass,
                        blackholes_pro.orb_ecc,
                        blackholes_binary.mass_1,
                        blackholes_binary.mass_1,
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.bin_sep,
                        blackholes_binary.bin_ecc,
                        blackholes_binary.bin_orb_ecc,
                        opts.timestep_duration_yr,
                        opts.disk_bh_pro_orb_ecc_crit,
                        opts.delta_energy_strong,
                        opts.disk_radius_outer,
                        opts.mean_harden_energy_delta,
                        opts.var_harden_energy_delta
                    )

                    # Update filing cabinet with new bin_sep
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="size",
                                          new_info=blackholes_binary.bin_sep)
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="orb_ecc",
                                          new_info=blackholes_binary.bin_orb_ecc)
                    filing_cabinet.update(id_num=blackholes_pro.id_num,
                                          attr="orb_a",
                                          new_info=blackholes_pro.orb_a)
                    filing_cabinet.update(id_num=blackholes_pro.id_num,
                                          attr="orb_ecc",
                                          new_info=blackholes_pro.orb_ecc)

                    # Check for mergers
                    # Check closeness of binary. Are black holes at merger condition separation
                    blackholes_binary.bin_sep, blackholes_binary.flag_merging = evolve.bin_contact_check(blackholes_binary.mass_1, blackholes_binary.mass_2, blackholes_binary.bin_sep, blackholes_binary.flag_merging, opts.smbh_mass)
                    bh_binary_id_num_merger = blackholes_binary.id_num[blackholes_binary.flag_merging < 0]

                    if opts.verbose:
                        print("Merger ID numbers")
                        print(bh_binary_id_num_merger)

                    if (bh_binary_id_num_merger.size > 0):

                        bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary.mass_1,
                                                                               blackholes_binary.mass_2,
                                                                               blackholes_binary.orb_a_1,
                                                                               blackholes_binary.orb_a_2,
                                                                               blackholes_binary.bin_ecc,
                                                                               blackholes_binary.id_num)
                        if bh_binary_id_num_unphysical.size > 0:
                            blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                            filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)

                        blackholes_merged, blackholes_pro = merge.merge_blackholes(blackholes_binary,
                                                                                   blackholes_pro,
                                                                                   blackholes_merged,
                                                                                   bh_binary_id_num_merger,
                                                                                   opts.smbh_mass,
                                                                                   opts.flag_use_surrogate,
                                                                                   disk_aspect_ratio,
                                                                                   disk_density,
                                                                                   time_passed,
                                                                                   galaxy)

                        # Update filing cabinet
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="category",
                                              new_info=np.full(bh_binary_id_num_merger.size, 0))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="mass",
                                              new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "mass"))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="orb_ecc",
                                              new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "orb_ecc"))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="size",
                                              new_info=np.full(bh_binary_id_num_merger.size, -1.5))
                        blackholes_binary.remove_id_num(bh_binary_id_num_merger)

                        if opts.verbose:
                            print("New BH locations", blackholes_pro.orb_a)
                    else:
                        # No merger
                        # do nothing! hardening should happen FIRST (and now it does!)
                        if (opts.verbose):
                            print("No mergers yet")

                    # Soften/ ionize binaries due to encounters with eccentric singletons
                    # Return 3 things: perturbed biary_bh_array, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc
                    blackholes_binary.bin_sep, blackholes_binary.bin_ecc, blackholes_binary.bin_orb_ecc, blackholes_pro.orb_a, blackholes_pro.orb_ecc = dynamics.circular_binaries_encounters_ecc_prograde(
                        opts.smbh_mass,
                        blackholes_pro.orb_a,
                        blackholes_pro.mass,
                        blackholes_pro.orb_ecc,
                        blackholes_binary.mass_1,
                        blackholes_binary.mass_1,
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.bin_sep,
                        blackholes_binary.bin_ecc,
                        blackholes_binary.bin_orb_ecc,
                        opts.timestep_duration_yr,
                        opts.disk_bh_pro_orb_ecc_crit,
                        opts.delta_energy_strong,
                        opts.disk_radius_outer
                    )

                    # Update filing cabinet with new bin_sep
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="size",
                                          new_info=blackholes_binary.bin_sep)
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="orb_ecc",
                                          new_info=blackholes_binary.bin_orb_ecc)
                    filing_cabinet.update(id_num=blackholes_pro.id_num,
                                          attr="orb_a",
                                          new_info=blackholes_pro.orb_a)
                    filing_cabinet.update(id_num=blackholes_pro.id_num,
                                          attr="orb_ecc",
                                          new_info=blackholes_pro.orb_ecc)

                    # Check for mergers
                    # Check closeness of binary. Are black holes at merger condition separation
                    blackholes_binary.bin_sep, blackholes_binary.flag_merging = evolve.bin_contact_check(blackholes_binary.mass_1, blackholes_binary.mass_2, blackholes_binary.bin_sep, blackholes_binary.flag_merging, opts.smbh_mass)
                    bh_binary_id_num_merger = blackholes_binary.id_num[blackholes_binary.flag_merging < 0]

                    if opts.verbose:
                        print("Merger ID numbers")
                        print(bh_binary_id_num_merger)

                    if (bh_binary_id_num_merger.size > 0):

                        bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary.mass_1,
                                                                               blackholes_binary.mass_2,
                                                                               blackholes_binary.orb_a_1,
                                                                               blackholes_binary.orb_a_2,
                                                                               blackholes_binary.bin_ecc,
                                                                               blackholes_binary.id_num)
                        if bh_binary_id_num_unphysical.size > 0:
                            blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                            filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)

                        blackholes_merged, blackholes_pro = merge.merge_blackholes(blackholes_binary,
                                                                                   blackholes_pro,
                                                                                   blackholes_merged,
                                                                                   bh_binary_id_num_merger,
                                                                                   opts.smbh_mass,
                                                                                   opts.flag_use_surrogate,
                                                                                   disk_aspect_ratio,
                                                                                   disk_density,
                                                                                   time_passed,
                                                                                   galaxy)

                        # Update filing cabinet
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="category",
                                              new_info=np.full(bh_binary_id_num_merger.size, 0))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="mass",
                                              new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "mass"))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="orb_ecc",
                                              new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "orb_ecc"))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="size",
                                              new_info=np.full(bh_binary_id_num_merger.size, -1.5))
                        blackholes_binary.remove_id_num(bh_binary_id_num_merger)

                        if opts.verbose:
                            print("New BH locations", blackholes_pro.orb_a)
                    else:
                        # No merger
                        # do nothing! hardening should happen FIRST (and now it does!)
                        if (opts.verbose):
                            print("No mergers yet")

                # Check for hyperbolic eccentricity (binary ejected from disk)
                bh_binary_id_num_ecc_hyperbolic = blackholes_binary.id_num[blackholes_binary.bin_ecc >= 1.]
                if bh_binary_id_num_ecc_hyperbolic.size > 0:
                    blackholes_binary.remove_id_num(bh_binary_id_num_ecc_hyperbolic)
                    filing_cabinet.remove_id_num(bh_binary_id_num_ecc_hyperbolic)

                # Harden binaries via gas
                # Choose between Baruteau et al. 2011 gas hardening, or gas hardening from LANL simulations. To do: include dynamical hardening/softening from encounters
                blackholes_binary.bin_sep, blackholes_binary.flag_merging, blackholes_binary.time_merged, blackholes_binary.time_to_merger_gw = evolve.bin_harden_baruteau(
                    blackholes_binary.mass_1,
                    blackholes_binary.mass_2,
                    blackholes_binary.bin_sep,
                    blackholes_binary.bin_ecc,
                    blackholes_binary.time_to_merger_gw,
                    blackholes_binary.flag_merging,
                    blackholes_binary.time_merged,
                    opts.smbh_mass,
                    opts.timestep_duration_yr,
                    time_gw_normalization,
                    time_passed,
                )

                # Update filing cabinet with new bin_sep
                filing_cabinet.update(id_num=blackholes_binary.id_num,
                                      attr="size",
                                      new_info=blackholes_binary.bin_sep)

                # Check for mergers
                # Check closeness of binary. Are black holes at merger condition separation
                blackholes_binary.bin_sep, blackholes_binary.flag_merging = evolve.bin_contact_check(blackholes_binary.mass_1, blackholes_binary.mass_2, blackholes_binary.bin_sep, blackholes_binary.flag_merging, opts.smbh_mass)
                bh_binary_id_num_merger = blackholes_binary.id_num[blackholes_binary.flag_merging < 0]

                if opts.verbose:
                    print("Merger ID numbers")
                    print(bh_binary_id_num_merger)

                if (bh_binary_id_num_merger.size > 0):
                    bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary.mass_1,
                                                                           blackholes_binary.mass_2,
                                                                           blackholes_binary.orb_a_1,
                                                                           blackholes_binary.orb_a_2,
                                                                           blackholes_binary.bin_ecc,
                                                                           blackholes_binary.id_num)
                    if bh_binary_id_num_unphysical.size > 0:
                        blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                        filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)
                    blackholes_merged, blackholes_pro = merge.merge_blackholes(blackholes_binary,
                                                                               blackholes_pro,
                                                                               blackholes_merged,
                                                                               bh_binary_id_num_merger,
                                                                               opts.smbh_mass,
                                                                               opts.flag_use_surrogate,
                                                                               disk_aspect_ratio,
                                                                               disk_density,
                                                                               time_passed,
                                                                               galaxy)

                    # Update filing cabinet
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="category",
                                          new_info=np.full(bh_binary_id_num_merger.size, 0))
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="mass",
                                          new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "mass"))
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="orb_ecc",
                                          new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "orb_ecc"))
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="size",
                                          new_info=np.full(bh_binary_id_num_merger.size, -1.5))
                    blackholes_binary.remove_id_num(bh_binary_id_num_merger)

                    if opts.verbose:
                        print("New BH locations", blackholes_pro.orb_a)
                else:
                    # No merger
                    # do nothing! hardening should happen FIRST (and now it does!)
                    if (opts.verbose):
                        print("No mergers yet")

                # Accrete gas onto binary components
                blackholes_binary.mass_1, blackholes_binary.mass_2 = evolve.change_bin_mass(
                    blackholes_binary.mass_1,
                    blackholes_binary.mass_2,
                    blackholes_binary.flag_merging,
                    opts.disk_bh_eddington_ratio,
                    disk_bh_eddington_mass_growth_rate,
                    opts.timestep_duration_yr,
                )

                # Update filing cabinet
                filing_cabinet.update(id_num=blackholes_binary.id_num,
                                      attr="mass",
                                      new_info=blackholes_binary.mass_1 + blackholes_binary.mass_2)

                # Spin up binary components
                blackholes_binary.spin_1, blackholes_binary.spin_2 = evolve.change_bin_spin_magnitudes(
                    blackholes_binary.spin_1,
                    blackholes_binary.spin_2,
                    blackholes_binary.flag_merging,
                    opts.disk_bh_eddington_ratio,
                    opts.disk_bh_torque_condition,
                    opts.timestep_duration_yr,
                )

                # Torque angle of binary spin components
                blackholes_binary.spin_angle_1, blackholes_binary.spin_angle_2 = evolve.change_bin_spin_angles(
                    blackholes_binary.spin_angle_1,
                    blackholes_binary.spin_angle_2,
                    blackholes_binary.flag_merging,
                    opts.disk_bh_eddington_ratio,
                    opts.disk_bh_torque_condition,
                    disk_bh_spin_resolution_min,
                    opts.timestep_duration_yr,
                )

                if (opts.flag_dynamic_enc > 0):
                    # Spheroid encounters
                    # FIX THIS: Replace nsc_imf_bh below with nsc_imf_stars_ since pulling from stellar MF
                    blackholes_binary.bin_sep, blackholes_binary.bin_ecc, blackholes_binary.bin_orb_ecc, blackholes_binary.bin_orb_inc = dynamics.bin_spheroid_encounter(
                        opts.smbh_mass,
                        opts.timestep_duration_yr,
                        blackholes_binary.mass_1,
                        blackholes_binary.mass_2,
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.bin_sep,
                        blackholes_binary.bin_ecc,
                        blackholes_binary.bin_orb_ecc,
                        blackholes_binary.bin_orb_inc,
                        time_passed,
                        opts.nsc_imf_bh_powerlaw_index,
                        opts.delta_energy_strong,
                        opts.nsc_spheroid_normalization,
                        opts.mean_harden_energy_delta,
                        opts.var_harden_energy_delta
                    )
                    # Update filing cabinet with new bin_sep
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="size",
                                          new_info=blackholes_binary.bin_sep)
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="orb_ecc",
                                          new_info=blackholes_binary.bin_orb_ecc)

                    # Check for mergers
                    # Check closeness of binary. Are black holes at merger condition separation
                    blackholes_binary.bin_sep, blackholes_binary.flag_merging = evolve.bin_contact_check(blackholes_binary.mass_1, blackholes_binary.mass_2, blackholes_binary.bin_sep, blackholes_binary.flag_merging, opts.smbh_mass)
                    bh_binary_id_num_merger = blackholes_binary.id_num[blackholes_binary.flag_merging < 0]

                    if opts.verbose:
                        print("Merger ID numbers")
                        print(bh_binary_id_num_merger)

                    if (bh_binary_id_num_merger.size > 0):

                        bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary.mass_1,
                                                                               blackholes_binary.mass_2,
                                                                               blackholes_binary.orb_a_1,
                                                                               blackholes_binary.orb_a_2,
                                                                               blackholes_binary.bin_ecc,
                                                                               blackholes_binary.id_num)
                        if bh_binary_id_num_unphysical.size > 0:
                            blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                            filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)

                        blackholes_merged, blackholes_pro = merge.merge_blackholes(blackholes_binary,
                                                                                   blackholes_pro,
                                                                                   blackholes_merged,
                                                                                   bh_binary_id_num_merger,
                                                                                   opts.smbh_mass,
                                                                                   opts.flag_use_surrogate,
                                                                                   disk_aspect_ratio,
                                                                                   disk_density,
                                                                                   time_passed,
                                                                                   galaxy)

                        # Update filing cabinet
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="category",
                                              new_info=np.full(bh_binary_id_num_merger.size, 0))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="mass",
                                              new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "mass"))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="orb_ecc",
                                              new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "orb_ecc"))
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="size",
                                              new_info=np.full(bh_binary_id_num_merger.size, -1.5))
                        blackholes_binary.remove_id_num(bh_binary_id_num_merger)

                        if opts.verbose:
                            print("New BH locations", blackholes_pro.orb_a)
                    else:
                        # No merger
                        # do nothing! hardening should happen FIRST (and now it does!)
                        if (opts.verbose):
                            print("No mergers yet")

                if (opts.flag_dynamic_enc > 0):
                    # Recapture bins out of disk plane.
                    # FIX THIS: Replace this with orb_inc_damping but for binary bhbh OBJECTS (KN)
                    blackholes_binary.bin_orb_inc = dynamics.bin_recapture(
                        blackholes_binary.mass_1,
                        blackholes_binary.mass_2,
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.bin_orb_inc,
                        opts.timestep_duration_yr
                    )

                # Migrate binaries
                # First if feedback present, find ratio of feedback heating torque to migration torque
                if opts.flag_thermal_feedback > 0:
                    ratio_heat_mig_torques_bin_com = evolve.bin_com_feedback_hankla(
                        blackholes_binary.bin_orb_a,
                        disk_surface_density,
                        disk_opacity,
                        opts.disk_bh_eddington_ratio,
                        opts.disk_alpha_viscosity,
                        opts.disk_radius_outer
                    )
                else:
                    ratio_heat_mig_torques_bin_com = np.ones(blackholes_binary.num)

                # Migrate binaries center of mass
                # Choose torque prescription for binary migration
                # Old is the original approximation used in v.0.1.0, based off (but not identical to Paardekooper 2010)-usually within factor [0.5-2]
                if opts.torque_prescription == 'old':
                    blackholes_binary.bin_orb_a = migration.type1_migration_binary(
                        opts.smbh_mass, blackholes_binary.mass_1, blackholes_binary.mass_2,
                        blackholes_binary.bin_orb_a, blackholes_binary.bin_orb_ecc,
                        opts.disk_bh_pro_orb_ecc_crit,
                        disk_surface_density, disk_aspect_ratio, ratio_heat_mig_torques_bin_com,
                        opts.disk_radius_trap, opts.disk_radius_outer, opts.timestep_duration_yr)

                # Alternatively, calculate actual torques from disk profiles.
                # Paardekooper torque coeff (default)
                if opts.torque_prescription == 'paardekooper':
                    paardekooper_torque_coeff_bh = migration.paardekooper10_torque(
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.bin_orb_ecc,
                        opts.disk_bh_pro_orb_ecc_crit,
                        disk_dlog10surfdens_dlog10R_func,
                        disk_dlog10temp_dlog10R_func
                    )

                # Jimenez-Masset torque coeff (from Grishin+24)
                if opts.torque_prescription == 'jimenez_masset':
                    jimenez_masset_torque_coeff_bh = migration.jimenezmasset17_torque(
                        opts.smbh_mass,
                        disk_surface_density,
                        disk_opacity,
                        disk_aspect_ratio,
                        temp_func,
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.bin_orb_ecc,
                        opts.disk_bh_pro_orb_ecc_crit,
                        disk_dlog10surfdens_dlog10R_func,
                        disk_dlog10temp_dlog10R_func
                    )
                    jimenez_masset_thermal_torque_coeff_bh = migration.jimenezmasset17_thermal_torque_coeff(
                        opts.smbh_mass,
                        disk_surface_density,
                        disk_opacity,
                        disk_aspect_ratio,
                        temp_func,
                        opts.disk_bh_eddington_ratio,
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.bin_orb_ecc,
                        opts.disk_bh_pro_orb_ecc_crit,
                        blackholes_binary.mass_1 + blackholes_binary.mass_2,
                        opts.flag_thermal_feedback,
                        disk_dlog10pressure_dlog10R_func
                    )
                    if opts.flag_thermal_feedback > 0:
                        total_jimenez_masset_torque_bh = jimenez_masset_torque_coeff_bh + jimenez_masset_thermal_torque_coeff_bh
                    else:
                        total_jimenez_masset_torque_bh = jimenez_masset_torque_coeff_bh
                # Normalized torque (multiplies torque coeff)
                if opts.torque_prescription == 'paardekooper' or opts.torque_prescription == 'jimenez_masset':
                    normalized_torque_bh = migration.normalized_torque(
                        opts.smbh_mass,
                        blackholes_binary.bin_orb_a,
                        blackholes_binary.mass_1 + blackholes_binary.mass_2,
                        blackholes_binary.bin_orb_ecc,
                        opts.disk_bh_pro_orb_ecc_crit,
                        disk_surface_density,
                        disk_aspect_ratio
                    )

                    if np.size(normalized_torque_bh) > 0:
                        if opts.torque_prescription == 'paardekooper':
                            torque = paardekooper_torque_coeff_bh*normalized_torque_bh
                            disk_trap_radius = opts.disk_radius_trap
                            disk_anti_trap_radius = opts.disk_radius_trap
                        if opts.torque_prescription == 'jimenez_masset':
                            torque = total_jimenez_masset_torque_bh*normalized_torque_bh
                        # Set up trap scaling as a function of mass for Jimenez-Masset (for SG-like disk)
                        # No traps if M_smbh >10^8Msun (approx.)
                            if opts.smbh_mass > 1.e8:
                                disk_trap_radius = opts.disk_inner_stable_circ_orb
                                disk_anti_trap_radius = opts.disk_inner_stable_circ_orb
                            if opts.smbh_mass == 1.e8:
                                disk_trap_radius = opts.disk_radius_trap
                                disk_anti_trap_radius = opts.disk_radius_trap
                            # Trap changes as a function of r_g if M_smbh <10^8Msun (default trap radius ~700r_g). Grishin+24
                            if opts.smbh_mass < 1.e8 and opts.smbh_mass > 1.e6:
                                disk_trap_radius = opts.disk_radius_trap * (opts.smbh_mass/1.e8)**(-1.225)
                                disk_anti_trap_radius = opts.disk_radius_trap * (opts.smbh_mass/1.e8)**(0.099)
                            # Trap location changes again at low SMBH mass (Grishin+24)
                            if opts.smbh_mass < 1.e6:
                                disk_trap_radius = opts.disk_radius_trap * (opts.smbh_mass/1.e8)**(-0.97)
                                disk_anti_trap_radius = opts.disk_radius_trap * (opts.smbh_mass/1.e8)**(0.099)

                        torque_mig_timescales_bh = migration.torque_mig_timescale(
                            opts.smbh_mass,
                            blackholes_binary.bin_orb_a,
                            blackholes_binary.mass_1 + blackholes_binary.mass_2,
                            blackholes_binary.bin_orb_ecc,
                            opts.disk_bh_pro_orb_ecc_crit,
                            torque
                        )
                        # Calculate new bh_orbs_a using torque
                        blackholes_binary.bin_orb_a = migration.type1_migration_distance(
                            opts.smbh_mass,
                            blackholes_binary.bin_orb_a,
                            blackholes_binary.mass_1 + blackholes_binary.mass_2,
                            blackholes_binary.bin_orb_ecc,
                            opts.disk_bh_pro_orb_ecc_crit,
                            torque_mig_timescales_bh,
                            ratio_heat_mig_torques_bin_com,
                            disk_trap_radius,
                            disk_anti_trap_radius,
                            opts.disk_radius_outer,
                            opts.timestep_duration_yr,
                            opts.flag_phenom_turb,
                            opts.phenom_turb_centroid,
                            opts.phenom_turb_std_dev,
                            opts.nsc_imf_bh_mode,
                            opts.torque_prescription
                        )

                # Update filing cabinet
                filing_cabinet.update(id_num=blackholes_binary.id_num,
                                      attr="orb_a",
                                      new_info=blackholes_binary.bin_orb_a)

                # Test to see if any binaries separation is O(1r_g)
                # If so, track them for GW freq, strain.
                # Minimum BBH separation (in units of r_g)
                min_bbh_gw_separation = 2.0
                # If there are binaries AND if any separations are < min_bbh_gw_separation
                bh_binary_id_num_gw = blackholes_binary.id_num[(blackholes_binary.bin_sep < min_bbh_gw_separation) & (blackholes_binary.bin_sep > 0)]
                if (bh_binary_id_num_gw.size > 0):
                    # 1st time around.
                    if num_bbh_gw_tracked == 0:
                        old_bbh_gw_freq = 9.e-7*np.ones(bh_binary_id_num_gw.size)
                    if num_bbh_gw_tracked > 0:
                        old_bbh_gw_freq = blackholes_binary.at_id_num(bh_binary_id_num_gw, "gw_freq")

                    num_bbh_gw_tracked = bh_binary_id_num_gw.size

                    bbh_gw_strain, bbh_gw_freq = gw.bbh_gw_params(
                        blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_1"),
                        blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_2"),
                        blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_sep"),
                        opts.smbh_mass,
                        opts.timestep_duration_yr,
                        old_bbh_gw_freq,
                        agn_redshift
                        )

                    blackholes_binary_gw.add_binaries(
                        new_id_num=bh_binary_id_num_gw,
                        new_mass_1=blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_1"),
                        new_mass_2=blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_2"),
                        new_orb_a_1=blackholes_binary.at_id_num(bh_binary_id_num_gw, "orb_a_1"),
                        new_orb_a_2=blackholes_binary.at_id_num(bh_binary_id_num_gw, "orb_a_2"),
                        new_spin_1=blackholes_binary.at_id_num(bh_binary_id_num_gw, "spin_1"),
                        new_spin_2=blackholes_binary.at_id_num(bh_binary_id_num_gw, "spin_2"),
                        new_spin_angle_1=blackholes_binary.at_id_num(bh_binary_id_num_gw, "spin_angle_1"),
                        new_spin_angle_2=blackholes_binary.at_id_num(bh_binary_id_num_gw, "spin_angle_2"),
                        new_bin_sep=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_sep"),
                        new_bin_orb_a=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_orb_a"),
                        new_time_to_merger_gw=blackholes_binary.at_id_num(bh_binary_id_num_gw, "time_to_merger_gw"),
                        new_flag_merging=blackholes_binary.at_id_num(bh_binary_id_num_gw, "flag_merging"),
                        new_time_merged=np.full(bh_binary_id_num_gw.size, time_passed),
                        new_bin_ecc=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_ecc"),
                        new_gen_1=blackholes_binary.at_id_num(bh_binary_id_num_gw, "gen_1"),
                        new_gen_2=blackholes_binary.at_id_num(bh_binary_id_num_gw, "gen_2"),
                        new_bin_orb_ang_mom=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_orb_ang_mom"),
                        new_bin_orb_inc=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_orb_inc"),
                        new_bin_orb_ecc=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_orb_ecc"),
                        new_gw_freq=bbh_gw_freq,
                        new_gw_strain=bbh_gw_strain,
                        new_galaxy=np.full(bh_binary_id_num_gw.size, galaxy),
                    )

                # Evolve GW frequency and strain
                blackholes_binary.gw_freq, blackholes_binary.gw_strain = gw.evolve_gw(
                    blackholes_binary.mass_1,
                    blackholes_binary.mass_2,
                    blackholes_binary.bin_sep,
                    opts.smbh_mass,
                    agn_redshift
                )

                # Check and see if any binaries are ionized.
                bh_binary_id_num_ionization = evolve.bin_ionization_check(blackholes_binary.mass_1, blackholes_binary.mass_2, blackholes_binary.bin_orb_a, blackholes_binary.bin_sep, blackholes_binary.id_num, opts.smbh_mass)
                if bh_binary_id_num_ionization.size > 0:
                    # Append 2 new BH to arrays of single BH locations, masses, spins, spin angles & gens
                    # For now add 2 new orb ecc term of 0.01. inclination is 0.0 as well. TO DO: calculate v_kick and resulting perturbation to orb ecc.

                    new_orb_ecc = eccentricity.ionized_orb_ecc(bh_binary_id_num_ionization.size * 2, opts.disk_bh_orb_ecc_max_init)
                    new_id_nums = np.arange(filing_cabinet.id_max+1, filing_cabinet.id_max + 1 + bh_binary_id_num_ionization.size * 2, 1)
                    blackholes_pro.add_blackholes(
                        new_mass=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "mass_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "mass_2")]),
                        new_spin=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "spin_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "spin_2")]),
                        new_spin_angle=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "spin_angle_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "spin_angle_2")]),
                        new_orb_a=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "orb_a_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "orb_a_2")]),
                        new_gen=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "gen_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "gen_2")]),
                        new_orb_ecc=new_orb_ecc,
                        new_orb_inc=np.full(bh_binary_id_num_ionization.size * 2, 0.0),
                        new_orb_ang_mom=np.ones(bh_binary_id_num_ionization.size * 2),
                        new_orb_arg_periapse=np.full(bh_binary_id_num_ionization.size * 2, -1.5),
                        new_gw_freq=np.full(bh_binary_id_num_ionization.size * 2, -1.5),
                        new_gw_strain=np.full(bh_binary_id_num_ionization.size * 2, -1.5),
                        new_galaxy=np.full(bh_binary_id_num_ionization.size * 2, galaxy),
                        new_time_passed=np.full(bh_binary_id_num_ionization.size * 2, time_passed),
                        new_id_num=new_id_nums
                    )

                    # Update filing cabinet
                    filing_cabinet.add_objects(
                        new_id_num=new_id_nums,
                        new_category=np.zeros(bh_binary_id_num_ionization.size * 2),
                        new_orb_a=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "orb_a_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "orb_a_2")]),
                        new_mass=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "mass_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "mass_2")]),
                        new_orb_ecc=new_orb_ecc,
                        new_size=np.full(bh_binary_id_num_ionization.size * 2, -1.5),
                        new_direction=np.ones(bh_binary_id_num_ionization.size * 2),
                        new_disk_inner_outer=np.ones(bh_binary_id_num_ionization.size * 2)
                    )

                    blackholes_binary.remove_id_num(bh_binary_id_num_ionization)
                    filing_cabinet.remove_id_num(bh_binary_id_num_ionization)

                bh_binary_id_num_merger = blackholes_binary.id_num[blackholes_binary.flag_merging < 0]

                if opts.verbose:
                    print("Merger ID numbers")
                    print(bh_binary_id_num_merger)

                if (bh_binary_id_num_merger.size > 0):

                    bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary.mass_1,
                                                                           blackholes_binary.mass_2,
                                                                           blackholes_binary.orb_a_1,
                                                                           blackholes_binary.orb_a_2,
                                                                           blackholes_binary.bin_ecc,
                                                                           blackholes_binary.id_num)
                    if bh_binary_id_num_unphysical.size > 0:
                        blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                        filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)

                    blackholes_merged, blackholes_pro = merge.merge_blackholes(blackholes_binary,
                                                                               blackholes_pro,
                                                                               blackholes_merged,
                                                                               bh_binary_id_num_merger,
                                                                               opts.smbh_mass,
                                                                               opts.flag_use_surrogate,
                                                                               disk_aspect_ratio,
                                                                               disk_density,
                                                                               time_passed,
                                                                               galaxy)

                    # Update filing cabinet
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="category",
                                          new_info=np.full(bh_binary_id_num_merger.size, 0))
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="mass",
                                          new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "mass"))
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="orb_ecc",
                                          new_info=blackholes_pro.at_id_num(bh_binary_id_num_merger, "orb_ecc"))
                    filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                          attr="size",
                                          new_info=np.full(bh_binary_id_num_merger.size, -1.5))
                    blackholes_binary.remove_id_num(bh_binary_id_num_merger)

                    if opts.verbose:
                        print("New BH locations", blackholes_pro.orb_a)
                else:
                    # No merger
                    # do nothing! hardening should happen FIRST (and now it does!)
                    if (opts.verbose):
                        print("No mergers yet")
            else:
                if opts.verbose:
                    print("No binaries formed yet")
                    # No Binaries present in bin_array. Nothing to do.
                # Finished evolving binaries

            # If a close encounter within mutual Hill sphere objects will form a binary or otherwise interact
            # check which objects have a close encounter within mutual Hill sphere
            id_nums_check = filing_cabinet.id_num[((filing_cabinet.category == 0) | (filing_cabinet.category == 1)) &  # single BH (0) or star (1)
                                                   (filing_cabinet.direction == 1) &  # prograde
                                                   (filing_cabinet.disk_inner_outer == 1)]  # in the outer disk

            close_encounters_id_nums = formation.close_encounters_check(id_nums_check,
                                                                        filing_cabinet,
                                                                        opts.smbh_mass,
                                                                        opts.disk_bh_pro_orb_ecc_crit)

            if (close_encounters_id_nums.shape != (0,)):
                # Get ID nums for BH pairs, star pairs, and BH-star pairs
                bhbh_id_nums, starstar_id_nums, bhstar_id_nums = formation.divide_types_encounters(close_encounters_id_nums, [[0, 0], [1, 1], [0, 1]], filing_cabinet)

                if (bhbh_id_nums.size > 0):
                    # BH and BH encounter each other: form a binary
                    blackholes_binary, bh_binary_id_num_new = formation.add_to_binary_obj(
                        blackholes_binary,
                        blackholes_pro,
                        bhbh_id_nums,
                        filing_cabinet.id_max,
                        opts.fraction_bin_retro,
                        opts.smbh_mass,
                        agn_redshift,
                        opts.disk_bh_pro_orb_ecc_crit
                    )

                    # Add new BH binaries to filing cabinet and delete prograde singleton black holes
                    filing_cabinet.add_objects(new_id_num=bh_binary_id_num_new,
                                               new_category=np.full(bh_binary_id_num_new.size, 2),
                                               new_orb_a=blackholes_binary.at_id_num(bh_binary_id_num_new, "bin_orb_a"),
                                               new_mass=blackholes_binary.at_id_num(bh_binary_id_num_new, "mass_1") + blackholes_binary.at_id_num(bh_binary_id_num_new, "mass_2"),
                                               new_orb_ecc=blackholes_binary.at_id_num(bh_binary_id_num_new, "bin_orb_ecc"),
                                               new_size=blackholes_binary.at_id_num(bh_binary_id_num_new, "bin_sep"),
                                               new_direction=np.full(bh_binary_id_num_new.size, 1),
                                               new_disk_inner_outer=np.full(bh_binary_id_num_new.size, 1))
                    filing_cabinet.remove_id_num(id_num_remove=bhbh_id_nums.flatten())

                    # delete corresponding entries for new binary members from singleton arrays
                    blackholes_pro.remove_id_num(id_num_remove=bhbh_id_nums.flatten())

                if (bhstar_id_nums.size > 0):
                    # BH and star encounter: star blows up, BH accretes mass
                    # Separate out into BH ID and star ID
                    bhstar_id_nums = bhstar_id_nums.flatten()
                    star_id_nums = bhstar_id_nums[np.nonzero(filing_cabinet.at_id_num(bhstar_id_nums, "category") == 1)]
                    bh_id_nums = bhstar_id_nums[np.nonzero(filing_cabinet.at_id_num(bhstar_id_nums, "category") == 0)]
                    stars_explode.add_stars(new_id_num_star=star_id_nums,
                                            new_id_num_bh=bh_id_nums,
                                            new_mass_star=stars_pro.at_id_num(star_id_nums, "mass"),
                                            new_mass_bh=blackholes_pro.at_id_num(bh_id_nums, "mass"),
                                            new_orb_a_star=stars_pro.at_id_num(star_id_nums, "orb_a"),
                                            new_orb_a_bh=blackholes_pro.at_id_num(bh_id_nums, "orb_a"),
                                            new_star_log_radius=stars_pro.at_id_num(star_id_nums, "log_radius"),
                                            new_orb_inc_star=stars_pro.at_id_num(star_id_nums, "orb_inc"),
                                            new_orb_inc_bh=blackholes_pro.at_id_num(bh_id_nums, "orb_inc"),
                                            new_orb_ecc_star=stars_pro.at_id_num(star_id_nums, "orb_ecc"),
                                            new_orb_ecc_bh=blackholes_pro.at_id_num(bh_id_nums, "orb_ecc"),
                                            new_gen_star=stars_pro.at_id_num(star_id_nums, "gen"),
                                            new_gen_bh=blackholes_pro.at_id_num(bh_id_nums, "gen"),
                                            new_galaxy=stars_pro.at_id_num(star_id_nums, "galaxy"),
                                            new_time_sn=np.full(star_id_nums.size, time_passed),
                                            )
                    disk_mass_gained.append(stars_pro.at_id_num(star_id_nums, "mass").sum())
                    # Delete exploded stars from regular array and filing cabinet
                    stars_pro.remove_id_num(star_id_nums)
                    filing_cabinet.remove_id_num(star_id_nums)

                    # BHs accrete mass and spin up
                    _, bh_id_mask = np.where(blackholes_pro.id_num == bh_id_nums[:, None])
                    blackholes_pro.mass[bh_id_mask] = accretion.change_bh_mass(
                        blackholes_pro.mass[bh_id_mask],
                        opts.disk_bh_eddington_ratio,
                        disk_bh_eddington_mass_growth_rate,
                        opts.timestep_duration_yr)

                    blackholes_pro.spin[bh_id_mask] = accretion.change_bh_spin_magnitudes(
                        blackholes_pro.spin[bh_id_mask],
                        opts.disk_bh_eddington_ratio,
                        opts.disk_bh_torque_condition,
                        opts.timestep_duration_yr,
                        blackholes_pro.orb_ecc[bh_id_mask],
                        opts.disk_bh_pro_orb_ecc_crit,
                    )

                    blackholes_pro.spin_angle[bh_id_mask] = accretion.change_bh_spin_angles(
                        blackholes_pro.spin_angle[bh_id_mask],
                        opts.disk_bh_eddington_ratio,
                        opts.disk_bh_torque_condition,
                        disk_bh_spin_resolution_min,
                        opts.timestep_duration_yr,
                        blackholes_pro.orb_ecc[bh_id_mask],
                        opts.disk_bh_pro_orb_ecc_crit
                    )

                    # Update filing cabinet
                    filing_cabinet.update(id_num=bh_id_nums,
                                          attr="mass",
                                          new_info=blackholes_pro.mass[bh_id_mask])

                if (starstar_id_nums.size > 0):
                    # Star and star encounter each other: stellar merger
                    # Generate new ID numbers
                    star_merged_id_num_new = np.arange(filing_cabinet.id_max + 1, filing_cabinet.id_max + 1 + starstar_id_nums.shape[1], 1)
                    # Merged mass is just masses added together. Any over disk_star_initial_mass_cutoff get set to disk_star_initial_mass_cutoff
                    star_merged_mass = stars_pro.at_id_num(starstar_id_nums[0], "mass") + stars_pro.at_id_num(starstar_id_nums[1], "mass")
                    # New orb_a is the center of mass of the two stars
                    star_merged_orbs_a = ((stars_pro.at_id_num(starstar_id_nums[0], "mass") * stars_pro.at_id_num(starstar_id_nums[0], "orb_a")) +
                                          (stars_pro.at_id_num(starstar_id_nums[1], "mass") * stars_pro.at_id_num(starstar_id_nums[1], "orb_a"))) / star_merged_mass
                    assert np.all(star_merged_orbs_a < opts.disk_radius_outer), "star_merged_orbs_a has values greater than disk_radius_outer"
                    # After doing the weighted average for orb_a we then cut off stars with mass > disk_star_initial_mass_cutoff
                    star_merged_mass[star_merged_mass > opts.disk_star_initial_mass_cutoff] = opts.disk_star_initial_mass_cutoff
                    # Radius, luminosity, Teff are all interpolated based on the new mass
                    star_merged_logR, star_merged_logL, star_merged_logTeff = stellar_interpolation.interp_star_params(star_merged_mass)
                    # New gen is the maximum between the pair's gen plus one
                    star_merged_gen = np.maximum(stars_pro.at_id_num(starstar_id_nums[0], "gen"), stars_pro.at_id_num(starstar_id_nums[1], "gen")) + 1.0

                    # Add to stars object
                    stars_pro.add_stars(new_id_num=star_merged_id_num_new,
                                        new_mass=star_merged_mass,
                                        new_orb_a=star_merged_orbs_a,
                                        new_log_radius=star_merged_logR,
                                        new_log_teff=star_merged_logTeff,
                                        new_log_luminosity=star_merged_logL,
                                        new_X=stars_pro.at_id_num(starstar_id_nums[0], "star_X"),  # no metallicity evolution
                                        new_Y=stars_pro.at_id_num(starstar_id_nums[0], "star_Y"),
                                        new_Z=stars_pro.at_id_num(starstar_id_nums[0], "star_Z"),
                                        new_orb_ang_mom=np.ones(starstar_id_nums.shape[1]),  # orb_ang_mom is +1 because two prograde stars merge
                                        new_orb_ecc=np.full(starstar_id_nums.shape[1], opts.disk_bh_pro_orb_ecc_crit),  # orb_ecc is initially very small
                                        new_orb_inc=np.zeros(starstar_id_nums.shape[1]),  # orb_inc is zero
                                        new_orb_arg_periapse=stars_pro.at_id_num(starstar_id_nums[0], "orb_arg_periapse"),  # Assume orb_arg_periapse is same as before
                                        new_gen=star_merged_gen,
                                        new_galaxy=stars_pro.at_id_num(starstar_id_nums[0], "galaxy"),
                                        new_time_passed=stars_pro.at_id_num(starstar_id_nums[0], "time_passed"))

                    # Add new merged stars to merged stars object
                    stars_merge.add_stars(new_id_num=star_merged_id_num_new,
                                          new_galaxy=stars_pro.at_id_num(starstar_id_nums[0], "galaxy"),
                                          new_orb_a_final=star_merged_orbs_a,
                                          new_gen_final=star_merged_gen,
                                          new_mass_final=star_merged_mass,
                                          new_mass_1=stars_pro.at_id_num(starstar_id_nums[0], "mass"),
                                          new_mass_2=stars_pro.at_id_num(starstar_id_nums[1], "mass"),
                                          new_gen_1=stars_pro.at_id_num(starstar_id_nums[0], "gen"),
                                          new_gen_2=stars_pro.at_id_num(starstar_id_nums[1], "gen"),
                                          new_log_radius_final=star_merged_logR,
                                          new_orb_ecc=np.full(starstar_id_nums.shape[1], opts.disk_bh_pro_orb_ecc_crit),
                                          new_time_merged=np.full(starstar_id_nums.shape[1], time_passed))

                    # Add new merged stars to filing cabinet and delete previous stars
                    filing_cabinet.add_objects(new_id_num=star_merged_id_num_new,
                                               new_category=np.ones(star_merged_id_num_new.size),
                                               new_orb_a=stars_pro.at_id_num(star_merged_id_num_new, "orb_a"),
                                               new_mass=stars_pro.at_id_num(star_merged_id_num_new, "mass"),
                                               new_orb_ecc=stars_pro.at_id_num(star_merged_id_num_new, "orb_ecc"),
                                               new_size=point_masses.r_g_from_units(opts.smbh_mass, (10 ** stars_pro.at_id_num(star_merged_id_num_new, "log_radius")) * u.Rsun).value,
                                               new_direction=np.ones(star_merged_id_num_new.size),
                                               new_disk_inner_outer=np.ones(star_merged_id_num_new.size))
                    filing_cabinet.remove_id_num(starstar_id_nums.flatten())
                    stars_pro.remove_id_num(starstar_id_nums.flatten())

            # After this time period, was there a disk capture via orbital grind-down?
            # To do: What eccentricity do we want the captured BH to have? Right now ecc=0.0? Should it be ecc<h at a?             
            # Assume 1st gen BH captured and orb ecc =0.0
            # To do: Bias disk capture to more massive BH!
            # Assuming captured objects are not in the inner disk? (KN)
            capture = time_passed % opts.capture_time_yr
            if capture == 0:
                bh_orb_a_captured = setupdiskblackholes.setup_disk_blackholes_location_NSC_powerlaw(
                    1, opts.disk_radius_capture_outer, opts.disk_inner_stable_circ_orb,
                    opts.smbh_mass, opts.nsc_radius_crit, opts.nsc_density_index_inner,
                    opts.nsc_density_index_outer, volume_scaling=True)
                bh_mass_captured = setupdiskblackholes.setup_disk_blackholes_masses(
                    1, opts.nsc_imf_bh_mode, opts.nsc_imf_bh_mass_max, opts.nsc_imf_bh_powerlaw_index, opts.mass_pile_up)
                bh_spin_captured = setupdiskblackholes.setup_disk_blackholes_spins(
                    1, opts.nsc_bh_spin_dist_mu, opts.nsc_bh_spin_dist_sigma)
                bh_spin_angle_captured = setupdiskblackholes.setup_disk_blackholes_spin_angles(
                    1, bh_spin_captured)
                bh_gen_captured = [1]
                bh_orb_ecc_captured = [0.0]
                bh_orb_inc_captured = [0.0]
                bh_id_num_captured = np.arange(filing_cabinet.id_max+1, len(bh_mass_captured) + filing_cabinet.id_max+1, 1)
                # Append captured BH to existing singleton arrays. Assume prograde and 1st gen BH.
                blackholes_pro.add_blackholes(new_mass=bh_mass_captured,
                                              new_spin=bh_spin_captured,
                                              new_spin_angle=bh_spin_angle_captured,
                                              new_orb_a=bh_orb_a_captured,
                                              new_orb_inc=bh_orb_inc_captured,
                                              new_orb_ang_mom=np.ones(bh_mass_captured.size),
                                              new_orb_ecc=bh_orb_ecc_captured,
                                              new_orb_arg_periapse=np.full(bh_mass_captured.size, -1.5),
                                              new_gen=bh_gen_captured,
                                              new_galaxy=np.full(len(bh_mass_captured),galaxy),
                                              new_time_passed=np.full(len(bh_mass_captured),time_passed),
                                              new_id_num=bh_id_num_captured)
                # Update filing cabinet
                filing_cabinet.add_objects(new_id_num=bh_id_num_captured,
                                           new_category=np.array([0.0]),
                                           new_orb_a=bh_orb_a_captured,
                                           new_mass=bh_mass_captured,
                                           new_orb_ecc=bh_orb_ecc_captured,
                                           new_size=np.array([-1]),
                                           new_direction=np.array([1.0]),
                                           new_disk_inner_outer=np.array([1.0]))

            # Starter code for stars being captured
            # Conditions copy BH capture right now, but need to incorporate WZL2024 conditions
            if opts.flag_add_stars:
                capture_stars = time_passed % opts.capture_time_yr
                if capture_stars == 0:
                    num_star_captured = 2
                    star_mass_captured = setupdiskstars.setup_disk_stars_masses(star_num=num_star_captured,
                                                                                disk_star_mass_min_init=opts.disk_star_mass_min_init,
                                                                                disk_star_mass_max_init=opts.disk_star_mass_max_init,
                                                                                nsc_imf_star_powerlaw_index=opts.nsc_imf_star_powerlaw_index)
                    star_orb_a_captured = setupdiskstars.setup_disk_stars_orb_a(star_num=num_star_captured,
                                                                                disk_radius_outer=opts.disk_radius_outer,
                                                                                disk_inner_stable_circ_orb=opts.disk_inner_stable_circ_orb)
                    star_orb_inc_captured = np.full(num_star_captured, 0.0)  # setupdiskstars.setup_disk_stars_inclination(num_star_captured)
                    star_orb_ang_mom_captured = setupdiskstars.setup_disk_stars_orb_ang_mom(num_star_captured)
                    star_orb_arg_periapse_captured = setupdiskstars.setup_disk_stars_arg_periapse(num_star_captured)
                    star_orb_ecc_captured = np.full(num_star_captured, 0.0)
                    star_X_captured, star_Y_captured, star_Z_captured = setupdiskstars.setup_disk_stars_comp(star_num=num_star_captured,
                                                                                                             star_ZAMS_metallicity=opts.nsc_star_metallicity_z_init,
                                                                                                             star_ZAMS_helium=opts.nsc_star_metallicity_y_init)
                    star_log_radius_captured, star_log_luminosity_captured, star_log_teff_captured = stellar_interpolation.interp_star_params(star_mass_captured)
                    # Append captured stars to stars_pro array. Assume prograde and 1st gen.
                    stars_pro.add_stars(new_mass=star_mass_captured,
                                        new_orb_a=star_orb_a_captured,
                                        new_log_radius=star_log_radius_captured,
                                        new_log_luminosity=star_log_luminosity_captured,
                                        new_log_teff=star_log_teff_captured,
                                        new_X=star_X_captured,
                                        new_Y=star_Y_captured,
                                        new_Z=star_Z_captured,
                                        new_orb_inc=star_orb_inc_captured,
                                        new_orb_ang_mom=star_orb_ang_mom_captured,
                                        new_orb_ecc=star_orb_ecc_captured,
                                        new_orb_arg_periapse=star_orb_arg_periapse_captured,
                                        new_gen=np.full(num_star_captured, 1),
                                        new_galaxy=np.full(num_star_captured, galaxy),
                                        new_time_passed=np.full(num_star_captured, time_passed),
                                        new_id_num=np.arange(filing_cabinet.id_max + 1, num_star_captured + filing_cabinet.id_max + 1, 1)
                                        )
                    # Update filing cabinet
                    filing_cabinet.add_objects(new_id_num=np.arange(filing_cabinet.id_max + 1, num_star_captured + filing_cabinet.id_max + 1, 1),
                                               new_category=np.ones(num_star_captured),
                                               new_orb_a=star_orb_a_captured,
                                               new_mass=star_mass_captured,
                                               new_orb_ecc=star_orb_ecc_captured,
                                               new_size=point_masses.r_g_from_units(opts.smbh_mass, (10 ** star_log_radius_captured) * u.Rsun).value,
                                               new_direction=np.ones(num_star_captured),
                                               new_disk_inner_outer=np.zeros(num_star_captured))

            # Test if any BH or BBH are in the danger-zone (<mininum_safe_distance, default =50r_g) from SMBH.
            # Potential EMRI/BBH EMRIs.
            # Find prograde BH in inner disk. Define inner disk as <=50r_g. 
            # Since a 10Msun BH will decay into a 10^8Msun SMBH at 50R_g in ~38Myr and decay time propto a^4.
            # e.g at 25R_g, decay time is only 2.3Myr.

            # Check if any prograde BHs are in the inner disk
            bh_id_num_pro_inner_disk = blackholes_pro.id_num[blackholes_pro.orb_a < opts.inner_disk_outer_radius]
            if (bh_id_num_pro_inner_disk.size > 0):
                # Add BH to inner_disk_arrays
                blackholes_inner_disk.add_blackholes(
                    new_mass=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "mass"),
                    new_spin=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "spin"),
                    new_spin_angle=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "spin_angle"),
                    new_orb_a=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "orb_a"),
                    new_orb_inc=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "orb_inc"),
                    new_orb_ang_mom=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "orb_ang_mom"),
                    new_orb_ecc=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "orb_ecc"),
                    new_orb_arg_periapse=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "orb_arg_periapse"),
                    new_gen=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "gen"),
                    new_galaxy=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "galaxy"),
                    new_time_passed=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "time_passed"),
                    new_id_num=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "id_num"),
                    new_gw_freq=9.e-7 * np.ones(len(bh_id_num_pro_inner_disk))
                )

                # Remove from blackholes_pro and update filing_cabinet
                blackholes_pro.remove_id_num(bh_id_num_pro_inner_disk)
                filing_cabinet.update(id_num=bh_id_num_pro_inner_disk,
                                      attr="disk_inner_outer",
                                      new_info=np.full(len(bh_id_num_pro_inner_disk), -1))

            # Check if any prograde stars are in the inner disk
            star_id_num_pro_inner_disk = stars_pro.id_num[stars_pro.orb_a < opts.inner_disk_outer_radius]
            if (star_id_num_pro_inner_disk.size > 0):
                # Add BH to inner_disk_arrays
                stars_inner_disk.add_stars(
                    new_mass=stars_pro.at_id_num(star_id_num_pro_inner_disk, "mass"),
                    new_log_radius=stars_pro.at_id_num(star_id_num_pro_inner_disk, "log_radius"),
                    new_log_teff=stars_pro.at_id_num(star_id_num_pro_inner_disk, "log_teff"),
                    new_log_luminosity=stars_pro.at_id_num(star_id_num_pro_inner_disk, "log_luminosity"),
                    new_X=stars_pro.at_id_num(star_id_num_pro_inner_disk, "star_X"),
                    new_Y=stars_pro.at_id_num(star_id_num_pro_inner_disk, "star_Y"),
                    new_Z=stars_pro.at_id_num(star_id_num_pro_inner_disk, "star_Z"),
                    new_orb_a=stars_pro.at_id_num(star_id_num_pro_inner_disk, "orb_a"),
                    new_orb_inc=stars_pro.at_id_num(star_id_num_pro_inner_disk, "orb_inc"),
                    new_orb_ang_mom=stars_pro.at_id_num(star_id_num_pro_inner_disk, "orb_ang_mom"),
                    new_orb_ecc=stars_pro.at_id_num(star_id_num_pro_inner_disk, "orb_ecc"),
                    new_orb_arg_periapse=stars_pro.at_id_num(star_id_num_pro_inner_disk, "orb_arg_periapse"),
                    new_gen=stars_pro.at_id_num(star_id_num_pro_inner_disk, "gen"),
                    new_galaxy=stars_pro.at_id_num(star_id_num_pro_inner_disk, "galaxy"),
                    new_time_passed=stars_pro.at_id_num(star_id_num_pro_inner_disk, "time_passed"),
                    new_id_num=stars_pro.at_id_num(star_id_num_pro_inner_disk, "id_num")
                )

                # Remove from stars_pro and update filing_cabinet
                stars_pro.remove_id_num(star_id_num_pro_inner_disk)
                filing_cabinet.update(id_num=star_id_num_pro_inner_disk,
                                      attr="disk_inner_outer",
                                      new_info=np.full(len(star_id_num_pro_inner_disk), -1))

            # Check if any retrograde BHs are in the inner disk
            bh_id_num_retro_inner_disk = blackholes_retro.id_num[blackholes_retro.orb_a < opts.inner_disk_outer_radius]
            if (bh_id_num_retro_inner_disk.size > 0):
                # Add BH to inner_disk_arrays
                blackholes_inner_disk.add_blackholes(
                    new_mass=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "mass"),
                    new_spin=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "spin"),
                    new_spin_angle=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "spin_angle"),
                    new_orb_a=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "orb_a"),
                    new_orb_inc=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "orb_inc"),
                    new_orb_ang_mom=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "orb_ang_mom"),
                    new_orb_ecc=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "orb_ecc"),
                    new_orb_arg_periapse=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "orb_arg_periapse"),
                    new_gen=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "gen"),
                    new_galaxy=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "galaxy"),
                    new_time_passed=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "time_passed"),
                    new_id_num=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "id_num"),
                    new_gw_freq=9.e-7*np.ones(len(bh_id_num_retro_inner_disk))
                )
                # Remove from blackholes_retro and update filing_cabinet
                blackholes_retro.remove_id_num(bh_id_num_retro_inner_disk)
                filing_cabinet.update(id_num=bh_id_num_retro_inner_disk,
                                      attr="disk_inner_outer",
                                      new_info=np.full(len(bh_id_num_retro_inner_disk), -1))

            # Check if any retrograde stars are in the inner disk
            star_id_num_retro_inner_disk = stars_retro.id_num[stars_retro.orb_a < opts.inner_disk_outer_radius]
            if (star_id_num_retro_inner_disk.size > 0):
                # Add BH to inner_disk_arrays
                stars_inner_disk.add_stars(
                    new_mass=stars_retro.at_id_num(star_id_num_retro_inner_disk, "mass"),
                    new_log_radius=stars_retro.at_id_num(star_id_num_retro_inner_disk, "log_radius"),
                    new_log_luminosity=stars_retro.at_id_num(star_id_num_retro_inner_disk, "log_luminosity"),
                    new_log_teff=stars_retro.at_id_num(star_id_num_retro_inner_disk, "log_teff"),
                    new_X=stars_retro.at_id_num(star_id_num_retro_inner_disk, "star_X"),
                    new_Y=stars_retro.at_id_num(star_id_num_retro_inner_disk, "star_Y"),
                    new_Z=stars_retro.at_id_num(star_id_num_retro_inner_disk, "star_Z"),
                    new_orb_a=stars_retro.at_id_num(star_id_num_retro_inner_disk, "orb_a"),
                    new_orb_inc=stars_retro.at_id_num(star_id_num_retro_inner_disk, "orb_inc"),
                    new_orb_ang_mom=stars_retro.at_id_num(star_id_num_retro_inner_disk, "orb_ang_mom"),
                    new_orb_ecc=stars_retro.at_id_num(star_id_num_retro_inner_disk, "orb_ecc"),
                    new_orb_arg_periapse=stars_retro.at_id_num(star_id_num_retro_inner_disk, "orb_arg_periapse"),
                    new_gen=stars_retro.at_id_num(star_id_num_retro_inner_disk, "gen"),
                    new_galaxy=stars_retro.at_id_num(star_id_num_retro_inner_disk, "galaxy"),
                    new_time_passed=stars_retro.at_id_num(star_id_num_retro_inner_disk, "time_passed"),
                    new_id_num=stars_retro.at_id_num(star_id_num_retro_inner_disk, "id_num")
                )
                # Remove from stars_retro and update filing_cabinet
                stars_retro.remove_id_num(star_id_num_retro_inner_disk)
                filing_cabinet.update(id_num=star_id_num_retro_inner_disk,
                                      attr="disk_inner_outer",
                                      new_info=np.full(len(star_id_num_retro_inner_disk), -1))

            if (blackholes_inner_disk.num > 0):
                # FIX THIS: Return the new evolved bh_orb_ecc_inner_disk as they decay inwards.
                # Potentially move inner disk behaviour to module that is not dynamics (e.g inner disk module)
                blackholes_inner_disk.orb_a = dynamics.bh_near_smbh(
                    opts.smbh_mass,
                    blackholes_inner_disk.orb_a,
                    blackholes_inner_disk.mass,
                    blackholes_inner_disk.orb_ecc,
                    opts.timestep_duration_yr,
                    opts.inner_disk_outer_radius,
                    opts.disk_inner_stable_circ_orb,
                )
                # Update filing cabinet
                filing_cabinet.update(id_num=blackholes_inner_disk.id_num,
                                      attr="orb_a",
                                      new_info=blackholes_inner_disk.orb_a)

                # On 1st run through define old GW freqs (at say 9.e-7 Hz, since evolution change is 1e-6Hz)
                if blackholes_emris.num == 0:
                    old_gw_freq = 9.e-7*np.ones(blackholes_inner_disk.num)
                if blackholes_emris.num > 0:
                    old_gw_freq = blackholes_inner_disk.gw_freq

                emri_gw_strain, emri_gw_freq = emri.evolve_emri_gw(
                    blackholes_inner_disk,
                    opts.timestep_duration_yr,
                    old_gw_freq,
                    opts.smbh_mass,
                    agn_redshift
                )

                blackholes_inner_disk.gw_freq = emri_gw_freq
                blackholes_inner_disk.gw_strain = emri_gw_strain

            if (stars_inner_disk.num > 0):
                # FIX THIS: Return the new evolved bh_orb_ecc_inner_disk as they decay inwards.
                # Potentially move inner disk behaviour to module that is not dynamics (e.g inner disk module)
                stars_inner_disk.orb_a = dynamics.bh_near_smbh( # KN: TDEs need their own method here bc drag
                    opts.smbh_mass,
                    stars_inner_disk.orb_a,
                    stars_inner_disk.mass,
                    stars_inner_disk.orb_ecc,
                    opts.timestep_duration_yr,
                    opts.inner_disk_outer_radius,
                    opts.disk_inner_stable_circ_orb,
                )
                # Update filing cabinet
                filing_cabinet.update(id_num=stars_inner_disk.id_num,
                                      attr="orb_a",
                                      new_info=stars_inner_disk.orb_a)

                # On 1st run through define old GW freqs (at say 9.e-7 Hz, since evolution change is 1e-6Hz)
                if stars_tdes.num == 0:
                    old_gw_tde_freq = 9.e-7*np.ones(stars_inner_disk.num)
                if (stars_tdes.num > 0):
                    old_gw_tde_freq = tde_gw_freq

                tde_gw_strain, tde_gw_freq = emri.evolve_emri_gw( # KN: TDEs need their own method here bc drag
                    stars_inner_disk,
                    opts.timestep_duration_yr,
                    old_gw_tde_freq,
                    opts.smbh_mass,
                    agn_redshift
                )

            if blackholes_inner_disk.num > 0:
                blackholes_emris.add_blackholes(new_mass=blackholes_inner_disk.mass,
                                                new_spin=blackholes_inner_disk.spin,
                                                new_spin_angle=blackholes_inner_disk.spin_angle,
                                                new_orb_a=blackholes_inner_disk.orb_a,
                                                new_orb_inc=blackholes_inner_disk.orb_inc,
                                                new_orb_ang_mom=blackholes_inner_disk.orb_ang_mom,
                                                new_orb_ecc=blackholes_inner_disk.orb_ecc,
                                                new_orb_arg_periapse=blackholes_inner_disk.orb_arg_periapse,
                                                new_gw_freq=emri_gw_freq,
                                                new_gw_strain=emri_gw_strain,
                                                new_gen=blackholes_inner_disk.gen,
                                                new_galaxy=np.full(emri_gw_freq.size, galaxy),
                                                new_time_passed=np.full(emri_gw_freq.size, time_passed),
                                                new_id_num=blackholes_inner_disk.id_num)

            #merger_dist = 1.0
            emri_merger_id_num = blackholes_inner_disk.id_num[blackholes_inner_disk.orb_a <= opts.disk_inner_stable_circ_orb]
            star_rlof_smbh_id_num = stars_inner_disk.id_num[stars_inner_disk.orb_a <= opts.disk_inner_stable_circ_orb]

            # if mergers occurs, remove from inner_disk arrays and stop evolving
            # still getting some nans, but I think that's bc there's retros that should have been
            #  moved to prograde arrays

            if np.size(emri_merger_id_num) > 0:
                blackholes_inner_disk.remove_id_num(emri_merger_id_num)
                # Remove merged EMRIs from filing_cabinet
                filing_cabinet.remove_id_num(emri_merger_id_num)

            # If stars plunge into SMBH record them and then remove from disk
            if np.size(star_rlof_smbh_id_num) > 0:
                stars_plunge.add_stars(
                    new_mass=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "mass"),
                    new_log_radius=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "log_radius"),
                    new_log_luminosity=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "log_luminosity"),
                    new_log_teff=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "log_teff"),
                    new_X=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "star_X"),
                    new_Y=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "star_Y"),
                    new_Z=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "star_Z"),
                    new_orb_a=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "orb_a"),
                    new_orb_inc=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "orb_inc"),
                    new_orb_ang_mom=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "orb_ang_mom"),
                    new_orb_ecc=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "orb_ecc"),
                    new_orb_arg_periapse=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "orb_arg_periapse"),
                    new_galaxy=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "galaxy"),
                    new_time_passed=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "time_passed"),
                    new_gen=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "gen"),
                    new_id_num=stars_inner_disk.at_id_num(star_rlof_smbh_id_num, "id_num")
                    )
                stars_inner_disk.remove_id_num(star_rlof_smbh_id_num)
                filing_cabinet.remove_id_num(star_rlof_smbh_id_num)

            # Here is where we need to move retro to prograde if they've flipped in this timestep
            # If they're IN the disk prograde, OR if they've circularized:
            # stop treating them with crude retro evolution--it will be sad
            # SF: fix the inc threshhold later to be truly 'in disk' but should be non-stupid as-is!!!
            inc_threshhold = 5.0 * np.pi/180.0
            bh_id_num_flip_to_pro = blackholes_retro.id_num[np.where((np.abs(blackholes_retro.orb_inc) <= inc_threshhold) | (blackholes_retro.orb_ecc == 0.0))]
            if (bh_id_num_flip_to_pro.size > 0):
                # add to prograde arrays
                blackholes_pro.add_blackholes(
                    new_mass=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "mass"),
                    new_orb_a=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "orb_a"),
                    new_spin=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "spin"),
                    new_spin_angle=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "spin_angle"),
                    new_orb_inc=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "orb_inc"),
                    new_orb_ang_mom=np.ones(bh_id_num_flip_to_pro.size),
                    new_orb_ecc=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "orb_ecc"),
                    new_orb_arg_periapse=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "orb_arg_periapse"),
                    new_galaxy=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "galaxy"),
                    new_time_passed=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "time_passed"),
                    new_gen=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "gen"),
                    new_id_num=bh_id_num_flip_to_pro
                )
                # delete from retro arrays
                blackholes_retro.remove_id_num(id_num_remove=bh_id_num_flip_to_pro)
                # Update filing_cabinet
                filing_cabinet.update(id_num=bh_id_num_flip_to_pro,
                                      attr="direction",
                                      new_info=np.ones(bh_id_num_flip_to_pro.size))

            star_id_num_flip_to_pro_or_tde = stars_retro.id_num[np.where((np.abs(stars_retro.orb_inc) <= inc_threshhold) | (stars_retro.orb_ecc == 0.0))]
            star_id_num_tde, star_id_num_flip_to_pro = tde.check_tde_or_flip(star_id_num_flip_to_pro_or_tde,
                                                                             stars_retro.at_id_num(star_id_num_flip_to_pro_or_tde, "mass"),
                                                                             stars_retro.at_id_num(star_id_num_flip_to_pro_or_tde, "log_radius"),
                                                                             stars_retro.at_id_num(star_id_num_flip_to_pro_or_tde, "orb_ecc"),
                                                                             stars_retro.at_id_num(star_id_num_flip_to_pro_or_tde, "orb_a"),
                                                                             opts.smbh_mass)
            if (star_id_num_flip_to_pro.size > 0):
                # add to prograde arrays
                stars_pro.add_stars(
                    new_mass=stars_retro.at_id_num(star_id_num_flip_to_pro, "mass"),
                    new_log_radius=stars_retro.at_id_num(star_id_num_flip_to_pro, "log_radius"),
                    new_log_luminosity=stars_retro.at_id_num(star_id_num_flip_to_pro, "log_luminosity"),
                    new_log_teff=stars_retro.at_id_num(star_id_num_flip_to_pro, "log_teff"),
                    new_X=stars_retro.at_id_num(star_id_num_flip_to_pro, "star_X"),
                    new_Y=stars_retro.at_id_num(star_id_num_flip_to_pro, "star_Y"),
                    new_Z=stars_retro.at_id_num(star_id_num_flip_to_pro, "star_Z"),
                    new_orb_a=stars_retro.at_id_num(star_id_num_flip_to_pro, "orb_a"),
                    new_orb_inc=stars_retro.at_id_num(star_id_num_flip_to_pro, "orb_inc"),
                    new_orb_ang_mom=np.ones(star_id_num_flip_to_pro.size),
                    new_orb_ecc=stars_retro.at_id_num(star_id_num_flip_to_pro, "orb_ecc"),
                    new_orb_arg_periapse=stars_retro.at_id_num(star_id_num_flip_to_pro, "orb_arg_periapse"),
                    new_galaxy=stars_retro.at_id_num(star_id_num_flip_to_pro, "galaxy"),
                    new_time_passed=stars_retro.at_id_num(star_id_num_flip_to_pro, "time_passed"),
                    new_gen=stars_retro.at_id_num(star_id_num_flip_to_pro, "gen"),
                    new_id_num=stars_retro.at_id_num(star_id_num_flip_to_pro, "id_num")
                )
                # delete from retro arrays
                stars_retro.remove_id_num(id_num_remove=star_id_num_flip_to_pro)
                # Update filing_cabinet
                filing_cabinet.update(id_num=star_id_num_flip_to_pro,
                                      attr="direction",
                                      new_info=np.ones(star_id_num_flip_to_pro.size))

            if (star_id_num_tde.size > 0):
                # add to TDE arrays
                stars_tdes.add_stars(
                    new_mass=stars_retro.at_id_num(star_id_num_tde, "mass"),
                    new_log_radius=stars_retro.at_id_num(star_id_num_tde, "log_radius"),
                    new_log_luminosity=stars_retro.at_id_num(star_id_num_tde, "log_luminosity"),
                    new_log_teff=stars_retro.at_id_num(star_id_num_tde, "log_teff"),
                    new_X=stars_retro.at_id_num(star_id_num_tde, "star_X"),
                    new_Y=stars_retro.at_id_num(star_id_num_tde, "star_Y"),
                    new_Z=stars_retro.at_id_num(star_id_num_tde, "star_Z"),
                    new_orb_a=stars_retro.at_id_num(star_id_num_tde, "orb_a"),
                    new_orb_inc=stars_retro.at_id_num(star_id_num_tde, "orb_inc"),
                    new_orb_ang_mom=np.ones(star_id_num_tde.size),
                    new_orb_ecc=stars_retro.at_id_num(star_id_num_tde, "orb_ecc"),
                    new_orb_arg_periapse=stars_retro.at_id_num(star_id_num_tde, "orb_arg_periapse"),
                    new_galaxy=stars_retro.at_id_num(star_id_num_tde, "galaxy"),
                    new_time_passed=stars_retro.at_id_num(star_id_num_tde, "time_passed"),
                    new_gen=stars_retro.at_id_num(star_id_num_tde, "gen"),
                    new_id_num=stars_retro.at_id_num(star_id_num_tde, "id_num")
                )
                # delete from retro arrays
                stars_retro.remove_id_num(id_num_remove=star_id_num_tde)

            # Record mass cycled parameters
            disk_arr_timestep.append(time_passed)
            disk_arr_galaxy.append(galaxy)
            disk_arr_mass_gained.append(sum(disk_mass_gained))
            disk_arr_mass_lost.append(sum(disk_mass_lost))

            # Iterate the time step
            time_passed = time_passed + opts.timestep_duration_yr
            # Print time passed every 10 timesteps for now
            time_galaxy_tracker = 10.0*opts.timestep_duration_yr
            if time_passed % time_galaxy_tracker == 0:
                print("Time passed=", time_passed)

        # End Loop of Timesteps at Final Time, end all changes & print out results

        print("End Loop!")
        print("Final Time (yrs) = ", time_passed)
        if opts.verbose:
            print("BH locations at Final Time")
            print(blackholes_pro.orb_a)
        print("Number of binaries = ", blackholes_binary.num)
        print("Total number of mergers = ", blackholes_merged.num)
        print("Total number of immortal stars = ", len(stars_pro.mass[stars_pro.mass == opts.disk_star_initial_mass_cutoff]))
        print("Nbh_disk", disk_bh_num)

        # Write out all singletons after AGN episode so we can use as input to another AGN phase

        # Assume that all BH binaries break apart
        # Note: eccentricity will relax, ignore
        # Inclination assumed 0deg
        blackholes_pro.add_blackholes(
            new_mass=np.concatenate([blackholes_binary.mass_1, blackholes_binary.mass_1]),
            new_spin=np.concatenate([blackholes_binary.spin_1, blackholes_binary.spin_2]),
            new_spin_angle=np.concatenate([blackholes_binary.spin_angle_1, blackholes_binary.spin_angle_2]),
            new_orb_a=np.concatenate([blackholes_binary.orb_a_1, blackholes_binary.orb_a_2]),
            new_orb_inc=np.zeros(blackholes_binary.num * 2),  # Assume orb_inc = 0.0
            new_orb_ang_mom=np.ones(blackholes_binary.num * 2),  # Assume all are prograde
            new_orb_ecc=np.zeros(blackholes_binary.num * 2),  # Assume orb_ecc = 0.0
            new_orb_arg_periapse=np.full(blackholes_binary.num * 2, -1.5),  # Assume orb_arg_periapse = -1
            new_galaxy=np.full(blackholes_binary.num * 2, galaxy),
            new_time_passed=np.full(blackholes_binary.num * 2, time_passed),
            new_gen=np.concatenate([blackholes_binary.gen_1, blackholes_binary.gen_2]),
            new_id_num=np.arange(filing_cabinet.id_max + 1, filing_cabinet.id_max + 1 + blackholes_binary.num * 2, 1)
        )

        # Update filing_cabinet
        filing_cabinet.add_objects(
            new_id_num=np.arange(filing_cabinet.id_max + 1, filing_cabinet.id_max + 1 + blackholes_binary.num * 2, 1),
            new_category=np.zeros(blackholes_binary.num * 2),
            new_orb_a=np.concatenate([blackholes_binary.orb_a_1, blackholes_binary.orb_a_2]),
            new_mass=np.concatenate([blackholes_binary.mass_1, blackholes_binary.mass_1]),
            new_orb_ecc=np.zeros(blackholes_binary.num * 2),
            new_size=np.full(blackholes_binary.num * 2, -1.5),
            new_direction=np.ones(blackholes_binary.num * 2),
            new_disk_inner_outer=np.zeros(blackholes_binary.num * 2)
        )
        blackholes_binary.remove_id_num(blackholes_binary.id_num)
        filing_cabinet.remove_id_num(blackholes_binary.id_num)

        # Add merged BH to the population level object
        blackholes_merged_pop.add_blackholes(new_id_num=blackholes_merged.id_num,
                                             new_galaxy=blackholes_merged.galaxy,
                                             new_bin_orb_a=blackholes_merged.bin_orb_a,
                                             new_mass_final=blackholes_merged.mass_final,
                                             new_spin_final=blackholes_merged.spin_final,
                                             new_spin_angle_final=blackholes_merged.spin_angle_final,
                                             new_mass_1=blackholes_merged.mass_1,
                                             new_mass_2=blackholes_merged.mass_2,
                                             new_spin_1=blackholes_merged.spin_1,
                                             new_spin_2=blackholes_merged.spin_2,
                                             new_spin_angle_1=blackholes_merged.spin_angle_1,
                                             new_spin_angle_2=blackholes_merged.spin_angle_2,
                                             new_gen_1=blackholes_merged.gen_1,
                                             new_gen_2=blackholes_merged.gen_2,
                                             new_chi_eff=blackholes_merged.chi_eff,
                                             new_chi_p=blackholes_merged.chi_p,
                                             new_v_kick=blackholes_merged.v_kick,
                                             new_lum_shock=blackholes_merged.lum_shock,
                                             new_lum_jet=blackholes_merged.lum_jet,
                                             new_time_merged=blackholes_merged.time_merged)

        # Add list of all binaries formed to the population level object
        blackholes_binary_gw_pop.add_binaries(new_id_num=blackholes_binary_gw.id_num,
                                              new_mass_1=blackholes_binary_gw.mass_1,
                                              new_mass_2=blackholes_binary_gw.mass_2,
                                              new_orb_a_1=blackholes_binary_gw.orb_a_1, 
                                              new_orb_a_2=blackholes_binary_gw.orb_a_2,
                                              new_spin_1=blackholes_binary_gw.spin_1,
                                              new_spin_2=blackholes_binary_gw.spin_2,
                                              new_spin_angle_1=blackholes_binary_gw.spin_angle_1,
                                              new_spin_angle_2=blackholes_binary_gw.spin_angle_2,
                                              new_bin_sep=blackholes_binary_gw.bin_sep,
                                              new_bin_orb_a=blackholes_binary_gw.bin_orb_a,
                                              new_time_to_merger_gw=blackholes_binary_gw.time_to_merger_gw,
                                              new_flag_merging=blackholes_binary_gw.flag_merging,
                                              new_time_merged=blackholes_binary_gw.time_merged,
                                              new_bin_ecc=blackholes_binary_gw.bin_ecc,
                                              new_gen_1=blackholes_binary_gw.gen_1,
                                              new_gen_2=blackholes_binary_gw.gen_2,
                                              new_bin_orb_ang_mom=blackholes_binary_gw.bin_orb_ang_mom,
                                              new_bin_orb_inc=blackholes_binary_gw.bin_orb_inc,
                                              new_bin_orb_ecc=blackholes_binary_gw.bin_orb_ecc,
                                              new_gw_freq=blackholes_binary_gw.gw_freq,
                                              new_gw_strain=blackholes_binary_gw.gw_strain,
                                              new_galaxy=blackholes_binary_gw.galaxy)

        # Save the mergers
        galaxy_save_name = f"gal{galaxy_zfilled_str}/{opts.fname_output_mergers}"
        blackholes_merged.to_txt(os.path.join(opts.work_directory, galaxy_save_name), cols=merger_cols)

        # Append each galaxy result to outputs

        emris_pop.add_blackholes(new_id_num=blackholes_emris.id_num,
                                 new_mass=blackholes_emris.mass,
                                 new_spin=blackholes_emris.spin,
                                 new_spin_angle=blackholes_emris.spin_angle,
                                 new_orb_a=blackholes_emris.orb_a,
                                 new_orb_inc=blackholes_emris.orb_inc,
                                 new_orb_ang_mom=blackholes_emris.orb_ang_mom,
                                 new_orb_ecc=blackholes_emris.orb_ecc,
                                 new_orb_arg_periapse=blackholes_emris.orb_arg_periapse,
                                 new_galaxy=blackholes_emris.galaxy,
                                 new_gen=blackholes_emris.gen,
                                 new_time_passed=blackholes_emris.time_passed,
                                 new_gw_freq=blackholes_emris.gw_freq,
                                 new_gw_strain=blackholes_emris.gw_strain)
        
        tdes_pop.add_stars(new_id_num=stars_tdes.id_num,
                           new_mass=stars_tdes.mass,
                           new_log_radius=stars_tdes.log_radius,
                           new_log_teff=stars_tdes.log_teff,
                           new_log_luminosity=stars_tdes.log_luminosity,
                           new_X=stars_tdes.star_X,
                           new_Y=stars_tdes.star_Y,
                           new_Z=stars_tdes.star_Z,
                           new_orb_a=stars_tdes.orb_a,
                           new_orb_inc=stars_tdes.orb_inc,
                           new_orb_ang_mom=stars_tdes.orb_ang_mom,
                           new_orb_ecc=stars_tdes.orb_ecc,
                           new_orb_arg_periapse=stars_tdes.orb_arg_periapse,
                           new_galaxy=stars_tdes.galaxy,
                           new_gen=stars_tdes.gen,
                           new_time_passed=stars_tdes.time_passed)
        
        stars_plunge_pop.add_stars(new_id_num=stars_plunge.id_num,
                                   new_mass=stars_plunge.mass,
                                   new_log_radius=stars_plunge.log_radius,
                                   new_log_teff=stars_plunge.log_teff,
                                   new_log_luminosity=stars_plunge.log_luminosity,
                                   new_X=stars_plunge.star_X,
                                   new_Y=stars_plunge.star_Y,
                                   new_Z=stars_plunge.star_Z,
                                   new_orb_a=stars_plunge.orb_a,
                                   new_orb_inc=stars_plunge.orb_inc,
                                   new_orb_ang_mom=stars_plunge.orb_ang_mom,
                                   new_orb_ecc=stars_plunge.orb_ecc,
                                   new_orb_arg_periapse=stars_plunge.orb_arg_periapse,
                                   new_galaxy=stars_plunge.galaxy,
                                   new_gen=stars_plunge.gen,
                                   new_time_passed=stars_plunge.time_passed)
        
        stars_pop.add_stars(new_id_num=stars_pro.id_num,
                            new_mass=stars_pro.mass,
                            new_log_radius=stars_pro.log_radius,
                            new_log_teff=stars_pro.log_teff,
                            new_log_luminosity=stars_pro.log_luminosity,
                            new_X=stars_pro.star_X,
                            new_Y=stars_pro.star_Y,
                            new_Z=stars_pro.star_Z,
                            new_orb_a=stars_pro.orb_a,
                            new_orb_ang_mom=stars_pro.orb_ang_mom,
                            new_orb_ecc=stars_pro.orb_ecc,
                            new_orb_inc=stars_pro.orb_inc,
                            new_orb_arg_periapse=stars_pro.orb_arg_periapse,
                            new_galaxy=stars_pro.galaxy,
                            new_gen=stars_pro.gen,
                            new_time_passed=stars_pro.time_passed)
        
        stars_explode_pop.add_stars(new_id_num_star=stars_explode.id_num_star,
                                    new_id_num_bh=stars_explode.id_num_bh,
                                    new_mass_star=stars_explode.mass_star,
                                    new_mass_bh=stars_explode.mass_bh,
                                    new_orb_a_star=stars_explode.orb_a_star,
                                    new_orb_a_bh=stars_explode.orb_a_bh,
                                    new_star_log_radius=stars_explode.star_log_radius,
                                    new_orb_inc_star=stars_explode.orb_inc_star,
                                    new_orb_inc_bh=stars_explode.orb_inc_bh,
                                    new_orb_ecc_star=stars_explode.orb_ecc_star,
                                    new_orb_ecc_bh=stars_explode.orb_ecc_bh,
                                    new_gen_star=stars_explode.gen_star,
                                    new_gen_bh=stars_explode.gen_bh,
                                    new_galaxy=stars_explode.galaxy,
                                    new_time_sn=stars_explode.time_sn
                                    )

        stars_merge_pop.add_stars(new_id_num=stars_merge.id_num,
                                  new_galaxy=stars_merge.galaxy,
                                  new_orb_a_final=stars_merge.orb_a_final,
                                  new_gen_final=stars_merge.gen_final,
                                  new_mass_final=stars_merge.mass_final,
                                  new_mass_1=stars_merge.mass_1,
                                  new_mass_2=stars_merge.mass_2,
                                  new_gen_1=stars_merge.gen_1,
                                  new_gen_2=stars_merge.gen_2,
                                  new_log_radius_final=stars_merge.log_radius_final,
                                  new_orb_ecc=stars_merge.orb_ecc,
                                  new_time_merged=stars_merge.time_merged)

        # Add mass cycled info to population arrays
        disk_arr_timestep_pop = np.concatenate([disk_arr_timestep_pop, disk_arr_timestep])
        disk_arr_mass_gained_pop = np.concatenate([disk_arr_mass_gained_pop, disk_arr_mass_gained])
        disk_arr_mass_lost_pop = np.concatenate([disk_arr_mass_lost_pop, disk_arr_mass_lost])

    # save all mergers from Monte Carlo
    basename, extension = os.path.splitext(opts.fname_output_mergers)
    population_save_name = f"{basename}_population{extension}"
    survivors_save_name = f"{basename}_survivors{extension}"
    emris_save_name = f"{basename}_emris{extension}"
    tdes_save_name = f"{basename}_tdes{extension}"
    gws_save_name = f"{basename}_lvk{extension}"
    stars_save_name = f"{basename}_stars_population{extension}"
    stars_explode_save_name = f"{basename}_stars_exploded{extension}"
    stars_merge_save_name = f"{basename}_stars_merged{extension}"
    stars_plunge_save_name = f"{basename}_stars_plunge{extension}"
    basename_disk, extension_disk = os.path.splitext(opts.fname_output)
    disk_mass_cycled_save_name = f"{basename_disk}_diskmasscycled{extension_disk}"


    # Save things
    emris_pop.to_txt(os.path.join(opts.work_directory, emris_save_name),
                     cols=emri_cols)        

    blackholes_pro.to_txt(os.path.join(opts.work_directory, survivors_save_name),
                          cols=bh_surviving_cols)
    
    blackholes_binary_gw_pop.to_txt(os.path.join(opts.work_directory, gws_save_name),
                                    cols=binary_gw_cols)

    # Include initial seed in header
    blackholes_merged_pop.to_txt(os.path.join(opts.work_directory, population_save_name),
                                 cols=population_cols, extra_header=f"Initial seed: {opts.seed}\n")
    
    if opts.flag_add_stars:
        stars_pop.to_txt(os.path.join(opts.work_directory, stars_save_name),
                                    cols=stars_cols, extra_header=f"Initial seed: {opts.seed}\n")
        tdes_pop.to_txt(os.path.join(opts.work_directory, tdes_save_name),
                     cols=tde_cols)
        stars_plunge_pop.to_txt(os.path.join(opts.work_directory, stars_plunge_save_name),
                     cols=tde_cols)
        stars_explode_pop.to_txt(os.path.join(opts.work_directory, stars_explode_save_name),
                     cols=stars_explode_cols)
        stars_merge_pop.to_txt(os.path.join(opts.work_directory, stars_merge_save_name),
                               cols=stars_merge_cols)
        temp_mass_cycled = np.column_stack((disk_arr_galaxy, disk_arr_timestep_pop, disk_arr_mass_gained_pop, disk_arr_mass_lost_pop))
        np.savetxt(os.path.join(opts.work_directory, disk_mass_cycled_save_name), temp_mass_cycled, header="galaxy timestep mass_gained mass_lost")

    toc_perf = time.perf_counter()
    print("Perf time: %0.2f"%(toc_perf - tic_perf))

if __name__ == "__main__":
    main()
