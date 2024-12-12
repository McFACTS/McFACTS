#!/usr/bin/env python3
import os
import warnings
from importlib import resources as impresources
from os.path import isfile, isdir
from pathlib import Path
import time

import numpy as np
from astropy import units as astropy_units

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
from mcfacts.physics import point_masses

from mcfacts.inputs import ReadInputs
from mcfacts.inputs import data as input_data
from mcfacts.mcfacts_random_state import reset_random
from mcfacts.objects.agnobject import AGNBlackHole, AGNBinaryBlackHole, AGNMergedBlackHole, AGNStar, AGNFilingCabinet
from mcfacts.setup import setupdiskblackholes, setupdiskstars, initializediskstars

binary_field_names = "bin_orb_a1 bin_orb_a2 mass1 mass2 spin1 spin2 theta1 theta2 sep bin_com time_gw merger_flag time_mgr  gen_1 gen_2  bin_ang_mom bin_ecc bin_incl bin_orb_ecc nu_gw h_bin"

# columns to write for incremental data files
merger_cols = ["galaxy", "bin_orb_a", "mass_final", "chi_eff", "spin_final", "spin_angle_final", "mass_1", "mass_2",
               "spin_1", "spin_2", "spin_angle_1", "spin_angle_2", "gen_1", "gen_2", "time_merged", ]
binary_cols = ["orb_a_1", "orb_a_2", "mass_1", "mass_2", "spin_1", "spin_2", "spin_angle_1", "spin_angle_2",
               "bin_sep", "bin_orb_a", "time_to_merger_gw", "flag_merging", "time_merged", "bin_ecc",
               "gen_1", "gen_2", "bin_orb_ang_mom", "bin_orb_inc", "bin_orb_ecc", "gw_freq", "gw_strain", "id_num"]

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
    parser.add_argument("--fname-snapshots-bh",
                        default="output_bh_[single|binary]_$(index).dat",
                        help="output of BH index file ")
    parser.add_argument("--save-snapshots", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-w", "--work-directory",
                        default=Path().parent.resolve(),
                        help="Set the working directory for saving output. Default: current working directory",
                        type=str
                        )
    parser.add_argument("--seed", type=int, default=None,
                        help="Set the random seed. Randomly sets one if not passed. Default: None")
    parser.add_argument("--fname-log", default="mcfacts_sim.log", type=str,
                        help="Specify a file to save the arguments for mcfacts")

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
                            default=_default,
                            type=_dtype,
                            metavar=_metavar,
                            )

    # Parse arguments
    opts = parser.parse_args()
    # Check that the inifile exists
    assert isfile(opts.fname_ini)
    # Convert to path objects
    opts.fname_ini = Path(opts.fname_ini)
    assert opts.fname_ini.is_file()
    opts.fname_snapshots_bh = Path(opts.fname_snapshots_bh)
    opts.fname_output_mergers = Path(opts.fname_output_mergers)

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
        if getattr(opts, name) != _variable_inputs[name]:
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
    disk_surface_density, disk_aspect_ratio, disk_opacity, disk_sound_speed, disk_density = \
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
        bh_orb_a_initial = setupdiskblackholes.setup_disk_blackholes_location(
                disk_bh_num, opts.disk_radius_outer, opts.disk_inner_stable_circ_orb)
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
                                  galaxy=np.zeros(disk_bh_num),
                                  time_passed=np.zeros(disk_bh_num))

        # Initialize filing_cabinet
        filing_cabinet = AGNFilingCabinet(id_num=blackholes.id_num,
                                          category=np.full(blackholes.num, 0),
                                          orb_a=blackholes.orb_a,
                                          mass=blackholes.mass,
                                          size=np.full(blackholes.num, -1.5),
                                          )

        # Initialize stars
        if opts.flag_add_stars:
            stars, disk_star_num = initializediskstars.init_single_stars(opts, disk_aspect_ratio, id_start_val=filing_cabinet.id_max+1)
        else:
            stars, disk_star_num = AGNStar(), 0

        print(f"{disk_bh_num} black holes, {disk_star_num} stars")
        filing_cabinet.add_objects(new_id_num=stars.id_num,
                                   new_category=np.full(stars.num, 1),
                                   new_orb_a=stars.orb_a,
                                   new_mass=stars.mass,
                                   new_size=point_masses.r_g_from_units(opts.smbh_mass, (10 ** stars.log_radius) * astropy_units.Rsun).value,
                                   new_direction=np.zeros(stars.num),
                                   new_disk_inner_outer=np.zeros(stars.num))

        # Writing initial parameters to file
        if opts.flag_add_stars:
            stars.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/initial_params_star.dat"))
        blackholes.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/initial_params_bh.dat"))

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

        # Create empty EMRIs object
        blackholes_emris = AGNBlackHole()

        # Create empty TDEs object
        stars_tdes = AGNStar()

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

            # then migrate as usual
            blackholes_pro.orb_a = migration.type1_migration_single(
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

            stars_pro.orb_a = migration.type1_migration_single(
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

            # Check for orb_a unphysical
            bh_pro_id_num_unphysical = blackholes_pro.id_num[blackholes_pro.orb_a == 0.]
            if bh_pro_id_num_unphysical.size > 0:
                # The binary has unphysical eccentricity. Delete
                blackholes_pro.remove_id_num(bh_pro_id_num_unphysical)
                filing_cabinet.remove_id_num(bh_pro_id_num_unphysical)

            star_pro_id_num_unphysical = stars_pro.id_num[stars_pro.orb_a == 0.]
            if star_pro_id_num_unphysical.size > 0:
                # The binary has unphysical eccentricity. Delete
                stars_pro.remove_id_num(star_pro_id_num_unphysical)
                filing_cabinet.remove_id_num(star_pro_id_num_unphysical)

            # Accrete
            blackholes_pro.mass = accretion.change_bh_mass(
                blackholes_pro.mass,
                opts.disk_bh_eddington_ratio,
                disk_bh_eddington_mass_growth_rate,
                opts.timestep_duration_yr
            )

            disk_star_luminosity_factor = 4.  # Hardcoded from Cantiello+2021 and Fabj+2024
            stars_pro.mass = accretion.change_star_mass(
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

            # Change stars' radii, luminosity, and temp
            stars_pro.log_radius, stars_pro.log_luminosity, stars_pro.log_teff = stellar_interpolation.interp_star_params(stars_pro.mass)

            # Update filing cabinet
            filing_cabinet.update(id_num=stars_pro.id_num,
                                  attr="size",
                                  new_info=point_masses.r_g_from_units(opts.smbh_mass, (10 ** stars_pro.log_radius) * astropy_units.Rsun).value)

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

            # Check for bin_ecc unphysical
            bh_retro_id_num_unphysical_ecc = blackholes_retro.id_num[blackholes_retro.orb_ecc >= 1.]
            if bh_retro_id_num_unphysical_ecc.size > 0:
                # The BH has unphysical eccentricity. Delete
                blackholes_retro.remove_id_num(bh_retro_id_num_unphysical_ecc)
                filing_cabinet.remove_id_num(bh_retro_id_num_unphysical_ecc)

            star_retro_id_num_unphysical_ecc = stars_retro.id_num[stars_retro.orb_ecc >= 1.]
            if star_retro_id_num_unphysical_ecc.size > 0:
                # The star has unphysical eccentricity. Delete
                stars_retro.remove_id_num(star_retro_id_num_unphysical_ecc)
                filing_cabinet.remove_id_num(star_retro_id_num_unphysical_ecc)

            # Perturb eccentricity via dynamical encounters
            if opts.flag_dynamic_enc > 0:

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

                stars_pro.orb_a, stars_pro.orb_ecc = dynamics.circular_singles_encounters_prograde(
                    opts.smbh_mass,
                    stars_pro.orb_a,
                    stars_pro.mass,
                    stars_pro.orb_ecc,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit,
                    opts.delta_energy_strong,
                    opts.disk_radius_outer
                )

            # Do things to the binaries--first check if there are any:
            if blackholes_binary.num > 0:

                # First check that binaries are real. Discard any columns where the location or the mass is 0.
                # SF: I believe this step is handling an error checking thing that may have been
                #     set up in the previous timeloop if e.g. a binary either merged or was ionized?
                #     Please explain what this is and how it works right here?
                bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary)
                if bh_binary_id_num_unphysical.size > 0:

                    # One of the key parameter (mass or location is zero or bin_ecc > 1). Not real. Delete binary. Remove column at index = ionization_flag
                    blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                    filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)

                # Check for bin_ecc unphysical
                # bh_binary_id_num_unphysical_ecc = blackholes_binary.id_num[blackholes_binary.bin_ecc >= 1.]
                # if bh_binary_id_num_unphysical_ecc.size > 0:
                #     # The binary has unphysical eccentricity. Delete
                #     blackholes_binary.remove_id_num(bh_binary_id_num_unphysical_ecc)
                #     filing_cabinet.remove_id_num(bh_binary_id_num_unphysical_ecc)

                # If there are binaries, evolve them
                # Damp binary orbital eccentricity
                blackholes_binary = eccentricity.orbital_bin_ecc_damping(
                    opts.smbh_mass,
                    blackholes_binary,
                    disk_surface_density,
                    disk_aspect_ratio,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit
                )

                if (opts.flag_dynamic_enc > 0):
                    # Harden/soften binaries via dynamical encounters
                    # Harden binaries due to encounters with circular singletons (e.g. Leigh et al. 2018)
                    # FIX THIS: RETURN perturbed circ singles (orb_a, orb_ecc)

                    blackholes_binary = dynamics.circular_binaries_encounters_circ_prograde(
                        opts.smbh_mass,
                        blackholes_pro.orb_a,
                        blackholes_pro.mass,
                        blackholes_pro.orb_ecc,
                        opts.timestep_duration_yr,
                        opts.disk_bh_pro_orb_ecc_crit,
                        opts.delta_energy_strong,
                        blackholes_binary,
                        opts.disk_radius_outer
                    )

                    # Update filing cabinet with new bin_sep
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="size",
                                          new_info=blackholes_binary.bin_sep)

                    # Soften/ ionize binaries due to encounters with eccentric singletons
                    # Return 3 things: perturbed biary_bh_array, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc
                    blackholes_binary = dynamics.circular_binaries_encounters_ecc_prograde(
                        opts.smbh_mass,
                        blackholes_pro.orb_a,
                        blackholes_pro.mass,
                        blackholes_pro.orb_ecc,
                        opts.timestep_duration_yr,
                        opts.disk_bh_pro_orb_ecc_crit,
                        opts.delta_energy_strong,
                        blackholes_binary,
                        opts.disk_radius_outer
                    )

                    # Update filing cabinet with new bin_sep
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="size",
                                          new_info=blackholes_binary.bin_sep)

                # Check for bin_ecc unphysical
                # We need a second check here
                bh_binary_id_num_unphysical_ecc = blackholes_binary.id_num[blackholes_binary.bin_ecc >= 1.]
                if bh_binary_id_num_unphysical_ecc.size > 0:
                    # The binary has unphysical eccentricity. Delete
                    blackholes_binary.remove_id_num(bh_binary_id_num_unphysical_ecc)
                    filing_cabinet.remove_id_num(bh_binary_id_num_unphysical_ecc)

                # Harden binaries via gas
                # Choose between Baruteau et al. 2011 gas hardening, or gas hardening from LANL simulations. To do: include dynamical hardening/softening from encounters
                blackholes_binary = evolve.bin_harden_baruteau(
                    blackholes_binary,
                    opts.smbh_mass,
                    opts.timestep_duration_yr,
                    time_gw_normalization,
                    time_passed,
                )

                # Update filing cabinet with new bin_sep
                filing_cabinet.update(id_num=blackholes_binary.id_num,
                                      attr="size",
                                      new_info=blackholes_binary.bin_sep)

                # Check closeness of binary. Are black holes at merger condition separation
                blackholes_binary = evolve.bin_contact_check(blackholes_binary, opts.smbh_mass)

                # Update filing cabinet with new bin_sep
                filing_cabinet.update(id_num=blackholes_binary.id_num,
                                      attr="size",
                                      new_info=blackholes_binary.bin_sep)

                # Accrete gas onto binary components
                blackholes_binary = evolve.change_bin_mass(
                    blackholes_binary,
                    opts.disk_bh_eddington_ratio,
                    disk_bh_eddington_mass_growth_rate,
                    opts.timestep_duration_yr,
                )

                # Spin up binary components
                blackholes_binary = evolve.change_bin_spin_magnitudes(
                    blackholes_binary,
                    opts.disk_bh_eddington_ratio,
                    opts.disk_bh_torque_condition,
                    opts.timestep_duration_yr,
                )

                # Torque angle of binary spin components
                blackholes_binary = evolve.change_bin_spin_angles(
                    blackholes_binary,
                    opts.disk_bh_eddington_ratio,
                    opts.disk_bh_torque_condition,
                    disk_bh_spin_resolution_min,
                    opts.timestep_duration_yr,
                )

                if (opts.flag_dynamic_enc > 0):
                    # Spheroid encounters
                    # FIX THIS: Replace nsc_imf_bh below with nsc_imf_stars_ since pulling from stellar MF
                    blackholes_binary = dynamics.bin_spheroid_encounter(
                        opts.smbh_mass,
                        opts.timestep_duration_yr,
                        blackholes_binary,
                        time_passed,
                        opts.nsc_imf_bh_powerlaw_index,
                        opts.delta_energy_strong,
                        opts.nsc_spheroid_normalization,
                    )
                    # Update filing cabinet with new bin_sep
                    filing_cabinet.update(id_num=blackholes_binary.id_num,
                                          attr="size",
                                          new_info=blackholes_binary.bin_sep)

                if (opts.flag_dynamic_enc > 0):
                    # Recapture bins out of disk plane.
                    # FIX THIS: Replace this with orb_inc_damping but for binary bhbh OBJECTS (KN)
                    blackholes_binary = dynamics.bin_recapture(
                        blackholes_binary,
                        opts.timestep_duration_yr
                    )

                # Migrate binaries
                # First if feedback present, find ratio of feedback heating torque to migration torque
                if opts.flag_thermal_feedback > 0:
                    ratio_heat_mig_torques_bin_com = evolve.bin_com_feedback_hankla(
                        blackholes_binary,
                        disk_surface_density,
                        disk_opacity,
                        opts.disk_bh_eddington_ratio,
                        opts.disk_alpha_viscosity,
                        opts.disk_radius_outer
                    )

                else:
                    ratio_heat_mig_torques_bin_com = np.ones(blackholes_binary.num)

                # Migrate binaries center of mass
                blackholes_binary = migration.type1_migration_binary(
                    opts.smbh_mass, blackholes_binary,
                    opts.disk_bh_pro_orb_ecc_crit,
                    disk_surface_density, disk_aspect_ratio, ratio_heat_mig_torques_bin_com,
                    opts.disk_radius_trap, opts.disk_radius_outer, opts.timestep_duration_yr)

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
                        old_bbh_gw_freq = bbh_gw_freq

                    num_bbh_gw_tracked = bh_binary_id_num_gw.size

                    bbh_gw_strain, bbh_gw_freq = gw.bbh_gw_params(
                        blackholes_binary,
                        bh_binary_id_num_gw,
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
                blackholes_binary = gw.evolve_gw(
                    blackholes_binary,
                    opts.smbh_mass,
                    agn_redshift
                )

                # Check and see if any binaries are ionized.
                bh_binary_id_num_ionization = evolve.bin_ionization_check(blackholes_binary, opts.smbh_mass)
                if bh_binary_id_num_ionization.size > 0:
                    # Append 2 new BH to arrays of single BH locations, masses, spins, spin angles & gens
                    # For now add 2 new orb ecc term of 0.01. inclination is 0.0 as well. TO DO: calculate v_kick and resulting perturbation to orb ecc.
                    new_orb_ecc = eccentricity.ionized_orb_ecc(bh_binary_id_num_ionization.size * 2, opts.disk_bh_orb_ecc_max_init)
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
                        new_id_num=np.arange(filing_cabinet.id_max+1, filing_cabinet.id_max + 1 + bh_binary_id_num_ionization.size * 2, 1)
                    )

                    # Update filing cabinet
                    filing_cabinet.add_objects(
                        new_id_num=np.arange(filing_cabinet.id_max+1, filing_cabinet.id_max + 1 + bh_binary_id_num_ionization.size * 2, 1),
                        new_category=np.zeros(bh_binary_id_num_ionization.size * 2),
                        new_orb_a=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "orb_a_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "orb_a_2")]),
                        new_mass=np.concatenate([
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "mass_1"),
                            blackholes_binary.at_id_num(bh_binary_id_num_ionization, "mass_2")]),
                        new_size=np.full(bh_binary_id_num_ionization.size * 2, -1.5),
                        new_direction=np.ones(bh_binary_id_num_ionization.size * 2),
                        new_disk_inner_outer=np.zeros(bh_binary_id_num_ionization.size * 2)
                    )

                    blackholes_binary.remove_id_num(bh_binary_id_num_ionization)
                    filing_cabinet.remove_id_num(bh_binary_id_num_ionization)

                bh_binary_id_num_merger = blackholes_binary.id_num[blackholes_binary.flag_merging < 0]

                if opts.verbose:
                    print("Merger ID numbers")
                    print(bh_binary_id_num_merger)

                if (bh_binary_id_num_merger.size > 0):

                    # Check for primary masses of zero
                    if (np.sum(blackholes_binary.mass_1 == 0) > 0):
                        blackholes_binary.remove_id_num(blackholes_binary.id_num[blackholes_binary.mass_1 == 0])
                        bh_binary_id_num_merger = blackholes_binary.id_num[np.nonzero(blackholes_binary.flag_merging)]

                    # Check for NaNs in flag_merging
                    if (np.sum(np.isnan(blackholes_binary.flag_merging)) > 0):
                        blackholes_binary.remove_id_num(blackholes_binary.id_num[np.isnan(blackholes_binary.flag_merging)])
                        bh_binary_id_num_merger = blackholes_binary.id_num[np.nonzero(blackholes_binary.flag_merging)]

                    bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary=blackholes_binary)
                    if bh_binary_id_num_unphysical.size > 0:
                        # One of the key parameter (mass or location is zero). Not real. Delete binary. Remove column at index = ionization_flag
                        blackholes_binary.remove_id_num(bh_binary_id_num_unphysical)
                        filing_cabinet.remove_id_num(bh_binary_id_num_unphysical)
                        bh_binary_id_num_merger = blackholes_binary.id_num[np.nonzero(blackholes_binary.flag_merging)]

                    if (bh_binary_id_num_merger.size > 0):

                        bh_mass_merged = merge.merged_mass(
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2")
                            )

                        bh_spin_merged = merge.merged_spin(
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2")
                            )

                        bh_chi_eff_merged = merge.chi_effective(
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_ang_mom")
                                )

                        bh_chi_p_merged = merge.chi_p(
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
                                blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_inc")
                                )

                        # TODO: calculate v_kick and resulting perturbation to orb ecc. For now set v_kick to 200 km/s
                        v_kick = 200.
                        bh_orb_ecc_merged = merge.merged_orb_ecc(blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
                                                                 np.full(bh_binary_id_num_merger.size, v_kick),
                                                                 opts.smbh_mass)

                        # Append new merged BH to arrays of single BH locations, masses, spins, spin angles & gens
                        blackholes_merged.add_blackholes(new_id_num=bh_binary_id_num_merger,
                                                         new_galaxy=np.full(bh_binary_id_num_merger.size, galaxy),
                                                         new_bin_orb_a=blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
                                                         new_mass_final=bh_mass_merged,
                                                         new_spin_final=bh_spin_merged,
                                                         new_spin_angle_final=np.zeros(bh_binary_id_num_merger.size),
                                                         new_mass_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
                                                         new_mass_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
                                                         new_spin_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
                                                         new_spin_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
                                                         new_spin_angle_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
                                                         new_spin_angle_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
                                                         new_gen_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_1"),
                                                         new_gen_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_2"),
                                                         new_chi_eff=bh_chi_eff_merged,
                                                         new_chi_p=bh_chi_p_merged,
                                                         new_time_merged=np.full(bh_binary_id_num_merger.size, time_passed))

                        # # New bh generation is max of generations involved in merger plus 1
                        blackholes_pro.add_blackholes(new_mass=blackholes_merged.at_id_num(bh_binary_id_num_merger, "mass_final"),
                                                      new_orb_a=blackholes_merged.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
                                                      new_spin=blackholes_merged.at_id_num(bh_binary_id_num_merger, "spin_final"),
                                                      new_spin_angle=np.zeros(bh_binary_id_num_merger.size),
                                                      new_orb_inc=np.zeros(bh_binary_id_num_merger.size),
                                                      new_orb_ang_mom=np.ones(bh_binary_id_num_merger.size),
                                                      new_orb_ecc=bh_orb_ecc_merged,
                                                      new_gen=np.maximum(blackholes_merged.at_id_num(bh_binary_id_num_merger, "gen_1"),
                                                                         blackholes_merged.at_id_num(bh_binary_id_num_merger, "gen_2")) + 1.0,
                                                      new_orb_arg_periapse=np.full(bh_binary_id_num_merger.size, -1.5),
                                                      new_galaxy=np.full(bh_binary_id_num_merger.size, galaxy),
                                                      new_time_passed=np.full(bh_binary_id_num_merger.size, time_passed),
                                                      new_id_num=bh_binary_id_num_merger)

                        # Update filing cabinet
                        filing_cabinet.update(id_num=bh_binary_id_num_merger,
                                              attr="category",
                                              new_info=np.full(bh_binary_id_num_merger.size, 0))
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

            # If a close encounter within mutual Hill sphere add a new Binary
            # check which binaries should get made
            close_encounters_indices = formation.binary_check(
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                opts.smbh_mass,
                blackholes_pro.orb_ecc,
                opts.disk_bh_pro_orb_ecc_crit
            )
            close_encounters_id_num = blackholes_pro.id_num[close_encounters_indices]

            if (close_encounters_id_num.size > 0):
                blackholes_binary, bh_binary_id_num_new = formation.add_to_binary_obj(
                    blackholes_binary,
                    blackholes_pro,
                    close_encounters_id_num,
                    filing_cabinet.id_max,
                    opts.fraction_bin_retro,
                    opts.smbh_mass,
                    agn_redshift,
                    opts.disk_bh_pro_orb_ecc_crit
                )

                # Add new binaries to filing cabinet and delete prograde singleton black holes
                filing_cabinet.add_objects(new_id_num=bh_binary_id_num_new,
                                           new_category=np.full(bh_binary_id_num_new.size, 2),
                                           new_orb_a=blackholes_binary.at_id_num(bh_binary_id_num_new, "bin_orb_a"),
                                           new_mass=blackholes_binary.at_id_num(bh_binary_id_num_new, "mass_1") + blackholes_binary.at_id_num(bh_binary_id_num_new, "mass_2"),
                                           new_size=blackholes_binary.at_id_num(bh_binary_id_num_new, "bin_sep"),
                                           new_direction=np.full(bh_binary_id_num_new.size, 1),
                                           new_disk_inner_outer=np.full(bh_binary_id_num_new.size, 1))
                filing_cabinet.remove_id_num(id_num_remove=close_encounters_id_num)

                # delete corresponding entries for new binary members from singleton arrays
                blackholes_pro.remove_id_num(id_num_remove=close_encounters_id_num)

            # After this time period, was there a disk capture via orbital grind-down?
            # To do: What eccentricity do we want the captured BH to have? Right now ecc=0.0? Should it be ecc<h at a?             
            # Assume 1st gen BH captured and orb ecc =0.0
            # To do: Bias disk capture to more massive BH!
            capture = time_passed % opts.capture_time_yr
            if capture == 0:
                bh_orb_a_captured = setupdiskblackholes.setup_disk_blackholes_location(
                    1, opts.disk_radius_capture_outer, opts.disk_inner_stable_circ_orb)
                bh_mass_captured = setupdiskblackholes.setup_disk_blackholes_masses(
                    1, opts.nsc_imf_bh_mode, opts.nsc_imf_bh_mass_max, opts.nsc_imf_bh_powerlaw_index, opts.mass_pile_up)
                bh_spin_captured = setupdiskblackholes.setup_disk_blackholes_spins(
                    1, opts.nsc_bh_spin_dist_mu, opts.nsc_bh_spin_dist_sigma)
                bh_spin_angle_captured = setupdiskblackholes.setup_disk_blackholes_spin_angles(
                    1, bh_spin_captured)
                bh_gen_captured = [1]
                bh_orb_ecc_captured = [0.0]
                bh_orb_inc_captured = [0.0]
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
                                              new_id_num=np.arange(filing_cabinet.id_max+1, len(bh_mass_captured) + filing_cabinet.id_max+1, 1))
                # Update filing cabinet
                filing_cabinet.add_objects(new_id_num=np.arange(filing_cabinet.id_max+1, len(bh_mass_captured) + filing_cabinet.id_max+1, 1),
                                           new_category=np.array([0.0]),
                                           new_orb_a=bh_orb_a_captured,
                                           new_mass=bh_mass_captured,
                                           new_size=np.array([-1]),
                                           new_direction=np.array([1.0]),
                                           new_disk_inner_outer=np.array([0.0]))

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
                                               new_size=point_masses.r_g_from_units(opts.smbh_mass, (10 ** star_log_radius_captured) * astropy_units.Rsun).value,
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
                    new_id_num=blackholes_pro.at_id_num(bh_id_num_pro_inner_disk, "id_num")
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
                    new_id_num=blackholes_retro.at_id_num(bh_id_num_retro_inner_disk, "id_num")
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

                # On 1st run through define old GW freqs (at say 9.e-7 Hz, since evolution change is 1e-6Hz)
                if blackholes_emris.num == 0:
                    old_gw_freq = 9.e-7*np.ones(blackholes_inner_disk.num)
                if (blackholes_emris.num > 0):
                    old_gw_freq = emri_gw_freq

                emri_gw_strain, emri_gw_freq = emri.evolve_emri_gw(
                    blackholes_inner_disk,
                    opts.timestep_duration_yr,
                    old_gw_freq,
                    opts.smbh_mass,
                    agn_redshift
                )

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
                    new_id_num=blackholes_retro.at_id_num(bh_id_num_flip_to_pro, "id_num")
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
        
    # save all mergers from Monte Carlo
    basename, extension = os.path.splitext(opts.fname_output_mergers)
    population_save_name = f"{basename}_population{extension}"
    survivors_save_name = f"{basename}_survivors{extension}"
    emris_save_name = f"{basename}_emris{extension}"
    tdes_save_name = f"{basename}_tdes{extension}"
    gws_save_name = f"{basename}_lvk{extension}"
    stars_save_name = f"{basename}_stars{extension}"

    # Define columns to write
    emri_cols = ["galaxy", "time_passed", "orb_a", "mass", "orb_ecc", "gw_strain", "gw_freq", "id_num"]
    bh_surviving_cols = ["galaxy", "orb_a", "mass", "spin", "spin_angle", "gen", "id_num"]
    population_cols = ["galaxy", "bin_orb_a", "mass_final", "chi_eff", "spin_final", "spin_angle_final",
                       "mass_1", "mass_2", "spin_1", "spin_2", "spin_angle_1", "spin_angle_2",
                       "gen_1", "gen_2", "time_merged", "chi_p"]
    binary_gw_cols = ["galaxy", "time_merged", "bin_sep", "mass_total", "bin_ecc", "gw_strain", "gw_freq", "gen_1", "gen_2"]
    stars_cols = ["galaxy", "time_passed", "orb_a", "mass", "log_radius", "log_teff", "log_luminosity", "orb_ecc", "star_X", "star_Y", "star_Z", "gen", "id_num"]
    tde_cols = ["galaxy", "time_passed", "orb_a", "mass", "log_radius", "log_teff", "log_luminosity", "orb_ecc", "star_X", "star_Y", "star_Z", "gen", "id_num"]

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

    toc_perf = time.perf_counter()
    print("Perf time: %0.2f"%(toc_perf - tic_perf))

if __name__ == "__main__":
    main()
