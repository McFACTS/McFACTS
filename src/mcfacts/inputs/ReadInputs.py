"""Define input handling functions for mcfacts_sim

Inifile
-------
    "disk_model_name"               : str
        'sirko_goodman' or 'thompson_etal'
    "flag_use_pagn"                 : int
        Use pAGN to generate disk model?
    "flag_add_stars"                : int
        Add stars to the disk
    "flag_coalesce_initial_stars"   : int
        Keep stars as is (0) or coalesce before time loop starts (1)
    "flag_initial_stars_BH_immortal": int
        If stars over disk_star_initial_mass_cutoff turn into BH (0) or hold at cutoff (1, immortal)
    "smbh_mass"                     : float
        Mass of the supermassive black hole (solMass)
    "disk_radius_trap"              : float
        Radius of migration trap in gravitational radii (r_g = G*`smbh_mass`/c^2)
        Should be set to zero if disk model has no trap
    "disk_radius_outer"             : float
        final element of disk_model_radius_array (units of r_g)
    "disk_radius_max_pc"            : float
        Maximum disk size in parsecs (0. for off)
    "disk_alpha_viscosity"          : float
        disk viscosity 'alpha'
    "nsc_radius_outer"              : float
        Radius of NSC (units of pc)
    "nsc_mass"                      : float
        Mass of NSC (units of M_sun)
    "nsc_radius_crit"               : float
        Radius where NSC density profile flattens (transition to Bahcall-Wolf) (units of pc)
    "nsc_ratio_bh_num_star_num"     : float
        Ratio of number of BH to stars in NSC (typically spans 3x10^-4 to 10^-2 in Generozov+18)
    "nsc_ratio_bh_mass_star_mass"   : float
        Ratio of mass of typical BH to typical star in NSC (typically 10:1 in Generozov+18)
    "nsc_density_index_inner"       : float
        Index of radial density profile of NSC inside r_nsc_crit (usually Bahcall-Wolf, 1.75)
    "nsc_density_index_outer"       : float
        Index of radial density profile of NSC outside r_nsc_crit
        (e.g. 2.5 in Generozov+18 or 2.25 if Peebles)
    "disk_aspect_ratio_avg"    : float
        Average disk scale height (e.g. about 3% in Sirko & Goodman 2003 out to ~0.3pc)
    "nsc_spheroid_normalization"    : float
        Spheroid normalization
    "nsc_imf_bh_mode"               : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--mode of initial mass dist (M_sun)
    "nsc_imf_bh_powerlaw_index"     : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--powerlaw index for Pareto dist
    "nsc_imf_bh_mass_max"           : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--mass of cutoff (M_sun)
    "nsc_bh_spin_dist_mu"           : float
        Initial spin distribution for stellar bh is assumed to be Gaussian
        --mean of spin dist
    "nsc_bh_spin_dist_sigma"        : float
        Initial spin distribution for stellar bh is assumed to be Gaussian
        --standard deviation of spin dist
    "disk_bh_torque_condition"      : float
        fraction of initial mass required to be accreted before BH spin is torqued
        fully into alignment with the AGN disk. We don't know for sure but
        Bogdanovic et al. says between 0.01=1% and 0.1=10% is what is required.
    "disk_bh_eddington_ratio"       : float
        Eddington ratio for disk bh
    "disk_bh_orb_ecc_max_init"      : float
        assumed accretion rate onto stellar bh from disk gas, in units of Eddington
        accretion rate
    "disk_star_mass_max_init"       : float
        Initial mass distribution for stars is assumed Salpeter
    "disk_star_mass_min_init"       : float
        Initial mass distribution for stars is assumed Salpeter
    "nsc_imf_star_powerlaw_index"   : float
        Initial mass distribution for stars is assumed Salpeter, disk_alpha_viscosity = 2.35
    "disk_star_scale_factor"        : float
        Scale factor to go from number of BH to number of stars.
    "disk_star_initial_mass_cutoff" : float
        Cutoff for initial star behavior
    "nsc_imf_star_mass_mode"       : float
        Mass mode for star IMF
    "disk_star_torque_condition"    : float
        fraction of initial mass required to be accreted before star spin is torqued
        fully into alignment with the AGN disk. We don't know for sure but
        Bogdanovic et al. says between 0.01=1% and 0.1=10% is what is required.
    "disk_star_eddington_ratio"     : float
        assumed accretion rate onto stars from disk gas, in units of Eddington
        accretion rate
    "disk_star_orb_ecc_max_init"    : float
        assuming initially flat eccentricity distribution among single orbiters around SMBH
        out to max_initial_eccentricity. Eventually this will become smarter.
    "nsc_star_metallicity_x_init"   : float
        Stellar initial hydrogen mass fraction
    "nsc_star_metallicity_y_init"   : float
        Stellar initial helium mass fraction
    "nsc_star_metallicity_z_init"   : float
        Stellar initial metallicity mass fraction
    "timestep_duration_yr"          : float
        How long is your timestep in years?
    "timestep_num"                  : int
        How many timesteps are you taking (timestep*number_of_timesteps = disk_lifetime)
    "galaxy_num"                    : int
        Number of galaxies of code run (e.g. 1 for testing, 30 for a quick run)
    "fraction_bin_retro"            : float
        Fraction of BBH that form retrograde to test (q,X_eff) relation. Default retro=0.1
    "flag_thermal_feedback"         : int
        Switch (1) turns feedback from embedded BH on.
    "flag_orb_ecc_damping"          : int
        Switch (1) turns orb. ecc damping on.
        If switch = 0, assumes all bh are circularized (at e=e_crit)
    "capture_time_yr"              : float
        Capture time in years Secunda et al. (2021) assume capture rate 1/0.1 Myr
    "disk_radius_capture_outer"     : float
        Disk capture outer radius (units of r_g)
        Secunda et al. (2001) assume <2000r_g from Fabj et al. (2020)
    "disk_bh_pro_orb_ecc_crit"      : float
        Critical eccentricity (limiting eccentricity, below which assumed circular orbit)
    "flag_dynamic_enc"              : int
        Switch (1) turns dynamical encounters between embedded BH on.
    "delta_energy_strong"           : float
        Average energy change per strong interaction.
        de can be 20% in cluster interactions. May be 10% on average (with gas)
    "inner_disk_outer_radius"       : float
        Outer radius of the inner disk (Rg)
    "disk_inner_stable_circ_orb"    : float
        Innermost Stable Circular Orbit around SMBH
    "mass_pile_up"                  : float
        Pile-up of masses caused by cutoff (M_sun)
    "save_snapshots"                : int
        Save snapshots of the disk and NSC at each timestep
    "mean_harden_energy_delta"      : float
        The Gaussian mean value for the energy change during a strong interaction
    "var_harden_energy_delta"       : float
        The Gaussian variance value for the energy change during a strong interaction
    "flag_use_surrogate"            : int
        Switch (0) uses analytical kick prescription from Akiba et al. (2024). Switch (1) sets 200 km/s for each merger's kick velocity.
"""
# Things everyone needs
import configparser as ConfigParser
from io import StringIO
from importlib import resources as impresources
# Third party
import numpy as np
import scipy.interpolate
# pAGN imports 
import pagn.constants as pagn_ct
# Local imports 
import mcfacts.external.DiskModelsPAGN as dm_pagn
from mcfacts.inputs import data as mcfacts_input_data
from astropy import constants as ct

# Dictionary of types
INPUT_TYPES = {
    "disk_model_name"               : str,
    "flag_use_pagn"                 : int,
    "flag_add_stars"                : int,
    "flag_coalesce_initial_stars"   : int,
    "flag_initial_stars_BH_immortal": int,
    "smbh_mass"                     : float,
    "disk_radius_trap"              : float,
    "disk_radius_outer"             : float,
    "disk_radius_max_pc"            : float,
    "disk_alpha_viscosity"          : float,
    "nsc_radius_outer"              : float,
    "nsc_mass"                      : float,
    "nsc_radius_crit"               : float,
    "nsc_ratio_bh_num_star_num"     : float,
    "nsc_ratio_bh_mass_star_mass"   : float,
    "nsc_density_index_inner"       : float,
    "nsc_density_index_outer"       : float,
    "disk_aspect_ratio_avg"         : float,
    "nsc_spheroid_normalization"    : float,
    "nsc_imf_bh_mode"               : float,
    "nsc_imf_bh_powerlaw_index"     : float,
    "nsc_imf_bh_mass_max"           : float,
    "nsc_bh_spin_dist_mu"           : float,
    "nsc_bh_spin_dist_sigma"        : float,
    "disk_bh_torque_condition"      : float,
    "disk_bh_eddington_ratio"       : float,
    "disk_bh_orb_ecc_max_init"      : float,
    "disk_star_mass_max_init"       : float,
    "disk_star_mass_min_init"       : float,
    "nsc_imf_star_powerlaw_index"   : float,
    "disk_star_scale_factor"        : float,
    "disk_star_initial_mass_cutoff" : float,
    "nsc_imf_star_mass_mode"        : float,
    "disk_star_torque_condition"    : float,
    "disk_star_eddington_ratio"     : float,
    "disk_star_orb_ecc_max_init"    : float,
    "nsc_star_metallicity_x_init"   : float,
    "nsc_star_metallicity_y_init"   : float,
    "nsc_star_metallicity_z_init"   : float,
    "timestep_duration_yr"          : float,
    "timestep_num"                  : int,
    "galaxy_num"                    : int,
    "fraction_bin_retro"            : float,
    "flag_thermal_feedback"         : int,
    "flag_orb_ecc_damping"          : int,
    "capture_time_yr"               : float,
    "disk_radius_capture_outer"     : float,
    "disk_bh_pro_orb_ecc_crit"      : float,
    "flag_dynamic_enc"              : int,
    "delta_energy_strong"           : float,
    "inner_disk_outer_radius"       : float,
    "disk_inner_stable_circ_orb"    : float,
    "mass_pile_up"                  : float,
    "save_snapshots"                : int,
    "mean_harden_energy_delta"      : float,
    "var_harden_energy_delta"       : float,
    "torque_prescription"           : str,
    "flag_phenom_turb"              : int,
    "phenom_turb_centroid"          : float,
    "phenom_turb_std_dev"           : float,
    "flag_use_surrogate"            : int
}
# Ensure none of the data types are bool to avoid issues casting ascii to boolean
if bool in INPUT_TYPES.values():
    raise ValueError("[ReadInputs.py] Boolean data types are not allowed in"
                     "the INPUT_TYPES dictionary. Please use int instead.")

def ReadInputs_ini(fname_ini, verbose=0):
    """Input file parser

    This function reads your input choices from a file user specifies or
    default (inputs/model_choice.txt), and returns the chosen variables for
    manipulation by main.

    Required input formats and units are given in IOdocumentation.txt file.

    Parameters
    ----------
    fname_ini : str
        Name of inifile for mcfacts
    verbose : int
        Print extra things when 1. Default is 0.

    Returns
    -------
    input_variables : dict
        Dictionary of input variables
    """
    # Initialize the config parser
    config = ConfigParser.ConfigParser()
    config.optionxform=str # force preserve case! Important for --choose-data-LI-seglen

    # Default format has no section headings ...
    config.read(fname_ini)

    # convert to dict
    input_variables = dict(config.items('top'))


    # try to pretty-convert these to quantites
    for name in input_variables:
        # If we know what the type should be, use the type from INPUT_TYPES
        if name in INPUT_TYPES:
            input_variables[name] = INPUT_TYPES[name](input_variables[name])
        # If we can't figure it out, check if it's a floating point number
        elif '.' in input_variables[name]:
            input_variables[name] = float(input_variables[name])
        # If it's not a floating point number, try an integer
        elif input_variables[name].isdigit():
            input_variables[name] = int(input_variables[name])
        # If it's a boolean string, raise an error
        elif input_variables[name] in ["False", "false", "F", "True", "true", "T"]:
            raise ValueError(f"[ReadInputs.py] Encountered `{{{name}: {input_variables[name]}}}` "
                              "in the ini file. Boolean data types are not allowed. "
                              "Please use int instead.")
        # If all else fails, leave it the way we found it
        else:
            input_variables[name] = str(input_variables[name])

    # Clean up strings
    for name in input_variables:
        if isinstance(input_variables[name], str):
            input_variables[name] = input_variables[name].strip("'")

    # Set default : not use pagn.  this allows us not to provide it
    if not ('flag_use_pagn' in input_variables):
        input_variables['flag_use_pagn'] = 0

    ## Check outer disk radius in parsecs
    # Scale factor for parsec distance in r_g
    pc_dist = 2.e5*((input_variables["smbh_mass"]/1.e8)**(-1.0))
    # Calculate outer disk radius in pc
    disk_radius_outer_pc = input_variables["disk_radius_outer"]/pc_dist
    # Check disk_radius_max_pc argument
    if input_variables["disk_radius_max_pc"] == 0.:
        # Case 1: disk_radius_max_pc is disabled
        pass
    elif input_variables["disk_radius_max_pc"] < 0.:
        # Case 2: disk_radius_max_pc is negative
        # Always assign disk_radius_outer to given distance in parsecs
        input_variables["disk_radius_outer"] = \
            -1. * input_variables["disk_radius_max_pc"] * pc_dist
    else:
        # Case 3: disk_radius_max_pc is positive
        # Cap disk_radius_outer at given value
        if disk_radius_outer_pc > input_variables["disk_radius_max_pc"]:
            # calculate scale factor
            disk_radius_scale = input_variables["disk_radius_max_pc"] / disk_radius_outer_pc
            # Adjust disk_radius_outer as needed
            input_variables["disk_radius_outer"] = \
                input_variables["disk_radius_outer"] * disk_radius_scale

    # Print out the dictionary if we are in verbose mode
    if verbose:
        print("input_variables:")
        for key in input_variables:
            print(key, input_variables[key], type(input_variables[key]))
        print("I put your variables where they belong")

    # Return the arguments
    return input_variables


def load_disk_arrays(
    disk_model_name,
    disk_radius_outer,
    verbose=0
    ):
    """Load the dictionary arrays from file (pAGN_off)

    Use import resources to load datafile from src/mcfacts/inputs/data

    Parameters
    ----------
    disk_model_name : str
        sirko_goodman or thompson_etal
    disk_radius_outer : float
        Outer disk radius we truncate at
    verbose : int
        Print extra things when 1. Default is 0.

    Returns
    -------
    truncated_disk_radii : NumPy array (float)
        The disk radius array
    truncated_surface_densities : NumPy array (float)
        The surface density array
    truncated_aspect_ratio : NumPy array (float)
        The aspect ratio array
    truncated_temperature : NumPy array (float)
        The temperature array
    """

    # Get density filename
    fname_disk_surf_density = disk_model_name + '_surface_density.txt'
    # Look in the source data
    fname_disk_surf_density = impresources.files(mcfacts_input_data) / fname_disk_surf_density
    # Load data from the surface density file
    disk_surf_density_data = np.loadtxt(fname_disk_surf_density)
    # Truncate surface density data
    surf_density_mask = disk_surf_density_data[:,1] < disk_radius_outer
    trunc_surf_density_data = np.flip(disk_surf_density_data[surf_density_mask].T,axis=0)

    # open the disk model aspect ratio file and read it in
    # Note format is assumed to be comments with #
    #   aspect ratio in first column
    #   radius in r_g in second column must be identical to surface density file
    #       (radius is actually ignored in this file!)
    #   filename = model_aspect_ratio.txt, where model is user choice
    fname_disk_aspect_ratio = disk_model_name + "_aspect_ratio.txt"
    fname_disk_aspect_ratio = impresources.files(mcfacts_input_data) / fname_disk_aspect_ratio
    # Load data from the aspect ratio file
    disk_aspect_ratio_data = np.loadtxt(fname_disk_aspect_ratio)
    # Truncate aspect ratio data
    aspect_ratio_mask = disk_aspect_ratio_data[:,1] < disk_radius_outer
    trunc_aspect_ratio_data = np.flip(disk_aspect_ratio_data[aspect_ratio_mask].T,axis=0)

    # Get opacity filename
    fname_disk_opacity = disk_model_name + '_opacity.txt'
    # Look in the source data
    fname_disk_opacity = impresources.files(mcfacts_input_data) / fname_disk_opacity
    # Load data from opacity file
    disk_opacity_data = np.loadtxt(fname_disk_opacity)
    # Truncate opacity data
    opacity_mask = disk_opacity_data[:,1] < disk_radius_outer
    trunc_opacity_data = np.flip(disk_opacity_data[opacity_mask].T,axis=0)

    # Get sound speed filename
    fname_disk_sound_speed = disk_model_name + '_sound_speed.txt'
    # Look in the source data
    fname_disk_sound_speed = impresources.files(mcfacts_input_data) / fname_disk_sound_speed
    # Load data from opacity file
    disk_sound_speed_data = np.loadtxt(fname_disk_sound_speed)
    # Truncate disk at outer radius
    sound_speed_mask = disk_sound_speed_data[:,1] < disk_radius_outer
    trunc_sound_speed_data = np.flip(disk_sound_speed_data[sound_speed_mask].T,axis=0)

    # Get density filename
    fname_disk_density = disk_model_name + '_density.txt'
    # Look in the source data
    fname_disk_density = impresources.files(mcfacts_input_data) / fname_disk_density
    # Load data from opacity file
    disk_density_data = np.loadtxt(fname_disk_density)
    # Truncate disk at outer radius
    density_mask = disk_density_data[:,1] < disk_radius_outer
    trunc_density_data = np.flip(disk_density_data[density_mask].T,axis=0)

    # Get omega filename
    fname_disk_omega = disk_model_name + '_omega.txt'
    # Look in the source data
    fname_disk_omega = impresources.files(mcfacts_input_data) / fname_disk_omega
    # Load data from opacity file
    disk_omega_data = np.loadtxt(fname_disk_omega)
    # Truncate disk at outer radius
    omega_mask = disk_omega_data[:,1] < disk_radius_outer
    trunc_omega_data = np.flip(disk_omega_data[omega_mask].T,axis=0)

    # Get pressure grad filename
    fname_disk_pressure_gradient = disk_model_name + '_pressure_gradient.txt'
    # Look in the source data
    fname_disk_pressure_gradient = impresources.files(mcfacts_input_data) / fname_disk_pressure_gradient
    # Load data from opacity file
    disk_pressure_gradient_data = np.loadtxt(fname_disk_pressure_gradient)
    # Truncate disk at outer radius
    pressure_mask = disk_pressure_gradient_data[:,1] < disk_radius_outer
    trunc_pressure_data = np.flip(disk_pressure_gradient_data[pressure_mask].T,axis=0)

    # Get temp filename
    fname_disk_temperature = disk_model_name + '_temperature.txt'
    # Look in the source data
    fname_disk_temperature = impresources.files(mcfacts_input_data) / fname_disk_temperature
    # Load data from opacity file
    disk_temperature_data = np.loadtxt(fname_disk_temperature)
    # Truncate disk at outer radius
    temperature_mask = disk_temperature_data[:,1] < disk_radius_outer
    trunc_temperature_data = np.flip(disk_temperature_data[temperature_mask].T,axis=0)

    # Now redefine arrays used to generate interpolating functions in terms of truncated arrays
    return trunc_surf_density_data, trunc_aspect_ratio_data, trunc_opacity_data, trunc_sound_speed_data, trunc_density_data, trunc_omega_data, trunc_pressure_data, trunc_temperature_data


def construct_disk_direct(
    disk_model_name,
    disk_radius_outer,
    verbose=0
    ):
    """Construct a disk interpolation without pAGN

    Construct a disk interpolation without pAGN by reading
        files with the load_disk_arrays function

    Parameters
    ----------
    disk_model_name : str
        sirko_goodman or thompson_etal
    disk_radius_outer : float
        Outer disk radius we truncate at
    verbose : int
        Print extra things when 1. Default is 0.

    Returns
    -------
    disk_surf_dens_func : lambda
        Surface density (radius)
    disk_aspect_ratio_func : lambda
        Aspect ratio (radius)
    disk_opacity_func : lambda
        Opacity (radius)
    sound_speed_func : lambda
        sound speed (radius) [m/s] 
    disk_model_properties : dict
        Other disk model things we may want
    """
    # Call the load_disk_arrays function
    trunc_surf_density_data, trunc_aspect_ratio_data, \
            trunc_opacity_data, trunc_sound_speed_data, \
            trunc_density_data, trunc_omega_data, \
            trunc_pressure_data, trunc_temperature_data = \
        load_disk_arrays(
        disk_model_name,
        disk_radius_outer,
        verbose=verbose
        )
    if verbose:
        print("disk_model_radii\n", trunc_surf_density_data[0])
    # Now generate interpolating functions
    # Create surface density function from input arrays
    disk_surf_dens_func_log = scipy.interpolate.CubicSpline(
        np.log(trunc_surf_density_data[0]), np.log(trunc_surf_density_data[1]))
    disk_surf_dens_func = lambda x, f=disk_surf_dens_func_log: np.exp(f(np.log(x)))

    # Create aspect ratio function from input arrays
    disk_aspect_ratio_func_log = scipy.interpolate.CubicSpline(
        np.log(trunc_aspect_ratio_data[0]), np.log(trunc_aspect_ratio_data[1]))
    disk_aspect_ratio_func = lambda x, f=disk_aspect_ratio_func_log: np.exp(f(np.log(x)))

    # Create opacity function from input arrays
    disk_opacity_func_log = scipy.interpolate.CubicSpline(
        np.log(trunc_opacity_data[0]), np.log(trunc_opacity_data[1]))
    disk_opacity_func = lambda x, f=disk_opacity_func_log: np.exp(f(np.log(x)))

    # Create sound speeds function from input arrays
    sound_speeds_func_log = scipy.interpolate.CubicSpline(
        np.log(trunc_sound_speed_data[0]), np.log(trunc_sound_speed_data[1]))
    disk_sound_speed_func = lambda x, f=sound_speeds_func_log: np.exp(f(np.log(x)))

    # Create densities function from input arrays
    disk_densities_func_log = scipy.interpolate.CubicSpline(
        np.log(trunc_density_data[0]), np.log(trunc_density_data[1]))
    disk_density_func = lambda x, f=disk_densities_func_log: np.exp(f(np.log(x)))

    # Create omegas function from input arrays
    disk_omegas_func_log = scipy.interpolate.CubicSpline(
        np.log(trunc_omega_data[0]), np.log(trunc_omega_data[1]))
    disk_omega_func = lambda x, f=disk_omegas_func_log: np.exp(f(np.log(x)))

    # Create pressure gradients function from input arrays
    # Need to preserve signs, since log only takes positive values
    # Multiply final value by the correct sign
    disk_pressure_gradient_func = scipy.interpolate.CubicSpline(
        trunc_pressure_data[0], trunc_pressure_data[1])
    #disk_pressure_gradient_func = lambda x, f=disk_pressure_gradients_func_raw: np.exp(f(np.log(x)))

    # Create temperatures function from input arrays
    disk_temperatures_func_log = scipy.interpolate.CubicSpline(
        np.log(trunc_temperature_data[0]), np.log(trunc_temperature_data[1]))
    disk_temperature_func = lambda x, f=disk_temperatures_func_log: np.exp(f(np.log(x)))

    # Create log10 Sigma function
    disk_surf_dens_func_log10 = scipy.interpolate.CubicSpline(
        np.log10(trunc_surf_density_data[0]), np.log10(trunc_surf_density_data[1]))
    disk_surf_dens_func_log10_derivative = disk_surf_dens_func_log10.derivative()

    # Create log10 temp function
    disk_temp_func_log10 = scipy.interpolate.CubicSpline(
        np.log10(trunc_temperature_data[0]), np.log10(trunc_temperature_data[1]))
    disk_temp_func_log10_derivative = disk_temp_func_log10.derivative()

    # Create log10 midplane pressure function
    disk_midplane_pressure = (trunc_sound_speed_data[1] ** 2) / trunc_density_data[1]
    disk_pressure_func_log10 = scipy.interpolate.CubicSpline(
        np.log10(trunc_density_data[0]), np.log10(disk_midplane_pressure))
    disk_pressure_func_log10_derivative = disk_pressure_func_log10.derivative()

    # Define properties we want to return
    disk_model_properties ={}
    disk_model_properties['Sigma'] = disk_surf_dens_func
    disk_model_properties['h_over_r'] = disk_aspect_ratio_func
    disk_model_properties['kappa'] = disk_opacity_func

    return disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, disk_sound_speed_func, disk_density_func, disk_pressure_gradient_func, disk_omega_func, disk_surf_dens_func_log, disk_temperature_func, disk_surf_dens_func_log10_derivative, disk_temp_func_log10_derivative, disk_pressure_func_log10_derivative, disk_model_properties


def construct_disk_pAGN(
    disk_model_name,
    smbh_mass,
    disk_radius_outer,
    disk_alpha_viscosity,
    disk_bh_eddington_ratio,
    rad_efficiency=0.1,
    ):
    """Construct AGN disk model using the pAGN code.

    Get 1d functions of radius for your choice of disk model. Disk model can be
    Sirko & Goodman (2003) or Thompson, Quataert, & Murray (2005)

    Sirko and Goodman. “Spectral Energy Distributions of Marginally
    Self-Gravitating Quasi-Stellar Object Discs.” 2003MNRAS.341..501S.
    [DOI](https://doi.org/10.1046/j.1365-8711.2003.06431.x).

    Thompson, Quataert, & Murray. “Radiation Pressure-Supported
    Starburst Disks and Active Galactic Nucleus Fueling.” 2005ApJ.630..167.
    [DOI](https://doi.org/10.1086/431923).

    Parameters
    ----------
    disk_model_name : str
        sirko_goodman or thompson_etal
    smbh_mass : float
        Mass of the supermassive black hole (M_sun)
    disk_radius_outer : float
        final element of disk_model_radius_array (units of r_g)
    disk_alpha_viscosity : float
        disk viscosity 'alpha'
    rad_efficiency : float
        An input for pAGN

    Returns
    -------
    disk_surf_dens_func : lambda
        Surface density (radius)
    disk_aspect_ratio_func : lambda
        Aspect ratio (radius)
    disk_opacity_func : lambda
        Opacity (radius)
    sound_speed_func : lambda
        sound speed (radius) [m/s]
    disk_density_func : lambda
        disk density (radius) [kg/m^3]
    disk_model_properties : dict
        Other disk model things we may want
    bonus_structures : dict
        Other disk model things we may want, which are only available
        for pAGN models
    """
    # instead, populate with pagn
    if "sirko" in disk_model_name:
        pagn_name = "Sirko"
        base_args = {
            'Mbh': smbh_mass*pagn_ct.MSun,
            'alpha': disk_alpha_viscosity, 
            'le': disk_bh_eddington_ratio,
            'eps': rad_efficiency
        }
    elif 'thompson' in disk_model_name:
        pagn_name = 'Thompson'
        base_args = {
            'Mbh': smbh_mass*pagn_ct.MSun,
            'm': disk_alpha_viscosity, 
        }
            #'epsilon': rad_efficiency
            #'le': disk_bh_eddington_ratio,\
        Rg = smbh_mass * ct.M_sun * ct.G / (ct.c**2)
        # pAGN TQM disk models exclude `Rout`, so feed pAGN a slightly
        # larger value (+1%) than the user set for `disk_radius_outer`
        base_args['Rout'] = 1.01 * disk_radius_outer * Rg.to('m').value
    else:
        raise RuntimeError("unknown disk model: %s"%(disk_model_name))

    # note Rin default is 3 Rs

    # Run pAGN
    pagn_model = dm_pagn.AGNGasDiskModel(disk_type=pagn_name, **base_args)
    disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, sound_speed_func, disk_density_func, disk_pressure_grad_func, disk_omega_func, disk_surf_dens_func_log, temp_func, surf_dens_log10_derivative_func, temp_log10_derivative_func, pressure_log10_derivative_func, bonus_structures = \
        pagn_model.return_disk_surf_model()

    # Define properties we want to return
    disk_model_properties = {}
    disk_model_properties['Sigma'] = disk_surf_dens_func
    disk_model_properties['h_over_r'] = disk_aspect_ratio_func
    disk_model_properties['kappa'] = disk_opacity_func
    disk_model_properties['dSigmadR'] = disk_surf_dens_func_log
    disk_model_properties['T'] = temp_func
    return disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, sound_speed_func, disk_density_func, disk_pressure_grad_func, disk_omega_func, disk_surf_dens_func_log, temp_func, surf_dens_log10_derivative_func, temp_log10_derivative_func, pressure_log10_derivative_func, disk_model_properties, bonus_structures


def construct_disk_interp(
    smbh_mass,
    disk_radius_outer,
    disk_model_name,
    disk_alpha_viscosity,
    disk_bh_eddington_ratio,
    disk_radius_max_pc=0.,
    flag_use_pagn=0,
    verbose=0,
    ):
    """Construct the disk array interpolators

    Parameters
    ----------
        smbh_mass : float
            Mass of the supermassive black hole (M_sun)
        disk_radius_outer : float
            final element of disk_model_radius_array (units of r_g)
        disk_alpha_viscosity : float
            disk viscosity 'alpha'
        disk_radius_max_pc : float
            Maximum disk size in parsecs (0. for off)
        flag_use_pagn : int
            use pAGN if 1. Default is 0.
        disk_model_name : str
            Choice of disk model
        verbose : int
            Print extra stuff if 1. Default is 0.

    Returns
    ------
    disk_surf_dens_func : lambda
        Surface density (radius)
    disk_aspect_ratio_func : lambda
        Aspect ratio (radius)
    disk_opacity_func : lambda
        Opacity (radius)
    """
    ## Check inputs ##
    # Check smbh_mass
    assert type(smbh_mass) == float, "smbh_mass expected float, got %s"%(type(smbh_mass))

    # open the disk model surface density file and read it in
    # Note format is assumed to be comments with #
    #   density in SI in first column
    #   radius in r_g in second column
    #   infile = model_surface_density.txt, where model is user choice
    if not(flag_use_pagn):
        # Load interpolators
        disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, sound_speed_func, disk_density_func, disk_pressure_grad_func, disk_omega_func, disk_surf_dens_func_log, temp_func, surf_dens_log10_derivative_func, temp_log10_derivative_func, pressure_log10_derivative_func, disk_model_properties = \
            construct_disk_direct(
                disk_model_name,
                disk_radius_outer,
                verbose=verbose
            )

    else:
        # instead, populate with pagn
        disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, sound_speed_func, disk_density_func, disk_pressure_grad_func, disk_omega_func, disk_surf_dens_func_log, temp_func, surf_dens_log10_derivative_func, temp_log10_derivative_func, pressure_log10_derivative_func, disk_model_properties, bonus_structures = \
            construct_disk_pAGN(
                disk_model_name,
                smbh_mass,
                disk_radius_outer,
                disk_alpha_viscosity,
                disk_bh_eddington_ratio,
            )

    # Truncate disk models at outer disk radius
    if verbose:
        print("I read and digested your disk model")
        print("Sending variables back")

    return disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, sound_speed_func, disk_density_func, disk_pressure_grad_func, disk_omega_func, disk_surf_dens_func_log, temp_func, surf_dens_log10_derivative_func, temp_log10_derivative_func, pressure_log10_derivative_func

def ReadInputs_prior_mergers(fname='recipes/sg1Myrx2_survivors.dat', verbose=0):
    """This function reads your prior mergers from a file user specifies or
    default (recipies/prior_mergers_population.dat), and returns the chosen variables for
    manipulation by main.

    Required input formats and units are given in IOdocumentation.txt file.

    See below for full output list, including units & formats

    Example
    -------
    To run, ensure a prior_mergers_population.dat is in the same directory and type:

        $ python ReadInputs_prior_mergers.py

    Notes
    -----
    Function will tell you what it is doing via print statements along the way.

    Attributes
    ----------
    Output variables:
    radius_bh : float
        Location of BH in disk
    mass_bh : float
        Mass of BH (M_sun)
    spin_bh : float
        Magnitude of BH spin (dimensionless)
    spin_angle_bh : float
        Angle of BH spin wrt L_disk (radians). 0(pi) radians = aligned (anti-aligned) with L_disk
    gen_bh: float
        Generation of BH (integer). 1.0 =1st gen
        (wasn't involved in merger in previous episode; but accretion=mass/spin changed)
    )
    """
    with open(fname, 'r') as filedata:
        prior_mergers_file = np.genfromtxt(filedata, unpack = True)


    #Clean the file of galaxy lines (of form 3.0 3.0 3.0 3.0 3.0 etc for it=3.0, same value across each column)
    cleaned_prior_mergers_file = prior_mergers_file

    radius_list = []
    masses_list = []
    spins_list = []
    spin_angles_list = []
    gens_list = []
    len_columns = prior_mergers_file.shape[1]
    rows_to_be_removed = []

    for i in range(0,len_columns):
        # If 1st and 2nd entries in row i are same, it's an galaxy marker, delete row.
        if prior_mergers_file[0,i] == prior_mergers_file[1,i]:
            rows_to_be_removed = np.append(rows_to_be_removed,int(i))

    rows_to_be_removed=rows_to_be_removed.astype('int32')
    cleaned_prior_mergers_file = np.delete(cleaned_prior_mergers_file,rows_to_be_removed,axis=1)

    radius_list = cleaned_prior_mergers_file[0,:]
    masses_list = cleaned_prior_mergers_file[1,:]
    spins_list = cleaned_prior_mergers_file[2,:]
    spin_angles_list = cleaned_prior_mergers_file[3,:]
    gens_list = cleaned_prior_mergers_file[4,:]

    return radius_list,masses_list,spins_list,spin_angles_list,gens_list

