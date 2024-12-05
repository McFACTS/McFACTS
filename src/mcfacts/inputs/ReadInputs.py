"""Define input handling functions for mcfacts_sim

Inifile
-------
    "disk_model_name"               : str
        'sirko_goodman' or 'thompson_etal'
    "flag_use_pagn"                 : bool
        Use pAGN to generate disk model?
    "flag_add_stars"                : bool
        Add stars to the disk
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
    "nsc_imf_star_mass_modex"       : float
        Mass mode for star IMF
    "nsc_star_spin_dist_mu"         : float
        Initial spin distribution for stars is assumed to be Gaussian
    "nsc_star_spin_dist_sigma"      : float
        Initial spin distribution for stars is assumed to be Gaussian
        --standard deviation of spin dist
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
    "flag_use_pagn"                 : bool,
    "flag_add_stars"                : bool,
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
    "nsc_imf_star_mass_mode"        : float,
    "nsc_star_spin_dist_mu"         : float,
    "nsc_star_spin_dist_sigma"      : float,
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
    "save_snapshots"                : bool,
}


def ReadInputs_ini(fname_ini, verbose=False):
    """Input file parser

    This function reads your input choices from a file user specifies or
    default (inputs/model_choice.txt), and returns the chosen variables for
    manipulation by main.

    Required input formats and units are given in IOdocumentation.txt file.

    Parameters
    ----------
    fname_ini : str
        Name of inifile for mcfacts
    verbose : bool
        Print extra things

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
            # Bools can behave strangely, so cast as int then convert back to bool
            if INPUT_TYPES[name] == bool:
                input_variables[name] = bool(int(input_variables[name]))
            else:
                input_variables[name] = INPUT_TYPES[name](input_variables[name])
        # If we can't figure it out, check if it's a floating point number
        elif '.' in input_variables[name]:
            input_variables[name]=float(input_variables[name])
        # If it's not a floating point number, try an integer
        elif input_variables[name].isdigit():
            input_variables[name] =int(input_variables[name])
        # If all else fails, leave it the way we found it
        else:
            input_variables[name] = str(input_variables[name])

    # Clean up strings
    for name in input_variables:
        if isinstance(input_variables[name], str):
            input_variables[name] = input_variables[name].strip("'")

    # Set default : not use pagn.  this allows us not to provide it
    if not ('flag_use_pagn' in input_variables):
        input_variables['flag_use_pagn'] = False

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
    verbose=False
    ):
    """Load the dictionary arrays from file (pAGN_off)

    Use import resources to load datafile from src/mcfacts/inputs/data

    Parameters
    ----------
    disk_model_name : str
        sirko_goodman or thompson_etal
    disk_radius_outer : float
        Outer disk radius we truncate at
    verbose : bool
        Print extra things

    Returns
    -------
    truncated_disk_radii : NumPy array (float)
        The disk radius array
    truncated_surface_densities : NumPy array (float)
        The surface density array
    truncated_aspect_ratio : NumPy array (float)
        The aspect ratio array
    """

    # Get density filename
    fname_disk_density = disk_model_name + '_surface_density.txt'
    # Look in the source data
    fname_disk_density = impresources.files(mcfacts_input_data) / fname_disk_density
    # Load data from the surface density file
    disk_density_data = np.loadtxt(fname_disk_density)

    # Get the radii from the data (second column)
    disk_model_radii = disk_density_data[:,1]
    if verbose:
        print("disk_radius_outer", disk_radius_outer)
        print("disk_model_radii", disk_model_radii)
    
    # Get the surface densities from the data (first column)
    disk_surface_densities = disk_density_data[:,0]
    # truncate disk at outer radius
    truncated_disk_radii = np.extract(
        np.where(disk_model_radii < disk_radius_outer),
        disk_model_radii,
    )
    # Truncate surface density array
    truncated_surface_densities = disk_surface_densities[0:len(truncated_disk_radii)]

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
    disk_aspect_ratios = disk_aspect_ratio_data[:,0]
    # Truncate the aspect ratio array
    truncated_aspect_ratios = disk_aspect_ratios[0:len(truncated_disk_radii)]

    # Get opacity filename
    fname_disk_opacity = disk_model_name + '_opacity.txt'
    # Look in the source data
    fname_disk_opacity = impresources.files(mcfacts_input_data) / fname_disk_opacity
    # Load data from opacity file
    disk_opacity_data = np.loadtxt(fname_disk_opacity)
    # Get the opacities from the data (first column)
    disk_opacities = disk_opacity_data[:,0]
    # Truncate disk at outer radius
    truncated_opacities = disk_opacities[0:len(truncated_disk_radii)]

    # Now redefine arrays used to generate interpolating functions in terms of truncated arrays
    return truncated_disk_radii, truncated_surface_densities, truncated_aspect_ratios, truncated_opacities

def construct_disk_direct(
    disk_model_name,
    disk_radius_outer,
    verbose=False
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
    verbose : bool
        Print extra things

    Returns
    -------
    disk_surf_dens_func : lambda
        Surface density (radius)
    disk_aspect_ratio_func : lambda
        Aspect ratio (radius)
    disk_opacity_func : lambda
        Opacity (radius)
    disk_model_properties : dict
        Other disk model things we may want
    """
    # Call the load_disk_arrays function
    disk_model_radii, surface_densities, aspect_ratios, opacities = \
        load_disk_arrays(
        disk_model_name,
        disk_radius_outer,
        verbose=verbose
        )
    print(disk_model_radii)
    # Now generate interpolating functions
    # Create surface density function from input arrays
    disk_surf_dens_func_log = scipy.interpolate.CubicSpline(
        np.log(disk_model_radii), np.log(surface_densities))
    disk_surf_dens_func = lambda x, f=disk_surf_dens_func_log: np.exp(f(np.log(x)))

    # Create aspect ratio function from input arrays
    disk_aspect_ratio_func_log = scipy.interpolate.CubicSpline(
        np.log(disk_model_radii), np.log(aspect_ratios))
    disk_aspect_ratio_func = lambda x, f=disk_aspect_ratio_func_log: np.exp(f(np.log(x)))

    # Create opacity function from input arrays
    disk_opacity_func_log = scipy.interpolate.CubicSpline(
        np.log(disk_model_radii), np.log(opacities))
    disk_opacity_func = lambda x, f=disk_opacity_func_log: np.exp(f(np.log(x)))

    # Define properties we want to return
    disk_model_properties ={}
    disk_model_properties['Sigma'] = disk_surf_dens_func
    disk_model_properties['h_over_r'] = disk_aspect_ratio_func
    disk_model_properties['kappa'] = disk_opacity_func

    return disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, disk_model_properties

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
        base_args['Rout'] = disk_radius_outer * Rg.to('m').value
    else:
        raise RuntimeError("unknown disk model: %s"%(disk_model_name))

    # note Rin default is 3 Rs

    # Run pAGN
    pagn_model =dm_pagn.AGNGasDiskModel(disk_type=pagn_name,**base_args)
    disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, bonus_structures = \
        pagn_model.return_disk_surf_model()

    # Define properties we want to return
    disk_model_properties ={}
    disk_model_properties['Sigma'] = disk_surf_dens_func
    disk_model_properties['h_over_r'] = disk_aspect_ratio_func
    disk_model_properties['kappa'] = disk_opacity_func

    return disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, disk_model_properties, bonus_structures


def construct_disk_interp(
    smbh_mass,
    disk_radius_outer,
    disk_model_name,
    disk_alpha_viscosity,
    disk_bh_eddington_ratio,
    disk_radius_max_pc=0.,
    flag_use_pagn=False,
    verbose=False,
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
        flag_use_pagn : bool
            use pAGN?
        verbose : bool
            Print extra stuff?

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
        disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, disk_model_properties = \
            construct_disk_direct(
                disk_model_name,
                disk_radius_outer,
                verbose=verbose
            )

    else:
        # instead, populate with pagn
        disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, disk_model_properties, bonus_structures = \
            construct_disk_pAGN(
                disk_model_name,
                smbh_mass,
                disk_radius_outer,
                disk_alpha_viscosity,
                disk_bh_eddington_ratio,
            )

    #Truncate disk models at outer disk radius
    if verbose:
        print("I read and digested your disk model")
        print("Sending variables back")

    return disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func

def ReadInputs_prior_mergers(fname='recipes/sg1Myrx2_survivors.dat', verbose=False):
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

