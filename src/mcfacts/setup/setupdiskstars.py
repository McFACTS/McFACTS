import numpy as np
from astropy.constants import G
from mcfacts.mcfacts_random_state import rng
from mcfacts.setup import setupdiskblackholes


def setup_disk_stars_orb_a(star_num, disk_radius_outer, disk_inner_stable_circ_orb):
    """Generates initial single star semi-major axes

    BH semi-major axes are distributed randomly uniformly through disk of radial size :math:`\\mathtt{disk_outer_radius}`

    Parameters
    ----------
    star_num : int
        number of stars
    disk_radius_outer : float
        outer radius of disk, maximum semimajor axis for stars
    disk_inner_stable_circ_orb : float
        Inner radius of disk [r_{g,SMBH}]

    Returns
    -------
    star_orb_a_initial : numpy.ndarray
        semi-major axes for stars
    """

    # Generating star locations in an x^2 distribution
    x_vals = rng.uniform(low=(disk_inner_stable_circ_orb/disk_radius_outer) ** 2, high=1, size=star_num)
    star_orb_a_initial = np.sqrt(x_vals) * disk_radius_outer

    return (star_orb_a_initial)


def setup_disk_stars_masses(star_num,
                            disk_star_mass_min_init,
                            disk_star_mass_max_init,
                            nsc_imf_star_powerlaw_index):
    """
    Generate star masses using powerlaw IMF.

    Parameters
    ----------
    star_num : int
        number of stellar masses
    disk_star_mass_min_init : float
        minimum star mass
    disk_star_mass_max_init : float
        maximum star mass
    nsc_imf_star_powerlaw_index : float
        powerlaw index

    Returns
    -------
    masses : numpy.ndarray
        stellar masses
    """

    # Convert min and max mass to x = m ^ {-p + 1} format
    x_min = disk_star_mass_min_init**(-nsc_imf_star_powerlaw_index+1.)
    x_max = disk_star_mass_max_init**(-nsc_imf_star_powerlaw_index+1.)

    # Get array of uniform random numbers
    p_vals = rng.uniform(low=0.0, high=1.0, size=star_num)

    # Calculate x values
    x_vals = x_min - p_vals * (x_min - x_max)

    # Convert back to mass
    masses = (x_vals**(1./(-nsc_imf_star_powerlaw_index+1)))

    return (masses)


def setup_disk_stars_radius(masses):
    """Calculate stellar radii. Radii is set with the mass-radius relation.

    Parameters
    ----------
    masses : numpy.ndarray
        stellar masses

    Returns
    -------
    star_log_radius : numpy.ndarray
        log of stellar radii
    """
    star_radius_initial = (masses**0.8)
    star_log_radius = np.log10(star_radius_initial)
    return (star_log_radius)


def setup_disk_stars_comp(star_num,
                          star_ZAMS_metallicity,
                          star_ZAMS_helium):
    """
    Set the initial chemical composition. For now all stars have
    the same initial composition.

    Parameters
    ----------
    star_num : int
        number of stars
    star_ZAMS_metallicity : float
        initial metals mass fraction
    star_ZAMS_hydrogen : float
        initial hydrogen mass fraction
    star_ZAMS_helium : float
        initial helium mass fraction

    Returns
    -------
    star_X : numpy array
        array for the hydrogen mass fraction
    star_Y : numpy array
        array for the helium mass fraction
    star_Z : numpy array
        array for the metals mass fraction
    """
    # For now the numbers are set in the input file. Maybe we want to
    # just set 2 of them (X and Y?) and calculate the final so it adds to 1?
    # Distribution function option at some point?

    star_ZAMS_hydrogen = 1. - star_ZAMS_helium - star_ZAMS_metallicity

    star_Z_initial = np.full(star_num, star_ZAMS_metallicity)
    star_X_initial = np.full(star_num, star_ZAMS_hydrogen)
    star_Y_initial = np.full(star_num, star_ZAMS_helium)
    return (star_X_initial, star_Y_initial, star_Z_initial)


def setup_disk_stars_orb_ang_mom_full(star_num,
                                 mass, smbh_mass,
                                 orb_a, orb_inc,):
    """
    Calculate initial orbital angular momentum from Keplerian orbit formula
    for L and add a random direction (+ or -) for prograde vs retrograde
    orbiters.

    Parameters
    ----------
    star_num : int
        number of stars
    mass_reduced : float
        reduced mass, calculated as mass*smbh_mass/(mass+smbh_mass)
    mass_total : float
        total mass, calculated as smbh_mass + mass
    orb_a : numpy.ndarray
        orbital semi-major axis
    orb_inc : numpy.ndarray
        orbital inclination

    Returns
    -------
    star_orb_ang_mom_initial : numpy.ndarray
        orbital angular momentum
    """
    mass_total = mass + smbh_mass
    mass_reduced = (mass * smbh_mass) / mass_total
    star_orb_ang_mom_initial_sign = rng.choice(a=[1., -1.], size=star_num)
    star_orb_ang_mom_initial_value = mass_reduced*np.sqrt(G.to('m^3/(M_sun s^2)').value*mass_total*orb_a*(1-orb_inc**2))
    star_orb_ang_mom_initial = star_orb_ang_mom_initial_sign*star_orb_ang_mom_initial_value
    return (star_orb_ang_mom_initial)


def setup_disk_stars_orb_ang_mom(star_num):
    """Generates disk star initial orbital angular momenta [unitless]

    Assume either initially fully prograde (+1) or retrograde (-1)

    Parameters
    ----------
        star_num : int
            Integer number of BH initially embedded in disk

    Returns
    -------
        disk_bh_initial_orb_ang_mom : numpy.ndarray
            Initial BH orb ang mom [unitless] with :obj:`float` type. No units because it is an on/off switch.
    """

    disk_star_initial_orb_ang_mom = rng.choice(a=[1.,-1.],size=star_num)
    return disk_star_initial_orb_ang_mom


def setup_disk_stars_arg_periapse(star_num):
    """
    Return an array of star arguments of periapse
    Should be fine to do a uniform distribution between 0-2pi
    Someday we should probably pick all the orbital variables
      to be consistent with a thermal distribution or something
      otherwise physically well-motivated...

    Hahahaha, actually, no, for now everything will be at 0.0 or pi/2
    For now, bc crude retro evol only treats arg periapse = (0.0, pi) or pi/2
    and intermediate values are fucking up evolution
    (and pi is symmetric w/0, pi/2 symmetric w/ 3pi/2 so doesn't matter)
    when code can handle arbitrary arg of periapse, uncomment relevant line

    Parameters
    ----------
    star_num : int
        number of stars

    Returns
    -------
    star_initial_orb_arg_periapse : numpy.ndarray
        arguments for orbital periapse
    """

    star_orb_arg_periapse_initial = rng.choice(a=[0., 0.5*np.pi],size=star_num)

    return (star_orb_arg_periapse_initial)


def setup_disk_stars_eccentricity_thermal(star_num):
    """
    Return an array of star orbital eccentricities
    For a thermal initial distribution of eccentricities, select from a
    uniform distribution in e^2.
    Thus (e=0.7)^2 is 0.49 (half the eccentricities are <0.7) and
    (e=0.9)^2=0.81(about 1/5th eccentricities are >0.9)
    So rnd= draw from a uniform [0,1] distribution, allows ecc=sqrt(rnd) for
    thermal distribution.
    Thermal distribution in limit of equipartition of energy after multiple
    dynamical encounters


    Parameters
    ----------
    star_num : int
        number of stars

    Returns
    -------
    star_initial_orb_ecc : float array
        orbital eccentricities
    """
    random_uniform_number = rng.uniform(size=star_num)
    star_orb_ecc_initial = np.sqrt(random_uniform_number)
    return (star_orb_ecc_initial)


def setup_disk_stars_eccentricity_uniform(star_num, disk_star_orb_ecc_max_init):
    """
    Return an array of star orbital eccentricities
    For a uniform initial distribution of eccentricities, select from
    a uniform distribution in e.
    Thus half the eccentricities are <0.5 and about 1/10th eccentricities
    are >0.9
    So rnd = draw from a uniform [0,1] distribution,
    allows ecc = rnd for uniform distribution
    Most real clusters/binaries lie between thermal & uniform
    (e.g. Geller et al. 2019, ApJ, 872, 165)


    Parameters
    ----------
    star_num : int
        number of stars

    Returns
    -------
    star_initial_orb_ecc : float array
        orbital eccentricities
    """
    random_uniform_number = rng.uniform(size=star_num)
    star_orb_ecc_initial = random_uniform_number * disk_star_orb_ecc_max_init
    return (star_orb_ecc_initial)


def setup_disk_stars_inc(star_num, star_orb_a, star_orb_ang_mom,
                         disk_aspect_ratio):
    """
    NEED TO UPDATE STAR FUNCTIONS TO CALL THS INSTEAD
    Return an array of star orbital inclinations
    Return an initial distribution of inclination angles that are 0.0

    To do: initialize inclinations so random draw with i <h (so will need to
    input star locations and disk_aspect_ratio)
    and then damp inclination.
    To do: calculate v_kick for each merger and then the (i,e) orbital elements
    for the newly merged star.
    Then damp (i,e) as appropriate

    Parameters
    ----------
    star_num : int
        number of stars
    star_orb_a : numpy array
        orbital semi-major axes of the star
    star_orb_ang_mom : numpy array
        orbital angular momentum
    disk_aspect_ratio : float
        aspect ratio of the disk

    Returns
    -------
    star_orb_inc_initial : numpy array
        orbital inclinations
    """

    # Return an array of star orbital inclinations
    # initial distribution is not 0.0
    # what is the max height at the orbiter location that keeps it in the disk
    max_height = star_orb_a * disk_aspect_ratio(star_orb_a)
    # reflect that height to get the min
    min_height = -max_height
    random_uniform_number = rng.uniform(size=star_num)
    # pick the actual height between the min and max, then reset zero point
    height_range = max_height - min_height
    actual_height_range = height_range * random_uniform_number
    actual_height = actual_height_range + min_height
    # inclination is arctan of height over radius, modulo pro or retrograde
    star_orb_inc_initial = np.arctan(actual_height/star_orb_a)
    # for retrogrades, add 180 degrees
    star_orb_inc_initial[star_orb_ang_mom < 0.0] = star_orb_inc_initial[star_orb_ang_mom < 0.0] + np.pi

    return (star_orb_inc_initial)


def setup_disk_stars_inclination(star_num):
    """
    Return an array of star orbital inclinations
    Return an initial distribution of inclination angles that are 0.0

    To do: initialize inclinations so random draw with i <h (so will need to
    input star locations and disk_aspect_ratio)
    and then damp inclination.
    To do: calculate v_kick for each merger and then the (i,e) orbital elements
    for the newly merged star.
    Then damp (i,e) as appropriate

    Parameters
    ----------
    star_num : int
        number of stars

    Returns
    -------
    star_orb_inc_initial : numpy array
        orbital inclinations
    """

    # For now, inclinations are zeros
    star_orb_inc_initial = np.zeros(shape=star_num, dtype=float)
    return (star_orb_inc_initial)


def setup_disk_stars_circularized(star_num, crit_ecc):
    """
    Return an array of BH orbital inclinations
    Return an initial distribution of inclination angles that are 0.0

    To do: initialize inclinations so random draw with i <h (so will need to
    input star_orb_as and disk_aspect_ratio)
    and then damp inclination.
    To do: calculate v_kick for each merger and then the (i,e) orbital elements
    for the newly merged BH.
    Then damp (i,e) as appropriate

    Parameters
    ----------
    star_num : int
        number of stars
    crit_ecc : float
        critical eccentricity

    Returns
    -------
    star_orb_ecc_initial : numpy array
        orbital eccentricities
    """

    # For now, inclinations are zeros
    # Try zero eccentricities
    star_orb_ecc_initial = crit_ecc*np.ones(shape=star_num, dtype=float)
    return (star_orb_ecc_initial)


def setup_disk_stars_num(nsc_mass, nsc_ratio_bh_num_star_num, nsc_ratio_mbh_mass_star_mass,
                         disk_star_scale_factor,
                         nsc_radius_outer, nsc_density_index_outer, smbh_mass, disk_radius_outer,
                         disk_aspect_ratio_avg, nsc_radius_crit, nsc_density_index_inner):
    """Calculates integer number of BH in the AGN disk as calculated from user inputs for NSC and disk

    Parameters
    ----------
        nsc_mass : float
            Mass of Nuclear Star Cluster [M_sun]. Set by user. Default is mass of Milky Way NSC = 3e7M_sun.
        nsc_ratio_bh_num_star_num : float
            Ratio of number of BH in NSC to number of stars [unitless]. Set by user. Default is 1.e-3.
        nsc_ratio_mbh_mass_star_mass : float
            Ratio of mass of typical BH in NSC to typical star in NSC [unitless]. Set by user. Default is 10 (BH=10M_sun,star=1M_sun)
        disk_star_scale_factor : float
            Scale factor [unitless] to go from number of BH to number of stars. Set by user.
        nsc_radius_outer : float
            Outer radius of NSC [pc]. Set by user. Default is 5pc.
        nsc_density_index_outer : float
            NSC density powerlaw index in outer regions. Set by user. 
            NSC density n(r) is assumed to consist of a broken powerlaw distribution,
            with one powerlaw in inner regions (Bahcall-Wolf, r^{-7/4} usually) and one in the outer regions.
            This is the outer region NSC powerlaw density index. Default is :math:`n(r) \\propto r^{-5/2}`
        smbh_mass : float
            Mass of the SMBH [M_sun]. Set by user. Default is 1.e8M_sun.
        disk_radius_outer : float
            Outer radius of disk [r_{g,SMBH}]. Set by user. Default is 5.e4r_g (or 0.25pc around a 10^8M_sun)
        disk_aspect_ratio_avg : float
            Average disk aspect ratio [unitless]. Set by user. Default is h=0.03.
        nsc_radius_crit : float
            NSC critical radius [pc]. Set by user.
            Radius at which NSC density changes from inner powerlaw index to outer powerlaw index.
        nsc_density_index_inner : float
            NSC density powerlaw index in inner regions [unitless]. Set by user.
            Default is :math:`n(r) \propto r^{-7/4}` (Bahcall-Wolf)

    Returns
    -------
        disk_star_num : int
            Number of stars in the AGN disk
    """

    # Get number of BH in disk
    disk_bh_num = setupdiskblackholes.setup_disk_nbh(nsc_mass, nsc_ratio_bh_num_star_num, nsc_ratio_mbh_mass_star_mass,
                                                     nsc_radius_outer, nsc_density_index_outer, smbh_mass, disk_radius_outer,
                                                     disk_aspect_ratio_avg, nsc_radius_crit, nsc_density_index_inner)

    # Number of stars is number of BH divided by the number ratio of BH to stars and multiplied by scale factor
    disk_star_num = (disk_bh_num / nsc_ratio_bh_num_star_num) * disk_star_scale_factor

    return (int(disk_star_num))
