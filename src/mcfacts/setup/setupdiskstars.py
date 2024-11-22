import numpy as np
from astropy.constants import G 
from mcfacts.mcfacts_random_state import rng


def setup_disk_stars_orb_a(star_num, disk_radius_outer):
    """Generates initial single star semi-major axes

    BH semi-major axes are distributed randomly uniformly through disk of radial size :math:`\\mathtt{disk_outer_radius}`

    Parameters
    ----------
    star_num : int
        number of stars
    disk_radius_outer : float
        outer radius of disk, maximum semimajor axis for stars

    Returns
    -------
    star_orb_a_initial : numpy.ndarray
        semi-major axes for stars
    """
    star_orb_a_initial = disk_radius_outer*rng.uniform(size=star_num)
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
    radii : numpy.ndarray
        stellar radii
    """
    star_radius_initial = (masses**0.8)
    return (star_radius_initial)


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


def setup_disk_stars_spins(star_num,
                           nsc_star_spin_dist_mu, nsc_star_spin_dist_sigma):
    """
    Generate initial spins for stars.

    Parameters
    ----------
    star_num : int
        number of stars
    nsc_star_spin_dist_mu : float
        mean for Gaussian distribution
    nsc_star_spin_dist_sigma : float
        standard deviation for Gaussian distribution

    Returns
    -------
    star_spins_initial : numpy array
        initial spins for stars
    """
    star_spins_initial = rng.normal(loc=nsc_star_spin_dist_mu,
                                    scale=nsc_star_spin_dist_sigma,
                                    size=star_num)
    return (star_spins_initial)


def setup_disk_stars_spin_angles(star_num, star_spins_initial):
    """
    Return an array of star initial spin angles (in radians). Positive
    (negative) spin magnitudes have spin angles
    [0,1.57]([1.5701,3.14])rads. All star spin angles drawn
    from [0,1.57]rads and +1.57rads to negative spin indices

    Parameters
    ----------
    star_num : int
        number of stars
    star_spins_initial : numpy array
        spins of stars

    Returns
    -------
    star_spin_angles_initial : numpy array
        spin angles of stars
    """

    star_initial_spin_indices = np.array(star_spins_initial)
    negative_spin_indices = np.where(star_initial_spin_indices < 0.)
    star_spin_angles_initial = rng.uniform(low=0., high=1.57, size=star_num)
    star_spin_angles_initial[negative_spin_indices] = star_spin_angles_initial[negative_spin_indices] + 1.57
    return (star_spin_angles_initial)


def setup_disk_stars_orb_ang_mom(star_num,
                                 mass_reduced, mass_total,
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
    random_uniform_number = rng.uniform(size=star_num)
    star_orb_ang_mom_initial_sign = (2.0*np.around(random_uniform_number)) - 1.0
    star_orb_ang_mom_initial_value = mass_reduced*np.sqrt(G.to('m^3/(M_sun s^2)').value*mass_total*orb_a*(1-orb_inc**2))
    star_orb_ang_mom_initial = star_orb_ang_mom_initial_sign*star_orb_ang_mom_initial_value
    return (star_orb_ang_mom_initial)


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

    random_uniform_number = rng.uniform(size=star_num)
    star_orb_arg_periapse_initial = 0.5 * np.pi * np.around(random_uniform_number)

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


def setup_disk_stars_eccentricity_uniform(star_num):
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
    star_orb_ecc_initial = random_uniform_number
    return (star_orb_ecc_initial)


def setup_disk_stars_inclination_toinclude(star_num,
                                           star_orb_a,
                                           star_orb_ang_mom,
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
    star_orb_ecc_initial = crit_ecc*np.zeros(shape=star_num, dtype=float)
    return (star_orb_ecc_initial)


def setup_disk_stars_num(nsc_mass,
                         nsc_ratio_bh_num_star_num,
                         nsc_ratio_bh_mass_star_mass,
                         nsc_radius_outer,
                         nsc_density_index_outer,
                         smbh_mass,
                         disk_radius_outer,
                         disk_aspect_ratio_avg,
                         nsc_radius_crit,
                         nsc_density_index_inner):
    """
    Return the integer number of stars in the AGN disk as calculated from NSC
    inputs assuming isotropic distribution of NSC orbits
    To do: Calculate when R_disk_outer is not equal to the nsc_radius_crit
    To do: Calculate when disky NSC population of BH in plane/out of plane.


    Parameters
    ----------
    nsc_mass : float
        mass of the NSC
    nsc_ratio_bh_num_star_num : float
        ratio of number of black holes to number of stars
    nsc_ratio_bh_mass_star_mass : float
        ratio of mass of black holes to mass of stars
    nsc_radius_outer : float
        outer radius of the NSC
    nsc_density_index_outer : float
        inner radius of the NSC
    smbh_mass : float
        mass of the SMBH
    disk_radius_outer : float
        outer radius of the disk
    disk_aspect_ratio_avg : float
        aspect ratio of the disk
    nsc_radius_crit : float
        critical radius of the NSC
    nsc_density_index_inner : float

    Returns
    -------
    disk_star_num_total : int
        number of stars in the disk in total
    """
    # Housekeeping:
    # Convert disk_radius_outer in r_g to units of pc. 1r_g =1AU (M_smbh/10^8Msun)
    # 1pc =2e5AU =2e5 r_g(M/10^8Msun)^-1
    pc_to_rg = 2.e5*((smbh_mass/1.e8)**(-1.0))
    critical_disk_radius_pc = disk_radius_outer/pc_to_rg
    # Total average mass of stars in NSC
    nsc_star_mass = nsc_mass * (1./nsc_ratio_bh_num_star_num) * (1./nsc_ratio_bh_mass_star_mass)
    # Total number of stars in NSC
    nsc_star_num = nsc_star_mass * nsc_ratio_bh_mass_star_mass
    # Relative volumes:
    #   of central 1 pc^3 to size of NSC
    relative_volumes_at1pc = (1.0/nsc_radius_outer)**(3.0)
    #   of nsc_radius_crit^3 to size of NSC
    relative_volumes_at_r_nsc_crit = (nsc_radius_crit/nsc_radius_outer)**(3.0)
    # Total number of stars
    #   at R<1pc (should be about 10^4 for Milky Way parameters; 3x10^7Msun, 5pc, r^-5/2 in outskirts)
    nsc_star_num_orb_a_in_1pc = nsc_star_num * relative_volumes_at1pc * (1.0/nsc_radius_outer)**(-nsc_density_index_outer)
    #   at nsc_radius_crit
    nsc_star_num_orb_a_in_nsc_radius_crit = nsc_star_num * relative_volumes_at_r_nsc_crit * (nsc_radius_crit/nsc_radius_outer)**(-nsc_density_index_outer)

    # Calculate Total number of stars in volume R < disk_radius_outer,
    # assuming disk_radius_outer<=1pc.

    if critical_disk_radius_pc >= nsc_radius_crit:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/1.0)**(3.0)
        disk_star_num_volume_in_disk_radius_outer = nsc_star_num_orb_a_in_1pc * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/1.0)**(-nsc_density_index_outer))
    else:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/nsc_radius_crit)**(3.0)
        disk_star_num_volume_in_disk_radius_outer = nsc_star_num_orb_a_in_nsc_radius_crit * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/nsc_radius_crit)**(-nsc_density_index_inner))

    # Total number of BH in disk
    disk_star_num_total = np.rint(disk_star_num_volume_in_disk_radius_outer * disk_aspect_ratio_avg)
    return np.int64(disk_star_num_total)
