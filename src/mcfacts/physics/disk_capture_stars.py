from mcfacts.setup import setupdiskstars
from mcfacts.physics.point_masses import r_g_from_units
from mcfacts.mcfacts_random_state import rng
from mcfacts.physics import stellar_interpolation
import astropy.constants as const
import astropy.units as u
import numpy as np


def stellar_mass_captured_nsc(disk_lifetime, smbh_mass, nsc_density_index_inner, nsc_mass,
                              nsc_ratio_bh_num_star_num, nsc_ratio_bh_mass_star_mass, disk_surface_density_func,
                              disk_star_mass_min_init, disk_star_mass_max_init, nsc_imf_star_powerlaw_index):
    """Calculate total stellar mass captured from the NSC over the lifetime of the disk

    Note from WZL2024: We assume the surface density scales with r^-3/2,
    which is true for self-gravitating disk models with constant accretion rate

    Parameters
    ----------
    disk_lifetime : float
        Lifetime of the disk [yr]
    smbh_mass : float
        Mass [Msun] of the SMBH
    nsc_density_index_inner : float
        Powerlaw index for the NSC density
    nsc_mass : float
        Mass of the NSC
    nsc_ratio_bh_num_star_num : float
        Ratio of number of BH to number of stars in the NSC
    nsc_ratio_bh_mass_star_mass : float
        Ratio of mass of typical BH to typical star in the NSC
    disk_surface_density_func : lambda function
        Disk density
    disk_star_mass_min_init : float
        Star mass [Msun] lower bound for IMF
    disk_star_mass_max_init : float
        Star mass [Msun] upper bound for IMF
    nsc_imf_star_powerlaw_index : float
        Powerlaw index for IMF

    Returns
    -------
    captured_mass : float
        Amount of stellar mass captured by the NSC in the disk lifetime
    """

    # Convert to SI units
    disk_lifetime_si = disk_lifetime * u.year
    smbh_mass_si = smbh_mass * u.Msun
    nsc_mass_si = nsc_mass * u.Msun

    # Total mass of BH in NSC
    total_mass_bh_in_nsc = nsc_mass_si * nsc_ratio_bh_num_star_num * nsc_ratio_bh_mass_star_mass
    # Total mass of star in NSC (we assume nsc_mass = mass_bh_total + mass_star_total)
    total_mass_star_in_nsc = nsc_mass_si - total_mass_bh_in_nsc

    # Mass fraction of stars in NSC
    f_star = total_mass_star_in_nsc / nsc_mass_si

    # sigma_NSC = 2.3 km/s (M_SMBH / Msun) ^ (1/4.38) given by Kormendy and Ho (2013), 2013ARA&A..51..511K
    disk_velocity_dispersion = (2.3 * u.km / u.second) * ((smbh_mass_si / u.Msun) ** (1. / 4.38))

    # Gravitational influence radius for disk
    disk_radius_of_gravitational_influence_si = ((const.G * smbh_mass_si) / (disk_velocity_dispersion ** 2)).to("pc")
    disk_radius_of_gravitational_influence_rg = r_g_from_units(smbh_mass_si, disk_radius_of_gravitational_influence_si)

    # Surface density at the gravitational influence radius
    disk_surface_density_at_rm_rg = disk_surface_density_func(disk_radius_of_gravitational_influence_rg) * u.kg / u.m**2

    # Disk orbital period at gravitational influence radius
    disk_orbital_period = (2 * np.pi * np.sqrt((disk_radius_of_gravitational_influence_si ** 3) / (const.G * smbh_mass_si))).to("second")

    star_mass_average = (setupdiskstars.setup_disk_stars_mass_avg(disk_star_mass_min_init, disk_star_mass_max_init, nsc_imf_star_powerlaw_index)) * u.Msun

    # given by eqn 46 in WZL2024, Sigma_sun * (m / Msun) ^-1/2
    star_surface_density = ((1.39e11) * (u.gram/u.cm**2) * ((star_mass_average / u.Msun) ** -0.5))

    captured_mass = (2. * smbh_mass_si * f_star * ((disk_surface_density_at_rm_rg / star_surface_density) * (disk_lifetime_si / disk_orbital_period)) ** (1. - (nsc_density_index_inner / 3.))).to("Msun")

    return (captured_mass.value)


def setup_captured_stars_masses(captured_star_mass, disk_star_mass_min_init, disk_star_mass_max_init, nsc_imf_star_powerlaw_index):
    """Generate array of captured star masses, sorted by greatest to least mass

    Parameters
    ----------
    captured_star_mass : float
        Total stellar mass [Msun] captured by the NSC over the lifetime of the disk
    disk_star_mass_min_init : float
        Lower bound [Msun] for stellar IMF
    disk_star_mass_max_init : float
        Upper bound [Msun] for stellar IMF
    nsc_imf_star_powerlaw_index : float
        Powerlaw index for stellar IMF


    Returns
    -------
    star_masses : array
        Captured star masses, sorted from greatest to least
    """

    star_mass_average = setupdiskstars.setup_disk_stars_mass_avg(disk_star_mass_min_init, disk_star_mass_max_init, nsc_imf_star_powerlaw_index)

    star_num = int(np.rint(captured_star_mass / star_mass_average))

    star_masses = setupdiskstars.setup_disk_stars_masses(star_num, disk_star_mass_min_init, disk_star_mass_max_init, nsc_imf_star_powerlaw_index)
    star_masses = np.sort(star_masses)[::-1]

    return (star_masses)


def setup_captured_stars_orbs_a(num_stars_captured, disk_lifetime, smbh_mass, disk_surface_density_func,
                                disk_star_mass_min_init, disk_star_mass_max_init, nsc_imf_star_powerlaw_index):
    """Generate array of orb_a values for captured stars

    Captured stars are placed in the disk with radial distribution dN/dr \propto r^-1/4

    Parameters
    ----------
    num_stars_captured : int
        Number of stars captured
    disk_lifetime : float
        Disk lifetime [yr]
    smbh_mass : float
        SMBH mass [Msun]
    disk_inner_stable_circ_orb : float
        Disk innermost stable circular radius [R_{g,SMBH}]
    disk_radius_outer : float
        Disk outer radius [R_{g,SMBH}]
    disk_surface_density_func : lambda function
        Disk surface density
    disk_star_mass_min_init : float
        Lower bound [Msun] for stellar IMF
    disk_star_mass_max_init : float
        Upper bound [Msun] for stellar IMF
    nsc_imf_star_powerlaw_index : float
        Powerlaw index for stellar IMF

    Returns
    -------
    captured_star_orb_a : array
        Array of orb_a values for captured stars
    """

    # Convert to SI units
    disk_lifetime_si = disk_lifetime * u.year
    smbh_mass_si = smbh_mass * u.Msun

    # sigma_NSC = 2.3 km/s (M_SMBH / Msun) ^ (1/4.38) given by Kormendy and Ho (2013), 2013ARA&A..51..511K
    disk_velocity_dispersion = (2.3 * u.km / u.second) * ((smbh_mass_si / u.Msun) ** (1. / 4.38))

    # Gravitational influence radius for disk
    disk_radius_of_gravitational_influence_si = ((const.G * smbh_mass_si) / (disk_velocity_dispersion ** 2)).to("pc")
    disk_radius_of_gravitational_influence_rg = r_g_from_units(smbh_mass_si, disk_radius_of_gravitational_influence_si)

    # Surface density at the gravitational influence radius
    disk_surface_density_at_rm_rg = disk_surface_density_func(disk_radius_of_gravitational_influence_rg) * u.kg / u.m**2

    star_mass_average = setupdiskstars.setup_disk_stars_mass_avg(disk_star_mass_min_init, disk_star_mass_max_init, nsc_imf_star_powerlaw_index)
    star_mass_average_si = star_mass_average * u.Msun

    star_logR_avg, star_logL_avg, star_logT_avg = stellar_interpolation.interp_star_params(np.array([star_mass_average]))

    star_radius_avg = (10 ** star_logR_avg) * u.Rsun

    radius_tde_si = (star_radius_avg * (smbh_mass_si / star_mass_average_si) ** (1/3)).to("pc")
    radius_tde_rg = r_g_from_units(smbh_mass_si, radius_tde_si)

    # given by eqn 46 in WZL2024, Sigma_sun * (m / Msun) ^-1/2
    star_surface_density = ((1.39e11) * (u.gram/u.cm**2) * ((star_mass_average_si / u.Msun) ** -0.5))

    # Disk orbital period at gravitational influence radius
    disk_orbital_period = (2 * np.pi * np.sqrt((disk_radius_of_gravitational_influence_si ** 3) / (const.G * smbh_mass_si))).to("second")

    star_orb_a_max_si = (disk_radius_of_gravitational_influence_si * ((disk_surface_density_at_rm_rg * disk_lifetime_si) / (star_surface_density * disk_orbital_period)) ** (1./3.)).to("pc")
    star_orb_a_max_rg = r_g_from_units(smbh_mass_si, star_orb_a_max_si)

    # Captured stars are distributed between disk_inner_stable_circ_orb and star_orb_a_max with a powerlaw distribution r^-1/4
    x_vals = rng.uniform(low=0, high=1, size=num_stars_captured)
    captured_star_orb_a = (x_vals ** (4/3)) * (star_orb_a_max_rg - radius_tde_rg) + radius_tde_rg

    return (captured_star_orb_a.value)


def distribute_captured_stars(captured_stars_masses, captured_stars_orb_a, timestep_num, timestep_duration_yr):
    """Distribute captured stars' masses and orb_a based on how many are captured per timestep

    Assumes timesteps are uniform and evenly spaced

    Parameters
    ----------
    captured_stars_masses : numpy.ndarray
        Masses [Msun] of captured stars, sorted by greatest to least
    captured_stars_orb_a : numpy.ndarray
        Semi-major axis [R_{g,SMBH}] of captured stars
    timestep_num : int
        Number of timesteps per galaxy

    Returns
    -------
    captured_stars : dict
        Captured stars' mass and orb_a values. Dictionary key is timestep they are captured in, and value is a tuple
        of number of captured stars, array of captured masses, and array of captured orb_a
    """

    assert captured_stars_masses.size == captured_stars_orb_a.size

    star_num = len(captured_stars_masses)

    per_timestep = int(star_num / timestep_num)
    extra_stars = star_num - (per_timestep * timestep_num)
    stars_per_timestep = np.full(timestep_num, per_timestep)

    # add in the extra n stars to the first n elements of the array
    stars_per_timestep[:extra_stars] = stars_per_timestep[:extra_stars] + 1

    # randomize so we don't capture all the "extras" right at the beginning
    rng.shuffle(stars_per_timestep)

    assert star_num == np.sum(stars_per_timestep), "star_num != np.sum(stars_per_timestep): Not all captured stars accounted for by end of galaxy"

    timestep_arr = np.arange(timestep_num) * timestep_duration_yr

    assert len(timestep_arr) == len(stars_per_timestep), "Lengths of timestep_arr and stars_per_timestep do not match"

    # Shift masses backwards/forwards with a Gaussian
    shift_idx_by = np.rint(rng.normal(loc=0, scale=1, size=star_num)).astype(int)
    shift_max = np.abs(shift_idx_by).max()

    # Pad mass indices because otherwise some masses will end up with negative indices and wrap around to back
    masses_idx = np.arange(star_num) + shift_max * 2
    shifted_idx = masses_idx + shift_idx_by

    assert np.all(shifted_idx >= 0), "shifted_idx contains negative values, will wrap around to back"

    blank_arr = np.zeros(shifted_idx.max() + 1)
    blank_arr = np.insert(blank_arr, shifted_idx, captured_stars_masses)
    masses_shuffled = blank_arr[np.nonzero(blank_arr)]

    # Put captured masses and orb_a in dictionary with designated timestep as key
    start_idx = 0
    end_idx = 0
    captured_stars = {}
    check_sum = 0
    for timestep, cap_num in zip(timestep_arr, stars_per_timestep):
        end_idx += cap_num
        cap_masses = masses_shuffled[start_idx:end_idx]
        check_sum += len(cap_masses)
        cap_orb_a = captured_stars_orb_a[start_idx:end_idx]
        captured_stars[timestep] = (cap_num, cap_masses, cap_orb_a)
        start_idx += cap_num

    assert check_sum == star_num, "Number of stars stored in captured_stars != number of captured stars"

    return (captured_stars)
