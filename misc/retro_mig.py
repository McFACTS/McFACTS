import numpy as np
import astropy.constants as const
import astropy.units as u

def retro_mig(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
              disk_bh_retro_orbs_e, disk_bh_retro_orbs_inc, disk_bh_retro_arg_periapse,
              timestep_duration_yr, disk_surf_density_func):
    """Apply change to retrograde orbiters' semi-major axes (migration) due to dynamical friction.

    This function calculates how fast the semi-major axis of a retrograde single orbiter
    changes due to dynamical friction (appropriate for BH, NS, maaaybe WD?--check) using
    Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL). It returns the new locations of the retrograde
    orbiters after 1 timestep. Note we have assumed the masses of the orbiters are
    negligible compared to the SMBH (<1% should be fine).

    Funny story: if inc = pi exactly, the semi-major axis decay is stupid fast--in fact
    at modest initial a, the decay speed (a_0/tau_a_dyn) is > c, so that's wrong.
    I think this is due to the sin(inc) in tau_a_dyn. However, if you're just a bit
    away from inc = pi (say, pi - 1e-6--but haven't done thorough param search) 
    you get something like sensible answers.
    So we gotta watch out for this

    Parameters
    ----------
    smbh_mass : float/ndarray
        Mass of supermassive black hole in units of solar masses.
    disk_bh_retro_orbs_a : float/ndarray
        Semi-major axes of retrograde singleton BH at start of timestep in units of
        gravitational radii (r_g=GM_SMBH/c^2).
    disk_bh_retro_masses : float/ndarray
        Mass of retrograde singleton BH at start of timestep in units of solar masses.
    disk_bh_retro_orbs_e : float/ndarray
        Orbital eccentricity of retrograde singleton BH at start of timestep.
    disk_bh_retro_orbs_inc : float/ndarray
        Orbital inclination of retrograde singleton BH at start of timestep.
    disk_bh_retro_arg_periapse : float/ndarray
        Argument of periapse of retrograde singleton BH at start of timestep.
    timestep_duration_yr : float
        Size of timestep in years
    disk_surf_density_func : function
        Returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    disk_bh_retro_orbs_a_new : float array
        locations of retrograde singleton BH at end of timestep in units of
        gravitational radii (r_g=GM_SMBH/c^2)

    """

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    smbh_mass_kg = smbh_mass * const.M_sun.si.value
    semi_maj_axis = disk_bh_retro_orbs_a * const.G * smbh_mass_kg \
                    / (const.c ** 2)  # m
    retro_mass = disk_bh_retro_masses * const.M_sun.si.value  # kg
    omega = disk_bh_retro_arg_periapse  # radians
    ecc = disk_bh_retro_orbs_e  # unitless
    inc = disk_bh_retro_orbs_inc  # radians
    timestep_duration_yr = timestep_duration_yr * (1 * u.yr).to(u.s)  # sec

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt((semi_maj_axis ** 3)/(const.G * smbh_mass_kg))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0 - (ecc ** 2))
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + (ecc ** 2) + 2.0 * ecc * np.cos(omega))
    sigma_minus = np.sqrt(1.0 + (ecc ** 2) - 2.0 * ecc * np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc * np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc * np.cos(omega))
    # WZL Eqn 65
    kappa_bar = 0.5 * (np.sqrt(1.0/(eta_plus ** 7)) + np.sqrt(1.0/(eta_minus ** 7)))
    # WZL Eqn 66
    xi_bar = 0.5 * (np.sqrt((sigma_plus ** 4)/(eta_plus ** 13)) + np.sqrt((sigma_minus ** 4)/(eta_minus ** 13)))
    # WZL Eqn 67
    zeta_bar = xi_bar / kappa_bar
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus/(eta_plus ** 2) + sigma_minus/(eta_minus ** 2))
    # WZL Eqn 72
    #   NOTE: preserved retrograde_bh_locations in r_g to feed to disk_surf_model function
    #   tau in units of sec
    tau_a_dyn = (1.0-(ecc ** 2)) * np.sin(inc) * ((delta - np.cos(inc)) ** 1.5) \
                * (smbh_mass_kg ** 2) * period \
                / (retro_mass * disk_surf_density_func(disk_bh_retro_orbs_a) * np.pi * (semi_lat_rec ** 2)) \
                / (np.sqrt(2)) * kappa_bar * np.abs(np.cos(inc) - zeta_bar)

    # assume the fractional change in semi-major axis is the fraction
    #   of tau_a_dyn represented by one timestep; in case of semi-maj axis
    #   always moving inwards (because drag)
    frac_change = timestep_duration_yr / tau_a_dyn

    # if the timescale for change of semi-major axis is larger than the timestep
    #    send the new location to zero (btw we should probably have a special
    #    handling procedure for these--like, remove & count them)
    # we may also want to add a check for if a_0/tau_a_dyn > c (should only cause
    #    issues not handled here at very small a_0 or very large timesteps)
    frac_change[frac_change > 1.0] = 1.0

    disk_bh_retro_orbs_a_new = disk_bh_retro_orbs_a * (1.0 - frac_change)

    return disk_bh_retro_orbs_a_new
