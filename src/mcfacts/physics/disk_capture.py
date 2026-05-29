"""
Module for computing disk-orbiter interactions, which may lead to capture.
"""
import numpy as np
import astropy.constants as const
import astropy.units as u
import mcfacts.setup.setupdiskblackholes as setup
from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.point_masses import si_from_r_g


def orb_inc_damping(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses, disk_bh_retro_orbs_ecc,
                    disk_bh_retro_orbs_inc, disk_bh_retro_arg_periapse, timestep_duration_yr, disk_surf_density_func):
    """Calculates how fast the inclination angle of an arbitrary single orbiter changes due to dynamical friction.

    Appropriate for BH, NS, maaaybe WD?--check using Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL).

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_retro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of retrograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_retro_masses : numpy.ndarray
        Masses [M_sun] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_ecc : numpy.ndarray
        Orbital eccentricities [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_inc : numpy.ndarray
        Orbital inclinations [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_arg_periapse : numpy.ndarray
        Argument of periapse [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_surf_density_func : function
        Method provides the AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH

    Returns
    -------
    disk_bh_retro_orbs_ecc_new : numpy.ndarray
        Orbital inclinations [radian] of retrograde singletons BH at end of a timestep with :obj:`float` type

    Notes
    -----
    It returns the new locations of the retrograde
    orbiters after 1 timestep_duration_yr. Note we have assumed the masses of the orbiters are
    negligible compared to the SMBH (<1% should be fine).

    Unlike all the other orbital variables (semi-major axis, ecc, semi-latus rectum)
    the timescale won't necessarily do anything super horrible for inc = pi, since it only
    depends on the inclination angle itself, sin(inc)... however, it could do something
    horrible at inc=pi for some values of omega; and it WILL go to zero at inc=0, which
    could easily cause problems...

    Also, I think this function will still work fine if you feed it prograde bh
    just change the variable name in the function call... (this is not true for migration)
    Testing implies that inc=0 or pi is actually ok, at least for omega=0 or pi
    """

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    smbh_mass = smbh_mass * u.Msun.to("kg")  # kg
    semi_maj_axis = disk_bh_retro_orbs_a * const.G * smbh_mass \
                    / (const.c ** 2)  # m
    retro_mass = disk_bh_retro_masses * u.Msun.to("kg")  # kg
    omega = disk_bh_retro_arg_periapse  # radians
    ecc = disk_bh_retro_orbs_ecc  # unitless
    inc = disk_bh_retro_orbs_inc  # radians
    timestep_duration_yr = timestep_duration_yr * (1 * u.yr).to(u.s)  # sec

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt((semi_maj_axis ** 3) / (const.G * smbh_mass))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0 - (ecc ** 2))
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + (ecc ** 2) + 2.0 * ecc * np.cos(omega))
    sigma_minus = np.sqrt(1.0 + (ecc ** 2) - 2.0 * ecc * np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc * np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc * np.cos(omega))
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0 / (eta_plus ** 15)) + np.sqrt(1.0 / (eta_minus ** 15)))
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus / (eta_plus ** 2) + sigma_minus / (eta_minus ** 2))
    # WZL Eqn 71
    #   NOTE: preserved disk_bh_retro_orbs_a in r_g to feed to disk_surf_density_func function
    #   tau in units of sec
    tau_i_dyn = np.sqrt(2.0) * inc * ((delta - np.cos(inc)) ** 1.5) \
                * (smbh_mass ** 2) * period / (
                            retro_mass * disk_surf_density_func(disk_bh_retro_orbs_a) * np.pi * (semi_lat_rec ** 2)) \
                / kappa

    # assume the fractional change in inclination is the fraction
    #   of tau_i_dyn represented by one timestep_duration_yr
    frac_change = timestep_duration_yr / tau_i_dyn

    # if the timescale for change of inclination is larger than the timestep_duration_yr
    #    send the new inclination to zero
    frac_change[frac_change > 1.0] = 1.0

    disk_bh_retro_orbs_ecc_new = inc * (1.0 - frac_change)

    return disk_bh_retro_orbs_ecc_new


def retro_bh_orb_disk_evolve(smbh_mass, disk_bh_retro_masses, disk_bh_retro_orbs_a, disk_bh_retro_orbs_ecc,
                             disk_bh_retro_orbs_inc, disk_bh_retro_arg_periapse,
                             disk_inner_stable_circ_orb, disk_surf_density_func, timestep_duration_yr, disk_radius_outer):
    """Evolve the orbit of initially-embedded retrograde black hole orbiters due to disk interactions.

    This is a CRUDE version of evolution, future upgrades may couple to SpaceHub.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_retro_masses : numpy.ndarray
        Mass [M_sun] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of retrograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_retro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_inc : numpy.ndarray
        Orbital inclination [radians] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_arg_periapse : numpy.ndarray
        Argument of periapse [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
    timestep_duration_yr : float
        Length of a timestep [yr]

    Returns
    -------
    disk_bh_retro_orbs_ecc_new : numpy.ndarray
        Updated value of eccentricity [unitless] with :obj:`float` typeafter one timestep_duration_yr assuming gas only evolution hacked together badly...
    disk_bh_retro_orbs_a_new : numpy.ndarray
        Updated value of semi-major axis [r_{g,SMBH}] with :obj:`float` typeafter one timestep_duration_yr assuming gas only evolution hacked together badly...
    disk_bh_retro_orbs_inc_new : numpy.ndarray
        Updated value of orbital inclination [radians] with :obj:`float` typeafter one timestep_duration_yr assuming gas only evolution hacked together badly...

    Notes
    -----
    To avoid having to install and couple to SpaceHub, and run N-body code
    this is a distinctly half-assed treatment of retrograde orbiters, based
    LOOSELY on Wang, Zhu & Lin 2024 (WZL). Evolving all orbital params simultaneously.
    Using lots of if statements to pretend we're interpolating.
    Hardcoding some stuff from WZL figs 7, 8 & 12 (see comments).
    Arg of periapse = w in comments below

    """

    # first handle cos(w)=+/-1 (assume abs(cos(w))>0.5)
    #   this evolution is multistage:
    #       1. radialize, semimaj axis shrinks (slower), flip (very slowly)
    #       2. flip (very fast), circ (slowly), constant semimaj axis
    #       3. i->0.000 (very fast), circ & shrink semimaj axis slightly slower
    #
    #      For smbh_mass=1e8Msun, orbiter_mass=30Msun, SG disk surf dens (fig 12 WZL)
    #       1. in 1.5e5yrs e=0.7->0.9999 (roughly), a=100rg->60rg, i=175->165deg
    #       2. in 1e4yrs i=165->12deg, e=0.9999->0.9, a=60rg
    #       3. in 1e4yrs i=12->0.0deg, e=0.9->0.5, a=60->20rg

    # Housekeeping: if you're too close to ecc=1.0, nothing works, so
    epsilon = 1.0e-8

    # hardcoded awfulness coming up:
    smbh_mass_0 = 1e8  # solar masses, for scaling
    orbiter_mass_0 = 30.0  # solar masses
    periapse_1 = 0.0  # radians
    periapse_0 = np.pi / 2.0  # radians

    step1_ecc_0 = 0.7
    step1_inc_0 = np.pi * (175.0 / 180.0)  # rad
    step1_semi_maj_0 = 100.0  # r_g

    step2_ecc_0 = 0.9999
    step2_inc_0 = np.pi * (165.0 / 180.0)  # rad
    step2_semi_maj_0 = 60.0  # r_g

    step3_ecc_0 = 0.9
    step3_inc_0 = np.pi * (12.0 / 180.0)  # rad
    step3_semi_maj_0 = step2_semi_maj_0  # r_g

    step3_ecc_f = 0.5
    step3_inc_f = 0.0  # rad
    step3_semi_maj_f = 20.0  # r_g

    stepw0_ecc_0 = 0.7
    stepw0_inc_0 = np.pi * (175.0 / 180.0)  # rad
    stepw0_semi_maj_0 = 100.0  # r_g

    stepw0_ecc_f = 0.5
    stepw0_inc_f = np.pi * (170.0 / 180.0)  # rad
    stepw0_semi_maj_f = 60.0  # r_g

    step1_time = 1.5e5  # years
    step1_delta_ecc = step2_ecc_0 - step1_ecc_0
    step1_delta_semimaj = step1_semi_maj_0 - step2_semi_maj_0  #rg
    step1_delta_inc = step1_inc_0 - step2_inc_0  #rad

    step2_time = 1.4e4  # years
    step2_delta_ecc = step2_ecc_0 - step3_ecc_0
    step2_delta_semimaj = step2_semi_maj_0 - step3_semi_maj_0  #rg
    step2_delta_inc = step2_inc_0 - step3_inc_0

    step3_time = 1.4e4  # years
    step3_delta_ecc = step3_ecc_0 - step3_ecc_f
    step3_delta_semimaj = step3_semi_maj_0 - step3_semi_maj_f  #rg
    step3_delta_inc = step3_inc_0 - step3_inc_f

    # Then figure out cos(w)=0
    # this evolution does one thing: shrink semimaj axis, circ (slowly), flip (even slower)
    #   scaling from fig 8 WZL comparing cos(w)=0 to cos(w)=+/-1
    #       tau_semimaj~1/100, tau_ecc~1/1000, tau_inc~1/5000
    #       at high inc, large ecc
    #
    #      Estimating for smbh_mass=1e8Msun, orbiter_mass=30Msun, SG disk surf dens
    #       in 1.5e7yrs a=100rg->60rg, e=0.7->0.5, i=175->170deg
    stepw0_time = 1.5e7  # years
    stepw0_delta_ecc = stepw0_ecc_0 - stepw0_ecc_f
    stepw0_delta_semimaj = stepw0_semi_maj_0 - stepw0_semi_maj_f  #rg
    stepw0_delta_inc = stepw0_inc_0 - stepw0_inc_f

    # setup output arrays
    disk_bh_retro_orbs_ecc_new = np.zeros(len(disk_bh_retro_orbs_ecc))
    disk_bh_retro_orbs_inc_new = np.zeros(len(disk_bh_retro_orbs_inc))
    disk_bh_retro_orbs_a_new = np.zeros(len(disk_bh_retro_orbs_a))

    tau_e_current = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    tau_a_current = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    tau_e_ref = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    tau_a_ref = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    ecc_scale_factor = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    semimaj_scale_factor = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    inc_scale_factor = np.full(disk_bh_retro_arg_periapse.size, -100.5)

    # cosine masks
    # returns True for values where cos(w)~+/-1
    cos_pm1_mask = np.abs(np.cos(disk_bh_retro_arg_periapse)) >= 0.5
    # returns True for values where cos(w)=0 (assume abs(cos(w))<0.5)
    cos_0_mask = np.abs(np.cos(disk_bh_retro_arg_periapse)) < 0.5
    cos_unreliable_mask = ~(cos_pm1_mask | cos_0_mask)
    if cos_unreliable_mask.sum() > 0:
        print("COS Warning: retrograde orbital parameters out of range, behavior unreliable")

    # eccentricity/inclination masks
    # returns True for values where we haven't hit our max ecc for step 1, and remain somewhat retrograde
    no_max_ecc_retro_mask = (disk_bh_retro_orbs_ecc < step2_ecc_0) & (np.abs(disk_bh_retro_orbs_inc) >= np.pi / 2.0)
    # returns True for values where we have hit max ecc, which sends us to step 2
    max_ecc_mask = disk_bh_retro_orbs_ecc >= step2_ecc_0
    # returns True for values where our inc is even barely prograde... hopefully this works ok...
    # this should work as long as we're only tracking stuff originally retrograde
    barely_prograde_mask = np.abs(disk_bh_retro_orbs_inc) < (np.pi / 2.0)
    ecc_unreliable_mask = ~(no_max_ecc_retro_mask | max_ecc_mask | barely_prograde_mask)
    if ecc_unreliable_mask.sum() > 0:
        print("ECC Warning: retrograde orbital parameters out of range, behavior unreliable")

    # Set up arrays for hardcoded values
    semi_maj_0 = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    ecc_0 = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    inc_0 = np.full(disk_bh_retro_arg_periapse.size, -100.5)
    periapse = np.full(disk_bh_retro_arg_periapse.size, -100.5)

    # Fill with values
    semi_maj_0[cos_pm1_mask & no_max_ecc_retro_mask] = step1_semi_maj_0
    semi_maj_0[cos_pm1_mask & max_ecc_mask] = step2_semi_maj_0
    semi_maj_0[cos_pm1_mask & barely_prograde_mask] = step3_semi_maj_0
    semi_maj_0[cos_0_mask] = stepw0_semi_maj_0

    ecc_0[cos_pm1_mask & no_max_ecc_retro_mask] = step1_ecc_0
    ecc_0[cos_pm1_mask & max_ecc_mask] = step2_ecc_0
    ecc_0[cos_pm1_mask & barely_prograde_mask] = step3_ecc_0
    ecc_0[cos_0_mask] = stepw0_ecc_0

    inc_0[cos_pm1_mask & no_max_ecc_retro_mask] = step1_inc_0
    inc_0[cos_pm1_mask & max_ecc_mask] = step2_inc_0
    inc_0[cos_pm1_mask & barely_prograde_mask] = step3_inc_0
    inc_0[cos_0_mask] = stepw0_inc_0

    periapse[cos_pm1_mask] = periapse_1
    periapse[cos_0_mask] = periapse_0

    # Get current tau values
    tau_e_current, tau_a_current = tau_ecc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
                                               disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc,
                                               disk_surf_density_func)
    tau_inc_current = tau_inc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
                                  disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc,
                                  disk_bh_retro_orbs_inc, disk_surf_density_func)

    # Get reference tau values
    tau_e_ref, tau_a_ref = tau_ecc_dyn(smbh_mass_0, semi_maj_0, orbiter_mass_0, periapse, ecc_0, inc_0, disk_surf_density_func)
    tau_inc_ref = tau_inc_dyn(smbh_mass_0, semi_maj_0, orbiter_mass_0, periapse, ecc_0, inc_0, disk_surf_density_func)

    if (tau_e_current == -100.5).sum() > 0:
        print("TAU Warning: retrograde orbital parameters out of range, behavior unreliable")

    # Get ecc scale factors
    tau_e_div = tau_e_current / tau_e_ref
    ecc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask] = step1_time * tau_e_div[cos_pm1_mask & no_max_ecc_retro_mask]
    ecc_scale_factor[cos_pm1_mask & max_ecc_mask] = step2_time * tau_e_div[cos_pm1_mask & max_ecc_mask]
    ecc_scale_factor[cos_pm1_mask & barely_prograde_mask] = step3_time * tau_e_div[cos_pm1_mask & barely_prograde_mask]
    ecc_scale_factor[cos_0_mask] = stepw0_time * tau_e_div[cos_0_mask]
    # Get semimaj scale factors
    tau_a_div = tau_a_current / tau_a_ref
    semimaj_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask] = step1_time * tau_a_div[cos_pm1_mask & no_max_ecc_retro_mask]
    semimaj_scale_factor[cos_pm1_mask & max_ecc_mask] = step2_time * tau_a_div[cos_pm1_mask & max_ecc_mask]
    semimaj_scale_factor[cos_pm1_mask & barely_prograde_mask] = step3_time * tau_a_div[cos_pm1_mask & barely_prograde_mask]
    semimaj_scale_factor[cos_0_mask] = stepw0_time * tau_a_div[cos_0_mask]
    # Get inc scale factors
    tau_inc_div = tau_inc_current / tau_inc_ref
    inc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask] = step1_time * tau_inc_div[cos_pm1_mask & no_max_ecc_retro_mask]
    inc_scale_factor[cos_pm1_mask & max_ecc_mask] = step2_time * tau_inc_div[cos_pm1_mask & max_ecc_mask]
    inc_scale_factor[cos_pm1_mask & barely_prograde_mask] = step3_time * tau_inc_div[cos_pm1_mask & barely_prograde_mask]
    inc_scale_factor[cos_0_mask] = stepw0_time * tau_inc_div[cos_0_mask]

    # Calculate new orb_ecc values
    disk_bh_retro_orbs_ecc_new[cos_pm1_mask & no_max_ecc_retro_mask] = disk_bh_retro_orbs_ecc[cos_pm1_mask & no_max_ecc_retro_mask] * (
        1.0 + step1_delta_ecc / disk_bh_retro_orbs_ecc[cos_pm1_mask & no_max_ecc_retro_mask] * (timestep_duration_yr / ecc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask]))
    disk_bh_retro_orbs_ecc_new[cos_pm1_mask & max_ecc_mask] = disk_bh_retro_orbs_ecc[cos_pm1_mask & max_ecc_mask] * (
        1.0 - step2_delta_ecc / disk_bh_retro_orbs_ecc[cos_pm1_mask & max_ecc_mask] * (timestep_duration_yr / ecc_scale_factor[cos_pm1_mask & max_ecc_mask]))
    disk_bh_retro_orbs_ecc_new[cos_pm1_mask & barely_prograde_mask] = disk_bh_retro_orbs_ecc[cos_pm1_mask & barely_prograde_mask] * (
        1.0 - step3_delta_ecc / disk_bh_retro_orbs_ecc[cos_pm1_mask & barely_prograde_mask] * (timestep_duration_yr / ecc_scale_factor[cos_pm1_mask & barely_prograde_mask]))
    disk_bh_retro_orbs_ecc_new[cos_0_mask] = disk_bh_retro_orbs_ecc[cos_0_mask] * (
        1.0 - stepw0_delta_ecc / disk_bh_retro_orbs_ecc[cos_0_mask] * (timestep_duration_yr / ecc_scale_factor[cos_0_mask]))

    # Calculate new orb_a values
    disk_bh_retro_orbs_a_new[cos_pm1_mask & no_max_ecc_retro_mask] = disk_bh_retro_orbs_a[cos_pm1_mask & no_max_ecc_retro_mask] * (
        1.0 - step1_delta_semimaj / disk_bh_retro_orbs_a[cos_pm1_mask & no_max_ecc_retro_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask]))
    disk_bh_retro_orbs_a_new[cos_pm1_mask & max_ecc_mask] = disk_bh_retro_orbs_a[cos_pm1_mask & max_ecc_mask] * (
        1.0 - step2_delta_semimaj / disk_bh_retro_orbs_a[cos_pm1_mask & max_ecc_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_pm1_mask & max_ecc_mask]))
    disk_bh_retro_orbs_a_new[cos_pm1_mask & barely_prograde_mask] = disk_bh_retro_orbs_a[cos_pm1_mask & barely_prograde_mask] * (
        1.0 - step3_delta_semimaj / disk_bh_retro_orbs_a[cos_pm1_mask & barely_prograde_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_pm1_mask & barely_prograde_mask]))
    disk_bh_retro_orbs_a_new[cos_0_mask] = disk_bh_retro_orbs_a[cos_0_mask] * (
        1.0 - stepw0_delta_semimaj / disk_bh_retro_orbs_a[cos_0_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_0_mask]))

    # Calculate new orb_inc values
    disk_bh_retro_orbs_inc_new[cos_pm1_mask & no_max_ecc_retro_mask] = disk_bh_retro_orbs_inc[cos_pm1_mask & no_max_ecc_retro_mask] * (
        1.0 - step1_delta_inc / disk_bh_retro_orbs_inc[cos_pm1_mask & no_max_ecc_retro_mask] * (timestep_duration_yr / inc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask]))
    disk_bh_retro_orbs_inc_new[cos_pm1_mask & max_ecc_mask] = disk_bh_retro_orbs_inc[cos_pm1_mask & max_ecc_mask] * (
        1.0 - step2_delta_inc / disk_bh_retro_orbs_inc[cos_pm1_mask & max_ecc_mask] * (timestep_duration_yr / inc_scale_factor[cos_pm1_mask & max_ecc_mask]))
    disk_bh_retro_orbs_inc_new[cos_pm1_mask & barely_prograde_mask] = disk_bh_retro_orbs_inc[cos_pm1_mask & barely_prograde_mask] * (
        1.0 - step3_delta_inc / disk_bh_retro_orbs_inc[cos_pm1_mask & barely_prograde_mask] * (timestep_duration_yr / inc_scale_factor[cos_pm1_mask & barely_prograde_mask]))
    disk_bh_retro_orbs_inc_new[cos_0_mask] = disk_bh_retro_orbs_inc[cos_0_mask] * (
        1.0 - stepw0_delta_inc / disk_bh_retro_orbs_inc[cos_0_mask] * (timestep_duration_yr / inc_scale_factor[cos_0_mask]))

    # Catch overshooting ecc = 0
    disk_bh_retro_orbs_ecc_new[disk_bh_retro_orbs_ecc_new < 0.0] = 0.0
    # catch overshooting ecc=1, actually eqns not appropriate for ecc=1.0
    disk_bh_retro_orbs_ecc_new[disk_bh_retro_orbs_ecc_new >= 1.0 - epsilon] = 1.0 - epsilon
    # Catch overshooting semi-major axis, set to disk_inner_stable_circ_orb
    disk_bh_retro_orbs_a_new[disk_bh_retro_orbs_a_new <= 0.0] = disk_inner_stable_circ_orb
    # Catch overshooting inc, set to 0.0
    disk_bh_retro_orbs_inc_new[disk_bh_retro_orbs_inc_new <= 0.0] = 0.0

    # Check Finite
    nan_mask = (
        ~np.isfinite(disk_bh_retro_orbs_ecc_new) | \
        ~np.isfinite(disk_bh_retro_orbs_a_new) | \
        ~np.isfinite(disk_bh_retro_orbs_inc_new) \
    )
    if np.sum(nan_mask) > 0:
        # Check for objects inside 12.1 R_g
        if all(disk_bh_retro_orbs_a[nan_mask] < 12.1):
            disk_bh_retro_orbs_ecc_new[nan_mask] = disk_bh_retro_orbs_ecc[nan_mask]
            # Inside ACTUAL ISCO; might get caught better
            disk_bh_retro_orbs_a_new[nan_mask] = 5.9
            # It's been eaten
            disk_bh_retro_orbs_inc_new[nan_mask] = 0.
        else:
            print("nan_mask:",np.where(nan_mask))
            print("nan old ecc:",disk_bh_retro_orbs_ecc[nan_mask])
            print("disk_bh_retro_masses:", disk_bh_retro_masses[nan_mask])
            print("disk_bh_retro_orbs_a:", disk_bh_retro_orbs_a[nan_mask])
            print("disk_bh_retro_orbs_inc:", disk_bh_retro_orbs_inc[nan_mask])
            print("disk_bh_retro_arg_periapse:", disk_bh_retro_arg_periapse[nan_mask])
            disk_bh_retro_orbs_ecc_new[nan_mask] = 2.
            disk_bh_retro_orbs_a_new[nan_mask] = 0.
            disk_bh_retro_orbs_inc_new[nan_mask] = 0.
            raise RuntimeError("Finite check failed for disk_bh_retro_orbs_ecc_new")

    # Anything outside the disk is brought back in
    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon_orb_a = disk_radius_outer * ((disk_bh_retro_masses / (3 * (disk_bh_retro_masses + smbh_mass)))**(1. / 3.)) * rng.uniform(size=len(disk_bh_retro_masses))
    disk_bh_retro_orbs_a_new[disk_bh_retro_orbs_a_new > disk_radius_outer] = disk_radius_outer - epsilon_orb_a[disk_bh_retro_orbs_a_new > disk_radius_outer]

    assert np.all(disk_bh_retro_orbs_a_new < disk_radius_outer), \
        "disk_bh_retro_orbs_a_new has values greater than disk_radius_outer"
    assert np.all(disk_bh_retro_orbs_a_new >= 0), \
        "disk_bh_retro_orbs_a_new has values < 0"

    return disk_bh_retro_orbs_ecc_new, disk_bh_retro_orbs_a_new, disk_bh_retro_orbs_inc_new


def tau_inc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses, disk_bh_retro_arg_periapse,
                disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc, disk_surf_density_func):
    """Computes inclination damping timescale from actual variables; used only for scaling.


    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_retro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of retrograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_retro_masses : numpy.ndarray
        Mass [M_sun] of retrograde singleton BH at start of timestep_duration_yr with :obj:`float` type
    disk_bh_retro_arg_periapse : numpy.ndarray
        Argument of periapse [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_inc : numpy.ndarray
        Orbital inclination [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH

    Returns
    -------
    tau_i_dyn : numpy.ndarray
        Inclination damping timescale [s]
    """
    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    SI_smbh_mass = smbh_mass * u.Msun.to("kg")  # kg
    SI_semi_maj_axis = si_from_r_g(smbh_mass, disk_bh_retro_orbs_a).to("m").value
    SI_orbiter_mass = disk_bh_retro_masses * u.Msun.to("kg")  # kg
    omega = disk_bh_retro_arg_periapse  # radians
    ecc = disk_bh_retro_orbs_ecc  # unitless
    inc = disk_bh_retro_orbs_inc  # radians
    cos_omega = np.cos(omega)

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt((SI_semi_maj_axis ** 3) / (const.G * SI_smbh_mass))
    # semi-latus rectum in units of meters
    semi_lat_rec = SI_semi_maj_axis * (1.0 - (ecc ** 2))
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + (ecc ** 2) + 2.0 * ecc * cos_omega)
    sigma_minus = np.sqrt(1.0 + (ecc ** 2) - 2.0 * ecc * cos_omega)
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc * cos_omega)
    eta_minus = np.sqrt(1.0 - ecc * cos_omega)
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0 / (eta_plus ** 15)) + np.sqrt(1.0 / (eta_minus ** 15)))
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus / (eta_plus ** 2) + sigma_minus / (eta_minus ** 2))
    # WZL Eqn 71
    #   NOTE: preserved disk_bh_retro_orbs_a in r_g to feed to disk_surf_density_func function
    #   tau in units of sec
    tau_i_dyn = np.sqrt(2.0) * inc * ((delta - np.cos(inc)) ** 1.5) \
                * (SI_smbh_mass ** 2) * period / (
                            SI_orbiter_mass * disk_surf_density_func(disk_bh_retro_orbs_a) * np.pi * (semi_lat_rec ** 2)) \
                / kappa

    assert np.isfinite(tau_i_dyn).all(), \
        "Finite check failure: tau_i_dyn"

    return tau_i_dyn.value


def tau_semi_lat(smbh_mass, retrograde_bh_locations, retrograde_bh_masses, retrograde_bh_orb_ecc, retrograde_bh_orb_inc,
                 retro_arg_periapse, disk_surf_model):
    """Calculates how fast the semi-latus rectum of a retrograde single orbiter changes due to dynamical friction

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    retrograde_bh_locations : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of retrograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    retrograde_bh_masses : numpy.ndarray
        Mass [M_sun] of retrograde singleton BH at start of a timestep with :obj:`float` type
    retrograde_bh_orb_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
    retrograde_bh_orb_inc : numpy.ndarray
        Orbital inclination [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    retro_arg_periapse : numpy.ndarray
        Argument of periapse [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_surf_model : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH

    Returns
    -------
    tau_p_dyn : numpy.ndarray
        Timescale [s] for the evolution of the semi-latus rectum of each object

    Notes
    -----
    Uses Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL). It returns the timescale for the retrograde
    orbiters to change their semi-latus rectum (eqn 70). Note we have assumed the masses of
    the orbiters are negligible compared to the SMBH (<1% should be fine).

    Funny story: if inc = pi exactly, the semi-latus rectum decay is stupid fast
    due to the sin(inc) in tau_p_dyn. However, if you're just a bit
    away from inc = pi (say, pi - 1e-6--but haven't done thorough param search)
    you get something like sensible answers.
    So we gotta watch out for this

    Appropriate for BH, NS, maaaybe WD?--check
    """
    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    smbh_mass = smbh_mass * u.Msun.to("kg")  # kg
    semi_maj_axis = si_from_r_g(smbh_mass, retrograde_bh_locations).to("m").value
    retro_mass = retrograde_bh_masses * u.Msun.to("kg")  # kg
    omega = retro_arg_periapse  # radians
    ecc = retrograde_bh_orb_ecc  # unitless
    inc = retrograde_bh_orb_inc  # radians
    cos_omega = np.cos(omega)

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt((semi_maj_axis ** 3) / (const.G * smbh_mass))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0 - (ecc ** 2))
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + (ecc ** 2) + 2.0 * ecc * cos_omega)
    sigma_minus = np.sqrt(1.0 + (ecc ** 2) - 2.0 * ecc * cos_omega)
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc * cos_omega)
    eta_minus = np.sqrt(1.0 - ecc * cos_omega)
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0 / (eta_plus ** 15)) + np.sqrt(1.0 / (eta_minus ** 15)))
    # WZL Eqn 63
    xi = 0.5 * (np.sqrt(1.0 / (eta_plus ** 13)) + np.sqrt(1.0 / (eta_minus ** 13)))
    # WZL Eqn 64
    zeta = xi / kappa
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus / (eta_plus ** 2) + sigma_minus / (eta_minus ** 2))
    # WZL Eqn 70
    #   NOTE: preserved retrograde_bh_locations in r_g to feed to disk_surf_model function
    #   tau in units of sec
    #   NOTE: had to add an abs(sin(inc)) to avoid negative timescales(!)
    tau_p_dyn = np.abs(np.sin(inc)) * ((delta - np.cos(inc)) ** 1.5) \
                * (smbh_mass ** 2) * period / (
                            retro_mass * disk_surf_model(retrograde_bh_locations) * np.pi * (semi_lat_rec ** 2)) \
                / (np.sqrt(2)) * kappa * np.abs(np.cos(inc) - zeta)

    assert np.isfinite(tau_p_dyn).all(), \
        "Finite check failure: tau_p_dyn"

    return tau_p_dyn


def tau_ecc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses, disk_bh_retro_arg_periapse,
                disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc, disk_surf_density_func):
    """Computes eccentricity & semi-maj axis damping timescale from actual variables

    This does not including migration; used only for scaling.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_retro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of retrograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_retro_masses : float array
        Mass [M_sun] of retrograde singleton BH at start of timestep_duration_yr with :obj:`float` type
    disk_bh_retro_arg_periapse : numpy.ndarray
        Argument of periapse [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_bh_retro_orbs_inc : numpy.ndarray
        Orbital inclination [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH

    Returns
    -------
    tau_e_dyn : numpy.ndarray
        Eccentricity damping timescale [s]
    tau_a_dyn : numpy.ndarray
        Semi-major axis damping timescale [s]
    """
    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    omega = disk_bh_retro_arg_periapse  # radians
    ecc = disk_bh_retro_orbs_ecc  # unitless
    inc = disk_bh_retro_orbs_inc  # radians
    cos_omega = np.cos(omega)

    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + (ecc ** 2) + 2.0 * ecc * cos_omega)
    sigma_minus = np.sqrt(1.0 + (ecc ** 2) - 2.0 * ecc * cos_omega)
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc * cos_omega)
    eta_minus = np.sqrt(1.0 - ecc * cos_omega)
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0 / (eta_plus ** 15)) + np.sqrt(1.0 / (eta_minus ** 15)))
    # WZL Eqn 63
    xi = 0.5 * (np.sqrt(1.0 / (eta_plus ** 13)) + np.sqrt(1.0 / (eta_minus ** 13)))
    # WZL Eqn 64
    zeta = xi / kappa
    # WZL Eqn 65
    kappa_bar = 0.5 * (np.sqrt(1.0 / (eta_plus ** 7)) + np.sqrt(1.0 / (eta_minus ** 7)))
    # WZL Eqn 66
    xi_bar = 0.5 * (np.sqrt((sigma_plus ** 4) / (eta_plus ** 13)) + np.sqrt((sigma_minus ** 4) / (eta_minus ** 13)))
    # WZL Eqn 67
    zeta_bar = xi_bar / kappa_bar

    # call function for tau_p_dyn (WZL Eqn 70)
    tau_p_dyn = tau_semi_lat(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses, disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc, disk_bh_retro_arg_periapse,
                             disk_surf_density_func)
    #  also need to find tau_a_dyn, but
    #   fortunately it's a few factors off of tau_p_dyn (this may be a dumb way to handle it)
    tau_a_dyn = tau_p_dyn * (1.0 - (ecc ** 2)) * kappa * np.abs(np.cos(inc) - zeta) / (
                kappa_bar * np.abs(np.cos(inc) - zeta_bar))
    # WZL Eqn 73
    tau_e_dyn = (2.0 * (ecc ** 2) / (1.0 - (ecc ** 2))) * 1.0 / np.abs(1.0 / tau_a_dyn - 1.0 / tau_p_dyn)

    assert np.isfinite(tau_e_dyn).all(), \
        "Finite check failure: tau_e_dyn"
    assert np.isfinite(tau_a_dyn).all(), \
        "Finite check failure: tau_a_dyn"

    return tau_e_dyn.value, tau_a_dyn.value

def disk_capture_rate(smbh_mass, disk_inner_stable_circ_orb, disk_radius_outer, disk_surf_density_func, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer, nsc_imf_bh_mode, nsc_imf_bh_mass_max, nsc_imf_bh_powerlaw_index, mass_pile_up, nsc_imf_bh_method, disk_bh_num, disk_aspect_ratio_avg):
    """Function to compute the disk capture rate of BH, based on scaling to results of WZL and Fabj et al. 2020
    i.e. 10 captured BH per Myr around a 1e8 SMBH for an SG disk, default BH IMF, default NSC radial distribution,
    thermal eccentricity distribution. Scales for variable BH IMF, number of BH in NSC, radial distribution,
    eccentricity distribution, disk surface density and SMBH mass. BUT: crude average only.

    Parameters
    ----------
    smbh_mass : float
        from inputs
    disk_inner_stable_circ_orb : float
        from inputs
    disk_radius_outer : float
        from inputs
    disk_surf_density_func : function
        from inputs
    nsc_radius_crit : float
        from inputs
    nsc_density_index_inner : float
        from inputs
    nsc_density_index_outer : float
        from inputs
    nsc_imf_bh_mode : float
        from inputs
    nsc_imf_bh_mass_max : float
        from inputs
    nsc_imf_bh_powerlaw_index : float
        from inputs
    mass_pile_up : float
        from inputs
    nsc_imf_bh_method : string
        from inputs
    disk_bh_num : int
        computed--careful on ooo
    disk_aspect_ratio_avg : float
        from inputs, but should be computed (again careful on ooo)

    Returns
    -------
    rate : float
        number of captured BH per Myr of disk lifetime
    """
    rate_0 = 10.0 # fiducial captures per Myr
    smbh_mass_0 = 1.e8 # fiducial SMBH mass for capture rate--see Fabj et al. 2020 and WZL
    num_nsc_obj_0 = 6700.0 # fiducial number of NSC objects with semi-maj axis inside disk R_out
    avg_nsc_obj_mass_0 = 17.1 # fiducial avg mass of NSC orbiter (BH only) out to disk R_out
    avg_semi_lat_nsc_obj_0 = 13994.1 # fiducial avg semi-latus-rectum (p=a(1-e^2)) of NSC orbiter (BH only) to to disk R_out
    avg_disk_surf_dens_0 = 683587.08 # fiducial avg surface density of disk out to disk R_out

    sample_size = 1e4 # how many objects do you need to get a consistent average, hopefully this is ok.
    
    # undo the scaling with disk aspect ratio and count everything (may need to round to nearest int)
    # This will not work when Miranda fixes how we count the number of BH, but then we can just call their func
    num_nsc_obj = disk_bh_num/disk_aspect_ratio_avg

    # careful here: do we always use these methods to get orb_a, orb_ecc and masses? Or do we sometimes use other methods?
    nsc_obj_masses = setup.setup_disk_blackholes_masses(int(sample_size), nsc_imf_bh_mode, nsc_imf_bh_mass_max, nsc_imf_bh_powerlaw_index, mass_pile_up, nsc_imf_bh_method)
    avg_nsc_obj_mass = sum(nsc_obj_masses)/sample_size # draw 1e4 times from setup disk black holes mass func and take average?

    orb_a_dist = setup.setup_disk_blackholes_location_NSC_powerlaw(int(sample_size), disk_radius_outer, disk_inner_stable_circ_orb, smbh_mass, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer, volume_scaling=True)
    orb_ecc_dist = setup.setup_disk_blackholes_eccentricity_thermal(int(sample_size))
    orb_semi_lat_dist = orb_a_dist * (1.0 - orb_ecc_dist**2)
    avg_semi_lat_nsc_obj = sum(orb_semi_lat_dist)/sample_size # draw 1e4 times from setupdisk black holes orb a and ecc and take average?

    #disk_radii = np.arange(disk_inner_stable_circ_orb, disk_radius_outer, 0.01)
    # had to replace isco with 13.0 rg because pAGN can't handle radii < 13.0rg... fix later
    disk_radii = np.arange(13.0, disk_radius_outer, 0.01)
    disk_surf_density_array = disk_surf_density_func(disk_radii)
    avg_disk_surf_dens = sum(disk_surf_density_array)/len(disk_radii) # compute avg disk surf density to disk R_out for this disk

    rate = rate_0 * (num_nsc_obj/num_nsc_obj_0) * (avg_nsc_obj_mass/avg_nsc_obj_mass_0) * (avg_semi_lat_nsc_obj/avg_semi_lat_nsc_obj_0)**2 * (avg_disk_surf_dens/avg_disk_surf_dens_0) * (smbh_mass_0/smbh_mass)**2 

    return rate

def disk_capture_crit_radius(time, R_out, R_in):
    """computes time dependent critical radius for disk capture of BH; inside of this radius, probability of
    capture radius scales as r^-1/4, outside scales as r^-2; critical capture radius grows with time until it
    reaches the outer radius of the disk. Roughly should be SMBH mass invariant and scale up an order of magnitude
    in units of Rg per order of magnitude in time, starting with 100Rg at ~10^5yrs (see WZL fig 15).

    Parameters
    ----------
    time : float
        time in simulation at capture (units?)
    R_out : float
        Outer radius of disk in units of Rg (from inputs)
    R_in : float
        innermost radius of disk or innermost radius where objects can capture (R_tidal?) units of Rg

    Returns
    -------
    cap_crit_radius : float
        break radius between P(capture) propto r^-1/4 and r^-2 behavior units of Rg
    """
    
    # This is literally eyeballing a plot: roughly the critical capture radius seems to be 100rg at 10^5 yrs
    # then 1000rg at 10^6 yrs, and 10^4rg at 10^7 yrs
    idx_crit_rad = np.log10(time/year) - 3.0
    cap_crit_radius = pow(10.0, idx_crit_rad)

    # then make sure the crit radius isn't bigger than the disk or smaller than the disk
    if cap_crit_radius >= R_out:
        cap_crit_radius = R_out
    elif cap_crit_radius <= R_in:
        cap_crit_radius = R_in + 0.01

    return cap_crit_radius

def where_capture(R_in, R_out, R_cap_crit):

    

    return where_capture