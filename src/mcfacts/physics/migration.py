"""
Module for calculating the timescale of migrations.
"""

import numpy as np
import astropy.constants as const
import astropy.units as u
from mcfacts.mcfacts_random_state import rng
import scipy


def paardekooper10_torque(smbh_mass, disk_surf_density_func_log, disc_surf_density, temp_func, orbs_a, orbs_ecc, orb_ecc_crit):
    """Return the Paardekooper (2010) torque coefficient for Type 1 migration
    """
    #Sort the radii of BH and get their log (radii)
    sorted_orbs_a = np.sort(orbs_a)
    log_orbs_a = np.log10(orbs_a)
    sorted_log_orbs_a = np.sort(log_orbs_a)

    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]


    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
    #    return (orbs_a)
        return ()
    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    log_new_orbs_a = np.log10(new_orbs_a)
    #Evaluate disc surf density at locations of all BH
    #disc_surf_d = disc_surf_density(orbs_a)
    disc_surf_d = disc_surf_density(sorted_orbs_a)
    disc_temp=temp_func(sorted_orbs_a)
    #Evaluate disc surf density at only migrating BH
    disc_surf_d_mig = disc_surf_density(new_orbs_a)
    #Get log of disc surf density
    log_disc_surf_d = np.log10(disc_surf_d)
    #Get log of disc midplane temperature
    log_disc_temp = np.log10(disc_temp)

    log_disc_surf_d_mig = np.log10(disc_surf_d_mig)
    sort_log_orbs_a =np.sort(log_new_orbs_a)


    Sigmalog_spline = scipy.interpolate.CubicSpline(sorted_log_orbs_a, log_disc_surf_d, extrapolate=False)
    Templog_spline = scipy.interpolate.CubicSpline(sorted_log_orbs_a, log_disc_temp, extrapolate=False)
    #Find derivates of Sigmalog_spline
    dSigmadR_spline = Sigmalog_spline.derivative()
    dTempdR_spline = Templog_spline.derivative()
    #Evaluate dSigmadR_spline at the migrating orb_a values
    dSigmadR = dSigmadR_spline(log_new_orbs_a)
    dTempdR = dTempdR_spline(log_new_orbs_a)
    #sortdSigmadR = dSigmadR_spline(sort_log_orbs_a)

    #print("dSigmadR", dSigmadR)
    #print("dTempdR", dTempdR)
    #print("sortdSigmadR",sortdSigmadR)
    #print("log_new_orbs_a",log_new_orbs_a)
    #print("sorted_log_new_orbs_a",sort_log_orbs_a)

    Torque_paardekooper_coeff = -0.85 + dSigmadR +0.9*dTempdR

    return Torque_paardekooper_coeff

def normalized_torque(smbh_mass,orbs_a,masses, orbs_ecc, orb_ecc_crit,disk_surf_density_func,disk_aspect_ratio_func):
    """Calculates the normalized torque from e.g. Grishin et al. '24
    Gamma_0 = (q/h)^2 * Sigma* a^4 * Omega^2
        where q= mass_of_bh/smbh_mass, h= disk aspect ratio at location of bh (a_bh),
        Sigma= disk surface density at a_bh, a=a_bh, Omega = bh orbital frequency at a_bh.
        Units are kg m^-2 * m^4 *s^-2 = kg (m s^-1)^2 = Nm (= J)
    Args:
        smbh_mass : float
        Mass [M_sun] of the SMBH
    orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] wrt to SMBH of objects at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    masses : numpy.ndarray
        Masses [M_sun] of objects at start of timestep with :obj:`float` type
    orbs_ecc : numpy.ndarray
        Orbital ecc [unitless] wrt to SMBH of objects at start of timestep :math:`\\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated

    """
    #M_sun =2.e30kg
    m_sun = 2.0*10**(30)
    smbh_mass_in_kg = smbh_mass * m_sun
    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return ()

    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    # Get surface density function or process if just a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(orbs_a)[migration_indices]
    # Get aspect ratio function or process if just a float
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(orbs_a)[migration_indices]
    # find mass ratios
    mass_ratios = (masses[migration_indices]/smbh_mass)
    #Convert orb_a of migrating BH to meters. r_g =GM_smbh/c^2.
    # Usefully, 1_rg=GM_smbh/c^2= 6.7e-11*2.e38/(9e16)~1.5e11m=1AU
    orb_a_in_meters = new_orbs_a*smbh_mass_in_kg*scipy.constants.G / (scipy.constants.c)**(2.0)
    #Omega of migrating BH
    Omega_bh = np.sqrt(scipy.constants.G * smbh_mass_in_kg/((orb_a_in_meters)**(3.0)))
    #Normalized torque = (q/h)^2 * Sigma * a^4 * Omega^2
    normalized_torque = ((mass_ratios/disk_aspect_ratio)**(2.0))*disk_surface_density*((orb_a_in_meters)**(4.0))*(Omega_bh**(2.0))
    return normalized_torque

def torque_mig_timescale(smbh_mass,orbs_a,masses, orbs_ecc, orb_ecc_crit,migration_torque):
    """Calculates the migration timescale using an input migration torque
    t_mig = a/-(dot(a)) where dot(a)=-2aGamma_tot/L so
    t_mig = L/2Gamma_tot
    with Gamma_tot=migration torque, L = orb ang mom = m (GMa)^1/2=m Omega a^2 and so
    t_mig = mOmega a^2/2Gamma_tot in units of s.
    Gamma_0 = (q/h)^2 * Sigma* a^4 * Omega^2
        where q= mass_of_bh/smbh_mass, h= disk aspect ratio at location of bh (a_bh),
        Sigma= disk surface density at a_bh, a=a_bh, Omega = bh orbital frequency at a_bh.
        Units are kg m^-2 * m^4 *s^-2 = kg (m s^-1)^2 = Nm (= J)
    Args:
        smbh_mass : float
        Mass [M_sun] of the SMBH
    orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] wrt to SMBH of objects at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    masses : numpy.ndarray
        Masses [M_sun] of objects at start of timestep with :obj:`float` type
    orbs_ecc : numpy.ndarray
        Orbital ecc [unitless] wrt to SMBH of objects at start of timestep :math:`\\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : numpy.ndarray
        Migration torque array. E.g. calculated from torque_paardekooper (units = Nm=J)


    """
    #M_sun =2.e30kg
    m_sun = 2.0*10**(30)
    smbh_mass_in_kg = smbh_mass * m_sun
    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return ()

    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    # Get surface density function or process if just a float
    #if isinstance(disk_surf_density_func, float):
    #    disk_surface_density = disk_surf_density_func
    #else:
    #    disk_surface_density = disk_surf_density_func(orbs_a)[migration_indices]
    # Get aspect ratio function or process if just a float
    #if isinstance(disk_aspect_ratio_func, float):
    #    disk_aspect_ratio = disk_aspect_ratio_func
    #else:
    #    disk_aspect_ratio = disk_aspect_ratio_func(orbs_a)[migration_indices]
    # find mass ratios
    #mass_ratios = (masses[migration_indices]/smbh_mass)
    #Convert orb_a of migrating BH to meters. r_g =GM_smbh/c^2.
    # Usefully, 1_rg=GM_smbh/c^2= 6.7e-11*2.e38/(9e16)~1.5e11m=1AU
    orb_a_in_meters = new_orbs_a*smbh_mass_in_kg*scipy.constants.G / (scipy.constants.c)**(2.0)
    #Omega of migrating BH in s^-1
    Omega_bh = np.sqrt(scipy.constants.G * smbh_mass_in_kg/((orb_a_in_meters)**(3.0)))
    #masses of BH in kg
    bh_masses = m_sun*masses[migration_indices]
    #Normalized torque = (q/h)^2 * Sigma * a^4 * Omega^2
    torque_mig_timescale = bh_masses*Omega_bh*((orb_a_in_meters)**(2.0))/(2.0*migration_torque)
    #print("torque_mig_timescale",torque_mig_timescale)
    return torque_mig_timescale

def jiminezmasset17_torque(smbh_mass, disc_surf_density, disk_opacity_func, temp_func, orbs_a, orbs_ecc, orb_ecc_crit):
    """Return the Jiminez & Masset (2017) torque coefficient for Type 1 migration
    """
    #Sort the radii of BH and get their log (radii)
    sorted_orbs_a = np.sort(orbs_a)
    log_orbs_a = np.log10(orbs_a)
    sorted_log_orbs_a = np.sort(log_orbs_a)

    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]


    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
    #    return (orbs_a)
        return ()
    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    log_new_orbs_a = np.log10(new_orbs_a)
    #Evaluate disc surf density at locations of all BH
    #disc_surf_d = disc_surf_density(orbs_a)
    disc_surf_d = disc_surf_density(sorted_orbs_a)
    disc_temp=temp_func(sorted_orbs_a)
    #Evaluate disc surf density at only migrating BH
    disc_surf_d_mig = disc_surf_density(new_orbs_a)
    #Get log of disc surf density
    log_disc_surf_d = np.log10(disc_surf_d)
    #Get log of disc midplane temperature
    log_disc_temp = np.log10(disc_temp)

    log_disc_surf_d_mig = np.log10(disc_surf_d_mig)
    sort_log_orbs_a =np.sort(log_new_orbs_a)


    Sigmalog_spline = scipy.interpolate.CubicSpline(sorted_log_orbs_a, log_disc_surf_d, extrapolate=False)
    Templog_spline = scipy.interpolate.CubicSpline(sorted_log_orbs_a, log_disc_temp, extrapolate=False)
    #Find derivates of Sigmalog_spline
    dSigmadR_spline = Sigmalog_spline.derivative()
    dTempdR_spline = Templog_spline.derivative()
    #Evaluate dSigmadR_spline at the migrating orb_a values
    dSigmadR = dSigmadR_spline(log_new_orbs_a)
    dTempdR = dTempdR_spline(log_new_orbs_a)
    #sortdSigmadR = dSigmadR_spline(sort_log_orbs_a)

    #print("dSigmadR", dSigmadR)
    #print("dTempdR", dTempdR)
    #print("sortdSigmadR",sortdSigmadR)
    #print("log_new_orbs_a",log_new_orbs_a)
    #print("sorted_log_new_orbs_a",sort_log_orbs_a)

    Torque_jiminezmasset_coeff = -0.85 + dSigmadR +0.9*dTempdR

    return Torque_jiminezmasset_coeff


def type1_migration(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                    disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                    disk_radius_trap, disk_radius_outer, timestep_duration_yr):
    """Calculates how far an object migrates in an AGN gas disk in a single timestep

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] wrt to SMBH of objects at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    masses : numpy.ndarray
        Masses [M_sun] of objects at start of timestep with :obj:`float` type
    orbs_ecc : numpy.ndarray
        Orbital ecc [unitless] wrt to SMBH of objects at start of timestep :math:`\\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    orbs_a : float array
        Semi-major axes [r_{g,SMBH}] of objects at end of timestep
    """

    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return (orbs_a)

    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    # Get surface density function or process if just a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(orbs_a)[migration_indices]
    # Get aspect ratio function or process if just a float
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(orbs_a)[migration_indices]

    disk_feedback_ratio = disk_feedback_ratio_func[migration_indices]

    # Compute migration timescale for each orbiter in seconds
    # Eqn from Paardekooper 2014, rewritten for R in terms of r_g of SMBH = GM_SMBH/c^2
    # tau = (pi/2) h^2/(q_d*q) * (1/Omega)
    #   where h is aspect ratio, q is m/M_SMBH, q_d = pi R^2 disk_surface_density/M_SMBH
    #   and Omega is the Keplerian orbital frequency around the SMBH
    # Here smbh_mass/disk_bh_mass_pro are both in M_sun, so units cancel
    # c, G and disk_surface_density in SI units
    tau = ((disk_aspect_ratio ** 2.0) * const.c.value / (3.0 * const.G.value) * (smbh_mass/masses[migration_indices]) / disk_surface_density) / np.sqrt(new_orbs_a)
    #print("tau",tau)
    # ratio of timestep to tau_mig (timestep in years so convert)
    year = (1 * u.yr).to(u.s).value # o.g. vers: (365 * u.day).to(u.s).value; julian yr: (1 * u.yr).to(u.s).value
    dt = timestep_duration_yr * (1 * u.yr).to(u.s).value / tau
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = new_orbs_a.copy() * dt

    # Calculate epsilon --amount to adjust from disk_radius_trap for objects that will be set to disk_radius_trap
    epsilon_trap_radius = disk_radius_trap * ((masses[migration_indices] / (3 * (masses[migration_indices] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=migration_indices.size)

    # Get masks for if objects are inside or outside the trap radius
    mask_out_trap = new_orbs_a > disk_radius_trap
    mask_in_trap = new_orbs_a < disk_radius_trap

    # Get mask for objects where feedback_ratio <1; these still migrate inwards, but more slowly
    mask_mig_in = disk_feedback_ratio < 1
    if (np.sum(mask_mig_in) > 0):
        # If outside trap migrate inwards
        temp_orbs_a = new_orbs_a[mask_mig_in & mask_out_trap] - migration_distance[mask_mig_in & mask_out_trap] * (1 - disk_feedback_ratio[mask_mig_in & mask_out_trap])
        # If migration takes object inside trap, fix at trap
        temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap - epsilon_trap_radius[mask_mig_in & mask_out_trap][temp_orbs_a <= disk_radius_trap]
        new_orbs_a[mask_mig_in & mask_out_trap] = temp_orbs_a

        # If inside trap, migrate outwards
        temp_orbs_a = new_orbs_a[mask_mig_in & mask_in_trap] + migration_distance[mask_mig_in & mask_in_trap] * (1 - disk_feedback_ratio[mask_mig_in & mask_in_trap])
        # If migration takes object outside trap, fix at trap
        temp_orbs_a[temp_orbs_a >= disk_radius_trap] = disk_radius_trap + epsilon_trap_radius[mask_mig_in & mask_in_trap][temp_orbs_a >= disk_radius_trap]
        new_orbs_a[mask_mig_in & mask_in_trap] = temp_orbs_a

    # Get mask for objects where feedback_ratio > 1: these migrate outwards
    mask_mig_out = disk_feedback_ratio > 1
    if (np.sum(mask_mig_out) > 0):
        new_orbs_a[mask_mig_out] = new_orbs_a[mask_mig_out] + migration_distance[mask_mig_out] * (disk_feedback_ratio[mask_mig_out] - 1)

    # Get mask for objects where feedback_ratio == 1. Shouldn't happen if feedback = 1 (on), but will happen if feedback = 0 (off)
    mask_mig_stay = disk_feedback_ratio == 1
    if (np.sum(mask_mig_stay) > 0):
        # If outside trap migrate inwards
        temp_orbs_a = new_orbs_a[mask_mig_stay & mask_out_trap] - migration_distance[mask_mig_stay & mask_out_trap]
        # If migration takes object inside trap, fix at trap
        temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap - epsilon_trap_radius[mask_mig_stay & mask_out_trap][temp_orbs_a <= disk_radius_trap]
        new_orbs_a[mask_mig_stay & mask_out_trap] = temp_orbs_a

        # If inside trap migrate outwards
        temp_orbs_a = new_orbs_a[mask_mig_stay & mask_in_trap] + migration_distance[mask_mig_stay & mask_in_trap]
        # If migration takes object outside trap, fix at trap
        temp_orbs_a[temp_orbs_a >= disk_radius_trap] = disk_radius_trap + epsilon_trap_radius[mask_mig_stay & mask_in_trap][temp_orbs_a >= disk_radius_trap]
        new_orbs_a[mask_mig_stay & mask_in_trap] = temp_orbs_a

    # Assert that things cannot migrate out of the disk
    epsilon = disk_radius_outer * ((masses[migration_indices][new_orbs_a > disk_radius_outer] / (3 * (masses[migration_indices][new_orbs_a > disk_radius_outer] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=np.sum(new_orbs_a > disk_radius_outer))
    new_orbs_a[new_orbs_a > disk_radius_outer] = disk_radius_outer - epsilon

    # Update orbs_a
    orbs_a[migration_indices] = new_orbs_a
    return (orbs_a)


def type1_migration_grishin(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                    disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                    disk_radius_trap, disk_radius_outer, timestep_duration_yr):
    """Calculates how far an object migrates in an AGN gas disk in a single timestep using Grishin+24

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.



    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] wrt to SMBH of objects at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    masses : numpy.ndarray
        Masses [M_sun] of objects at start of timestep with :obj:`float` type
    orbs_ecc : numpy.ndarray
        Orbital ecc [unitless] wrt to SMBH of objects at start of timestep :math:`\\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    orbs_a : float array
        Semi-major axes [r_{g,SMBH}] of objects at end of timestep
    """
    #If SMBH >7.e7M_sun, define dummy trap,anti-trap radii
    #if smbh_mass > 7.e7:
    #    trap_radius = 1.0
    #    anti_trap_radius = 0.0
    #If SMBH <7.e7Msun, scale trap, anti-trap radii to regular value (Grishin find Bellovary trap in limit)
    #if smbh_mass < 7.e7:
    #    trap_radius = disk_radius_trap *(smbh_mass/7.e7)^{-1.225}
    #    anti_trap_radius = disk_radius_trap *(smbh_mass/7.e7)^{0.1}


    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return (orbs_a)

    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    # Get surface density function or process if just a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(orbs_a)[migration_indices]
    # Get aspect ratio function or process if just a float
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(orbs_a)[migration_indices]

    disk_feedback_ratio = disk_feedback_ratio_func[migration_indices]

    # Compute migration timescale for each orbiter in seconds
    # Use Grishin+24. Also used in Darmgardt+24 (pAGN).
    # Need modification of Type 1 migration and thermal feedback combined.
    # At ~10^8Msun (at alpha~0.01 or higher), no migration tramp, migration always inwards
    #   but swamp at smaller radii (<10^3R_g), see e.g. Fig. 7 in Grishin+24
    #   Type 1 torque is smaller in Grishin, but migration timescale is pretty quick.
    # Two key radii if M_smbh <6x10^7Msun
    #   R_trap~ 10^3r_{g} (M_smbh/6.7 x 10^7Msun)^-1.225 ~1000AU
    #   Anti-trap ~ 10^3r_{g} (M_smbh/6.7x10^7Msun)^0.1 ~ 1000AU
    # So, e.g. for M_smbh=10^7Msun, R_trap~ 10^4r_g (~1000AU), R_anti_trap ~827r_g (~83AU). (Compare with Fig.7 of Grishin+24)
    #        & for M_smbh=10^6Msun, R_trap~ 1.7x10^5R_g (~1700AU), R_anti_trap~657r_g (~7AU).
    # So, order of operation:
    # 1.given M_smbh figure out where in the disk are R_trap, R_anti-trap.
    #   If M_smbh > 6.7e10^7Msun, no trap/anti-trap (assuming alpha>=0.01) R_trap=1, R_anti_trap=2 (<R_isco)
    #   If M_smbh < 6.7e10^7Msun, R_trap=10^3r_g (M_smbh/6.7x10^7Msun)^-1.225
    #                             R_anti_trap = 10^3r_g (M_smbh/6.7x10^7Msun)^0.1
    # 2. Given circularized prograde BH of mass m, semi-major axis a
    #   If a > R_trap then migration inwards on timescale t_grishin
    #   If a < R_trap and a > R_anti_trap then migration *outwards* on timescale ~kyr (Fig. 7). < Fiducial timestep (10kyr)
    #   If a < R_trap and a < R_anti_trap then migration *inwards* on timescale t_grishin
    #
    #  Eqn from Paardekooper 2014, rewritten for R in terms of r_g of SMBH = GM_SMBH/c^2
    # tau = (pi/2) h^2/(q_d*q) * (1/Omega)
    #   where h is aspect ratio, q is m/M_SMBH, q_d = pi R^2 disk_surface_density/M_SMBH
    #   and Omega is the Keplerian orbital frequency around the SMBH
    # Here smbh_mass/disk_bh_mass_pro are both in M_sun, so units cancel
    # c, G and disk_surface_density in SI units
    tau = ((disk_aspect_ratio ** 2.0) * scipy.constants.c / (3.0 * scipy.constants.G) * (smbh_mass/masses[migration_indices]) / disk_surface_density) / np.sqrt(new_orbs_a)
    # ratio of timestep to tau_mig (timestep in years so convert)
    dt = timestep_duration_yr * scipy.constants.year / tau
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = new_orbs_a.copy() * dt

    # Calculate epsilon --amount to adjust from disk_radius_trap for objects that will be set to disk_radius_trap
    epsilon_trap_radius = disk_radius_trap * ((masses[migration_indices] / (3 * (masses[migration_indices] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=migration_indices.size)

    # Get masks for if objects are inside or outside the trap radius
    mask_out_trap = new_orbs_a > disk_radius_trap
    mask_in_trap = new_orbs_a < disk_radius_trap

    # Get mask for objects where feedback_ratio <1; these still migrate inwards, but more slowly
    mask_mig_in = disk_feedback_ratio < 1
    if (np.sum(mask_mig_in) > 0):
        # If outside trap migrate inwards
        temp_orbs_a = new_orbs_a[mask_mig_in & mask_out_trap] - migration_distance[mask_mig_in & mask_out_trap] * (1 - disk_feedback_ratio[mask_mig_in & mask_out_trap])
        # If migration takes object inside trap, fix at trap #BUG
        temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap - epsilon_trap_radius[mask_mig_in & mask_out_trap][temp_orbs_a <= disk_radius_trap]
        new_orbs_a[mask_mig_in & mask_out_trap] = temp_orbs_a

        # If inside trap, migrate outwards
        temp_orbs_a = new_orbs_a[mask_mig_in & mask_in_trap] + migration_distance[mask_mig_in & mask_in_trap] * (1 - disk_feedback_ratio[mask_mig_in & mask_in_trap])
        # If migration takes object outside trap, fix at trap
        temp_orbs_a[temp_orbs_a >= disk_radius_trap] = disk_radius_trap + epsilon_trap_radius[mask_mig_in & mask_in_trap][temp_orbs_a >= disk_radius_trap]
        new_orbs_a[mask_mig_in & mask_in_trap] = temp_orbs_a

    # Get mask for objects where feedback_ratio > 1: these migrate outwards
    mask_mig_out = disk_feedback_ratio > 1
    if (np.sum(mask_mig_out) > 0):
        new_orbs_a[mask_mig_out] = new_orbs_a[mask_mig_out] + migration_distance[mask_mig_out] * (disk_feedback_ratio[mask_mig_out] - 1)

    # Get mask for objects where feedback_ratio == 1. Shouldn't happen if feedback = 1 (on), but will happen if feedback = 0 (off)
    mask_mig_stay = disk_feedback_ratio == 1
    if (np.sum(mask_mig_stay) > 0):
        # If outside trap migrate inwards
        temp_orbs_a = new_orbs_a[mask_mig_stay & mask_out_trap] - migration_distance[mask_mig_stay & mask_out_trap]
        # If migration takes object inside trap, fix at trap
        temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap - epsilon_trap_radius[mask_mig_stay & mask_out_trap][temp_orbs_a <= disk_radius_trap]
        new_orbs_a[mask_mig_stay & mask_out_trap] = temp_orbs_a

        # If inside trap migrate outwards
        temp_orbs_a = new_orbs_a[mask_mig_stay & mask_in_trap] + migration_distance[mask_mig_stay & mask_in_trap]
        # If migration takes object outside trap, fix at trap
        temp_orbs_a[temp_orbs_a >= disk_radius_trap] = disk_radius_trap + epsilon_trap_radius[mask_mig_stay & mask_in_trap][temp_orbs_a >= disk_radius_trap]
        new_orbs_a[mask_mig_stay & mask_in_trap] = temp_orbs_a

    # Assert that things cannot migrate out of the disk
    epsilon = disk_radius_outer * ((masses[migration_indices][new_orbs_a > disk_radius_outer] / (3 * (masses[migration_indices][new_orbs_a > disk_radius_outer] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=np.sum(new_orbs_a > disk_radius_outer))
    new_orbs_a[new_orbs_a > disk_radius_outer] = disk_radius_outer - epsilon

    # Update orbs_a
    orbs_a[migration_indices] = new_orbs_a
    return (orbs_a)

def type1_migration_single(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                           disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                           disk_radius_trap, disk_radius_outer, timestep_duration_yr):
    """Wrapper function for type1_migration for single objects in the disk.

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] wrt to SMBH of objects at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    masses : numpy.ndarray
        Masses [M_sun] of objects at start of timestep with :obj:`float` type
    orbs_ecc : numpy.ndarray
        Orbital ecc [unitless] wrt to SMBH of objects at start of timestep :math:`\\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    new_orbs_a : float array
        Semi-major axes [r_{g,SMBH}] of objects at end of timestep
    """

    new_orbs_a = type1_migration(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                                 disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                                 disk_radius_trap, disk_radius_outer, timestep_duration_yr)

    return (new_orbs_a)


def type1_migration_binary(smbh_mass, blackholes_binary, orb_ecc_crit,
                           disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                           disk_radius_trap, disk_radius_outer, timestep_duration_yr):
    """Wrapper function for type1_migration for binaries in the disk.

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters, including mass_1, mass_2, bin_orb_a, and bin_orb_ecc
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    orbs_a : float array
        Semi-major axes [r_{g,SMBH}] of objects at end of timestep
    """

    orbs_a = blackholes_binary.bin_orb_a
    masses = blackholes_binary.mass_1 + blackholes_binary.mass_2
    orbs_ecc = blackholes_binary.bin_orb_ecc

    blackholes_binary.bin_orb_a = type1_migration(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                                                  disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                                                  disk_radius_trap, disk_radius_outer, timestep_duration_yr)

    return (blackholes_binary)