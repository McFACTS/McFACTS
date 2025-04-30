"""
Module for calculating the timescale of migrations.
"""

import numpy as np
import astropy.constants as const
import astropy.units as u
import scipy.interpolate
from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.point_masses import si_from_r_g
import scipy


def paardekooper10_torque(orbs_a, orbs_ecc, orb_ecc_crit, disk_dlog10surfdens_dlog10R_func, disk_dlog10temp_dlog10R_func):
    """Return the Paardekooper (2010) torque coefficient for Type 1 migration
        Paardekooper_Coeff = [-0.85+0.9dTdR +dSigmadR]
    """
    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return np.array([])

    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    log_new_orbs_a = np.log10(new_orbs_a)

    # Evaluate dSigmadR_spline at the migrating orb_a values
    dlogSigmadlogR = disk_dlog10surfdens_dlog10R_func(log_new_orbs_a)
    dlogTempdlogR = disk_dlog10temp_dlog10R_func(log_new_orbs_a)

    Torque_paardekooper_coeff = -0.85 + dlogSigmadlogR + (0.9 * dlogTempdlogR)

    # Check for nans
    nan_mask = np.isnan(Torque_paardekooper_coeff)
    if any(nan_mask):
        if all(orbs_a[migration_indices][nan_mask] < 12.1):
            # They are not migrating if they have already been captured
            Torque_paardekooper_coeff[nan_mask] = 0.
        else:
            print(f"dlogSigmadlogR[nan_mask]: {dlogSigmadlogR[nan_mask]}")
            print(f"dlogTempdlogR[nan_mask]: {dlogTempdlogR[nan_mask]}")
            print(f"orbs_a[nan_mask]: {orbs_a[migration_indices][nan_mask]}")
            raise ValueError("nans in Torque_paardekooper_coeff")

    assert np.isfinite(Torque_paardekooper_coeff).all(), \
        "Finite check failure: Torque_paardekooper_coeff"

    return Torque_paardekooper_coeff


def normalized_torque(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit, disk_surf_density_func, disk_aspect_ratio_func):
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
    smbh_mass_in_kg = smbh_mass * u.Msun.to("kg")
    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return np.array([])

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
    # Convert orb_a of migrating BH to meters. r_g =GM_smbh/c^2.
    orb_a_in_meters = si_from_r_g(smbh_mass, new_orbs_a).to("m").value
    # Omega of migrating BH
    Omega_bh = np.sqrt(scipy.constants.G * smbh_mass_in_kg/((orb_a_in_meters)**(3.0)))
    # Normalized torque = (q/h)^2 * Sigma * a^4 * Omega^2
    normalized_torque = ((mass_ratios/disk_aspect_ratio)**(2.0))*disk_surface_density*((orb_a_in_meters)**(4.0))*(Omega_bh**(2.0))
    # Check for nans
    nan_mask = np.isnan(normalized_torque)
    if any(nan_mask):
        if all(orbs_a[migration_indices][nan_mask] < 12.1):
            # They are not migrating if they have already been captured
            normalized_torque[nan_mask] = 0.
        else:
            print(orbs_a[migration_indices][nan_mask])
            raise ValueError("nans in normalized_torque")

    assert np.isfinite(normalized_torque).all(), \
        "Finite check failure: normalized_torque"

    return normalized_torque


def torque_mig_timescale(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit, migration_torque):
    """Calculates the migration timescale using an input migration torque
    t_mig = a/-(dot(a)) where dot(a)=-2aGamma_tot/L so
    t_mig = L/2Gamma_tot
    with Gamma_tot=migration torque, L = orb ang mom = m (GMa)^1/2=m Omega a^2 and so
    t_mig = m Omega a^2/2Gamma_tot in units of s.
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
    migration_torque : numpy.ndarray
        Migration torque array. E.g. calculated from torque_paardekooper (units = Nm=J)


    """
    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return np.array([])

    smbh_mass_si = smbh_mass * u.Msun
    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    orb_a_si = si_from_r_g(smbh_mass, new_orbs_a).to("m")
    migration_torque_si = migration_torque * u.newton * u.meter
    # Omega of migrating BH in s^-1
    Omega_bh = np.sqrt(const.G * smbh_mass_si/((orb_a_si)**(3.0)))
    bh_masses = u.Msun*masses[migration_indices]
    # Normalized torque = (q/h)^2 * Sigma * a^4 * Omega^2 (in units of seconds)
    torque_mig_timescale = (bh_masses*Omega_bh*((orb_a_si)**(2.0))/(2.0*migration_torque_si)).to("s")

    assert np.isfinite(torque_mig_timescale).all(), \
        "Finite check failure: torque_mig_timescale"
    return torque_mig_timescale.value


def jimenezmasset17_torque(smbh_mass, disk_surf_density_func, disk_opacity_func, disk_aspect_ratio_func, disk_temp_func, orbs_a, orbs_ecc, orb_ecc_crit, disk_dlog10surfdens_dlog10R_func, disk_dlog10temp_dlog10R_func):
    """Return the Jimenez & Masset (2017) torque coefficient for Type 1 migration
        Jimenez-Masset_torque = [0.46 + 0.96dSigmadR -1/8dTdR]/gamma
                                +[-2.34 -0.1dSigmadR +1.5dTdR]*factor
            where    factor = ((x/2)^{1/2} + (1/gamma))/((x/2)^{1/2} + 1)
            and      x=(16/3)*gamma*(gamma-1)*sigma_SB*T^4/kappa*rho^2*H^4*Omega^3
            with gamma is the adiabatic index (Cp/Cv=5/3=1.66 for monatomic gas; 1.4 for diatomic gas)
                 sigma_SB = Stefan Boltzmann constant (5.67*10^-8 J s^-1 m^-2 K^-4)
                 kappa = disk opacity at location a
                 rho = disk density at location a
                 H = disk height at location a
                 Omega = orbital frequency at location a
            Can rewrite x using Sigma = rho*H = rho*a*h
                so rho^2H^4 = (rho*H)^2*H^2 = Sigma^2 (a*h)^2 and so
                x=(16/3)*gamma*(gamma-1)*sigma_SB*T^4/kappa*Sigma^2*a^2*h^2*Omega^3
    """
    # Constants
    # Define adiabatic index (assume monatomic gas)
    gamma = 5./3.
    # Stefan-Boltzmann constant
    sigma_SB = scipy.constants.Stefan_Boltzmann
    smbh_mass_in_kg = smbh_mass * u.Msun.to("kg")

    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return np.array([])
    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(orbs_a)[migration_indices]

    # Convert migrating orbs_a to meters
    orb_a_in_meters = si_from_r_g(smbh_mass, new_orbs_a).to("m").value
    # Omega of migrating BH in s^-1
    Omega_bh = np.sqrt(scipy.constants.G * smbh_mass_in_kg/((orb_a_in_meters)**(3.0)))
    log_new_orbs_a = np.log10(new_orbs_a)

    # Evaluate disk surf density at only migrating BH
    disk_surf_d_mig = disk_surf_density_func(new_orbs_a)

    # Evaluate dSigmadR_spline at the migrating orb_a values
    dlogSigmadlogR = disk_dlog10surfdens_dlog10R_func(log_new_orbs_a)
    # Evaluate dTempdR_spline at the migrating orb_a values
    dlogTempdlogR = disk_dlog10temp_dlog10R_func(log_new_orbs_a)
    # Evaluate temp at the migrating orb_a values
    temp_migrators = disk_temp_func(new_orbs_a)
    # Evaluate opacity at the migrating orb_a values
    opacity_migrators = disk_opacity_func(new_orbs_a)

    xfactor_1 = (16./3.)*gamma*(gamma-1.0)*sigma_SB*(temp_migrators**(4.0))
    xfactor_2 = opacity_migrators * (disk_surf_d_mig**(2.0))*(disk_aspect_ratio**(2.0))*(orb_a_in_meters**(2.0))*(Omega_bh**(3.0))
    xfactor = xfactor_1/xfactor_2
    sqrtfactor = np.sqrt(xfactor/2)
    factor = (sqrtfactor + 1.0/gamma)/(sqrtfactor + 1.0)

    Torque_jimenezmasset_coeff = (0.46 + 0.96 * dlogSigmadlogR - 1.8 * dlogTempdlogR)/gamma + (-2.34 - 0.1*dlogSigmadlogR + 1.5 * dlogTempdlogR) * factor

    assert np.isfinite(Torque_jimenezmasset_coeff).all(), \
        "Finite check failure: Torque_jimenezmasset_coeff"

    return Torque_jimenezmasset_coeff


def jimenezmasset17_thermal_torque_coeff(smbh_mass, disk_surf_density_func, disk_opacity_func, disk_aspect_ratio_func, disk_temp_func, disk_bh_eddington_ratio, orbs_a, orbs_ecc, orb_ecc_crit, bh_masses, flag_thermal_feedback, disk_dlog10pressure_dlog10R_func):
    """Return the Jimenez & Masset (2017) thermal torque coefficient for Type 1 migration
        Jimenez-Masset_thermal_torque_coeff = Torque_hot*(4mu_thermal/(1+4.*mu_thermal))+ Torque_cold*(2mu_thermal/(1+2.*mu_thermal))
            Given   Torque_hot=thermal_factor*(L/L_c)
                    Torque_cold =-thermal_factor
                with  L= 4piGm_bh*c/kappa_e_scattering (assuming f_Edd=1, the Eddington fraction of the luminosity)
                and L_c = 4pi G*m_bh*rho*Xi/gamma = 4pi G*m_bh Sigma x c_s/gamma since Xi=xH^2*Omega=x*c_s*H
                and with thermal_factor = 1.61*(gamma-1/gamma)*(x_c/lambda)
                where x_c = (dP/dr)*H^2/(3*gamma*R), lambda = sqrt(2Xi/3*gamma*Omega) & (x_c << lambda is assumed for approximation)

            and mu_thermal = Xi/c_s*r_Bondi = x*H/r_Bondi where r_Bondi maximum is capped at H (cannot accrete from outside disk!)
            where c_s is local sound speed, r_Bondi = GM/c_s^2 and Xi = x*H^2*Omega =x*c_s*H
            where  x=(16/3)*gamma*(gamma-1)*sigma_SB*T^4/kappa*rho^2*H^4*Omega^3

        with gamma is the adiabatic index (Cp/Cv=5/3=1.66 for monatomic gas; 1.4 for diatomic gas)
             sigma_SB = Stefan Boltzmann constant (5.67*10^-8 J s^-1 m^-2 K^-4)
             kappa = disk opacity at location a
             rho = disk density at location a
             H = disk height at location a
             Omega = orbital frequency at location a
        Note can rewrite x using Sigma = rho*H = rho*a*h
             so rho^2H^4 = (rho*H)^2*H^2 = Sigma^2 (a*h)^2 and so
             x=(16/3)*gamma*(gamma-1)*sigma_SB*T^4/kappa*Sigma^2*a^2*h^2*Omega^3

    """
    # If no feedback then end the function
    if flag_thermal_feedback == 0:
        return np.array([])

    # Constants
    # Define adiabatic index (assume monatomic gas)
    gamma = 5.0/3.0
    # kappa electron scattering
    kappa_e_scattering = 0.7
    # Stefan-Boltzmann constant
    sigma_SB = scipy.constants.Stefan_Boltzmann
    smbh_mass_in_kg = smbh_mass * u.Msun.to("kg")

    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]

    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        return np.array([])
    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(orbs_a)[migration_indices]

    # Convert migrating orbs_a to meters
    # Convert orb_a of migrating BH to meters. r_g =GM_smbh/c^2.
    orb_a_in_meters = si_from_r_g(smbh_mass, new_orbs_a).to("m").value
    # Omega of migrating BH in s^-1
    Omega_bh = np.sqrt(scipy.constants.G * smbh_mass_in_kg/((orb_a_in_meters)**(3.0)))

    # Height of disk in meters  H=orb_a_in_meters*disk_aspect_ratio
    disk_height_in_meters = orb_a_in_meters*disk_aspect_ratio
    # disk_height_in_meters_squared
    disk_height_m_sq = disk_height_in_meters * disk_height_in_meters
    # sound speed at location of migrating BH in m/s  (c_s = H*Omega_bh)
    sound_speed = disk_height_in_meters*Omega_bh

    # BH masses in kg
    bh_masses_in_kg = bh_masses[migration_indices]*u.Msun.to("kg")
    # Bondi radii for migrating BH
    r_bondi = scipy.constants.G*bh_masses_in_kg/(sound_speed**2.0)
    # If r_bondi for a migrating BH is > disk_height, set effective Bondi radius to disk height
    effective_bondi_radius = np.where(r_bondi < disk_height_in_meters, r_bondi, disk_height_in_meters)
    # Luminosity of migrating BH
    lum = disk_bh_eddington_ratio*4.0*np.pi*scipy.constants.G*bh_masses_in_kg*scipy.constants.c/kappa_e_scattering

    log_new_orbs_a = np.log10(new_orbs_a)

    # For migrating BH
    # Evaluate disk surf density at only migrating BH
    disk_surf_d_mig = disk_surf_density_func(new_orbs_a)
    # Evaluate sound speed at only migrating BH
    disk_sound_speed = sound_speed

    # Evaluate dPressuredR_spline at the migrating orb_a values
    dlogPressuredlogR = disk_dlog10pressure_dlog10R_func(log_new_orbs_a)

    # Evaluate temp at the migrating orb_a values
    temp_migrators = disk_temp_func(new_orbs_a)
    # Evaluate opacity at the migrating orb_a values
    opacity_migrators = disk_opacity_func(new_orbs_a)

    xfactor_1 = (16./3.)*gamma*(gamma-1.0)*sigma_SB*(temp_migrators**(4.0))
    xfactor_2 = opacity_migrators*(disk_surf_d_mig**(2.0))*(disk_aspect_ratio**(2.0))*(orb_a_in_meters**(2.0))*(Omega_bh**(3.0))
    xfactor = xfactor_1/xfactor_2

    # Critical Luminosity of migrating BH
    lum_crit = 4.0*np.pi*scipy.constants.G*bh_masses_in_kg*disk_surf_d_mig*xfactor*disk_sound_speed/gamma

    # mu_thermal = Xi/c_s*r_Bondi and Xi=x*H^2*Omega so mu_thermal = x*H^2*Omega/c_s*r_Bondi = x*H/r_B (where r_B<=H)
    mu_thermal = xfactor*disk_height_in_meters/effective_bondi_radius
    # Saturate the torque
    mu_thermal = np.where(mu_thermal < 1.0, mu_thermal, 1.0)

    # lambda (length) = sqrt(2 Xi/3gamma Omega) =sqrt(2/3 x H^2) since Xi=x*H^2*Omega
    length = np.sqrt(2.0*xfactor*disk_height_m_sq/3.0)

    # x_crit = (dP/dr)*H^2/(3 gamma R)
    x_crit = dlogPressuredlogR*disk_height_m_sq/(3.0*gamma*orb_a_in_meters)
    # Thermal torque calculation
    thermal_factor = 1.61*((gamma-1)/gamma)*(x_crit/length)
    Torque_hot = thermal_factor*lum/lum_crit
    Torque_cold = -thermal_factor

    Thermal_torque_coeff = Torque_hot * (4.0*mu_thermal/(1.0+4.0*mu_thermal)) + Torque_cold*(2.0*mu_thermal/(1.0+2.0*mu_thermal))

    # decay factor of (1- exp(-length*tau/H) where tau is optical depth) and tau=kappa*Sigma/2
    optical_depth = disk_surf_d_mig*opacity_migrators/2.0
    exp_factor = length * optical_depth/disk_height_in_meters
    decay_factor = (1 - np.exp(-exp_factor))

    Thermal_torque_coeff = Thermal_torque_coeff * decay_factor

    assert np.isfinite(Thermal_torque_coeff).all(), \
        "Finite check failure: Thermal_torque_coeff"

    return Thermal_torque_coeff


def type1_migration_distance(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit, torque_mig_timescale, disk_feedback_ratio,
                             disk_radius_trap, disk_radius_anti_trap, disk_radius_outer, timestep_duration_yr, flag_phenom_turb, phenom_turb_centroid, phenom_turb_std_dev, bh_min_mass, torque_prescription):
    """Calculates how far an object migrates in an AGN gas disk in a single timestep given a torque migration timescale
    calculated elsewhere (e.g. torque_migration_timescale)
    Returns their new locations after migration over one timestep.

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
    torque_mig_timescale: numpy.ndarray
        Array of timescale of torque to migrate onto SMBH (units in seconds)
    disk_feedback_ratio : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_antitrap : float
        Radius [r_{g,SMBH}] of disk anti-trap (divergent trap)
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]
    flag_phenom_turb : int
        Is phenomenological turbulence model on (1) or off (0).
    phenom_turb_centroid : float
        Centroid of Gaussian draw of turbulent modification to migration distance (default is 0: no net drift!)
    phenom_turb_std_dev : float
        Standard deviation of Gaussian draw of turbulent perturbation (default is 0.1)
    bh_min_mass : float
        Minimum mass of BH IMF. Phenom. turbulence is largest for this value. Decreases with bh_mass^2 since normalized torque propto m_bh^2.
    torque_prescription : str
        Torque prescription 'paardekooper' or 'jimenez_masset' ('old' is deprecated)
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

    # Array of migration timescales for each orbiter in seconds as calculated from torques elsewhere
    tau = np.abs(torque_mig_timescale)

    # Normalized masses of migrators (normalized to BH minimum mass)
    normalized_migrating_masses = masses[migration_indices]/bh_min_mass
    normalized_mig_masses_sq = normalized_migrating_masses**2.0

    # ratio of timestep to tau_mig (timestep in years so convert)
    dt = timestep_duration_yr * (1 * u.yr).to(u.s).value / tau
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = new_orbs_a.copy() * dt

    # Calculate epsilon for trap radius --amount to adjust from disk_radius_trap for objects that will be set to disk_radius_trap
    epsilon_trap_radius = disk_radius_trap * ((masses[migration_indices] / (3 * (masses[migration_indices] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=migration_indices.size)

    # Calculate epsilon for outer edge of disk
    epsilon_outer_radius = disk_radius_outer * ((masses[migration_indices] / (3 * (masses[migration_indices] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=migration_indices.size)

    if flag_phenom_turb == 1:
        # Only need to perturb migrators for now
        # size_of_turbulent_array = np.size(migration_indices)
        # Assume migration is always inwards (true for 'old' and for 'jimenez_masset' for M_smbh>10^8Msun)
        # Calc migration distance as modified by turbulence.
        migration_distance = migration_distance*(1.0 + rng.normal(phenom_turb_centroid, phenom_turb_std_dev, size=migration_indices.size))/normalized_mig_masses_sq

    if torque_prescription == 'old' or torque_prescription == 'paardekooper':
        # Assume migration is always inwards (true for 'old' and for 'jimenez_masset' for M_smbh>10^8Msun)
        # Disk feedback ratio
        disk_feedback_ratio = disk_feedback_ratio[migration_indices]

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

    if torque_prescription == 'jimenez_masset':
        # If smbh_mass >10^8Msun --assume migration is always inwards
        if smbh_mass > 1.e8:
            new_orbs_a = new_orbs_a - migration_distance
        # If smbh_mass = 1.e8, assume trap at disk_radius_trap, but Type 1 migration inward everywhere.
        # ie. migrators interior & exterior to trap migrate inwards, but exteriors at trap stay there.
        if smbh_mass == 1.e8:
            # Get masks for if objects are inside or outside the trap radius (fixed to disk_radius_trap)
            mask_out_trap = new_orbs_a > disk_radius_trap
            mask_in_trap = new_orbs_a < (disk_radius_trap - epsilon_trap_radius)

            # If outside trap migrate inwards
            temp_orbs_a = new_orbs_a[mask_out_trap] - migration_distance[mask_out_trap]
            # If migration takes object inside trap, fix at trap
            temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap - epsilon_trap_radius[mask_out_trap][temp_orbs_a <= disk_radius_trap]
            new_orbs_a[mask_out_trap] = temp_orbs_a
            # If inside trap migrate inwards
            temp_orbs_a = new_orbs_a[mask_in_trap] - migration_distance[mask_in_trap]
            new_orbs_a[mask_in_trap] = temp_orbs_a

        if smbh_mass < 1.e8 and smbh_mass > 1.e6:
            # Trap radius changes as a function of mass.
            # Also new(!) anti-trap radius. Region between trap and anti-trap migrates out, all others migrate inwards
            # Calc new trap radius from Grishin+24
            # temp_disk_radius_trap = disk_radius_trap*((smbh_mass/1.e8)**(-1.225))
            # temp_disk_radius_anti_trap = disk_radius_trap*((smbh_mass/1.e8)**(0.1))
            # Get masks for objects outside trap, inside trap and inside anti-trap
            mask_out_trap = new_orbs_a > disk_radius_trap
            mask_in_anti_trap = new_orbs_a < disk_radius_anti_trap
            # mask_in_trap = new_orbs_a > disk_radius_anti_trap and new_orbs_a < disk_radius_trap
            mask_in_trap = (new_orbs_a > disk_radius_anti_trap) & (new_orbs_a < disk_radius_trap)
            # If outside trap migrate inwards
            temp_orbs_a = new_orbs_a[mask_out_trap] + migration_distance[mask_out_trap]
            # If migration takes object inside trap, fix at trap
            temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap - epsilon_trap_radius[mask_out_trap][temp_orbs_a <= disk_radius_trap]
            new_orbs_a[mask_out_trap] = temp_orbs_a

            # If inside trap, but outside anti_trap, migrate outwards
            # Commented out this out-migration line
            # temp_orbs_a = new_orbs_a[mask_in_trap] + migration_distance[mask_in_trap]
            # Anything in out-migrating region ends up on trap in <0.01Myr (ie 1 timestep)

            # If migration takes object outside trap, fix at trap. No use, outside trap to keep at trap.
            new_orbs_a[mask_in_trap] = disk_radius_trap + epsilon_trap_radius[mask_in_trap]
            # If inside anti_trap migrate inwards
            temp_orbs_a = new_orbs_a[mask_in_anti_trap] + migration_distance[mask_in_anti_trap]
            new_orbs_a[mask_in_anti_trap] = temp_orbs_a
        if smbh_mass < 1.e6:
            # Trap radius changes as a function of mass.
            # Also new(!) anti-trap radius. Region between trap and anti-trap migrates out, all others migrate inwards
            # Calc new trap radius from Grishin+24
            disk_radius_trap = disk_radius_trap * (smbh_mass/1.e8)**(-0.97)
            disk_radius_anti_trap = disk_radius_trap * (smbh_mass/1.e8)**(0.099)
            # Get masks for objects outside trap, inside trap and inside anti-trap
            mask_out_trap = new_orbs_a > disk_radius_trap
            mask_in_anti_trap = new_orbs_a < disk_radius_anti_trap
            mask_in_trap = new_orbs_a > disk_radius_anti_trap and new_orbs_a < disk_radius_trap

            # If outside trap migrate inwards
            temp_orbs_a = new_orbs_a[mask_out_trap] - migration_distance[mask_out_trap]
            # If migration takes object inside trap, fix at trap
            temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap - epsilon_trap_radius[mask_out_trap][temp_orbs_a <= disk_radius_trap]
            new_orbs_a[mask_out_trap] = temp_orbs_a

            # If inside trap, but outside anti_trap, migrate outwards
            temp_orbs_a = new_orbs_a[mask_in_trap] + migration_distance[mask_in_trap]
            # If migration takes object outside trap, fix at trap
            temp_orbs_a[temp_orbs_a >= disk_radius_trap] = disk_radius_trap + epsilon_trap_radius[mask_out_trap][temp_orbs_a <= disk_radius_trap]
            new_orbs_a[mask_in_trap] = temp_orbs_a

            # If inside anti_trap migrate inwards
            temp_orbs_a = new_orbs_a[mask_in_anti_trap] - migration_distance[mask_in_anti_trap]
            new_orbs_a[mask_in_anti_trap] = temp_orbs_a

    # Assert that objects cannot migrate out of the disk
    new_orbs_a[new_orbs_a > disk_radius_outer] = disk_radius_outer - epsilon_outer_radius[new_orbs_a > disk_radius_outer]

    # Update orbs_a
    orbs_a[migration_indices] = new_orbs_a

    assert np.isfinite(orbs_a).all(), \
        "Finite check failure: orbs_a"
    assert np.all(new_orbs_a < disk_radius_outer), \
        "new_orbs_a contains values greater than disk_radius_outer"
    assert np.all(new_orbs_a > 0), \
        "new_orbs_a contains values <= 0"

    return (orbs_a)


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
    # If SMBH >7.e7M_sun, define dummy trap,anti-trap radii
    # if smbh_mass > 7.e7:
    #    trap_radius = 1.0
    #    anti_trap_radius = 0.0
    # If SMBH <7.e7Msun, scale trap, anti-trap radii to regular value (Grishin find Bellovary trap in limit)
    # if smbh_mass < 7.e7:
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

    assert np.isfinite(new_orbs_a).all(), \
        "Finite check failure: new_orbs_a"
    assert np.all(new_orbs_a < disk_radius_outer), \
        "new_orbs_a has values greater than disk_radius_outer"

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


def type1_migration_binary(smbh_mass, bin_mass_1, bin_mass_2, bin_orb_a, bin_orb_ecc, orb_ecc_crit,
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

    new_bin_orb_a = type1_migration(smbh_mass, bin_orb_a, bin_mass_1 + bin_mass_2, bin_orb_ecc, orb_ecc_crit,
                                    disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                                    disk_radius_trap, disk_radius_outer, timestep_duration_yr)

    return (new_bin_orb_a)
