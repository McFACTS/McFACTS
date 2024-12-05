"""
Module for calculating the timescale of migrations.
"""

import numpy as np
import scipy
from mcfacts.mcfacts_random_state import rng


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