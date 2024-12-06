"""
Module for calculating the orbital and binary eccentricity damping.
"""

import numpy as np
from mcfacts.mcfacts_random_state import rng


def orbital_ecc_damping(smbh_mass, disk_bh_pro_orbs_a, disk_bh_pro_orbs_masses, disk_surf_density_func,
                        disk_aspect_ratio_func, disk_bh_pro_orbs_ecc, timestep_duration_yr, disk_bh_pro_orb_ecc_crit):
    """Calculates damping of BH orbital eccentricities according to a prescription.

    Using Tanaka & Ward (2004)  t_damp = M^3/2 h^4 / (2^1/2 m Sigma a^1/2 G )
    where M is the central mass, h is the disk aspect ratio (H/a), m is the orbiter mass,
    Sigma is the disk surface density, a is the semi-major axis, G is the universal gravitational constant.

    From McKernan & Ford (2023) eqn 4. we can parameterize t_damp as
    t_damp ~ 0.1Myr (q/10^-7)^-1 (h/0.03)^4 (Sigma/10^5 kg m^-2)^-1 (a/10^4r_g)^-1/2

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_pro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_pro_masses : numpy.ndarray
        Masses [M_sun] of prograde singleton BH at start of timestep with :obj:`float` type
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of singleton prograde BH with :obj:`float` type
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_bh_pro_orb_ecc_crit : float
        Critical orbital eccentricity [unitless] below which orbit is close enough to circularize
    delta_energy_strong : float
        Average energy change [units??] per strong encounter
    nsc_spheroid_normalization : float
        Normalization factor [unitless] determines the departures from sphericity of
        the initial distribution of perturbers (1.0=spherical)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated

    Returns
    -------
    bh_new_orb_ecc : float array
        updated orbital eccentricities damped by AGN gas

    Notes
    -----
    For eccentricity e<2h
    e(t)=e0*exp(-t/t_damp)......(1)

    So
    in 0.1 damping time, e(t_damp)=0.90*e0
    in 1 damping time,  e(t_damp)=0.37*e0
    in 2 damping times, e(t_damp)=0.135*e0
    in 3 damping times, e(t_damp)=0.05*e0

    For now assume e<2h condition. To do: Add condition below (if ecc>2h..)

    For eccentricity e>2h eqn. 9 in McKernan & Ford (2023), based on Horn et al. (2012) the scaling time is now t_ecc.
    :math:`t_{ecc} = (t_{damp}/0.78)*[1 - (0.14*(e/h)^2) + (0.06*(e/h)^3)]` ......(2)
    which in the limit of e>0.1 for most disk models becomes
    :math:`t_{ecc} \\propto (t_{damp}/0.78)*[1 + (0.06*(e/h)^3)]`
    """
    # Check incoming eccentricities for nans
    assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for disk_bh_pro_orbs_ecc"

    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(disk_bh_pro_orbs_a)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(disk_bh_pro_orbs_a)

    # Set up new_disk_bh_pro_orbs_ecc
    new_disk_bh_pro_orbs_ecc = np.zeros_like(disk_bh_pro_orbs_ecc)

    # Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7
    mass_ratio = disk_bh_pro_orbs_masses / smbh_mass

    normalized_mass_ratio = mass_ratio / (10 ** -7)
    normalized_bh_locations = disk_bh_pro_orbs_a / 1.e4
    normalized_disk_surf_density_func = disk_surface_density / 1.e5
    normalized_aspect_ratio = disk_aspect_ratio / 0.03

    # Assume all incoming eccentricities are prograde (for now)
    prograde_disk_bh_pro_orbs_ecc = disk_bh_pro_orbs_ecc

    # Calculate (e/h) ratio for all prograde BH for use in eqn. 2 above
    e_h_ratio = prograde_disk_bh_pro_orbs_ecc / disk_aspect_ratio

    # Modest orb eccentricities: e < 2h (experience simple exponential damping): mask entries > 2*aspect_ratio;
    # only show BH with e<2h
    modest_ecc_prograde_indices = np.asarray(prograde_disk_bh_pro_orbs_ecc <= 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a)).nonzero()[0]

    # Large orb eccentricities: e > 2h (experience more complicated damping)
    large_ecc_prograde_indices = np.asarray(prograde_disk_bh_pro_orbs_ecc > 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a)).nonzero()[0]

    # print('modest ecc indices', modest_ecc_prograde_indices)
    # print('large ecc indices', large_ecc_prograde_indices)
    # Calculate the 1-d array of damping times at all locations since we need t_damp for both modest & large ecc
    # (see eqns above)
    t_damp = 1.e5 * (1.0 / normalized_mass_ratio) * (normalized_aspect_ratio ** 4) * (
            1.0 / normalized_disk_surf_density_func) * (1.0 / np.sqrt(normalized_bh_locations))

    # timescale ratio for modest ecc damping
    modest_timescale_ratio = timestep_duration_yr / t_damp

    # timescale for large ecc damping from eqn. 2 above
    t_ecc = (t_damp / 0.78) * (1 - (0.14 * (e_h_ratio ** 2.0)) + (0.06 * (e_h_ratio ** 3.0)))
    large_timescale_ratio = timestep_duration_yr / t_ecc
    # print("t_damp",timestep_duration_yr/t_damp)
    # print("t_ecc",timestep_duration_yr/t_ecc)

    # print("timescale_ratio",timescale_ratio)
    new_disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] * np.exp(
        -modest_timescale_ratio[modest_ecc_prograde_indices])
    new_disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] * np.exp(
        -large_timescale_ratio[large_ecc_prograde_indices])

    new_disk_bh_pro_orbs_ecc = np.where(new_disk_bh_pro_orbs_ecc < disk_bh_pro_orb_ecc_crit,
                                        disk_bh_pro_orb_ecc_crit, new_disk_bh_pro_orbs_ecc)

    # print("Old ecc, New ecc",disk_bh_pro_orbs_ecc,new_disk_bh_pro_orbs_ecc)
    assert np.isfinite(new_disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for new_disk_bh_pro_orbs_ecc"
    return new_disk_bh_pro_orbs_ecc


def orbital_bin_ecc_damping(smbh_mass, blackholes_binary, disk_surf_density_func, disk_aspect_ratio_func, timestep_duration_yr,
                            disk_bh_pro_orb_ecc_crit):
    """Calculates damping of BBH orbital eccentricities according to a prescription.

    Use same mechanisms as for prograde singleton BH.

    E.g. Tanaka & Ward (2004)  t_damp = M^3/2 h^4 / (2^1/2 m Sigma a^1/2 G )
    where M is the central mass, h is the disk aspect ratio (H/a), m is the orbiter mass,
    Sigma is the disk surface density, a is the semi-major axis, G is the universal gravitational constant.
    From McKernan & Ford (2023) eqn 4. we can parameterize t_damp as
    t_damp ~ 0.1Myr (q/10^-7)^-1 (h/0.03)^4 (Sigma/10^5 kg m^-2)^-1 (a/10^4r_g)^-1/2

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    blackholes_binary : AGNBinaryBlackHole
        binary black hole parameters
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        binaries with updated orbital eccentricities damped by AGN gas

    Notes
    -----
    For eccentricity e<2h
        e(t)=e0*exp(-t/t_damp)......(1)
        So
        in 0.1 damping time, e(t_damp)=0.90*e0
        in 1 damping time,  e(t_damp)=0.37*e0
        in 2 damping times, e(t_damp)=0.135*e0
        in 3 damping times, e(t_damp)=0.05*e0

    For now assume e<2h condition. To do: Add condition below (if ecc>2h..)

    For eccentricity e>2h eqn. 9 in McKernan & Ford (2023), based on Horn et al. (2012) the scaling time is now t_ecc.
        t_ecc = (t_damp/0.78)*[1 - (0.14*(e/h)^2) + (0.06*(e/h)^3)] ......(2)
        which in the limit of e>0.1 for most disk models becomes
        t_ecc ~ (t_damp/0.78)*[1 + (0.06*(e/h)^3)]
    """
    # Check incoming eccentricities for nans
    assert np.isfinite(blackholes_binary.bin_ecc).all(), \
        "Finite check failed for blackholes_binary.bin_ecc"

    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(blackholes_binary.bin_orb_a)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(blackholes_binary.bin_orb_a)

    # Set up new_bin_orb_ecc
    new_bin_orb_ecc = np.zeros_like(blackholes_binary.bin_orb_ecc)

    # Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7
    mass_ratio = (blackholes_binary.mass_1 + blackholes_binary.mass_2) / smbh_mass

    normalized_mass_ratio = mass_ratio / (10 ** -7)
    normalized_bh_locations = blackholes_binary.bin_orb_a / 1.e4
    normalized_disk_surf_density_func = disk_surface_density / 1.e5
    normalized_aspect_ratio = disk_aspect_ratio / 0.03

    # Calculate the damping time for all bins
    t_damp = 1.e5 * (1.0 / normalized_mass_ratio) * (normalized_aspect_ratio ** 4) * (
            1.0 / normalized_disk_surf_density_func) * (1.0 / np.sqrt(normalized_bh_locations))
    modest_timescale_ratio = timestep_duration_yr / t_damp

    # Calculate (e/h) ratio for all prograde BH for use in eqn. 2 above
    e_h_ratio = blackholes_binary.bin_orb_ecc / disk_aspect_ratio

    # Calculate damping time for large orbital eccentricity binaries
    t_ecc = (t_damp / 0.78) * (1 - (0.14 * (e_h_ratio ** 2.0)) + (0.06 * (e_h_ratio ** 3.0)))
    large_timescale_ratio = timestep_duration_yr / t_ecc

    # If bin orb ecc <= disk_bh_pro_orb_ecc_crit, do nothing (no damping needed)
    mask1 = blackholes_binary.bin_orb_ecc <= disk_bh_pro_orb_ecc_crit
    new_bin_orb_ecc[mask1] = blackholes_binary.bin_orb_ecc[mask1]

    # If bin orb ecc > disk_bh_pro_orb_ecc_crit, but <2*h then damp modest orb eccentricity
    mask2 = (blackholes_binary.bin_orb_ecc > disk_bh_pro_orb_ecc_crit) & (blackholes_binary.bin_orb_ecc < (2 * disk_aspect_ratio))
    new_bin_orb_ecc[mask2] = blackholes_binary.bin_orb_ecc[mask2] * np.exp(-modest_timescale_ratio[mask2])

    # If bin orb ecc > 2*h then damp large orb eccentricity
    mask3 = (blackholes_binary.bin_orb_ecc > disk_bh_pro_orb_ecc_crit) & (blackholes_binary.bin_orb_ecc > (2 * disk_aspect_ratio))
    new_bin_orb_ecc[mask3] = blackholes_binary.bin_orb_ecc[mask3] * np.exp(-large_timescale_ratio[mask3])

    new_bin_orb_ecc[new_bin_orb_ecc < disk_bh_pro_orb_ecc_crit] = disk_bh_pro_orb_ecc_crit
    # Check output
    assert np.isfinite(new_bin_orb_ecc).all(), \
        "Finite check failed for new_bin_orb_ecc"

    blackholes_binary.bin_orb_ecc = new_bin_orb_ecc

    return (blackholes_binary)


def bin_ecc_damping(smbh_mass, disk_bh_pro_orbs_a, disk_bh_pro_orbs_masses, disk_surf_density_func,
                    disk_aspect_ratio_func,
                    disk_bh_pro_orbs_ecc, timestep_duration_yr, disk_bh_pro_orb_ecc_crit):
    """"Calculate modified eccentricities for BBH according to Calcino et al. (2023), arXiv:2312.13727

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_pro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_pro_masses : numpy.ndarray
        Masses [M_sun] of prograde singleton BH at start of timestep with :obj:`float` type
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of singleton prograde BH with :obj:`float` type
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    bh_new_orb_ecc : float array
        updated orbital eccentricities damped by AGN gas

    Notes
    -----
    dot{e}_b increases in magnitude as e_b is larger.
    dot{e}_b is always negative for prograde bins (ie e_b is damped for prograde bins)
    dot{e}_b is always positive for retrograde bins (ie e_b is pumped for retrograde bins)

    For retrograde bins:
        dot{e_b} ~ 1 in units of dot{M_bondi}/M_bondi for e_b~ 0.5, dot_{e_b}>1 for e_b >0.5, <1 for e_b<0.5
    For prograde bins:
        dot{e_b} ~ -0.25 in the same units for e_b~0.5, dot{e_b} ~ -0.28 at e_b=0.7, dot{e_b}~ -0.04 at e_b=0.1
    and
        dot{m_bin} is ~0.05 M_bondi in these sims.
    with (from their eqn. 19)
        M_bondi/M_edd ~ 5e5 (R_H/H)^3  (rho/10^-14 g/cc) (M_smbh/10^6M_sun)^-1/2 (R0/0.1pc)^3/2 (e/0.1)
        where R_H=Hill sphere radius, H = disk scale height, rho = disk midplane density,
        R0=location of binary, e=acc. efficiency onto SMBH (L=e*dot{M}c^2)
    Convert to 10^8Msun, *1/10

    Use Tanaka & Ward (2004)  t_damp = M^3/2 h^4 / (2^1/2 m Sigma a^1/2 G )
    where M is the central mass, h is the disk aspect ratio (H/a), m is the orbiter mass,
    Sigma is the disk surface density, a is the semi-major axis, G is the universal gravitational constant.

    From McKernan & Ford (2023) eqn 4. we can parameterize t_damp as
        t_damp ~ 0.1Myr (q/10^-7)^-1 (h/0.03)^4 (Sigma/10^5 kg m^-2)^-1 (a/10^4r_g)^-1/2

    For eccentricity e<2h
        e(t)=e0*exp(-t/t_damp)......(1)
        So
            in 0.1 damping time, e(t_damp)=0.90*e0
            in 1 damping time,  e(t_damp)=0.37*e0
            in 2 damping times, e(t_damp)=0.135*e0
            in 3 damping times, e(t_damp)=0.05*e0

    For now assume e<2h condition. To do: Add condition below (if ecc>2h..)

    For eccentricity e>2h eqn. 9 in McKernan & Ford (2023), based on Horn et al. (2012) the scaling time is now t_ecc.
        t_ecc = (t_damp/0.78)*[1 - (0.14*(e/h)^2) + (0.06*(e/h)^3)] ......(2)
        which in the limit of e>0.1 for most disk models becomes
        t_ecc ~ (t_damp/0.78)*[1 + (0.06*(e/h)^3)]
    """
    # Check incoming eccentricities for nans
    assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for disk_bh_pro_orbs_ecc"
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(disk_bh_pro_orbs_a)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(disk_bh_pro_orbs_a)
    # Set up new_disk_bh_pro_orbs_ecc
    new_disk_bh_pro_orbs_ecc = np.zeros_like(disk_bh_pro_orbs_ecc)

    # Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7
    mass_ratio = disk_bh_pro_orbs_masses / smbh_mass

    normalized_mass_ratio = mass_ratio / (10 ** -7)
    normalized_bh_locations = disk_bh_pro_orbs_a / 1.e4
    normalized_disk_surf_density_func = disk_surface_density / 1.e5
    normalized_aspect_ratio = disk_aspect_ratio / 0.03

    # Assume all incoming eccentricities are prograde (for now)
    prograde_disk_bh_pro_orbs_ecc = disk_bh_pro_orbs_ecc

    # Calculate (e/h) ratio for all prograde BH for use in eqn. 2 above
    e_h_ratio = prograde_disk_bh_pro_orbs_ecc / disk_aspect_ratio

    # Modest orb eccentricities: e < 2h (experience simple exponential damping): mask entries > 2*aspect_ratio;
    # only show BH with e<2h
    modest_ecc_prograde_indices = np.asarray(prograde_disk_bh_pro_orbs_ecc <= 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a)).nonzero()[0]

    # Large orb eccentricities: e > 2h (experience more complicated damping)
    large_ecc_prograde_indices = np.asarray(prograde_disk_bh_pro_orbs_ecc > 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a)).nonzero()[0]

    # print('modest ecc indices', modest_ecc_prograde_indices)
    # print('large ecc indices', large_ecc_prograde_indices)
    # Calculate the 1-d array of damping times at all locations since we need t_damp for both modest & large ecc
    # (see eqns above)
    t_damp = 1.e5 * (1.0 / normalized_mass_ratio) * (normalized_aspect_ratio ** 4) * (
            1.0 / normalized_disk_surf_density_func) * (1.0 / np.sqrt(normalized_bh_locations))

    # timescale ratio for modest ecc damping
    modest_timescale_ratio = timestep_duration_yr / t_damp

    # timescale for large ecc damping from eqn. 2 above
    t_ecc = (t_damp / 0.78) * (1 - (0.14 * (e_h_ratio ** 2.0)) + (0.06 * (e_h_ratio ** 3.0)))
    large_timescale_ratio = timestep_duration_yr / t_ecc
    # print("t_damp",timestep_duration_yr/t_damp)
    # print("t_ecc",timestep_duration_yr/t_ecc)

    # print("timescale_ratio",timescale_ratio)
    new_disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] * np.exp(
        -modest_timescale_ratio[modest_ecc_prograde_indices])
    new_disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] * np.exp(
        -large_timescale_ratio[large_ecc_prograde_indices])
    new_disk_bh_pro_orbs_ecc = np.where(new_disk_bh_pro_orbs_ecc < disk_bh_pro_orb_ecc_crit, disk_bh_pro_orb_ecc_crit,
                                        new_disk_bh_pro_orbs_ecc)

    # print("Old ecc, New ecc",disk_bh_pro_orbs_ecc,new_disk_bh_pro_orbs_ecc)
    # Check new eccentricities
    assert np.isfinite(new_disk_bh_pro_orbs_ecc).all(),\
        "Finite check failed for new_disk_bh_pro_orbs_ecc"
    return new_disk_bh_pro_orbs_ecc


def ionized_orb_ecc(num_bh, orb_ecc_max):
    """Calculate new eccentricity for each component of an ionized binary.

    Parameters
    ----------
    num_bh : int
        Number of BHs (num of ionized binaries * 2)
    orb_ecc_max : float
        Maximum allowed orb_ecc
    """
    orb_eccs = rng.uniform(low=0.0, high=orb_ecc_max, size=num_bh)

    return (orb_eccs)
