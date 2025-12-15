"""Module for handling dynamical interactions.

Contains multiple functions which are each mocked up versions of a
dynamical mechanism. Of varying fidelity to reality. Also contains
GW orbital evolution for BH in the inner disk, which should probably
move elsewhere.
"""
import time
import numpy as np
import scipy

#import astropy.units as u
#import astropy.constants as const
from astropy import units as u, constants as const
from astropy.units import cds

from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.point_masses import time_of_orbital_shrinkage
from mcfacts.physics.point_masses import si_from_r_g, r_g_from_units


def circular_singles_encounters_prograde(
        smbh_mass,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        disk_radius_outer
        ):
    """Adjust orb ecc due to encounters between 2 single circ pro BH

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
    disk_radius_outer : float
        Outer radius of the inner disk (Rg)

    Returns
    -------
    disk_bh_pro_orbs_a : numpy.ndarray
        Updated BH semi-major axis [r_{g,SMBH}] perturbed by dynamics with :obj:`float` type
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Updated BH orbital eccentricities [unitless] perturbed by dynamics with :obj:`float` type

    Notes
    -----
    Return array of modified singleton BH orbital eccentricities perturbed
    by encounters within :math:`f*R_{Hill}`, where f is some fraction/multiple of
    Hill sphere radius R_H

    Assume encounters between damped BH (e<e_crit) and undamped BH
    (e>e_crit) are the only important ones for now.
    Since the e<e_crit population is the most likely BBH merger source.

    1, find those orbiters with e<e_crit and their
        associated semi-major axes a_circ =[a_circ1, a_circ2, ..] and masses m_circ =[m_circ1,m_circ2, ..].

    2, calculate orbital timescales for a_circ1 and a_i and N_orbits/timestep. 
        For example, since
        :math:`T_orb =2\\pi \sqrt(a^3/GM_{smbh})`
        and
        .. math::
        a^3/GM_{smbh} = (10^3r_g)^3/GM_{smbh} = 10^9 (a/10^3r_g)^3 (GM_{smbh}/c^2)^3/GM_{smbh} \\
                    = 10^9 (a/10^3r_g)^3 (G M_{smbh}/c^3)^2 

        So
        .. math::
            T_orb   = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} GM_{smbh}/c^3 \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (6.7e-11*2e38/(3e8)^3) \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (13.6e27/27e24) \\
                    = \\pi 10^{7.5}  (a/10^3r_g)^{3/2} \\
                    ~ 3yr (a/10^3r_g)^3/2 (M_{smbh}/10^8M_{sun}) \\
        i.e. Orbit~3yr at 10^3r_g around a 10^8M_{sun} SMBH.
        Therefore in a timestep=1.e4yr, a BH at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.

    3, among population of orbiters with e>e_crit,
        find those orbiters (a_i,e_i) where a_i*(1-e_i)< a_circ1,j <a_i*(1-e_i) for all members a_circ1,j of the circularized population 
        so we can test for possible interactions.

    4, calculate mutual Hill sphere R_H of candidate binary (a_circ1,j ,a_i).

    5, calculate ratio of 2R_H of binary to size of circular orbit, or (2R_H/2pi a_circ1,j)
        Hill sphere possible on both crossing inwards and outwards once per orbit, 
        so 2xHill sphere =4R_H worth of circular orbit will have possible encounter. 
        Thus, (4R_H/2pi a_circ1)= odds that a_circ1 is in the region of cross-over per orbit.
        For example, for BH at a_circ1 = 1e3r_g, 
            .. math:: R_h = a_{circ1}*(m_{circ1} + m_i/3M_{smbh})^1/3
            .. math:: = 0.004a_{circ1} (m_{circ1}/10M_{sun})^1/3 (m_i/10M_{sun})^1/3 (M_{smbh}/1e8M_{sun})^-1/3
        then
            ratio (4R_H/2pi a_circ1) = 0.008/pi ~ 0.0026 
            (ie around 1/400 odds that BH at a_circ1 is in either area of crossing)         

    6, calculate number of orbits of a_i in 1 timestep. 
        If e.g. N_orb(a_i)/timestep = 200 orbits per timestep of 10kyr, then 
        probability of encounter = (200orbits/timestep)*(4R_H/2pi a_circ1) ~ 0.5, 
                                or 50% odds of an encounter on this timestep between (a_circ1,j , a_i).
        If probability > 1, set probability = 1.
    7, draw a random number from the uniform [0,1] distribution and 
        if rng < probability of encounter, there is an encounter during the timestep
        if rng > probability of encounter, there is no encounter during the timestep

    8, if encounter:
        Take energy (de) from high ecc. a_i and give energy (de) to a_circ1,j
        de is average fractional energy change per encounter.
            So, a_circ1,j ->(1+de)a_circ1,j.    
                e_circ1,j ->(crit_ecc + de)
            and
                a_i       ->(1-de)a_i
                e_i       ->(1-de)e_i              
        Could be that average energy in gas-free cluster case is  
        assume average energy transfer = 20% perturbation (from Sigurdsson & Phinney 1993). 

        Further notes for self:
        sigma_ecc = sqrt(ecc^2 + incl^2)v_kep so if incl=0 deg (for now)
        En of ecc. interloper = 1/2 m_i sigma_ecc^2.
            Note: Can also use above logic for binary encounters except use binary binding energy instead.

        or later could try 
            Deflection angle defl = tan (defl) = dV_perp/V = 2GM/bV^2 kg^-1 m^3 s^-2 kg / m (m s^-1)^2
        so :math:`de/e =2GM/bV^2 = 2 G M_{bin}/0.5R_{hill}*\sigma^2`
        and :math:`R_hill = a_{circ1}*(M_{bin}/3M_{smbh})^1/3 and \sigma^2 =ecc^2*v_{kep}^2`
        So :math:`de/e = 4GM_{bin}/a_{circ1}(M_{bin}/3M_{smbh})^1/3 ecc^2 v_{kep}^2`
        and :math:`v_{kep} = \sqrt(GM_{smbh}/a_i)`
        So :math:`de/e = 4GM_{bin}^{2/3}M_{smbh}^1/3 a_i/a_{circ1} ecc^2 GM_{smbh} = 4(M_{bin}/M_{smbh})^{2/3} (a_i/a_{circ1})(1/ecc^2)
        where :math:`V_{rel} = \sigma` say and :math:`b=R_H = a_{circ1} (q/3)^{1/3}`
        So :math:`defl = 2GM/ a_{circ1}(q/3)^2/3 ecc^2 10^14 (m/s)^2 (R/10^3r_g)^-1`
            :math:`= 2 6.7e-11 2.e31/`
        !!Note: when doing this for binaries. 
            Calculate velocity of encounter compared to a_bin.
            If binary is hard ie GM_bin/a_bin > m3v_rel^2 then:
            harden binary 
                a_bin -> a_bin -da_bin and
            new binary eccentricity 
                e_bin -> e_bin + de  
            and give  da_bin worth of binding energy to extra eccentricity of m3.
            If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
            soften binary 
                a_bin -> a_bin + da_bin and
            new binary eccentricity
                e_bin -> e_bin + de
            and remove da_bin worth of binary energy from eccentricity of m3.
    """
    # Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc <= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc > disk_bh_pro_orb_ecc_crit).nonzero()[0]

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon = disk_radius_outer * ((disk_bh_pro_masses[circ_prograde_population_indices] / (3 * (disk_bh_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=circ_prograde_population_indices.size)

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = np.pi*((disk_bh_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*const.G.value)/(const.c.value**(3.0)*3.15e7)
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    # Generate all possible needed random numbers ahead of time
    chance_of_enc = rng.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
    num_poss_ints = 0
    num_encounters = 0
    if len(circ_prograde_population_indices) > 0:
        for i, circ_idx in enumerate(circ_prograde_population_indices):
            for j, ecc_idx in enumerate(ecc_prograde_population_indices):
                if (disk_bh_pro_orbs_a[circ_idx] < ecc_orb_max[j] and disk_bh_pro_orbs_a[circ_idx] > ecc_orb_min[j]):
                    # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                    # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                    temp_bin_mass = disk_bh_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]
                    bh_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                    mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
                    prob_orbit_overlap = (1./np.pi)*mass_ratio_factor
                    prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                    if prob_enc_per_timestep > 1:
                        prob_enc_per_timestep = 1
                    if chance_of_enc[i][j] < prob_enc_per_timestep:
                        num_encounters = num_encounters + 1
                        # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                        # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                        # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                        if disk_bh_pro_orbs_ecc[circ_idx] <= disk_bh_pro_orb_ecc_crit:
                            disk_bh_pro_orbs_ecc[circ_idx] = delta_energy_strong
                            disk_bh_pro_orbs_a[circ_idx] = disk_bh_pro_orbs_a[circ_idx]*(1.0 + delta_energy_strong)
                            # Catch for if orb_a > disk_radius_outer
                            if (disk_bh_pro_orbs_a[circ_idx] > disk_radius_outer):
                                disk_bh_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon[i]
                            disk_bh_pro_orbs_ecc[ecc_idx] = disk_bh_pro_orbs_ecc[ecc_idx]*(1 - delta_energy_strong)
                            disk_bh_pro_orbs_a[ecc_idx] = disk_bh_pro_orbs_a[ecc_idx]*(1 - delta_energy_strong)
                    num_poss_ints = num_poss_ints + 1
            num_poss_ints = 0
            num_encounters = 0

    # Check finite
    assert np.isfinite(disk_bh_pro_orbs_a).all(), \
        "Finite check failed for disk_bh_pro_orbs_a"
    assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for disk_bh_pro_orbs_ecc"
    assert np.all(disk_bh_pro_orbs_a < disk_radius_outer), \
        "disk_bh_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_bh_pro_orbs_a > 0), \
        "disk_bh_pro_orbs_a contains values <= 0"

    return (disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)


def circular_singles_encounters_prograde_stars(
        smbh_mass,
        disk_star_pro_orbs_a,
        disk_star_pro_masses,
        disk_star_pro_radius,
        disk_star_pro_orbs_ecc,
        disk_star_pro_id_nums,
        rstar_rhill_exponent,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        disk_radius_outer
        ):
    """Adjust orb ecc due to encounters between 2 single circ pro stars

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_pro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton star at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_pro_masses : numpy.ndarray
        Masses [M_sun] of prograde singleton star at start of timestep with :obj:`float` type
    disk_star_pro_radius : numpy.ndarray
        Radii [Rsun] of prograde singleton star at start of timestep with :obj: `float` type
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of singleton prograde star with :obj:`float` type
    disk_star_pro_id_nums : numpy.ndarray
        ID numbers of singleton prograde stars
    rstar_rhill_exponent : float
        Exponent for the ratio of R_star / R_Hill. Default is 2
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_bh_pro_orb_ecc_crit : float
        Critical orbital eccentricity [unitless] below which orbit is close enough to circularize
    delta_energy_strong : float
        Average energy change [units??] per strong encounter

    Returns
    -------
    disk_star_pro_orbs_a : numpy.ndarray
        Updated BH semi-major axis [r_{g,SMBH}] perturbed by dynamics with :obj:`float` type
    disk_star_pro_orbs_ecc : numpy.ndarray
        Updated BH orbital eccentricities [unitless] perturbed by dynamics with :obj:`float` type
    disk_star_pro_id_nums_touch : numpy.ndarray
        ID numbers of stars that will touch each other

    Notes
    -----
    Return array of modified singleton star orbital eccentricities perturbed
    by encounters within :math:`f*R_{Hill}`, where f is some fraction/multiple of
    Hill sphere radius R_H

    Assume encounters between damped star (e<e_crit) and undamped star
    (e>e_crit) are the only important ones for now.
    Since the e<e_crit population is the most likely BBH merger source.

    1, find those orbiters with e<e_crit and their
        associated semi-major axes a_circ =[a_circ1, a_circ2, ..] and masses m_circ =[m_circ1,m_circ2, ..].

    2, calculate orbital timescales for a_circ1 and a_i and N_orbits/timestep. 
        For example, since
        :math:`T_orb =2\\pi \sqrt(a^3/GM_{smbh})`
        and
        .. math::
        a^3/GM_{smbh} = (10^3r_g)^3/GM_{smbh} = 10^9 (a/10^3r_g)^3 (GM_{smbh}/c^2)^3/GM_{smbh} \\
                    = 10^9 (a/10^3r_g)^3 (G M_{smbh}/c^3)^2 

        So
        .. math::
            T_orb   = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} GM_{smbh}/c^3 \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (6.7e-11*2e38/(3e8)^3) \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (13.6e27/27e24) \\
                    = \\pi 10^{7.5}  (a/10^3r_g)^{3/2} \\
                    ~ 3yr (a/10^3r_g)^3/2 (M_{smbh}/10^8M_{sun}) \\
        i.e. Orbit~3yr at 10^3r_g around a 10^8M_{sun} SMBH.
        Therefore in a timestep=1.e4yr, a BH at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.

    3, among population of orbiters with e>e_crit,
        find those orbiters (a_i,e_i) where a_i*(1-e_i)< a_circ1,j <a_i*(1-e_i) for all members a_circ1,j of the circularized population 
        so we can test for possible interactions.

    4, calculate mutual Hill sphere R_H of candidate binary (a_circ1,j ,a_i).

    5, calculate ratio of 2R_H of binary to size of circular orbit, or (2R_H/2pi a_circ1,j)
        Hill sphere possible on both crossing inwards and outwards once per orbit, 
        so 2xHill sphere =4R_H worth of circular orbit will have possible encounter. 
        Thus, (4R_H/2pi a_circ1)= odds that a_circ1 is in the region of cross-over per orbit.
        For example, for BH at a_circ1 = 1e3r_g, 
            .. math:: R_h = a_{circ1}*(m_{circ1} + m_i/3M_{smbh})^1/3
            .. math:: = 0.004a_{circ1} (m_{circ1}/10M_{sun})^1/3 (m_i/10M_{sun})^1/3 (M_{smbh}/1e8M_{sun})^-1/3
        then
            ratio (4R_H/2pi a_circ1) = 0.008/pi ~ 0.0026 
            (ie around 1/400 odds that BH at a_circ1 is in either area of crossing)         

    6, calculate number of orbits of a_i in 1 timestep. 
        If e.g. N_orb(a_i)/timestep = 200 orbits per timestep of 10kyr, then 
        probability of encounter = (200orbits/timestep)*(4R_H/2pi a_circ1) ~ 0.5, 
                                or 50% odds of an encounter on this timestep between (a_circ1,j , a_i).
        If probability > 1, set probability = 1.
    7, draw a random number from the uniform [0,1] distribution and 
        if rng < probability of encounter, there is an encounter during the timestep
        if rng > probability of encounter, there is no encounter during the timestep

    8, if encounter:
        Take energy (de) from high ecc. a_i and give energy (de) to a_circ1,j
        de is average fractional energy change per encounter.
            So, a_circ1,j ->(1+de)a_circ1,j.    
                e_circ1,j ->(crit_ecc + de)
            and
                a_i       ->(1-de)a_i
                e_i       ->(1-de)e_i              
        Could be that average energy in gas-free cluster case is  
        assume average energy transfer = 20% perturbation (from Sigurdsson & Phinney 1993). 

        Further notes for self:
        sigma_ecc = sqrt(ecc^2 + incl^2)v_kep so if incl=0 deg (for now)
        En of ecc. interloper = 1/2 m_i sigma_ecc^2.
            Note: Can also use above logic for binary encounters except use binary binding energy instead.

        or later could try 
            Deflection angle defl = tan (defl) = dV_perp/V = 2GM/bV^2 kg^-1 m^3 s^-2 kg / m (m s^-1)^2
        so :math:`de/e =2GM/bV^2 = 2 G M_{bin}/0.5R_{hill}*\sigma^2`
        and :math:`R_hill = a_{circ1}*(M_{bin}/3M_{smbh})^1/3 and \sigma^2 =ecc^2*v_{kep}^2`
        So :math:`de/e = 4GM_{bin}/a_{circ1}(M_{bin}/3M_{smbh})^1/3 ecc^2 v_{kep}^2`
        and :math:`v_{kep} = \sqrt(GM_{smbh}/a_i)`
        So :math:`de/e = 4GM_{bin}^{2/3}M_{smbh}^1/3 a_i/a_{circ1} ecc^2 GM_{smbh} = 4(M_{bin}/M_{smbh})^{2/3} (a_i/a_{circ1})(1/ecc^2)
        where :math:`V_{rel} = \sigma` say and :math:`b=R_H = a_{circ1} (q/3)^{1/3}`
        So :math:`defl = 2GM/ a_{circ1}(q/3)^2/3 ecc^2 10^14 (m/s)^2 (R/10^3r_g)^-1`
            :math:`= 2 6.7e-11 2.e31/`
        !!Note: when doing this for binaries. 
            Calculate velocity of encounter compared to a_bin.
            If binary is hard ie GM_bin/a_bin > m3v_rel^2 then:
            harden binary 
                a_bin -> a_bin -da_bin and
            new binary eccentricity 
                e_bin -> e_bin + de  
            and give  da_bin worth of binding energy to extra eccentricity of m3.
            If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
            soften binary 
                a_bin -> a_bin + da_bin and
            new binary eccentricity
                e_bin -> e_bin + de
            and remove da_bin worth of binary energy from eccentricity of m3.
    """
    # Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population_indices = np.asarray(disk_star_pro_orbs_ecc <= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_star_pro_orbs_ecc > disk_bh_pro_orb_ecc_crit).nonzero()[0]

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon = disk_radius_outer * ((disk_star_pro_masses[circ_prograde_population_indices] / (3 * (disk_star_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=circ_prograde_population_indices.size)

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = scipy.constants.pi*((disk_star_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*scipy.constants.G)/(scipy.constants.c**(3.0)*3.15e7) 
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_star_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_star_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
    # Generate all possible needed random numbers ahead of time
    chance_of_enc = rng.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
    num_poss_ints = 0
    num_encounters = 0
    id_nums_poss_touch = []
    frac_rhill_sep = []
    if len(circ_prograde_population_indices) > 0:
        for i, circ_idx in enumerate(circ_prograde_population_indices):
            for j, ecc_idx in enumerate(ecc_prograde_population_indices):
                if (disk_star_pro_orbs_a[circ_idx] < ecc_orb_max[j] and disk_star_pro_orbs_a[circ_idx] > ecc_orb_min[j]):
                    # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                    # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                    temp_bin_mass = disk_star_pro_masses[circ_idx] + disk_star_pro_masses[ecc_idx]
                    star_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                    mass_ratio_factor = (star_smbh_mass_ratio)**(1./3.)
                    prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                    prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                    if prob_enc_per_timestep > 1:
                        prob_enc_per_timestep = 1
                    if chance_of_enc[i][j] < prob_enc_per_timestep:
                        num_encounters = num_encounters + 1
                        # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                        # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                        # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                        if disk_star_pro_orbs_ecc[circ_idx] <= disk_bh_pro_orb_ecc_crit:
                            disk_star_pro_orbs_ecc[circ_idx] = delta_energy_strong
                            disk_star_pro_orbs_a[circ_idx] = disk_star_pro_orbs_a[circ_idx]*(1.0 + delta_energy_strong)
                            # Catch for if orb_a > disk_radius_outer
                            if (disk_star_pro_orbs_a[circ_idx] > disk_radius_outer):
                                disk_star_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon[i]
                            disk_star_pro_orbs_ecc[ecc_idx] = disk_star_pro_orbs_ecc[ecc_idx]*(1 - delta_energy_strong)
                            disk_star_pro_orbs_a[ecc_idx] = disk_star_pro_orbs_a[ecc_idx]*(1 - delta_energy_strong)
                            # Look for stars that are inside each other's Hill spheres and if so return them as mergers
                            separation = np.abs(disk_star_pro_orbs_a[circ_idx] - disk_star_pro_orbs_a[ecc_idx])
                            center_of_mass = np.average([disk_star_pro_orbs_a[circ_idx], disk_star_pro_orbs_a[ecc_idx]],
                                                        weights=[disk_star_pro_masses[circ_idx], disk_star_pro_masses[ecc_idx]])
                            rhill_poss_encounter = center_of_mass * ((disk_star_pro_masses[circ_idx] + disk_star_pro_masses[ecc_idx]) / (3. * smbh_mass)) ** (1./3.)
                            if (separation - rhill_poss_encounter < 0):
                                id_nums_poss_touch.append(np.array([disk_star_pro_id_nums[circ_idx], disk_star_pro_id_nums[ecc_idx]]))
                                frac_rhill_sep.append(separation / rhill_poss_encounter)
                    num_poss_ints = num_poss_ints + 1
            num_poss_ints = 0
            num_encounters = 0

    # Check finite
    assert np.isfinite(disk_star_pro_orbs_a).all(), \
        "Finite check failed for disk_star_pro_orbs_a"
    assert np.isfinite(disk_star_pro_orbs_ecc).all(), \
        "Finite check failed for disk_star_pro_orbs_ecc"
    assert np.all(disk_star_pro_orbs_a < disk_radius_outer), \
        "disk_star_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_star_pro_orbs_a > 0), \
        "disk_star_pro_orbs_a contains values <= 0"

    id_nums_poss_touch = np.array(id_nums_poss_touch)
    frac_rhill_sep = np.array(frac_rhill_sep)

    # Test if there are any duplicate pairs, if so only return ID numbers of pair with smallest fractional Hill sphere separation
    if np.unique(id_nums_poss_touch).shape != id_nums_poss_touch.flatten().shape:
        sort_idx = np.argsort(frac_rhill_sep)
        id_nums_poss_touch = id_nums_poss_touch[sort_idx]
        uniq_vals, unq_counts = np.unique(id_nums_poss_touch, return_counts=True)
        dupe_vals = uniq_vals[unq_counts > 1]
        dupe_rows = id_nums_poss_touch[np.any(np.isin(id_nums_poss_touch, dupe_vals), axis=1)]
        uniq_rows = id_nums_poss_touch[np.all(~np.isin(id_nums_poss_touch, dupe_vals), axis=1)]

        rm_rows = []
        for row in dupe_rows:
            dupe_indices = np.any(np.isin(dupe_rows, row), axis=1).nonzero()[0][1:]
            rm_rows.append(dupe_indices)
        rm_rows = np.unique(np.concatenate(rm_rows))
        keep_mask = np.ones(len(dupe_rows))
        keep_mask[rm_rows] = 0

        id_nums_touch = np.concatenate((dupe_rows[keep_mask.astype(bool)], uniq_rows))

    else:
        id_nums_touch = id_nums_poss_touch

    id_nums_touch = id_nums_touch.T
    return (disk_star_pro_orbs_a, disk_star_pro_orbs_ecc, id_nums_touch)


def circular_singles_encounters_prograde_star_bh(
        smbh_mass,
        disk_star_pro_orbs_a,
        disk_star_pro_masses,
        disk_star_pro_radius,
        disk_star_pro_orbs_ecc,
        disk_star_pro_id_nums,
        rstar_rhill_exponent,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        disk_bh_pro_id_nums,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        disk_radius_outer
        ):
    """Adjust orb ecc due to encounters between 2 single circ pro stars

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_pro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton star at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_pro_masses : numpy.ndarray
        Masses [M_sun] of prograde singleton star at start of timestep with :obj:`float` type
    disk_star_pro_radius : numpy.ndarray
        Radii [Rsun] of prograde singleton star at start of timestep with :obj: `float` type
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of singleton prograde star with :obj:`float` type
    disk_star_pro_id_nums : numpy.ndarray
        ID numbers of singleton prograde stars
    rstar_rhill_exponent : float
        Exponent for the ratio of R_star / R_Hill. Default is 2
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_bh_pro_orb_ecc_crit : float
        Critical orbital eccentricity [unitless] below which orbit is close enough to circularize
    delta_energy_strong : float
        Average energy change [units??] per strong encounter

    Returns
    -------
    disk_star_pro_orbs_a : numpy.ndarray
        Updated stars semi-major axis [r_{g,SMBH}] perturbed by dynamics with :obj:`float` type
    disk_star_pro_orbs_ecc : numpy.ndarray
        Updated stars orbital eccentricities [unitless] perturbed by dynamics with :obj:`float` type
    disk_star_pro_id_nums_touch : numpy.ndarray
        ID numbers of stars that will touch each other
    disk_bh_pro_orbs_a : numpy.ndarray
        Updated BH semi-major axis [r_{g,SMBH}] perturbed by dynamics with :obj:`float` type
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Updated BH orbital eccentricities [unitless] perturbed by dynamics with :obj:`float` type

    Notes
    -----
    Return array of modified singleton star orbital eccentricities perturbed
    by encounters within :math:`f*R_{Hill}`, where f is some fraction/multiple of
    Hill sphere radius R_H

    Assume encounters between damped star (e<e_crit) and undamped star
    (e>e_crit) are the only important ones for now.
    Since the e<e_crit population is the most likely BBH merger source.

    1, find those orbiters with e<e_crit and their
        associated semi-major axes a_circ =[a_circ1, a_circ2, ..] and masses m_circ =[m_circ1,m_circ2, ..].

    2, calculate orbital timescales for a_circ1 and a_i and N_orbits/timestep. 
        For example, since
        :math:`T_orb =2\\pi \sqrt(a^3/GM_{smbh})`
        and
        .. math::
        a^3/GM_{smbh} = (10^3r_g)^3/GM_{smbh} = 10^9 (a/10^3r_g)^3 (GM_{smbh}/c^2)^3/GM_{smbh} \\
                    = 10^9 (a/10^3r_g)^3 (G M_{smbh}/c^3)^2 

        So
        .. math::
            T_orb   = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} GM_{smbh}/c^3 \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (6.7e-11*2e38/(3e8)^3) \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (13.6e27/27e24) \\
                    = \\pi 10^{7.5}  (a/10^3r_g)^{3/2} \\
                    ~ 3yr (a/10^3r_g)^3/2 (M_{smbh}/10^8M_{sun}) \\
        i.e. Orbit~3yr at 10^3r_g around a 10^8M_{sun} SMBH.
        Therefore in a timestep=1.e4yr, a BH at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.

    3, among population of orbiters with e>e_crit,
        find those orbiters (a_i,e_i) where a_i*(1-e_i)< a_circ1,j <a_i*(1-e_i) for all members a_circ1,j of the circularized population 
        so we can test for possible interactions.

    4, calculate mutual Hill sphere R_H of candidate binary (a_circ1,j ,a_i).

    5, calculate ratio of 2R_H of binary to size of circular orbit, or (2R_H/2pi a_circ1,j)
        Hill sphere possible on both crossing inwards and outwards once per orbit, 
        so 2xHill sphere =4R_H worth of circular orbit will have possible encounter. 
        Thus, (4R_H/2pi a_circ1)= odds that a_circ1 is in the region of cross-over per orbit.
        For example, for BH at a_circ1 = 1e3r_g, 
            .. math:: R_h = a_{circ1}*(m_{circ1} + m_i/3M_{smbh})^1/3
            .. math:: = 0.004a_{circ1} (m_{circ1}/10M_{sun})^1/3 (m_i/10M_{sun})^1/3 (M_{smbh}/1e8M_{sun})^-1/3
        then
            ratio (4R_H/2pi a_circ1) = 0.008/pi ~ 0.0026 
            (ie around 1/400 odds that BH at a_circ1 is in either area of crossing)         

    6, calculate number of orbits of a_i in 1 timestep. 
        If e.g. N_orb(a_i)/timestep = 200 orbits per timestep of 10kyr, then 
        probability of encounter = (200orbits/timestep)*(4R_H/2pi a_circ1) ~ 0.5, 
                                or 50% odds of an encounter on this timestep between (a_circ1,j , a_i).
        If probability > 1, set probability = 1.
    7, draw a random number from the uniform [0,1] distribution and 
        if rng < probability of encounter, there is an encounter during the timestep
        if rng > probability of encounter, there is no encounter during the timestep

    8, if encounter:
        Take energy (de) from high ecc. a_i and give energy (de) to a_circ1,j
        de is average fractional energy change per encounter.
            So, a_circ1,j ->(1+de)a_circ1,j.    
                e_circ1,j ->(crit_ecc + de)
            and
                a_i       ->(1-de)a_i
                e_i       ->(1-de)e_i              
        Could be that average energy in gas-free cluster case is  
        assume average energy transfer = 20% perturbation (from Sigurdsson & Phinney 1993). 

        Further notes for self:
        sigma_ecc = sqrt(ecc^2 + incl^2)v_kep so if incl=0 deg (for now)
        En of ecc. interloper = 1/2 m_i sigma_ecc^2.
            Note: Can also use above logic for binary encounters except use binary binding energy instead.

        or later could try 
            Deflection angle defl = tan (defl) = dV_perp/V = 2GM/bV^2 kg^-1 m^3 s^-2 kg / m (m s^-1)^2
        so :math:`de/e =2GM/bV^2 = 2 G M_{bin}/0.5R_{hill}*\sigma^2`
        and :math:`R_hill = a_{circ1}*(M_{bin}/3M_{smbh})^1/3 and \sigma^2 =ecc^2*v_{kep}^2`
        So :math:`de/e = 4GM_{bin}/a_{circ1}(M_{bin}/3M_{smbh})^1/3 ecc^2 v_{kep}^2`
        and :math:`v_{kep} = \sqrt(GM_{smbh}/a_i)`
        So :math:`de/e = 4GM_{bin}^{2/3}M_{smbh}^1/3 a_i/a_{circ1} ecc^2 GM_{smbh} = 4(M_{bin}/M_{smbh})^{2/3} (a_i/a_{circ1})(1/ecc^2)
        where :math:`V_{rel} = \sigma` say and :math:`b=R_H = a_{circ1} (q/3)^{1/3}`
        So :math:`defl = 2GM/ a_{circ1}(q/3)^2/3 ecc^2 10^14 (m/s)^2 (R/10^3r_g)^-1`
            :math:`= 2 6.7e-11 2.e31/`
        !!Note: when doing this for binaries. 
            Calculate velocity of encounter compared to a_bin.
            If binary is hard ie GM_bin/a_bin > m3v_rel^2 then:
            harden binary 
                a_bin -> a_bin -da_bin and
            new binary eccentricity 
                e_bin -> e_bin + de  
            and give  da_bin worth of binding energy to extra eccentricity of m3.
            If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
            soften binary 
                a_bin -> a_bin + da_bin and
            new binary eccentricity
                e_bin -> e_bin + de
            and remove da_bin worth of binary energy from eccentricity of m3.
    """
    # We are comparing the CIRCULARIZED stars and the ECCENTRIC black holes
    # Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population_indices = np.asarray(disk_star_pro_orbs_ecc <= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc > disk_bh_pro_orb_ecc_crit).nonzero()[0]

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon_star = disk_radius_outer * ((disk_star_pro_masses[circ_prograde_population_indices] / (3 * (disk_star_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)) * rng.uniform(size=circ_prograde_population_indices.size)

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = scipy.constants.pi*((disk_star_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*scipy.constants.G)/(scipy.constants.c**(3.0)*3.15e7) 
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_bh_pro_orbs_a[ecc_prograde_population_indices])
    ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_bh_pro_orbs_a[ecc_prograde_population_indices])
    num_poss_ints = 0
    num_encounters = 0
    # Generate all possible needed random numbers ahead of time
    chance_of_enc = rng.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
    id_nums_poss_touch = []
    frac_rhill_sep = []
    if len(circ_prograde_population_indices) > 0:
        for i, circ_idx in enumerate(circ_prograde_population_indices):
            for j, ecc_idx in enumerate(ecc_prograde_population_indices):
                if (disk_star_pro_orbs_a[circ_idx] < ecc_orb_max[j] and disk_star_pro_orbs_a[circ_idx] > ecc_orb_min[j]):
                    # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                    # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                    temp_bin_mass = disk_star_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]
                    star_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                    mass_ratio_factor = (star_smbh_mass_ratio)**(1./3.)
                    prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                    prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                    if prob_enc_per_timestep > 1:
                        prob_enc_per_timestep = 1
                    if chance_of_enc[i][j] < prob_enc_per_timestep:
                        num_encounters = num_encounters + 1
                        # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                        # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                        # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                        if disk_star_pro_orbs_ecc[circ_idx] <= disk_bh_pro_orb_ecc_crit:
                            disk_star_pro_orbs_ecc[circ_idx] = delta_energy_strong
                            disk_star_pro_orbs_a[circ_idx] = disk_star_pro_orbs_a[circ_idx]*(1.0 + delta_energy_strong)
                            # Catch for if orb_a > disk_radius_outer
                            if (disk_star_pro_orbs_a[circ_idx] > disk_radius_outer):
                                disk_star_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon_star[i]
                            disk_bh_pro_orbs_ecc[ecc_idx] = disk_bh_pro_orbs_ecc[ecc_idx]*(1 - delta_energy_strong)
                            disk_bh_pro_orbs_a[ecc_idx] = disk_bh_pro_orbs_a[ecc_idx]*(1 - delta_energy_strong)
                            # Look for stars that are inside each other's Hill spheres and if so return them as mergers
                            separation = np.abs(disk_star_pro_orbs_a[circ_idx] - disk_bh_pro_orbs_a[ecc_idx])
                            center_of_mass = np.average([disk_star_pro_orbs_a[circ_idx], disk_bh_pro_orbs_a[ecc_idx]],
                                                        weights=[disk_star_pro_masses[circ_idx], disk_bh_pro_masses[ecc_idx]])
                            rhill_poss_encounter = center_of_mass * ((disk_star_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]) / (3. * smbh_mass)) ** (1./3.)
                            if (separation - rhill_poss_encounter < 0):
                                id_nums_poss_touch.append(np.array([disk_star_pro_id_nums[circ_idx], disk_bh_pro_id_nums[ecc_idx]]))
                                frac_rhill_sep.append(separation / rhill_poss_encounter)
                    num_poss_ints = num_poss_ints + 1
            num_poss_ints = 0
            num_encounters = 0

    # Check finite
    assert np.isfinite(disk_star_pro_orbs_a).all(), \
        "Finite check failed for disk_star_pro_orbs_a"
    assert np.isfinite(disk_star_pro_orbs_ecc).all(), \
        "Finite check failed for disk_star_pro_orbs_ecc"
    assert np.isfinite(disk_bh_pro_orbs_a).all(), \
        "Finite check failed for disk_bh_pro_orbs_a"
    assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for disk_bh_pro_orbs_ecc"
    assert np.all(disk_star_pro_orbs_a < disk_radius_outer), \
        "disk_star_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_bh_pro_orbs_a < disk_radius_outer), \
        "disk_bh_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_bh_pro_orbs_a > 0), \
        "disk_bh_pro_orbs_a contains values <= 0"
    assert np.all(disk_star_pro_orbs_a > 0), \
        "disk_star_pro_orbs_a contains values <= 0"

    # Put ID nums array into correct shape
    id_nums_poss_touch = np.array(id_nums_poss_touch)
    frac_rhill_sep = np.array(frac_rhill_sep)

    # Test if there are any duplicate pairs, if so only return ID numbers of pair with smallest fractional Hill sphere separation
    if np.unique(id_nums_poss_touch).shape != id_nums_poss_touch.flatten().shape:
        sort_idx = np.argsort(frac_rhill_sep)
        id_nums_poss_touch = id_nums_poss_touch[sort_idx]
        uniq_vals, unq_counts = np.unique(id_nums_poss_touch, return_counts=True)
        dupe_vals = uniq_vals[unq_counts > 1]
        dupe_rows = id_nums_poss_touch[np.any(np.isin(id_nums_poss_touch, dupe_vals), axis=1)]
        uniq_rows = id_nums_poss_touch[np.all(~np.isin(id_nums_poss_touch, dupe_vals), axis=1)]

        rm_rows = []
        for row in dupe_rows:
            dupe_indices = np.any(np.isin(dupe_rows, row), axis=1).nonzero()[0][1:]
            rm_rows.append(dupe_indices)
        rm_rows = np.unique(np.concatenate(rm_rows))
        keep_mask = np.ones(len(dupe_rows))
        keep_mask[rm_rows] = 0

        id_nums_touch = np.concatenate((dupe_rows[keep_mask.astype(bool)], uniq_rows))

    else:
        id_nums_touch = id_nums_poss_touch

    id_nums_touch = id_nums_touch.T
    return (disk_star_pro_orbs_a, disk_star_pro_orbs_ecc, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc, id_nums_touch)


def circular_binaries_encounters_ecc_prograde(
        smbh_mass,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        bin_mass_1,
        bin_mass_2,
        bin_orb_a,
        bin_sep,
        bin_ecc,
        bin_orb_ecc,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        disk_radius_outer
        ):
    """"Adjust orb eccentricities due to encounters between BBH and eccentric single BHs

    Return array of modified binary BH separations and eccentricities
    perturbed by encounters within f*R_Hill, for eccentric singleton
    population, where f is some fraction/multiple of Hill sphere radius R_H
    Right now assume f=1.

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
    disk_bins_bhbh : numpy.ndarray
        [21, bindex] mixed array containing properties of binary BBH, see add_to_binary_array function for
        complete description
    disk_radius_outer : float
        Outer radius of the inner disk (Rg)

    Returns
    -------
    disk_bins_bhbh : numpy.ndarray
        [21, bindex] mixed array, updated version of input after dynamical perturbations

    Notes
    -----
    Logic:
            0.  Find number of binaries in this timestep given by bindex
            1.  Find the binary center of mass (c.o.m.) and corresponding orbital velocities & binary total masses.
                disk_bins_bhbh[9,:] = bin c.o.m. = [R_bin1_orb_a,R_bin2_orb_a,...]. These are the orbital radii of the bins.
                disk_bins_bhbh[8,;] = bin_separation =[a_bin1,a_bin2,...]
                disk_bins_bhbh[2,:]+disk_bins_bhbh[3,:] = mass of binaries
                disk_bins_bhbh[13,:] = ecc of binary around com
                disk_bins_bhbh[18,:] = orb. ecc of binary com around SMBH
                Keplerian orbital velocity of the bin c.o.m. around SMBH: v_bin,i= sqrt(GM_SMBH/R_bin,i_com)= c/sqrt(R_bin,i_com)
            2.  Calculate the binary orbital time and N_orbits/timestep
                For example, since
                T_orb =2pi sqrt{bin,orb a}^3/GM_smbh)
                and {bin,orb a}^3/GM_smbh = (10^3r_g)^3/GM_smbh = 10^9 ({bin,orb a}/10^3r_g)^3 (GM_smbh/c^2)^3/GM_smbh 
                    = 10^9 ({bin,orb a}/10^3r_g)^3 (G M_smbh/c^3)^2 

                So,
                .. math::
                    T_{orb}
                    = 2\\pi 10^{4.5} (R_{bin,orb a}/10^3r_g)^{3/2} GM_{smbh}/c^3
                    = 2\\pi 10^{4.5} (R_{bin,orb a}/10^3r_g)^{3/2} (6.7e-11*2e38/(3e8)^3)
                    = 2\\pi 10^{4.5} (R_{bin,orb a}/10^3r_g)^{3/2} (13.6e27/27e24)
                    = \\pi 10^{7.5}  (R_{bin,orb a}/10^3r_g)^{3/2}
                    ~ 3.15 yr (R_{bin,orb a}/10^3r_g)^3/2 (M_smbh/10^8Msun)
                i.e. Orbit~3.15yr at 10^3r_g around a 10^8M_{sun} SMBH.
                Therefore in a timestep=1.e4yr, a binary at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.
            3.  Calculate binding energy of bins = [GM1M2/sep_bin1, GMiMi+1,sep_bin2, ....] where sep_bin1 is in meters and M1,M2 are binary mass components in kg.
            4.  Find those single BH with e>e_crit and their
                associated semi-major axes a_ecc =[a_ecc1, a_ecc2, ..] and masses m_ecc =[m_ecc1,m_ecc2, ..]
                and calculate their average velocities v_ecc = [GM_smbh/a_ecc1, GM_smbh/a_ecc2,...]
            5.  Where (1-ecc_i)*a_ecc_i < R_bin_j_com < (1+ecc_i)*a_ecc_i, interaction possible
            6.  Among candidate encounters, calculate relative velocity of encounter.
                        :math:`v_{peri,i}=\\sqrt(Gm_{ecc,i}/a_{ecc,i}[1+ecc,i/1-ecc,i])`
                        :math:`v_{apo,i} =\\sqrt(Gm_{ecc,i}/a_{ecc,i}[1-ecc,i/1+ecc,i])`
                        :math:`v_{ecc,i} =\\sqrt(GM/a_{ecc_i})` ..average Keplerian vel.

                    :math:`v_{rel} = abs(v_{bin,i} - v_{ecc,i})`
            7. Calculate relative K.E. of tertiary, (1/2)m_ecc_i*v_rel_^2     
            8. Compare binding en of binary to K.E. of tertiary.
                Critical velocity for ionization of binary is v_crit, given by:
                    :math:`v_{crit} = \\sqrt(GM_1M_2(M_1+M_2+M_3)/M_3(M_1+M_2)a_{bin})
                If binary is hard ie GM_1M_2/a_bin > m3v_rel^2 then:
                    harden binary
                        a_bin -> a_bin -da_bin and
                    new binary eccentricity
                        e_bin -> e_bin + de 
                    and give  +da_bin worth of binding energy (GM_bin/(a_bin -da_bin) - GM_bin/a_bin) 
                    to extra eccentricity ecc_i and a_ecc,i of m_ecc,i.
                    Say average en of encounter is de=0.1 (10%) then binary a_bin shrinks by 10%, ecc_bin is pumped by 10%
                    And a_ecc_i shrinks by 10% and ecc_i also shrinks by 10%
                If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
                    if v_rel (effectively v_infty) > v_crit
                        ionize binary
                            update singleton array with 2 new BH with orbital eccentricity e_crit+de
                            remove binary from binary array
                    else if v_rel < v_crit
                        soften binary
                            a_bin -> a_bin + da_bin and
                        new binary eccentricity
                            e_bin -> e_bin + de
                        and remove -da_bin worth of binary energy from eccentricity of m3.
            Note1: Will need to test binary eccentricity each timestep.
                If bin_ecc> some value (0.9), check for da_bin due to GW bremsstrahlung at pericenter.
            9. As 4, except now include interactions between binaries and circularized BH. This should give us primarily
                hardening encounters as in Leigh+2018, since the v_rel is likely to be small for more binaries.

    Given array of binaries at locations [a_bbh1,a_bbh2] with 
    binary semi-major axes [a_bin1,a_bin2,...] and binary eccentricities [e_bin1,e_bin2,...],
    find all the single BH at locations a_i that within timestep 
        either pass between a_i(1-e_i)< a_bbh1 <a_i(1+e_i)

    Calculate velocity of encounter compared to a_bin.
    If binary is hard ie GM1M2/a_bin > m3v_rel^2 then:
      harden binary to a_bin = a_bin -da_bin and
      new binary eccentricity e_bin = e_bin + de around com and
      new binary orb eccentricity e_orb_com = e_orb_com + de and 
      now give  da_bin worth of binding energy to extra eccentricity of m3.
    If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
      soften binary to a_bin = a_bin + da_bin and
      new binary eccentricity e_bin = e_bin + de
      and take da_bin worth of binary energy from eccentricity of m3. 
    If binary is unbound ie GM_bin/a_bin << m3v_rel^2 then:
      remove binary from binary array
      add binary components m1,m2 back to singleton arrays with new orbital eccentricities e_1,e_2 from energy of encounter.
      Equipartition energy so m1v1^2 =m2 v_2^2 and 
      generate new individual orbital eccentricities e1=v1/v_kep_circ and e_2=v_2/v_kep_circ
      Take energy put into destroying binary from orb. eccentricity of m3.
    """

    # Set up constants
    solar_mass = u.solMass.to("kg")
    # eccentricity correction--do not let ecc>=1, catch and reset to 1-epsilon
    epsilon = 1e-8

    # Set up other values we need
    bin_masses = bin_mass_1 + bin_mass_2
    bin_velocities = const.c.value / np.sqrt(bin_orb_a)
    bin_binding_energy = const.G.value * (solar_mass ** 2) * bin_mass_1 * bin_mass_2 / (si_from_r_g(smbh_mass, bin_sep).to("meter")).value
    bin_orbital_times = 3.15 * (smbh_mass / 1.e8) * ((bin_orb_a / 1.e3) ** 1.5)
    bin_orbits_per_timestep = timestep_duration_yr/bin_orbital_times

    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc >= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find their locations and masses
    ecc_prograde_population_locations = disk_bh_pro_orbs_a[ecc_prograde_population_indices]
    ecc_prograde_population_masses = disk_bh_pro_masses[ecc_prograde_population_indices]
    ecc_prograde_population_eccentricities = disk_bh_pro_orbs_ecc[ecc_prograde_population_indices]
    # Find min and max radii around SMBH for eccentric orbiters
    ecc_orb_min = ecc_prograde_population_locations * (1.0-ecc_prograde_population_eccentricities)
    ecc_orb_max = ecc_prograde_population_locations * (1.0+ecc_prograde_population_eccentricities)
    # Keplerian velocity of ecc prograde orbiter around SMBH (=c/sqrt(a/r_g))
    ecc_velocities = const.c.value / np.sqrt(ecc_prograde_population_locations)

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon_orb_a = disk_radius_outer * ((ecc_prograde_population_masses / (3 * (ecc_prograde_population_masses + smbh_mass)))**(1. / 3.)) * rng.uniform(size=len(ecc_prograde_population_masses))

    if np.size(bin_mass_1) == 0:
        return (bin_sep, bin_ecc, bin_orb_ecc, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)

    # Create array of random numbers for the chances of encounters
    chances = rng.uniform(size=(np.size(bin_mass_1), ecc_prograde_population_indices.size))

    # For each binary in blackholes_binary
    for i in range(0, np.size(bin_mass_1)):
        # We compare each single BH to that binary
        for j in range(0, len(ecc_prograde_population_indices)):
            # If binary com orbit lies inside eccentric orbit [min,max] radius
            # i.e. if R_m3_minimum lie inside R_bin_maximum and does R_m3_max lie outside R_bin_minimum
            if (1.0 - bin_orb_ecc[i]) * bin_orb_a[i] < ecc_orb_max[j] and (1.0 + bin_orb_ecc[i]) * bin_orb_a[i] > ecc_orb_min[j]:

                # Make a temporary Hill sphere treating binary + ecc interloper as a 'binary' = M_1+M_2+M_3
                # r_h = a_circ1(temp_bin_mass/3mass_smbh)^1/3 so prob_enc/orb = mass_ratio^1/3/pi

                temp_bin_mass = bin_masses[i] + ecc_prograde_population_masses[j]
                bh_smbh_mass_ratio = temp_bin_mass / (3.0 * smbh_mass)
                mass_ratio_factor = bh_smbh_mass_ratio ** (1./3.)
                prob_orbit_overlap = (1. / np.pi) * mass_ratio_factor
                prob_enc_per_timestep = prob_orbit_overlap * bin_orbits_per_timestep[i]
                # Cap prob_enc_per_timestep at 1
                if prob_enc_per_timestep > 1:
                    prob_enc_per_timestep = 1
                chances_of_encounter = chances[i][j]

                if chances_of_encounter < prob_enc_per_timestep:
                    # Perturb *this* ith binary depending on how hard it already is.
                    relative_velocities = np.abs(bin_velocities[i] - ecc_velocities[j])

                    # K.E. of interloper
                    ke_interloper = 0.5 * ecc_prograde_population_masses[j] * solar_mass * (relative_velocities ** 2.0)
                    hard = bin_binding_energy[i] - ke_interloper

                    if hard > 0:
                        # Binary is hard w.r.t interloper
                        # Change binary parameters; decr separation, incr ecc around bin_orb_a and orb_ecc
                        bin_sep[i] = bin_sep[i] * (1 - delta_energy_strong)
                        bin_ecc[i] = bin_ecc[i] * (1 + delta_energy_strong)
                        bin_orb_ecc[i] = bin_orb_ecc[i] * (1 + delta_energy_strong)
                        # Change interloper parameters; increase a_ecc, increase e_ecc
                        ecc_prograde_population_locations[j] = ecc_prograde_population_locations[j] * (1 + delta_energy_strong)
                        # Catch for if location > disk_radius_outer #
                        if (ecc_prograde_population_locations[j] > disk_radius_outer):
                            ecc_prograde_population_locations[j] = disk_radius_outer - epsilon_orb_a[j]
                        ecc_prograde_population_eccentricities[j] = ecc_prograde_population_eccentricities[j] * (1 + delta_energy_strong)

                    if hard < 0:
                        # Binary is soft w.r.t. interloper
                        # Check to see if binary is ionized
                        # Change binary parameters; incr bin separation, decr ecc around com, incr orb_ecc
                        bin_sep[i] = bin_sep[i] * (1 + delta_energy_strong)
                        bin_ecc[i] = bin_ecc[i] * (1 - delta_energy_strong)
                        bin_orb_ecc[i] = bin_orb_ecc[i] * (1 + delta_energy_strong)
                        # Change interloper parameters; decrease a_ecc, decrease e_ecc
                        ecc_prograde_population_locations[j] = ecc_prograde_population_locations[j] * (1 - delta_energy_strong)
                        ecc_prograde_population_eccentricities[j] = ecc_prograde_population_eccentricities[j] * (1 - delta_energy_strong)

                    # Catch if bin_ecc or bin_orb_ecc >= 1
                    if bin_ecc[i] >= 1:
                        bin_ecc[i] = 1.0 - epsilon
                    if bin_orb_ecc[i] >= 1:
                        bin_orb_ecc[i] = 1.0 - epsilon
                    # Catch if single BHs have ecc >= 1
                    if ecc_prograde_population_eccentricities[j] >= 1:
                        ecc_prograde_population_eccentricities[j] = 1.0 - epsilon

    # TODO: ALSO return new array of singletons with changed params.
    disk_bh_pro_orbs_a[ecc_prograde_population_indices] = ecc_prograde_population_locations
    disk_bh_pro_orbs_ecc[ecc_prograde_population_indices] = ecc_prograde_population_eccentricities

    # Check finite
    assert np.isfinite(bin_sep).all(), \
        "Finite check failure: bin_separations"
    assert np.isfinite(bin_orb_ecc).all(), \
        "Finite check failure: bin_orbital_eccentricities"
    assert np.isfinite(bin_ecc).all(), \
        "Finite check failure: bin_eccentricities"
    assert np.all(ecc_prograde_population_locations < disk_radius_outer), \
        "ecc_prograde_population_locations has values greater than disk_radius_outer"
    assert np.all(ecc_prograde_population_locations > 0), \
        "ecc_prograde_population_locations contains values <= 0"
    assert np.all(bin_sep >= 0), \
        "bin_sep contains values < 0"

    return bin_sep, bin_ecc, bin_orb_ecc, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc


def circular_binaries_encounters_circ_prograde(
        smbh_mass,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        bin_mass_1,
        bin_mass_2,
        bin_orb_a,
        bin_sep,
        bin_ecc,
        bin_orb_ecc,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        disk_radius_outer,
        mean_harden_energy_delta,
        var_harden_energy_delta
        ):
    """"Adjust orb ecc due to encounters btw BBH and circularized singles

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
    disk_bins_bhbh : numpy.ndarray
        [21, bindex] mixed array containing properties of binary BBH, see add_to_binary_array function for
        complete description
    disk_radius_outer : float
        Outer radius of the inner disk (Rg)
    var_harden_energy_delta : float
        Average energy exchanged in a strong 2 + 1 interaction that hardens the binary
    mean_harden_energy_delta : float
        Variance of the energy exchanged in a strong 2 + 1 interaction that hardens the binary

    Returns
    -------
    disk_bins_bhbh : numpy.ndarray
        [21, bindex] mixed array, updated version of input after dynamical perturbations

    Notes
    -----
    Return array of modified binary BH separations and eccentricities
    perturbed by encounters within f*R_Hill, for circularized singleton
    population, where f is some fraction/multiple of Hill sphere radius
    R_H
    Right now assume f=1.
    Logic:  
            0.  Find number of binaries in this timestep given by bindex
            1.  Find the binary center of mass (c.o.m.) and corresponding orbital velocities & binary total masses.
                disk_bins_bhbh[9,:] = bin c.o.m. = [R_bin1_orb_a,R_bin2_orb_a,...]. These are the orbital radii of the bins.
                disk_bins_bhbh[8,;] = bin_separation =[a_bin1,a_bin2,...]
                disk_bins_bhbh[2,:]+disk_bins_bhbh[3,:] = mass of binaries
                disk_bins_bhbh[13,:] = ecc of binary around com
                disk_bins_bhbh[18,:] = orb. ecc of binary com around SMBH
                Keplerian orbital velocity of the bin c.o.m. around SMBH: v_bin,i= sqrt(GM_SMBH/R_bin,i_com)= c/sqrt(R_bin,i_com)
            2.  Calculate the binary orbital time and N_orbits/timestep
                For example, since
                T_orb =2pi sqrt(R_bin_com^3/GM_smbh)
                and R_bin_com^3/GM_smbh = (10^3r_g)^3/GM_smbh = 10^9 (R_bin_com/10^3r_g)^3 (GM_smbh/c^2)^3/GM_smbh 
                    = 10^9 (R_bin_com/10^3r_g)^3 (G M_smbh/c^3)^2 

                So,
                .. math::
                    T_{orb}
                    = 2\\pi 10^{4.5} (R_{bin,orb a}/10^3r_g)^{3/2} GM_{smbh}/c^3
                    = 2\\pi 10^{4.5} (R_{bin,orb a}/10^3r_g)^{3/2} (6.7e-11*2e38/(3e8)^3)
                    = 2\\pi 10^{4.5} (R_{bin,orb a}/10^3r_g)^{3/2} (13.6e27/27e24)
                    = \\pi 10^{7.5}  (R_{bin,orb a}/10^3r_g)^{3/2}
                    ~ 3.15 yr (R_{bin,orb a}/10^3r_g)^3/2 (M_smbh/10^8Msun)
                i.e. Orbit~3.15yr at 10^3r_g around a 10^8M_{sun} SMBH.
                Therefore in a timestep=1.e4yr, a binary at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.
            3.  Calculate binding energy of bins = [GM1M2/sep_bin1, GMiMi+1,sep_bin2, ....] where sep_bin1 is in meters and M1,M2 are binary mass components in kg.
            4.  Find those single BH with e>e_crit and their
                associated semi-major axes a_ecc =[a_ecc1, a_ecc2, ..] and masses m_ecc =[m_ecc1,m_ecc2, ..]
                and calculate their average velocities v_ecc = [GM_smbh/a_ecc1, GM_smbh/a_ecc2,...]
            5.  Where (1-ecc_i)*a_ecc_i < R_bin_j_com < (1+ecc_i)*a_ecc_i, interaction possible
            6.  Among candidate encounters, calculate relative velocity of encounter.
                        :math:`v_{peri,i}=\\sqrt(Gm_{ecc,i}/a_{ecc,i}[1+ecc,i/1-ecc,i])`
                        :math:`v_{apo,i} =\\sqrt(Gm_{ecc,i}/a_{ecc,i}[1-ecc,i/1+ecc,i])`
                        :math:`v_{ecc,i} =\\sqrt(GM/a_{ecc_i})` ..average Keplerian vel.

                    :math:`v_{rel} = abs(v_{bin,i} - v_{ecc,i})`
            7. Calculate relative K.E. of tertiary, (1/2)m_ecc_i*v_rel_^2
            8. Compare binding en of binary to K.E. of tertiary.
                Critical velocity for ionization of binary is v_crit, given by:
                    :math:`v_{crit} = \\sqrt(GM_1M_2(M_1+M_2+M_3)/M_3(M_1+M_2)a_{bin})
                If binary is hard ie GM_1M_2/a_bin > m3v_rel^2 then:
                    harden binary 
                        a_bin -> a_bin -da_bin and
                    new binary eccentricity
                        e_bin -> e_bin + de 
                    and give  +da_bin worth of binding energy (GM_bin/(a_bin -da_bin) - GM_bin/a_bin)
                    to extra eccentricity ecc_i and a_ecc,i of m_ecc,i.
                    Say average en of encounter is de=0.1 (10%) then binary a_bin shrinks by 10%, ecc_bin is pumped by 10%
                    And a_ecc_i shrinks by 10% and ecc_i also shrinks by 10%
                If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
                    if v_rel (effectively v_infty) > v_crit
                        ionize binary
                            update singleton array with 2 new BH with orbital eccentricity e_crit+de
                            remove binary from binary array
                    else if v_rel < v_crit
                        soften binary 
                            a_bin -> a_bin + da_bin and
                        new binary eccentricity
                            e_bin -> e_bin + de
                        and remove -da_bin worth of binary energy from eccentricity of m3.
            Note1: Will need to test binary eccentricity each timestep.
                If bin_ecc> some value (0.9), check for da_bin due to GW bremsstrahlung at pericenter.
            9. As 4, except now include interactions between binaries and circularized BH. This should give us primarily
                hardening encounters as in Leigh+2018, since the v_rel is likely to be small for more binaries.

    Given array of binaries at locations [a_bbh1,a_bbh2] with
    binary semi-major axes [a_bin1,a_bin2,...] and binary eccentricities [e_bin1,e_bin2,...],
    find all the single BH at locations a_i that within timestep
        either pass between a_i(1-e_i)< a_bbh1 <a_i(1+e_i)

    Calculate velocity of encounter compared to a_bin.
    If binary is hard ie GM1M2/a_bin > m3v_rel^2 then:
      harden binary to a_bin = a_bin -da_bin and
      new binary eccentricity e_bin = e_bin + de around com and
      new binary orb eccentricity e_orb_com = e_orb_com + de and
      now give  da_bin worth of binding energy to extra eccentricity of m3.
    If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
      soften binary to a_bin = a_bin + da_bin and
      new binary eccentricity e_bin = e_bin + de
      and take da_bin worth of binary energy from eccentricity of m3.
    If binary is unbound ie GM_bin/a_bin << m3v_rel^2 then:
      remove binary from binary array
      add binary components m1,m2 back to singleton arrays with new orbital eccentricities e_1,e_2 from energy of encounter.
      Equipartition energy so m1v1^2 =m2 v_2^2 and
      generate new individual orbital eccentricities e1=v1/v_kep_circ and e_2=v_2/v_kep_circ
      Take energy put into destroying binary from orb. eccentricity of m3.
    """

    # Housekeeping
    solar_mass = u.solMass.to("kg")

    # Magnitude of energy change to drive binary to merger in ~2 interactions in a strong encounter. Say de_strong=0.9
    # de_strong here refers to the perturbation of the binary around its center of mass
    # The energy in the exchange is assumed to come from the binary binding energy around its c.o.m.
    # delta_energy_strong (read into this module) refers to the perturbation of the orbit of the binary c.o.m. around the SMBH, which is not as strongly perturbed (we take an 'average' perturbation)

    # Pick from a normal distribution defined by the user, and bound it between 0 and 1.
    de_strong = max(0., min(1., rng.normal(mean_harden_energy_delta, var_harden_energy_delta)))

    # eccentricity correction--do not let ecc>=1, catch and reset to 1-epsilon
    epsilon = 1e-8

    # Set up arrays for later
    bin_masses = bin_mass_1 + bin_mass_2
    bin_velocities = const.c.value/np.sqrt(bin_orb_a)
    bin_orbital_times = 3.15 * (smbh_mass / 1.e8) * ((bin_orb_a / 1.e3) ** 1.5)
    bin_orbits_per_timestep = timestep_duration_yr / bin_orbital_times
    bin_binding_energy = const.G.value * (solar_mass ** 2.0) * bin_mass_1 * bin_mass_2 / (si_from_r_g(smbh_mass, bin_sep).to("meter")).value

    # Find the e< crit_ecc population. These are the interlopers w. low encounter vel that can harden the circularized population
    circ_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc <= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find their locations and masses
    circ_prograde_population_locations = disk_bh_pro_orbs_a[circ_prograde_population_indices]
    circ_prograde_population_masses = disk_bh_pro_masses[circ_prograde_population_indices]
    circ_prograde_population_eccentricities = disk_bh_pro_orbs_ecc[circ_prograde_population_indices]
    # Find min and max radii around SMBH for eccentric orbiters
    ecc_orb_min = disk_bh_pro_orbs_a[circ_prograde_population_indices]*(1.0-disk_bh_pro_orbs_ecc[circ_prograde_population_indices])
    ecc_orb_max = disk_bh_pro_orbs_a[circ_prograde_population_indices]*(1.0+disk_bh_pro_orbs_ecc[circ_prograde_population_indices])
    # Keplerian velocity of ecc prograde orbiter around SMBH (=c/sqrt(a/r_g))
    circ_velocities = const.c.value/np.sqrt(circ_prograde_population_locations)

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon_orb_a = disk_radius_outer * ((circ_prograde_population_masses / (3 * (circ_prograde_population_masses + smbh_mass)))**(1. / 3.)) * rng.uniform(size=len(circ_prograde_population_masses))

    if np.size(bin_mass_1) == 0:
        return (bin_sep, bin_ecc, bin_orb_ecc, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)

    # Set up random numbers
    chances = rng.uniform(size=(np.size(bin_mass_1), len(circ_prograde_population_locations)))

    for i in range(0, np.size(bin_mass_1)):
        for j in range(0, len(circ_prograde_population_locations)):
            # If binary com orbit lies inside circ orbit [min,max] radius
            # i.e. does R_m3_minimum lie inside R_bin_maximum and does R_m3_max lie outside R_bin_minimum
            if (1.0 - bin_orb_ecc[i]) * bin_orb_a[i] < ecc_orb_max[j] and (1.0 + bin_orb_ecc[i]) * bin_orb_a[i] > ecc_orb_min[j]:
                # Make a temporary Hill sphere treating binary + ecc interloper as a 'binary' = M_1+M_2+M_3
                # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                temp_bin_mass = bin_masses[i] + circ_prograde_population_masses[j]
                bh_smbh_mass_ratio = temp_bin_mass/(3.0 * smbh_mass)
                mass_ratio_factor = (bh_smbh_mass_ratio ** (1./3.))
                prob_orbit_overlap = (1. / np.pi) * mass_ratio_factor
                prob_enc_per_timestep = prob_orbit_overlap * bin_orbits_per_timestep[i]
                if prob_enc_per_timestep > 1:
                    prob_enc_per_timestep = 1

                chance_of_encounter = chances[i][j]
                if chance_of_encounter < prob_enc_per_timestep:
                    # Perturb *this* ith binary depending on how hard it already is.
                    # Find relative velocity of interloper in km/s so divide by 1.e3
                    rel_vel_ms = abs(bin_velocities[i] - circ_velocities[j])
                    # K.E. of interloper
                    ke_interloper = 0.5 * circ_prograde_population_masses[j] * solar_mass * (rel_vel_ms ** 2.0)
                    hard = bin_binding_energy[i] - ke_interloper

                    if (hard > 0):
                        # Binary is hard w.r.t interloper
                        # Change binary parameters; decr separation, incr ecc around com and orb_ecc
                        # de_strong here refers to the perturbation of the binary around its center of mass
                        # The energy in the exchange is assumed to come from the binary binding energy around its c.o.m.
                        # delta_energy_strong refers to the perturbation of the orbit of the binary c.o.m. around the SMBH, which is not as strongly perturbed (we take an 'average' perturbation) 
                        bin_sep[i] = bin_sep[i] * (1 - de_strong)
                        bin_ecc[i] = bin_ecc[i] * (1 + de_strong)
                        bin_orb_ecc[i] = bin_orb_ecc[i] * (1 + delta_energy_strong)
                        # Change interloper parameters; increase a_ecc, increase e_ecc
                        circ_prograde_population_locations[j] = circ_prograde_population_locations[j] * (1 + delta_energy_strong)
                        if (circ_prograde_population_locations[j] > disk_radius_outer):
                            circ_prograde_population_locations[j] = disk_radius_outer - epsilon_orb_a[j]
                        circ_prograde_population_eccentricities[j] = circ_prograde_population_eccentricities[j] * (1 + delta_energy_strong)

                    if hard < 0:
                        # Binary is soft w.r.t. interloper
                        # Check to see if binary is ionized
                        # Change binary parameters; incr bin separation, decr ecc around com, incr orb_ecc
                        bin_sep[i] = bin_sep[i] * (1 + delta_energy_strong)
                        bin_ecc[i] = bin_ecc[i] * (1 - delta_energy_strong)
                        bin_orb_ecc[i] = bin_orb_ecc[i] * (1 + delta_energy_strong)
                        # Change interloper parameters; decrease a_ecc, decrease e_ecc
                        circ_prograde_population_locations[j] = circ_prograde_population_locations[j] * (1 - delta_energy_strong)
                        if (circ_prograde_population_locations[j] > disk_radius_outer):
                            circ_prograde_population_locations[j] = disk_radius_outer - epsilon_orb_a[j]
                        circ_prograde_population_eccentricities[j] = circ_prograde_population_eccentricities[j] * (1 - delta_energy_strong)

                    # Catch where bin_orb_ecc and bin_ecc >= 1
                    if (bin_ecc[i] >= 1):
                        bin_ecc[i] = 1.0 - epsilon
                    if (bin_orb_ecc[i] >= 1):
                        bin_orb_ecc[i] = 1.0 - epsilon
                    # Catch where single BHs have ecc >= 1
                    if circ_prograde_population_eccentricities[j] >= 1:
                        circ_prograde_population_eccentricities[j] = 1.0 - epsilon

    disk_bh_pro_orbs_a[circ_prograde_population_indices] = circ_prograde_population_locations
    disk_bh_pro_orbs_ecc[circ_prograde_population_indices] = circ_prograde_population_eccentricities

    # Check finite
    assert np.isfinite(bin_sep).all(), \
        "Finite check failure: bin_separations"
    assert np.isfinite(bin_orb_ecc).all(), \
        "Finite check failure: bin_orbital_eccentricities"
    assert np.isfinite(bin_ecc).all(), \
        "Finite check failure: bin_eccentricities"
    assert np.all(circ_prograde_population_locations < disk_radius_outer), \
        "ecc_prograde_population_locations has values greater than disk_radius_outer"
    assert np.all(circ_prograde_population_locations > 0), \
        "circ_prograde_population_locations contains values <= 0"
    assert np.all(bin_sep >= 0), \
        "bin_sep contains values < 0"

    return (bin_sep, bin_ecc, bin_orb_ecc, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)


def bin_spheroid_encounter(
        smbh_mass,
        timestep_duration_yr,
        bin_mass_1_all,
        bin_mass_2_all,
        bin_orb_a_all,
        bin_sep_all,
        bin_ecc_all,
        bin_orb_ecc_all,
        bin_orb_inc_all,
        time_passed,
        nsc_bh_imf_powerlaw_index,
        delta_energy_strong,
        nsc_spheroid_normalization,
        mean_harden_energy_delta,
        var_harden_energy_delta
        ):
    """Perturb orbits due to encounters with spheroid (NSC) objects

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bins_bhbh : numpy.ndarray
        [21, bindex] mixed array containing properties of binary BBH, see add_to_binary_array function for
        complete description
    time_passed : float
        Current time set [yr]
            nsc_bh_imf_powerlaw_index : float
            Powerlaw index of nuclear star cluster BH IMF (e.g. M^-2) [unitless]. User set (default = 2).
    timestep_duration_yr : float
        Length of timestep [yr]
    nsc_bh_imf_powerlaw_index : float
        Powerlaw index of nuclear star cluster BH IMF (e.g. M^-2) [unitless]. User set (default = 2).
    delta_energy_strong : float
        Average energy change [units??] per strong encounter
    nsc_spheroid_normalization : float
        Normalization factor [unitless] determines the departures from sphericity of
        the initial distribution of perturbers (1.0=spherical)
    var_harden_energy_delta : float
        Average energy exchanged in a strong 2 + 1 interaction that hardens the binary
    mean_harden_energy_delta : float
        Variance of the energy exchanged in a strong 2 + 1 interaction that hardens the binary


    Returns
    -------
    disk_bins_bhbh : [21, bindex] mixed array
        updated version of input after dynamical perturbations

    Notes
    -----
    Warning: the powerlaw index for the mass of perturbers is for BH
    but should be for stars, and the mode mass is hardcoded inside the fn

    Use Leigh+18 to figure out the rate at which spheroid encounters happen to binaries embedded in the disk
    Binaries at small disk radii encounter spheroid objects at high rate, particularly early on in the disk lifetime
    However, orbits at those small radii get captured quickly by the disk.

    From Fig.1 in Leigh+18, Rate of sph. encounter = 20/Myr at t=0, normalized to a_bin=1AU, R_disk=10^3r_g or 0.2/10kyr timestep.
    Introduce a spheroid normalization factor nsc_spheroid_normalization=1 (default) allowing for non-ideal NSC (previous episodes; disky populations etc). 
    Within 1Myr, for a dense model disk (e.g. Sirko & Goodman), most of those inner stellar orbits have been captured by the disk.
    So rate of sph. encounter ->0/Myr at t=1Myr since those orbits are gone (R<10^3r_g; assuming approx circular orbits!) for SG disk model
    For TQM disk model, rate of encounter slightly lower but non-zero.

    So, inside R_com<10^3r_g: (would be rt of enc =0.2 if nsc_spheroid_normalization=1)
    Assume: :math:`\text{Rate of encounter} = 0.2 (\text{nsc_spheroid_normalization}/1)(\text{timestep}/10kyr)^{-1} (R_{com}/10^3r_g)^{-1} (a_{bin}/1r_gM8)^{-2}`
    Generate random number from uniform [0,1] distribution and if <0.2 (normalized to above condition) then encounter

    Encounter rt starts at = :math:` 0.2 (\text{nsc_spheroid_normalization}/1)(\text{timestep}/10kyr)^{-1} (R_{com}/10^3r_g)^{-1} (a_{bin}/1r_gM8)^{-2}` at t=0
    decreases to          = :math:` 0. (\text{nsc_spheroid_normalization}/1)(\text{timestep}/10kyr)^{-1} (R_{com}/10^3r_g)^{-1} (a_{bin}/1r_gM8)^{-2}` (time_passed/1Myr)
    at R<10^3r_g.
    Outside: R_com>10^3r_g
    Normalize to rate at (R_com/10^4r_g) so that rate is non-zero at R_com=[1e3,1e4]r_g after 1Myr.
    Decrease rate with time, but ensure it goes to zero at R_com<1.e3r_g.

    So, rate of sph. encounter = 2/Myr at t=0, normalized to a_bin=1AU, R_disk=10^4r_g which is equivalently
    Encounter rate = 0.02 (nsc_spheroid_normalization/1)(timestep/10kyr)^-1 (R_com/10^4r_g)^-1 (a_bin/1r_gM8)^2
    Drop this by an order of magnitude over 1Myr.
    Encounter rate = 0.02 (timestep/10kyr)^-1 (R_com/10^4r_g)^-1 (a_bin/1r_gM8)^2 (time_passed/10kyr)^-1/2
    so ->0.002 after a Myr
    For R_com < 10^3r_g:
        if time_passed <=1Myr
            Encounter rt = 0.02*(nsc_spheroid_normalization/0.1)*(1-(1Myr/time_passed))(timestep/10kyr)^{-1}(R_com/10^3r_g)^-1 (a_bin/1r_gM8)^2 ....(1)
        if time_passed >1Myr
            Encounter rt = 0
    For R_com > 10^3r_g:
        Encounter rt = 0.002 *(nsc_spheroid_normalization/0.1)* (timestep/10kyr)^-1 (R_com/10^4r_g)^-1 (a_bin/1r_gM8)^2 (time_passed/10kyr)^-1/2 ....(2)

    Return corrected binary with spin angles projected onto new L_bin. So can calculate chi_p (in plane components of spin)
    Return new binary inclination angle w.r.t disk
    Harden/soften/ionize binary as appropriate

    Orbital angular momentum:
    Binary orbital angular momentum is
        L_bin =M_bin*v_orb_bin X R_com
    Spheroid orbital angular momentum is
        L3=m3*v3 X R3
    where m3,v3,R3 are the mass, velocity and semi-major axis of tertiary encounter.

    Draw m3 from IMF random distrib. BUT mostly stars early on!
    TO DO: Switch from spheroid stars to spheroid BH at late time
    Draw a3 from uniform distribution a3=[10^-0.5,0.5]a_bbh say. v_3= c/sqrt(R_3)
    Ratio of L3/Lbin =(m3/M_bin)*sqrt(R3/R_com)....(3)
    so L3 = ratio*Lbin

    Resultant L_bin must be the resultant in a parallelogram of L3 (one side) and L_bin(other side)

    Angle of encounter:
    If angle of encounter between BBH and M3 (angle_enc)<|90deg|, ie angle_enc is in [0-90deg,270-360deg] then: 
    L_bin_new = sqrt(L3^2 + L_bin^2 + 2L3L_bin cos(angle_enc)) ....(4)
              = sqrt( (1+ratio^2)L_bin^2 + 2ratioL_bin^2 cos(angle_enc))
              = sqrt((1+ratio^2) + 2ratio*cos(angle_enc)) L_bin_old
    else if angle_enc is in [90deg,270 deg]
    L_bin_new = sqrt(L3^2 + L_bin^2 - 2L3L_bin cos(angle_enc)) ....(5)
              = sqrt((1+ratio^2) - 2ratio*cos(angle_end))L_bin_old
    and
    L_bin_new/L_bin = v_b_new x R_com_new/ v_b_old x R_com_old
    and for Keplerian vels
    v_b_com = sqrt(GM_smbh/a_com) so

    L_bin_new/L_bin_old = sqrt(a_com_new/a_com_old) ....(6)

    So new BBH semi-major axis:
    a_com_new = (L_bin_new/L_bin_old)^2 *(a_com_old) ....(7)

    Angle of encounter:
    M3 has some random angle (i3) in the spheroid wrt disk (i=0deg) & BBH (also presumed i=0deg).
    But, over time, spheroid population (of STARS) with small inclination angles wrt disk
    (i=0 deg) are captured by disk (takes ~1Myr in SG disk; Fabj+20)
    So, at t=0, start with drawing from uniform distribution of i3=[0,360]
    After 1Myr in a SG disk, we want all the spheroid (star!) encounters inside R=1000r_g to go to zero.
    Over time remove e.g. i3=[0,+/-15], so draw from [15,345] next timestep
    Then remove i3 =+/-[15,30] so draw from [30,330] etc.
    So,
    if crit_time =1.e6 #1Myr
    then
    excluded_angles =(time_passed/crit_time)*180
    select from i3 = [excluded angles,360-excluded angles]
    So:

    crit_time=1.e6
    if time_passed < crit_time
        excluded_angles = (time_passed/crit_time)*180
        if R<10^3r_g
            #Draw random integer in range [excluded_angles,360-(excluded_angles)]
            i3 = rng.randint(excluded_angles, 360-(excluded_angles))....(8)

    Calculate velocity of encounter compared to a_bin.
    Ignore what happens to m3, since it's a random draw from the NSC and we are not tracking individual NSC components.
    If binary is hard ie GM1M2/a_bin > m3v_rel^2 then:
      harden binary to a_bin = a_bin -da_bin and
      new binary eccentricity e_bin = e_bin + de around com and
      new binary orb eccentricity e_orb_com = e_orb_com + de
    If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
      soften binary to a_bin = a_bin + da_bin and
      new binary eccentricity e_bin = e_bin + de
    """

    # Units of r_g normalized to 1AU around a 10^8Msun SMBH
    dist_in_rg_m8 = 1.0 * (1.0e8/smbh_mass)

    # Critical time (in yrs) for capture of all BH with a<1e3r_g (default is 1Myr for Sirko & Goodman (2003) disk)
    crit_time = 1.e6
    # Critical disk radius (in units of r_g,SMBH) where after crit_time, all the spheroid orbits are captured.
    crit_radius = 1.e3
    # Solar mass in units of kg
    solar_mass = u.solMass.to("kg")
    # Magnitude of energy change to drive binary to merger in ~2 interactions in a strong encounter. Say de_strong=0.9
    # de_strong here refers to the perturbation of the binary around its center of mass
    # The energy in the exchange is assumed to come from the binary binding energy around its c.o.m.
    # delta_energy_strong refers to the perturbation of the orbit of the binary c.o.m. around the SMBH, which is not as strongly perturbed (we take an 'average' perturbation) 

    # Pick from a normal distribution defined by the user, and bound it between 0 and 1.
    de_strong = max(0., min(1., rng.normal(mean_harden_energy_delta, var_harden_energy_delta)))

    # eccentricity correction--do not let ecc>=1, catch and reset to 1-epsilon
    epsilon = 1e-8
    # Spheroid normalization to allow for non-ideal NSC (cored/previous AGN episodes/disky population concentration/whatever)

    # Set up binary properties we need for later
    bin_mass = bin_mass_1_all + bin_mass_2_all
    bin_velocities = const.c.value / np.sqrt(bin_orb_a_all)
    bin_binding_energy = const.G.value * (solar_mass ** 2) * bin_mass_1_all * bin_mass_2_all / (si_from_r_g(smbh_mass, bin_sep_all).to("meter")).value

    # Calculate encounter rate for each binary based on bin_orb_a, binary size, and time_passed
    # Set up array of encounter rates filled with -1
    enc_rate = np.full(np.size(bin_mass_1_all), -1.5)

    # Set encounter rate if bin_orb_a < crit_radius
    enc_rate[(bin_orb_a_all < crit_radius) & (time_passed <= crit_time)] = 0.02 * (nsc_spheroid_normalization / 0.1) * (1.0 - (time_passed / 1.e6)) * ((bin_sep_all[(bin_orb_a_all < crit_radius) & (time_passed <= crit_time)] / dist_in_rg_m8) ** 2.0) / ((timestep_duration_yr / 1.e4) * (bin_orb_a_all[(bin_orb_a_all < crit_radius) & (time_passed <= crit_time)] / 1.e3))
    enc_rate[(bin_orb_a_all < crit_radius) & (time_passed > crit_time)] = 0.0

    # Set encounter rate if bin_orb_a > crit_radius
    enc_rate[(bin_orb_a_all > crit_radius)] = 0.002 * (nsc_spheroid_normalization / 0.1) * ((bin_sep_all[(bin_orb_a_all > crit_radius)] / dist_in_rg_m8) ** 2.0) / ((timestep_duration_yr / 1.e4) * (bin_orb_a_all[(bin_orb_a_all > crit_radius)] / 1.e4) * np.sqrt(time_passed / 1.e4))
    # If enc_rate still has negative values throw error
    if (np.sum(enc_rate < 0) > 0):
        print("enc_rate",enc_rate)
        raise RuntimeError("enc_rate not being set in bin_spheroid_encounter")

    # If bin_orb_a == crit_radius throw error
    if (np.sum(bin_orb_a_all == crit_radius) > 0):
        print("SMBH mass:", smbh_mass)
        print("bin_orb_a:", bin_orb_a_all[bin_orb_a_all == crit_radius])
        print("crit_radius:", crit_radius)
        raise RuntimeError("Unrecognized bin_orb_a")

    # Based on estimated encounter rate, calculate if binary actually has a spheroid encounter
    chances_of_encounter = rng.uniform(size=np.size(bin_mass_1_all))
    encounter_index = np.where(chances_of_encounter < enc_rate)[0]
    num_encounters = np.size(encounter_index)

    if (num_encounters > 0):

        # Set up arrays for changed blackholes_binary parameters
        bin_orb_a = bin_orb_a_all[encounter_index].copy()
        bin_sep = bin_sep_all[encounter_index].copy()
        bin_ecc = bin_ecc_all[encounter_index].copy()
        bin_orb_ecc = bin_orb_ecc_all[encounter_index].copy()
        bin_orb_inc = bin_orb_inc_all[encounter_index].copy()

        # Have already generated spheroid interaction, so a_3 is not far off a_bbh (unless super high ecc). 
        # Assume a_3 is similar to a_bbh (within a factor of O(3), so allowing for modest relative eccentricity)    
        # i.e. a_3=[10^-0.5,10^0.5]*a_bbh.

        # Calculate interloper parameters
        # NOTE: Stars should be most common sph component. Switch to BH after some long time.
        mode_star = 2.0
        mass_3 = (rng.pareto(nsc_bh_imf_powerlaw_index, size=num_encounters) + 1) * mode_star
        radius_3 = bin_orb_a * (10 ** (-0.5 + rng.uniform(size=num_encounters)))
        # K.E_3 in Joules
        # Keplerian velocity of ecc prograde orbiter around SMBH (=c/sqrt(a/r_g))
        velocity_3 = const.c.value / np.sqrt(radius_3)
        relative_velocities = np.abs(bin_velocities[encounter_index] - velocity_3)
        ke_3 = 0.5 * mass_3 * solar_mass * (relative_velocities ** 2.0)

        # Compare orbital angular momentum for interloper and binary
        # Ratio of L3/Lbin =(m3/M_bin)*sqrt(R3/R_com)
        L_ratio = (mass_3 / bin_mass[encounter_index]) * np.sqrt(radius_3 / bin_orb_a)

        excluded_angles = np.full(num_encounters, -100.5)

        # If time_passed < crit_time then gradually decrease angles i3 available at a < 1000r_g
        if (time_passed < crit_time):
            # Set up arrays for angles
            excluded_angles[radius_3 < crit_radius] = (time_passed/crit_time) * 180

            # If radius_3 > crit_radius make grind down much slower at >1000r_g (say all captured in 20 Myr for < 5e4r_g)
            excluded_angles[radius_3 > crit_radius] = 0.05 * (time_passed/crit_time) * 180

        elif time_passed >= crit_time:
            # No encounters inside R < 10^3 r_g
            excluded_angles[radius_3 < crit_radius] = 360

            # If radius_3 > crit_radius all stars captured out to 1e4r_g after 100 Myr
            excluded_angles[radius_3 > crit_radius] = 0.01 * (time_passed / crit_time) * 180

        # If excluded_angles has any negative elements throw error
        if (np.sum(excluded_angles < 0) > 0):
            print("excluded_angles",excluded_angles)
            raise RuntimeError("excluded_angles not being set in bin_spheroid_encounter")

        # Draw random integer in range [excluded_angles,360-(excluded_angles)]
        # i3 in units of degrees
        # where 0 deg = disk mid-plane prograde, 180 deg= disk mid-plane retrograde,
        # 90deg = aligned with L_disk, 270 deg = anti-aligned with disk)
        #
        # Vera: somebody set excluded_angles = 360 to indicate that all
        #   angles should be excluded. However, this causes the whole
        #   pipeline to crash. This is a temporary fix.
        include_mask = excluded_angles < 360
        include_index = encounter_index[include_mask]
        excluded_angles[~include_mask] = 0.
        i3 = rng.randint(low=excluded_angles, high=360-excluded_angles)
        # Convert i3 to radians
        i3_rad = np.radians(i3)

        # Ionize/soften/harden binary if appropriate
        hard = bin_binding_energy[encounter_index] - ke_3
        # Create mask for hard and soft
        mask_hard = hard > 0
        mask_soft = hard < 0

        # If hard > 0 binary is hard wrt interloper
        # Change binary parameters: decrease separation, increase ecc around bin_orb_a and orb_ecc
        bin_sep[mask_hard] = bin_sep[mask_hard] * (1 - de_strong)
        bin_ecc[mask_hard] = bin_ecc[mask_hard] * (1 + de_strong)
        bin_orb_ecc[mask_hard] = bin_orb_ecc[mask_hard] * (1 + delta_energy_strong)
        # Ignore interloper parameters, since just drawing randomly from IMF population

        # If hard < 0 binary is soft wrt interloper
        # Change binary parameters: increase separation, decrease ecc around bin_orb_a, increase orb_ecc
        bin_sep[mask_soft] = bin_sep[mask_soft] * (1 + delta_energy_strong)
        bin_ecc[mask_soft] = bin_ecc[mask_soft] * (1 - delta_energy_strong)
        bin_orb_ecc[mask_soft] = bin_orb_ecc[mask_soft] * (1 + delta_energy_strong)

        # Catch if bin_ecc or bin_orb_ecc >= 1
        bin_ecc[bin_ecc >= 1.0] = 1.0 - epsilon
        bin_orb_ecc[bin_orb_ecc >= 1.0] = 1.0 - epsilon

        # New angle of binary wrt disk (in radians)
        bin_orb_inc[(L_ratio < 1)] = bin_orb_inc[(L_ratio < 1)] + L_ratio[L_ratio < 1] * (i3_rad[L_ratio < 1]/2.0)
        bin_orb_inc[(L_ratio > 1)] = bin_orb_inc[(L_ratio > 1)] + (1./L_ratio[L_ratio > 1]) * (i3_rad[L_ratio > 1]/2.0)

        bin_sep_all[include_index] = bin_sep[include_mask]
        bin_ecc_all[include_index] = bin_ecc[include_mask]
        bin_orb_ecc_all[include_index] = bin_orb_ecc[include_mask]
        bin_orb_inc_all[include_index] = bin_orb_inc[include_mask]

    # Test new values
    assert np.isfinite(bin_sep_all).all(), \
        "Finite check failure: bin_sep_all"
    assert np.isfinite(bin_orb_inc_all).all(), \
        "Finite check failure: bin_orb_inc_all"
    assert np.all(bin_sep_all >= 0), \
        "bin_sep_all contains values < 0"


    return (bin_sep_all, bin_ecc_all, bin_orb_ecc_all, bin_orb_inc_all)


def bin_recapture(bin_mass_1_all, bin_mass_2_all, bin_orb_a_all, bin_orb_inc_all, timestep_duration_yr):
    """Recapture BBH that has orbital inclination >0 post spheroid encounter

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        binary black holes
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes with binary orbital inclination [radian] updated

    Notes
    -----
    Purely bogus scaling does not account for real disk surface density.
    From Fabj+20, if i<5deg (=(5deg/180deg)*pi=0.09rad), time to recapture a BH in SG disk is 1Myr (M_b/10Msun)^-1(R/10^4r_g)
    if i=[5,15]deg =(0.09-0.27rad), time to recapture a BH in SG disk is 50Myrs(M_b/10Msun)^-1 (R/10^4r_g)
    For now, ignore if i>15deg (>0.27rad)
    """
    # Critical inclinations (5deg,15deg for SG disk model)
    crit_inc1 = 0.09
    crit_inc2 = 0.27

    idx_gtr_0 = bin_orb_inc_all > 0

    if (idx_gtr_0.shape[0] == 0):
        return (bin_orb_inc_all)

    bin_orb_inc = bin_orb_inc_all[idx_gtr_0]
    bin_mass = bin_mass_1_all[idx_gtr_0] + bin_mass_2_all[idx_gtr_0]
    bin_orb_a = bin_orb_a_all[idx_gtr_0]

    less_crit_inc1_mask = bin_orb_inc < crit_inc1
    bwtwn_crit_inc1_inc2_mask = (bin_orb_inc > crit_inc1) & (bin_orb_inc < crit_inc2)

    # is bin orbital inclination <5deg in SG disk?
    bin_orb_inc[less_crit_inc1_mask] = bin_orb_inc[less_crit_inc1_mask] * (1. - ((timestep_duration_yr/1e6) * (bin_mass[less_crit_inc1_mask] / 10.) * (bin_orb_a[less_crit_inc1_mask] / 1.e4)))
    bin_orb_inc[bwtwn_crit_inc1_inc2_mask] = bin_orb_inc[bwtwn_crit_inc1_inc2_mask] * (1. - ((timestep_duration_yr/5.e7) * (bin_mass[bwtwn_crit_inc1_inc2_mask] / 10.) * (bin_orb_a[bwtwn_crit_inc1_inc2_mask] / 1.e4)))

    bin_orb_inc_all[idx_gtr_0] = bin_orb_inc

    assert np.isfinite(bin_orb_inc_all).all(), \
        "Finite check failure: bin_orb_inc_all"

    return (bin_orb_inc_all)

def quiescence_relaxation_time(
        smbh_mass,
        blackholes_survivors_pop,
        quiescence_time,
        galaxy_num,
        ):
    """Calculate relaxation times according to equations in Panamarev & Kocsis '23
      
    Intuition:
    Approximately in order of fastest to slowest:
    Vector resonant relaxation allows orb_inc to relax
    Scalar resonant relaxation allows orb_ecc to relax
    2-body relaxation time allows orb_a to relax
    Timescales for a sphere of mass N*m around M, scale approximately (from Kocsis & Tremaine 2015)
    t_vrr  = T_orb *(M/m sqrt(N))
    t_srr  = T_orb *(M/m)
    t_2bdy = T_orb *(M/m)^2*(1/N)
    
    Maths:
    Spheroidal interactions:
        t_sph_2body_relax = 0.34 sigma^3(r)/[G^2 rho(r) m ln Lambda].....from Binney & Tremaine '87
    can be written as
        t_sph_2body_relax = T_orb * (M_smbh/m_2)^2 * 1/(N*ln Lambda)
    and scalar resonant relaxation (change in eccentricity and orbital angular momentum)
        t_sph_scalar_rr = T_orb *2*(M_smbh/m2) *(N*m/M(r))
    and vector resonant relaxation (change in angle of orbital angular momentum or inclination angle)
        t_vector_rr = T_orb* (M_smbh/sqrt(m_2*M(r)))* 1/bv^2
    where bv~1.83.
    Disky interactions:
    2-body relaxation rate from disky interactions:
        t_disk_2body_relax = T_orb *(<e^2>^2/(9*pi))*(M_smbh^2/m_2*Nm)*(1/ln(<e^2>^3/2 M_smbh/m)))
    and vector resonant relaxation from disky interactions
        t_disk_vector_rr = T_orb *(M_smbh/m_2)*(1/<i^2>^1/2)
        where <i^2>1/2 = RMS(i) ~ 0.5 RMS(e)

    T_orb = 2pi R/v = 2pi R(r_g)^3/2 * r_g_in_meters/c 
                    = 2pi 31*10^3*(1.5e11/3e8)=pi*31*10^6 =10^8s = 3yrs at 1000r_g around 10^8Msun SMBH e.g.       
    t_rel = [6*sqrt(2)/(32 pi^1/2)]* v_kep^3/[G^2 ln (M_smbh/M_i)*M_i^2 n_i]
    where v_kep= Keplerian velocity (m/s)
    m2 = < m^2>/<m> is the effective mass where m is the mean mass of stars (say 1Msun)
    G = Universal Gravitational Constant
    M_smbh = SMBH mass (in kg)
    M_i = mass of BH i (in kg)
    n_i = number density of these BH_i (m^-3).
    NB: 1pc^3 = (3.1e16m)^3. Assume 10^4 BH in central pc^3.  Assume Bahcall-Wolf density profile so
      n_i = 10^4* (r/pc)^-7/4 in units of pc^-3. So divide by (3.1e16m)^3
      n_i = 10^4* (r/2.e5)^-1.75 

    Compare to relaxation time from Alexander, Begelman & Armitage (2007) also in McKernan+2012 eqn.5
    Assume an annulus of objects centered on R, of width dR, containing N stars of mass m_i, then
    
    t_rel_disk = C_1* R* dR * sigma^4 *T_orb/[G^2 N m_i^2 ln Lambda]
    where C_1 is a constant O(1), T_orb is orbital timescale in disk
    
    Args:
        smbh_mass (_type_): _description_
    """
    #Bahcall-Wolf profile (r/pc)^-7/4
    density_index = -1.75
    year= u.yr.to(u.s) *u.s
    yr_s = np.pi*10**7 
    # 1 Myr
    Myr =((10**6)*u.yr).to(u.s)
    Myr_yr = (10**6)
    #Relevant timescale in seconds. Convert quiescence time in Myr to s. 100Myr = pix10^7x10^8s=3e15s
    relevant_timescale_s = quiescence_time*Myr
    
    #Read in relevant survivor info
    print("shape",np.shape(blackholes_survivors_pop))
    bh_radii = blackholes_survivors_pop[:,1]
    bh_masses = blackholes_survivors_pop[:,2]
    bh_orb_ecc = blackholes_survivors_pop[:,6]
    bh_orb_inc = blackholes_survivors_pop[:,8]
    #Make a copy of survivors array and change parameters for relaxed population
    updated_survivors = blackholes_survivors_pop
    #Note updated_surivors radii is [:,1], orb_ecc =[:,6]
    
    number_of_bh = len(bh_radii)
    print("number of BH", number_of_bh)
    
    #Relevant energy & orbital angular momentum totals among survivors (so we conserve Energy & Orb angular momentum)
    mass_in_kg = (bh_masses * cds.Msun).to(u.kg)
    v_kep = const.c/(np.sqrt(bh_radii))
    v_kep2 = v_kep**2
    # 1rg =1AU=1.5e11m for 1e8Msun
    rg = 1.5e11 * (smbh_mass/1.e8) * u.meter
    bh_radii_m = bh_radii*rg
    bh_ecc_sq = bh_orb_ecc**2 
    #Total Kinetic energy (disk)
    ke_total = np.sum(mass_in_kg*v_kep2)
    #Total Orbital Angular momentum (disk)
    l_total = np.sum(mass_in_kg*v_kep*bh_radii_m*(1-bh_ecc_sq)**0.5)
    
    #Total spheroid energy (half P.E. = GM_cluster*M_cluster/2*a_cluster)
    #Assume spheroid has mass 3e7Msun and a_cluster ~1pc
    #Normalization for stars per pc^3 (start with 10^7 or ~10^7Msun worth of stars)
    norm_stars =1.e7
    mass_sun = cds.Msun.to(u.kg)
    mass_nsc = norm_stars*mass_sun*u.kg
    mass_nsc_sq = mass_nsc**2
    # 1pc = 3.1e16m = 2e5 r_g(M_smbh/10^8Msun)
    pc = (1.0*u.pc).to(u.m)
    a_nsc = pc
    e_total_spheroid = const.G*mass_nsc_sq/(2*a_nsc)
    # Note for a fully isotropic spheroid, the total orbital angular momentum should be ~0. 
    # This is because all the orbital angular momentum vectors should net cancel 
    # (think of this as analagous to the peak~0 for a chi_eff distribution for a gas-free stellar cluster)

    print("ke total/disk (J)",ke_total/galaxy_num)
    print("l total/disk (kg m^2/s)",l_total/galaxy_num)
    print("e_total_spheroid (J)",e_total_spheroid)
    #Operate on survivor info
    
    bh_mass_ratios = smbh_mass/bh_masses
    bh_mass_ratios_sq = bh_mass_ratios**2
    ln_mass_ratios = np.log(bh_mass_ratios)
    #average parameters from disk population
    average_bh_mass = np.mean(bh_masses)
    average_bh_orb_ecc = np.mean(bh_orb_ecc)
    
    # N.B. If make average bh_orb_ecc thermal (0.7), t_disk_2bdy goes from 10Myr (e=0.01) to 100Gyr since propto e^4 !
    #average_bh_orb_ecc = 0.7

    average_bh_mass_ratio = smbh_mass/average_bh_mass
    ms_bh_orb_ecc = average_bh_orb_ecc**2
    rms_orb_ecc = np.sqrt(ms_bh_orb_ecc)
    rms_orb_inc = 0.5*rms_orb_ecc
    ms_orb_ecc_sq = ms_bh_orb_ecc**2
    power_orb_ecc = (ms_bh_orb_ecc)**(1.5)

    
    # Calc t_orb as 2pi (radii)^3/2 *rg/c (s)
    t_orb =(2*np.pi*rg/const.c)*(bh_radii)**(1.5) 
    print("t_orb(s)",t_orb)
    #orb time in years
    t_orb_yrs = t_orb/(np.pi*(10**7)*(u.s))
    print("t_orb_yrs",t_orb_yrs)
    #Mass average star = 1.0Msun
    mass_av_star = 1.0
    #Mass ratio of SMBH to average star
    mass_ratio_smbh_star = smbh_mass/mass_av_star
    Coulomb_log = np.log(mass_ratio_smbh_star)
    Coulomb_log_disk = np.log(power_orb_ecc*average_bh_mass_ratio)
    mass_ratio_smbh_star_sq = mass_ratio_smbh_star**2

    #Stellar density.
    # Assume 10^7 stars in pc^3. 
    # Assume a Bahcall-Wolf profile to start (density_index=-1.75) consistent with Panamarev & Kocsis 2023
    # Then num_stars = 10^7 (pc-3) (r/pc)^-density_index
    # at 1pc (2e5r_gM_8), num_stars = 10^7
    # at 0.1pc(2e4r_gM_8), num_stars = 10^7*56.2 *10^-3 pc^3 = 5.6e5
    # at 0.01pc(2e3r_gM_8), num_stars = 10^7*3.2e3 *10^-6pc^3 = 3.2e4
    # at 0.001pc(2e2r_gM_8), num_stars = 10^7*1.8e5 *10^-9pc^3 = 1.8e3
    
    
    #radii as a fraction of pc
    norm_radii = bh_radii*rg/pc
    
    #Number of stars within a sphere of radius radii that could interact with BH at radius radii.
    num_stars = norm_stars * (norm_radii**density_index) *(norm_radii**3)
    sqrt_num_stars = np.sqrt(num_stars)
    #Number * mass in disk. Average over a galaxy so sum all BH masses and divide by galaxy_num
    disk_mass = np.sum(bh_masses)/galaxy_num
    disk_mass_ratio = smbh_mass/disk_mass

    #Spheroidal relaxation times
    #   spheroidal 2-body relaxation time in Myr
    t_sph_2bdy_relax = t_orb_yrs * mass_ratio_smbh_star_sq *(1/Coulomb_log*num_stars)/Myr_yr
    #   spheroidal scalar resonant relaxation time in Myr
    t_sph_srr = t_orb_yrs * (2*mass_ratio_smbh_star)/Myr_yr
    #   spheroidal vector resonant relaxation time in Myr
    t_sph_vrr = t_orb_yrs * mass_ratio_smbh_star *(1/sqrt_num_stars)/Myr_yr

    #Disky relaxation times
    # 2-body disk interactions
    t_disk_2bdy_relax = t_orb_yrs * disk_mass_ratio* average_bh_mass_ratio * (ms_orb_ecc_sq/9*np.pi)*(1/Coulomb_log_disk)/Myr_yr
    # disk vrr timescale
    t_disk_vrr = t_orb_yrs *average_bh_mass_ratio *(1/rms_orb_inc)/Myr_yr
    
    
    print("quiescence time",quiescence_time)
    print("t_disk_2bdy",t_disk_2bdy_relax)
    #print("t_disk_vrr",t_disk_vrr)
    masked_sph_2bdy = np.where(t_sph_2bdy_relax < quiescence_time, t_sph_2bdy_relax,0)
    masked_sph_srr = np.where(t_sph_srr <quiescence_time,t_sph_srr,0)
    masked_sph_vrr = np.where(t_sph_vrr<quiescence_time,t_sph_vrr,0)
    masked_disk_2bdy = np.where(t_disk_2bdy_relax< quiescence_time,t_disk_2bdy_relax,0)
    masked_disk_vrr = np.where(t_disk_vrr<quiescence_time,t_disk_vrr,0)
    
    #Where are BH relaxed?
    indices_sph_2bdy = np.nonzero(masked_sph_2bdy)
    indices_sph_srr = np.nonzero(masked_sph_srr)
    indices_sph_vrr = np.nonzero(masked_sph_vrr)
    indices_disk_2bdy = np.nonzero(masked_disk_2bdy)
    indices_disk_vrr = np.nonzero(masked_disk_vrr)
    print("indices_sph_2bdy",indices_sph_2bdy)
    print("indices_sph_srr",indices_sph_srr)
    print("indices_sph_vrr",indices_sph_vrr)
    print("indices_disk_2bdy",indices_disk_2bdy)
    print("indices_disk_vrr",indices_disk_vrr)
    
    #Ratio of quiescence to relaxation time.
    ratio_q_sph_2bdy = t_sph_2bdy_relax/quiescence_time
    ratio_q_sph_srr = t_sph_srr/quiescence_time
    ratio_q_sph_vrr = t_sph_vrr/quiescence_time
    ratio_q_disk_2bdy = t_disk_2bdy_relax/quiescence_time
    ratio_q_disk_vrr = t_disk_vrr/quiescence_time

    print("ratio_q_disk_2bdy",ratio_q_disk_2bdy)


    density_factor = 1.e4*norm_radii**density_index
    #print("density factor",density_factor)
    pc3 = pc**3
    #print("pc3",pc3)
    mass_in_kg = (bh_masses * cds.Msun).to(u.kg)
    mass_in_kg_sq = (mass_in_kg)**2
    num_density = density_factor/pc3
    num_factor = 6*np.sqrt(2)/(32*np.sqrt(np.pi))
    v_kep = const.c/(np.sqrt(bh_radii))
    v_kep3 = v_kep**3
    v_kep4 = v_kep**4
    
    # Calculate change in velocity and change in semi-major axis for relaxed disky population
    # Relevant populations that are relaxed
    num_relaxed = len(indices_disk_2bdy)
    relaxed_masses = bh_masses[indices_disk_2bdy]
    relaxed_orb_ecc = bh_orb_ecc[indices_disk_2bdy]
    relaxed_radii = bh_radii[indices_disk_2bdy]
    #Factors needed
    #mass_factor = np.sqrt(relaxed_masses/average_bh_mass)
    full_mass_factor = np.sqrt(bh_masses/average_bh_mass)
    print("bh_masses",bh_masses)
    print("full mass factor",full_mass_factor)
    #dv = v_kep[indices_disk_2bdy]*(mass_factor -1)/(mass_factor + 1)
    full_dv = v_kep*(full_mass_factor -1)/(full_mass_factor +1)
    
    #vnew = v_kep[indices_disk_2bdy] + dv
    full_vnew = v_kep + full_dv
    #frac_change = 1 + (dv/v_kep[indices_disk_2bdy])
    full_frac_change = 1 +(full_dv/v_kep)
    #frac_change_factor = (1 + ((1-frac_change**2)/4))
    full_frac_change_factor = (1 + ((1-full_frac_change**2)/4))
    print("full frac change",full_frac_change_factor)
    masked_gt = np.where(full_frac_change_factor -1<0,0,full_frac_change_factor)
    indices_gt = np.nonzero(masked_gt)
    gt_masses = bh_masses[indices_gt]
    print("gt_masses",gt_masses)
    #num_retro = np.count_nonzero(masked_retros)
    num_gt = np.count_nonzero(masked_gt)
    print("num_gt",num_gt)
    #print("change_a_factor",full_frac_change_factor)
    #anew = relaxed_radii/frac_change_factor
    #Final orb_a
    full_a_new = bh_radii/full_frac_change_factor
    old_bh_radii = np.copy(updated_survivors[:,1])
    print("survivors[:,1]",old_bh_radii)
    print("factor",full_frac_change_factor)
    #print("len a,b", len(updated_survivors[:,1]),len(full_frac_change_factor))
    #Scale the relaxation by factor (quiescence time/relaxation time) where this is <1.
    # i.e. If relaxation time < quiesence  then BH relaxes fully. 
    #      If relaxation time > quiescence, scale the relaxation by the ratio of times
    factor_t_disk_2bdy = np.where(t_disk_2bdy_relax<quiescence_time,1,ratio_q_disk_2bdy)
    print("factor_t",factor_t_disk_2bdy)
    new_radii = np.copy(old_bh_radii)*full_frac_change_factor
    #*(1+(1/factor_t_disk_2bdy)/2)
    masked_new_r = np.where(new_radii - updated_survivors[:,1] <0,0,new_radii)
    indices_r = np.nonzero(masked_new_r)
    r_masses = bh_masses[indices_r]
    old_mass_radii = old_bh_radii[indices_r]
    new_mass_radii = new_radii[indices_r]
    print("r_masses",r_masses)
    print("old radii",old_mass_radii)
    print("full_frac_factor",full_frac_change_factor[indices_r])
    print("old*frac",old_mass_radii*full_frac_change_factor[indices_r])
    print("new radii",new_mass_radii)
    num_r = np.count_nonzero(masked_new_r)
    print("num_r",num_r)

    
    updated_survivors[:,1] = new_radii

    #print("vkep(indices)",v_kep[indices_disk_2bdy])
    #print("dv",dv)
    #print("vnew",vnew)
    #print("frac change",frac_change)
    #print("radii[indices]",relaxed_radii)
    #print("anew",anew)
    #print("average BH mass",average_bh_mass)
    #print("relaxed masses",relaxed_masses)
    
    #scaled_ecc is 0.1 at 1Myr. Scaling from Mikhaloff & Perets (2017) after 1Myr, initially circular pop disk ecc~0.1.
    scaled_ecc =0.1
    #From Panamarev, std dev is approx 0.1 in ecc.
    new_ecc_std_dev =0.1
    # Median rms ecc (centroid for Gaussian draw) where quiescence time is in units of Myrs. ecc_rms = 0.1 at 1Myr
    ecc_new_rms = (scaled_ecc)*(quiescence_time)**(0.25)
    #print("ecc_new_rms",ecc_new_rms)
    #Actual new ecc depend on mass scaling and draw from Gaussian
    draws_Gaussian = rng.normal(ecc_new_rms,new_ecc_std_dev,num_relaxed)
    ecc_new = (average_bh_mass/relaxed_masses)*draws_Gaussian
    #print("ecc old",bh_orb_ecc)
    #print("ecc_new",ecc_new)
    
    #Eccentricity:
    #Actually change all the ecc with scaling that goes as (1/ratio_q_..)
    #If ratio_q <1, then factor =1 if ratio_q >1 then  factor = 1/ratio_q..
    actual_draws_Gaussian = rng.normal(ecc_new_rms,new_ecc_std_dev,number_of_bh)
    #Make sure all eccentricities draws are actually positive!
    positive_draws_Gaussian =np.where(actual_draws_Gaussian<0.01,0.05,actual_draws_Gaussian)
    #Scale the eccentricity excitation by factor (quiescence time/relaxation time) where this is <1.
    # i.e. If relaxation time < quiesence  then BH relaxes fully. 
    #      If relaxation time > quiescence, scale the relaxation by the ratio of times
    factor_t_disk_2bdy = np.where(t_disk_2bdy_relax<quiescence_time,1,ratio_q_disk_2bdy)
    #print("factor_t_disk_2bdy",factor_t_disk_2bdy)
    actual_new_orb_ecc = bh_orb_ecc + ((average_bh_mass/bh_masses)*(1/factor_t_disk_2bdy)*positive_draws_Gaussian)
    #Cap the eccentricity (since Gaussian draw can be very large!)
    actual_new_orb_ecc = np.where(actual_new_orb_ecc>0.999,0.999,actual_new_orb_ecc)
    #print("actual_new_orb_ecc",actual_new_orb_ecc)
    #Update survivors array
    #updated_survivors[indices_disk_2bdy,1] = anew
    updated_survivors[:,6] = actual_new_orb_ecc
    
    #Orbital Inclination:
    # Update the inclination angles based on t_sph_vrr
    # The spheroid of stars is an efficient randomizer of the direction (but not magnitude)  of orb ang mom.
    # Orb_inc is in radians (2pi rad =360 degrees; pi rad = 180deg)
    # If fully randomized, ie t_relax_sph_vrr < t_quiescence draw from [-pi,pi] radians.
    # Scale by t_quiescence/t_sph_vrr as 
    # draw from [-lim,lim] where lim=pi*(t_quiescence/t_relax_sph_vrr) for t_relax_sph_vrr >t_quiescence
    factor_t_sph_vrr = np.where(ratio_q_sph_vrr<1,1,ratio_q_sph_vrr)
    lim = np.pi/factor_t_sph_vrr
    #print("lim",lim)
    rand_lim = rng.uniform(-lim,lim)
    masked_rand_lim = np.where(rand_lim<0,0,rand_lim)
    num_pos=np.count_nonzero(masked_rand_lim)
    #print("num_pos",num_pos)
    #print("rand_lim",rand_lim)
    actual_draws_inclination = bh_orb_inc + rand_lim
    #normalized_inclination = np.where(actual_draws_inclination > np.pi,)
    #print("old inc", bh_orb_inc)
    masked_retros = np.where(np.abs(bh_orb_inc - np.pi)<0.14,bh_orb_inc,0)
    masked_pros = np.where(np.abs(bh_orb_inc -0)<0.14,bh_orb_inc,0)
    indices_retro = np.nonzero(masked_retros)
    #print("indices retro",indices_retro)
    num_retro = np.count_nonzero(masked_retros)
    num_pros = np.count_nonzero(masked_pros)
    #print("num retro",num_retro)  
    #print("num pros",num_pros)
    #print("average num pro/galaxy =", num_pros/galaxy_num)  
    temp_updated_survivors = np.abs(actual_draws_inclination)
    #Normalize to <3.14 orb. inc.
    actual_updated_survivors = np.where(temp_updated_survivors>np.pi,np.pi+(np.pi-temp_updated_survivors),temp_updated_survivors)
    updated_survivors[:,8] = actual_updated_survivors
    updated_retro_mask =np.where(np.abs(actual_draws_inclination - np.pi)<0.14,actual_draws_inclination,0)
    new_num_retro = np.count_nonzero(updated_retro_mask)
    #print("new num retro",new_num_retro)
    
    #Compare energy and ang mom
    v_kep_new = const.c/(np.sqrt(new_radii))
    v_kep_new2 = v_kep_new**2 
    new_radii_m = new_radii*rg
    new_ecc_sq = actual_new_orb_ecc**2 
    #Total Kinetic energy (disk)
    ke_new_total = np.sum(mass_in_kg*v_kep_new2)
    #Total Orbital Angular momentum (disk)
    l_new_total = np.sum(mass_in_kg*v_kep_new*new_radii_m*(1-new_ecc_sq)**0.5)
    #print("v_kep_new",v_kep_new)
    #print("new_radii_m",new_radii_m)
    #print("(1-ecc^2)^(1/2)",(1-new_ecc_sq)**0.5)
    #print("ecc^2",new_ecc_sq)
    #print("1-ecc^2",(1-new_ecc_sq))
    print("ke total/disk (J)",ke_total/galaxy_num)
    print("l total/disk (kg m^2/s)",l_total/galaxy_num)
    print("e_total_spheroid (J)",e_total_spheroid)
    print("ke new total/disk (J)",ke_new_total/galaxy_num)
    print("l new total/disk (kg m^2/s)",l_new_total/galaxy_num)
    print("en ratio",ke_new_total/ke_total)
    print("l ratio",l_new_total/l_total)

    return(updated_survivors)

def bh_near_smbh(
        smbh_mass,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        timestep_duration_yr,
        inner_disk_outer_radius,
        disk_inner_stable_circ_orb,
        ):
    """Evolve semi-major axis of single BH near SMBH according to Peters64

    Test whether there are any BH near SMBH. 
    Flag if anything within min_safe_distance (default=50r_g) of SMBH.
    Time to decay into SMBH can be parameterized from Peters(1964) as:
    .. math:: t_{gw} =38Myr (1-e^2)(7/2) (a/50r_{g})^4 (M_{smbh}/10^8M_{sun})^3 (m_{bh}/10M_{sun})^{-1}

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
    inner_disk_outer_radius : float
        Outer radius of the inner disk [r_{g,SMBH}]
    disk_inner_stable_circ_orb : float
        Innermost stable circular orbit around the SMBH [r_{g,SMBH}]

    Returns
    -------
    disk_bh_pro_orbs_a : numpy.ndarray
        Semi-major axis [r_{g,SMBH}] of prograde singleton BH at end of timestep assuming only GW evolution
    """
    num_bh = disk_bh_pro_orbs_a.shape[0]
    # Calculate min_safe_distance in r_g
    min_safe_distance = max(disk_inner_stable_circ_orb, inner_disk_outer_radius)

    # Create a new bh_pro_orbs array
    new_disk_bh_pro_orbs_a = disk_bh_pro_orbs_a.copy()
    # Estimate the eccentricity factor for orbital decay time
    ecc_factor_arr = (1.0 - (disk_bh_pro_orbs_ecc)**(2.0))**(7/2)
    # Estimate the orbital decay time of each bh
    decay_time_arr = time_of_orbital_shrinkage(
        smbh_mass*u.solMass,
        disk_bh_pro_masses*u.solMass,
        si_from_r_g(smbh_mass*u.solMass, disk_bh_pro_orbs_a),
        0*u.m,
    )
    # Estimate the number of timesteps to decay
    decay_timesteps = decay_time_arr.to('yr').value / timestep_duration_yr
    # Estimate decrement
    decrement_arr = (1.0-(1./decay_timesteps))
    # Fix decrement
    decrement_arr[decay_timesteps == 0.] = 0.
    # Estimate new location
    new_location_r_g = decrement_arr * disk_bh_pro_orbs_a
    # Check location
    new_location_r_g[new_location_r_g < 1.] = 1.
    # Only update when less than min_safe_distance
    new_disk_bh_pro_orbs_a[disk_bh_pro_orbs_a < min_safe_distance] = new_location_r_g

    assert np.isfinite(new_disk_bh_pro_orbs_a).all(), \
        "Finite check failure: new_disk_bh_pro_orbs_a"

    return new_disk_bh_pro_orbs_a
