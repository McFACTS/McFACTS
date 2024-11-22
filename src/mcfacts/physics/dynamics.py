"""Module for handling dynamical interactions.

Contains multiple functions which are each mocked up versions of a
dynamical mechanism. Of varying fidelity to reality. Also contains
GW orbital evolution for BH in the inner disk, which should probably
move elsewhere.
"""
import numpy as np
import scipy

from astropy import units as astropy_units

from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.point_masses import time_of_orbital_shrinkage
from mcfacts.physics.point_masses import si_from_r_g


def circular_singles_encounters_prograde(
        smbh_mass,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong
        ):
    """"Adjust orb ecc due to encounters between 2 single circ pro BH

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
        :math:`T_orb =2\pi \sqrt(a^3/GM_{smbh})`
        and
        .. math::
        a^3/GM_{smbh} = (10^3r_g)^3/GM_{smbh} = 10^9 (a/10^3r_g)^3 (GM_{smbh}/c^2)^3/GM_{smbh} \\
                    = 10^9 (a/10^3r_g)^3 (G M_{smbh}/c^3)^2 

        So
        .. math::
            T_orb   = 2\pi 10^{4.5} (a/10^3r_g)^{3/2} GM_{smbh}/c^3 \\
                    = 2\pi 10^{4.5} (a/10^3r_g)^{3/2} (6.7e-11*2e38/(3e8)^3) \\
                    = 2\pi 10^{4.5} (a/10^3r_g)^{3/2} (13.6e27/27e24) \\
                    = \pi 10^{7.5}  (a/10^3r_g)^{3/2} \\
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
    ecc_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc >= disk_bh_pro_orb_ecc_crit).nonzero()[0]

    # Find their locations and masses
    circ_prograde_population_locations = disk_bh_pro_orbs_a[circ_prograde_population_indices]
    circ_prograde_population_masses = disk_bh_pro_masses[circ_prograde_population_indices]
    # Ecc locs
    ecc_prograde_population_locations = disk_bh_pro_orbs_a[ecc_prograde_population_indices]
    ecc_prograde_population_masses = disk_bh_pro_masses[ecc_prograde_population_indices]

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = scipy.constants.pi*((circ_prograde_population_locations)**(1.5))*(2.e30*smbh_mass*scipy.constants.G)/(scipy.constants.c**(3.0)*3.15e7) 
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    num_poss_ints = 0
    num_encounters = 0
    if len(circ_prograde_population_locations) > 0:
        for i in range(0, len(circ_prograde_population_locations)):
            for j in range(0, len(ecc_prograde_population_locations)):
                if circ_prograde_population_locations[i] < ecc_orb_max[j] and circ_prograde_population_locations[i] > ecc_orb_min[j]:
                    # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                    # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                    temp_bin_mass = circ_prograde_population_masses[i] + ecc_prograde_population_masses[j]
                    bh_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                    mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
                    prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                    prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                    if prob_enc_per_timestep > 1:
                        prob_enc_per_timestep = 1
                    random_uniform_number = rng.uniform(size=1)
                    if random_uniform_number < prob_enc_per_timestep:
                        indx_array = circ_prograde_population_indices
                        num_encounters = num_encounters + 1
                        # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                        # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                        # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                        if disk_bh_pro_orbs_ecc[indx_array[i]] <= disk_bh_pro_orb_ecc_crit:
                            disk_bh_pro_orbs_ecc[indx_array[i]] = delta_energy_strong
                            disk_bh_pro_orbs_a[indx_array[i]] = disk_bh_pro_orbs_a[indx_array[i]]*(1.0 + delta_energy_strong)
                            disk_bh_pro_orbs_ecc[j] = disk_bh_pro_orbs_ecc[j]*(1 - delta_energy_strong)
                            disk_bh_pro_orbs_a[j] = disk_bh_pro_orbs_a[j]*(1 - delta_energy_strong)                        
                    num_poss_ints = num_poss_ints + 1
            num_poss_ints = 0
            num_encounters = 0

    # Check finite
    assert np.isfinite(disk_bh_pro_orbs_a).all(), \
        "Finite check failed for disk_bh_pro_orbs_a"
    assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for disk_bh_pro_orbs_ecc"
    return (disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)


def circular_binaries_encounters_ecc_prograde(
        smbh_mass,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        blackholes_binary,
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
    bindex : int
        Number of binaries at time of function call

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
    solar_mass = astropy_units.solMass.to("kg")
    rg_in_meters = scipy.constants.G * (solar_mass*smbh_mass) / (scipy.constants.c ** 2.0)
    # eccentricity correction--do not let ecc>=1, catch and reset to 1-epsilon
    epsilon = 1e-8

    # Set up other values we need
    bin_masses = blackholes_binary.mass_1 + blackholes_binary.mass_2
    bin_velocities = scipy.constants.c / np.sqrt(blackholes_binary.bin_orb_a)
    bin_binding_energy = scipy.constants.G * (solar_mass ** 2) * blackholes_binary.mass_1 * blackholes_binary.mass_2 / (blackholes_binary.bin_sep * rg_in_meters)
    bin_orbital_times = 3.15 * (smbh_mass / 1.e8) * ((blackholes_binary.bin_orb_a / 1.e3) ** 1.5)
    bin_orbits_per_timestep = timestep_duration_yr/bin_orbital_times

    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc >= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find their locations and masses
    ecc_prograde_population_locations = disk_bh_pro_orbs_a[ecc_prograde_population_indices]
    ecc_prograde_population_masses = disk_bh_pro_masses[ecc_prograde_population_indices]
    ecc_prograde_population_eccentricities = disk_bh_pro_orbs_ecc[ecc_prograde_population_indices]
    # Find min and max radii around SMBH for eccentric orbiters
    ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices] * (1.0-disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices] * (1.0+disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    # Keplerian velocity of ecc prograde orbiter around SMBH (=c/sqrt(a/r_g))
    ecc_velocities = scipy.constants.c/np.sqrt(ecc_prograde_population_locations)

    if blackholes_binary.num == 0:
        return (blackholes_binary)

    # Create array of random numbers for the chances of encounters
    chances = rng.uniform(size=(blackholes_binary.num, len(ecc_prograde_population_locations)))

    # For each binary in blackholes_binary
    for i in range(0, blackholes_binary.num):
        # We compare each single BH to that binary
        for j in range(0, len(ecc_prograde_population_locations)):
            # If binary com orbit lies inside eccentric orbit [min,max] radius
            # i.e. if R_m3_minimum lie inside R_bin_maximum and does R_m3_max lie outside R_bin_minimum
            if (1.0-blackholes_binary.bin_orb_ecc[i]) * blackholes_binary.bin_orb_a[i] < ecc_orb_max[j] and (1.0+blackholes_binary.bin_orb_ecc[i]) * blackholes_binary.bin_orb_a[i] > ecc_orb_min[j]:

                # Make a temporary Hill sphere treating binary + ecc interloper as a 'binary' = M_1+M_2+M_3
                # r_h = a_circ1(temp_bin_mass/3mass_smbh)^1/3 so prob_enc/orb = mass_ratio^1/3/pi

                temp_bin_mass = bin_masses[i] + ecc_prograde_population_masses[j]
                bh_smbh_mass_ratio = temp_bin_mass / (3.0 * smbh_mass)
                mass_ratio_factor = bh_smbh_mass_ratio ** (1./3.)
                prob_orbit_overlap = (1. / scipy.constants.pi) * mass_ratio_factor
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
                        blackholes_binary.bin_sep[i] = blackholes_binary.bin_sep[i] * (1 - delta_energy_strong)  # BUG check if should be de_strong
                        blackholes_binary.bin_ecc[i] = blackholes_binary.bin_ecc[i] * (1 + delta_energy_strong)  # BUG should be de_strong
                        blackholes_binary.bin_orb_ecc[i] = blackholes_binary.bin_orb_ecc[i] * (1 + delta_energy_strong)
                        # Change interloper parameters; increase a_ecc, increase e_ecc
                        ecc_prograde_population_locations[j] = ecc_prograde_population_locations[j] * (1 + delta_energy_strong)
                        ecc_prograde_population_eccentricities[j] = ecc_prograde_population_eccentricities[j] * (1 + delta_energy_strong)

                    if hard < 0:
                        # Binary is soft w.r.t. interloper
                        # Check to see if binary is ionized
                        # Change binary parameters; incr bin separation, decr ecc around com, incr orb_ecc
                        blackholes_binary.bin_sep[i] = blackholes_binary.bin_sep[i] * (1 + delta_energy_strong)  # BUG check if should be de_strong
                        blackholes_binary.bin_ecc[i] = blackholes_binary.bin_ecc[i] * (1 - delta_energy_strong)  # BUG check if should be de_strong
                        blackholes_binary.bin_orb_ecc[i] = blackholes_binary.bin_orb_ecc[i] * (1 + delta_energy_strong)
                        # Change interloper parameters; decrease a_ecc, decrease e_ecc
                        ecc_prograde_population_locations[j] = ecc_prograde_population_locations[j] * (1 - delta_energy_strong)
                        ecc_prograde_population_eccentricities[j] = ecc_prograde_population_eccentricities[j] * (1 - delta_energy_strong)

                    # Catch if bin_ecc or bin_orb_ecc >= 1
                    if blackholes_binary.bin_ecc[i] >= 1:
                        blackholes_binary.bin_ecc[i] = 1.0 - epsilon
                    if blackholes_binary.bin_orb_ecc[i] >= 1:
                        blackholes_binary.bin_orb_ecc[i] = 1.0 - epsilon

    # TODO: ALSO return new array of singletons with changed params.

    # Check finite
    assert np.isfinite(blackholes_binary.bin_sep).all(), \
        "Finite check failure: bin_separations"
    assert np.isfinite(blackholes_binary.bin_orb_ecc).all(), \
        "Finite check failure: bin_orbital_eccentricities"
    assert np.isfinite(blackholes_binary.bin_ecc).all(), \
        "Finite check failure: bin_eccentricities"
    assert np.sum(blackholes_binary.bin_ecc > 1) == 0, "bin_ecc has values greater than 1"

    return (blackholes_binary)


def circular_binaries_encounters_circ_prograde(
        smbh_mass,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_orbs_ecc,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        blackholes_binary,
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
    bindex : int
        Number of binaries at time of function call

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
    solar_mass = astropy_units.solMass.to("kg")
    rg_in_meters = scipy.constants.G * (solar_mass * smbh_mass) / (scipy.constants.c ** 2.0)

    # Magnitude of energy change to drive binary to merger in ~2 interactions in a strong encounter. Say de_strong=0.9
    # de_strong here refers to the perturbation of the binary around its center of mass
    # The energy in the exchange is assumed to come from the binary binding energy around its c.o.m.
    # delta_energy_strong (read into this module) refers to the perturbation of the orbit of the binary c.o.m. around the SMBH, which is not as strongly perturbed (we take an 'average' perturbation) 
    de_strong = 0.9
    # eccentricity correction--do not let ecc>=1, catch and reset to 1-epsilon
    epsilon = 1e-8

    # Set up arrays for later
    bin_masses = blackholes_binary.mass_1 + blackholes_binary.mass_2
    bin_velocities = scipy.constants.c/np.sqrt(blackholes_binary.bin_orb_a)
    bin_orbital_times = 3.15 * (smbh_mass / 1.e8) * ((blackholes_binary.bin_orb_a / 1.e3) ** 1.5)
    bin_orbits_per_timestep = timestep_duration_yr / bin_orbital_times
    bin_binding_energy = scipy.constants.G * (solar_mass ** 2.0) * blackholes_binary.mass_1 * blackholes_binary.mass_2 / (blackholes_binary.bin_sep * rg_in_meters)

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
    circ_velocities = scipy.constants.c/np.sqrt(circ_prograde_population_locations)

    if (blackholes_binary.num == 0):
        return (blackholes_binary)

    # Set up random numbers
    chances = rng.uniform(size=(blackholes_binary.num, len(circ_prograde_population_locations)))

    for i in range(0, blackholes_binary.num):
        for j in range(0, len(circ_prograde_population_locations)):
            # If binary com orbit lies inside circ orbit [min,max] radius
            # i.e. does R_m3_minimum lie inside R_bin_maximum and does R_m3_max lie outside R_bin_minimum
            if (1.0-blackholes_binary.bin_orb_ecc[i]) * blackholes_binary.bin_orb_a[i] < ecc_orb_max[j] and (1.0+blackholes_binary.bin_orb_ecc[i]) * blackholes_binary.bin_orb_a[i] > ecc_orb_min[j]:
                # Make a temporary Hill sphere treating binary + ecc interloper as a 'binary' = M_1+M_2+M_3
                # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                temp_bin_mass = bin_masses[i] + circ_prograde_population_masses[j]
                bh_smbh_mass_ratio = temp_bin_mass/(3.0 * smbh_mass)
                mass_ratio_factor = (bh_smbh_mass_ratio ** (1./3.))
                prob_orbit_overlap = (1. / scipy.constants.pi) * mass_ratio_factor
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
                        blackholes_binary.bin_sep[i] = blackholes_binary.bin_sep[i] * (1 - de_strong)
                        blackholes_binary.bin_ecc[i] = blackholes_binary.bin_ecc[i] * (1 + de_strong)
                        blackholes_binary.bin_orb_ecc[i] = blackholes_binary.bin_orb_ecc[i]*(1 + delta_energy_strong)
                        # Change interloper parameters; increase a_ecc, increase e_ecc
                        circ_prograde_population_locations[j] = circ_prograde_population_locations[j] * (1 + delta_energy_strong)
                        circ_prograde_population_eccentricities[j] = circ_prograde_population_eccentricities[j] * (1 + delta_energy_strong)

                    if hard < 0:
                        # Binary is soft w.r.t. interloper
                        # Check to see if binary is ionized
                        # Change binary parameters; incr bin separation, decr ecc around com, incr orb_ecc
                        blackholes_binary.bin_sep[i] = blackholes_binary.bin_sep[i] * (1 + delta_energy_strong)  # BUG should this be de_strong?
                        blackholes_binary.bin_ecc[i] = blackholes_binary.bin_ecc[i] * (1 - delta_energy_strong)  # BUG should this be de_strong?
                        blackholes_binary.bin_orb_ecc[i] = blackholes_binary.bin_orb_ecc[i] * (1 + delta_energy_strong)
                        # Change interloper parameters; decrease a_ecc, decrease e_ecc
                        circ_prograde_population_locations[j] = circ_prograde_population_locations[j] * (1 - delta_energy_strong)
                        circ_prograde_population_eccentricities[j] = circ_prograde_population_eccentricities[j] * (1 - delta_energy_strong)

                    # Catch where bin_orb_ecc and bin_ecc >= 1
                    if (blackholes_binary.bin_ecc[i] >= 1):
                        blackholes_binary.bin_ecc[i] = 1.0 - epsilon
                    if (blackholes_binary.bin_orb_ecc[i] >= 1):
                        blackholes_binary.bin_orb_ecc = 1.0 - epsilon
    return (blackholes_binary)


def bin_spheroid_encounter(
        smbh_mass,
        timestep_duration_yr,
        blackholes_binary,
        time_passed,
        nsc_bh_imf_powerlaw_index,
        delta_energy_strong,
        nsc_spheroid_normalization
        ):
    """Perturb orbits due to encounters with spheroid (NSC) objects

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
    solar_mass = astropy_units.solMass.to("kg")
    # Magnitude of energy change to drive binary to merger in ~2 interactions in a strong encounter. Say de_strong=0.9
    # de_strong here refers to the perturbation of the binary around its center of mass
    # The energy in the exchange is assumed to come from the binary binding energy around its c.o.m.
    # delta_energy_strong refers to the perturbation of the orbit of the binary c.o.m. around the SMBH, which is not as strongly perturbed (we take an 'average' perturbation) 
    de_strong = 0.9
    # eccentricity correction--do not let ecc>=1, catch and reset to 1-epsilon
    epsilon = 1e-8
    # Spheroid normalization to allow for non-ideal NSC (cored/previous AGN episodes/disky population concentration/whatever)
    rg_in_meters = scipy.constants.G * (solar_mass * smbh_mass) / (scipy.constants.c ** 2.0)

    # Set up binary properties we need for later
    bin_mass = blackholes_binary.mass_1 + blackholes_binary.mass_2
    bin_velocities = scipy.constants.c / np.sqrt(blackholes_binary.bin_orb_a)
    bin_binding_energy = scipy.constants.G * (solar_mass ** 2) * blackholes_binary.mass_1 * blackholes_binary.mass_2 / (blackholes_binary.bin_sep * rg_in_meters)

    # Calculate encounter rate for each binary based on bin_orb_a, binary size, and time_passed
    # Set up array of encounter rates filled with -1
    enc_rate = np.full(blackholes_binary.num, -1.5)

    # Set encounter rate if bin_orb_a < crit_radius
    enc_rate[(blackholes_binary.bin_orb_a < crit_radius) & (time_passed <= crit_time)] = 0.02 * (nsc_spheroid_normalization / 0.1) * (1.0 - (time_passed / 1.e6)) * ((blackholes_binary.bin_sep[(blackholes_binary.bin_orb_a < crit_radius) & (time_passed <= crit_time)] / dist_in_rg_m8) ** 2.0) / ((timestep_duration_yr / 1.e4) * (blackholes_binary.bin_orb_a[(blackholes_binary.bin_orb_a < crit_radius) & (time_passed <= crit_time)] / 1.e3))
    enc_rate[(blackholes_binary.bin_orb_a < crit_radius) & (time_passed > crit_time)] = 0.0

    # Set encounter rate if bin_orb_a > crit_radius
    enc_rate[(blackholes_binary.bin_orb_a > crit_radius)] = 0.002 * (nsc_spheroid_normalization / 0.1) * ((blackholes_binary.bin_sep / dist_in_rg_m8) ** 2.0) / ((timestep_duration_yr / 1.e4) * (blackholes_binary.bin_orb_a / 1.e4) * np.sqrt(time_passed / 1.e4))

    # If enc_rate still has negative values throw error
    if (np.sum(enc_rate < 0) > 0):
        print("enc_rate",enc_rate)
        raise RuntimeError("enc_rate not being set in bin_spheroid_encounter")

    # If bin_orb_a == crit_radius throw error
    if (np.sum(blackholes_binary.bin_orb_a == crit_radius) > 0):
        print("SMBH mass:", smbh_mass)
        print("bin_orb_a:", blackholes_binary.bin_orb_a[blackholes_binary.bin_orb_a == crit_radius])
        print("crit_radius:", crit_radius)
        raise RuntimeError("Unrecognized bin_orb_a")

    # Based on estimated encounter rate, calculate if binary actually has a spheroid encounter
    chances_of_encounter = rng.uniform(size=blackholes_binary.num)
    num_encounters = np.sum(chances_of_encounter < enc_rate)

    if (num_encounters > 0):

        # Set up arrays for changed blackholes_binary parameters
        bin_orb_a = blackholes_binary.bin_orb_a[chances_of_encounter < enc_rate].copy()
        bin_sep = blackholes_binary.bin_sep[chances_of_encounter < enc_rate].copy()
        bin_ecc = blackholes_binary.bin_ecc[chances_of_encounter < enc_rate].copy()
        bin_orb_ecc = blackholes_binary.bin_orb_ecc[chances_of_encounter < enc_rate].copy()
        bin_orb_inc = blackholes_binary.bin_orb_inc[chances_of_encounter < enc_rate].copy()

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
        velocity_3 = scipy.constants.c / np.sqrt(radius_3)
        relative_velocities = np.abs(bin_velocities[chances_of_encounter < enc_rate] - velocity_3)
        ke_3 = 0.5 * mass_3 * solar_mass * (relative_velocities ** 2.0)

        # Compare orbital angular momentum for interloper and binary
        # Ratio of L3/Lbin =(m3/M_bin)*sqrt(R3/R_com)
        L_ratio = (mass_3 / bin_mass[chances_of_encounter < enc_rate]) * np.sqrt(radius_3 / bin_orb_a)

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
        i3 = rng.randint(low=excluded_angles, high=360-excluded_angles)
        # Convert i3 to radians
        i3_rad = np.radians(i3)

        # Ionize/soften/harden binary if appropriate
        hard = bin_binding_energy[chances_of_encounter < enc_rate] - ke_3
        # Create mask for hard and soft
        mask_hard = hard > 0
        mask_soft = hard < 0

        # If hard > 0 binary is hard wrt interloper
        # Change binary parameters: decrease separation, increase ecc around bin_orb_a and orb_ecc
        bin_sep[mask_hard] = bin_sep[mask_hard] * (1-de_strong)
        bin_ecc[mask_hard] = bin_ecc[mask_hard] * (1 + de_strong)
        bin_orb_ecc[mask_hard] = bin_orb_ecc[mask_hard] * (1 + delta_energy_strong)
        # Ignore interloper parameters, since just drawing randomly from IMF population

        # If hard < 0 binary is soft wrt interloper
        # Change binary parameters: increase separation, decrease ecc around bin_orb_a, increase orb_ecc
        bin_sep[mask_soft] = bin_sep[mask_soft] * (1 + delta_energy_strong)  # BUG this should be de_strong
        bin_ecc[mask_soft] = bin_ecc[mask_soft] * (1 - delta_energy_strong)  # BUG this should be de_strong
        bin_orb_ecc[mask_soft] = bin_orb_ecc[mask_soft] * (1 + delta_energy_strong)

        # Catch if bin_ecc or bin_orb_ecc >= 1
        bin_ecc[bin_ecc >= 1.0] = 1.0 - epsilon
        bin_orb_ecc[bin_orb_ecc >= 1.0] = 1.0 - epsilon

        # New angle of binary wrt disk (in radians)
        bin_orb_inc[(L_ratio < 1)] = bin_orb_inc[(L_ratio < 1)] + L_ratio[L_ratio < 1] * (i3_rad[L_ratio < 1]/2.0)
        bin_orb_inc[(L_ratio > 1)] = bin_orb_inc[(L_ratio > 1)] + (1./L_ratio[L_ratio > 1]) * (i3_rad[L_ratio > 1]/2.0)

        blackholes_binary.bin_sep[chances_of_encounter < enc_rate] = bin_sep
        blackholes_binary.bin_ecc[chances_of_encounter < enc_rate] = bin_ecc
        blackholes_binary.bin_orb_ecc[chances_of_encounter < enc_rate] = bin_orb_ecc
        blackholes_binary.bin_orb_inc[chances_of_encounter < enc_rate] = bin_orb_inc

    return (blackholes_binary)


def bin_recapture(blackholes_binary, timestep_duration_yr):
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

    idx_gtr_0 = blackholes_binary.bin_orb_inc > 0

    if (idx_gtr_0.shape[0] == 0):
        return (blackholes_binary)

    bin_orb_inc = blackholes_binary.bin_orb_inc[idx_gtr_0]
    bin_mass = blackholes_binary.mass_1[idx_gtr_0] + blackholes_binary.mass_2[idx_gtr_0]
    bin_orb_a = blackholes_binary.bin_orb_a[idx_gtr_0]

    less_crit_inc1_mask = bin_orb_inc < crit_inc1
    bwtwn_crit_inc1_inc2_mask = (bin_orb_inc > crit_inc1) & (bin_orb_inc < crit_inc2)

    # is bin orbital inclination <5deg in SG disk?
    bin_orb_inc[less_crit_inc1_mask] = bin_orb_inc[less_crit_inc1_mask] * (1. - ((timestep_duration_yr/1e6) * (bin_mass[less_crit_inc1_mask] / 10.) * (bin_orb_a[less_crit_inc1_mask] / 1.e4)))
    bin_orb_inc[bwtwn_crit_inc1_inc2_mask] = bin_orb_inc[bwtwn_crit_inc1_inc2_mask] * (1. - ((timestep_duration_yr/5.e7) * (bin_mass[bwtwn_crit_inc1_inc2_mask] / 10.) * (bin_orb_a[bwtwn_crit_inc1_inc2_mask] / 1.e4)))

    blackholes_binary.bin_orb_inc[idx_gtr_0] = bin_orb_inc

    return (blackholes_binary)


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
    min_safe_distance = max(disk_inner_stable_circ_orb,inner_disk_outer_radius)

    # Create a new bh_pro_orbs array
    new_disk_bh_pro_orbs_a = disk_bh_pro_orbs_a.copy()
    # Estimate the eccentricity factor for orbital decay time
    ecc_factor_arr = (1.0 - (disk_bh_pro_orbs_ecc)**(2.0))**(7/2)
    # Estimate the orbital decay time of each bh
    decay_time_arr = time_of_orbital_shrinkage(
        smbh_mass*astropy_units.solMass,
        disk_bh_pro_masses*astropy_units.solMass,
        si_from_r_g(smbh_mass*astropy_units.solMass, disk_bh_pro_orbs_a),
        0*astropy_units.m,
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

    return new_disk_bh_pro_orbs_a
