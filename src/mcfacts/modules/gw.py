"""
Module for calculating the gw strain and freq of a binary and handling simple GR orbital evolution (Peters 1964)

Contain functions for orbital evolution and converting between
    units of r_g and SI units
"""
import numpy as np
from astropy import units as u, constants as const
from numpy.random import Generator

from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBinaryBlackHoleArray, AGNBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities import unit_conversion, peters


def orbital_separation_evolve(mass_1, mass_2, sep_initial, evolve_time):
    """Calculates the final separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy.units.quantity.Quantity
        Mass of object 1
    mass_2 : astropy.units.quantity.Quantity
        Mass of object 2
    sep_initial : astropy.units.quantity.Quantity
        Initial separation of two bodies
    evolve_time : astropy.units.quantity.Quantity
        Time to evolve GW orbit

    Returns
    -------
    sep_final : astropy.units.quantity.Quantity
        Final separation [m] of two bodies
    """
    # Calculate c and G in SI
    c = const.c.to('m/s').value
    G = const.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_initial = sep_initial.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const_g_c = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const_g_c * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_initial ** 4) - (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_final = np.zeros_like(sep_initial)
    sep_final[quantity > 0] = np.sqrt(np.sqrt(quantity[quantity > 0]))

    assert np.isfinite(sep_final).all(), \
        "Finite check failure: sep_final"
    assert np.all(sep_final > 0), \
        "sep_final contains values <= 0"

    return sep_final * u.m


def orbital_separation_evolve_reverse(mass_1, mass_2, sep_final, evolve_time):
    """Calculates the initial separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy.units.quantity.Quantity
        Mass of object 1
    mass_2 : astropy.units.quantity.Quantity
        Mass of object 2
    sep_final : astropy.units.quantity.Quantity
        Final separation of two bodies
    evolve_time : astropy.units.quantity.Quantity
        Time to evolve GW orbit

    Returns
    -------
    sep_initial : astropy.units.quantity.Quantity
        Initial separation [m] of two bodies
    """
    # Calculate c and G in SI
    c = const.c.to('m/s').value
    G = const.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_final = sep_final.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const_g_c = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const_g_c * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_final ** 4) + (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_initial = np.sqrt(np.sqrt(quantity))

    assert np.isfinite(sep_initial).all(), \
        "Finite check failure: sep_initial"
    assert np.all(sep_initial > 0), \
        "sep_initial contains values <= 0"

    return sep_initial * u.m


def evolve_emri_gw(mass, orb_a, timestep_duration_yr, old_gw_freq, smbh_mass, agn_redshift):
    """Evaluates the EMRI gravitational wave frequency and strain at the end of each timestep_duration_yr

    Parameters
    ----------
    blackholes_inner_disk : AGNBlackHole
        Parameters of black holes in the inner disk
    timestep_duration_yr : float
        Length of timestep [yr]
    old_gw_freq : numpy.ndarray
        Previous GW frequency [Hz] with :obj:`float` type
    smbh_mass : float
        Mass [M_sun] of the SMBH
    agn_redshift : float
        Redshift [unitless] of the AGN
    """

    old_gw_freq = old_gw_freq * u.Hz

    # If number of EMRIs has grown since last timestep_duration_yr, add a new component to old_gw_freq to carry out dnu/dt calculation
    # while (blackholes_inner_disk.num < len(old_gw_freq)):
    #     old_gw_freq = np.delete(old_gw_freq, 0)
    # while blackholes_inner_disk.num > len(old_gw_freq):
    #     old_gw_freq = np.append(old_gw_freq, (9.e-7) * u.Hz)

    char_strain, strain, nu_gw = peters.gw_strain_freq(mass_1=smbh_mass,
                                        mass_2=mass,
                                        obj_sep=orb_a,
                                        timestep_duration_yr=timestep_duration_yr,
                                        old_gw_freq=old_gw_freq,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=1)

    return char_strain, nu_gw


def normalize_tgw(smbh_mass, inner_disk_outer_radius):
    """Normalizes Gravitational wave timescale.

    Calculate the normalization for timescale of a merger (in s) due to GW emission.
    From Peters(1964):

    .. math:: t_{gw} \approx (5/256)* c^5/G^3 *a_b^4/(M_{b}^{2}mu_{b})
    assuming ecc=0.0.

    For a_b in units of r_g=GM_smbh/c^2 we find

    .. math:: t_{gw}=(5/256)*(G/c^3)*(a/r_g)^{4} *(M_s^4)/(M_b^{2}mu_b)

    Put bin_mass_ref in units of 10Msun (is a reference mass).
    reduced_mass in units of 2.5Msun.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    inner_disk_outer_radius : float
        Outer radius of the inner disk [r_g]

    Returns
    -------
    time_gw_normalization : float
        Normalization to gravitational wave timescale [s]
    """

    bin_mass_ref = 10.0
    '''
    G = const.G
    c = const.c
    mass_sun = const.M_sun
    year = 3.1536e7
    reduced_mass = 2.5
    norm = (5.0/256.0)*(G/(c**(3)))*(smbh_mass**(4))*mass_sun/((bin_mass_ref**(2))*reduced_mass)
    time_gw_normalization = norm/year
    '''
    time_gw_normalization = peters.time_of_orbital_shrinkage(
        smbh_mass * u.solMass,
        bin_mass_ref * u.solMass,
        unit_conversion.si_from_r_g(smbh_mass * u.solMass, inner_disk_outer_radius),
        0 * u.m,
    )
    return time_gw_normalization.si.value


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
    also eccentricity

    Test whether there are any BH near SMBH.
    Flag if anything within min_safe_distance (default=50r_g) of SMBH.
    Time to decay into SMBH can be parameterized from Peters(1964) as:
    .. math:: t_{gw} =38Myr (1-e^2)(7/2) (a/50r_{g})^4 (M_{smbh}/10^8M_{sun})^3 (m_{bh}/10M_{sun})^{-1}
    Time to eccentricity decay to zero from Peters(1964) as an annoying piecewise function:
    .. math:: t_{e_0} =t_{gw} * f(e_0) where f(e_0)=(1-e_0^2)^4/(1+ (121/304)e_0^2)^(870/2299) if e_0<0.8
    .. math:: f(e_0) = (768/425) * (1-e_0^2)^3.5 if e_0>0.95
    .. math:: f(e_0) = some other function if 0.8 < e_0 < 0.95

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
    ecc_factor_arr = (1.0 - (disk_bh_pro_orbs_ecc) ** (2.0)) ** (7 / 2)
    # Estimate the orbital decay time of each bh
    decay_time_arr = peters.time_of_orbital_shrinkage(
        smbh_mass * u.solMass,
        disk_bh_pro_masses * u.solMass,
        unit_conversion.si_from_r_g(smbh_mass * u.solMass, disk_bh_pro_orbs_a),
        0 * u.m,
    )
    # Estimate the decay time to zero eccentricity

    # Estimate the number of timesteps to decay
    decay_timesteps = decay_time_arr.to('yr').value / timestep_duration_yr
    # Estimate decrement
    decrement_arr = (1.0 - (1. / decay_timesteps))
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

    # TODO: Update eccentricity as well
    return new_disk_bh_pro_orbs_a


def gw_hardening(mass_1, mass_2, bin_ecc, bin_sep, bin_time_to_merge, flag_merging, smbh_mass, timestep_length):
    array_length = len(mass_1)

    flag_not_merging = np.array([flag_merging[i] >= 0 for i in range(array_length)], dtype=np.bool_)

    # Find eccentricity factor (1-e_b^2)^7/2
    ecc_factor_1 = np.power(1 - np.power(bin_ecc[flag_not_merging], 2), 3.5)
    # and eccentricity factor [1+(73/24)e_b^2+(37/96)e_b^4]
    ecc_factor_2 = 1 + ((73 / 24) * np.power(bin_ecc[flag_not_merging], 2)) + (
                (37 / 96) * np.power(bin_ecc[flag_not_merging], 4))
    # overall ecc factor = ecc_factor_1/ecc_factor_2
    ecc_factor = ecc_factor_1 / ecc_factor_2

    sep_crit = (unit_conversion.r_schwarzschild_of_m(mass_1) +
                unit_conversion.r_schwarzschild_of_m(mass_2))

    time_to_merger_gw = (peters.time_of_orbital_shrinkage(
        mass_1[flag_not_merging] * u.Msun,
        mass_2[flag_not_merging] * u.Msun,
        unit_conversion.si_from_r_g(smbh_mass, bin_sep[flag_not_merging]),
        sep_final=sep_crit[flag_not_merging]
    ) * ecc_factor).value

    assert np.isfinite(time_to_merger_gw).all(), \
        "Finite check failure: time_to_merger_gw"

    new_time_to_merge = np.zeros(array_length)
    new_time_to_merge[~flag_not_merging] = bin_time_to_merge[~flag_not_merging]
    new_time_to_merge[flag_not_merging] = time_to_merger_gw

    timestep_duration_sec = (timestep_length * u.yr).to("second").value
    merge_mask = new_time_to_merge <= timestep_duration_sec

    new_bin_sep = np.zeros(array_length)
    new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
    new_bin_sep[~merge_mask] = bin_sep[~merge_mask]
    new_bin_sep[merge_mask] = sep_crit[merge_mask]

    new_flag_merging = np.zeros(array_length, dtype=np.int_)
    new_flag_merging[~flag_not_merging] = flag_merging[~flag_not_merging]
    new_flag_merging[~merge_mask]= flag_merging[~merge_mask]
    new_flag_merging[merge_mask] = -2

    return new_bin_sep, new_time_to_merge, new_flag_merging


class BinaryBlackHoleEvolveGW(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Binary Black Hole Evolve GW" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator) -> None:
        sm = self.settings

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

        gw_tracked_mask = blackholes_binary.bin_sep <= sm.min_bbh_gw_separation
        gw_tracked_ids = blackholes_binary.unique_id[gw_tracked_mask]

        if gw_tracked_ids.size > 0:
            bbh_gw_strain, bbh_gw_freq = peters.gw_strain_freq_prior(
                blackholes_binary.get_attribute("mass_1", gw_tracked_ids),
                blackholes_binary.get_attribute("mass_2", gw_tracked_ids),
                blackholes_binary.get_attribute("bin_sep", gw_tracked_ids),
                sm.smbh_mass,
                timestep_length,
                blackholes_binary.get_attribute("gw_freq", gw_tracked_ids), # old_bbh_gw_freq
                sm.agn_redshift
            )

            blackholes_binary.gw_freq[gw_tracked_mask] = bbh_gw_freq
            blackholes_binary.gw_strain[gw_tracked_mask] = bbh_gw_strain

            blackholes_gw = blackholes_binary.copy()
            blackholes_gw.keep_only(gw_tracked_ids)

            blackholes_gw.consistency_check()
            blackholes_binary.consistency_check()

            filing_cabinet.ignore_check_array(sm.bbh_gw_array_name)
            filing_cabinet.create_or_append_array(sm.bbh_gw_array_name, blackholes_gw)

        blackholes_binary.gw_freq[~gw_tracked_mask], blackholes_binary.gw_strain[~gw_tracked_mask] = peters.gw_strain_freq_no_prior(
            blackholes_binary.mass_1[~gw_tracked_mask],
            blackholes_binary.mass_2[~gw_tracked_mask],
            blackholes_binary.bin_sep[~gw_tracked_mask],
            sm.smbh_mass,
            sm.agn_redshift
        )

        blackholes_binary.bin_sep, blackholes_binary.time_to_merger_gw, blackholes_binary.flag_merging \
            = gw_hardening(blackholes_binary.mass_1, blackholes_binary.mass_2,
                     blackholes_binary.bin_ecc, blackholes_binary.bin_sep,
                     blackholes_binary.time_to_merger_gw, blackholes_binary.flag_merging,
                     sm.smbh_mass, timestep_length)

        blackholes_binary.consistency_check()


class InnerBlackHoleDynamics(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None, target_array: str = ""):
        super().__init__("Inner Black Hole Dynamics" if name is None else name, settings)
        self.target_array = target_array

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet,
                agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if self.target_array not in filing_cabinet:
            return

        inner_bh = filing_cabinet.get_array(self.target_array, AGNBlackHoleArray)

        # TODO: also get bh_near_smbh to return updated ecc and add here
        inner_bh.orb_a = bh_near_smbh(
            sm.smbh_mass,
            inner_bh.orb_a,
            inner_bh.mass,
            inner_bh.orb_ecc,
            timestep_length,
            sm.disk_radius_outer,
            sm.disk_inner_stable_circ_orb
        )

        zero_strain_mask = inner_bh.gw_strain == 0
        inner_bh.gw_strain[zero_strain_mask] = 9.e-7

        emri_gw_strain, emri_gw_freq = evolve_emri_gw(
            inner_bh.mass,
            inner_bh.orb_a,
            timestep_length,
            inner_bh.gw_freq,
            sm.smbh_mass,
            sm.agn_redshift
        )

        inner_bh.gw_freq = emri_gw_freq
        inner_bh.gw_strain = emri_gw_strain



