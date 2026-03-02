"""
Module for hardening the binary via gas.
"""
import astropy.units as u
import astropy.constants as const
import numpy as np
from numpy.random import Generator

import mcfacts.modules.gw
import mcfacts.utilities.checks
import mcfacts.utilities.peters
import mcfacts.utilities.unit_conversion
from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities import peters, checks, unit_conversion


def bin_harden_baruteau(bin_mass_1, bin_mass_2, bin_sep, bin_ecc, bin_time_to_merger_gw, bin_flag_merging, bin_time_merged, smbh_mass, timestep_duration_yr,
                        time_gw_normalization, time_passed, r_g_in_meters):
    """Harden black hole binaries using Baruteau+11 prescription

    Use Baruteau+11 prescription to harden a pre-existing binary.
    For every 1000 orbits of binary around its center of mass, the
    separation (between binary components) is halved.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    smbh_mass : float
        Mass [M_sun] of the SMBH
    timestep_duration_yr : float
        Length of timestep [yr]
    time_gw_normalization : float
        A normalization for GW decay timescale [s], set by `smbh_mass` & normalized for
        a binary total mass of 10 solar masses.
    bin_index : int
        Count of number of binaries
    time_passed : float
        Time elapsed [yr] since beginning of simulation.

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Black hole binaries with time_to_merger_gw, bin_sep, flag_merging, and time_merged updated
    """

    # 1. Find active binaries
    # 2. Find number of binary orbits around its center of mass within the timestep
    # 3. For every 10^3 orbits, halve the binary separation.

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(bin_flag_merging >= 0)[0]

    # If all binaries have merged then nothing to do
    if (idx_non_mergers.shape[0] == 0):
        return bin_sep, bin_flag_merging, bin_time_merged, bin_time_to_merger_gw

    # Set up variables
    mass_binary = bin_mass_1[idx_non_mergers] + bin_mass_2[idx_non_mergers]
    bin_sep_nomerge = bin_sep[idx_non_mergers]
    bin_ecc_nomerge = bin_ecc[idx_non_mergers]

    # Find eccentricity factor (1-e_b^2)^7/2
    ecc_factor_1 = np.power(1 - np.power(bin_ecc_nomerge, 2), 3.5)
    # and eccentricity factor [1+(73/24)e_b^2+(37/96)e_b^4]
    ecc_factor_2 = 1 + ((73/24) * np.power(bin_ecc_nomerge, 2)) + ((37/96) * np.power(bin_ecc_nomerge, 4))
    # overall ecc factor = ecc_factor_1/ecc_factor_2
    ecc_factor = ecc_factor_1/ecc_factor_2

    # Binary period = 2pi*sqrt((delta_r)^3/GM_bin)
    # or T_orb = 10^7s*(1r_g/m_smmbh=10^8Msun)^(3/2) *(M_bin/10Msun)^(-1/2) = 0.32yrs
    bin_period = 0.32 * np.power(bin_sep_nomerge, 1.5) * np.power(smbh_mass/1.e8, 1.5) * np.power(mass_binary/10.0, -0.5)

    # Find how many binary orbits in timestep. Binary separation is halved for every 10^3 orbits.
    num_orbits_in_timestep = np.zeros(len(bin_period))
    num_orbits_in_timestep[bin_period > 0] = timestep_duration_yr / bin_period[bin_period > 0]
    scaled_num_orbits = num_orbits_in_timestep / 1000.0

    # Timescale for binary merger via GW emission alone in seconds, scaled to bin parameters
    sep_crit = (unit_conversion.r_schwarzschild_of_m(bin_mass_1[idx_non_mergers]) +
                unit_conversion.r_schwarzschild_of_m(bin_mass_2[idx_non_mergers]))
    time_to_merger_gw = (peters.time_of_orbital_shrinkage(
        bin_mass_1[idx_non_mergers] * u.Msun,
        bin_mass_2[idx_non_mergers] * u.Msun,
        unit_conversion.si_from_r_g(smbh_mass, bin_sep_nomerge, r_g_defined=r_g_in_meters),
        sep_final=sep_crit
    ) * ecc_factor).value

    # Finite check
    assert np.isfinite(time_to_merger_gw).all(),\
        "Finite check failure: time_to_merger_gw"
    bin_time_to_merger_gw[idx_non_mergers] = time_to_merger_gw

    # Create mask for things that WILL merge in this timestep
    # need timestep_duration_yr in seconds
    timestep_duration_sec = (timestep_duration_yr * u.year).to("second").value
    merge_mask = time_to_merger_gw <= timestep_duration_sec

    # Binary will not merge in this timestep
    # new bin_sep according to Baruteau+11 prescription
    bin_sep_nomerge[~merge_mask] = bin_sep_nomerge[~merge_mask] * (0.5 ** scaled_num_orbits[~merge_mask])
    bin_sep[idx_non_mergers[~merge_mask]] = bin_sep_nomerge[~merge_mask]
    # Finite check
    assert np.isfinite(bin_sep_nomerge).all(),\
        "Finite check failure: bin_sep_nomerge"

    # Otherwise binary will merge in this timestep
    # Update flag_merging to -2 and time_merged to current time
    bin_flag_merging[idx_non_mergers[merge_mask]] = -2
    bin_time_merged[idx_non_mergers[merge_mask]] = time_passed
    # Finite check
    assert np.isfinite(bin_flag_merging).all(),\
        "Finite check failure: bin_flag_merging"
    # Finite check
    assert np.isfinite(bin_time_merged).all(),\
        "Finite check failure: bin_time_merged"

    return (bin_sep, bin_flag_merging, bin_time_merged, bin_time_to_merger_gw)


def baruteau_drag(mass_1, mass_2, bin_sep, smbh_mass, timestep_duration_yr):
    binary_mass = mass_1 + mass_2
    bin_period = 0.32 * np.power(bin_sep, 1.5) * np.power(smbh_mass / 1.e8, 1.5) * np.power(
        binary_mass / 10.0, -0.5)

    num_orbits_in_timestep = np.zeros(len(bin_period))
    num_orbits_in_timestep[bin_period > 0] = timestep_duration_yr / bin_period[bin_period > 0]
    scaled_num_orbits = num_orbits_in_timestep / 1000.0

    return bin_sep * (0.5 ** scaled_num_orbits)


def stahler_drag(mass_1, mass_2, bin_sep, orb_a, disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters):
    q = np.minimum(mass_1 / mass_2, mass_2 / mass_1)

    total_mass = ((mass_1 + mass_2) * const.M_sun).si

    scaling_constant = (15 / (35 * np.pi))
    ratio_component = (((1 + q) ** 2) / q)
    gas_component = (((disk_sound_speed(orb_a) * u.meter/u.second) ** 5) / (disk_density(orb_a) * (u.kg / u.m ** 3)))
    mass_component = 1 / ((const.G ** 3) * (total_mass ** 2))

    sep_unit = unit_conversion.si_from_r_g(smbh_mass, bin_sep, r_g_defined=r_g_in_meters)

    coalescence_time = sep_unit * (scaling_constant * ratio_component * gas_component * mass_component)

    timestep_units = (timestep_duration_yr * u.year).si

    new_bin_sep = bin_sep * (1 - (timestep_units / coalescence_time))

    contact_condition = (unit_conversion.r_schwarzschild_of_m(mass_1) +
                         unit_conversion.r_schwarzschild_of_m(mass_2))
    contact_condition = unit_conversion.r_g_from_units(smbh_mass, contact_condition).value

    new_bin_sep[new_bin_sep < contact_condition] = contact_condition[new_bin_sep < contact_condition]

    return new_bin_sep


def gas_hardening_no_stalling(mass_1, mass_2, bin_sep, flag_merging, smbh_mass, gas_hardening_prescription, orb_a, disk_sound_speed, disk_density, timestep_duration_yr, r_g_in_meters):
    flag_not_merging = np.array([(flag_merging[i] >= 0) for i in range(len(mass_1))], dtype=bool)

    if gas_hardening_prescription == "baruteau":
        calc_bin_sep = baruteau_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], smbh_mass, timestep_duration_yr)
    elif gas_hardening_prescription == "stahler":
        calc_bin_sep = stahler_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], orb_a[flag_not_merging], disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters)
    else:
        assert "No gas hardening prescription specified... Available values: (baruteau, stahler)"

    new_bin_sep = np.zeros(len(mass_1))
    new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
    new_bin_sep[flag_not_merging] = calc_bin_sep

    return new_bin_sep


def gas_hardening_variable_stalling(mass_1, mass_2, bin_sep, bin_orb_a, disk_sound_speed, flag_merging, smbh_mass, gas_hardening_prescription, disk_density, timestep_duration_yr, r_g_in_meters):
    rg_scale = (const.G * smbh_mass * const.M_sun / const.c ** 2).value
    bin_orb_velocity = np.sqrt((const.G.value * ((mass_1 + mass_2) * const.M_sun).si.value) / (bin_sep * rg_scale))
    sound_speed = disk_sound_speed(bin_orb_a)

    effective_stalling_separation = np.array([(sep if vel >= s_speed else 0) for vel, sep, s_speed in zip(bin_orb_velocity, bin_sep, sound_speed)])
    flag_not_merging = np.array([(flag_merging[i] >= 0 and bin_sep[i] > effective_stalling_separation[i]) for i in range(len(mass_1))], dtype=bool)

    if gas_hardening_prescription == "baruteau":
        calc_bin_sep = baruteau_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], smbh_mass, timestep_duration_yr)
    elif gas_hardening_prescription == "stahler":
        calc_bin_sep = stahler_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], bin_orb_a[flag_not_merging], disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters)
    else:
        assert "No gas hardening prescription specified... Available values: (baruteau, stahler)"

    new_bin_sep = np.zeros(len(mass_1))
    new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
    new_bin_sep[flag_not_merging] = calc_bin_sep

    return new_bin_sep


def gas_hardening_fixed_stalling(mass_1, mass_2, bin_sep, flag_merging, smbh_mass, stalling_separation, gas_hardening_prescription, orb_a, disk_sound_speed, disk_density,  timestep_duration_yr, r_g_in_meters):
    flag_not_merging = np.array([(flag_merging[i] >= 0 and bin_sep[i] > stalling_separation) for i in range(len(mass_1))], dtype=bool)

    if gas_hardening_prescription == "baruteau":
        calc_bin_sep = baruteau_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], smbh_mass, timestep_duration_yr)
    elif gas_hardening_prescription == "stahler":
        calc_bin_sep = stahler_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], orb_a[flag_not_merging], disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters)
    else:
        assert "No gas hardening prescription specified... Available values: (baruteau, stahler)"

    new_bin_sep = np.zeros(len(mass_1))
    new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
    new_bin_sep[flag_not_merging] = calc_bin_sep

    return new_bin_sep


class BinaryBlackHoleGasHardening(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None, reality_merge_checks: bool = False):
        super().__init__("Binary Black Hole Gas Hardening" if name is None else name, settings)

        self.reality_merge_checks = reality_merge_checks

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBlackHoleArray)

        if sm.stalling_separation > 0:
            blackholes_binary.bin_sep = gas_hardening_fixed_stalling(
                blackholes_binary.mass,
                blackholes_binary.mass_2,
                blackholes_binary.bin_sep,
                blackholes_binary.flag_merging,
                sm.smbh_mass,
                sm.stalling_separation,
                sm.gas_hardening_prescription,
                blackholes_binary.orb_a,
                agn_disk.disk_sound_speed,
                agn_disk.disk_density,
                timestep_length,
                sm.r_g_in_meters
            )
        elif sm.stalling_separation == 0:
            blackholes_binary.bin_sep = gas_hardening_no_stalling(
                blackholes_binary.mass,
                blackholes_binary.mass_2,
                blackholes_binary.bin_sep,
                blackholes_binary.flag_merging,
                sm.smbh_mass,
                sm.gas_hardening_prescription,
                blackholes_binary.orb_a,
                agn_disk.disk_sound_speed,
                agn_disk.disk_density,
                timestep_length,
                sm.r_g_in_meters
            )
        elif sm.stalling_separation == -1:
            blackholes_binary.bin_sep = gas_hardening_variable_stalling(
                blackholes_binary.mass,
                blackholes_binary.mass_2,
                blackholes_binary.bin_sep,
                blackholes_binary.orb_a,
                agn_disk.disk_sound_speed,
                blackholes_binary.flag_merging,
                sm.smbh_mass,
                sm.gas_hardening_prescription,
                agn_disk.disk_density,
                timestep_length,
                sm.r_g_in_meters
            )


        if not self.reality_merge_checks:
            return

        checks.binary_reality_check(sm, filing_cabinet, self.log)
        checks.flag_binary_mergers(sm, filing_cabinet)
