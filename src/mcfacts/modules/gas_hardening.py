"""
Module for hardening the binary via gas.
"""
import astropy.units as u
import numpy as np
from numpy.random import Generator

import mcfacts.modules.gw
import mcfacts.utilities.checks
import mcfacts.utilities.peters
import mcfacts.utilities.unit_conversion
from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities import peters, checks


def bin_harden_baruteau(bin_mass_1, bin_mass_2, bin_sep, bin_ecc, bin_time_to_merger_gw, bin_flag_merging,
                        bin_time_merged, smbh_mass, timestep_duration_yr,
                        time_gw_normalization, time_passed):
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
    ecc_factor_2 = 1 + ((73 / 24) * np.power(bin_ecc_nomerge, 2)) + ((37 / 96) * np.power(bin_ecc_nomerge, 4))
    # overall ecc factor = ecc_factor_1/ecc_factor_2
    ecc_factor = ecc_factor_1 / ecc_factor_2

    # Binary period = 2pi*sqrt((delta_r)^3/GM_bin)
    # or T_orb = 10^7s*(1r_g/m_smmbh=10^8Msun)^(3/2) *(M_bin/10Msun)^(-1/2) = 0.32yrs
    bin_period = 0.32 * np.power(bin_sep_nomerge, 1.5) * np.power(smbh_mass / 1.e8, 1.5) * np.power(mass_binary / 10.0,
                                                                                                    -0.5)

    # Find how many binary orbits in timestep. Binary separation is halved for every 10^3 orbits.
    num_orbits_in_timestep = np.zeros(len(bin_period))
    num_orbits_in_timestep[bin_period > 0] = timestep_duration_yr / bin_period[bin_period > 0]
    scaled_num_orbits = num_orbits_in_timestep / 1000.0

    # Timescale for binary merger via GW emission alone in seconds, scaled to bin parameters
    sep_crit = (mcfacts.utilities.unit_conversion.r_schwarzschild_of_m(bin_mass_1[idx_non_mergers]) +
                mcfacts.utilities.unit_conversion.r_schwarzschild_of_m(bin_mass_2[idx_non_mergers]))
    time_to_merger_gw = (peters.time_of_orbital_shrinkage(
        bin_mass_1[idx_non_mergers] * u.Msun,
        bin_mass_2[idx_non_mergers] * u.Msun,
        mcfacts.utilities.unit_conversion.si_from_r_g(smbh_mass, bin_sep_nomerge),
        sep_final=sep_crit
    ) * ecc_factor).value

    # Finite check
    assert np.isfinite(time_to_merger_gw).all(), \
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
    assert np.isfinite(bin_sep_nomerge).all(), \
        "Finite check failure: bin_sep_nomerge"

    # Otherwise binary will merge in this timestep
    # Update flag_merging to -2 and time_merged to current time
    bin_flag_merging[idx_non_mergers[merge_mask]] = -2
    bin_time_merged[idx_non_mergers[merge_mask]] = time_passed
    # Finite check
    assert np.isfinite(bin_flag_merging).all(), \
        "Finite check failure: bin_flag_merging"
    # Finite check
    assert np.isfinite(bin_time_merged).all(), \
        "Finite check failure: bin_time_merged"

    return (bin_sep, bin_flag_merging, bin_time_merged, bin_time_to_merger_gw)


class BinaryBlackHoleGasHardening(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None, reality_merge_checks: bool = False):
        super().__init__("Binary Black Hole Gas Hardening" if name is None else name, settings)

        self.reality_merge_checks = reality_merge_checks

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBlackHoleArray)

        time_gw_normalization = filing_cabinet.get_value("time_gw_normalization", mcfacts.modules.gw.normalize_tgw(sm.smbh_mass, sm.inner_disk_outer_radius))

        blackholes_binary.bin_sep, blackholes_binary.flag_merging, blackholes_binary.time_merged, blackholes_binary.time_to_merger_gw = bin_harden_baruteau(
            blackholes_binary.mass_1,
            blackholes_binary.mass_2,
            blackholes_binary.bin_sep,
            blackholes_binary.bin_ecc,
            blackholes_binary.time_to_merger_gw,
            blackholes_binary.flag_merging,
            blackholes_binary.time_merged,
            sm.smbh_mass,
            sm.timestep_duration_yr,
            time_gw_normalization,
            time_passed,
        )

        if not self.reality_merge_checks:
            return

        checks.binary_reality_check(sm, filing_cabinet, self.log)
        checks.flag_binary_mergers(sm, filing_cabinet)
