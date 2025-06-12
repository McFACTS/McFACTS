"""
Module for evolving the state of a binary.
"""
import astropy.units as u
import numpy as np
from numpy.random import Generator

from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBlackHoleArray

from mcfacts.objects.timeline import TimelineActor
from mcfacts.physics import point_masses, merge, reality_check


def bin_ionization_check(bin_mass_1, bin_mass_2, bin_orb_a, bin_sep, bin_id_num, smbh_mass):
    """Tests whether binary has been ionized beyond some limit

    This function tests whether a binary has been softened beyond some limit.
    Returns ID numbers of binaries to be ionized.
    The limit is set to some fraction of the binary Hill sphere, frac_R_hill

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    smbh_mass : float
        Mass [M_sun] of the SMBH

    Returns
    -------
    bh_id_nums : numpy.ndarray
        ID numbers of binaries to be removed from binary array

    Notes
    -----
    Default is frac_R_hill = 1.0 (ie binary is ionized at the Hill sphere). 
    Change frac_R_hill if you're testing binary formation at >R_hill.

    R_hill = a_com*(M_bin/3M_smbh)^1/3

    where a_com is the radial disk location of the binary center of mass,
    M_bin = M_1 + M_2 is the binary mass
    M_smbh is the SMBH mass (given by smbh_mass) 

    Condition is:
    if bin_separation > frac_R_hill*R_hill:
        Ionize binary.
        Remove binary from blackholes_binary!
        Add two new singletons to the singleton arrays.
    """

    # Remove returning -1 if that's not how it's supposed to work
    # Define ionization threshold as a fraction of Hill sphere radius
    # Default is 1.0, change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    # bin_orb_a is in units of r_g of the SMBH = GM_smbh/c^2
    mass_ratio = (bin_mass_1 + bin_mass_2) / smbh_mass
    hill_sphere = bin_orb_a * np.power(mass_ratio / 3, 1. / 3.)

    bh_id_nums = bin_id_num[np.where(bin_sep > (frac_rhill * hill_sphere))[0]]

    return (bh_id_nums)


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
    sep_crit = (point_masses.r_schwarzschild_of_m(bin_mass_1[idx_non_mergers]) +
                point_masses.r_schwarzschild_of_m(bin_mass_2[idx_non_mergers]))
    time_to_merger_gw = (point_masses.time_of_orbital_shrinkage(
        bin_mass_1[idx_non_mergers] * u.Msun,
        bin_mass_2[idx_non_mergers] * u.Msun,
        point_masses.si_from_r_g(smbh_mass, bin_sep_nomerge),
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

        time_gw_normalization = filing_cabinet.get_value("time_gw_normalization", merge.normalize_tgw(sm.smbh_mass, sm.inner_disk_outer_radius))

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

        reality_check.binary_reality_check(sm, filing_cabinet, self.log)
        merge.flag_binary_mergers(sm, filing_cabinet)