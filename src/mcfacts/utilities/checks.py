import numpy as np
from scipy.stats import truncnorm

import mcfacts.utilities
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBinaryBlackHoleArray
from mcfacts.utilities.unit_conversion import r_schwarzschild_of_m, r_g_from_units


def bin_check_params(bin_mass_1, bin_mass_2, bin_orb_a_1, bin_orb_a_2, bin_ecc, bin_id_num):
    """Tests if binaries are real (location and mass do not equal 0)

    This function tests to see if the binary is real. If location = 0 or mass = 0 *and* any other element is NON-ZERO then discard this binary element.
    Returns ID numbers of fake binaries.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters

    Returns
    -------
    id_nums or bh_bin_id_num_fakes : numpy.ndarray
        ID numbers of fake binaries with :obj:`float` type
    """
    bh_bin_id_num_fakes = np.array([])

    mass_1_id_num = bin_id_num[bin_mass_1 == 0]
    mass_2_id_num = bin_id_num[bin_mass_2 == 0]
    orb_a_1_id_num = bin_id_num[bin_orb_a_1 == 0]
    orb_a_2_id_num = bin_id_num[bin_orb_a_2 == 0]
    bin_ecc_id_num = bin_id_num[bin_ecc >= 1]

    id_nums = np.concatenate([mass_1_id_num, mass_2_id_num,
                              orb_a_1_id_num, orb_a_2_id_num, bin_ecc_id_num])

    if id_nums.size > 0:
        return (id_nums)
    else:
        return (bh_bin_id_num_fakes)


def bin_reality_check(bin_mass_1, bin_mass_2, bin_orb_a_1, bin_orb_a_2, bin_ecc, bin_id_num):
    """Tests if binaries are real (location and mass do not equal 0)

    DEPRECATED

    This function tests to see if the binary is real. If location = 0 or mass = 0 *and* any other element is NON-ZERO then discard this binary element.
    Returns ID numbers of fake binaries.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters

    Returns
    -------
    id_nums or bh_bin_id_num_fakes : numpy.ndarray
        ID numbers of fake binaries with :obj:`float` type
    """
    bh_bin_id_num_fakes = np.array([])

    mass_1_id_num = bin_id_num[bin_mass_1 == 0]
    mass_2_id_num = bin_id_num[bin_mass_2 == 0]
    orb_a_1_id_num = bin_id_num[bin_orb_a_1 == 0]
    orb_a_2_id_num = bin_id_num[bin_orb_a_2 == 0]
    bin_ecc_id_num = bin_id_num[bin_ecc >= 1]

    id_nums = np.concatenate([mass_1_id_num, mass_2_id_num,
                             orb_a_1_id_num, orb_a_2_id_num, bin_ecc_id_num])

    if id_nums.size > 0:
        return (id_nums)
    else:
        return (bh_bin_id_num_fakes)


def binary_reality_check(sm: SettingsManager, filing_cabinet: FilingCabinet, log_func: callable):
    if sm.bbh_array_name not in filing_cabinet:
        return

    blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

    # First check that binaries are real (mass and location are not zero)
    bh_binary_id_num_unphysical = bin_check_params(
        blackholes_binary.mass_1,
        blackholes_binary.mass_2,
        blackholes_binary.orb_a_1,
        blackholes_binary.orb_a_2,
        blackholes_binary.bin_ecc,
        blackholes_binary.id_num
    )
    blackholes_binary.remove_all(bh_binary_id_num_unphysical)

    # Check for binaries with hyperbolic eccentricity (ejected from disk)
    bh_binary_id_num_ecc_hyperbolic = blackholes_binary.id_num[blackholes_binary.bin_orb_ecc >= 1.]
    blackholes_binary.remove_all(bh_binary_id_num_ecc_hyperbolic)

    log_func(f"Removed {len(bh_binary_id_num_unphysical) + len(bh_binary_id_num_ecc_hyperbolic)} binaries with unphysical parameters.")


def bin_contact_check(bin_mass_1, bin_mass_2, bin_sep, bin_flag_merging, smbh_mass):
    """Tests if binary separation has shrunk so that binary is touching

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    smbh_mass : float
        Mass [M_sun] of the SMBH

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Returns modified blackholes_binary with updated bin_sep and flag_merging.

    Notes
    -----
    Touching condition is where binary separation is <= R_schw(M_1) + R_schw(M_2)
                                                      = 2(R_g(M_1) + R_g(M_2))
                                                      = 2G(M_1+M_2) / c^{2}

    Since binary separation is in units of r_g (GM_smbh/c^2) then condition is simply:
        binary_separation <= 2M_bin/M_smbh
    """

    # We assume bh are not spinning when in contact. TODO: Consider spin in future.
    contact_condition = (r_schwarzschild_of_m(bin_mass_1) +
                         r_schwarzschild_of_m(bin_mass_2))
    contact_condition = r_g_from_units(smbh_mass, contact_condition).value
    mask_condition = (bin_sep <= contact_condition)

    # If binary separation <= contact condition, set binary separation to contact condition
    bin_sep[mask_condition] = contact_condition[mask_condition]
    bin_flag_merging[mask_condition] = -2

    assert np.all(~np.isnan(bin_flag_merging)), \
        "blackholes_binary.flag_merging contains NaN values"

    return (bin_sep, bin_flag_merging)


def flag_binary_mergers(sm: SettingsManager, filing_cabinet: FilingCabinet):
    if sm.bbh_array_name not in filing_cabinet:
        return

    if sm.bh_prograde_array_name not in filing_cabinet:
        return

    blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

    blackholes_binary.bin_sep, blackholes_binary.flag_merging = bin_contact_check(
        blackholes_binary.mass_1,
        blackholes_binary.mass_2,
        blackholes_binary.bin_sep,
        blackholes_binary.flag_merging,
        sm.smbh_mass
    )


def generate_truncated_normal(mean=0, std=1, lower=0.75, upper=0.85, size=10):
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def spin_check(gen_1, gen_2, spin_merged):
    """ Since the Tichy and Marronetti '08 perscription generates spin values outside of the expected range for higher mass ratio objects this file checks spin values after merger and if the magnitude is too low, this function resets it to a random distribution between a set range in order to generate results similiar to that of the NRsurrogate model.

    Parameters
    ----------
        gen_1 : numpy.ndarray
            generation of m1 (before merger) (1=natal BH that has never been in a prior merger)
        gen_2 : numpy.darray
            generation of m2 (before merger) (1=natal BH that has never been in a prior merger)
        spin_merged : numpy.darray
            Final spin magnitude [unitless] of merger remnant with :obj:`float` type

    Returns
    -------
    merged_spins : numpy array
        Final spin magnitude [unitless] of merger remnant with :obj:`float` type
    """

    # print("initial gen 1: ", gen_1)
    # print("initial gen 2: ", gen_2)
    # print("initial spin_merged: ", spin_merged)

    new_spin_merged = []

    for i in range(len(spin_merged)):
        # Sorting first gen objects and keeping their parameters
        if (gen_1[i] == 1.) & (gen_2[i] == 1.):
            # print('gen 1', spin_merged[i])
            new_spin_merged.append(spin_merged[i])
        # Sorting 2nd gen objects and updating their spins as needed otherwise keeping them the same
        # If spins < 0.75, they are reset to a randomly selected gaussian distribution between 0.75 - 0.85
        elif ((gen_1[i] == 2.) | (gen_2[i] == 2.)) & ((gen_1[i] <= 2.) & (gen_2[i] <= 2.)):
            # print('gen 2', spin_merged[i])
            if spin_merged[i] < 0.75:
                spin_plus_noise = generate_truncated_normal(mean=0, std=1, lower=0.75, upper=0.85, size=1)
                # print('gen 2 plus noise', spin_plus_noise)
                new_spin_merged.append(float(spin_plus_noise))
            else:
                # print('gen 2', spin_merged[i])
                new_spin_merged.append(spin_merged[i])
        # Sorting 3+ gen objects and updating their spins as needed otherwise keeping them the same
        # If spins < 0.85, they are reset to a randomly selected gaussian distribution between 0.85 - 0.95
        elif (gen_1[i] >= 3.) | (gen_2[i] >= 3.):
            # print('gen x', spin_merged[i])
            if spin_merged[i] < 0.85:
                spin_plus_noise = generate_truncated_normal(mean=0, std=1, lower=0.85, upper=0.95, size=1)
                # print('gen 3+ plus noise', spin_plus_noise)
                new_spin_merged.append(float(spin_plus_noise))
            else:
                # print('gen 3+', spin_merged[i])
                new_spin_merged.append(spin_merged[i])

    # print("new gen 1: ", gen_1)
    # print("new gen 2: ", gen_2)
    # print("new spin_merged: ", spin_merged)
    return np.array(new_spin_merged)


def find_function_lower_bounds(disk_function):
    upper_bound = 20
    lower_bound = 5

    final_bound = 6

    for i in range(25):
        middle = lower_bound + ((upper_bound - lower_bound) / 2)

        if np.isfinite(disk_function(middle)):
            upper_bound = middle
            final_bound = middle
        else:
            lower_bound = middle


    return final_bound