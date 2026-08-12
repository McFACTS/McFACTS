from mcfacts.modules.gas_hardening import bin_harden_baruteau_optimized
from mcfacts.utilities import peters, unit_conversion
import numpy as np
import pandas as pd
import astropy.units as u
import pytest


# TODO: Update test to more accurately reflect current implementation of baruteau gas hardening.

def bin_harden_baruteau(bin_mass_1, bin_mass_2, bin_sep, bin_ecc, bin_time_to_merger_gw, bin_flag_merging,
                        bin_time_merged, smbh_mass, timestep_duration_yr, time_gw_normalization, time_passed,
                        r_g_in_meters):
    """
    Reference implementation of the Baruteau+11 hardening prescription, taken from previous McFACTS version.
    """
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
        unit_conversion.si_from_r_g_optimized(smbh_mass, bin_sep_nomerge),
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

# parse array out of the csv file
def parse_array(cell):
    if isinstance(cell, str) and cell.startswith('['):
        cleaned = cell.strip('[]')
        return np.fromstring(cleaned, sep=' ')
    return cell

# parse scalar value out of csv file
def parse_value(cell):
    if isinstance(cell, str):
        # check if it's an array
        if '[' in cell:
            cleaned = cell.replace('[', '').replace(']', '')
            return np.fromstring(cleaned, sep=' ')
        
        # try to extract number from string with units (e.g., "147662503805.01248 m")
        try:
            # split by whitespace and take the first part (the number)
            numeric_part = cell.split()[0]
            return float(numeric_part)
        except (ValueError, IndexError):
            return cell
    return cell


def setup_baruteau_params():
    """Return input parameters read from CSV."""
    inputs = pd.read_csv("tests/optimizations/baruteau_inputs.csv", header=None)
    params = []
    for _, row in inputs.iterrows():
        params.append(tuple(parse_value(row[i]) for i in range(11)))
    return params


@pytest.mark.parametrize(
    "bin_mass_1, bin_mass_2, bin_sep, bin_ecc, bin_time_to_merger_gw, bin_flag_merging, "
    "bin_time_merged, smbh_mass, timestep_duration_yr, time_gw_normalization, time_passed",
    setup_baruteau_params(),
)
def test_sweep_optimized_matches_original(
    bin_mass_1,
    bin_mass_2,
    bin_sep,
    bin_ecc,
    bin_time_to_merger_gw,
    bin_flag_merging,
    bin_time_merged,
    smbh_mass,
    timestep_duration_yr,
    time_gw_normalization,
    time_passed,
):

    out_sep, out_flag_merging, out_time_merged, out_time_to_merger_gw = bin_harden_baruteau(
        np.array([bin_mass_1]),
        np.array([ bin_mass_2 ]),
        np.array([ bin_sep ]),
        np.array([ bin_ecc ]),
        np.array([ bin_time_to_merger_gw ]),
        np.array([ bin_flag_merging ]),
        np.array([ bin_time_merged ]),
        smbh_mass,
        timestep_duration_yr,
        time_gw_normalization,
        time_passed,
        r_g_in_meters=None
    )
    out_sep_opt, out_flag_merging_opt, out_time_merged_opt, out_time_to_merger_gw_opt = bin_harden_baruteau_optimized(
        np.array([ bin_mass_1 ]),
        np.array([ bin_mass_2 ]),
        np.array([ bin_sep ]),
        np.array([ bin_ecc ]),
        np.array([ bin_time_to_merger_gw ]),
        np.array([ bin_flag_merging ]),
        np.array([ bin_time_merged ]),
        smbh_mass,
        timestep_duration_yr,
        time_gw_normalization,
        time_passed,
        r_g_in_meters=None
    )

    assert np.allclose(out_sep, out_sep_opt, rtol=1e-6)

    assert np.allclose(out_flag_merging, out_flag_merging_opt, rtol=1e-6)
    assert np.allclose(out_time_merged, out_time_merged_opt, rtol=1e-6)
    assert np.allclose(out_time_to_merger_gw, out_time_to_merger_gw_opt, rtol=1e-6)


