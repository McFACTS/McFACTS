from mcfacts.physics.binary.evolve import bin_harden_baruteau, bin_harden_baruteau_optimized
import numpy as np
import pandas as pd
import astropy.units as u
import pytest

# def test_baruteau():


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


