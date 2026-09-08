"""Unit tests for circular_singles_encounters_prograde_sweep_optimized"""
import numpy as np
import pytest
import pandas as pd

from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.dynamics import (
    circular_singles_encounters_prograde,
    circular_singles_encounters_prograde_sweep_optimized,
)

SEED = 483445


def parse_array(cell):
    if isinstance(cell, str) and cell.startswith('['):
        cleaned = cell.strip('[]')
        return np.fromstring(cleaned, sep=' ')
    return cell


def setup_sweep_params():
    """Return input parameters read from CSV."""
    inputs = pd.read_csv("tests/optimizations/sweep_inputs.csv", header=None)
    params = []
    for _, row in inputs.iterrows():
        params.append(tuple(parse_array(row[i]) for i in range(8)))
    return params


@pytest.mark.parametrize(
    "smbh_mass, disk_bh_pro_orbs_a, disk_bh_pro_masses, disk_bh_pro_orbs_ecc, "
    "timestep_duration_yr, disk_bh_pro_orb_ecc_crit, delta_energy_strong, disk_radius_outer",
    setup_sweep_params(),
)
def test_sweep_optimized_matches_original(
    smbh_mass,
    disk_bh_pro_orbs_a,
    disk_bh_pro_masses,
    disk_bh_pro_orbs_ecc,
    timestep_duration_yr,
    disk_bh_pro_orb_ecc_crit,
    delta_energy_strong,
    disk_radius_outer,
):
    """Test that optimized sweep matches original."""
    # Both functions mutate their array inputs, so we need copies
    orbs_a_orig = disk_bh_pro_orbs_a.copy()
    masses_orig = disk_bh_pro_masses.copy()
    orbs_ecc_orig = disk_bh_pro_orbs_ecc.copy()

    orbs_a_opt = disk_bh_pro_orbs_a.copy()
    masses_opt = disk_bh_pro_masses.copy()
    orbs_ecc_opt = disk_bh_pro_orbs_ecc.copy()

    rng.seed(seed=SEED)
    result_a_orig, result_ecc_orig = circular_singles_encounters_prograde(
        smbh_mass,
        orbs_a_orig,
        masses_orig,
        orbs_ecc_orig,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        disk_radius_outer,
        rng_here=rng,
    )

    rng.seed(seed=SEED)
    result_a_opt, result_ecc_opt = circular_singles_encounters_prograde_sweep_optimized(
        smbh_mass,
        orbs_a_opt,
        masses_opt,
        orbs_ecc_opt,
        timestep_duration_yr,
        disk_bh_pro_orb_ecc_crit,
        delta_energy_strong,
        disk_radius_outer,
        rng_here=rng,
    )

    assert np.allclose(result_a_orig, result_a_opt, rtol=1e-9), \
        f"orbs_a mismatch: max diff = {np.max(np.abs(result_a_orig - result_a_opt))}"
    assert np.allclose(result_ecc_orig, result_ecc_opt, rtol=1e-9), \
        f"orbs_ecc mismatch: max diff = {np.max(np.abs(result_ecc_orig - result_ecc_opt))}"
