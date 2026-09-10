"""Unit tests for setup_disk_blackholes_location_NSC_powerlaw"""
import numpy as np
import pytest
import pandas as pd

from mcfacts.setup.setupdiskblackholes import (
    setup_disk_blackholes_location_NSC_powerlaw,
    setup_disk_blackholes_location_NSC_powerlaw_optimized,
)

TEST_SEED = 483445


def setup_powerlaw_params():
    """Return input parameters read from CSV."""
    inputs = pd.read_csv("tests/optimizations/powerlaw_inputs.csv", header=None)
    inputs[0] = inputs[0].astype(int)
    return [row.tolist() for _, row in inputs.iterrows()]


@pytest.mark.parametrize(
    "disk_bh_num, disk_radius_outer, disk_inner_stable_circ_orb, "
    "smbh_mass, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer",
    setup_powerlaw_params(),
)
def test_powerlaw(
    disk_bh_num,
    disk_radius_outer,
    disk_inner_stable_circ_orb,
    smbh_mass,
    nsc_radius_crit,
    nsc_density_index_inner,
    nsc_density_index_outer,
):
    """Compare continuous draws with the discrete reference's grid precision."""
    original = setup_disk_blackholes_location_NSC_powerlaw(
        int(disk_bh_num),
        disk_radius_outer,
        disk_inner_stable_circ_orb,
        smbh_mass,
        nsc_radius_crit,
        nsc_density_index_inner,
        nsc_density_index_outer,
        np.random.default_rng(TEST_SEED),
        volume_scaling=True,
    )

    optimized = setup_disk_blackholes_location_NSC_powerlaw_optimized(
        int(disk_bh_num),
        disk_radius_outer,
        disk_inner_stable_circ_orb,
        smbh_mass,
        nsc_radius_crit,
        nsc_density_index_inner,
        nsc_density_index_outer,
        np.random.default_rng(TEST_SEED),
        volume_scaling=True,
    )

    grid_spacing = (disk_radius_outer - disk_inner_stable_circ_orb) / 999999
    np.testing.assert_allclose(original, optimized, rtol=0, atol=grid_spacing)
