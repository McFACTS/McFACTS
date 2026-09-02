"""Unit tests for setupdiskblackholes.py"""
import numpy as np
import pytest

# McFACTS modules
import conftest as provider
from conftest import InputParameterSet
from mcfacts.setup import setupdiskblackholes


def setup_disk_blackholes_location_NSC_powerlaw_param():
    """return input values"""

    # Get input parameters from the provider
    smbh_mass = provider.INPUT_PARAMETERS["smbh_mass"][InputParameterSet.BASE]
    disk_radius_outer = provider.INPUT_PARAMETERS["disk_radius_outer"][InputParameterSet.BASE]
    nsc_radius_crit = provider.INPUT_PARAMETERS["nsc_radius_crit"][InputParameterSet.BASE]
    nsc_density_index_inner = provider.INPUT_PARAMETERS["nsc_density_index_inner"][InputParameterSet.BASE]
    nsc_density_index_outer = provider.INPUT_PARAMETERS["nsc_density_index_outer"][InputParameterSet.BASE]

    # Construct the grid of all possible combinations of input parameters
    grids = np.meshgrid(smbh_mass, disk_radius_outer, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer, indexing='ij')
    # input_grid = np.array([grid.flatten() for grid in grids]).T.tolist()
    input_grid = np.array([grid.flatten() for grid in grids]).T

    return input_grid


@pytest.mark.parametrize("smbh_mass, disk_radius_outer, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer", setup_disk_blackholes_location_NSC_powerlaw_param())
def test_setup_disk_blackholes_location_NSC_powerlaw(smbh_mass, disk_radius_outer, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer):
    """test setup_disk_blackholes_location_NSC_powerlaw function"""

    # These parameters do not change so set them here.
    disk_bh_num = 1
    disk_inner_stable_circ_orb = 6

    # Both versions are handed their own generator seeded with the same value, so they draw
    # from the same stream and any difference between them comes from the implementations.
    location = setupdiskblackholes.setup_disk_blackholes_location_NSC_powerlaw(disk_bh_num,
                                  disk_radius_outer,
                                  disk_inner_stable_circ_orb,
                                  smbh_mass,
                                  nsc_radius_crit,
                                  nsc_density_index_inner,
                                  nsc_density_index_outer,
                                  np.random.default_rng(provider.TEST_SEED),
                                  volume_scaling=True)

    location_optimized = setupdiskblackholes.setup_disk_blackholes_location_NSC_powerlaw_optimized(disk_bh_num,
                                  disk_radius_outer,
                                  disk_inner_stable_circ_orb,
                                  smbh_mass,
                                  nsc_radius_crit,
                                  nsc_density_index_inner,
                                  nsc_density_index_outer,
                                  np.random.default_rng(provider.TEST_SEED),
                                  volume_scaling=True)

    # Don't use boolean operator `==` because of possible machine precision limitations
    assert np.allclose(location, location_optimized, rtol=1.e-9)