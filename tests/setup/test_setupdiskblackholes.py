"""Unit tests for setupdiskblackholes.py"""
import numpy as np
import pytest

# McFACTS modules
import conftest as provider
from conftest import InputParameterSet
from mcfacts.mcfacts_random_state import reset_random
from mcfacts.setup import setupdiskblackholes

# Set the seed using our test seed value to ensure values match expected results
rng = reset_random(provider.TEST_SEED)

def setup_disk_blackholes_location_NSC_powerlaw_param():
    """return input and excpected values"""

    # Expected values with which to compare
    expected = np.array([42573.683837683835, 31040.156492156493, 24968.329126329125, 7842.167384167384, 42286.01807401807, 47390.110614110614])

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

    params = np.hstack((input_grid, expected[:,np.newaxis]))
    # print(input_grid)
    # print(params)
    
    return params


@pytest.mark.parametrize("smbh_mass, disk_radius_outer, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer, expected", setup_disk_blackholes_location_NSC_powerlaw_param())
def test_setup_disk_blackholes_location_NSC_powerlaw(smbh_mass, disk_radius_outer, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer, expected):
    """test setup_disk_blackholes_location_NSC_powerlaw function"""

    # These parameters do not change so set them here.
    disk_bh_num = 1
    disk_inner_stable_circ_orb = 6

    # Draw a location
    location = setupdiskblackholes.setup_disk_blackholes_location_NSC_powerlaw(disk_bh_num,
                                  disk_radius_outer,
                                  disk_inner_stable_circ_orb,
                                  smbh_mass,
                                  nsc_radius_crit,
                                  nsc_density_index_inner,
                                  nsc_density_index_outer,
                                  volume_scaling=True)

    # Compare to the expected value.
    # Don't use boolean operator `==` because of possible machine precision limitations
    assert np.isclose(location, expected, rtol=1.e-4)