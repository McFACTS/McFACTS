"""Unit tests for accretion.py"""
import numpy as np
import pytest

# McFACTS modules
import conftest as provider
from conftest import InputParameterSet
from mcfacts.mcfacts_random_state import reset_random
from mcfacts.physics import accretion

# Set the seed using our test seed value to ensure values match expected results
rng = reset_random(provider.TEST_SEED)


def setup_change_bh_mass_param():
    """Return input and expected values"""

    # Expected values with which to compare
    expected = np.loadtxt('tests/bh_change_mass_outputs.csv', delimiter=',')

    # Get input values from the provider
    bh_pro_masses = provider.INPUT_PARAMETERS["bh_masses"][InputParameterSet.SINGLETON]
    bh_eddington_ratio = provider.INPUT_PARAMETERS["bh_eddington_ratio"][InputParameterSet.SINGLETON]
    bh_eddington_mass_growth_rate = provider.INPUT_PARAMETERS["bh_eddington_ratio_mass_growth_rate"][InputParameterSet.SINGLETON]
    timestep_duration_year = provider.INPUT_PARAMETERS["timestep_duration_yr"][InputParameterSet.SINGLETON]

    # Construct grid of all possible combinations of input parameters
    grids = np.meshgrid(bh_pro_masses, bh_eddington_ratio, bh_eddington_mass_growth_rate, timestep_duration_year, indexing='ij')
    input_grid = np.array([grid.flatten() for grid in grids]).T

    params = np.hstack((input_grid, expected[:, np.newaxis]))

    return params


@pytest.mark.parametrize("bh_pro_masses, bh_eddington_ratio, bh_eddington_mass_growth_rate, timestep_duration_year, expected", setup_change_bh_mass_param())
def test_change_bh_mass(bh_pro_masses, bh_eddington_ratio, bh_eddington_mass_growth_rate, timestep_duration_year, expected):
    """Test change_bh_mass function"""

    new_masses = accretion.change_bh_mass(
        bh_pro_masses,
        bh_eddington_ratio,
        bh_eddington_mass_growth_rate,
        timestep_duration_year
    )

    # Compare to the expected value
    # Don't use boolean operator `==` because of possible machine precision limitations
    assert np.isclose(new_masses, expected, rtol=1.e-9)