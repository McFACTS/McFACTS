import pytest
import numpy as np
import conftest as provider
from conftest import InputParameterSet
import mcfacts.physics.accretion as accretion

#def accretion_bh_spin_angle_params():
    #return input and expected values
    #disk_bh_spin_angle = provider.INPUT_PARAMETERS["bh_spin_angles"][InputParameterSet.SINGLETON]
    #disk_bh_orbital_eccentricity = provider.INPUT_PARAMETERS["bh_orbital_eccentricity"][InputParameterSet.SINGLETON]

    #expected_spin_angle = [0.0, 0.34906585, 0.6981317,  1.04719755, 1.3962634,  1.74532925, 2.0943951,  2.44346095, 2.7925268,  3.14159265]
    
    #return zip(disk_bh_spin_angle, disk_bh_orbital_eccentricity, expected_spin_angle)

#@pytest.mark.parametrize("disk_bh_spin_angle, disk_bh_orbital_eccentricity, expected_spin_angle", accretion_bh_spin_angle_params())

def accretion_bh_spin_params():
    #return input and expected values
    disk_bh_spin = provider.INPUT_PARAMETERS["bh_spins"][InputParameterSet.SINGLETON]
    disk_bh_spin_angle = provider.INPUT_PARAMETERS["bh_spin_angles"][InputParameterSet.SINGLETON]
    disk_bh_orbital_eccentricity = provider.INPUT_PARAMETERS["bh_orbital_eccentricity"][InputParameterSet.SINGLETON]

    expected_spin = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    expected_spin_angle = [0.0, 0.34906585, 0.6981317,  1.04719755, 1.3962634,  1.74532925, 2.0943951,  2.44346095, 2.7925268,  3.14159265]

    return zip(disk_bh_spin, disk_bh_spin_angle, disk_bh_orbital_eccentricity, expected_spin, expected_spin_angle)

@pytest.mark.parametrize("disk_bh_spin, disk_bh_spin_angle, disk_bh_orbital_eccentricity, expected_spin, expected_spin_angle", accretion_bh_spin_params())

def test_change_bh_spin(disk_bh_spin, disk_bh_spin_angle, disk_bh_orbital_eccentricity, expected_spin, expected_spin_angle):
    """test change_bh_spin function"""

    change_bh_spin, change_bh_spin_angle = accretion.change_bh_spin(
        np.array([disk_bh_spin]),
        np.array([disk_bh_spin_angle]),
        1.0,
        0.1,
        0.02,
        1E4,
        np.array([disk_bh_orbital_eccentricity]),
        0.01
        )

    assert np.abs(change_bh_spin - expected_spin) < 1.e4
    assert np.abs(change_bh_spin_angle - expected_spin_angle) < 1.e4