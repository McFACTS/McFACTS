import pytest
import numpy as np
import conftest as provider
from conftest import InputParameterSet
import mcfacts.physics.accretion as accretion

def accretion_bh_spin_params():
    #return input and expected values
    disk_bh_spin = provider.INPUT_PARAMETERS["bh_spin"][InputParameterSet.SINGLETON]
    disk_bh_orbital_eccentricity = provider.INPUT_PARAMETERS["bh_orbital_eccentricity"][InputParameterSet.SINGLETON]

    expected = [0.0, 0.34906585, 0.6981317,  1.04719755, 1.3962634,  1.74532925, 2.0943951,  2.44346095, 2.7925268,  3.14159265]

    return zip(disk_bh_spin, disk_bh_orbital_eccentricity, expected)

@pytest.mark.parametrize("disk_bh_spin_angles, disk_bh_orbital_eccentricity, expected", accretion_bh_spin_params())
def test_change_bh_spin(disk_bh_spin, disk_bh_pro_spin_angles, disk_bh_orbital_eccentricity, expected):
    """test change_bh_spin function"""

    change_bh_spin = accretion.change_bh_spin(
        np.array([disk_bh_spin]),
        np.array([disk_bh_pro_spin_angles]),
        1.0,
        0.1,
        0.02,
        1E4,
        np.array([disk_bh_orbital_eccentricity]),
        0.01
        )

    assert np.abs(change_bh_spin - expected) < 1.e4

