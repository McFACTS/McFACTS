"""Unit test for accretion.change_bh_spin_magnitudes"""
import numpy as np
import pytest

from mcfacts.physics import accretion

def param_change_bh_spin_magnitudes():
    """return input and expected values"""
    
    disk_bh_pro_orbs_ecc = np.array([0.0, 0.01, 0.03, 0.3, 0.5, 0.7, 0.9])
    disk_bh_pro_spins = 0.7 * np.ones(len(disk_bh_pro_orbs_ecc))
    # actually want to test on spins -1.0, -0.7, -0.1, 0.0, 0.1, 0.7, 1.0
    disk_bh_eddington_ratio = 0.5
    # actually want to test on [0.01, 0.1, 0.5, 1.0, 5.0]
    disk_bh_torque_condition = 0.01
    # actually want to test on [0.01, 0.1]
    timestep_duration_yr = 1e4
    # actually want to test on [1e2, 1e3, 1e4, 1e5]
    disk_bh_pro_orbs_ecc_crit = 0.03
    # this is fine as long as ecc array remains same--want a crit ecc in the middle
    
    "Why aren't the masses passed in, if we need to know the fraction of initial mass accreted?"

    expected = [0.70005, 0.70005, 0.70005, 0.69995, 0.69995, 0.69995, 0.69995]

    return zip(disk_bh_pro_spins,
               disk_bh_eddington_ratio,
               disk_bh_torque_condition,
               timestep_duration_yr,
               disk_bh_pro_orbs_ecc,
               disk_bh_pro_orbs_ecc_crit,
               expected)


@pytest.mark.parametrize("input_values, expected", param_change_bh_spin_magnitudes())
def test_change_bh_spin_magnitudes(disk_bh_pro_spins,
                                   disk_bh_eddington_ratio,
                                   disk_bh_torque_condition,
                                   timestep_duration_yr,
                                   disk_bh_pro_orbs_ecc,
                                   disk_bh_pro_orbs_ecc_crit,
                                   expected):
    """test function"""

    assert np.abs(accretion.change_bh_spin_magnitudes(disk_bh_pro_spins,
                                                      disk_bh_eddington_ratio,
                                                      disk_bh_torque_condition,
                                                      timestep_duration_yr,
                                                      disk_bh_pro_orbs_ecc,
                                                      disk_bh_pro_orbs_ecc_crit, 50) - expected) < 1.e4