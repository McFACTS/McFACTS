"""Unit test for test_tgw.py"""
import numpy as np
import pytest

from mcfacts.physics.binary.formation.add_new_binary import add_to_binary_array2
from mcfacts.utility import unit_test_variable_provider as provider
from mcfacts.utility.unit_test_variable_provider import InputParameterSet

INPUT_PARAMETERS = "disk_bins_bhbh, disk_bh_pro_orbs_a, disk_bh_pro_masses, disk_bh_pro_spins, \
                       disk_bh_pro_spin_angles, disk_bh_pro_gens, disk_bin_bhbh_pro_indices, bindex,\
                       fraction_bin_retro, smbh_mass"


def param_add_to_binary_array2():
    """return input and expected values"""

    #disk_bins_bhbh,
    #disk_bh_pro_orbs_a,
    #disk_bh_pro_masses,
    #disk_bh_pro_spins,
    #disk_bh_pro_spin_angles,
    #disk_bh_pro_gens,
    #disk_bin_bhbh_pro_indices,
    #bindex,
    #fraction_bin_retro,

    smbh_mass = provider.INPUT_PARAMETERS["smbh_mass"][InputParameterSet.BINARY]

    expected = []

    return zip(smbh_mass, expected)


@pytest.mark.parametrize("smbh_mass, expected", param_add_to_binary_array2())
def test_add_to_binary_array2(disk_bins_bhbh, disk_bh_pro_orbs_a, disk_bh_pro_masses, disk_bh_pro_spins,
                              disk_bh_pro_spin_angles, disk_bh_pro_gens, disk_bin_bhbh_pro_indices, bindex,
                              fraction_bin_retro, smbh_mass, expected):
    """test function"""

    return_value = add_to_binary_array2(disk_bins_bhbh, disk_bh_pro_orbs_a, disk_bh_pro_masses, disk_bh_pro_spins,
                                        disk_bh_pro_spin_angles, disk_bh_pro_gens, disk_bin_bhbh_pro_indices, bindex,
                                        fraction_bin_retro, smbh_mass)

