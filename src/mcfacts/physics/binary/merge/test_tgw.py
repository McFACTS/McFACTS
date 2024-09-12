"""Unit test for test_tgw.py"""
import numpy as np
import pytest

from mcfacts.physics.binary.merge import tgw
from mcfacts.utility import unit_test_variable_provider as provider
from mcfacts.utility.unit_test_variable_provider import InputParameterSet


def param_normalize_tgw():
    """return input and expected values"""
    smbh_mass = provider.INPUT_PARAMETERS["smbh_mass"][InputParameterSet.BASE]

    expected = [24404108.338690642, 244041083386.9064, 2440410833869065.0, 2.440410833869064e+19]

    return zip(smbh_mass, expected)


@pytest.mark.parametrize("smbh_mass, expected", param_normalize_tgw())
def test_normalize_tgw(smbh_mass, expected):
    """test function"""

    assert np.abs(tgw.normalize_tgw(smbh_mass) - expected) < 1.e4
