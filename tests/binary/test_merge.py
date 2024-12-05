"""Unit test for test_tgw.py"""
import numpy as np
import pytest

from mcfacts.physics.binary.merge import normalize_tgw
import conftest as provider
from conftest import InputParameterSet


# def param_normalize_tgw():
#     """return input and expected values"""
#     smbh_mass = provider.INPUT_PARAMETERS["smbh_mass"][InputParameterSet.BASE]
#
#     expected = [24404108.338690642, 244041083386.9064, 2440410833869065.0, 2.440410833869064e+19, 2.4404108338690643e+23, 19767327754339.426]
#
#     return zip(smbh_mass, expected)
#
#
# @pytest.mark.parametrize("smbh_mass, expected", param_normalize_tgw())
# def test_normalize_tgw(smbh_mass, expected):
#     """test function"""
#
#     assert np.abs(normalize_tgw(smbh_mass, 50) - expected) < 1.e4
