"""Unit test for feedback.py"""
import numpy as np
import pytest

from mcfacts.inputs.ReadInputs import construct_disk_direct
import conftest as provider
from conftest import InputParameterSet
import mcfacts.physics.feedback as feedback


def feedback_bh_hankla_param():
    """return input and expected values"""
    disk_bh_pro_orbs_a = provider.INPUT_PARAMETERS["bh_orbital_semi_major_axis_inner"][InputParameterSet.SINGLETON]

    expected = [0.01557252, 0.00485183, 0.0035772, 0.00279144, 0.00228692, 0.00194053, 0.00169016, 0.00150413, 0.00136354, 0.00125561]

    return zip(disk_bh_pro_orbs_a, expected)


@pytest.mark.parametrize("disk_bh_pro_orbs_a, expected", feedback_bh_hankla_param())
def test_feedback_bh_hankla(disk_bh_pro_orbs_a, expected):
    """test feedback_bh_hankla function"""

    surf_dens_func, spect_ratio_func, opacity_func, model_properties = construct_disk_direct("sirko_goodman", 50000, verbose=False)

    feedback_bh_hankla_values = feedback.feedback_bh_hankla(np.array([disk_bh_pro_orbs_a]), surf_dens_func, opacity_func, 1, 0.01, 50000.0)

    assert np.abs(feedback_bh_hankla_values - expected) < 1.e4
