"""Unit test for feedback.py"""
import numpy as np
import pytest

from mcfacts.inputs.ReadInputs import construct_disk_direct, construct_disk_pAGN
import conftest as provider
from conftest import InputParameterSet
import mcfacts.physics.feedback as feedback

disk_surf_dens_func, disk_aspect_ratio_func, disk_opacity_func, sound_speed_func, disk_density_func, disk_pressure_grad_func, disk_omega_func, disk_surf_dens_func_log, temp_func, disk_model_properties, bonus_structures = construct_disk_pAGN("sirko_goodman", 1.e8, 50000, 0.01, 1.0)

def feedback_bh_hankla_param():
    """return input and expected values"""
    disk_bh_pro_orbs_a = provider.INPUT_PARAMETERS["bh_orbital_semi_major_axis_inner"][InputParameterSet.SINGLETON]

    expected = [np.nan, np.nan, 0.08781061, 0.07893705, 0.06938105, 0.0613231, 0.05491281, 0.04974382, 0.0454978, 0.04200165]

    return zip(disk_bh_pro_orbs_a, expected)


@pytest.mark.parametrize("disk_bh_pro_orbs_a, expected", feedback_bh_hankla_param())
def test_feedback_bh_hankla(disk_bh_pro_orbs_a, expected):
    """test feedback_bh_hankla function"""

    feedback_bh_hankla_values = feedback.feedback_bh_hankla(np.array([disk_bh_pro_orbs_a]), disk_surf_dens_func, disk_opacity_func, 1, 0.01, 50000.0)

    print(feedback_bh_hankla_values)

    assert (np.isnan(expected) and np.isnan(feedback_bh_hankla_values)) or np.abs(feedback_bh_hankla_values - expected) < 1.e-4
