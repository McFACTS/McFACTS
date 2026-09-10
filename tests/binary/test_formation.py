"""Check binary formation with real encounters and gravitational-wave estimates."""
import uuid

import numpy as np
import pytest

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.modules.formation import BinaryBlackHoleFormation, add_to_binary_obj
from mcfacts.objects.agn_object_array import AGNBlackHoleArray, FilingCabinet
from mcfacts.objects.agnobject import AGNBlackHole, AGNBinaryBlackHole


@pytest.mark.parametrize("fraction_retro", [0.0, 0.5, 1.0])
def test_new_binaries_are_circular(fraction_retro):
    settings = SettingsManager({"fraction_bin_retro": fraction_retro})
    singles = AGNBlackHoleArray(
        unique_id=np.array([uuid.UUID(int=i) for i in range(1, 6)]),
        mass=np.array([10., 20., 30., 40., 50.]),
        orb_a=np.array([1000., 1001., 3000., 3001., 10000.]),
    )
    cabinet = FilingCabinet()
    cabinet.set_array(settings.bh_prograde_array_name, singles)
    actor = BinaryBlackHoleFormation(settings=settings)
    actor.perform(0, 10000., 0., cabinet, None, np.random.default_rng(123))

    binaries = cabinet.get_array(settings.bbh_array_name)
    assert len(binaries) == 2
    np.testing.assert_array_equal(binaries.bin_ecc, [0., 0.])
    np.testing.assert_array_equal(
        binaries.bin_orb_ecc, np.full(2, settings.disk_bh_pro_orb_ecc_crit)
    )
    assert np.all(np.isfinite(binaries.gw_freq))
    assert np.all(binaries.gw_freq > 0)
    if fraction_retro in (0., 1.):
        np.testing.assert_array_equal(binaries.bin_orb_ang_mom, 1 - 2 * fraction_retro)
    np.testing.assert_array_equal(singles.unique_id, [uuid.UUID(int=5)])
    cabinet.consistency_check()


def test_no_binaries_without_close_encounters():
    settings = SettingsManager()
    cabinet = FilingCabinet()
    cabinet.set_array(settings.bh_prograde_array_name, AGNBlackHoleArray(
        unique_id=np.array([uuid.UUID(int=1), uuid.UUID(int=2)]),
        mass=np.array([10., 20.]),
        orb_a=np.array([1000., 10000.]),
    ))
    BinaryBlackHoleFormation(settings=settings).perform(
        0, 10000., 0., cabinet, None, np.random.default_rng(123)
    )
    assert settings.bbh_array_name not in cabinet
    assert len(cabinet.get_array(settings.bh_prograde_array_name)) == 2


def test_legacy_binary_formation_is_circular():
    singles = AGNBlackHole(
        mass=np.array([10., 20.]), orb_a=np.array([1000., 1001.]),
        spin=np.zeros(2), spin_angle=np.zeros(2), orb_inc=np.zeros(2),
        orb_ecc=np.zeros(2), orb_ang_mom=np.ones(2),
        orb_arg_periapse=np.zeros(2), time_passed=np.zeros(2),
        galaxy=np.zeros(2), id_start_val=1,
    )
    binaries = AGNBinaryBlackHole()
    add_to_binary_obj(
        binaries, singles, np.array([[1], [2]]), 2, 0., 1.e8, 0.1, 0.01,
        np.random.default_rng(123),
    )
    np.testing.assert_array_equal(binaries.bin_ecc, [0.])
    np.testing.assert_array_equal(binaries.bin_orb_ecc, [0.01])
    assert np.all(np.isfinite(binaries.gw_freq))
    assert np.all(binaries.gw_freq > 0)
