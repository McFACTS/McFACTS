#!/usr/bin/env python3
"""Test saving and loading SettingsManager objects through the snapshot handlers"""
######## Imports ########
#### Standard Library ####
import warnings

#### Third Party ####
import pytest

#### Local ####
from mcfacts.inputs.settings_manager import SettingsManager, DEFAULT_SETTINGS, StaticSettingsProperty
from mcfacts.objects.snapshot import TxtSnapshotHandler, IniSnapshotHandler, HDF5SnapshotHandler

######## Setup ########
HANDLERS = [
    pytest.param(TxtSnapshotHandler, "settings.txt", id="txt"),
    pytest.param(IniSnapshotHandler, "settings.ini", id="ini"),
    pytest.param(HDF5SnapshotHandler, "settings.hdf5", id="hdf5"),
]

TEXT_HANDLERS = [
    pytest.param(TxtSnapshotHandler, "settings.txt", id="txt"),
    pytest.param(IniSnapshotHandler, "settings.ini", id="ini"),
]


def round_trip(handler_cls, file_name, settings, directory):
    handler_cls(settings=settings).save_settings(str(directory), file_name, settings)

    return handler_cls().load_settings(str(directory), file_name)


def assert_defaults_equal(a, b):
    for prop in DEFAULT_SETTINGS:
        # settings_file is mangled to "" by some handlers when unset
        if prop.name == "settings_file":
            continue
        assert getattr(a, prop.name) == getattr(b, prop.name), f"{prop.name} did not survive the round trip"


# region Default Settings Round Trip

# Check that untouched settings survive a save and load
@pytest.mark.parametrize("handler_cls, file_name", HANDLERS)
def test_round_trip_defaults(handler_cls, file_name, tmp_path):
    live = SettingsManager()
    assert_defaults_equal(live, round_trip(handler_cls, file_name, live, tmp_path))


# Check that a value changed with set_preprocessing survives a save and load
@pytest.mark.parametrize("handler_cls, file_name", HANDLERS)
def test_round_trip_set_preprocessing(handler_cls, file_name, tmp_path):
    live = SettingsManager()
    live.set_preprocessing("seed", 9001)

    loaded = round_trip(handler_cls, file_name, live, tmp_path)

    assert loaded.seed == 9001
    assert_defaults_equal(live, loaded)


# Check that each non-static default keeps its type after a save and load
@pytest.mark.parametrize("handler_cls, file_name", HANDLERS)
def test_round_trip_preserves_types(handler_cls, file_name, tmp_path):
    loaded = round_trip(handler_cls, file_name, SettingsManager(), tmp_path)

    for prop in DEFAULT_SETTINGS:
        if isinstance(prop, StaticSettingsProperty) or prop.name == "settings_file":
            continue
        assert type(getattr(loaded, prop.name)) is prop.prop_type, f"{prop.name} changed type"


# Check that a mixed-case setting name survives a save and load
@pytest.mark.parametrize("handler_cls, file_name", HANDLERS)
def test_round_trip_mixed_case_name(handler_cls, file_name, tmp_path):
    live = SettingsManager({"flag_initial_stars_BH_immortal": True})
    assert round_trip(handler_cls, file_name, live, tmp_path).flag_initial_stars_BH_immortal is True

# endregion


# region Aliased Settings Round Trip

# Check that a value given by alias survives a save and load under its canonical name
@pytest.mark.parametrize("handler_cls, file_name", HANDLERS)
def test_round_trip_value_set_by_alias(handler_cls, file_name, tmp_path):
    live = SettingsManager({"mean_harden_energy_delta": 0.5})
    loaded = round_trip(handler_cls, file_name, live, tmp_path)

    assert loaded.mu_circ_encounter == 0.5
    assert loaded.mean_harden_energy_delta == 0.5


# Check that snapshots are written with canonical names, not aliases
@pytest.mark.parametrize("handler_cls, file_name", TEXT_HANDLERS)
def test_snapshot_writes_canonical_names(handler_cls, file_name, tmp_path):
    live = SettingsManager({"mean_harden_energy_delta": 0.5})
    handler_cls(settings=live).save_settings(str(tmp_path), file_name, live)

    text = (tmp_path / file_name).read_text()
    assert "mu_circ_encounter" in text
    assert "mean_harden_energy_delta" not in text


# Check that a user-written ini file using an old (alias) name is applied
def test_ini_load_accepts_alias(tmp_path):
    (tmp_path / "user.ini").write_text("[bh]\nmean_harden_energy_delta = 0.5\n")
    assert IniSnapshotHandler().load_settings(str(tmp_path), "user.ini").mu_circ_encounter == 0.5


# Check that a user-written ini file using the canonical name is applied
def test_ini_load_accepts_canonical_name(tmp_path):
    (tmp_path / "user.ini").write_text("[bh]\nmu_circ_encounter = 0.5\n")
    assert IniSnapshotHandler().load_settings(str(tmp_path), "user.ini").mu_circ_encounter == 0.5

# endregion


# region Custom Category Round Trip

# Check that a custom category and its props survive a save and load (only the ini handler supports custom categories)
def test_round_trip_custom_category(tmp_path):
    live = SettingsManager()
    live.add_custom_category("mine", {"a": "hello"})

    loaded = round_trip(IniSnapshotHandler, "settings.ini", live, tmp_path)

    assert loaded.mine.a == "hello"


# Check that a custom category's aliases survive a save and load (only the ini handler supports custom categories)
def test_round_trip_custom_category_aliases(tmp_path):
    live = SettingsManager()
    live.add_custom_category("mine", {"a": "hello"}, {"b": "a"})

    loaded = round_trip(IniSnapshotHandler, "settings.ini", live, tmp_path)

    assert loaded.mine.b == "hello"


# Check that handlers without custom category support drop them instead of failing
@pytest.mark.parametrize("handler_cls, file_name", [
    pytest.param(TxtSnapshotHandler, "settings.txt", id="txt"),
    pytest.param(HDF5SnapshotHandler, "settings.hdf5", id="hdf5"),
])
def test_round_trip_custom_category_unsupported_is_dropped(handler_cls, file_name, tmp_path):
    live = SettingsManager()
    live.add_custom_category("mine", {"a": "hello"}, {"b": "a"})

    loaded = round_trip(handler_cls, file_name, live, tmp_path)

    assert "mine" not in loaded.categories


# Check that custom categories don't change the default settings after a save and load
@pytest.mark.parametrize("handler_cls, file_name", HANDLERS)
def test_round_trip_custom_category_keeps_defaults(handler_cls, file_name, tmp_path):
    live = SettingsManager()
    live.set_preprocessing("seed", 42)
    live.add_custom_category("mine", {"a": "hello"})

    loaded = round_trip(handler_cls, file_name, live, tmp_path)

    assert_defaults_equal(live, loaded)

# endregion
