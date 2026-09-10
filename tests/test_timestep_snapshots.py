"""Check selective snapshots using the real galaxy and text writer."""
import uuid

import numpy as np
import pytest

from mcfacts import simulation
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.agn_object_array import AGNBlackHoleArray
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.snapshot import TxtSnapshotHandler
from mcfacts.objects.timeline import SimulationTimeline


def populated_galaxy(tmp_path, **overrides):
    settings = SettingsManager({"save_each_timestep": True, **overrides})
    galaxy = Galaxy(123, str(tmp_path), "0", settings=settings)
    for name, start, count in [("blackholes_prograde", 1, 2), ("blackholes_binary_gw", 10, 20)]:
        galaxy.filing_cabinet.set_array(name, AGNBlackHoleArray(
            unique_id=np.array([uuid.UUID(int=i) for i in range(start, start + count)]),
            mass=np.full(count, 20.),
        ))
    galaxy.populated = True
    return galaxy


def test_default_snapshots_still_save_every_array_and_step(tmp_path):
    galaxy = populated_galaxy(tmp_path)
    assert galaxy.snapshot_handler.settings is galaxy.settings
    galaxy.run(SimulationTimeline("test", timesteps=5, timestep_length=1.), None)
    assert len(list(tmp_path.rglob("*_blackholes_prograde.txt"))) == 5
    assert len(list(tmp_path.rglob("*_blackholes_binary_gw.txt"))) == 5


def test_selected_snapshots_keep_final_state_complete(tmp_path):
    galaxy = populated_galaxy(
        tmp_path, save_every_n_timesteps=3,
        timestep_snapshot_arrays=" blackholes_prograde,not_created_yet ", save_state=True,
    )
    original = galaxy.filing_cabinet.get_array("blackholes_binary_gw")
    galaxy.run(SimulationTimeline("test", timesteps=5, timestep_length=1.), None)

    snapshots = sorted((tmp_path / "gal00" / "gal00_s00_to_s01").glob("*.txt"))
    assert [p.name for p in snapshots] == [
        "gal00_s00_to_s01_t02_blackholes_prograde.txt",
        "gal00_s00_to_s01_t04_blackholes_prograde.txt",
    ]
    # The final state still includes the history excluded from intermediate files.
    final = tmp_path / "gal00" / "gal00_s01_blackholes_binary_gw.txt"
    assert final.exists()
    assert galaxy.filing_cabinet.get_array("blackholes_binary_gw") is original
    assert len(original) == 20
    # The selected text snapshot can be read by the existing loader.
    loaded = TxtSnapshotHandler().load_cabinet(snapshots[0].parent, "gal00_s00_to_s01_t02")
    np.testing.assert_array_equal(loaded["blackholes_prograde"]["mass"], [20., 20.])


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_interval_fails_before_running(tmp_path, interval):
    galaxy = populated_galaxy(tmp_path, save_every_n_timesteps=interval)
    with pytest.raises(ValueError, match="save_every_n_timesteps"):
        galaxy.run(SimulationTimeline("test", timesteps=5, timestep_length=1.), None)
    assert galaxy.timeline_history == []
    assert list(tmp_path.iterdir()) == []


def test_save_each_timestep_off_writes_no_intermediate_files(tmp_path):
    galaxy = populated_galaxy(tmp_path, save_each_timestep=False)
    galaxy.run(SimulationTimeline("test", timesteps=5, timestep_length=1.), None)
    assert list(tmp_path.iterdir()) == []


def test_simulation_shares_snapshot_handler(tmp_path, monkeypatch):
    settings = SettingsManager({"galaxy_num": 1, "active_timestep_num": 0,
                                "flag_use_pagn": False, "output_dir": str(tmp_path / "run")})
    handlers = []

    class RecordingHandler(TxtSnapshotHandler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            handlers.append(self)

    class InspectGalaxy(Galaxy):
        def __init__(self, **kwargs):
            assert kwargs["snapshot_handler"] is handlers[0]
            super().__init__(**kwargs)

    monkeypatch.setattr(simulation, "TxtSnapshotHandler", RecordingHandler)
    monkeypatch.setattr(simulation, "Galaxy", InspectGalaxy)
    simulation.main(settings)
    assert len(handlers) == 1
