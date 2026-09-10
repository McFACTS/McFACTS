"""Exercise the simulation's population settings and seeded star setup."""
import numpy as np

from mcfacts import simulation
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.disk import AGNDisk
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.populators import SingleStarPopulator


def test_simulation_passes_settings_to_populators(tmp_path, monkeypatch):
    settings = SettingsManager({
        "galaxy_num": 1, "active_timestep_num": 0,
        "flag_use_pagn": False, "output_dir": str(tmp_path / "run"),
        "nsc_bh_spin_dist_mu": 0.4, "nsc_bh_spin_dist_sigma": 0.,
        "disk_star_mass_min_init": 1., "disk_star_mass_max_init": 2.,
    })
    populations = []

    class InspectGalaxy(Galaxy):
        def populate(self, populators, agn_disk, **kwargs):
            super().populate(populators, agn_disk, **kwargs)
            for populator in populators:
                assert populator.settings is settings
            blackholes = self.filing_cabinet.get_array(settings.bh_array_name)
            stars = self.filing_cabinet.get_array(settings.star_array_name)
            np.testing.assert_array_equal(blackholes.spin, 0.4)
            assert np.all((stars.mass >= 1.) & (stars.mass <= 2.))
            populations.append((len(blackholes), len(stars)))

    monkeypatch.setattr(simulation, "Galaxy", InspectGalaxy)
    simulation.main(settings)
    assert len(populations) == 1
    assert all(count > 0 for count in populations[0])
    assert (tmp_path / "run" / "settings.ini").exists()


def test_coalescing_stars_uses_supplied_generator():
    settings = SettingsManager({
        "flag_coalesce_initial_stars": True, "flag_use_pagn": False,
    })
    disk = AGNDisk(settings)
    populator = SingleStarPopulator(settings=settings)
    first = populator.populate(disk, np.random.default_rng(123))
    repeated = populator.populate(disk, np.random.default_rng(123))
    different = populator.populate(disk, np.random.default_rng(456))
    assert len(first) > 0
    for name, values in first.get_super_dict().items():
        np.testing.assert_array_equal(values, repeated.get_super_dict()[name])
    assert not np.array_equal(first.unique_id, different.unique_id)
