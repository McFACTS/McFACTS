"""Keep interpolation defined through the requested tabulated disk edge."""
from importlib import resources

import numpy as np
import pytest

from mcfacts.inputs import data
from mcfacts.inputs.ReadInputs import load_disk_arrays
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.disk import AGNDisk


@pytest.mark.parametrize("model, outer_radius", [
    ("sirko_goodman", 1000.), ("sirko_goodman", 50000.),
    ("thompson_etal", 1000.), ("thompson_etal", 30000.),
])
def test_tables_bracket_outer_radius(model, outer_radius):
    for table in load_disk_arrays(model, outer_radius):
        radius = table[0]
        assert radius[-2] < outer_radius <= radius[-1]
        assert np.all(np.diff(radius) > 0)

    disk = AGNDisk(SettingsManager({"flag_use_pagn": False,
                                   "disk_model_name": model,
                                   "disk_radius_outer": outer_radius}))
    probe = np.array([outer_radius - 1., outer_radius])
    for name in ["surface_density", "aspect_ratio", "opacity", "sound_speed",
                 "density", "omega", "temperature"]:
        values = getattr(disk, name)(probe)
        assert np.all(np.isfinite(values)), name
        assert np.all(values > 0), name


@pytest.mark.parametrize("model", ["sirko_goodman", "thompson_etal"])
def test_exact_boundary_keeps_the_existing_sample(model):
    source = np.loadtxt(resources.files(data) / f"{model}_surface_density.txt")
    edge = source[100, 1]
    loaded = load_disk_arrays(model, edge)[0]
    np.testing.assert_array_equal(loaded, source[:101].T[::-1])


def test_request_beyond_table_preserves_source_and_nan_boundary():
    source = np.loadtxt(resources.files(data) / "sirko_goodman_surface_density.txt")
    edge = float(source[-1, 1] * 2)
    loaded = load_disk_arrays("sirko_goodman", edge)[0]
    np.testing.assert_array_equal(loaded, source.T[::-1])
    disk = AGNDisk(SettingsManager({"flag_use_pagn": False, "disk_radius_outer": edge}))
    assert np.isnan(disk.surface_density(np.array([edge]))).all()
