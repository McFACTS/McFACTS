"""Unit tests for analytical_velocity"""
import numpy as np
import pytest
import pandas as pd

from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.analytical_velocity import (
    analytical_kick_velocity,
    analytical_kick_velocity_optimized,
)

SEED = 483445


def parse_array(cell):
    if isinstance(cell, str) and cell.startswith('['):
        cleaned = cell.strip('[]')
        return np.fromstring(cleaned, sep=' ')
    return cell


def setup_analytical_velocity_params():
    """Return input parameters read from CSV."""
    inputs = pd.read_csv("tests/analytical_inputs.csv", header=None)
    params = []
    for _, row in inputs.iterrows():
        params.append(tuple(parse_array(row[i]) for i in range(6)))
    return params


@pytest.mark.parametrize(
    "mass_1, mass_2, spin_1, spin_2, spin_angle_1, spin_angle_2",
    setup_analytical_velocity_params(),
)
def test_analytical_velocity(
    mass_1, mass_2, spin_1, spin_2, spin_angle_1, spin_angle_2
):
    """Test that optimized analytical_kick_velocity matches original."""
    rng.seed(seed=SEED)
    original = analytical_kick_velocity(
        mass_1, mass_2, spin_1, spin_2, spin_angle_1, spin_angle_2
    )

    rng.seed(seed=SEED)
    optimized = analytical_kick_velocity_optimized(
        mass_1, mass_2, spin_1, spin_2, spin_angle_1, spin_angle_2
    )

    assert np.allclose(original, optimized, rtol=1e-9)

