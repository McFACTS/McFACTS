import numpy as np

from conftest import TEST_SEED
from mcfacts.utilities.checks import spin_check


def test_spin_check_is_reproducible():
    """Test that the spin_check function only draws from the generator and redraws low spins into range."""
    gen_1 = np.array([1., 2., 3., 2.])
    gen_2 = np.array([1., 1., 1., 2.])
    spin_merged = np.array([0.1, 0.2, 0.3, 0.9])

    outputs = []

    for _ in range(2):
        # Scramble the global rng
        np.random.seed(None)

        outputs.append(spin_check(gen_1, gen_2, spin_merged, np.random.default_rng(TEST_SEED)))

    assert np.array_equal(outputs[0], outputs[1])

    new_spins = outputs[0]

    assert new_spins[0] == 0.1
    assert new_spins[3] == 0.9

    assert 0.75 <= new_spins[1] <= 0.85
    assert 0.85 <= new_spins[2] <= 0.95
