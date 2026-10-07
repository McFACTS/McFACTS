import numpy as np

from conftest import TEST_SEED
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.modules.merge import merge_blackholes_precession


def get_test_arrays():
    mass_1 = np.array([30., 20., 50.])
    mass_2 = np.array([10., 15., 45.])

    chi_1 = np.array([0.5, 0.1, 0.9])
    chi_2 = np.array([0.3, 0.7, 0.2])

    theta1 = np.array([0.2, np.pi / 2, 2.5])
    theta2 = np.array([1.0, 0.4, np.pi / 3])

    bin_sep = np.array([10., 20., 50.])
    bin_ecc = np.zeros(3)

    return mass_1, mass_2, chi_1, chi_2, theta1, theta2, bin_sep, bin_ecc


def test_precession_remnant_attributes():
    """Test if the precession prescription returns expected values."""

    sm = SettingsManager()
    rng = np.random.default_rng(TEST_SEED)
    mass_1, mass_2, chi_1, chi_2, theta_1, theta_2, bin_sep, bin_ecc = get_test_arrays()

    mass_merged, spin_merged, spin_angle_merged, v_kick, *_ = merge_blackholes_precession(
        mass_1, mass_2, chi_1, chi_2, theta_1, theta_2, bin_sep, bin_ecc,
        sm.smbh_mass, sm.r_g_in_meters, rng
    )

    # Remnant should lose some mass, but not too much
    assert np.all(mass_merged < mass_1 + mass_2)
    assert np.all(mass_merged > 0.9 * (mass_1 + mass_2))

    # Spin magnitude should be between [0, 1)
    assert np.all((spin_merged >= 0.) & (spin_merged < 1.))

    # Check if the spin angle is real
    assert np.all((spin_angle_merged >= 0.) & (spin_angle_merged <= np.pi))

    # Check that the kick velocity is positive, and finite
    assert np.all(np.isfinite(v_kick) & (v_kick >= 0.))


def test_precession_is_reproducible():
    """Test that precession only draws from the defined generator."""
    sm = SettingsManager()
    mass_1, mass_2, chi_1, chi_2, theta_1, theta_2, bin_sep, bin_ecc = get_test_arrays()

    global_bit_generator = np.random.get_bit_generator()

    outputs = []

    for _ in range(2):
        # Scramble the rng, which precession would otherwise draw from
        np.random.seed(None)

        outputs.append(merge_blackholes_precession(
            mass_1.copy(), mass_2.copy(), chi_1.copy(), chi_2.copy(), theta_1.copy(), theta_2.copy(),
            bin_sep.copy(), bin_ecc.copy(), sm.smbh_mass, sm.r_g_in_meters, np.random.default_rng(TEST_SEED)
        ))

    for first, second in zip(*outputs):
        assert np.array_equal(first, second)

    # make sure the global rng generator is restored after our call to merge_blackholes_precession
    assert np.random.get_bit_generator() is global_bit_generator
