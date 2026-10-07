import numpy as np

from conftest import TEST_SEED
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.modules.merge import merge_blackholes_precession


def test_precession_remnant_attributes():
    """Test if the precession prescription returns expected values."""

    sm = SettingsManager()
    rng = np.random.default_rng(TEST_SEED)

    # Definite test arrays
    mass_1 = np.array([30., 20., 50.])
    mass_2 = np.array([10., 15., 45.])

    chi_1 = np.array([0.5, 0.1, 0.9])
    chi_2 = np.array([0.3, 0.7, 0.2])

    theta1 = np.array([0.2, np.pi / 2, 2.5])
    theta2 = np.array([1.0, 0.4, np.pi / 3])

    bin_sep = np.array([10., 20., 50.])
    bin_ecc = np.zeros(3)

    mass_merged, spin_merged, spin_angle_merged, v_kick, *_ = merge_blackholes_precession(
        mass_1, mass_2, chi_1, chi_2, theta1, theta2, bin_sep, bin_ecc,
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
