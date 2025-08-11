import numpy as np
import fit_modeler
import mcfacts.external.sxs.surrogate as surrogate

import time

if __name__ == "__main__":
    mass_1 = 35.8
    mass_2 = 30.2
    spin_1_mag = 0.7
    spin_2_mag = 0.2
    spin_angle_1 = np.pi / 3
    spin_angle_2 = np.pi / 2
    phi_12 = np.pi / 4
    # This should be in units of mass_1 + mass_2
    bin_sep = 1000
    bin_inc = [0, 0, 1]
    bin_phase = 0
    # These next three are used to correct the remnant velocity;
    # If they are None, no correction is applied.
    bin_orb_a = None
    mass_SMBH = None
    spin_SMBH = None

    surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")

    spin_1 = spin_1_mag * np.array(
        [
            np.cos(phi_12) * np.sin(spin_angle_1),
            np.sin(phi_12) * np.sin(spin_angle_1),
            np.cos(spin_angle_1),
        ]
    )
    spin_2 = spin_2_mag * np.array(
        [
            np.cos(phi_12) * np.sin(spin_angle_2),
            np.sin(phi_12) * np.sin(spin_angle_2),
            np.cos(spin_angle_2),
        ]
    )
    print("Inputs:")
    print("-------")
    print("M_1 input = ", mass_1)
    print("M_2 input = ", mass_2)
    print("spin_1 input = ", spin_1)
    print("spin_2 input = ", spin_2)
    print()

    M_f, spin_f, v_f, mass_1_20Hz, mass_2_20Hz, spin_1_20Hz, spin_2_20Hz = surrogate.evolve_binary(
        mass_1,
        mass_2,
        spin_1_mag,
        spin_2_mag,
        spin_angle_1,
        spin_angle_2,
        phi_12,
        bin_sep,
        bin_inc,
        bin_phase,
        bin_orb_a,
        mass_SMBH,
        spin_SMBH,
        surrogate,
        verbose=True,
    )

    print()
    print("Outputs:")
    print("--------")
    print("M_f = ", M_f)
    print("spin_f = ", spin_f)
    print("v_f = ", v_f)
    print("M_1 @ 20Hz = ", mass_1_20Hz)
    print("M_2 @ 20Hz = ", mass_2_20Hz)
    print("spin_1 @ 20Hz = ", spin_1_20Hz)
    print("spin_2 @ 20Hz = ", spin_2_20Hz)
