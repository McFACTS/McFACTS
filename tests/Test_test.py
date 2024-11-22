"""Can I write a test?"""

import numpy as np

from mcfacts.physics import accretion

def main():
    """seeing if I can call shit and get output values?
    """
    disk_bh_pro_orbs_ecc = np.array([0.0, 0.01, 0.03, 0.3, 0.5, 0.7, 0.9])
    disk_bh_pro_spins = 0.7 * np.ones(len(disk_bh_pro_orbs_ecc))
    disk_bh_eddington_ratio = 0.5 # 0.01 [0.1, 0.5, 1.0, 5.0]
    disk_bh_torque_condition = 0.01 # [0.1]
    timestep_duration_yr = 1e4 # [1e2, 1e3, 1e4, 1e5]
    disk_bh_pro_orbs_ecc_crit = 0.03
    
    output_values = accretion.change_bh_spin_magnitudes(
        disk_bh_pro_spins,
        disk_bh_eddington_ratio,
        disk_bh_torque_condition,
        timestep_duration_yr,
        disk_bh_pro_orbs_ecc,
        disk_bh_pro_orbs_ecc_crit
    )

    print(output_values)

if __name__ == "__main__":
    main()