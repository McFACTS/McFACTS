"""Can I write a test?"""

import numpy as np

from mcfacts.physics import disk_capture

def main():
    """seeing if I can call shit and get output values?
    """
    smbh_mass = 1.0e8
    disk_inner_stable_circ_orb = 6.0
    disk_radius_outer = 5.e4
    disk_surf_density_func 
    nsc_radius_crit = 0.25
    nsc_density_index_inner = 1.75
    nsc_density_index_outer = 2.5
    nsc_bh_imf_mode = 10.0
    nsc_bh_imf_max_mass = 40.0
    nsc_bh_imf_powerlaw_index = 2.0
    mass_pile_up = 35.0
    nsc_imf_bh_method = 'default'
    disk_bh_num = 201
    disk_aspect_ratio_avg = 0.03


    rate = disk_capture.disk_capture_rate(smbh_mass, disk_inner_stable_circ_orb, disk_radius_outer, disk_surf_density_func, nsc_radius_crit, nsc_density_index_inner, nsc_density_index_outer, nsc_bh_imf_mode, nsc_bh_imf_max_mass, nsc_bh_imf_powerlaw_index, mass_pile_up, nsc_imf_bh_method, disk_bh_num, disk_aspect_ratio_avg)

    print(rate)

if __name__ == "__main__":
    main()