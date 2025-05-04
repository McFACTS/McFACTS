#!/usr/bin/env python3
"""Test for bh_near_smbh function
    in physics/dynamics.py
"""
######## Imports ########
#### Standard Library ####
import time
#### Third-Party ####
import numpy as np
#### Homemade ####
#### Local ####
from mcfacts.physics.dynamics import bh_near_smbh

######## Setup ########
INPUT_DATA = {
    "smbh_mass" : 100000000.0,
    "disk_bh_pro_orbs_a" : np.asarray(
[6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
 6., 6.]),
    "disk_bh_pro_masses" : np.asarray(
[10.90040936, 15.44896236, 19.72743684, 37.50105233, 15.88936831, 11.32254387,
 22.46305068, 10.1351487,  36.27818419, 26.40840304, 16.37462098, 12.53039078,
 30.41774755, 16.14268923, 11.14170817, 13.37738356, 16.87312757, 10.76633767,
 17.73980057, 10.27667614, 35.27727791, 10.1824792,  20.93057533, 12.27675557,
 12.00289819, 37.52689174]),
    "disk_bh_pro_orbs_ecc" : np.asarray(
[0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999,
 0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999,
 0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999,
 0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999, 0.99999999,
 0.99999999, 0.99999999]),
    "timestep_duration_yr" : 10000.0,
    "inner_disk_outer_radius" : 50.0,
    "disk_inner_stable_circ_orb" : 6.0,
}


OUTPUT_DATA = {
    "disk_bh_pro_orbs_a" : np.asarray(
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
}

######## Functions ########

######## Tests ########
def test_inputs():
    for key in INPUT_DATA:
        print(key)
    for key in OUTPUT_DATA:
        print(key)
        if key in INPUT_DATA:
            assert not np.allclose(INPUT_DATA[key],OUTPUT_DATA[key]), \
                "test bh_near_smbh: Bad OUTPUT DATA\n" +\
                "All input and output values are identical"

def test_function():
    tic = time.perf_counter()
    disk_bh_pro_orbs_a = bh_near_smbh(
        INPUT_DATA["smbh_mass"],
        INPUT_DATA["disk_bh_pro_orbs_a"].copy(),
        INPUT_DATA["disk_bh_pro_masses"].copy(),
        INPUT_DATA["disk_bh_pro_orbs_ecc"].copy(),
        INPUT_DATA["timestep_duration_yr"],
        INPUT_DATA["inner_disk_outer_radius"],
        INPUT_DATA["disk_inner_stable_circ_orb"],
    )
    toc = time.perf_counter()
    assert np.allclose(disk_bh_pro_orbs_a, OUTPUT_DATA["disk_bh_pro_orbs_a"])
    print("test_bh_near_smbh: pass")
    diff_mask = ~(np.isclose(disk_bh_pro_orbs_a, INPUT_DATA["disk_bh_pro_orbs_a"]))
    print(f"{np.sum(diff_mask)} / {np.size(diff_mask)} changes")
    print(f"Time: {toc-tic} seconds!")

def test_performance(n=9001):
    tic = time.perf_counter()
    disk_bh_pro_orbs_a = bh_near_smbh(
        INPUT_DATA["smbh_mass"],
        np.tile(INPUT_DATA["disk_bh_pro_orbs_a"],n),
        np.tile(INPUT_DATA["disk_bh_pro_masses"],n),
        np.tile(INPUT_DATA["disk_bh_pro_orbs_ecc"],n),
        INPUT_DATA["timestep_duration_yr"],
        INPUT_DATA["inner_disk_outer_radius"],
        INPUT_DATA["disk_inner_stable_circ_orb"],
    )
    toc = time.perf_counter()
    print(f"Time: {toc-tic} seconds!")


######## Algorithm ########
def tests():
    test_inputs()
    test_function()
    #test_performance()

######## Main ########
def main():
    tests()
    return
######## Execution ########
if __name__ == "__main__":
    main()
