#!/usr/bin/env python3
"""Test for bin_recapture function
    in physics/dynamics.py
"""
######## Imports ########
#### Standard Library ####
import time
#### Third-Party ####
import numpy as np
#### Homemade ####
#### Local ####
from mcfacts.physics.dynamics import bin_recapture

######## Setup ########
INPUT_DATA = {
    "bin_mass_1_all" : np.asarray(
    [14.54453867, 22.94940789, 15.32811881, 10.7847671,  15.44589222, 11.49062996,
 16.50199124, 12.35558125, 12.83018659, 12.28839276, 36.13006406, 10.17147256]),
    "bin_mass_2_all" : np.asarray(
    [18.12356777, 14.10766309, 11.283557,   10.20085314, 12.64002858, 26.76381266,
 12.68593058, 37.78197953, 21.59693903, 16.18398008, 29.00691522, 14.91878274]),
    "bin_orb_a_all" : np.asarray(
    [22775.78053854, 22357.01786987, 11215.85839829, 10897.83143391,
 19834.60481211,  7569.53974609,  6791.39049089, 20797.77090857,
  9988.9847044,  16623.732495,   26425.79700587,  1370.961792,  ]),
    "bin_orb_inc_all" : np.asarray(
    [1.90854002, 0.12038371, 0.63286273, 0.58833443, 0.43494666, 0.36873407,
 1.08070448, 0.16185984, 0.28563825, 0.08478488, 0.2627608,  0.        ]),
    "timestep_duration_yr" : 10000.0,
}

OUTPUT_DATA = {
    "bin_orb_inc_all" : np.asarray(
    [1.90854002, 0.12018424, 0.63286273, 0.58833443, 0.43494666, 0.36873407,
 1.08070448, 0.16152228, 0.28563825, 0.08077187, 0.26185622, 0.        ]),
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
                "test bin_recapture: Bad OUTPUT DATA\n" +\
                "All input and output values are identical"

def test_function():
    tic = time.perf_counter()
    bin_orb_inc_all = bin_recapture(
            INPUT_DATA["bin_mass_1_all"].copy(),
            INPUT_DATA["bin_mass_2_all"].copy(),
            INPUT_DATA["bin_orb_a_all"].copy(),
            INPUT_DATA["bin_orb_inc_all"].copy(),
            INPUT_DATA["timestep_duration_yr"],
    )
    toc = time.perf_counter()
    assert np.allclose(bin_orb_inc_all, OUTPUT_DATA["bin_orb_inc_all"])
    print("test_bin_recapture: pass")
    diff_mask = ~(np.isclose(bin_orb_inc_all, INPUT_DATA["bin_orb_inc_all"]))
    print(f"{np.sum(diff_mask)} / {np.size(diff_mask)} changes")
    print(f"Time: {toc-tic} seconds!")

def test_performance(n=9001):
    tic = time.perf_counter()
    bin_orb_inc_all = bin_recapture(
            np.tile(INPUT_DATA["bin_mass_1_all"],n),
            np.tile(INPUT_DATA["bin_mass_2_all"],n),
            np.tile(INPUT_DATA["bin_orb_a_all"],n),
            np.tile(INPUT_DATA["bin_orb_inc_all"],n),
            INPUT_DATA["timestep_duration_yr"],
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
