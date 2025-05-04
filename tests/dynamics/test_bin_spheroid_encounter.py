#!/usr/bin/env python3
"""Test for bin_spheroid_encounter function
    in physics/dynamics.py
"""
######## Imports ########
#### Standard Library ####
import time
#### Third-Party ####
import numpy as np
#### Homemade ####
#### Local ####
from mcfacts.physics.dynamics import bin_spheroid_encounter

######## Setup ########
INPUT_DATA = {
    "smbh_mass" : 100000000.0,
    "timestep_duration_yr" : 10000.0,
    "bin_mass_1_all" : np.asarray(
    [14.54453867, 22.94940789, 15.32811881, 10.7847671, 15.44589222, 11.49062996,
 16.50199124, 12.35558125, 12.83018659, 12.28839276, 36.13006406, 10.17147256]),
    "bin_mass_2_all" : np.asarray(
    [18.12356777, 14.10766309, 11.283557, 10.20085314, 12.64002858, 26.76381266,
 12.68593058, 37.78197953, 21.59693903, 16.18398008, 29.00691522, 14.91878274]),
    "bin_orb_a_all" : np.asarray(
    [22775.78053854, 22357.01786987, 11215.85839829, 10897.83143391,
 19834.60481211, 7569.53974609, 6791.39049089, 20797.77090857,
  9988.9847044, 16623.732495, 26425.79700587, 1370.961792, ]),
    "bin_sep_all" : np.asarray(
    [1.04142481e+02, 1.48968918e+01, 3.40373716e+01, 8.93802564e+00,
 6.64032952e+01, 6.45992218e+00, 8.70659904e+00, 1.00098540e+02,
 3.81787985e+01, 3.61096179e+01, 8.00437931e+01, 1.45140013e-02]),
    "bin_ecc_all" : np.asarray(
    [0.35618302, 0.45223104, 0.44070042, 0.37046261, 0.19593823, 0.52678181,
 0.45149061, 0.03599074, 0.47831825, 0.27930884, 0.64065049, 0.37058853]),
    "bin_orb_ecc_all" : np.asarray(
    [0.011, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.011, 0.01, ]),
    "bin_orb_inc_all" : np.asarray(
    [1.57594916, 0.10027048, 0.5625362, 0.58833443, 0.19432918, 0.36873407,
 1.08070448, 0.10147987, 0.13460533, 0.06009311, 0., 0., ]),
    "time_passed" : 440000.0,
    "nsc_bh_imf_powerlaw_index" : 2.0,
    "delta_energy_strong" : 0.1,
    "nsc_spheroid_normalization" : 1.0,
    "mean_harden_energy_delta" : 0.9,
    "var_harden_energy_delta" : 0.025,
}

OUTPUT_DATA = {
    "bin_sep_all" : np.asarray(
    [1.14556729e+02, 1.48968918e+01, 3.74411088e+01, 9.83182820e+00,
 7.30436247e+01, 6.45992218e+00, 8.70659904e+00, 1.10108394e+02,
 4.19966784e+01, 3.97205797e+01, 8.80481724e+01, 1.45140013e-02]),
    "bin_ecc_all" : np.asarray(
    [0.32056472, 0.45223104, 0.39663038, 0.33341635, 0.17634441, 0.52678181,
 0.45149061, 0.03239167, 0.43048643, 0.25137796, 0.57658544, 0.37058853]),
    "bin_orb_ecc_all" : np.asarray(
    [0.0121, 0.01, 0.011, 0.011, 0.011, 0.01, 0.01, 0.011, 0.011, 0.011,
 0.0121, 0.01, ]),
    "bin_orb_inc_all" : np.asarray(
    [1.59037063, 0.10027048, 0.66803537, 0.6757186, 0.65992293, 0.36873407,
 1.08070448, 0.102877, 0.22884434, 0.10328693, 0.18363498, 0.        ]),
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
                "test bin_spheroid_encounter: Bad OUTPUT DATA\n" +\
                "All input and output values are identical"

def test_function():
    tic = time.perf_counter()
    (bin_sep_all, bin_ecc_all, bin_orb_ecc_all, bin_orb_inc_all) = \
        bin_spheroid_encounter(
            INPUT_DATA["smbh_mass"],
            INPUT_DATA["timestep_duration_yr"],
            INPUT_DATA["bin_mass_1_all"].copy(),
            INPUT_DATA["bin_mass_2_all"].copy(),
            INPUT_DATA["bin_orb_a_all"].copy(),
            INPUT_DATA["bin_sep_all"].copy(),
            INPUT_DATA["bin_ecc_all"].copy(),
            INPUT_DATA["bin_orb_ecc_all"].copy(),
            INPUT_DATA["bin_orb_inc_all"].copy(),
            INPUT_DATA["time_passed"],
            INPUT_DATA["nsc_bh_imf_powerlaw_index"],
            INPUT_DATA["delta_energy_strong"],
            INPUT_DATA["nsc_spheroid_normalization"],
            INPUT_DATA["mean_harden_energy_delta"],
            INPUT_DATA["var_harden_energy_delta"],
    )
    toc = time.perf_counter()
    assert np.allclose(bin_sep_all, OUTPUT_DATA["bin_sep_all"]), \
            "test bin_spheroid_encounter: OUTPUT_DATA mismatch"
    assert np.allclose(bin_ecc_all, OUTPUT_DATA["bin_ecc_all"]), \
            "test bin_spheroid_encounter: OUTPUT_DATA mismatch"
    assert np.allclose(bin_orb_ecc_all, OUTPUT_DATA["bin_orb_ecc_all"]), \
            "test bin_spheroid_encounter: OUTPUT_DATA mismatch"
    assert np.allclose(bin_orb_inc_all, OUTPUT_DATA["bin_orb_inc_all"]), \
            "test bin_spheroid_encounter: OUTPUT_DATA mismatch"
    print("test_bin_spheroid_encounter: pass")
    diff_mask = ~(
        np.isclose(bin_sep_all, INPUT_DATA["bin_sep_all"]) & \
        np.isclose(bin_ecc_all, INPUT_DATA["bin_ecc_all"]) & \
        np.isclose(bin_orb_ecc_all, INPUT_DATA["bin_orb_ecc_all"]) & \
        np.isclose(bin_orb_inc_all, INPUT_DATA["bin_orb_inc_all"])
    )
    print(f"{np.sum(diff_mask)} / {np.size(diff_mask)} changes")
    print(f"Time: {toc-tic} seconds!")

def test_performance(n=9001):
    tic = time.perf_counter()
    (bin_sep_all, bin_ecc_all, bin_orb_ecc_all, bin_orb_inc_all) = \
        bin_spheroid_encounter(
            INPUT_DATA["smbh_mass"],
            INPUT_DATA["timestep_duration_yr"],
            np.tile(INPUT_DATA["bin_mass_1_all"],n),
            np.tile(INPUT_DATA["bin_mass_2_all"],n),
            np.tile(INPUT_DATA["bin_orb_a_all"],n),
            np.tile(INPUT_DATA["bin_sep_all"],n),
            np.tile(INPUT_DATA["bin_ecc_all"],n),
            np.tile(INPUT_DATA["bin_orb_ecc_all"],n),
            np.tile(INPUT_DATA["bin_orb_inc_all"],n),
            INPUT_DATA["time_passed"],
            INPUT_DATA["nsc_bh_imf_powerlaw_index"],
            INPUT_DATA["delta_energy_strong"],
            INPUT_DATA["nsc_spheroid_normalization"],
            INPUT_DATA["mean_harden_energy_delta"],
            INPUT_DATA["var_harden_energy_delta"],
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
