import numpy as np

import mcfacts.physics.binary.merge
from mcfacts.physics.binary.merge import tichy08, chieff, tgw

def test_merger():
    #This is a test of the merger 
    #1. Test a merger by calling modules
    print("Test Module merger!")
    test_bh_mass1 = 10.0
    test_bh_mass2 = 15.0
    test_bh_spin1 = 0.1
    test_bh_spin2 = 0.7
    test_bh_angle1 = 1.80
    test_bh_angle2 = 0.7
    test_bin_bh_ang_mom = 1.0
    test_bh_merg_mass = mcfacts.physics.binary.merge.merged_mass(test_bh_mass1, test_bh_mass2, test_bh_spin1, test_bh_spin2)
    test_bh_merg_spin = mcfacts.physics.binary.merge.merged_spin(test_bh_mass1, test_bh_mass2, test_bh_spin1, test_bh_spin2, test_bin_bh_ang_mom)
    test_bh_merg_chi = mcfacts.physics.binary.merge.chi_effective(test_bh_mass1, test_bh_mass2, test_bh_spin1, test_bh_spin2, test_bh_angle1, test_bh_angle2, test_bin_bh_ang_mom)
    print(test_bh_merg_mass,test_bh_merg_spin,test_bh_merg_chi)
    #Output should always be constant: 23.560384 0.8402299374639024 0.31214563487176167
    return