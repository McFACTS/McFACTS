"""
Module to process binary black hole mergers using the surfinBH surrogate model.
"""

import juliacall
import numpy as np
import random
#from scripts.sxs import fit_modeler
from mcfacts.external.sxs import evolve_surrogate

import pandas as pd
import time, os
from astropy import constants as const

#surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")

def surrogate(mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_1, phi_2, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate):

    #print(m1, m2, s1m, s2m, sa1, sa2, p12)
    mass_final, spin_final, spin_angle_final, kick_final, mass_1_20Hz_final, mass_2_20Hz_final, spin_1_20Hz_final, spin_2_20Hz_final = [], [], [], [], [], [], [], []

    phi_1_rand = np.random.uniform(0, 2 * np.pi, phi_1)
    phi_2_rand = np.random.uniform(0, 2 * np.pi, phi_2)
    bin_phase_rand = np.random.uniform(0, 2 * np.pi, bin_phase)

    for i in range(len(mass_1)):        
        # Variables are all sent to surrogate model 
        # McFACTS outputs arrays with all values and the surrogate requrires float values 
        # Calling the values by iterating through the arrays and running the surrogate and then assembling them back into an array        
        start = time.time()
        M_f, spin_f, v_f, mass_1_20Hz, mass_2_20Hz, spin_1_20Hz, spin_2_20Hz = evolve_surrogate.evolve_binary(
            mass_1[i],
            mass_2[i],
            spin_1_mag[i],
            spin_2_mag[i],
            spin_angle_1[i],
            spin_angle_2[i],
            phi_1_rand[i],
            phi_2_rand[i],
            bin_sep,
            bin_inc,
            bin_phase_rand[i],
            bin_orb_a,
            mass_SMBH,
            spin_SMBH,
            surrogate,
            verbose=True,
        )
        end = time.time()
        
        run_time = end - start
        print("Merger took ", run_time, " seconds")
        
        spin_f_mag = np.linalg.norm(spin_f)
        # Even though spin 1/2 at 20Hz are labeled as "final" they are still values taken prior to merger and are not post merger values
        spin_1_20Hz_final_mag = np.linalg.norm(spin_1_20Hz)
        spin_2_20Hz_final_mag = np.linalg.norm(spin_2_20Hz)
        
        v_f_mag = np.linalg.norm(v_f) * const.c.value / 1000
        
        #print(M_f, spin_f_mag, v_f_mag)
        
        mass_final.append(M_f)
        spin_final.append(spin_f_mag)
        spin_angle_final.append(np.arccos(spin_f[2]/spin_f_mag)) # contains remnant spin angles 
        kick_final.append(v_f_mag)
        mass_1_20Hz_final.append(mass_1_20Hz)
        mass_2_20Hz_final.append(mass_2_20Hz)
        spin_1_20Hz_final.append(spin_1_20Hz_final_mag)
        spin_2_20Hz_final.append(spin_2_20Hz_final_mag)
        
    #print(M_f, spin_f_mag, v_f_mag)
    
    print("M_f = ", mass_final)
    print("spin_f = ", spin_f_mag)
    print("spin_angle_f = ", spin_angle_final)
    print("v_f = ", kick_final)
    print("M_1 @ 20Hz = ", mass_1_20Hz)
    print("M_2 @ 20Hz = ", mass_2_20Hz)
    print("spin_1 @ 20Hz = ", spin_1_20Hz_final_mag)
    print("spin_2 @ 20Hz = ", spin_2_20Hz_final_mag)
    
    return np.array(mass_final), np.array(spin_final), np.array(spin_angle_final), np.array(kick_final), np.array(mass_1_20Hz_final), np.array(mass_2_20Hz_final), np.array(spin_1_20Hz_final), np.array(spin_2_20Hz_final)