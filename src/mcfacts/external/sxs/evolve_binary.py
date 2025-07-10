"""
Module to process binary black hole mergers using the surfinBH surrogate model.
"""

#import juliacall
import numpy as np
#from scripts.sxs import fit_modeler
from mcfacts.external.sxs import surrogate

import pandas as pd
import time, os
from astropy import constants as const

#surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")

def surrogate(m1, m2, s1m, s2m, sa1, sa2, p12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate):

    #print(m1, m2, s1m, s2m, sa1, sa2, p12)
    mass_final, spin_final, kick_final = [], [], []
    #mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12 = [], [], [], [], [], [], []
    
    for i in range(len(m1)):
        #print(mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate)
        
        # Variables are all sent to surrogate model 
        # McFACTS outputs arrays with all values and the surrogate requrires float values 
        # Calling the values by iterating through the arrays and running the surrogate and then assembling them back into an array
        start = time.time()
        M_f, spin_f, v_f = surrogate.evolve_binary(
            m1[i],
            m2[i],
            s1m[i],
            s2m[i],
            sa1[i],
            sa2[i],
            p12[i],
            bin_sep,
            bin_inc,
            bin_phase,
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
        v_f_mag = np.linalg.norm(v_f) * const.c.value / 1000
        
        #print(M_f, spin_f_mag, v_f_mag)
        
        mass_final.append(M_f)
        spin_final.append(spin_f_mag)
        kick_final.append(v_f_mag)
        
    
    #print(M_f, spin_f_mag, v_f_mag)
    
    print("M_f = ", mass_final)
    print("spin_f = ", spin_final)
    print("v_f = ", kick_final)
    
    return np.array(mass_final), np.array(spin_final), np.array(kick_final)