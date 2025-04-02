import numpy as np
from mcfacts.mcfacts_random_state import rng
from astropy import units as u
from astropy import constants as const


# def analytical_kick_velocity(length):
#     # took this from chat:
#     # Define shape and scale parameters for the gamma distribution
#     shape = 5  # Adjust to control peak sharpness
#     scale = 40  # Controls spread; tweak to shift peak to ~200

#     # Generate random numbers from gamma distribution
#     r = gamma.rvs(a=shape, scale=scale, size=length)
#     return r

# # bh_binary_id_num_merger

def analytical_kick_velocity(mass_1, mass_2, chi_1, chi_2, spin_angle_1, spin_angle_2):
    # check:
    v_kick = []

    for x in range(len(mass_1)):
        condition = mass_1[x] >= mass_2[x]

        m_1_new = mass_1[x] if condition else mass_2[x]
        m_2_new = mass_2[x] if condition else mass_1[x]
        chi_1_new = chi_1[x] if condition else chi_2[x]
        chi_2_new = chi_2[x] if condition else chi_1[x]
        spin_angle_1_new = spin_angle_1[x] if condition else spin_angle_2[x]
        spin_angle_2_new = spin_angle_2[x] if condition else spin_angle_1[x] 

        chi_1_new_par = chi_1_new * np.cos(spin_angle_1_new) # do i have to do anything to my cos?
        chi_1_new_perp = chi_1_new * np.sin(spin_angle_1_new)

        chi_2_new_par = chi_2_new * np.cos(spin_angle_2_new)
        chi_2_new_perp = chi_2_new * np.sin(spin_angle_2_new)

        q = m_1_new/m_2_new
        xi = 145 # deg
        A = 1.2*10**4
        eta = q/(1+q)**2
        B = -0.93
        H = 6.9*10**3
        q = m_1_new/m_2_new
        n = q/(1+q)**2
        S = (2 * (chi_2_new_par + (q**2 * chi_1_new_par)))/(1+q)**2
        angle = rng.uniform(0.0, 2*np.pi)
        V_11 = 3678
        V_A = 2481
        V_B = 1793
        V_C = 1507
        v_m = A * eta**2 * np.sqrt(1-4*eta)*(1 + B * n)
        v_perp = ( (H*eta**2)/(1+q) ) * (chi_2_new_par - q*chi_1_new_par)
        #print(v_perp)
        term_1 = ( (16*eta**2) / (1 + q)) * (V_11 + (V_A * S) + (V_B * S**2) + (V_C * S**3))
        term_2 = abs(chi_2_new_perp - q*chi_1_new_perp)*np.cos(angle)
        v_par = term_1 * term_2
        
        v_kick.append(np.sqrt((v_m + (v_perp * np.cos(xi)))**2 + (v_perp * (np.sin(xi)))**2 + v_par**2))
        # np.sqrt((v_m + v_perp * np.cos(xi))**2 + (v_perp * (np.sin(xi)))**2 + v_par**2)
    return v_kick

    