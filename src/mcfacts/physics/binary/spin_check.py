import numpy as np
from scipy.stats import truncnorm

def generate_truncated_normal(mean=0, std=1, lower=0.75, upper=0.85, size=10):
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def spin_check(gen_1, gen_2, spin_merged):
    """ Since the T&M '08 perscription generates spin values outside of the expected range for higher mass ratio objects this file checks spin values after merger and if the magnitude is too low, this function resets it to a random distribution between a set range in order to generate results similiar to that of the NRsurrogate model.

    Parameters
    ----------        
        gen_1 : numpy.ndarray
            generation of m1 (before merger) (1=natal BH that has never been in a prior merger)
        gen_2 : numpy.darray
            generation of m2 (before merger) (1=natal BH that has never been in a prior merger)
        spin_merged : numpy.darray
            Final spin magnitude [unitless] of merger remnant with :obj:`float` type

    Returns
    -------
    merged_spins : numpy array
        Final spin magnitude [unitless] of merger remnant with :obj:`float` type
    """
    
    #print("initial gen 1: ", gen_1)
    #print("initial gen 2: ", gen_2)
    #print("initial spin_merged: ", spin_merged)
    
    new_spin_merged = []
    
    for i in range(len(spin_merged)):
        # Sorting first gen objects and keeping their parameters
        if (gen_1[i] == 1.) & (gen_2[i] == 1.):
            #print('gen 1', spin_merged[i])
            new_spin_merged.append(spin_merged[i])
        # Sorting 2nd gen objects and updating their spins as needed otherwise keeping them the same
        # If spins < 0.75, they are reset to a randomly selected gaussian distribution between 0.75 - 0.85
        elif ((gen_1[i] == 2.) | (gen_2[i] == 2.)) & ((gen_1[i] <= 2.) & (gen_2[i] <= 2.)):
            #print('gen 2', spin_merged[i])
            if spin_merged[i] < 0.75:
                spin_plus_noise = generate_truncated_normal(mean=0, std=1, lower=0.75, upper=0.85, size=1)
                #print('gen 2 plus noise', spin_plus_noise)
                new_spin_merged.append(float(spin_plus_noise))
            else:
                #print('gen 2', spin_merged[i])
                new_spin_merged.append(spin_merged[i])
        # Sorting 3+ gen objects and updating their spins as needed otherwise keeping them the same
        # If spins < 0.85, they are reset to a randomly selected gaussian distribution between 0.85 - 0.95
        elif (gen_1[i] >= 3.) | (gen_2[i] >= 3.):
            #print('gen x', spin_merged[i])
            if spin_merged[i] < 0.85:
                spin_plus_noise = generate_truncated_normal(mean=0, std=1, lower=0.85, upper=0.95, size=1)
                #print('gen 3+ plus noise', spin_plus_noise)
                new_spin_merged.append(float(spin_plus_noise))
            else:
                #print('gen 3+', spin_merged[i])
                new_spin_merged.append(spin_merged[i])
        
    #print("new gen 1: ", gen_1)
    #print("new gen 2: ", gen_2)
    #print("new spin_merged: ", spin_merged)
    return np.array(new_spin_merged)
    