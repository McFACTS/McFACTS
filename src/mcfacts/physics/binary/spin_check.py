import numpy as np

def spin_check(gen_1, gen_2, spin_merged):
    
    print("initial gen 1: ", gen_1)
    print("initial gen 2: ", gen_2)
    print("initial spin_merged: ", spin_merged)
    
    new_spin_merged = []
    
    for i in range(len(spin_merged)):
        if (gen_1[i] == 1.) & (gen_2[i] == 1.):
            print('gen 1', spin_merged[i])
            new_spin_merged.append(spin_merged[i])
        elif ((gen_1[i] == 2.) | (gen_2[i] == 2.)) & ((gen_1[i] <= 2.) & (gen_2[i] <= 2.)):
            print('gen 2', spin_merged[i])
            if spin_merged[i] < 0.7:
                new_spin_merged.append(0.7)
            else:
                new_spin_merged.append(spin_merged[i])
        elif (gen_1[i] >= 3.) | (gen_2[i] >= 3.):
            print('gen x', spin_merged[i])
            if spin_merged[i] < 0.85:
                new_spin_merged.append(0.85)
            else:
                new_spin_merged.append(spin_merged[i])
        
    print("new gen 1: ", gen_1)
    print("new gen 2: ", gen_2)
    print("new spin_merged: ", spin_merged)
    print(new_spin_merged)
    return np.array(new_spin_merged)
    