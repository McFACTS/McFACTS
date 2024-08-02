#!/usr/bin/env python3

######## Globals ########
COLUMN_NAMES = "iter CM M chi_eff a_tot spin_angle m1 m2 a1 a2 theta1 theta2 gen1 gen2 t_merge chi_p"

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import os
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data


######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fname-survivors",
        default="sg1Myr_survivors.dat",
        type=str, help="output_survivors file")
    
    opts = parser.parse_args()
    assert os.path.isfile(opts.fname_survivors)
    
    return opts

######## Main ########
def main():
    #plt.style.use('seaborn-v0_8-poster')

    # need section for loading data
    #opts = arg()
    survivors = np.loadtxt("recipes/sg1Myrx2_survivors.dat")
    #emris = np.loadtxt(opts.fname_emris, skiprows=2)
    #lvk = np.loadtxt(opts.fname_lvk,skiprows=2)

    mask = np.isfinite(survivors[:,0])
    survivors = survivors[mask]

    
    # plt.figure(figsize=(10,6))
    # plt.figure()

    # Plot location distributions
    counts, bins = np.histogram(survivors[:,0])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(survivors[:,0].min()), int(survivors[:,0].max())+2, 500)
    plt.hist(survivors[:,0], bins=bins, align='left', color='rebeccapurple', alpha=0.9, rwidth=0.8)

    # plt.hist(bh_masses_by_sorted_location, bins=numbins, align='left', label='Final', color='purple', alpha=0.5)
    # plt.hist(binary_bh_array[2:4,:bin_index], bins=numbins, align='left', label='Final', color=['purple'], alpha=0.5)
    #plt.title(f'Black Hole Merger Remnant Masses\n'+
    #            f'Number of Mergers: {mergers.shape[0]}')
    plt.ylabel('Number of BH')
    plt.xlabel(r'Initial Radius ($r_{g}$)')

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig("./survivor_location.png", format='png')
    plt.tight_layout()
    plt.close()

######## Execution ########
if __name__ == "__main__":
    main()
