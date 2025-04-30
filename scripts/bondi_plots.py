#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import glob as g
import os
from scipy.optimize import curve_fit
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data
from mcfacts.vis import plotting
from mcfacts.vis import styles
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

figsize = "apj_col"


######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-directory",
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--fname-mergers",
                        default="output_mergers_population.dat",
                        type=str, help="output_mergers file")
    parser.add_argument("--fname-bondi",
                        default="output_mergers_bondi_variables.dat",
                        type=str, help="output_mergers file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isfile(opts.fname_mergers)
    return opts


def make_gen_masks(table, col1, col2):
    """Create masks for retrieving different sets of a merged or binary population based on generation.
    """
    # Column of generation data
    gen_obj1 = table[:, col1]
    gen_obj2 = table[:, col2]

    # Masks for hierarchical generations
    # g1 : all 1g-1g objects
    # g2 : 2g-1g and 2g-2g objects
    # g3 : >=3g-Ng (first object at least 3rd gen; second object any gen)
    # Pipe operator (|) = logical OR. (&)= logical AND.
    g1_mask = (gen_obj1 == 1) & (gen_obj2 == 1)
    g2_mask = ((gen_obj1 == 2) | (gen_obj2 == 2)) & ((gen_obj1 <= 2) & (gen_obj2 <= 2))
    gX_mask = (gen_obj1 >= 3) | (gen_obj2 >= 3)

    return g1_mask, g2_mask, gX_mask


def main():
    opts = arg()

    mergers = np.loadtxt(opts.fname_mergers, skiprows=2)
    bondi = np.loadtxt(opts.fname_bondi, skiprows=2)

    # ===============================
    ### testing
    # ===============================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(bondi[:, 0], bondi[:, 1], s=1)

    plt.ylabel(r'Migration Velocity')
    plt.xlabel(r'Sound Speed')
    #plt.xscale('log')
    # plt.yscale('log')
    #plt.grid(True, color='gray', ls='dashed')

    plt.savefig(opts.plots_directory + '/mig_vs_sound_speed.png', format='png')
    plt.close()

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(bondi[:, 0], bondi[:, 2], s=1)

    plt.ylabel(r'Shear Velocity')
    plt.xlabel(r'Sound Speed')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.grid(True, color='gray', ls='dashed')

    plt.savefig(opts.plots_directory + '/shear_vs_sound_speed.png', format='png')
    plt.close()


######## Execution ########
if __name__ == "__main__":
    main()
