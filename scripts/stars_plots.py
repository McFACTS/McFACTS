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
    parser.add_argument("--fname-stars",
                        default="output_mergers_stars.dat",
                        type=str, help="output_stars file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts


def main():
    opts = arg()

    stars = np.loadtxt(opts.fname_stars, skiprows=2)

    # folders = (g.glob(opts.runs_directory + "gal*"))

    # data = np.loadtxt(folders[0] + "/initial_params_star.dat",skiprows=2)


    # ========================================
    # Stars initial mass
    # ========================================

    # fig = plt.figure(figsize=plotting.set_size(figsize))
    # bins = np.linspace(np.log10(data[:,2]).min(), np.log10(data[:,2]).max(),20)

    # plt.hist(np.log10(data[:,2]), bins=bins)
    # plt.xlabel('log initial mass [$M_\odot$]')

    # if figsize == 'apj_col':
    #     plt.legend(fontsize=6)
    # elif figsize == 'apj_page':
    #     plt.legend()

    # plt.savefig(opts.plots_directory + r"/stars_initial_mass.png",format="png")


    # ========================================
    # Merger Mass vs Radius
    # ========================================

    # TQM has a trap at 500r_g, SG has a trap radius at 700r_g.
    # trap_radius = 500
    trap_radius = 700

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(stars[:, 2], stars[:, 3],
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label="1g-1g")

    plt.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')

    plt.ylabel(r'Star Mass [$M_\odot$]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.ylim(0.4, 325)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_mass_v_radius.png", format='png')
    plt.close()

    # ========================================
    # HRD
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(stars[:, 5], stars[:, 6],
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label="1g-1g")

    plt.gca().invert_xaxis()
    plt.ylabel(r"$\log L/L_\odot$")
    plt.xlabel(r"$\log T_\mathrm{eff}/\mathrm{K}$")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_hrd.png", format='png')
    plt.close()



######## Execution ########
if __name__ == "__main__":
    main()
