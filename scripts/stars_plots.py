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

#apj_col or apj_page
figsize = "apj_col"

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-directory",
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--fname-stars",
                        default="output_mergers_stars_population.dat",
                        type=str, help="output_stars file")
    parser.add_argument("--fname-stars-merge",
                        default="output_mergers_stars_merged.dat",
                        type=str, help="output merged stars file")
    parser.add_argument("--fname-stars-explode",
                        default="output_mergers_stars_exploded.dat",
                        type=str, help="output exploded stars file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts


def make_gen_masks(table, col):
    """Create masks for retrieving different sets of a stellar population

    Parameters
    ----------
    table : numpy.ndarray
        Data
    col : int
        Column index where generation information is stored
    """

    # Column with generation data
    gen_obj = table[:, col]

    # Masks for hierarchical generations
    # g1 : 1g objects
    # g2 : 2g objects
    # g3 : >=3g
    # Pipe operator (|) = logical OR. (&)= logical AND.

    g1_mask = (gen_obj == 1)
    g2_mask = (gen_obj == 2)
    gX_mask = (gen_obj >= 3)

    return (g1_mask, g2_mask, gX_mask)


def main():
    opts = arg()

    stars = np.loadtxt(opts.fname_stars, skiprows=2)
    stars_merge = np.loadtxt(opts.fname_stars_merge, skiprows=1)
    stars_explode = np.loadtxt(opts.fname_stars_explode, skiprows=1)

    merger_g1_mask, merger_g2_mask, merger_gX_mask = make_gen_masks(stars, 6)
    explode_g1_mask, explode_g2_mask, explode_gX_mask = make_gen_masks(stars_explode, 6)

    # Ensure no union between sets
    assert all(merger_g1_mask & merger_g2_mask) == 0
    assert all(merger_g1_mask & merger_gX_mask) == 0
    assert all(merger_g2_mask & merger_gX_mask) == 0
    assert all(explode_g1_mask & explode_g2_mask) == 0
    assert all(explode_g1_mask & explode_gX_mask) == 0
    assert all(explode_g2_mask & explode_gX_mask) == 0

    # Ensure no elements are missed
    assert all(merger_g1_mask | merger_g2_mask | merger_gX_mask) == 1
    assert all(explode_g1_mask | explode_g2_mask | explode_gX_mask) == 1


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
    # Number of Mergers vs Mass
    # ========================================

    # Plot intial and final mass distributions
    fig = plt.figure(figsize=plotting.set_size(figsize))
    counts, bins = np.histogram(stars[:, 3])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(stars[:, 3].min()), int(stars[:, 3].max()) + 2, 1)

    hist_data = [stars[:, 3][merger_g1_mask], stars[:, 3][merger_g2_mask], stars[:, 3][merger_gX_mask]]
    hist_label = ['1g', '2g', r'$\geq$3g']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Star Mass [$M_\odot$]')
    #plt.xscale('log')
    # plt.ylim(-5,max(counts))
    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    svf_ax.tick_params(axis='x', direction='out', which='both')
    #plt.grid(True, color='gray', ls='dashed')
    svf_ax.yaxis.grid(True, color='gray', ls='dashed')
    #plt.xticks(np.geomspace(int(stars[:, 3].min()), int(stars[:, 3].max()), 5).astype(int))
    #plt.xticks(np.geomspace(20, 200, 5).astype(int))

    #svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    #svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.savefig(opts.plots_directory + r"/star_merger_mass.png", format='png')

    plt.close()



    # ========================================
    # Merger Mass vs Radius
    # ========================================

    # TQM has a trap at 500r_g, SG has a trap radius at 700r_g.
    # trap_radius = 500
    trap_radius = 700

    # Separate generational subpopulations
    gen1_orb_a = stars[:, 2][merger_g1_mask]
    gen2_orb_a = stars[:, 2][merger_g2_mask]
    genX_orb_a = stars[:, 2][merger_gX_mask]
    gen1_mass = stars[:, 3][merger_g1_mask]
    gen2_mass = stars[:, 3][merger_g2_mask]
    genX_mass = stars[:, 3][merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(gen1_orb_a, gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g'
                )

    plt.scatter(gen2_orb_a, gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g'
                )

    plt.scatter(genX_orb_a, genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g'
                )

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

    plt.ylim(0.4, 400)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_mass_v_radius.png", format='png')
    plt.close()

    # ========================================
    # Merger Mass vs Radius for exploded stars
    # ========================================

    # TQM has a trap at 500r_g, SG has a trap radius at 700r_g.
    # trap_radius = 500
    trap_radius = 700

    # Separate generational subpopulations
    ex_gen1_orb_a = stars_explode[:, 2][explode_g1_mask]
    ex_gen2_orb_a = stars_explode[:, 2][explode_g2_mask]
    ex_genX_orb_a = stars_explode[:, 2][explode_gX_mask]
    ex_gen1_mass = stars_explode[:, 3][explode_g1_mask]
    ex_gen2_mass = stars_explode[:, 3][explode_g2_mask]
    ex_genX_mass = stars_explode[:, 3][explode_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(ex_gen1_orb_a, ex_gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g'
                )

    plt.scatter(ex_gen2_orb_a, ex_gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g'
                )

    plt.scatter(ex_genX_orb_a, ex_genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g'
                )

    plt.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')

    plt.ylabel(r'Star Mass [$M_\odot$]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6, title="Exploded stars")
    elif figsize == 'apj_page':
        plt.legend(title="Exploded stars")

    plt.ylim(0.4, 400)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_exploded_mass_v_radius.png", format='png')
    plt.close()

    # ========================================
    # HRD
    # ========================================

    # Separate generational subpopulations
    gen1_teff = stars[:, 8][merger_g1_mask]
    gen2_teff = stars[:, 8][merger_g2_mask]
    genX_teff = stars[:, 8][merger_gX_mask]
    gen1_lum = stars[:, 9][merger_g1_mask]
    gen2_lum = stars[:, 9][merger_g2_mask]
    genX_lum = stars[:, 9][merger_gX_mask]


    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(gen1_teff, gen1_lum,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g'
                )

    plt.scatter(gen2_teff, gen2_lum,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g'
                )

    plt.scatter(genX_teff, genX_lum,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g'
                )

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


    # ========================================
    # M1 vs M2 for merged stars
    # ========================================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    cm = plt.scatter(stars_merge[:,8], stars_merge[:,9], c=stars_merge[:,2], cmap="BuPu", norm="log",
                     s=styles.markersize_gen1,
                     marker=styles.marker_gen1)

    cbar = fig.colorbar(cm)
    cbar.set_label(r"Radius [$R_{\rm g}$]")

    plt.xlabel(r"$M_1$ [$M_\odot$]")
    plt.ylabel(r"$M_2$ [$M_\odot$]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6, title='Merged stars')
    elif figsize == 'apj_page':
        plt.legend(title="Merged stars")

    plt.savefig(opts.plots_directory + "/stars_m1m2.png", format="png")
    plt.close()


    # ========================================
    # Merged and exploded stars vs time
    # ========================================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(stars_merge[:, 1]/1e6, stars_merge[:, 3], s=styles.markersize_gen1, marker="o", edgecolor=styles.color_gen1, facecolor="None", alpha=styles.markeralpha_gen1, label="Merged star")
    plt.scatter(stars_explode[:, 1]/1e6, stars_explode[:, 3], s=styles.markersize_gen1, marker="o", edgecolor=styles.color_gen2, facecolor="None", alpha=styles.markeralpha_gen1, label="Exploded star")

    plt.xlabel(r"Time [Myr]")
    plt.ylabel(r"Mass [$M_\odot$]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.savefig(opts.plots_directory + "/stars_merge_explode.png", format="png")
    plt.close()




######## Execution ########
if __name__ == "__main__":
    main()
