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
    parser.add_argument("--fname-mergers",
                        default="output_mergers_population.dat",
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

    #shock plot
    fig = plt.figure(figsize=plotting.set_size(figsize))

    shock_bins = np.logspace(np.log10(mergers[:, 16].min()), np.log10(mergers[:, 16].max()), 50)

    plt.hist(mergers[:, 16], bins = shock_bins)

    plt.ylabel(r'n')
    plt.xlabel(r'Shock Luminsoity (erg/s)')
    plt.xscale('log')
    #plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    #plt.ylim(0.4, 325)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/luminosity_shock.png", format='png')
    plt.close()

    ### jet plot
    fig = plt.figure(figsize=plotting.set_size(figsize))

    jet_bins = np.logspace(np.log10(mergers[:, 17].min()), np.log10(mergers[:, 17].max()), 50)

    plt.hist(mergers[:, 17], bins = jet_bins)

    plt.ylabel(r'n')
    plt.xlabel(r'Jet Luminsoity (erg/s)')
    plt.xscale('log')
    #plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    #plt.ylim(0.4, 325)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/luminosity_jet.png", format='png')
    plt.close()


    ### comparison plot
    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.hist(mergers[:, 16], bins = shock_bins, label = 'shock')
    plt.hist(mergers[:, 17], bins = jet_bins, label = 'jet')

    plt.scatter(mergers[:, 16], mergers[:, 17])

    plt.ylabel(r'jet [erg/s]')
    plt.xlabel(r'shock lum [erg/s]')
    plt.xscale('log')
    #plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    #plt.ylim(0.4, 325)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/luminosity_comp.png", format='png')
    plt.close()

    # shocks vs time
    fig = plt.figure(figsize=plotting.set_size(figsize))

    merger_g1_mask, merger_g2_mask, merger_gX_mask = make_gen_masks(mergers, 12, 13)
        # Ensure no union between sets
    assert all(merger_g1_mask & merger_g2_mask) == 0
    assert all(merger_g1_mask & merger_gX_mask) == 0
    assert all(merger_g2_mask & merger_gX_mask) == 0
    assert all(merger_g1_mask | merger_g2_mask | merger_gX_mask) == 1

    gen1_shock = mergers[:, 16][merger_g1_mask]
    gen2_shock = mergers[:, 16][merger_g2_mask]
    genX_shock = mergers[:, 16][merger_gX_mask]

    all_time = mergers[:, 14]
    gen1_time = all_time[merger_g1_mask]
    gen2_time = all_time[merger_g2_mask]
    genX_time = all_time[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(gen1_time / 1e6, gen1_shock,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    ax3.scatter(gen2_time / 1e6, gen2_shock,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax3.scatter(genX_time / 1e6, genX_shock,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    ax3.set(
        xlabel='Time [Myr]',
        ylabel='Shock Lum [erg/s]',
        yscale="log",
        axisbelow=True
    )

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/time_vs_shock_lum.png', format='png')
    plt.close()

    # jet lum vs time

    fig = plt.figure(figsize=plotting.set_size(figsize))

    gen1_jet = mergers[:, 17][merger_g1_mask]
    gen2_jet = mergers[:, 17][merger_g2_mask]
    genX_jet= mergers[:, 17][merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(gen1_time / 1e6, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    ax3.scatter(gen2_time / 1e6, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax3.scatter(genX_time / 1e6, genX_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    ax3.set(
        xlabel='Time [Myr]',
        ylabel='Jet Lum [erg/s]',
        yscale="log",
        axisbelow=True
    )

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/time_vs_jet_lum.png', format='png')
    plt.close()


    #plt.plot(mergers[:, ], mergers[:, 16])
    #plt.plot(mergers[:, ], mergers[:, 17])

    # plt.ylabel(r'n')
    # plt.xlabel(r'Shock Luminsoity (erg/s)')
    # plt.xscale('log')
    # #plt.yscale('log')

    # if figsize == 'apj_col':
    #     plt.legend(fontsize=6)
    # elif figsize == 'apj_page':
    #     plt.legend()

    # #plt.ylim(0.4, 325)

    # svf_ax = plt.gca()
    # svf_ax.set_axisbelow(True)
    # plt.grid(True, color='gray', ls='dashed')
    # plt.savefig(opts.plots_directory + "/luminosity_shock.png", format='png')
    # plt.close()



######## Execution ########
if __name__ == "__main__":
    main()
