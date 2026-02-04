#!/usr/bin/env python3
"""Make visualizations for bolometric shock and jet luminosities.

Example usage: 
make plots
make em_plots
"""
######## Imports ########
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import glob as g
import os
# Grab those txt files
from scipy.optimize import curve_fit
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
                        default=".",
                        type=str, help="directory to go to runs")
    parser.add_argument("--fname-mergers",
                        default="output_mergers_population.dat",
                        type=str, help="output_mergers file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.fname_mergers)
    assert os.path.isfile(opts.fname_mergers)
    return opts

def linefunc(x, m):
    """Model for a line passing through (x,y) = (0,1).

    Function for a line used when fitting to the data.
    """
    return m * (x - 1)

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

    # Bolometric luminosity of the AGN [erg s**-1]
    lum_agn = (0.1 * 1.26) * 10**38 * (1e8 / 1)

    # Exclude all rows with NaNs or zeros in the final mass column
    merger_nan_mask = (np.isfinite(mergers[:, 2])) & (mergers[:, 2] != 0)
    mergers = mergers[merger_nan_mask]

    merger_g1_mask, merger_g2_mask, merger_gX_mask = make_gen_masks(mergers, 12, 13)

    # Ensure no union between sets
    assert all(merger_g1_mask & merger_g2_mask) == 0
    assert all(merger_g1_mask & merger_gX_mask) == 0
    assert all(merger_g2_mask & merger_gX_mask) == 0

    # Ensure no elements are missed
    assert all(merger_g1_mask | merger_g2_mask | merger_gX_mask) == 1

    # ===============================
    ### comparison of shock and jet luminosities histogram ###
    # ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    shock_bins = np.logspace(np.log10(mergers[:, 17].min()), np.log10(mergers[:, 17].max()), 50)
    jet_bins = np.logspace(np.log10(mergers[:, 18].min()), np.log10(mergers[:, 18].max()), 50)

    plt.hist(mergers[:, 17], bins=shock_bins, label='Shock')
    plt.hist(mergers[:, 18], bins=jet_bins, label='Jet', alpha=0.8)
    plt.axvline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    plt.ylabel(r'n')
    plt.xlabel(r'log Luminosity [erg s$^{-1}$]')
    plt.xscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/luminosity_comparison_dist.png", format='png')
    plt.close()

    # ===============================
    ### shocks vs time ###
    # ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    all_shocks = mergers[:, 17]
    gen1_shock = all_shocks[merger_g1_mask]
    gen2_shock = all_shocks[merger_g2_mask]
    genX_shock = all_shocks[merger_gX_mask]

    all_time = mergers[:, 14]
    gen1_time = all_time[merger_g1_mask]
    gen2_time = all_time[merger_g2_mask]
    genX_time = all_time[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plot the 1g mergers
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
        ylabel=r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]',
        yscale="log",
        axisbelow=True
    )
    
    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/shock_lum_vs_time.png', format='png')
    plt.close()

    # ===============================
    ### jet lum vs time ###
    # ===============================
    all_jets = mergers[:,18]
    gen1_jet = all_jets[merger_g1_mask]
    gen2_jet = all_jets[merger_g2_mask]
    genX_jet= all_jets[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    ax3.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    # plot the 1g mergers
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
        ylabel=r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]',
        yscale="log",
        axisbelow=True
    )

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/jet_lum_vs_time.png', format='png')
    plt.close()

    # =================================================
    ### shock luminosity distribution histogram ###
    # =================================================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    shock_log = np.log10(mergers[:, 17])
    bins = np.arange(int(shock_log.min()), int(shock_log.max())+1, 0.2)

    hist_data = [shock_log[merger_g1_mask], shock_log[merger_g2_mask], shock_log[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)
    plt.ylabel(r'n')
    plt.xlabel(r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/shock_lum_histogram.png", format='png')
    plt.close()

    # ===========================================
    ### jet luminosity distribution histogram ###
    # ===========================================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    bins = np.logspace(np.log10(all_jets.min()), np.log10(all_jets.max()), 50)
    hist_data = [all_jets[merger_g1_mask], all_jets[merger_g2_mask], all_jets[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)
    plt.axvline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")
    plt.ylabel(r'n')
    plt.xlabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]')

    if figsize == 'apj_col':
        plt.legend(fontsize=6, loc = 'best')
    elif figsize == 'apj_page':
        plt.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.xscale('log')
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/jet_lum_histogram.png", format='png')
    plt.close()

    # ===============================
    ### jet luminosity vs. a_bin ###
    # ===============================
    all_orb_a = mergers[:, 1]
    gen1_orb_a = all_orb_a[merger_g1_mask]
    gen2_orb_a = all_orb_a[merger_g2_mask]
    genX_orb_a = all_orb_a[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))

    ax3.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    ax3 = fig.add_subplot(111)
    # plot 1g-1g mergers
    ax3.scatter(gen1_orb_a, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )
    
    # plot the 2g+ mergers
    ax3.scatter(gen2_orb_a, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    
    # plot the 3g+ mergers
    ax3.scatter(genX_orb_a, genX_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    trap_radius = 700
    #if "sg" in opts.plots_directory:
     #   trap_radius = 700
    #elif "tqm" in opts.plots_directory:
     #   trap_radius = 500

    ax3.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.ylabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]')
    plt.xlabel(r'log Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/jet_lum_vs_radius.png', format='png')
    plt.close()

    # ===============================
    ### shock luminosity vs. a_bin ###
    # ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    ax3 = fig.add_subplot(111)

    # plot 1g-1g mergers
    ax3.scatter(gen1_orb_a, gen1_shock,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )
    
    # plot the 2g+ mergers
    ax3.scatter(gen2_orb_a, gen2_shock,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax3.scatter(genX_orb_a, genX_shock,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    ax3.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    plt.ylabel(r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]')
    plt.xlabel(r'log Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/shock_lum_vs_radius.png', format='png')
    plt.close()

    # ====================================================
    ### shock luminosity vs. a_bin with histogram ###
    # ====================================================
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 

    plot = axs[0]
    hist = axs[1]

    # plot 1g mergers
    plot.scatter(gen1_orb_a, gen1_shock,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g+ mergers
    plot.scatter(gen2_orb_a, gen2_shock,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot 3g+ mergers
    plot.scatter(genX_orb_a, genX_shock,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plot.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    plot.set_ylabel(r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]')
    plot.set_xlabel(r'log Radius [$R_g$]')
    plot.set_xscale('log')
    plot.set_yscale('log')
    plot.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        plot.legend(fontsize=6, loc = 'best')
    elif figsize == 'apj_page':
        plot.legend()

    lum_shuck_bins = np.logspace(np.log10(all_shocks.min()), np.log10(all_shocks.max()), 50)
    hist_lum_shock_data = [all_shocks[merger_g1_mask], all_shocks[merger_g2_mask], all_shocks[merger_gX_mask]]

    # configure histogram
    hist.hist(hist_lum_shock_data, bins=lum_shuck_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation = 'horizontal')
    hist.grid(True, color='gray', ls='dashed')
    hist.yaxis.tick_right()
    hist.set_xlabel(r'n')

    if figsize == 'apj_col':
        hist.legend(fontsize=6, loc='best')
    elif figsize == 'apj_page':
        hist.legend()

    plt.tight_layout()
    plt.savefig(opts.plots_directory + '/shock_lum_vs_radius_w_histogram.png', format='png')
    plt.close()

    # ==================================================
    ### jet luminosity vs. a_bin with histogram ###
    # ==================================================
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 

    plot = axs[0]
    hist = axs[1]

    # plot 1g-1g mergers
    plot.scatter(gen1_orb_a, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g-mg mergers
    plot.scatter(gen2_orb_a, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot 3g-ng mergers
    plot.scatter(genX_orb_a, genX_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plot.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    plot.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")
    plot.set_ylabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]')
    plot.set_xlabel(r'log Radius [$R_g$]')
    plot.set_xscale('log')
    plot.set_yscale('log')

    plot.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        plot.legend(fontsize=6, loc = 'best')
    elif figsize == 'apj_page':
        plot.legend()

    lum_bins = np.logspace(np.log10(all_jets.min()), np.log10(all_jets.max()), 50)
    hist_lum_data = [all_jets[merger_g1_mask], all_jets[merger_g2_mask], all_jets[merger_gX_mask]]

    # configure histogram
    hist.hist(hist_lum_data, bins=lum_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation = 'horizontal')
    hist.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")
    hist.grid(True, color='gray', ls='dashed')
    #hist.set_yscale('log')
    hist.yaxis.tick_right()
    hist.set_xlabel(r'n')

    #if figsize == 'apj_col':
    #    hist.legend(fontsize=6, loc='best')
    #elif figsize == 'apj_page':
    #    hist.legend()

    plt.tight_layout()
    plt.savefig(opts.plots_directory + '/jet_lum_vs_radius_w_histogram.png', format='png')
    plt.close()

    # =======================================
    ### shock luminosity vs. remnant mass ###
    # =======================================
    all_mass = mergers[:, 2]
    gen1_mass = all_mass[merger_g1_mask]
    gen2_mass = all_mass[merger_g2_mask]
    genX_mass = all_mass[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))

    ax3 = fig.add_subplot(111)

    # plot 1g mergers
    ax3.scatter(gen1_mass, gen1_shock,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g+ mergers
    ax3.scatter(gen2_mass, gen2_shock,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot 3g+ mergers
    ax3.scatter(genX_mass, genX_shock,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plt.ylabel(r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]')
    plt.xlabel(r'M$_{\mathrm{Remnant}}$ [$M_\odot$]')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/shock_lum_vs_remnant_mass.png', format='png')
    plt.close()

    # =======================================
    ### jet luminosity vs. remnant mass ###
    # =======================================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    ax3 = fig.add_subplot(111)

    ax3.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    # plot 1g mergers
    ax3.scatter(gen1_mass, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g+ mergers
    ax3.scatter(gen2_mass, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot 3g+ merger
    ax3.scatter(genX_mass, genX_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    plt.xlabel(r'Mass [$M_\odot$]')
    plt.ylabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]'),
    plt.yscale('log')
    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.savefig(opts.plots_directory + '/jet_lum_vs_remnant_mass.png', format='png')
    plt.close()

    # =================================
    ### shock luminosity vs. mass ratio
    # =================================
    mass_1 = mergers[:,6]
    mass_2 = mergers[:,7]
    mask = mass_1 <= mass_2 
    
    m_1_new = np.where(mask, mass_1, mass_2)
    m_2_new = np.where(mask, mass_2, mass_1)

    all_m_1_new = m_1_new
    gen1_m_1_new = all_m_1_new[merger_g1_mask]
    gen2_m_1_new = all_m_1_new[merger_g2_mask]
    genX_m_1_new = all_m_1_new[merger_gX_mask]

    all_m_2_new = m_2_new
    gen1_m__new = all_m_2_new[merger_g1_mask]
    gen2_m_2_new = all_m_2_new[merger_g2_mask]
    genX_m_2_new = all_m_2_new[merger_gX_mask]

    q = m_1_new/m_2_new

    all_q = q
    gen1_q = all_q[merger_g1_mask]
    gen2_q = all_q[merger_g2_mask]
    genX_q = all_q[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))

    ax3 = fig.add_subplot(111)

    # plot 1g mergers
    ax3.scatter(gen1_q, gen1_shock,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

    # plot the 2g+ mergers
    ax3.scatter(gen2_q, gen2_shock,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

    # plot the 3g+ mergers
    ax3.scatter(genX_q, genX_shock,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel='q',
            ylabel=r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]',
            yscale="log",
            axisbelow=True
        )
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/shock_lum_vs_q.png', format='png')
    plt.close()

    # ===============================
    ### jet luminosity vs. mass ratio
    # ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    ax3 = fig.add_subplot(111)

    ax3.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    # plot 1g mergers
    ax3.scatter(gen1_q, gen1_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

    # plot the 2g+ mergers
    ax3.scatter(gen2_q, gen2_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

    # plot the 3g+ mergers
    ax3.scatter(genX_q, genX_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel='q',
            ylabel=r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]',
            yscale="log",
            axisbelow=True
        )
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/jet_lum_vs_q.png', format='png')
    plt.close()

    # ========================================
    # q vs Chi Effective
    # ========================================

    # (q,X_eff) Figure details here:
    # Want to highlight higher generation mergers on this plot
    chi_eff = mergers[:, 3]

    # Get 1g-1g population
    gen1_chi_eff = chi_eff[merger_g1_mask]
    # 2g-1g and 2g-2g population
    gen2_chi_eff = chi_eff[merger_g2_mask]
    # >=3g-Ng population (i.e., N=1,2,3,4,...)
    genX_chi_eff = chi_eff[merger_gX_mask]
    # all 2+g mergers; H = hierarchical
    genH_chi_eff = chi_eff[(merger_g2_mask + merger_gX_mask)]
    genH_mass_ratio = q[(merger_g2_mask + merger_gX_mask)]

    # points for plotting line fit
    x = np.linspace(-1, 1, num=2)

    # fit the hierarchical mergers (any binaries with 2+g) to a line passing through 0,1
    # popt contains the model parameters, pcov the covariances
    # poptHigh, pcovHigh = curve_fit(linefunc, high_gen_mass_ratio, high_gen_chi_eff)

    # plot the 1g-1g population
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax2 = fig.add_subplot(111)
    # 1g-1g mergers
    ax2.scatter(gen1_chi_eff, gen1_q,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    ax2.scatter(gen2_chi_eff, gen2_q,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax2.scatter(genX_chi_eff, genX_q,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    if len(genH_chi_eff) > 0:
        poptHier, pcovHier = curve_fit(linefunc, genH_mass_ratio, genH_chi_eff)
        errHier = np.sqrt(np.diag(pcovHier))[0]
        # plot the line fitting the hierarchical mergers
        ax2.plot(linefunc(x, *poptHier), x,
                 ls='dashed',
                 lw=1,
                 color='gray',
                 zorder=3,
                 label=r'$d\chi/dq(\geq$2g)=' +
                       f'{poptHier[0]:.2f}' +
                       r'$\pm$' + f'{errHier:.2f}'
                 )
        #         #  alpha=linealpha,

    if len(chi_eff) > 0:
        poptAll, pcovAll = curve_fit(linefunc, q, chi_eff)
        errAll = np.sqrt(np.diag(pcovAll))[0]
        ax2.plot(linefunc(x, *poptAll), x,
                 ls='solid',
                 lw=1,
                 color='black',
                 zorder=3,
                 label=r'$d\chi/dq$(all)=' +
                       f'{poptAll[0]:.2f}' +
                       r'$\pm$' + f'{errAll:.2f}'
                 )
        #  alpha=linealpha,

    ax2.set(
        ylabel=r'$q = M_2 / M_1$',  # ($M_1 > M_2$)')
        xlabel=r'$\chi_{\rm eff}$',
        ylim=(0, 1),
        xlim=(-1, 1),
        axisbelow=True
    )

    if figsize == 'apj_col':
        ax2.legend(loc='lower left', fontsize=5)
    elif figsize == 'apj_page':
        ax2.legend(loc='lower left')

    ax2.grid('on', color='gray', ls='dotted')
    plt.savefig(opts.plots_directory + "/q_chi_eff.png", format='png')  # ,dpi=600)
    plt.close()


    # ========================================
    # Disk Radius vs Chi_p
    # ========================================

    # Can break out higher mass Chi_p events as test/illustration.
    # Set up default arrays for high mass BBH (>40Msun say) to overplot vs chi_p.
    chi_p = mergers[:, 15]
    gen1_chi_p = chi_p[merger_g1_mask]
    gen2_chi_p = chi_p[merger_g2_mask]
    genX_chi_p = chi_p[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax1 = fig.add_subplot(111)

    ax1.scatter(gen1_orb_a, gen1_chi_p,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g')

    # plot the 2g+ mergers
    ax1.scatter(gen2_orb_a, gen2_chi_p,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g')

    # plot the 3g+ mergers
    ax1.scatter(genX_orb_a, genX_chi_p,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng')
    
    plt.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    # plt.title("In-plane effective Spin vs. Merger radius")
    ax1.set(
        ylabel=r'$\chi_{\rm p}$',
        xlabel=r'Radius [$R_g$]',
        xscale='log',
        ylim=(0, 1),
        axisbelow=True)

    ax1.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax1.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax1.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.savefig(opts.plots_directory + "/r_chi_p.png", format='png')
    plt.close()


    # ===============================
    ### shock luminosity vs. kick velocity
    # ===============================
    all_vk = mergers[:, 16]
    gen1_vk = all_vk[merger_g1_mask]
    gen2_vk = all_vk[merger_g2_mask]
    genX_vk = all_vk[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plot the 1g mergers
    ax3.scatter(gen1_vk, gen1_shock,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

    # plot the 2g+ mergers
    ax3.scatter(gen2_vk, gen2_shock,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

    # plot the 3g+ mergers
    ax3.scatter(genX_vk, genX_shock,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'v_{\mathrm{Kick}} [km s$^{-1}$]',
            ylabel=r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]',
            yscale="log",
            axisbelow=True
        )
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/shock_lum_vs_vkick.png', format='png')
    plt.close()


    # ==================================
    ### jet luminosity vs. kick velocity
    # ==================================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    ax3 = fig.add_subplot(111)

    ax3.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    # plot 1g mergers
    ax3.scatter(gen1_vk,gen1_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

    # plot the 2g+ mergers
    ax3.scatter(gen2_vk, gen2_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

    # plot the 3g+ mergers
    ax3.scatter(genX_vk, genX_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'v$_{\mathrm{Kick}}$ [km s$^{-1}$]',
            ylabel=r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]',
            yscale="log",
            axisbelow=True
        )
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/jet_lum_vs_vkick.png', format='png')
    plt.close()

    # ==================================================
    ### jet luminosity vs. v_kick with histogram ###
    # ==================================================
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 

    plot = axs[0]
    hist = axs[1]

    # plot 1g-1g mergers
    plot.scatter(gen1_vk, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g-mg mergers
    plot.scatter(gen2_vk, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot 3g-ng mergers
    plot.scatter(genX_vk, genX_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    plot.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")
    plot.set_ylabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]')
    plot.set_xlabel(r'log v$_{\mathrm{Kick}}$ [km s$^{-1}$]')
    plot.set_xscale('log')
    plot.set_yscale('log')

    plot.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        plot.legend(fontsize=6, loc = 'best')
    elif figsize == 'apj_page':
        plot.legend()

    lum_bins = np.logspace(np.log10(all_jets.min()), np.log10(all_jets.max()), 50)
    hist_lum_data = [all_jets[merger_g1_mask], all_jets[merger_g2_mask], all_jets[merger_gX_mask]]

    # configure histogram
    hist.hist(hist_lum_data, bins=lum_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation = 'horizontal')
    hist.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")
    hist.grid(True, color='gray', ls='dashed')
    #hist.set_yscale('log')
    hist.yaxis.tick_right()
    hist.set_xlabel(r'n')

    #if figsize == 'apj_col':
    #    hist.legend(fontsize=6, loc='best')
    #elif figsize == 'apj_page':
    #    hist.legend()

    plt.tight_layout()

    plt.savefig(opts.plots_directory + '/jet_lum_vs_vkick_w_histogram.png', format='png')
    plt.close()

# ===============================
### shock luminosity vs. spin
# ===============================
    all_spin = mergers[:, 4]
    gen1_spin = all_spin[merger_g1_mask]
    gen2_spin = all_spin[merger_g2_mask]
    genX_spin = all_spin[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plot 1g mergers
    ax3.scatter(gen1_spin, gen1_shock,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

    # plot the 2g+ mergers
    ax3.scatter(gen2_spin, gen2_shock,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

    # plot the 3g+ mergers
    ax3.scatter(genX_spin, genX_shock,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'a$_{\mathrm{Remnant}}$',
            ylabel=r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]',
            yscale="log",
            axisbelow=True
        )
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/shock_lum_vs_spin.png', format='png')
    plt.close()

    # ===============================
    ### jet luminosity vs. spin
    # ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    ax3.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    # plot 1g mergers
    ax3.scatter(gen1_spin,gen1_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

    # plot the 2g+ mergers
    ax3.scatter(gen2_spin, gen2_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

    # plot the 3g+ mergers
    ax3.scatter(genX_spin, genX_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'a$_{\mathrm{Remnant}}$',
            ylabel=r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]',
            yscale="log",
            axisbelow=True
        )
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/jet_lum_vs_spin.png', format='png')
    plt.close() 

    # ===============================
    ### jet luminosity vs. eta
    # ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    
    ax3 = fig.add_subplot(111)

    ax3.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")

    # plot 1g mergers
    ax3.scatter(gen1_spin**2,gen1_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

    # plot the 2g+ mergers
    ax3.scatter(gen2_spin**2, gen2_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

    # plot the 3g+ mergers
    ax3.scatter(genX_spin**2, genX_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'$\eta$',
            ylabel=r'log L$_{Jet}$ [erg s$^{-1}$]',
            yscale="log",
            axisbelow=True
        )
    
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/jet_lum_vs_eta.png', format='png')
    plt.close()

######## Execution ########
if __name__ == "__main__":
    main()