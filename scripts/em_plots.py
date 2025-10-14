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

def f(x):
    return np.sqrt(x)

def g(y):
    return np.cos(y)

def jet_prob(mergers, q):
    """
    Compute jet-driving likelihood and return only mergers with jets "on".

    This function constructs a probability distribution for jet production
    based on a mass-weighted spin magnitude and the spin alignment angle.
    It then normalizes the resulting likelihood, treats it as a probability,
    and determines whether each merger produces a jet by checking a random
    number against the computed likelihood.

    What this function does is:
    1. Compute weighted spin = (|spin1| + q * |spin2|) / (1 + q).
    2. Compute spin misalignment Δθ = spin1 - spin2.
    3. Compute probability space using constants A, B and our chosen functions f and g.
    4. Normalize the probability to [0, 1].
    5. Draw random booleans with probability equal to the normalized likelihood.
    6. Return only the rows where the jet is active (True).

    Parameters
    ----------
    mergers : numpy.ndarray
        2D array where each row represents a merger event.
    q : float
        Mass ratio M1/M2, where M1 <= M2.

    Returns
    -------
    numpy.ndarray
        Subset of output_mergers_population where the final boolean (jet_on) is True.

    Notes
    -----
    - This function appends new columns to output_mergers_population:
        20: mass weighted spin
        21: delta_theta
        22: probability
        23: normalized probability
        24: boolean (jet_on)
    """
    np.random.seed(3456789108)

    # Constants
    A = -108.7
    B = 158.7

    # Compute mass weighted spin and append it to input data
    weighted_spin = (abs(mergers[:,8]) + (q * abs(mergers[:,9]))) / (1 + q)
    mergers = np.column_stack((mergers, weighted_spin))

    # Compute the absolute value of the difference between spin1 and spin2. Append to data.
    delta_theta = abs(mergers[:,8] - mergers[:,9])
    mergers = np.column_stack((mergers, delta_theta))

    # Compute the probability for each row. Append to data.
    probability_space = A * f(mergers[:,20]) * g(mergers[:,21]) + B * f(mergers[:,20])
    mergers = np.column_stack((mergers, probability_space))

    # Normalize the probability to [0, 1] range for jet on/off assignemnt. Append to data.
    jet_probability = (mergers[:,22] - mergers[:,22].min()) / (mergers[:,22].max() - mergers[:,22].min())
    mergers = np.column_stack((mergers, jet_probability))

    # Generate boolean: True with random number <= jet probability. Append to data.
    random_boolean = np.random.rand(len(mergers)) < mergers[:,23]
    mergers = np.column_stack((mergers, random_boolean))

    # Return only mergers with jets "on"
    jet_on = mergers[mergers[:,24] == 1]
    return jet_on

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

    # Bolometric luminosity of the AGN, which is an output from pAGN (in [erg s**-1])
    lum_agn = 1.4705385593922858e+46

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
        hist.legend(fontsize=6, loc='best', bbox_to_anchor=(1.0, 0.5))
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

    if figsize == 'apj_col':
        hist.legend(fontsize=6, loc='best', bbox_to_anchor=(1.0, 0.5))
    elif figsize == 'apj_page':
        hist.legend()

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

    if figsize == 'apj_col':
        hist.legend(fontsize=6, loc='best', bbox_to_anchor=(1.0, 0.5))
    elif figsize == 'apj_page':
        hist.legend()

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

    # ===============================
    ### jet luminosity vs. mass ratio
    # ===============================
    jets_on = jet_prob(mergers, q)

    jet_g1_mask, jet_g2_mask, jet_gX_mask = make_gen_masks(jets_on, 12, 13)

    # Ensure no union between sets
    assert all(jet_g1_mask & jet_g2_mask) == 0
    assert all(jet_g1_mask & jet_gX_mask) == 0
    assert all(jet_g2_mask & jet_gX_mask) == 0

    # Ensure no elements are missed
    assert all(jet_g1_mask | jet_g2_mask | jet_gX_mask) == 1
    
    all_orb_a_jets = jets_on[:, 1]
    gen1_orb_a_jets = all_orb_a_jets[jet_g1_mask]
    gen2_orb_a_jets = all_orb_a_jets[jet_g2_mask]
    genX_orb_a_jets = all_orb_a_jets[jet_gX_mask]

    all_jets_on = jets_on[:,18]
    gen1_jet_on = all_jets_on[jet_g1_mask]
    gen2_jet_on = all_jets_on[jet_g2_mask]
    genX_jet_on= all_jets_on[jet_gX_mask]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 

    plot = axs[0]
    hist = axs[1]

    # plot 1g-1g mergers
    plot.scatter(gen1_orb_a_jets, gen1_jet_on,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g-mg mergers
    plot.scatter(gen2_orb_a_jets, gen2_jet_on,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot 3g-ng mergers
    plot.scatter(genX_orb_a_jets, genX_jet_on,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    plot.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    
    plot.set_ylabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]')
    plot.set_xlabel(r'log Radius [$R_g$]')
    plot.set_xscale('log')
    plot.set_yscale('log')

    plot.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        plot.legend(fontsize=6, loc = 'best')
    elif figsize == 'apj_page':
        plot.legend()

    lum_bins_jets_on = np.logspace(np.log10(all_jets_on.min()), np.log10(all_jets_on.max()), 50)
    hist_lum_data_jets_on = [all_jets_on[jet_g1_mask], all_jets_on[jet_g2_mask], all_jets_on[jet_gX_mask]]

    # configure histogram
    hist.hist(hist_lum_data_jets_on, bins=lum_bins_jets_on, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation = 'horizontal')
    hist.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")
    hist.grid(True, color='gray', ls='dashed')
    #hist.set_yscale('log')
    hist.yaxis.tick_right()
    hist.set_xlabel(r'n')

    if figsize == 'apj_col':
        hist.legend(fontsize=6, loc='best', bbox_to_anchor=(1.0, 0.5))
    elif figsize == 'apj_page':
        hist.legend()

    plt.tight_layout()

    plt.savefig(opts.plots_directory + '/jet_lum_vs_radius_w_histogram_jets_on.png', format='png')
    plt.close()

    # ===============================
    ### jet luminosity vs. mass ratio
    # ===============================
    all_vk_jets = jets_on[:, 16]
    gen1_vk_jets = all_vk_jets[jet_g1_mask]
    gen2_vk_jets = all_vk_jets[jet_g2_mask]
    genX_vk_jets = all_vk_jets[jet_gX_mask]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 

    plot = axs[0]
    hist = axs[1]

    # plot 1g-1g mergers
    plot.scatter(gen1_vk_jets, gen1_jet_on,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g-mg mergers
    plot.scatter(gen2_vk_jets, gen2_jet_on,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot 3g-ng mergers
    plot.scatter(genX_vk_jets, genX_jet_on,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    plot.set_ylabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]')
    plot.set_xlabel(r'log v$_{\mathrm{Kick}}$ [km s$^{-1}$]')
    plot.set_xscale('log')
    plot.set_yscale('log')

    plot.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        plot.legend(fontsize=6, loc = 'best')
    elif figsize == 'apj_page':
        plot.legend()

    lum_bins_jets_on = np.logspace(np.log10(all_jets_on.min()), np.log10(all_jets_on.max()), 50)
    hist_lum_data_jets_on = [all_jets_on[jet_g1_mask], all_jets_on[jet_g2_mask], all_jets_on[jet_gX_mask]]

    # configure histogram
    hist.hist(hist_lum_data_jets_on, bins=lum_bins_jets_on, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation = 'horizontal')
    hist.axhline(lum_agn, color='black', linewidth=1, linestyle='dashdot', label=r'L$_{AGN}$ ='+f"{lum_agn:.2e}")
    hist.grid(True, color='gray', ls='dashed')
    hist.yaxis.tick_right()
    hist.set_xlabel(r'n')

    if figsize == 'apj_col':
        hist.legend(fontsize=6, loc='best', bbox_to_anchor=(1.0, 0.5))
    elif figsize == 'apj_page':
        hist.legend()

    plt.tight_layout()

    plt.savefig(opts.plots_directory + '/jet_lum_vs_vkick_w_histogram_jets_on.png', format='png')
    plt.close()

######## Execution ########
if __name__ == "__main__":
    main()