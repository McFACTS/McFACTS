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
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data
from mcfacts.vis import plotting
from mcfacts.vis import styles
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
#import seaborn as sns
#from scipy.stats import gaussian_kde


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
    parser.add_argument("--fname-emris",
                        default="output_mergers_emris.dat",
                        type=str, help="output_emris file")
    parser.add_argument("--fname-mergers",
                        default="output_mergers_population.dat",
                        type=str, help="output_mergers file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    parser.add_argument("--fname-lvk",
                        default="output_mergers_lvk.dat",
                        type=str, help="output_lvk file")
    opts = parser.parse_args()
    print(opts.fname_mergers)
    assert os.path.isfile(opts.fname_mergers)
    assert os.path.isfile(opts.fname_emris)
    assert os.path.isfile(opts.fname_lvk)
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
    emris = np.loadtxt(opts.fname_emris, skiprows=2)
    lvk = np.loadtxt(opts.fname_lvk, skiprows=2)

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
### comparison histogram ###
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    shock_bins = np.logspace(np.log10(mergers[:, 17].min()), np.log10(mergers[:, 17].max()), 50)
    jet_bins = np.logspace(np.log10(mergers[:, 18].min()), np.log10(mergers[:, 18].max()), 50)
    plt.hist(mergers[:, 17], bins = shock_bins, label = 'Shock')
    plt.hist(mergers[:, 18], bins = jet_bins, label = 'Jet', alpha = 0.8)
    plt.axvline(10**46, linewidth = 1, linestyle = 'dashed', color = 'red', label = r"Fiducial L$_{QSO}$")

    plt.ylabel(r'n')
    plt.xlabel(r'Luminosity [erg s$^{-1}$]')
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

    plt.savefig(opts.plots_directory + '/luminosity_shock_vs_time.png', format='png')
    plt.close()

# ===============================
### jet lum vs time ###
# ===============================
    gen1_jet = mergers[:, 18][merger_g1_mask]
    gen2_jet = mergers[:, 18][merger_g2_mask]
    genX_jet= mergers[:, 18][merger_gX_mask]

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

    plt.savefig(opts.plots_directory + '/luminosity_jet_vs_time.png', format='png')
    plt.close()

# ===============================
### shock luminosity distribution histogram ###
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    shock_log = np.log10(mergers[:, 17])
    #jet_bins = np.logspace(np.log10(mergers[:, 18].min()), np.log10(mergers[:, 18].max()), 50)
    counts, bins = np.histogram(shock_log)
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(shock_log.min()), int(shock_log.max()), 0.2)

    hist_data = [shock_log[merger_g1_mask], shock_log[merger_g2_mask], shock_log[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    #plt.hist(mergers[:, 18], bins = jet_bins)

    plt.ylabel(r'n')
    plt.xlabel(r'log L$_{\mathrm{Shock}}$ [erg s$^{-1}$]')
    #plt.xscale('log')
    #plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    #plt.ylim(0.4, 325)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/luminosity_shock_dist.png", format='png')
    plt.close()

# ===============================
### jet luminosity distribution histogram ###
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    jet_log = np.log10(mergers[:, 18])
    #jet_bins = np.logspace(np.log10(mergers[:, 18].min()), np.log10(mergers[:, 18].max()), 50)
    counts, bins = np.histogram(jet_log)
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(jet_log.min()), int(jet_log.max())+1, 0.2)
    # check end cases and check print()

    hist_data = [jet_log[merger_g1_mask], jet_log[merger_g2_mask], jet_log[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    #plt.hist(mergers[:, 18], bins = jet_bins)

    plt.ylabel(r'n')
    plt.xlabel(r'log L$_{\mathrm{Jet}}$ [erg s$^{-1}$]')
    #plt.xscale('log')
    #plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    #plt.ylim(0.4, 325)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/luminosity_jet_dist.png", format='png')
    plt.close()

# ===============================
### shock luminosity vs. a_bin ###
# ===============================
    all_orb_a = mergers[:, 1]
    gen1_orb_a = all_orb_a[merger_g1_mask]
    gen2_orb_a = all_orb_a[merger_g2_mask]
    genX_orb_a = all_orb_a[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    ax3.scatter(gen1_orb_a, gen1_shock,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    ax3.scatter(gen2_orb_a, gen2_shock,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    ax3.scatter(genX_orb_a, genX_shock,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    trap_radius = 700
    ax3.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Shock Lum [erg/s]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    # if figsize == 'apj_col':
    #     ax3.legend(fontsize=6)
    # elif figsize == 'apj_page':
    #     ax3.legend()

    plt.savefig(opts.plots_directory + '/radius_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. a_bin ###
# ===============================
    all_orb_a = mergers[:, 1]
    gen1_orb_a = all_orb_a[merger_g1_mask]
    gen2_orb_a = all_orb_a[merger_g2_mask]
    genX_orb_a = all_orb_a[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    ax3.scatter(gen1_orb_a, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    ax3.scatter(gen2_orb_a, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    ax3.scatter(genX_orb_a, genX_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    trap_radius = 700
    ax3.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    
    # if figsize == 'apj_col':
    #     ax3.legend(fontsize=6)
    # elif figsize == 'apj_page':
    #     ax3.legend()

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Jet Lum [erg/s]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    #plt.legend(loc ='best')
    plt.savefig(opts.plots_directory + '/radius_vs_jet_lum.png', format='png')
    plt.close()

# ===============================
### shock luminosity vs. remnant mass ###
# ===============================
    all_mass = mergers[:, 2]
    gen1_mass = all_mass[merger_g1_mask]
    gen2_mass = all_mass[merger_g2_mask]
    genX_mass = all_mass[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    ax3.scatter(gen1_mass, gen1_shock,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    ax3.scatter(gen2_mass, gen2_shock,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    ax3.scatter(genX_mass, genX_shock,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Shock Lum [erg/s]')
    plt.xlabel(r'Mass [$M_\odot$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/remnant_mass_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. remnant mass ###
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    ax3.scatter(gen1_mass, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    ax3.scatter(gen2_mass, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    ax3.scatter(genX_mass, genX_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    plt.ylabel(r'Jet Lum [erg/s]')
    plt.xlabel(r'Mass [$M_\odot$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()
    #plt.legend(loc ='best')
    plt.savefig(opts.plots_directory + '/remnant_mass_vs_jet_lum.png', format='png')
    plt.close()


# ===============================
### shock luminosity vs. mass ratio
# ===============================
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

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
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
            ylabel='Shock Lum [erg/s]',
            yscale="log",
            axisbelow=True
        )
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/q_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. mass ratio
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
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
            ylabel='Jet Lum [erg/s]',
            yscale="log",
            axisbelow=True
        )
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/q_vs_jet_lum.png', format='png')
    plt.close()

# ===============================
### testing color map stuff for jets ###
# ===============================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(mergers[:,2], mergers[:,16], c=np.log10(mergers[:,18]), cmap="viridis", marker="+", s=1)
    plt.colorbar(label='Jet Lum')
        
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.ylabel(r'Kick Velocity [km/s]')
    plt.xscale('log')
    #plt.yscale('log')
    plt.grid(True, color='gray', ls='dashed')

    plt.savefig(opts.plots_directory + '/mass_vs_vel_vs_jet_lum.png', format='png')
    plt.close()

# ===============================
### testing color map stuff for shocks ###
# ===============================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(mergers[:,2], mergers[:,16], c=np.log10(mergers[:,17]), cmap="viridis", marker="+", s=1)
    plt.colorbar(label='Shock Lum')
    
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.ylabel(r'Kick Velocity [km/s]')
    plt.xscale('log')
    #plt.yscale('log')
    plt.grid(True, color='gray', ls='dashed')

    plt.savefig(opts.plots_directory + '/mass_vs_vel_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### shock luminosity vs. time
# ===============================

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
            xlabel='Time Merged  / 1e6',
            ylabel='Shock Lum [erg/s]',
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/time_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. time
# ===============================
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
            xlabel='Time Merged  / 1e6',
            ylabel='Jet Lum [erg/s]',
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/time_vs_jet_lum.png', format='png')
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

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
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
            xlabel='Kick Velocity [km/s]',
            ylabel='Shock Lum [erg/s]',
            #xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/vk_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. kick velocity
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
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
            xlabel='Kick Velocity [km/s]',
            ylabel='Jet Lum [erg/s]',
            #xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/vk_vs_jet_lum.png', format='png')
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

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
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
            xlabel='Spin',
            ylabel='Shock Lum [erg/s]',
            #xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/spin_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. spin
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
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
            xlabel=r'a$_{\mathrm{remnant}}$',
            ylabel=r'L$_{Jet}$ [erg s$^{-1}$]',
            #xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/spin_vs_jet_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. eta
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
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
            xlabel='Spin',
            ylabel='Jet Lum [erg/s]',
            #xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/eta_vs_jet_lum.png', format='png')
    plt.close()


# ===============================
### shock luminosity vs. disk density
# ===============================
    factor = (mergers[:,18] / 2.5e45) * 1e-9
    density = factor * (0.1 / mergers[:,4]**2) * (100 / mergers[:,2])**2 * (mergers[:,16] / 200)**3

    all_rho = density
    gen1_rho = all_rho[merger_g1_mask]
    gen2_rho = all_rho[merger_g2_mask]
    genX_rho = all_rho[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(gen1_rho, gen1_shock,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

        # plot the 2g+ mergers
    ax3.scatter(gen2_rho, gen2_shock,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

        # plot the 3g+ mergers
    ax3.scatter(genX_rho, genX_shock,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'Density [cm$^3$/g]',
            ylabel='Shock Lum [erg/s]',
            xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/density_vs_shock_lum.png', format='png')
    plt.close()

# ===============================
### jet luminosity vs. disk density
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    ax3.scatter(gen1_rho,gen1_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

        # plot the 2g+ mergers
    ax3.scatter(gen2_rho, gen2_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

        # plot the 3g+ mergers
    ax3.scatter(genX_rho, genX_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'Density [cm$^3$/g]',
            ylabel='Jet Lum [erg/s]',
            xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/density_vs_jet_lum.png', format='png')
    plt.close()


## here
# ===============================
### luminous jet luminosity vs. remnant mass ###
# ===============================
    factor = (mergers[:,18] / 2.5e45) * 1e-9
    density = factor * (0.1 / mergers[:,4]**2) * (100 / mergers[:,2])**2 * (mergers[:,16] / 200)**3 
    mergers[:,19] = density
    luminous_stuff = mergers[mergers[:, 19] >= 10**-9]
        # Exclude all rows with NaNs or zeros in the final mass column
    luminous_stuff_nan_mask = (np.isfinite(luminous_stuff[:, 2])) & (luminous_stuff[:, 2] != 0)
    luminous_stuff = luminous_stuff[luminous_stuff_nan_mask]

    lum_g1_mask, lum_g2_mask, lum_gX_mask = make_gen_masks(luminous_stuff, 12, 13)

    # Ensure no union between sets
    assert all(lum_g1_mask & lum_g2_mask) == 0
    assert all(lum_g1_mask & lum_gX_mask) == 0
    assert all(lum_g2_mask & lum_gX_mask) == 0

    # Ensure no elements are missed
    assert all(lum_g1_mask | lum_g2_mask | lum_gX_mask) == 1

    all_luminous_jets = luminous_stuff[:,18]
    gen1_luminous_jet = all_luminous_jets[lum_g1_mask]
    gen2_luminous_jet = all_luminous_jets[lum_g2_mask]
    genX_luminous_jet = all_luminous_jets[lum_gX_mask]

    all_luminous_jets_mass = luminous_stuff[:,2]
    gen1_luminous_jet_mass = all_luminous_jets_mass[lum_g1_mask]
    gen2_luminous_jet_mass = all_luminous_jets_mass[lum_g2_mask]
    genX_luminous_jet_mass = all_luminous_jets_mass[lum_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_luminous_jet_mass, gen1_luminous_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_luminous_jet_mass, gen2_luminous_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_luminous_jet_mass, genX_luminous_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    plt.ylabel(r'Jet Lum [erg/s]')
    plt.xlabel(r'Mass [$M_\odot$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    #if figsize == 'apj_col':
     #   plt.legend(fontsize=6)
    #elif figsize == 'apj_page':
        #plt.legend()
    #plt.legend(loc ='best')
    plt.savefig(opts.plots_directory + '/remnant_mass_vs_luminous_jet_lum.png', format='png')
    plt.close()

# ===============================
### luminous jet luminosity vs. mass ratio
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    mass_1_lum = luminous_stuff[:,6]
    mass_2_lum = luminous_stuff[:,7]
    mask_lum = mass_1_lum <= mass_2_lum 
    
    m_1_new_lum = np.where(mask_lum, mass_1_lum, mass_2_lum)
    m_2_new_lum = np.where(mask_lum, mass_2_lum, mass_1_lum)

    q_lum = m_1_new_lum/m_2_new_lum

    all_q_lum = q_lum
    gen1_q_lum = all_q_lum[lum_g1_mask]
    gen2_q_lum = all_q_lum[lum_g2_mask]
    genX_q_lum = all_q_lum[lum_gX_mask]

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(gen1_q_lum, gen1_luminous_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

        # plot the 2g+ mergers
    ax3.scatter(gen2_q_lum, gen2_luminous_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

        # plot the 3g+ mergers
    ax3.scatter(genX_q_lum, genX_luminous_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel='q',
            ylabel='Jet Lum [erg/s]',
            yscale="log",
            axisbelow=True
        )
    
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/q_vs_luminous_jet_lum.png', format='png')
    plt.close()

# ===============================
### luminous jet luminosity vs. kick velocity
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    
    all_luminous_jets_vkick = luminous_stuff[:,16]
    gen1_luminous_jet_vkick = all_luminous_jets_vkick[lum_g1_mask]
    gen2_luminous_jet_vkick = all_luminous_jets_vkick[lum_g2_mask]
    genX_luminous_jet_vkick = all_luminous_jets_vkick[lum_gX_mask]

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(gen1_luminous_jet_vkick,gen1_luminous_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

        # plot the 2g+ mergers
    ax3.scatter(gen2_luminous_jet_vkick, gen2_luminous_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

        # plot the 3g+ mergers
    ax3.scatter(genX_luminous_jet_vkick, genX_luminous_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel='Kick Velocity [km/s]',
            ylabel='Jet Lum [erg/s]',
            #xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/vk_vs_luminous_jet_lum.png', format='png')
    plt.close()

# ===============================
### luminous_ jet luminosity vs. eta
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    
    all_luminous_stuff_spin = luminous_stuff[:, 4]
    luminous_stuff_gen1_spin = all_luminous_stuff_spin[lum_g1_mask]
    luminous_stuff_gen2_spin = all_luminous_stuff_spin[lum_g2_mask]
    luminous_stuff_genX_spin = all_luminous_stuff_spin[lum_gX_mask]

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(luminous_stuff_gen1_spin**2,gen1_luminous_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

        # plot the 2g+ mergers
    ax3.scatter(luminous_stuff_gen2_spin**2, gen2_luminous_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

        # plot the 3g+ mergers
    ax3.scatter(luminous_stuff_genX_spin**2, genX_luminous_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel='Spin',
            ylabel='Jet Lum [erg/s]',
            #xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/eta_vs_luminous_jet_lum.png', format='png')
    plt.close()


# ===============================
### luminous_ jet luminosity vs. disk density
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    all_luminous_jets_density = luminous_stuff[:,19]
    gen1_luminous_jet_density = all_luminous_jets_density[lum_g1_mask]
    gen2_luminous_jet_density = all_luminous_jets_density[lum_g2_mask]
    genX_luminous_jet_density = all_luminous_jets_density[lum_gX_mask]

    ax3.scatter(gen1_luminous_jet_density,gen1_luminous_jet,
                    s=styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    label='1g-1g'
                    )

        # plot the 2g+ mergers
    ax3.scatter(gen2_luminous_jet_density, gen2_luminous_jet,
                    s=styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    label='2g-1g or 2g-2g'
                    )

        # plot the 3g+ mergers
    ax3.scatter(genX_luminous_jet_density, genX_luminous_jet,
                    s=styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    label=r'$\geq$3g-Ng'
                    )

    ax3.set(
            xlabel=r'Density [cm$^3$/g]',
            ylabel='Jet Lum [erg/s]',
            xscale="log",
            yscale="log",
            axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/density_vs_luminous_jet_lum.png', format='png')
    plt.close()

# ===============================
### luminous shit a_bin vs. luminosity ###
# ===============================
    all_lum_orb_a = luminous_stuff[:, 1]
    gen1_orb_a_lum = all_lum_orb_a[lum_g1_mask]
    gen2_orb_a_lum = all_lum_orb_a[lum_g2_mask]
    genX_orb_a_lum = all_lum_orb_a[lum_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    ax3.scatter(gen1_orb_a_lum, gen1_luminous_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    ax3.scatter(gen2_orb_a_lum, gen2_luminous_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    ax3.scatter(genX_orb_a_lum, genX_luminous_jet,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    trap_radius = 700
    ax3.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    
    # if figsize == 'apj_col':
    #     ax3.legend(fontsize=6)
    # elif figsize == 'apj_page':
    #     ax3.legend()

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Jet Lum [erg/s]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    #plt.legend(loc ='best')
    plt.savefig(opts.plots_directory + '/radius_vs_luminous_jet_lum.png', format='png')
    plt.close()


"""# ===============================
### testing corr for jets ###
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(mergers[:,1], density, c=np.log10(mergers[:,18]), cmap="copper", marker="o", s=25)
    plt.colorbar(label='Jet Lum')
        
    plt.xlabel(r'Radius [R$_{g}$]')
    plt.ylabel(r'Density [cm$^3$/g]')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, color='gray', ls='dashed')

    plt.savefig(opts.plots_directory + '/radius_vs_density_vs_jet_lum.png', format='png')
    plt.close()
"""


######## Execution ########
if __name__ == "__main__":
    main()
