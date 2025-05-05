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
import seaborn as sns
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
    plt.axvline(10**46, linewidth = 1, linestyle = 'dashed', color = 'red', label = r"~Seyfert I AGN")

    plt.ylabel(r'N')
    plt.xlabel(r'Luminosity [erg/s]')
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
    bins = np.arange(int(shock_log.min()), int(shock_log.max()), 0.1)

    hist_data = [shock_log[merger_g1_mask], shock_log[merger_g2_mask], shock_log[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    #plt.hist(mergers[:, 18], bins = jet_bins)

    plt.ylabel(r'n')
    plt.xlabel(r'log_10(Shock Luminsoity) (erg/s)')
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
    bins = np.arange(int(jet_log.min()), int(jet_log.max()), 0.2)
    # check end cases and check print()

    hist_data = [jet_log[merger_g1_mask], jet_log[merger_g2_mask], jet_log[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    #plt.hist(mergers[:, 18], bins = jet_bins)

    plt.ylabel(r'n')
    plt.xlabel(r'log_10(Jet Luminsoity) (erg/s)')
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

    fig = plt.figure()
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
    plt.scatter(gen1_mass, gen1_jet,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_mass, gen2_jet,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_mass, genX_jet,
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
    plt.legend(loc ='best')
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
### jet luminosity vs. time
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
            xlabel='Spin',
            ylabel='Jet Lum [erg/s]',
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


# ===============================
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
    # ========================================
    # LVK and LISA Strain vs Freq
    # ========================================

    # plot for each merger generation seperately with heatmap


    # Read LIGO O3 sensitivity data (https://git.ligo.org/sensitivity-curves/o3-sensitivity-curves)
    H1 = impresources.files(data) / 'O3-H1-C01_CLEAN_SUB60HZ-1262197260.0_sensitivity_strain_asd.txt'
    L1 = impresources.files(data) / 'O3-L1-C01_CLEAN_SUB60HZ-1262141640.0_sensitivity_strain_asd.txt'

    # Adjust sep according to your delimiter (e.g., '\t' for tab-delimited files)
    dfh1 = pd.read_csv(H1, sep='\t', header=None)  # Use header=None if the file doesn't contain header row
    dfl1 = pd.read_csv(L1, sep='\t', header=None)

    # Access columns as df[0], df[1], ...
    f_H1 = dfh1[0]
    h_H1 = dfh1[1]

    # H - hanford
    # L - Ligvston

    # Using https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/blob/master/LISA.py
    # Create LISA object
    lisa = li.LISA()

    #   lisa_freq is the frequency (x-axis) being created
    #   lisa_sn is the sensitivity curve of LISA
    lisa_freq = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
    lisa_sn = lisa.Sn(lisa_freq)

    # Create figure and ax
    fig, svf_ax = plt.subplots(1, figsize=(plotting.set_size(figsize)[0], 2.9))

    svf_ax.set_xlabel(r'f [Hz]')  # , fontsize=20, labelpad=10)
    svf_ax.set_ylabel(r'${\rm h}_{\rm char}$')  # , fontsize=20, labelpad=10)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    svf_ax.set_xlim(0.5e-7, 1.0e+4)
    svf_ax.set_ylim(1.0e-28, 1.0e-15)

    # ----------Finding the rows in which EMRIs signals are either identical or zeroes and removing them----------
    identical_rows_emris = np.where(emris[:, 5] == emris[:, 6])
    zero_rows_emris = np.where(emris[:, 6] == 0)
    emris = np.delete(emris, identical_rows_emris, 0)
    # emris = np.delete(emris,zero_rows_emris,0)
    emris[~np.isfinite(emris)] = 1.e-40

    # ----------Finding the rows in which LVKs signals are either identical or zeroes and removing them----------
    identical_rows_lvk = np.where(lvk[:, 5] == lvk[:, 6])
    zero_rows_lvk = np.where(lvk[:, 6] == 0)
    lvk = np.delete(lvk, identical_rows_lvk, 0)
    # lvk = np.delete(lvk,zero_rows_lvk,0)
    lvk[~np.isfinite(lvk)] = 1.e-40


    # ----------Setting the values for the EMRIs and LVKs signals and inverting them----------
    inv_freq_emris = 1 / emris[:, 6]
    # inv_freq_lvk = 1/lvk[:,6]
    # ma_freq_emris = np.ma.where(freq_emris == 0)
    # ma_freq_lvk = np.ma.where(freq_lvk == 0)
    # indices_where_zeros_emris = np.where(freq_emris = 0.)
    # freq_emris = freq_emris[freq_emris !=0]
    # freq_lvk = freq_lvk[freq_lvk !=0]

    # inv_freq_emris = 1.0/ma_freq_emris
    # inv_freq_lvk = 1.0/ma_freq_lvk
    # timestep =1.e4yr
    timestep = 1.e4
    strain_per_freq_emris = emris[:, 5] * inv_freq_emris / timestep

    strain_per_freq_lvk = lvk[:, 5] * (1 / lvk[:, 6]) / timestep

    # plot the characteristic detector strains
    svf_ax.loglog(lisa_freq, np.sqrt(lisa_freq * lisa_sn),
              label='LISA Sensitivity',
              #   color='darkred',
              zorder=0)

    svf_ax.loglog(f_H1, h_H1,
              label='LIGO O3, H1 Sensitivity',
              #   color='darkblue',
              zorder=0)

    svf_ax.scatter(emris[:, 6], strain_per_freq_emris,
               s=0.4 * styles.markersize_gen1,
               alpha=styles.markeralpha_gen1
               )

    scatter = svf_ax.scatter(
    lvk[:, 6], strain_per_freq_lvk,
    s=0.4 * styles.markersize_gen1,
    marker=styles.marker_gen1,
    edgecolor=styles.color_gen1,
    facecolor='none',
    alpha=styles.markeralpha_gen1,
    label='1g-1g',
    c=jet_log, cmap='viridis'  # <-- fixed position, no missing comma
    )

    # Add colorbar for metallicity
    cbar = plt.colorbar(scatter, ax=svf_ax)
    cbar.set_label('Metallicity [Fe/H]')

    # Set log-log scale
    svf_ax.set_yscale('log')
    svf_ax.set_xscale('log')

    # ax.loglog(f_L1, h_L1,label = 'LIGO O3, L1 Sensitivity') # plot the characteristic strain
    # ax.loglog(f_gw,h,color ='black', label='GW150914')

    if figsize == 'apj_col':
        plt.legend(fontsize=7, loc="upper right")
    elif figsize == 'apj_page':
        plt.legend(loc="upper right")

    svf_ax.set_xlabel(r'$\nu_{\rm GW}$ [Hz]')  # , fontsize=20, labelpad=10)
    svf_ax.set_ylabel(r'$h_{\rm char}/\nu_{\rm GW}$')  # , fontsize=20, labelpad=10)

    plt.savefig(opts.plots_directory + './gw_strain_em.png', format='png')
    plt.close()
    """
# ===============================
### kde test
# ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    sns.kdeplot(x=np.log10(gen1_rho), y=np.log10(gen1_jet), fill=False, levels=10, color="green")
    sns.kdeplot(x=np.log10(gen2_rho), y=np.log10(gen2_jet), fill=False, levels=10, color="purple")
    sns.kdeplot(x=np.log10(genX_rho), y=np.log10(genX_jet), fill=False, levels=10, color="red")


    plt.xlabel(r'log10Density [cm$^3$/g]')
    plt.ylabel("log10Jet Luminosity [erg/s]")

    plt.savefig(opts.plots_directory + '/kde.png', format='png')
    plt.close()

"""
# ===============================
### KDE test
# ===============================
    log_rho_all = (mergers[:,4])
    log_shock_all = np.log10(mergers[:, 18])

    # Generation-wise log values
    log_rho_gen1 = log_rho_all[merger_g1_mask]
    log_shock_gen1 = log_shock_all[merger_g1_mask]

    log_rho_gen2 = log_rho_all[merger_g2_mask]
    log_shock_gen2 = log_shock_all[merger_g2_mask]

    log_rho_genX = log_rho_all[merger_gX_mask]
    log_shock_genX = log_shock_all[merger_gX_mask]

    # Set up the plot
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    # Grid for evaluation
    xmin, xmax = log_rho_all.min(), log_rho_all.max()
    ymin, ymax = log_shock_all.min(), log_shock_all.max()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    # KDE and contour for gen1 (dotted lines)
    kde_gen1 = gaussian_kde(np.vstack([log_rho_gen1, log_shock_gen1]))
    zz1 = kde_gen1(grid_coords).reshape(xx.shape)
    levels1 = np.linspace(zz1.min(), zz1.max(), 10)[1:]  # 6 levels, skipping the lowest
    alphas1 = np.linspace(0.5, 1.0, len(levels1))
    for i, level in enumerate(levels1):
        ax.contour(xx, yy, zz1, levels=[level], colors=[styles.color_gen1], alpha=alphas1[i], linestyles='dotted', linewidths=1.)

    # KDE and contour for gen2 (dashed lines)
    kde_gen2 = gaussian_kde(np.vstack([log_rho_gen2, log_shock_gen2]))
    zz2 = kde_gen2(grid_coords).reshape(xx.shape)
    levels2 = np.linspace(zz2.min(), zz2.max(), 10)[1:]
    alphas2 = np.linspace(0.5, 1.0, len(levels2))
    for i, level in enumerate(levels2):
        ax.contour(xx, yy, zz2, levels=[level], colors=[styles.color_gen2], alpha=alphas2[i], linestyles='dashed', linewidths=1.)

    # KDE and filled contour for genX
    kde_genX = gaussian_kde(np.vstack([log_rho_genX, log_shock_genX]))
    zzX = kde_genX(grid_coords).reshape(xx.shape)
    levelsX = np.linspace(zzX.min(), zzX.max(), 10)[1:]
    alphasX = np.linspace(0.5, 1.0, len(levelsX))
    ax.contourf(xx, yy, zzX, levels=levelsX, cmap='Reds', alpha=0.8)  # Filled contour for genX

    # Labels and styling
    ax.set(
        xlabel=r'log$_{10}$(Density) [cm$^3$/g]',
        ylabel=r'log$_{10}$(Shock Luminosity) [erg/s]',
        axisbelow=True
    )
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + '/kde_by_generation.png', format='png')
    plt.close()
"""


######## Execution ########
if __name__ == "__main__":
    main()
