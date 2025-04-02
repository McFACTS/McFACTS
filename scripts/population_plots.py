#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import os
from scipy.optimize import curve_fit
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data
from mcfacts.vis import plotting
from mcfacts.vis import styles
from mcfacts.outputs.ReadOutputs import ReadLog

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

figsize = "apj_col"


######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--fname-log",
                        default="mcfacts.log",
                        type=str, help="log file")
    opts = parser.parse_args()
    print(opts.fname_mergers)
    assert os.path.isfile(opts.fname_mergers)
    assert os.path.isfile(opts.fname_emris)
    assert os.path.isfile(opts.fname_lvk)
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


######## Main ########
def main():
    # plt.style.use('seaborn-v0_8-poster')

    # Load data from output files
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


    # ========================================
    # Number of Mergers vs Mass
    # ========================================

    # Plot intial and final mass distributions
    fig = plt.figure(figsize=plotting.set_size(figsize))
    counts, bins = np.histogram(mergers[:, 2])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(mergers[:, 2].min()), int(mergers[:, 2].max()) + 2, 1)

    hist_data = [mergers[:, 2][merger_g1_mask], mergers[:, 2][merger_g2_mask], mergers[:, 2][merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.xscale('log')
    # plt.ylim(-5,max(counts))
    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    svf_ax.tick_params(axis='x', direction='out', which='both')
    #plt.grid(True, color='gray', ls='dashed')
    svf_ax.yaxis.grid(True, color='gray', ls='dashed')

    plt.xticks(np.geomspace(int(mergers[:, 2].min()), int( mergers[:, 2].max()), 5).astype(int))
    #plt.xticks(np.geomspace(20, 200, 5).astype(int))

    svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.savefig(opts.plots_directory + r"/merger_remnant_mass.png", format='png')

    plt.close()


    # ========================================
    # Merger Mass vs Radius
    # ========================================

    # Read the log file
    log_data = ReadLog(opts.fname_log)

    # Retrieve the migration trap radius used in run
    trap_radius = log_data["disk_radius_trap"]

    # plt.title('Migration Trap influence')
    for i in range(len(mergers[:, 1])):
        if mergers[i, 1] < 10.0:
            mergers[i, 1] = 10.0

    # Separate generational subpopulations
    gen1_orb_a = mergers[:, 1][merger_g1_mask]
    gen2_orb_a = mergers[:, 1][merger_g2_mask]
    genX_orb_a = mergers[:, 1][merger_gX_mask]
    gen1_mass = mergers[:, 2][merger_g1_mask]
    gen2_mass = mergers[:, 2][merger_g2_mask]
    genX_mass = mergers[:, 2][merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_orb_a, gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_orb_a, gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_orb_a, genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plt.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Remnant Mass [$M_\odot$]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.ylim(18, 1000)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/merger_mass_v_radius.png", format='png')
    plt.close()


    # ========================================
    # q vs Chi Effective
    # ========================================

    # retrieve component masses and mass ratio
    m1 = np.zeros(mergers.shape[0])
    m2 = np.zeros(mergers.shape[0])
    mass_ratio = np.zeros(mergers.shape[0])
    for i in range(mergers.shape[0]):
        if mergers[i, 6] < mergers[i, 7]:
            m1[i] = mergers[i, 7]
            m2[i] = mergers[i, 6]
            mass_ratio[i] = mergers[i, 6] / mergers[i, 7]
        else:
            mass_ratio[i] = mergers[i, 7] / mergers[i, 6]
            m1[i] = mergers[i, 6]
            m2[i] = mergers[i, 7]

    # (q,X_eff) Figure details here:
    # Want to highlight higher generation mergers on this plot
    chi_eff = mergers[:, 3]

    # Get 1g-1g population
    gen1_chi_eff = chi_eff[merger_g1_mask]
    gen1_mass_ratio = mass_ratio[merger_g1_mask]
    # 2g-1g and 2g-2g population
    gen2_chi_eff = chi_eff[merger_g2_mask]
    gen_mass_ratio = mass_ratio[merger_g2_mask]
    # >=3g-Ng population (i.e., N=1,2,3,4,...)
    genX_chi_eff = chi_eff[merger_gX_mask]
    genX_mass_ratio = mass_ratio[merger_gX_mask]
    # all 2+g mergers; H = hierarchical
    genH_chi_eff = chi_eff[(merger_g2_mask + merger_gX_mask)]
    genH_mass_ratio = mass_ratio[(merger_g2_mask + merger_gX_mask)]

    # points for plotting line fit
    x = np.linspace(-1, 1, num=2)

    # fit the hierarchical mergers (any binaries with 2+g) to a line passing through 0,1
    # popt contains the model parameters, pcov the covariances
    # poptHigh, pcovHigh = curve_fit(linefunc, high_gen_mass_ratio, high_gen_chi_eff)

    # plot the 1g-1g population
    fig = plt.figure(figsize=(plotting.set_size(figsize)[0], 2.8))
    ax2 = fig.add_subplot(111)
    # 1g-1g mergers
    ax2.scatter(gen1_chi_eff, gen1_mass_ratio,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    ax2.scatter(gen2_chi_eff, gen_mass_ratio,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax2.scatter(genX_chi_eff, genX_mass_ratio,
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
        poptAll, pcovAll = curve_fit(linefunc, mass_ratio, chi_eff)
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
        ax2.legend(loc='lower left', fontsize=6)
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

    ax1.scatter(np.log10(gen1_orb_a), gen1_chi_p,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g')

    # plot the 2g+ mergers
    ax1.scatter(np.log10(gen2_orb_a), gen2_chi_p,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g')

    # plot the 3g+ mergers
    ax1.scatter(np.log10(genX_orb_a), genX_chi_p,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng')
    
    plt.axvline(np.log10(trap_radius), color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    # plt.title("In-plane effective Spin vs. Merger radius")
    ax1.set(
        ylabel=r'$\chi_{\rm p}$',
        xlabel=r'$\log_{10} (R)$ [$R_g$]',
        ylim=(0, 1),
        axisbelow=True)

    ax1.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax1.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax1.legend()

    plt.savefig(opts.plots_directory + "/r_chi_p.png", format='png')
    plt.close()

    # plt.figure()
    # index = 2
    # mode = 10
    # pareto = (np.random.pareto(index, 1000) + 1) * mode

    # x = np.linspace(1,100)
    # p = index*mode**index / x**(index+1)

    # # count, bins, _ = plt.hist(pareto, 100)
    # plt.plot(x, p)
    # plt.xlim(0,100)
    # plt.show()


    # ========================================
    # Mass 2 vs Mass 1
    # ========================================

    all_time = mergers[:, 14]
    gen1_time = all_time[merger_g1_mask]
    gen2_time = all_time[merger_g2_mask]
    genX_time = all_time[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(gen1_time / 1e6, gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    ax3.scatter(gen2_time / 1e6, gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax3.scatter(genX_time / 1e6, genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    ax3.set(
        xlabel='Time [Myr]',
        ylabel=r'Remnant Mass [$M_\odot$]',
        yscale="log",
        axisbelow=True
    )

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    plt.savefig(opts.plots_directory + '/time_of_merger.png', format='png')
    plt.close()


    # ========================================
    # Number of Mergers vs Mass
    # ========================================

    # Sort Objects into Mass 1 and Mass 2 by generation
    mass_mask_g1 = mergers[merger_g1_mask, 6] > mergers[merger_g1_mask, 7]
    gen1_mass_1 = np.zeros(np.sum(merger_g1_mask))
    gen1_mass_1[mass_mask_g1] = mergers[merger_g1_mask, 6][mass_mask_g1]
    gen1_mass_1[~mass_mask_g1] = mergers[merger_g1_mask, 7][~mass_mask_g1]
    gen1_mass_2 = np.zeros(np.sum(merger_g1_mask))
    gen1_mass_2[~mass_mask_g1] = mergers[merger_g1_mask, 6][~mass_mask_g1]
    gen1_mass_2[mass_mask_g1] = mergers[merger_g1_mask, 7][mass_mask_g1]

    mass_mask_g2 = mergers[merger_g2_mask, 6] > mergers[merger_g2_mask, 7]
    gen2_mass_1 = np.zeros(np.sum(merger_g2_mask))
    gen2_mass_1[mass_mask_g2] = mergers[merger_g2_mask, 6][mass_mask_g2]
    gen2_mass_1[~mass_mask_g2] = mergers[merger_g2_mask, 7][~mass_mask_g2]
    gen2_mass_2 = np.zeros(np.sum(merger_g2_mask))
    gen2_mass_2[~mass_mask_g2] = mergers[merger_g2_mask, 6][~mass_mask_g2]
    gen2_mass_2[mass_mask_g2] = mergers[merger_g2_mask, 7][mass_mask_g2]

    mass_mask_gX = mergers[merger_gX_mask, 6] > mergers[merger_gX_mask, 7]
    genX_mass_1 = np.zeros(np.sum(merger_gX_mask))
    genX_mass_1[mass_mask_gX] = mergers[merger_gX_mask, 6][mass_mask_gX]
    genX_mass_1[~mass_mask_gX] = mergers[merger_gX_mask, 7][~mass_mask_gX]
    genX_mass_2 = np.zeros(np.sum(merger_gX_mask))
    genX_mass_2[~mass_mask_gX] = mergers[merger_gX_mask, 6][~mass_mask_gX]
    genX_mass_2[mass_mask_gX] = mergers[merger_gX_mask, 7][mass_mask_gX]

    # Check that there aren't any zeros remaining.
    assert (gen1_mass_1 > 0).all()
    assert (gen1_mass_2 > 0).all()
    assert (gen2_mass_1 > 0).all()
    assert (gen2_mass_2 > 0).all()
    assert (genX_mass_1 > 0).all()
    assert (genX_mass_2 > 0).all()

    pointsize_m1m2 = 5
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax4 = fig.add_subplot(111)

    # plt.scatter(m1, m2, s=pointsize_m1m2, color='k')
    ax4.scatter(gen1_mass_1, gen1_mass_2,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    ax4.scatter(gen2_mass_1, gen2_mass_2,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax4.scatter(genX_mass_1, genX_mass_2,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    ax4.set(
        xlabel=r'$M_1$ [$M_\odot$]',
        ylabel=r'$M_2$ [$M_\odot$]',
        xscale='log',
        yscale='log',
        axisbelow=(True),
        # aspect=('equal')
    )

    ax4.legend(fontsize=6)

    # plt.grid(True, color='gray', ls='dotted')
    plt.savefig(opts.plots_directory + '/m1m2.png', format='png')
    plt.close()

    # ===============================
    ### kick velocity histogram ###
    # ===============================
    fig = plt.figure(figsize=plotting.set_size(figsize))

    kick_bins = np.logspace(np.log10(mergers[:, 16].min()), np.log10(mergers[:, 16].max()), 50)

    plt.hist(mergers[:, 16], bins = kick_bins, label = 'kick')

    plt.ylabel(r'n')
    plt.xlabel(r'v$_{kick}$ [km/s]')
    #plt.loglog()
    plt.xscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    #plt.ylim(0.4, 325)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/v_kick_distribution.png", format='png')
    plt.close()

    # ===============================
    ### a_bin vs. kick velocity with kick velocity histogram ###
    # ===============================

    all_kick = mergers[:, 16]
    gen1_vkick = all_kick[merger_g1_mask]
    gen2_vkick = all_kick[merger_g2_mask]
    genX_vkick = all_kick[merger_gX_mask]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 
    #plt.subplots_adjust(wspace=0.0, hspace=0.6)

    axs[0].scatter(gen1_orb_a, gen1_vkick,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    axs[0].scatter(gen2_orb_a, gen2_vkick,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    axs[0].scatter(genX_orb_a, genX_vkick,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    trap_radius = 700
    axs[0].axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    axs[0].set_ylabel(r'$v_{kick}$ [km/s]')
    axs[0].set_xlabel(r'Radius [$R_g$]')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        axs[0].legend(fontsize=6)
    elif figsize == 'apj_page':
        axs[0].legend()

    kick_bins = np.logspace(np.log10(mergers[:, 16].min()), np.log10(mergers[:, 16].max()), 50)

    axs[1].hist(mergers[:, 16], bins = kick_bins, orientation = 'horizontal')
    axs[1].set_yscale('log')
    mean_kick = np.mean(mergers[:, 16])
    axs[1].axhline(mean_kick, color = 'red', linewidth = 1, label = r'$\langle v_{kick}\rangle $ =' + f"{mean_kick:.2f}")
    axs[1].yaxis.tick_right()
    axs[1].set_xlabel(r'n')
    #fig.tight_layout()

    if figsize == 'apj_col':
        axs[1].legend(fontsize=6, loc = 'lower right')
    elif figsize == 'apj_page':
        axs[1].legend()
    #plt.tight_layout()
    plt.savefig(opts.plots_directory + '/v_kick_vs_radius.png', format='png')
    plt.close()

    # ===============================
    ### kick velocity vs. chi_eff ###
    # ===============================

    all_chi_eff = mergers[:, 3]
    gen1_chi_eff = all_chi_eff[merger_g1_mask]
    gen2_chi_eff  = all_chi_eff[merger_g2_mask]
    genX_chi_eff  = all_chi_eff[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = fig.add_subplot(111)
    ax3.scatter(gen1_chi_eff, gen1_vkick,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    ax3.scatter(gen2_chi_eff, gen2_vkick,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    ax3.scatter(genX_chi_eff, genX_vkick,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    mean_chieff_gen_1 = np.mean(gen1_vkick)
    mean_chieff_gen_2 = np.mean(gen2_vkick)
    mean_chieff_gen_X = np.mean(genX_vkick)
    # ax3.axhline(mean_chieff_gen_1, color='gold', linestyle='--', zorder=0,
    #             label= f"{mean_chieff_gen_1:.2f}")
    # ax3.axhline(mean_chieff_gen_2, color='purple', linestyle='--', zorder=0,
    #             label= f"{mean_chieff_gen_2:.2f}")
    # ax3.axhline(mean_chieff_gen_X, color='red', linestyle='--', zorder=0,
    #             label= f"{mean_chieff_gen_X:.2f}")
    
    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.xlabel(r'$\chi_{eff}$')
    plt.ylabel(r'$v_{kick}$ [km/s]')
    #plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
         ax3.legend(fontsize=4)
    # elif figsize == 'apj_page':
    #     ax3.legend()

    plt.savefig(opts.plots_directory + '/v_kick_vs_chi_eff.png', format='png')
    plt.close()


    # ========================================
    # LVK and LISA Strain vs Freq
    # ========================================

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

    lvk_g1_mask, lvk_g2_mask, lvk_gX_mask = make_gen_masks(lvk, 7, 8)

    lvk_g1 = lvk[lvk_g1_mask]
    lvk_g2 = lvk[lvk_g2_mask]
    lvk_gX = lvk[lvk_gX_mask]

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

    strain_per_freq_lvk_g1 = lvk_g1[:, 5] * (1 / lvk_g1[:, 6]) / timestep
    strain_per_freq_lvk_g2 = lvk_g2[:, 5] * (1 / lvk_g2[:, 6]) / timestep
    strain_per_freq_lvk_gX = lvk_gX[:, 5] * (1 / lvk_gX[:, 6]) / timestep

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

    svf_ax.scatter(lvk_g1[:, 6], strain_per_freq_lvk_g1,
                   s=0.4 * styles.markersize_gen1,
                   marker=styles.marker_gen1,
                   edgecolor=styles.color_gen1,
                   facecolor='none',
                   alpha=styles.markeralpha_gen1,
                   label='1g-1g'
                   )

    svf_ax.scatter(lvk_g2[:, 6], strain_per_freq_lvk_g2,
                   s=0.4 * styles.markersize_gen2,
                   marker=styles.marker_gen2,
                   edgecolor=styles.color_gen2,
                   facecolor='none',
                   alpha=styles.markeralpha_gen2,
                   label='2g-1g or 2g-2g'
                   )

    svf_ax.scatter(lvk_gX[:, 6], strain_per_freq_lvk_gX,
                   s=0.4 * styles.markersize_genX,
                   marker=styles.marker_genX,
                   edgecolor=styles.color_genX,
                   facecolor='none',
                   alpha=styles.markeralpha_genX,
                   label=r'$\geq$3g-Ng'
                   )

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

    plt.savefig(opts.plots_directory + './gw_strain.png', format='png')
    plt.close()


######## Execution ########
if __name__ == "__main__":
    main()
