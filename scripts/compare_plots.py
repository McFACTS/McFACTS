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
    parser.add_argument("--fname-surmergers",
                        default="sur_output_mergers_population.dat",
                        type=str, help="sur_output file")
    parser.add_argument("--fname-nosurmergers",
                        default="nosur_output_mergers_population.dat",
                        type=str, help="nosur_output file")
    opts = parser.parse_args()
    assert os.path.isfile(opts.fname_mergers)
    assert os.path.isfile(opts.fname_emris)
    assert os.path.isfile(opts.fname_lvk)
    assert os.path.isfile(opts.fname_surmergers)
    assert os.path.isfile(opts.fname_nosurmergers)
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

    sur_mergers = np.loadtxt(opts.fname_surmergers, skiprows=2)
    mergers = np.loadtxt(opts.fname_mergers, skiprows=2)
    sur_emris = np.loadtxt(opts.fname_emris, skiprows=2)
    sur_lvk = np.loadtxt(opts.fname_lvk, skiprows=2)

    nosur_mergers = np.loadtxt(opts.fname_nosurmergers, skiprows=2)
    nosur_emris = np.loadtxt(opts.fname_emris, skiprows=2)
    nosur_lvk = np.loadtxt(opts.fname_lvk, skiprows=2)
    
    # Exclude all rows with NaNs or zeros in the final mass column
    sur_merger_nan_mask = (np.isfinite(sur_mergers[:, 2])) & (sur_mergers[:, 2] != 0)
    sur_mergers = sur_mergers[sur_merger_nan_mask]
    
    # Exclude all rows with NaNs or zeros in the final mass column
    nosur_merger_nan_mask = (np.isfinite(nosur_mergers[:, 2])) & (nosur_mergers[:, 2] != 0)
    nosur_mergers = nosur_mergers[nosur_merger_nan_mask]

    sur_merger_g1_mask, sur_merger_g2_mask, sur_merger_gX_mask = make_gen_masks(sur_mergers, 12, 13)
    nosur_merger_g1_mask, nosur_merger_g2_mask, nosur_merger_gX_mask = make_gen_masks(nosur_mergers, 12, 13)


    # Ensure no union between sets
    assert all(sur_merger_g1_mask & sur_merger_g2_mask) == 0
    assert all(sur_merger_g1_mask & sur_merger_gX_mask) == 0
    assert all(sur_merger_g2_mask & sur_merger_gX_mask) == 0
    
    assert all(nosur_merger_g1_mask & nosur_merger_g2_mask) == 0
    assert all(nosur_merger_g1_mask & nosur_merger_gX_mask) == 0
    assert all(nosur_merger_g2_mask & nosur_merger_gX_mask) == 0

    # Ensure no elements are missed
    assert all(sur_merger_g1_mask | sur_merger_g2_mask | sur_merger_gX_mask) == 1
    
    assert all(nosur_merger_g1_mask | nosur_merger_g2_mask | nosur_merger_gX_mask) == 1
    
    print("success")    
    

    # ========================================
    # SUR - Number of Mergers vs Mass
    # ========================================

    # Plot intial and final mass distributions
    #figsize=plotting.set_size(figsize)
    fig, ax = plt.subplots(2, 1, figsize=plotting.set_size(figsize), sharex=True)
    
    sur = ax[1]
    nosur = ax[0]
    
    counts, bins = np.histogram(sur_mergers[:, 2])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(sur_mergers[:, 2].min()), int(sur_mergers[:, 2].max()) + 2, 1)

    sur_hist_data = [sur_mergers[:, 2][sur_merger_g1_mask], sur_mergers[:, 2][sur_merger_g2_mask], sur_mergers[:, 2][sur_merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    sur.hist(sur_hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    sur.set_ylabel('Number of Mergers - (sur)', fontsize=5, wrap=True)
    #sur.set_xlabel(r'Remnant Mass [$M_\odot$]')
    sur.set_xscale('log')
    # plt.ylim(-5,max(counts))
    #svf_ax = sur.gca()
    #svf_ax.set_axisbelow(True)
    #svf_ax.tick_params(axis='x', direction='out', which='both')
    #plt.grid(True, color='gray', ls='dashed')
    #svf_ax.yaxis.grid(True, color='gray', ls='dashed')

    #sur.set_xticks(np.geomspace(int(sur_mergers[:, 2].min()), int( sur_mergers[:, 2].max()), 5).astype(int))
    #plt.xticks(np.geomspace(20, 200, 5).astype(int))

    #svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    #svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        sur.legend(fontsize=5)
    elif figsize == 'apj_page':
        sur.legend()

    #plt.savefig(opts.plots_directory + r"/merger_remnant_mass.png", format='png')

    #sur.close()
    
    # ========================================
    # NO SUR - Number of Mergers vs Mass
    # ========================================

    # Plot intial and final mass distributions
    counts, bins = np.histogram(nosur_mergers[:, 2])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(nosur_mergers[:, 2].min()), int(nosur_mergers[:, 2].max()) + 2, 1)

    nosur_hist_data = [nosur_mergers[:, 2][nosur_merger_g1_mask], nosur_mergers[:, 2][nosur_merger_g2_mask], nosur_mergers[:, 2][nosur_merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    nosur.hist(nosur_hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    nosur.set_ylabel('Number of Mergers - (nosur)', fontsize=5, wrap=True)
    nosur.set_xlabel(r'Remnant Mass [$M_\odot$]')
    nosur.set_xscale('log')

    # plt.ylim(-5,max(counts))
    #svf_ax = nosur.gca()
    #svf_ax.set_axisbelow(True)
    #svf_ax.tick_params(axis='x', direction='out', which='both')
    #plt.grid(True, color='gray', ls='dashed')
    #svf_ax.yaxis.grid(True, color='gray', ls='dashed')

    #nosur.set_xticks(np.geomspace(int(nosur_mergers[:, 2].min()), int( nosur_mergers[:, 2].max()), 5).astype(int))
    #plt.xticks(np.geomspace(20, 200, 5).astype(int))

    #svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    #svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        nosur.legend(fontsize=5)
    elif figsize == 'apj_page':
        nosur.legend()

    #nosur.savefig(opts.plots_directory + r"/merger_remnant_mass.png", format='png')
    plt.savefig(opts.plots_directory + r"/merger_remnant_mass.png", format='png')
    #plt.show()


    # ========================================
    # SUR - Merger Mass vs Radius
    # ========================================

    # Read the log file
    log_data = ReadLog(opts.fname_log)

    # Retrieve the migration trap radius used in run
    trap_radius = log_data["disk_radius_trap"]

    # plt.title('Migration Trap influence')
    for i in range(len(sur_mergers[:, 1])):
        if sur_mergers[i, 1] < 10.0:
            sur_mergers[i, 1] = 10.0

    # Separate generational subpopulations
    sur_gen1_orb_a = sur_mergers[:, 1][sur_merger_g1_mask]
    sur_gen2_orb_a = sur_mergers[:, 1][sur_merger_g2_mask]
    sur_genX_orb_a = sur_mergers[:, 1][sur_merger_gX_mask]
    sur_gen1_mass = sur_mergers[:, 2][sur_merger_g1_mask]
    sur_gen2_mass = sur_mergers[:, 2][sur_merger_g2_mask]
    sur_genX_mass = sur_mergers[:, 2][sur_merger_gX_mask]

    fig, ax = plt.subplots(1, 2, figsize=(5,2), sharey=True, layout='constrained', gridspec_kw={'wspace':0, 'hspace':0})
    sur = ax[1]
    nosur = ax[0]
    
    sur.scatter(sur_gen1_orb_a, sur_gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    sur.scatter(sur_gen2_orb_a, sur_gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    sur.scatter(sur_genX_orb_a, sur_genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    sur.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    #sur.set_ylabel(r'Remnant Mass [$M_\odot$]')
    sur.set_xlabel(r'Radius$_{sur}$ [$R_g$]')
    sur.set_xscale('log')
    sur.set_yscale('log')

    if figsize == 'apj_col':
        sur.legend(fontsize=5)
    elif figsize == 'apj_page':
        sur.legend()

    sur.set_ylim(18, 1000)

    #svf_ax = sur.gca()
    #svf_ax.set_axisbelow(True)
    sur.grid(True, color='gray', ls='dashed')
    #sur.savefig(opts.plots_directory + "/merger_mass_v_radius.png", format='png')
    #sur.close()


    # ========================================
    # NOSUR - Merger Mass vs Radius
    # ========================================

    # Read the log file
    #log_data = ReadLog(opts.fname_log)

    # Retrieve the migration trap radius used in run
    #trap_radius = log_data["disk_radius_trap"]

    # plt.title('Migration Trap influence')
    for i in range(len(nosur_mergers[:, 1])):
        if nosur_mergers[i, 1] < 10.0:
            nosur_mergers[i, 1] = 10.0

    # Separate generational subpopulations
    nosur_gen1_orb_a = nosur_mergers[:, 1][nosur_merger_g1_mask]
    nosur_gen2_orb_a = nosur_mergers[:, 1][nosur_merger_g2_mask]
    nosur_genX_orb_a = nosur_mergers[:, 1][nosur_merger_gX_mask]
    nosur_gen1_mass = nosur_mergers[:, 2][nosur_merger_g1_mask]
    nosur_gen2_mass = nosur_mergers[:, 2][nosur_merger_g2_mask]
    nosur_genX_mass = nosur_mergers[:, 2][nosur_merger_gX_mask]

    nosur.scatter(nosur_gen1_orb_a, nosur_gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    nosur.scatter(nosur_gen2_orb_a, nosur_gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    nosur.scatter(nosur_genX_orb_a, nosur_genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    nosur.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    nosur.set_ylabel(r'Remnant Mass [$M_\odot$]')
    nosur.set_xlabel(r'Radius$_{nosur}$ [$R_g$]')
    nosur.set_xscale('log')
    nosur.set_yscale('log')

    if figsize == 'apj_col':
        nosur.legend(fontsize=5)
    elif figsize == 'apj_page':
        nosur.legend()

    nosur.set_ylim(18, 1000)

    #svf_ax = nosur.gca()
    #svf_ax.set_axisbelow(True)
    nosur.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/merger_mass_v_radius.png", format='png')
    #plt.show()


    # ========================================
    # SUR - q vs Chi Effective
    # ========================================

    # retrieve component masses and mass ratio
    m1 = np.zeros(sur_mergers.shape[0])
    m2 = np.zeros(sur_mergers.shape[0])
    mass_ratio = np.zeros(sur_mergers.shape[0])
    for i in range(sur_mergers.shape[0]):
        if sur_mergers[i, 6] < sur_mergers[i, 7]:
            m1[i] = sur_mergers[i, 7]
            m2[i] = sur_mergers[i, 6]
            mass_ratio[i] = sur_mergers[i, 6] / sur_mergers[i, 7]
        else:
            mass_ratio[i] = sur_mergers[i, 7] / sur_mergers[i, 6]
            m1[i] = sur_mergers[i, 6]
            m2[i] = sur_mergers[i, 7]
        #mass_ratio[i] = 1 / mass_ratio[i]

    # (q,X_eff) Figure details here:
    # Want to highlight higher generation mergers on this plot
    sur_chi_eff = sur_mergers[:, 3]

    # Get 1g-1g population
    sur_gen1_chi_eff = sur_chi_eff[sur_merger_g1_mask]
    sur_gen1_mass_ratio = mass_ratio[sur_merger_g1_mask]
    # 2g-1g and 2g-2g population
    sur_gen2_chi_eff = sur_chi_eff[sur_merger_g2_mask]
    sur_gen_mass_ratio = mass_ratio[sur_merger_g2_mask]
    # >=3g-Ng population (i.e., N=1,2,3,4,...)
    sur_genX_chi_eff = sur_chi_eff[sur_merger_gX_mask]
    sur_genX_mass_ratio = mass_ratio[sur_merger_gX_mask]
    # all 2+g mergers; H = hierarchical
    sur_genH_chi_eff = sur_chi_eff[(sur_merger_g2_mask + sur_merger_gX_mask)]
    sur_genH_mass_ratio = mass_ratio[(sur_merger_g2_mask + sur_merger_gX_mask)]

    # points for plotting line fit
    x = np.linspace(-1, 1, num=2)

    # fit the hierarchical mergers (any binaries with 2+g) to a line passing through 0,1
    # popt contains the model parameters, pcov the covariances
    # poptHigh, pcovHigh = curve_fit(linefunc, high_gen_mass_ratio, high_gen_chi_eff)

    # plot the 1g-1g population
    fig, ax = plt.subplots(1, 2, figsize=(5,3), sharey=True, gridspec_kw={'wspace':0, 'hspace':0})
    #ax2 = fig.add_subplot(111)
    sur = ax[1]
    nosur = ax[0]
    
    # 1g-1g mergers
    sur.scatter(sur_gen1_chi_eff, sur_gen1_mass_ratio,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    sur.scatter(sur_gen2_chi_eff, sur_gen_mass_ratio,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    sur.scatter(sur_genX_chi_eff, sur_genX_mass_ratio,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    if len(sur_genH_chi_eff) > 0:
        poptHier, pcovHier = curve_fit(linefunc, sur_genH_mass_ratio, sur_genH_chi_eff)
        errHier = np.sqrt(np.diag(pcovHier))[0]
        # plot the line fitting the hierarchical mergers
        sur.plot(linefunc(x, *poptHier), x,
                 ls='dashed',
                 lw=1,
                 color='gray',
                 zorder=3,
                 label=r'$d\chi/dq(\geq$2g)=' +
                       f'{poptHier[0]:.2f}' +
                       r'$\pm$' + f'{errHier:.2f}'
                 )
        #         #  alpha=linealpha,

    if len(sur_chi_eff) > 0:
        poptAll, pcovAll = curve_fit(linefunc, mass_ratio, sur_chi_eff)
        errAll = np.sqrt(np.diag(pcovAll))[0]
        sur.plot(linefunc(x, *poptAll), x,
                 ls='solid',
                 lw=1,
                 color='black',
                 zorder=3,
                 label=r'$d\chi/dq$(all)=' +
                       f'{poptAll[0]:.2f}' +
                       r'$\pm$' + f'{errAll:.2f}'
                 )
        #  alpha=linealpha,

    #sur.set_ylabel(r'$q = M_2 / M_1$')  # ($M_1 > M_2$)')
    sur.set_xlabel(r'$\chi_{\rm eff}^{sur}$')
    sur.set_ylim(0, 1)
    sur.set_xlim(-1, 1)
    sur.set_axisbelow=True

    if figsize == 'apj_col':
        sur.legend(loc='lower left', fontsize=4)
    elif figsize == 'apj_page':
        sur.legend(loc='lower left')

    sur.grid('on', color='gray', ls='dotted')
    #plt.savefig(opts.plots_directory + "/q_chi_eff.png", format='png')  # ,dpi=600)
    
    # ========================================
    # NOSUR - q vs Chi Effective
    # ========================================

    # retrieve component masses and mass ratio
    m1 = np.zeros(nosur_mergers.shape[0])
    m2 = np.zeros(nosur_mergers.shape[0])
    mass_ratio = np.zeros(nosur_mergers.shape[0])
    for i in range(nosur_mergers.shape[0]):
        if nosur_mergers[i, 6] < nosur_mergers[i, 7]:
            m1[i] = nosur_mergers[i, 7]
            m2[i] = nosur_mergers[i, 6]
            mass_ratio[i] = nosur_mergers[i, 6] / nosur_mergers[i, 7]
        else:
            mass_ratio[i] = nosur_mergers[i, 7] / nosur_mergers[i, 6]
            m1[i] = nosur_mergers[i, 6]
            m2[i] = nosur_mergers[i, 7]

    # (q,X_eff) Figure details here:
    # Want to highlight higher generation mergers on this plot
    nosur_chi_eff = nosur_mergers[:, 3]

    # Get 1g-1g population
    nosur_gen1_chi_eff = nosur_chi_eff[nosur_merger_g1_mask]
    nosur_gen1_mass_ratio = mass_ratio[nosur_merger_g1_mask]
    # 2g-1g and 2g-2g population
    nosur_gen2_chi_eff = nosur_chi_eff[nosur_merger_g2_mask]
    nosur_gen_mass_ratio = mass_ratio[nosur_merger_g2_mask]
    # >=3g-Ng population (i.e., N=1,2,3,4,...)
    nosur_genX_chi_eff = nosur_chi_eff[nosur_merger_gX_mask]
    nosur_genX_mass_ratio = mass_ratio[nosur_merger_gX_mask]
    # all 2+g mergers; H = hierarchical
    nosur_genH_chi_eff = nosur_chi_eff[(nosur_merger_g2_mask + nosur_merger_gX_mask)]
    nosur_genH_mass_ratio = mass_ratio[(nosur_merger_g2_mask + nosur_merger_gX_mask)]

    # points for plotting line fit
    x = np.linspace(-1, 1, num=2)

    # fit the hierarchical mergers (any binaries with 2+g) to a line passing through 0,1
    # popt contains the model parameters, pcov the covariances
    # poptHigh, pcovHigh = curve_fit(linefunc, high_gen_mass_ratio, high_gen_chi_eff)

    # plot the 1g-1g population
    #fig = plt.figure(figsize=(plotting.set_size(figsize)[0], 2.8))
    #ax2 = fig.add_subplot(111)
    # 1g-1g mergers
    nosur.scatter(nosur_gen1_chi_eff, nosur_gen1_mass_ratio,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    nosur.scatter(nosur_gen2_chi_eff, nosur_gen_mass_ratio,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    nosur.scatter(nosur_genX_chi_eff, nosur_genX_mass_ratio,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    if len(nosur_genH_chi_eff) > 0:
        poptHier, pcovHier = curve_fit(linefunc, nosur_genH_mass_ratio, nosur_genH_chi_eff)
        errHier = np.sqrt(np.diag(pcovHier))[0]
        # plot the line fitting the hierarchical mergers
        nosur.plot(linefunc(x, *poptHier), x,
                 ls='dashed',
                 lw=1,
                 color='gray',
                 zorder=3,
                 label=r'$d\chi/dq(\geq$2g)=' +
                       f'{poptHier[0]:.2f}' +
                       r'$\pm$' + f'{errHier:.2f}'
                 )
        #         #  alpha=linealpha,

    if len(nosur_chi_eff) > 0:
        poptAll, pcovAll = curve_fit(linefunc, mass_ratio, nosur_chi_eff)
        errAll = np.sqrt(np.diag(pcovAll))[0]
        nosur.plot(linefunc(x, *poptAll), x,
                 ls='solid',
                 lw=1,
                 color='black',
                 zorder=3,
                 label=r'$d\chi/dq$(all)=' +
                       f'{poptAll[0]:.2f}' +
                       r'$\pm$' + f'{errAll:.2f}'
                 )
        #  alpha=linealpha,

    nosur.set_ylabel(r'$q = M_2 / M_1$')  # ($M_1 > M_2$)')
    nosur.set_xlabel(r'$\chi_{\rm eff}^{nosur}$')
    nosur.set_ylim(0, 1)
    nosur.set_xlim(-1, 1)
    nosur.set_axisbelow=True

    if figsize == 'apj_col':
        nosur.legend(loc='best', fontsize=4)
    elif figsize == 'apj_page':
        nosur.legend(loc='best')

    nosur.grid('on', color='gray', ls='dotted')
    plt.savefig(opts.plots_directory + "/q_chi_eff.png", format='png')  # ,dpi=600)
    #plt.show()


    # ========================================
    # NOSUR - q vs Chi Effective            STAND ALONE PLOT
    # ========================================

    # retrieve component masses and mass ratio
    m1 = np.zeros(nosur_mergers.shape[0])
    m2 = np.zeros(nosur_mergers.shape[0])
    mass_ratio = np.zeros(nosur_mergers.shape[0])
    for i in range(nosur_mergers.shape[0]):
        if nosur_mergers[i, 6] < nosur_mergers[i, 7]:
            m1[i] = nosur_mergers[i, 7]
            m2[i] = nosur_mergers[i, 6]
            mass_ratio[i] = nosur_mergers[i, 6] / nosur_mergers[i, 7]
        else:
            mass_ratio[i] = nosur_mergers[i, 7] / nosur_mergers[i, 6]
            m1[i] = nosur_mergers[i, 6]
            m2[i] = nosur_mergers[i, 7]

    # (q,X_eff) Figure details here:
    # Want to highlight higher generation mergers on this plot
    nosur_chi_eff = nosur_mergers[:, 3]

    # Get 1g-1g population
    nosur_gen1_chi_eff = nosur_chi_eff[nosur_merger_g1_mask]
    nosur_gen1_mass_ratio = mass_ratio[nosur_merger_g1_mask]
    # 2g-1g and 2g-2g population
    nosur_gen2_chi_eff = nosur_chi_eff[nosur_merger_g2_mask]
    nosur_gen_mass_ratio = mass_ratio[nosur_merger_g2_mask]
    # >=3g-Ng population (i.e., N=1,2,3,4,...)
    nosur_genX_chi_eff = nosur_chi_eff[nosur_merger_gX_mask]
    nosur_genX_mass_ratio = mass_ratio[nosur_merger_gX_mask]
    # all 2+g mergers; H = hierarchical
    nosur_genH_chi_eff = nosur_chi_eff[(nosur_merger_g2_mask + nosur_merger_gX_mask)]
    nosur_genH_mass_ratio = mass_ratio[(nosur_merger_g2_mask + nosur_merger_gX_mask)]

    # points for plotting line fit
    x = np.linspace(-1, 1, num=2)

    # fit the hierarchical mergers (any binaries with 2+g) to a line passing through 0,1
    # popt contains the model parameters, pcov the covariances
    # poptHigh, pcovHigh = curve_fit(linefunc, high_gen_mass_ratio, high_gen_chi_eff)

    fig, ax = plt.subplots(1, 1, figsize=(4,3), sharey=True, gridspec_kw={'wspace':0, 'hspace':0})
    #ax2 = fig.add_subplot(111)
    #sur = ax[1]
    nosur = ax
    
    # 1g-1g mergers
    nosur.scatter(nosur_gen1_chi_eff, nosur_gen1_mass_ratio,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    nosur.scatter(nosur_gen2_chi_eff, nosur_gen_mass_ratio,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    nosur.scatter(nosur_genX_chi_eff, nosur_genX_mass_ratio,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    if len(nosur_genH_chi_eff) > 0:
        poptHier, pcovHier = curve_fit(linefunc, nosur_genH_mass_ratio, nosur_genH_chi_eff)
        errHier = np.sqrt(np.diag(pcovHier))[0]
        # plot the line fitting the hierarchical mergers
        nosur.plot(linefunc(x, *poptHier), x,
                 ls='dashed',
                 lw=1,
                 color='gray',
                 zorder=3,
                 label=r'$d\chi/dq(\geq$2g)=' +
                       f'{poptHier[0]:.2f}' +
                       r'$\pm$' + f'{errHier:.2f}'
                 )
        #         #  alpha=linealpha,

    if len(nosur_chi_eff) > 0:
        poptAll, pcovAll = curve_fit(linefunc, mass_ratio, nosur_chi_eff)
        errAll = np.sqrt(np.diag(pcovAll))[0]
        nosur.plot(linefunc(x, *poptAll), x,
                 ls='solid',
                 lw=1,
                 color='black',
                 zorder=3,
                 label=r'$d\chi/dq$(all)=' +
                       f'{poptAll[0]:.2f}' +
                       r'$\pm$' + f'{errAll:.2f}'
                 )
        #  alpha=linealpha,

    nosur.set_ylabel(r'$q = M_2 / M_1$')  # ($M_1 > M_2$)')
    nosur.set_xlabel(r'$\chi_{\rm eff}^{nosur}$')
    nosur.set_ylim(0, 1)
    nosur.set_xlim(-1, 1)
    nosur.set_axisbelow=True

    if figsize == 'apj_col':
        nosur.legend(loc='best', fontsize=4)
    elif figsize == 'apj_page':
        nosur.legend(loc='best')

    nosur.grid('on', color='gray', ls='dotted')
    plt.savefig(opts.plots_directory + "/q_chi_eff_nosur.png", format='png')  # ,dpi=600)
    #plt.show()


    # ========================================
    # SUR - Disk Radius vs Chi_p
    # ========================================

    # Can break out higher mass Chi_p events as test/illustration.
    # Set up default arrays for high mass BBH (>40Msun say) to overplot vs chi_p.
    sur_chi_p = sur_mergers[:, 15]
    sur_gen1_chi_p = sur_chi_p[sur_merger_g1_mask]
    sur_gen2_chi_p = sur_chi_p[sur_merger_g2_mask]
    sur_genX_chi_p = sur_chi_p[sur_merger_gX_mask]

    fig, ax = plt.subplots(1, 2, figsize=plotting.set_size(figsize), sharey=True)
    #ax1 = fig.add_subplot(111)
    
    sur = ax[1]
    nosur = ax[0]

    sur.scatter(np.log10(sur_gen1_orb_a), sur_gen1_chi_p,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g')

    # plot the 2g+ mergers
    sur.scatter(np.log10(sur_gen2_orb_a), sur_gen2_chi_p,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g')

    # plot the 3g+ mergers
    sur.scatter(np.log10(sur_genX_orb_a), sur_genX_chi_p,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng')
    
    sur.axvline(np.log10(trap_radius), color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    # plt.title("In-plane effective Spin vs. Merger radius")
    sur.set(
        ylabel=r'$\chi_{\rm p}$',
        xlabel=r'$\log_{10} (R)$ [$R_g$] - (sur)',
        ylim=(0, 1),
        axisbelow=True)

    sur.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        sur.legend(fontsize=5)
    elif figsize == 'apj_page':
        sur.legend()

    #plt.savefig(opts.plots_directory + "/r_chi_p.png", format='png')
    #plt.close()

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
    # NOSUR - Disk Radius vs Chi_p
    # ========================================

    # Can break out higher mass Chi_p events as test/illustration.
    # Set up default arrays for high mass BBH (>40Msun say) to overplot vs chi_p.
    nosur_chi_p = nosur_mergers[:, 15]
    nosur_gen1_chi_p = nosur_chi_p[nosur_merger_g1_mask]
    nosur_gen2_chi_p = nosur_chi_p[nosur_merger_g2_mask]
    nosur_genX_chi_p = nosur_chi_p[nosur_merger_gX_mask]

    #fig = plt.figure(figsize=plotting.set_size(figsize))
    #ax1 = fig.add_subplot(111)

    nosur.scatter(np.log10(nosur_gen1_orb_a), nosur_gen1_chi_p,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g')

    # plot the 2g+ mergers
    nosur.scatter(np.log10(nosur_gen2_orb_a), nosur_gen2_chi_p,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g')

    # plot the 3g+ mergers
    nosur.scatter(np.log10(nosur_genX_orb_a), nosur_genX_chi_p,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng')
    
    nosur.axvline(np.log10(trap_radius), color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    # plt.title("In-plane effective Spin vs. Merger radius")
    nosur.set(
        #ylabel=r'$\chi_{\rm p}$',
        xlabel=r'$\log_{10} (R)$ [$R_g$] - (nosur)',
        ylim=(0, 1),
        axisbelow=True)

    nosur.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        nosur.legend(fontsize=5)
    elif figsize == 'apj_page':
        nosur.legend()

    plt.savefig(opts.plots_directory + "/r_chi_p.png", format='png')
    #plt.show()

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
    # SUR - Time of Merger vs Remnant Mass
    # ========================================

    sur_all_time = sur_mergers[:, 14]
    sur_gen1_time = sur_all_time[sur_merger_g1_mask]
    sur_gen2_time = sur_all_time[sur_merger_g2_mask]
    sur_genX_time = sur_all_time[sur_merger_gX_mask]

    fig, ax = plt.subplots(1, 2, figsize=plotting.set_size(figsize))
    #ax3 = fig.add_subplot(111)
    
    sur = ax[1]
    nosur = ax[0]

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    sur.scatter(sur_gen1_time / 1e6, sur_gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    sur.scatter(sur_gen2_time / 1e6, sur_gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    sur.scatter(sur_genX_time / 1e6, sur_genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    sur.set(
        xlabel='Time [Myr] - (sur)',
        ylabel=r'Remnant Mass [$M_\odot$]',
        yscale="log",
        axisbelow=True
    )

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        sur.legend(fontsize=6)
    elif figsize == 'apj_page':
        sur.legend()

    #plt.savefig(opts.plots_directory + '/time_of_merger.png', format='png')
    #plt.close()
    
    # ========================================
    # NOSUR - Time of Merger vs Remnant Mass
    # ========================================

    nosur_all_time = nosur_mergers[:, 14]
    nosur_gen1_time = nosur_all_time[nosur_merger_g1_mask]
    nosur_gen2_time = nosur_all_time[nosur_merger_g2_mask]
    nosur_genX_time = nosur_all_time[nosur_merger_gX_mask]

    #fig = plt.figure(figsize=plotting.set_size(figsize))
    #ax3 = fig.add_subplot(111)

    # plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    nosur.scatter(nosur_gen1_time / 1e6, nosur_gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    nosur.scatter(nosur_gen2_time / 1e6, nosur_gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    nosur.scatter(nosur_genX_time / 1e6, nosur_genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    nosur.set(
        xlabel='Time [Myr] - (nosur)',
        ylabel=r'Remnant Mass [$M_\odot$]',
        yscale="log",
        axisbelow=True
    )

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        nosur.legend(fontsize=6)
    elif figsize == 'apj_page':
        nosur.legend()

    plt.savefig(opts.plots_directory + '/time_of_merger.png', format='png')
    #plt.show()



    # ========================================
    # SUR - Mass 1 vs Mass 2
    # ========================================

    # Sort Objects into Mass 1 and Mass 2 by generation
    sur_mass_mask_g1 = sur_mergers[sur_merger_g1_mask, 6] > sur_mergers[sur_merger_g1_mask, 7]
    sur_gen1_mass_1 = np.zeros(np.sum(sur_merger_g1_mask))
    sur_gen1_mass_1[sur_mass_mask_g1] = sur_mergers[sur_merger_g1_mask, 6][sur_mass_mask_g1]
    sur_gen1_mass_1[~sur_mass_mask_g1] = sur_mergers[sur_merger_g1_mask, 7][~sur_mass_mask_g1]
    sur_gen1_mass_2 = np.zeros(np.sum(sur_merger_g1_mask))
    sur_gen1_mass_2[~sur_mass_mask_g1] = sur_mergers[sur_merger_g1_mask, 6][~sur_mass_mask_g1]
    sur_gen1_mass_2[sur_mass_mask_g1] = sur_mergers[sur_merger_g1_mask, 7][sur_mass_mask_g1]

    sur_mass_mask_g2 = sur_mergers[sur_merger_g2_mask, 6] > sur_mergers[sur_merger_g2_mask, 7]
    sur_gen2_mass_1 = np.zeros(np.sum(sur_merger_g2_mask))
    sur_gen2_mass_1[sur_mass_mask_g2] = sur_mergers[sur_merger_g2_mask, 6][sur_mass_mask_g2]
    sur_gen2_mass_1[~sur_mass_mask_g2] = sur_mergers[sur_merger_g2_mask, 7][~sur_mass_mask_g2]
    sur_gen2_mass_2 = np.zeros(np.sum(sur_merger_g2_mask))
    sur_gen2_mass_2[~sur_mass_mask_g2] = sur_mergers[sur_merger_g2_mask, 6][~sur_mass_mask_g2]
    sur_gen2_mass_2[sur_mass_mask_g2] = sur_mergers[sur_merger_g2_mask, 7][sur_mass_mask_g2]

    sur_mass_mask_gX = sur_mergers[sur_merger_gX_mask, 6] > sur_mergers[sur_merger_gX_mask, 7]
    sur_genX_mass_1 = np.zeros(np.sum(sur_merger_gX_mask))
    sur_genX_mass_1[sur_mass_mask_gX] = sur_mergers[sur_merger_gX_mask, 6][sur_mass_mask_gX]
    sur_genX_mass_1[~sur_mass_mask_gX] = sur_mergers[sur_merger_gX_mask, 7][~sur_mass_mask_gX]
    sur_genX_mass_2 = np.zeros(np.sum(sur_merger_gX_mask))
    sur_genX_mass_2[~sur_mass_mask_gX] = sur_mergers[sur_merger_gX_mask, 6][~sur_mass_mask_gX]
    sur_genX_mass_2[sur_mass_mask_gX] = sur_mergers[sur_merger_gX_mask, 7][sur_mass_mask_gX]

    # Check that there aren't any zeros remaining.
    assert (sur_gen1_mass_1 > 0).all()
    assert (sur_gen1_mass_2 > 0).all()
    assert (sur_gen2_mass_1 > 0).all()
    assert (sur_gen2_mass_2 > 0).all()
    assert (sur_genX_mass_1 > 0).all()
    assert (sur_genX_mass_2 > 0).all()

    pointsize_m1m2 = 5
    fig, ax = plt.subplots(1, 2, figsize=(5,3), sharey=True)
    #ax4 = fig.add_subplot(111)
    
    sur = ax[1]
    nosur = ax[0]

    # plt.scatter(m1, m2, s=pointsize_m1m2, color='k')
    sur.scatter(sur_gen1_mass_1, sur_gen1_mass_2,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    sur.scatter(sur_gen2_mass_1, sur_gen2_mass_2,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    sur.scatter(sur_genX_mass_1, sur_genX_mass_2,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    sur.set(
        xlabel=r'$M_1^{sur}$ [$M_\odot$]',
        xscale='log',
        yscale='log',
        axisbelow=(True),
        # aspect=('equal')
    )

    sur.legend(fontsize=5)

    # plt.grid(True, color='gray', ls='dotted')
    #plt.savefig(opts.plots_directory + '/m1m2.png', format='png')
    #plt.show()
    
    # ========================================
    # NOSUR - Mass 1 vs Mass 2
    # ========================================

    # Sort Objects into Mass 1 and Mass 2 by generation
    nosur_mass_mask_g1 = nosur_mergers[nosur_merger_g1_mask, 6] > nosur_mergers[nosur_merger_g1_mask, 7]
    nosur_gen1_mass_1 = np.zeros(np.sum(nosur_merger_g1_mask))
    nosur_gen1_mass_1[nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 6][nosur_mass_mask_g1]
    nosur_gen1_mass_1[~nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 7][~nosur_mass_mask_g1]
    nosur_gen1_mass_2 = np.zeros(np.sum(nosur_merger_g1_mask))
    nosur_gen1_mass_2[~nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 6][~nosur_mass_mask_g1]
    nosur_gen1_mass_2[nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 7][nosur_mass_mask_g1]

    nosur_mass_mask_g2 = nosur_mergers[nosur_merger_g2_mask, 6] > nosur_mergers[nosur_merger_g2_mask, 7]
    nosur_gen2_mass_1 = np.zeros(np.sum(nosur_merger_g2_mask))
    nosur_gen2_mass_1[nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 6][nosur_mass_mask_g2]
    nosur_gen2_mass_1[~nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 7][~nosur_mass_mask_g2]
    nosur_gen2_mass_2 = np.zeros(np.sum(nosur_merger_g2_mask))
    nosur_gen2_mass_2[~nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 6][~nosur_mass_mask_g2]
    nosur_gen2_mass_2[nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 7][nosur_mass_mask_g2]

    nosur_mass_mask_gX = nosur_mergers[nosur_merger_gX_mask, 6] > nosur_mergers[nosur_merger_gX_mask, 7]
    nosur_genX_mass_1 = np.zeros(np.sum(nosur_merger_gX_mask))
    nosur_genX_mass_1[nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 6][nosur_mass_mask_gX]
    nosur_genX_mass_1[~nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 7][~nosur_mass_mask_gX]
    nosur_genX_mass_2 = np.zeros(np.sum(nosur_merger_gX_mask))
    nosur_genX_mass_2[~nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 6][~nosur_mass_mask_gX]
    nosur_genX_mass_2[nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 7][nosur_mass_mask_gX]

    # Check that there aren't any zeros remaining.
    assert (nosur_gen1_mass_1 > 0).all()
    assert (nosur_gen1_mass_2 > 0).all()
    assert (nosur_gen2_mass_1 > 0).all()
    assert (nosur_gen2_mass_2 > 0).all()
    assert (nosur_genX_mass_1 > 0).all()
    assert (nosur_genX_mass_2 > 0).all()

    pointsize_m1m2 = 5
    #fig = plt.figure(figsize=plotting.set_size(figsize))
    #ax4 = fig.add_subplot(111)

    # plt.scatter(m1, m2, s=pointsize_m1m2, color='k')
    nosur.scatter(nosur_gen1_mass_1, nosur_gen1_mass_2,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    nosur.scatter(nosur_gen2_mass_1, nosur_gen2_mass_2,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    nosur.scatter(nosur_genX_mass_1, nosur_genX_mass_2,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    nosur.set(
        xlabel=r'$M_1^{nosur}$ [$M_\odot$]',
        ylabel=r'$M_2$ [$M_\odot$]',
        xscale='log',
        yscale='log',
        axisbelow=(True),
        # aspect=('equal')
    )
        #ylabel=r'$M_2$ [$M_\odot$]',

    nosur.legend(fontsize=5)

    # plt.grid(True, color='gray', ls='dotted')
    plt.savefig(opts.plots_directory + '/m1m2.png', format='png')
    #plt.show()
    
    
    # ========================================
    # NOSUR - Mass 1 vs Mass 2                STAND ALONE PLOTS
    # ========================================

    # Sort Objects into Mass 1 and Mass 2 by generation
    nosur_mass_mask_g1 = nosur_mergers[nosur_merger_g1_mask, 6] > nosur_mergers[nosur_merger_g1_mask, 7]
    nosur_gen1_mass_1 = np.zeros(np.sum(nosur_merger_g1_mask))
    nosur_gen1_mass_1[nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 6][nosur_mass_mask_g1]
    nosur_gen1_mass_1[~nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 7][~nosur_mass_mask_g1]
    nosur_gen1_mass_2 = np.zeros(np.sum(nosur_merger_g1_mask))
    nosur_gen1_mass_2[~nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 6][~nosur_mass_mask_g1]
    nosur_gen1_mass_2[nosur_mass_mask_g1] = nosur_mergers[nosur_merger_g1_mask, 7][nosur_mass_mask_g1]

    nosur_mass_mask_g2 = nosur_mergers[nosur_merger_g2_mask, 6] > nosur_mergers[nosur_merger_g2_mask, 7]
    nosur_gen2_mass_1 = np.zeros(np.sum(nosur_merger_g2_mask))
    nosur_gen2_mass_1[nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 6][nosur_mass_mask_g2]
    nosur_gen2_mass_1[~nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 7][~nosur_mass_mask_g2]
    nosur_gen2_mass_2 = np.zeros(np.sum(nosur_merger_g2_mask))
    nosur_gen2_mass_2[~nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 6][~nosur_mass_mask_g2]
    nosur_gen2_mass_2[nosur_mass_mask_g2] = nosur_mergers[nosur_merger_g2_mask, 7][nosur_mass_mask_g2]

    nosur_mass_mask_gX = nosur_mergers[nosur_merger_gX_mask, 6] > nosur_mergers[nosur_merger_gX_mask, 7]
    nosur_genX_mass_1 = np.zeros(np.sum(nosur_merger_gX_mask))
    nosur_genX_mass_1[nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 6][nosur_mass_mask_gX]
    nosur_genX_mass_1[~nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 7][~nosur_mass_mask_gX]
    nosur_genX_mass_2 = np.zeros(np.sum(nosur_merger_gX_mask))
    nosur_genX_mass_2[~nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 6][~nosur_mass_mask_gX]
    nosur_genX_mass_2[nosur_mass_mask_gX] = nosur_mergers[nosur_merger_gX_mask, 7][nosur_mass_mask_gX]

    # Check that there aren't any zeros remaining.
    assert (nosur_gen1_mass_1 > 0).all()
    assert (nosur_gen1_mass_2 > 0).all()
    assert (nosur_gen2_mass_1 > 0).all()
    assert (nosur_gen2_mass_2 > 0).all()
    assert (nosur_genX_mass_1 > 0).all()
    assert (nosur_genX_mass_2 > 0).all()

    pointsize_m1m2 = 5
    #fig = plt.figure(figsize=plotting.set_size(figsize))
    fig, ax = plt.subplots(1, 1, figsize=plotting.set_size(figsize), sharey=True)
    #ax4 = fig.add_subplot(111)

    # plt.scatter(m1, m2, s=pointsize_m1m2, color='k')
    ax.scatter(nosur_gen1_mass_1, nosur_gen1_mass_2,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    ax.scatter(nosur_gen2_mass_1, nosur_gen2_mass_2,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax.scatter(nosur_genX_mass_1, nosur_genX_mass_2,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    ax.set(
        xlabel=r'$M_1$ [$M_\odot$] - (nosur)',
        ylabel=r'$M_2$ [$M_\odot$]',
        xscale='log',
        yscale='log',
        axisbelow=(True),
        # aspect=('equal')
    )
        #ylabel=r'$M_2$ [$M_\odot$]',

    ax.legend(fontsize=5)

    # plt.grid(True, color='gray', ls='dotted')
    plt.savefig(opts.plots_directory + '/m1m2_nosur.png', format='png')
    #plt.show()


    # ===============================
    ### SUR - kick velocity histogram ###
    # ===============================
    fig, ax = plt.subplots(1, 2, figsize=plotting.set_size(figsize), constrained_layout=True)
    
    sur = ax[1]
    nosur = ax[0]

    # make your bins...
    kick_bins = np.logspace(np.log10(sur_mergers[:, 16].min()), np.log10(sur_mergers[:, 16].max()+10), 50)

    sur_hist_data = [sur_mergers[:, 16][sur_merger_g1_mask], sur_mergers[:, 16][sur_merger_g2_mask], sur_mergers[:, 16][sur_merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    # plot the distribution of mergers as a function of generation
    sur.hist(sur_hist_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)
    sur.set_ylabel(r'n')
    sur.set_xlabel(r'v$_{kick}$ [km/s] - (sur)')
    sur.set_xscale('log')

    if figsize == 'apj_col':
        sur.legend(fontsize=6)
    elif figsize == 'apj_page':
        sur.legend()

    # plt.title(r"Distribution of v$_{kick}$")
    #sur.grid(True, color='gray', ls='dashed')
    #plt.savefig(opts.plots_directory + "/v_kick_distribution.png", format='png')
    #plt.close()
    
    # ===============================
    ### NOSUR - kick velocity histogram ###
    # ===============================
    #fig = plt.figure(figsize=plotting.set_size(figsize))

    # make your bins...
    kick_bins = np.logspace(np.log10(nosur_mergers[:, 16].min()), np.log10(nosur_mergers[:, 16].max()+10), 50)

    nosur_hist_data = [nosur_mergers[:, 16][nosur_merger_g1_mask], nosur_mergers[:, 16][nosur_merger_g2_mask], nosur_mergers[:, 16][nosur_merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    # plot the distribution of mergers as a function of generation
    nosur.hist(nosur_hist_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)
    nosur.set_ylabel(r'n')
    nosur.set_xlabel(r'v$_{kick}$ [km/s] - (nosur)')
    nosur.set_xscale('log')

    if figsize == 'apj_col':
        nosur.legend(fontsize=6)
    elif figsize == 'apj_page':
        nosur.legend()

    # plt.title(r"Distribution of v$_{kick}$")
    #plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/v_kick_distribution.png", format='png')
    #plt.show()

    # ===============================
    ### SUR - Kick Velocity vs Radius ###
    # ===============================

    sur_all_kick = sur_mergers[:, 16]
    sur_gen1_vkick = sur_all_kick[sur_merger_g1_mask]
    sur_gen2_vkick = sur_all_kick[sur_merger_g2_mask]
    sur_genX_vkick = sur_all_kick[sur_merger_gX_mask]

    # figsize is hardcoded here. don't change, shrink everything illegibly
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(4,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}, constrained_layout=True) 

    sur1 = axs[1][0]
    sur2 = axs[1][1]
    nosur1 = axs[0][0]
    nosur2 = axs[0][1]
    
    # plot 1g-1g mergers
    sur1.scatter(sur_gen1_orb_a, sur_gen1_vkick,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g-mg mergers
    sur1.scatter(sur_gen2_orb_a, sur_gen2_vkick,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    
    # plot 3g-ng mergers
    sur1.scatter(sur_genX_orb_a, sur_genX_vkick,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    # plot trap radius
    trap_radius = 700
    sur1.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    
    # configure scatter plot
    sur1.set_ylabel(r'$v_{kick}^{sur}$ [km/s]')
    sur1.set_xlabel(r'Radius [$R_g$]')
    sur1.set_xscale('log')
    sur1.set_yscale('log')
    sur1.set_xlim(3e2, 7e4)
    sur1.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        sur1.legend(fontsize=4)
    elif figsize == 'apj_page':
        sur1.legend()

    # calculate mean kick velocity for all mergers
    sur_mean_kick = np.mean(sur_mergers[:, 16])
    
    kick_bins = np.logspace(np.log10(sur_mergers[:, 16].min()), np.log10(sur_mergers[:, 16].max()+10), 50)

    # configure histogram
    sur2.hist(sur_hist_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation = 'horizontal')
    sur2.axhline(sur_mean_kick, color = 'black', linewidth = 1, linestyle = 'dashdot', label = r'$\langle v_{kick}\rangle $ =' + f"{sur_mean_kick:.2f}")
    sur2.grid(True, color='gray', ls='dashed')
    sur2.set_yscale('log')
    sur2.yaxis.tick_right()
    sur2.set_xlabel(r'n')
    sur2.set_xlim(0, 500)
    sur2.set_xticks([100,400])

    if figsize == 'apj_col':
        sur2.legend(fontsize=4, loc = 'lower right')
    elif figsize == 'apj_page':
        sur2.legend()

    # plt.title(r"v$_{kick} vs. semi-major axis with distribution of v$_{kick}$")
    #plt.tight_layout()
    #plt.savefig(opts.plots_directory + '/v_kick_vs_radius.png', format='png')
    #plt.close()
    
    
    # ===============================
    ### NOSUR - Kick Velocity vs Radius ### 
    # ===============================

    nosur_all_kick = nosur_mergers[:, 16]
    nosur_gen1_vkick = nosur_all_kick[nosur_merger_g1_mask]
    nosur_gen2_vkick = nosur_all_kick[nosur_merger_g2_mask]
    nosur_genX_vkick = nosur_all_kick[nosur_merger_gX_mask]

    # figsize is hardcoded here. don't change, shrink everything illegibly
    #fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 

    # plot 1g-1g mergers
    nosur1.scatter(nosur_gen1_orb_a, nosur_gen1_vkick,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g-mg mergers
    nosur1.scatter(nosur_gen2_orb_a, nosur_gen2_vkick,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    
    # plot 3g-ng mergers
    nosur1.scatter(nosur_genX_orb_a, nosur_genX_vkick,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    # plot trap radius
    trap_radius = 700
    nosur1.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    
    # configure scatter plot
    nosur1.set_ylabel(r'$v_{kick}^{nosur}$ [km/s]')
    nosur1.set_xlabel(r'Radius [$R_g$]')
    nosur1.set_xscale('log')
    nosur1.set_yscale('log')
    nosur1.set_xlim(3e2, 7e4)
    nosur1.grid(True, color='gray', ls='dashed')
    #if figsize == 'apj_col':
    #    nosur1.legend(fontsize=5, loc='upper left')
    #elif figsize == 'apj_page':
    #    nosur1.legend()

    # calculate mean kick velocity for all mergers
    nosur_mean_kick = np.mean(nosur_mergers[:, 16])
    
    kick_bins = np.logspace(np.log10(nosur_mergers[:, 16].min()), np.log10(nosur_mergers[:, 16].max()+10), 50)

    # configure histogram
    nosur2.hist(nosur_hist_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation='horizontal')
    nosur2.axhline(nosur_mean_kick, color = 'black', linewidth = 1, linestyle = 'dashdot', label = r'$\langle v_{kick}\rangle $ =' + f"{nosur_mean_kick:.2f}")
    nosur2.grid(True, color='gray', ls='dashed')
    nosur2.set_yscale('log')
    nosur2.yaxis.tick_right()
    nosur2.set_xlim(0, 500)
    nosur2.set_xlabel(r'n')
    nosur2.set_xticks([100,400])

    if figsize == 'apj_col':
        nosur2.legend(fontsize=4, loc = 'lower right')
    elif figsize == 'apj_page':
        nosur2.legend()

    # plt.title(r"v$_{kick} vs. semi-major axis with distribution of v$_{kick}$")
    plt.tight_layout()
    plt.savefig(opts.plots_directory + '/v_kick_vs_radius.png', format='png')
    #plt.savefig(opts.plots_directory + '/v_kick_vs_radius_nosur.png', format='png')
    #plt.show()
    
    # ===============================
    ### NOSUR - Kick Velocity vs Radius ### STAND ALONE PLOT
    # ===============================

    nosur_all_kick = nosur_mergers[:, 16]
    nosur_gen1_vkick = nosur_all_kick[nosur_merger_g1_mask]
    nosur_gen2_vkick = nosur_all_kick[nosur_merger_g2_mask]
    nosur_genX_vkick = nosur_all_kick[nosur_merger_gX_mask]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(4,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 
    kvel1 = axs[0]
    kvel2 = axs[1]
    # figsize is hardcoded here. don't change, shrink everything illegibly
    #fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5,3), gridspec_kw={'width_ratios': [3, 1], 'wspace':0, 'hspace':0}) 

    # plot 1g-1g mergers
    kvel1.scatter(nosur_gen1_orb_a, nosur_gen1_vkick,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot 2g-mg mergers
    kvel1.scatter(nosur_gen2_orb_a, nosur_gen2_vkick,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    
    # plot 3g-ng mergers
    kvel1.scatter(nosur_genX_orb_a, nosur_genX_vkick,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )
    
    # plot trap radius
    trap_radius = 700
    kvel1.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')
    
    # configure scatter plot
    kvel1.set_ylabel(r'$v_{kick}^{nosur}$ [km/s]')
    kvel1.set_xlabel(r'Radius [$R_g$]')
    kvel1.set_xscale('log')
    kvel1.set_yscale('log')
    kvel1.set_xlim(3e2, 7e4)
    kvel1.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        kvel1.legend(fontsize=5, loc='lower left')
    elif figsize == 'apj_page':
        kvel1.legend()

    # calculate mean kick velocity for all mergers
    nosur_mean_kick = np.mean(nosur_mergers[:, 16])
    
    kick_bins = np.logspace(np.log10(nosur_mergers[:, 16].min()), np.log10(nosur_mergers[:, 16].max()+10), 50)

    # configure histogram
    kvel2.hist(nosur_hist_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True, orientation='horizontal')
    kvel2.axhline(nosur_mean_kick, color = 'black', linewidth = 1, linestyle = 'dashdot', label = r'$\langle v_{kick}\rangle $ =' + f"{nosur_mean_kick:.2f}")
    kvel2.grid(True, color='gray', ls='dashed')
    kvel2.set_yscale('log')
    kvel2.yaxis.tick_right()
    kvel2.set_xlim(0, 500)
    kvel2.set_xlabel(r'n')
    kvel2.set_xticks([100,400])

    if figsize == 'apj_col':
        kvel2.legend(fontsize=4, loc = 'lower right')
    elif figsize == 'apj_page':
       kvel2.legend()

    # plt.title(r"v$_{kick} vs. semi-major axis with distribution of v$_{kick}$")
    plt.tight_layout()
    plt.savefig(opts.plots_directory + '/v_kick_vs_radius_nosur.png', format='png')
    #plt.show()
    
    # ===============================
    ### SUR - kick velocity histogram across disk radius ###
    # ===============================
    fig, ax = plt.subplots(1, 2, figsize=plotting.set_size(figsize), sharey=True)
    
    sur = ax[1]
    nosur = ax[0]

    # make your bins...
    sur_radius_bins = np.logspace(np.log10(sur_mergers[:, 1].min()), np.log10(sur_mergers[:, 1].max()+10), 50)

    sur_hist_data = [sur_mergers[:, 1][sur_merger_g1_mask], sur_mergers[:, 1][sur_merger_g2_mask], sur_mergers[:, 1][sur_merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    # plot the distribution of mergers as a function of generation
    sur.hist(sur_hist_data, bins=sur_radius_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)
    sur.set_xlabel(r'Radius$_{sur}$ [$R_g$]')
    #sur.set_ylabel(r'No. of mergers')
    sur.set_xscale('log')

    if figsize == 'apj_col':
        sur.legend(fontsize=6)
    elif figsize == 'apj_page':
        sur.legend()

    # plt.title(r"Distribution of v$_{kick}$")
    #sur.grid(True, color='gray', ls='dashed')
    #plt.savefig(opts.plots_directory + "/v_kick_distribution.png", format='png')
    #plt.close()
    
    # ===============================
    ### NOSUR - kick velocity histogram across disk radius ###
    # ===============================
    #fig = plt.figure(figsize=plotting.set_size(figsize))

    # make your bins...
    nosur_radius_bins = np.logspace(np.log10(nosur_mergers[:, 1].min()), np.log10(nosur_mergers[:, 1].max()+10), 50)

    nosur_hist_data = [nosur_mergers[:, 1][nosur_merger_g1_mask], nosur_mergers[:, 1][nosur_merger_g2_mask], nosur_mergers[:, 1][nosur_merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    # plot the distribution of mergers as a function of generation
    nosur.hist(nosur_hist_data, bins=nosur_radius_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)
    nosur.set_xlabel(r'Radius$_{nosur}$ [$R_g$]')
    nosur.set_ylabel(r'No. of mergers')
    nosur.set_xscale('log')

    if figsize == 'apj_col':
        nosur.legend(fontsize=6)
    elif figsize == 'apj_page':
        nosur.legend()

    # plt.title(r"Distribution of v$_{kick}$")
    #plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/v_kick_dist_radius.png", format='png')
    #plt.show()

    
    # ========================================
    # SUR - Kick Velocity vs Spin
    # ========================================

    sur_spin = sur_mergers[:, 4]
    sur_gen1_spin = sur_spin[sur_merger_g1_mask]
    sur_gen2_spin = sur_spin[sur_merger_g2_mask]
    sur_genX_spin = sur_spin[sur_merger_gX_mask]
    
    sur_all_kick = sur_mergers[:, 16]
    sur_gen1_vkick = sur_all_kick[sur_merger_g1_mask]
    sur_gen2_vkick = sur_all_kick[sur_merger_g2_mask]
    sur_genX_vkick = sur_all_kick[sur_merger_gX_mask]

    fig, ax = plt.subplots(1, 2, figsize=(4.5,2.5), sharey=True, gridspec_kw={'wspace':0, 'hspace':0})
    #ax3 = fig.add_subplot(111)
    
    sur = ax[1]
    nosur = ax[0]

    # plot the 1g-1g mergers
    sur.scatter(sur_gen1_vkick, sur_gen1_spin,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    sur.scatter(sur_gen2_vkick, sur_gen2_spin,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    sur.scatter(sur_genX_vkick, sur_genX_spin,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    sur.set(
        xlabel=r'$v_{kick}^{sur}$ [km/s]',
        xscale="log",
        axisbelow=True,
        xlim=([2e0,4e3])
    )

    sur.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        sur.legend(fontsize=6)
    elif figsize == 'apj_page':
        sur.legend()

    #plt.savefig(opts.plots_directory + '/time_of_merger.png', format='png')
    #plt.close()
    
    # ========================================
    # NOSUR - Kick Velocity vs Spin
    # ========================================

    nosur_spin = nosur_mergers[:, 4]
    nosur_gen1_spin = nosur_spin[nosur_merger_g1_mask]
    nosur_gen2_spin = nosur_spin[nosur_merger_g2_mask]
    nosur_genX_spin = nosur_spin[nosur_merger_gX_mask]
    
    nosur_all_kick = nosur_mergers[:, 16]
    nosur_gen1_vkick = nosur_all_kick[nosur_merger_g1_mask]
    nosur_gen2_vkick = nosur_all_kick[nosur_merger_g2_mask]
    nosur_genX_vkick = nosur_all_kick[nosur_merger_gX_mask]

    # plot the 1g-1g mergers
    nosur.scatter(nosur_gen1_vkick, nosur_gen1_spin,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # plot the 2g+ mergers
    nosur.scatter(nosur_gen2_vkick, nosur_gen2_spin,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    nosur.scatter(nosur_genX_vkick, nosur_genX_spin,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    nosur.set(
        xlabel=r'$v_{kick}^{nosur}$ [km/s]',
        ylabel='Spin',
        xscale="log",
        axisbelow=True,
        xlim=([2e0,4e3])
    )

    nosur.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        nosur.legend(fontsize=6, loc='lower left')
    elif figsize == 'apj_page':
        nosur.legend()

    plt.savefig(opts.plots_directory + '/vkick_spin.png', format='png')
    #plt.close()


    # ========================================
    # SUR - LVK and LISA Strain vs Freq
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
    sur_lisa = li.LISA()

    #   lisa_freq is the frequency (x-axis) being created
    #   lisa_sn is the sensitivity curve of LISA
    sur_lisa_freq = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
    sur_lisa_sn = sur_lisa.Sn(sur_lisa_freq)

    # Create figure and ax
    fig, svf_ax = plt.subplots(1, 2, figsize=(plotting.set_size(figsize)[0], 2.9), sharey=True)
    
    sur = svf_ax[1]
    nosur = svf_ax[0]

    sur.set_xlabel(r'f [Hz]')  # , fontsize=20, labelpad=10)
    sur.set_ylabel(r'${\rm h}_{\rm char}$')  # , fontsize=20, labelpad=10)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    sur.set_xlim(0.5e-7, 1.0e+4)
    sur.set_ylim(1.0e-28, 1.0e-15)

    # ----------Finding the rows in which EMRIs signals are either identical or zeroes and removing them----------
    sur_identical_rows_emris = np.where(sur_emris[:, 5] == sur_emris[:, 6])
    sur_zero_rows_emris = np.where(sur_emris[:, 6] == 0)
    sur_emris = np.delete(sur_emris, sur_identical_rows_emris, 0)
    # emris = np.delete(emris,zero_rows_emris,0)
    sur_emris[~np.isfinite(sur_emris)] = 1.e-40

    # ----------Finding the rows in which LVKs signals are either identical or zeroes and removing them----------
    sur_identical_rows_lvk = np.where(sur_lvk[:, 5] == sur_lvk[:, 6])
    sur_zero_rows_lvk = np.where(sur_lvk[:, 6] == 0)
    sur_lvk = np.delete(sur_lvk, sur_identical_rows_lvk, 0)
    # lvk = np.delete(lvk,zero_rows_lvk,0)
    sur_lvk[~np.isfinite(sur_lvk)] = 1.e-40

    sur_lvk_g1_mask, sur_lvk_g2_mask, sur_lvk_gX_mask = make_gen_masks(sur_lvk, 7, 8)

    sur_lvk_g1 = sur_lvk[sur_lvk_g1_mask]
    sur_lvk_g2 = sur_lvk[sur_lvk_g2_mask]
    sur_lvk_gX = sur_lvk[sur_lvk_gX_mask]

    # ----------Setting the values for the EMRIs and LVKs signals and inverting them----------
    sur_inv_freq_emris = 1 / sur_emris[:, 6]
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
    sur_strain_per_freq_emris = sur_emris[:, 5] * sur_inv_freq_emris / timestep

    sur_strain_per_freq_lvk_g1 = sur_lvk_g1[:, 5] * (1 / sur_lvk_g1[:, 6]) / timestep
    sur_strain_per_freq_lvk_g2 = sur_lvk_g2[:, 5] * (1 / sur_lvk_g2[:, 6]) / timestep
    sur_strain_per_freq_lvk_gX = sur_lvk_gX[:, 5] * (1 / sur_lvk_gX[:, 6]) / timestep

    # plot the characteristic detector strains
    sur.loglog(sur_lisa_freq, np.sqrt(sur_lisa_freq * sur_lisa_sn),
              label='LISA Sensitivity',
              #   color='darkred',
              zorder=0)

    sur.loglog(f_H1, h_H1,
              label='LIGO O3, H1 Sensitivity',
              #   color='darkblue',
              zorder=0)

    sur.scatter(sur_emris[:, 6], sur_strain_per_freq_emris,
               s=0.4 * styles.markersize_gen1,
               alpha=styles.markeralpha_gen1
               )

    sur.scatter(sur_lvk_g1[:, 6], sur_strain_per_freq_lvk_g1,
                   s=0.4 * styles.markersize_gen1,
                   marker=styles.marker_gen1,
                   edgecolor=styles.color_gen1,
                   facecolor='none',
                   alpha=styles.markeralpha_gen1,
                   label='1g-1g'
                   )

    sur.scatter(sur_lvk_g2[:, 6], sur_strain_per_freq_lvk_g2,
                   s=0.4 * styles.markersize_gen2,
                   marker=styles.marker_gen2,
                   edgecolor=styles.color_gen2,
                   facecolor='none',
                   alpha=styles.markeralpha_gen2,
                   label='2g-1g or 2g-2g'
                   )

    sur.scatter(sur_lvk_gX[:, 6], sur_strain_per_freq_lvk_gX,
                   s=0.4 * styles.markersize_genX,
                   marker=styles.marker_genX,
                   edgecolor=styles.color_genX,
                   facecolor='none',
                   alpha=styles.markeralpha_genX,
                   label=r'$\geq$3g-Ng'
                   )

    sur.set_yscale('log')
    sur.set_xscale('log')

    # ax.loglog(f_L1, h_L1,label = 'LIGO O3, L1 Sensitivity') # plot the characteristic strain
    # ax.loglog(f_gw,h,color ='black', label='GW150914')

    if figsize == 'apj_col':
        plt.legend(fontsize=7, loc="best")
    elif figsize == 'apj_page':
        plt.legend(loc="upper right")

    sur.set_xlabel(r'$\nu_{\rm GW}$ [Hz]')  # , fontsize=20, labelpad=10)
    sur.set_ylabel(r'$h_{\rm char}/\nu_{\rm GW}$')  # , fontsize=20, labelpad=10)

    #plt.savefig(opts.plots_directory + './gw_strain.png', format='png')
    #plt.show()
    
    # ========================================
    # NOSUR - LVK and LISA Strain vs Freq
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
    nosur_lisa = li.LISA()

    #   lisa_freq is the frequency (x-axis) being created
    #   lisa_sn is the sensitivity curve of LISA
    nosur_lisa_freq = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
    nosur_lisa_sn = nosur_lisa.Sn(nosur_lisa_freq)

    # Create figure and ax
    #fig, svf_ax = plt.subplots(1, 2, figsize=(plotting.set_size(figsize)[0], 2.9))

    nosur.set_xlabel(r'f [Hz]')  # , fontsize=20, labelpad=10)
    #nosur.set_ylabel(r'${\rm h}_{\rm char}$')  # , fontsize=20, labelpad=10)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    nosur.set_xlim(0.5e-7, 1.0e+4)
    nosur.set_ylim(1.0e-28, 1.0e-15)

    # ----------Finding the rows in which EMRIs signals are either identical or zeroes and removing them----------
    nosur_identical_rows_emris = np.where(nosur_emris[:, 5] == nosur_emris[:, 6])
    nosur_zero_rows_emris = np.where(nosur_emris[:, 6] == 0)
    nosur_emris = np.delete(nosur_emris, nosur_identical_rows_emris, 0)
    # emris = np.delete(emris,zero_rows_emris,0)
    nosur_emris[~np.isfinite(nosur_emris)] = 1.e-40

    # ----------Finding the rows in which LVKs signals are either identical or zeroes and removing them----------
    nosur_identical_rows_lvk = np.where(nosur_lvk[:, 5] == nosur_lvk[:, 6])
    nosur_zero_rows_lvk = np.where(nosur_lvk[:, 6] == 0)
    nosur_lvk = np.delete(nosur_lvk, nosur_identical_rows_lvk, 0)
    # lvk = np.delete(lvk,zero_rows_lvk,0)
    nosur_lvk[~np.isfinite(nosur_lvk)] = 1.e-40

    nosur_lvk_g1_mask, nosur_lvk_g2_mask, nosur_lvk_gX_mask = make_gen_masks(nosur_lvk, 7, 8)

    nosur_lvk_g1 = nosur_lvk[nosur_lvk_g1_mask]
    nosur_lvk_g2 = nosur_lvk[nosur_lvk_g2_mask]
    nosur_lvk_gX = nosur_lvk[nosur_lvk_gX_mask]

    # ----------Setting the values for the EMRIs and LVKs signals and inverting them----------
    nosur_inv_freq_emris = 1 / nosur_emris[:, 6]
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
    nosur_strain_per_freq_emris = nosur_emris[:, 5] * nosur_inv_freq_emris / timestep

    nosur_strain_per_freq_lvk_g1 = nosur_lvk_g1[:, 5] * (1 / nosur_lvk_g1[:, 6]) / timestep
    nosur_strain_per_freq_lvk_g2 = nosur_lvk_g2[:, 5] * (1 / nosur_lvk_g2[:, 6]) / timestep
    nosur_strain_per_freq_lvk_gX = nosur_lvk_gX[:, 5] * (1 / nosur_lvk_gX[:, 6]) / timestep

    # plot the characteristic detector strains
    nosur.loglog(nosur_lisa_freq, np.sqrt(nosur_lisa_freq * nosur_lisa_sn),
              label='LISA Sensitivity',
              #   color='darkred',
              zorder=0)

    nosur.loglog(f_H1, h_H1,
              label='LIGO O3, H1 Sensitivity',
              #   color='darkblue',
              zorder=0)

    nosur.scatter(nosur_emris[:, 6], nosur_strain_per_freq_emris,
               s=0.4 * styles.markersize_gen1,
               alpha=styles.markeralpha_gen1
               )

    nosur.scatter(nosur_lvk_g1[:, 6], nosur_strain_per_freq_lvk_g1,
                   s=0.4 * styles.markersize_gen1,
                   marker=styles.marker_gen1,
                   edgecolor=styles.color_gen1,
                   facecolor='none',
                   alpha=styles.markeralpha_gen1,
                   label='1g-1g'
                   )

    nosur.scatter(nosur_lvk_g2[:, 6], nosur_strain_per_freq_lvk_g2,
                   s=0.4 * styles.markersize_gen2,
                   marker=styles.marker_gen2,
                   edgecolor=styles.color_gen2,
                   facecolor='none',
                   alpha=styles.markeralpha_gen2,
                   label='2g-1g or 2g-2g'
                   )

    nosur.scatter(nosur_lvk_gX[:, 6], nosur_strain_per_freq_lvk_gX,
                   s=0.4 * styles.markersize_genX,
                   marker=styles.marker_genX,
                   edgecolor=styles.color_genX,
                   facecolor='none',
                   alpha=styles.markeralpha_genX,
                   label=r'$\geq$3g-Ng'
                   )

    nosur.set_yscale('log')
    nosur.set_xscale('log')

    # ax.loglog(f_L1, h_L1,label = 'LIGO O3, L1 Sensitivity') # plot the characteristic strain
    # ax.loglog(f_gw,h,color ='black', label='GW150914')

    if figsize == 'apj_col':
        plt.legend(fontsize=6, loc="best")
    elif figsize == 'apj_page':
        plt.legend(loc="upper right")

    nosur.set_xlabel(r'$\nu_{\rm GW}$ [Hz]')  # , fontsize=20, labelpad=10)
    #nosur.set_ylabel(r'$h_{\rm char}/\nu_{\rm GW}$')  # , fontsize=20, labelpad=10)

    plt.savefig(opts.plots_directory + './gw_strain.png', format='png')
    plt.show()



######## Execution ########
if __name__ == "__main__":
    main()
