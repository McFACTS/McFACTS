import matplotlib.ticker as mticker
import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from importlib import resources as impresources

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.snapshot import TxtSnapshotHandler

import mcfacts.vis.LISA as li
from mcfacts.vis import data
from mcfacts.vis import styles
from mcfacts.vis import plotting

def make_gen_masks(gen_obj1, gen_obj2):
    """Create masks for retrieving different sets of a merged or binary population based on generation.
    """
    # Column of generation data

    # Masks for hierarchical generations
    # g1 : all 1g-1g objects
    # g2 : 2g-1g and 2g-2g objects
    # g3 : >=3g-Ng (first object at least 3rd gen; second object any gen)
    # Pipe operator (|) = logical OR. (&)= logical AND.
    g1_mask = (gen_obj1 == 1) & (gen_obj2 == 1)
    g2_mask = ((gen_obj1 == 2) | (gen_obj2 == 2)) & ((gen_obj1 <= 2) & (gen_obj2 <= 2))
    gX_mask = (gen_obj1 >= 3) | (gen_obj2 >= 3)

    return g1_mask, g2_mask, gX_mask


def linefunc(x, m):
    """Model for a line passing through (x,y) = (0,1).

    Function for a line used when fitting to the data.
    """
    return m * (x - 1)


def num_mergers_vs_mass(settings, figsize, save_dir, merger_masks, mass_final):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks

    fig = plt.figure(figsize=plotting.set_size(figsize))
    counts, bins = np.histogram(mass_final)
    bins = np.arange(int(mass_final.min()), int(mass_final.max()) + 2, 1)

    hist_data = [mass_final[merger_g1_mask], mass_final[merger_g2_mask], mass_final[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.xscale('log')
    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    svf_ax.tick_params(axis='x', direction='out', which='both')
    svf_ax.yaxis.grid(True, color='gray', ls='dashed')

    plt.xticks(np.geomspace(int(mass_final.min()), int(mass_final.max()), 5).astype(int))

    svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "merger_remnant_mass.png"), format='png', dpi=300)
    plt.close()


def merger_vs_radius(settings, figsize, save_dir, merger_masks, mass, orb_a):
    # for i in range(len(mergers[:, 1])):
    #     if mergers[i, 1] < 10.0:
    #         mergers[i, 1] = 10.0

    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks

    # Separate generational subpopulations
    gen1_orb_a = orb_a[merger_g1_mask]
    gen2_orb_a = orb_a[merger_g2_mask]
    genX_orb_a = orb_a[merger_gX_mask]
    gen1_mass = mass[merger_g1_mask]
    gen2_mass = mass[merger_g2_mask]
    genX_mass = mass[merger_gX_mask]

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

    plt.axvline(settings.disk_radius_trap, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {settings.disk_radius_trap:.0f} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Remnant Mass [$M_\odot$]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.ylim(15, 1000)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "merger_mass_v_radius.png"), format='png', dpi=300)
    plt.close()


def q_vs_chi_effective(settings, figsize, save_dir, merger_masks, mass_1, mass_2, chi_eff):
    item_len = len(mass_1)
    mass_ratio = np.zeros(item_len)

    for i in range(item_len):
        if mass_1[i] < mass_2[i]:
            mass_ratio[i] = mass_1[i] / mass_2[i]
        else:
            mass_ratio[i] = mass_2[i] / mass_1[i]

    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks

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

    fig = plt.figure(figsize=(plotting.set_size(figsize)[0], 2.8))
    ax2 = fig.add_subplot(111)

    # Plot the 1g-1g mergers
    ax2.scatter(gen1_chi_eff, gen1_mass_ratio,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    # Plot the 2g+ mergers
    ax2.scatter(gen2_chi_eff, gen_mass_ratio,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # Plot the 3g+ mergers
    ax2.scatter(genX_chi_eff, genX_mass_ratio,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    # fit the hierarchical mergers (any binaries with 2+g) to a line passing through 0,1
    # popt contains the model parameters, pcov the covariances
    # poptHigh, pcovHigh = curve_fit(linefunc, high_gen_mass_ratio, high_gen_chi_eff)

    # Plot the line fitting the hierarchical population
    if len(genH_chi_eff) > 0:
        popt_hier, pcov_hier = curve_fit(linefunc, genH_mass_ratio, genH_chi_eff)
        err_hier = np.sqrt(np.diag(pcov_hier))[0]
        ax2.plot(linefunc(x, *popt_hier), x,
                 ls='dashed',
                 lw=1,
                 color='gray',
                 zorder=3,
                 label=r'$d\chi/dq(\geq$2g)=' +
                       f'{popt_hier[0]:.2f}' +
                       r'$\pm$' + f'{err_hier:.2f}'
                 )

    # Plot the line fitting for the entire population
    if len(chi_eff) > 0:
        popt_all, pcov_all = curve_fit(linefunc, mass_ratio, chi_eff)
        err_all = np.sqrt(np.diag(pcov_all))[0]
        ax2.plot(linefunc(x, *popt_all), x,
                 ls='solid',
                 lw=1,
                 color='black',
                 zorder=3,
                 label=r'$d\chi/dq$(all)=' +
                       f'{popt_all[0]:.2f}' +
                       r'$\pm$' + f'{err_all:.2f}'
                 )

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "q_chi_eff.png"), format='png', dpi=300)  # ,dpi=600)
    plt.close()


def disk_radius_vs_chi_p(settings, figsize, save_dir, merger_masks, orb_a, chi_p):
    # Can break out higher mass Chi_p events as test/illustration.
    # Set up default arrays for high mass BBH (>40Msun say) to overplot vs chi_p.

    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_orb_a = orb_a[merger_g1_mask]
    gen2_orb_a = orb_a[merger_g2_mask]
    genX_orb_a = orb_a[merger_gX_mask]
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

    plt.axvline(settings.disk_radius_trap, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {settings.disk_radius_trap:.0f} ' + r'$R_g$')

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "r_chi_p.png"), format='png', dpi=300)
    plt.close()


def time_vs_merger(settings, figsize, save_dir, merger_masks, mass, time_merged):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_time = time_merged[merger_g1_mask]
    gen2_time = time_merged[merger_g2_mask]
    genX_time = time_merged[merger_gX_mask]
    gen1_mass = mass[merger_g1_mask]
    gen2_mass = mass[merger_g2_mask]
    genX_mass = mass[merger_gX_mask]

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
        axisbelow=True,
        ylim=(1.7e1, 1.8e2)
    )

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'time_of_merger.png'), format='png', dpi=300)
    plt.close()


def mass_1_vs_mass_2(settings, figsize, save_dir, merger_masks, mass_1, mass_2):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks

    # Sort Objects into Mass 1 and Mass 2 by generation
    mass_mask_g1 = mass_1[merger_g1_mask] > mass_2[merger_g1_mask]
    gen1_mass_1 = np.zeros(np.sum(merger_g1_mask))
    gen1_mass_1[mass_mask_g1] = mass_1[merger_g1_mask][mass_mask_g1]
    gen1_mass_1[~mass_mask_g1] = mass_2[merger_g1_mask][~mass_mask_g1]
    gen1_mass_2 = np.zeros(np.sum(merger_g1_mask))
    gen1_mass_2[~mass_mask_g1] = mass_1[merger_g1_mask][~mass_mask_g1]
    gen1_mass_2[mass_mask_g1] = mass_2[merger_g1_mask][mass_mask_g1]

    mass_mask_g2 = mass_1[merger_g2_mask] > mass_2[merger_g2_mask]
    gen2_mass_1 = np.zeros(np.sum(merger_g2_mask))
    gen2_mass_1[mass_mask_g2] = mass_1[merger_g2_mask][mass_mask_g2]
    gen2_mass_1[~mass_mask_g2] = mass_2[merger_g2_mask][~mass_mask_g2]
    gen2_mass_2 = np.zeros(np.sum(merger_g2_mask))
    gen2_mass_2[~mass_mask_g2] = mass_1[merger_g2_mask][~mass_mask_g2]
    gen2_mass_2[mass_mask_g2] = mass_2[merger_g2_mask][mass_mask_g2]

    mass_mask_gX = mass_1[merger_gX_mask] > mass_2[merger_gX_mask]
    genX_mass_1 = np.zeros(np.sum(merger_gX_mask))
    genX_mass_1[mass_mask_gX] = mass_1[merger_gX_mask][mass_mask_gX]
    genX_mass_1[~mass_mask_gX] = mass_2[merger_gX_mask][~mass_mask_gX]
    genX_mass_2 = np.zeros(np.sum(merger_gX_mask))
    genX_mass_2[~mass_mask_gX] = mass_1[merger_gX_mask][~mass_mask_gX]
    genX_mass_2[mass_mask_gX] = mass_2[merger_gX_mask][mass_mask_gX]

    # Check that there aren't any zeros remaining.
    # assert (gen1_mass_1 > 0).all()
    # assert (gen1_mass_2 > 0).all()
    # assert (gen2_mass_1 > 0).all()
    # assert (gen2_mass_2 > 0).all()
    # assert (genX_mass_1 > 0).all()
    # assert (genX_mass_2 > 0).all()

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
        xlim=(9, 110),
        ylim=(0.9e1, 1.1e2)
        # aspect=('equal')
    )

    ax4.legend(fontsize=6)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'm1m2.png'), format='png', dpi=300)
    plt.close()


def kick_velocity_hist(settings, figsize, save_dir, merger_masks, v_kick):
    fig = plt.figure(figsize=plotting.set_size(figsize))

    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks

    kick_bins = np.logspace(np.log10(v_kick.min()), np.log10(v_kick.max()), 50)
    hist_kick_data = [v_kick[merger_g1_mask], v_kick[merger_g2_mask], v_kick[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_kick_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    # plot the distribution of mergers as a function of generation
    #plt.hist(hist_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)
    plt.ylabel(r'n')
    plt.xlabel(r'v$_{kick}$ [km/s]')
    plt.xscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    # plt.title(r"Distribution of v$_{kick}$")
    plt.grid(True, color='gray', ls='dashed')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "v_kick_distribution.png"), format='png', dpi=300)
    plt.close()


def kick_velocity_vs_radius(settings, figsize, save_dir, merger_masks, orb_a, v_kick):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_vkick = v_kick[merger_g1_mask]
    gen2_vkick = v_kick[merger_g2_mask]
    genX_vkick = v_kick[merger_gX_mask]
    gen1_orb_a = orb_a[merger_g1_mask]
    gen2_orb_a = orb_a[merger_g2_mask]
    genX_orb_a = orb_a[merger_gX_mask]

    # figsize is hardcoded here. don't change, shrink everything illegibly
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5.5, 3),
                            gridspec_kw={'width_ratios': [3, 1], 'wspace': 0, 'hspace': 0})

    plot = axs[0]
    hist = axs[1]

    # plot 1g-1g mergers
    plot.scatter(gen1_orb_a, gen1_vkick,
                 s=styles.markersize_gen1,
                 marker=styles.marker_gen1,
                 edgecolor=styles.color_gen1,
                 facecolors="none",
                 alpha=styles.markeralpha_gen1,
                 label='1g-1g'
                 )

    # plot 2g-mg mergers
    plot.scatter(gen2_orb_a, gen2_vkick,
                 s=styles.markersize_gen2,
                 marker=styles.marker_gen2,
                 edgecolor=styles.color_gen2,
                 facecolors="none",
                 alpha=styles.markeralpha_gen2,
                 label='2g-1g or 2g-2g'
                 )

    # plot 3g-ng mergers
    plot.scatter(genX_orb_a, genX_vkick,
                 s=styles.markersize_genX,
                 marker=styles.marker_genX,
                 edgecolor=styles.color_genX,
                 facecolors="none",
                 alpha=styles.markeralpha_genX,
                 label=r'$\geq$3g-Ng'
                 )

    # plot trap radius
    plot.axvline(settings.disk_radius_trap, color='k', linestyle='--', zorder=0,
                 label=f'Trap Radius = {settings.disk_radius_trap} ' + r'$R_g$')

    # configure scatter plot
    plot.set_ylabel(r'$v_{kick}$ [km/s]')
    plot.set_xlabel(r'Radius [$R_g$]')
    plot.set_xscale('log')
    plot.set_yscale('log')
    plot.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        plot.legend(fontsize=6, loc='lower right')
        plot.legend(fontsize=6, loc='lower right')
    elif figsize == 'apj_page':
        plot.legend()

    # calculate mean kick velocity for all mergers
    mean_kick = np.mean(v_kick)

    kick_bins = np.logspace(np.log10(v_kick.min()), np.log10(v_kick.max()), 50)
    hist_kick_data = [v_kick[merger_g1_mask], v_kick[merger_g2_mask], v_kick[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    # configure histogram
    hist.hist(hist_kick_data, bins=kick_bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label,
              stacked=True, orientation='horizontal')
    hist.axhline(mean_kick, color='black', linewidth=1, linestyle='dashdot',
                 label=r'$\langle v_{kick}\rangle $ =' + f"{mean_kick:.2f}")
    hist.grid(True, color='gray', ls='dashed')
    hist.set_yscale('log')
    hist.yaxis.tick_right()
    hist.set_xlabel(r'n')

    if figsize == 'apj_col':
        hist.legend(fontsize=6, loc='best')
    elif figsize == 'apj_page':
        hist.legend()

    # plt.title(r"v$_{kick} vs. semi-major axis with distribution of v$_{kick}$")
    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'v_kick_vs_radius.png'), format='png', dpi=300)
    plt.close()


def kick_velocity_vs_chi_eff(settings, figsize, save_dir, merger_masks, chi_eff, v_kick):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_vkick = v_kick[merger_g1_mask]
    gen2_vkick = v_kick[merger_g2_mask]
    genX_vkick = v_kick[merger_gX_mask]
    gen1_chi_eff = chi_eff[merger_g1_mask]
    gen2_chi_eff = chi_eff[merger_g2_mask]
    genX_chi_eff = chi_eff[merger_gX_mask]

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
    # plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, color='gray', ls='dashed')

    if figsize == 'apj_col':
        ax3.legend(fontsize=5, loc='lower right')
        ax3.legend(fontsize=5, loc='lower right')
    # elif figsize == 'apj_page':
    #     ax3.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "v_kick_vs_chi_eff.png"), format='png')
    plt.close()


def spin_vs_mass(settings, figsize, save_dir, merger_masks, mass, spin):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_mass = mass[merger_g1_mask]
    gen2_mass = mass[merger_g2_mask]
    genX_mass = mass[merger_gX_mask]
    gen1_spin = spin[merger_g1_mask]
    gen2_spin = spin[merger_g2_mask]
    genX_spin = spin[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_mass, gen1_spin,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_mass, gen2_spin,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_mass, genX_spin,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.ylabel(r'$a_{final}$')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlim(9, 140)
    plt.ylim(0.41, 1.01)

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.grid(True, color='gray', ls='dashed')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'spin_v_mass.png'), format='png', dpi=300)
    plt.close()


def spin_vs_radius(settings, figsize, save_dir, merger_masks, orb_a, spin):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_orb_a = orb_a[merger_g1_mask]
    gen2_orb_a = orb_a[merger_g2_mask]
    genX_orb_a = orb_a[merger_gX_mask]
    gen1_spin = spin[merger_g1_mask]
    gen2_spin = spin[merger_g2_mask]
    genX_spin = spin[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_orb_a, gen1_spin,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_orb_a, gen2_spin,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_orb_a, genX_spin,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plt.axvline(settings.disk_radius_trap, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {settings.disk_radius_trap} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.xlabel(r'Radius [$R_g$]')
    plt.ylabel(r'$a_{final}$')
    plt.xscale('log')
    #plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=4, loc = 'lower right')
    elif figsize == 'apj_page':
        plt.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'spin_vs_radius.png'), format='png', dpi=300)
    plt.close()


def spin_vs_kick(settings, figsize, save_dir, merger_masks, v_kick, spin):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_vkick = v_kick[merger_g1_mask]
    gen2_vkick = v_kick[merger_g2_mask]
    genX_vkick = v_kick[merger_gX_mask]
    gen1_spin = spin[merger_g1_mask]
    gen2_spin = spin[merger_g2_mask]
    genX_spin = spin[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_vkick, gen1_spin,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_vkick, gen2_spin,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_vkick, genX_spin,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plt.axvline(v_kick.mean(), color='k', linestyle='--', zorder=0,
                label=f'Analytical Kick Velocity = {v_kick.mean()} ' + r'$[km/s]$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.xlabel(r'$v_{kick}$ [km/s]')
    plt.ylabel(r'$a_{final}$')
    plt.xscale('log')
    plt.ylim(0.4, 1.01)

    if figsize == 'apj_col':
        plt.legend(fontsize=4)
    elif figsize == 'apj_page':
        plt.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'spin_vs_kick.png'), format='png', dpi=300)
    plt.close()


def kick_velocity_vs_mass(settings, figsize, save_dir, merger_masks, mass, v_kick):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks
    gen1_vkick = v_kick[merger_g1_mask]
    gen2_vkick = v_kick[merger_g2_mask]
    genX_vkick = v_kick[merger_gX_mask]
    gen1_mass = mass[merger_g1_mask]
    gen2_mass = mass[merger_g2_mask]
    genX_mass = mass[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_mass, gen1_vkick,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_mass, gen2_vkick,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_mass, genX_vkick,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plt.axhline(v_kick.mean(), color='k', linestyle='--', zorder=0,
                label=f'Analytical Kick Velocity = {v_kick.mean()} ' + r'$[km/s]$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.ylabel(r'$v_{kick}$ [km/s]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(9, 100)
    #plt.ylim(0.51, 1.01)

    if figsize == 'apj_col':
        plt.legend(fontsize=4)
    elif figsize == 'apj_page':
        plt.legend()

    #plt.ylim(18, 1000)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "v_kick_mass.png"), format='png', dpi=300)  # ,dpi=600)
    plt.close()


def strain_vs_freq(settings, figsize, save_dir, merger_masks, lvk):
    # H - hanford, L - Livingston
    H1 = impresources.files(data) / 'O3-H1-C01_CLEAN_SUB60HZ-1262197260.0_sensitivity_strain_asd.txt'
    L1 = impresources.files(data) / 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'

    # Adjust sep according to your delimiter (e.g., '\t' for tab-delimited files)
    dfh1 = pd.read_csv(H1, sep='\t', header=None)  # Use header=None if the file doesn't contain header row
    dfl1 = pd.read_csv(L1, sep='\t', header=None)

    # Using https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/blob/master/LISA.py
    # Create LISA object
    lisa = li.LISA()

    #   lisa_freq is the frequency (x-axis) being created
    #   lisa_sn is the sensitivity curve of LISA
    lisa_freq = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
    lisa_sn = lisa.Sn(lisa_freq)

    # Create figure and ax
    fig, axs = plt.subplots(1, 2, figsize=(plotting.set_size(figsize)[0] * 2, 2.9))

    lisa_axs = axs[0]

    lisa_axs.set_xlim(0.5e-7, 1.0e+1)
    lisa_axs.set_ylim(1.0e-26, 1.0e-15)

    # ----------Finding the rows in which EMRIs signals are either identical or zeroes and removing them----------
    # identical_rows_emris = np.where(emris[:, 5] == emris[:, 6])
    # zero_rows_emris = np.where(emris[:, 6] == 0)
    # emris = np.delete(emris, identical_rows_emris, 0)
    # # emris = np.delete(emris,zero_rows_emris,0)
    # emris[~np.isfinite(emris)] = 1.e-40

    # ----------Finding the rows in which LVKs signals are either identical or zeroes and removing them----------
    # identical_rows_lvk = np.where(lvk[:, 5] == lvk[:, 6])
    # zero_rows_lvk = np.where(lvk[:, 6] == 0)
    # lvk = np.delete(lvk, identical_rows_lvk, 0)
    # # lvk = np.delete(lvk,zero_rows_lvk,0)
    #lvk[~np.isfinite(lvk)] = 1.e-40

    lvk_g1_mask, lvk_g2_mask, lvk_gX_mask = make_gen_masks(lvk["gen_1"], lvk["gen_2"])

    # ----------Setting the values for the EMRIs and LVKs signals and inverting them----------
    #inv_freq_emris = 1 / emris[:, 6]
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
    #strain_per_freq_emris = emris[:, 5] # * inv_freq_emris / timestep

    # plot the characteristic detector strains
    lisa_axs.loglog(lisa_freq, np.sqrt(lisa_freq * lisa_sn),
                    label='LISA Sensitivity',
                    linewidth=1,
                    color='tab:orange',
                    zorder=0)

    # svf_ax.scatter(emris[:, 6], strain_per_freq_emris,
    #            s=0.4 * styles.markersize_gen1,
    #            alpha=styles.markeralpha_gen1
    #            )

    lisa_axs.scatter(lvk["gw_freq"][lvk_g1_mask], lvk["gw_strain"][lvk_g1_mask],
                   s=0.4 * styles.markersize_gen1,
                   marker=styles.marker_gen1,
                   edgecolor=styles.color_gen1,
                   facecolor='none',
                   alpha=styles.markeralpha_gen1,
                   label='1g-1g'
                   )

    lisa_axs.scatter(lvk["gw_freq"][lvk_g2_mask], lvk["gw_strain"][lvk_g2_mask],
                   s=0.4 * styles.markersize_gen2,
                   marker=styles.marker_gen2,
                   edgecolor=styles.color_gen2,
                   facecolor='none',
                   alpha=styles.markeralpha_gen2,
                   label='2g-1g or 2g-2g'
                   )

    lisa_axs.scatter(lvk["gw_freq"][lvk_gX_mask], lvk["gw_strain"][lvk_gX_mask],
                   s=0.4 * styles.markersize_genX,
                   marker=styles.marker_genX,
                   edgecolor=styles.color_genX,
                   facecolor='none',
                   alpha=styles.markeralpha_genX,
                   label=r'$\geq$3g-Ng'
                   )

    lisa_axs.loglog()

    if figsize == 'apj_col':
        lisa_axs.legend(fontsize=7, loc="upper right")
    elif figsize == 'apj_page':
        lisa_axs.legend(loc="upper right")

    lisa_axs.set_xlabel(r'$\nu_{\rm GW}$ [Hz]')
    lisa_axs.set_ylabel(r'$h_{\rm char}$')


    lvk_axs = axs[1]

    lvk_axs.set_xlim(5e0, 1e4)
    lvk_axs.set_ylim(2.0e-24, 1.0e-19)

    lvk_axs.loglog(dfh1[0], dfh1[1],
                   label='LIGO O3, H1 Sens.',
                   color='tab:blue',
                   linewidth=1,
                   zorder=0)

    lvk_axs.loglog(dfl1[0], dfl1[1],
                   label='LIGO O3, L1 Sens.',
                   color='tab:orange',
                   linewidth=1,
                   zorder=0)

    lvk_axs.scatter(lvk["gw_freq"][lvk_g1_mask], lvk["gw_strain"][lvk_g1_mask],
                    s=0.4 * styles.markersize_gen1,
                    marker=styles.marker_gen1,
                    edgecolor=styles.color_gen1,
                    facecolor='none',
                    alpha=styles.markeralpha_gen1,
                    )

    lvk_axs.scatter(lvk["gw_freq"][lvk_g2_mask], lvk["gw_strain"][lvk_g2_mask],
                    s=0.4 * styles.markersize_gen2,
                    marker=styles.marker_gen2,
                    edgecolor=styles.color_gen2,
                    facecolor='none',
                    alpha=styles.markeralpha_gen2,
                    )

    lvk_axs.scatter(lvk["gw_freq"][lvk_gX_mask], lvk["gw_strain"][lvk_gX_mask],
                    s=0.4 * styles.markersize_genX,
                    marker=styles.marker_genX,
                    edgecolor=styles.color_genX,
                    facecolor='none',
                    alpha=styles.markeralpha_genX,
                    )

    if figsize == 'apj_col':
        lvk_axs.legend(fontsize=7, loc="upper left")
    elif figsize == 'apj_page':
        lvk_axs.legend(loc="upper left")

    lvk_axs.set_xlabel(r'$\nu_{\rm GW}$ [Hz]')
    lvk_axs.yaxis.tick_right()
    lvk_axs.yaxis.set_label_position("right")
    lvk_axs.set_ylabel(r'$h_{\rm 0}$', rotation=-90, labelpad=15)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'gw_strain.png'), format='png')
    plt.close()


def main():
    # TODO: Handle settings import through arguments
    settings = SettingsManager({
        "verbose": False,
        "override_files": True,
        "save_state": True,
        "save_each_timestep": True
    })

    plt.style.use("mcfacts.vis.mcfacts_figures")

    figsize = "apj_col"

    snapshot_handler = TxtSnapshotHandler(settings)

    population_cabinet = snapshot_handler.load_cabinet("./runs", "population")

    mergers = population_cabinet["blackholes_merged"]
    lvk = population_cabinet["blackholes_lvk"]

    mass_1 = mergers["mass_1"]
    mass_2 = mergers["mass_2"]
    chi_eff = mergers["chi_eff"]
    mass_final = mergers["mass_final"]
    orb_a = mergers["orb_a"]
    chi_p = mergers["chi_p"]
    time_merged = mergers["time_merged"]
    v_kick = mergers["v_kick"]
    spin_final = mergers["spin_final"]

    merger_masks = (make_gen_masks(mergers["gen_1"], mergers["gen_2"])) # Man, I hate python

    plots_dir = "./runs/plots"

    num_mergers_vs_mass(settings, figsize, plots_dir, merger_masks, mass_final)
    merger_vs_radius(settings, figsize, plots_dir, merger_masks, mass_final, orb_a)
    q_vs_chi_effective(settings, figsize, plots_dir, merger_masks, mass_1, mass_2, chi_eff)
    disk_radius_vs_chi_p(settings, figsize, plots_dir, merger_masks, orb_a, chi_p)
    time_vs_merger(settings, figsize, plots_dir, merger_masks, mass_final, time_merged)
    mass_1_vs_mass_2(settings, figsize, plots_dir, merger_masks, mass_1, mass_2)
    kick_velocity_hist(settings, figsize, plots_dir, merger_masks, v_kick)
    kick_velocity_vs_radius(settings, figsize, plots_dir, merger_masks, orb_a, v_kick)
    kick_velocity_vs_chi_eff(settings, figsize, plots_dir, merger_masks, chi_eff, v_kick)
    spin_vs_mass(settings, figsize, plots_dir, merger_masks, mass_final, spin_final)
    spin_vs_radius(settings, figsize, plots_dir, merger_masks, orb_a, spin_final)
    spin_vs_kick(settings, figsize, plots_dir, merger_masks, v_kick, spin_final)
    kick_velocity_vs_mass(settings, figsize, plots_dir, merger_masks, mass_final, v_kick)

    strain_vs_freq(settings, figsize, plots_dir, merger_masks, lvk)


if __name__ == "__main__":
    main()