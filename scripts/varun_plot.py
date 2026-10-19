"""
fiducial_plots.py contains methods for creating default diagnostic and scientific plots from a computed McFACTS run.
"""
from pathlib import Path

#### IMPORTS
import matplotlib.ticker as mticker
import numpy as np
import os

import pandas as pd
import astropy.constants as const
from matplotlib import pyplot as plt, patches
from scipy.optimize import curve_fit
from importlib import resources as impresources

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.snapshot import TxtSnapshotHandler

import mcfacts.vis.LISA as li
from mcfacts.vis import data
from mcfacts.vis import styles
from mcfacts.vis import plotting


#### METHODS
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

    g1_mass = mass_final[merger_g1_mask]
    g2_mass = mass_final[merger_g2_mask]
    gX_mass = mass_final[merger_gX_mask]

    hist_data = [g1_mass, g2_mass, gX_mass]
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

    mass_sum = len(g1_mass) + len(g2_mass) + len(gX_mass)

    g1_ratio = len(g1_mass) / mass_sum
    g2_ratio = len(g2_mass) / mass_sum
    gX_ratio = len(gX_mass) / mass_sum

    rect_x = 180
    rect_y = 2
    rect_height = 25
    rect_width = 20

    rect1 = patches.Rectangle((rect_x, rect_y), rect_width, rect_height * g1_ratio, facecolor=styles.color_gen1)
    rect2 = patches.Rectangle((rect_x, rect_y + rect_height * g1_ratio), rect_width, rect_height * g2_ratio, facecolor=styles.color_gen2)
    rectX = patches.Rectangle((rect_x, rect_y + (rect_height * g1_ratio) + (rect_height * g2_ratio)), rect_width, rect_height * gX_ratio,
                              facecolor=styles.color_genX)

    svf_ax.add_patch(rect1)
    svf_ax.add_patch(rect2)
    svf_ax.add_patch(rectX)

    svf_ax.text(rect1.get_x() - 25, rect1.get_y() + rect1.get_height() / 2.0,
                f'{g1_ratio * 100:.2f}%', ha='center', va='center', fontsize=5)
    svf_ax.text(rect2.get_x() - 25, rect2.get_y() + rect2.get_height() / 2.0,
                f'{g2_ratio * 100:.2f}%', ha='center', va='center', fontsize=5)
    svf_ax.text(rectX.get_x() - 25, rectX.get_y() + rectX.get_height() / 2.0,
                f'{gX_ratio * 100:.2f}%', ha='center', va='center', fontsize=5)


    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "merger_remnant_mass.png"), format='png', dpi=300)
    plt.close()

    # ========================================
    # Number of Mergers vs Radius
    # ========================================


    # Plot intial and final mass distributions
    fig = plt.figure(figsize=plotting.set_size(figsize))
    counts, bins = np.histogram(mergers[:, 1]) #1 is position in radius, 2 is mass
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(mergers[:, 1].min()), int(mergers[:, 1].max()) + 2, 1)

    hist_data = [mergers[:, 1][merger_g1_mask], mergers[:, 1][merger_g2_mask], mergers[:, 1][merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    # plt.ylim(-5,max(counts))
    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    svf_ax.tick_params(axis='x', direction='in', which='both')
    #plt.grid(True, color='gray', ls='dashed')
    svf_ax.yaxis.grid(True, color='gray', ls='dashed')

    plt.xticks(np.geomspace(int(mergers[:, 1].min()), int( mergers[:, 1].max()), 5).astype(int))
    #plt.xticks(np.geomspace(20, 200, 5).astype(int))

    svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    print("Saving plot to:", os.path.abspath(opts.plots_directory + r"/mergers_vs_radius.png"))

    plt.savefig(opts.plots_directory + r"/mergers_vs_radius.png", format='png')

    plt.close()

if __name__ == "__main__":
    main()