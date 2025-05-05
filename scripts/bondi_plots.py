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
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--fname-mergers",
                        default="output_mergers_population.dat",
                        type=str, help="output_mergers file")
    parser.add_argument("--fname-bondi",
                        default="output_mergers_bondi_variables.dat",
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
    #bondi = np.loadtxt(opts.fname_bondi, skiprows=2)

    merger_nan_mask = (np.isfinite(mergers[:, 2])) & (mergers[:, 2] != 0)
    mergers = mergers[merger_nan_mask]

    merger_g1_mask, merger_g2_mask, merger_gX_mask = make_gen_masks(mergers, 12, 13)

    # Ensure no union between sets
    assert all(merger_g1_mask & merger_g2_mask) == 0
    assert all(merger_g1_mask & merger_gX_mask) == 0
    assert all(merger_g2_mask & merger_gX_mask) == 0

    # Ensure no elements are missed
    assert all(merger_g1_mask | merger_g2_mask | merger_gX_mask) == 1

    # retrieve component masses and mass ratio
    m1 = np.zeros(mergers.shape[0])
    m2 = np.zeros(mergers.shape[0])
    mass_ratio = np.zeros(mergers.shape[0])

    for i in range(mergers.shape[0]):
        if mergers[i, 6] < mergers[i, 7]:
            m1[i] = mergers[i, 7]
            m2[i] = mergers[i, 6]
            mass_ratio[i] = mergers[i, 6]
        else:
            mass_ratio[i] = mergers[i, 7]
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

    ax2.scatter(gen1_mass_ratio, gen1_chi_eff,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    #plot the 2g+ mergers
    ax2.scatter(gen_mass_ratio, gen2_chi_eff,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    # plot the 3g+ mergers
    ax2.scatter(genX_mass_ratio, genX_chi_eff,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plt.xscale('log')

    # if len(genH_chi_eff) > 0:
    #     poptHier, pcovHier = curve_fit(linefunc, genH_mass_ratio, genH_chi_eff)
    #     errHier = np.sqrt(np.diag(pcovHier))[0]
    #     # plot the line fitting the hierarchical mergers
    #     ax2.plot(linefunc(x, *poptHier), x,
    #              ls='dashed',
    #              lw=1,
    #              color='gray',
    #              zorder=3,
    #              label=r'$d\chi/dq(\geq$2g)=' +
    #                    f'{poptHier[0]:.2f}' +
    #                    r'$\pm$' + f'{errHier:.2f}'
    #              )
    #     #         #  alpha=linealpha,
    #
    # if len(chi_eff) > 0:
    #     poptAll, pcovAll = curve_fit(linefunc, mass_ratio, chi_eff)
    #     errAll = np.sqrt(np.diag(pcovAll))[0]
    #     ax2.plot(linefunc(x, *poptAll), x,
    #              ls='solid',
    #              lw=1,
    #              color='black',
    #              zorder=3,
    #              label=r'$d\chi/dq$(all)=' +
    #                    f'{poptAll[0]:.2f}' +
    #                    r'$\pm$' + f'{errAll:.2f}'
    #              )
    #     #  alpha=linealpha,

    # ax2.set(
    #     ylabel=r'$q = M_2 / M_1$',  # ($M_1 > M_2$)')
    #     xlabel=r'$\chi_{\rm eff}$',
    #     ylim=(0, 1),
    #     xlim=(-1, 1),
    #     axisbelow=True
    # )
    #
    # if figsize == 'apj_col':
    #     ax2.legend(loc='lower left', fontsize=6)
    # elif figsize == 'apj_page':
    #     ax2.legend(loc='lower left')

    #ax2.grid('on', color='gray', ls='dotted')
    plt.savefig(opts.plots_directory + "/chi_eff_mass_ratio.png", format='png')  # ,dpi=600)
    plt.close()

######## Execution ########
if __name__ == "__main__":
    main()
