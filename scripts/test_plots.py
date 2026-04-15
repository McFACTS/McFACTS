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
    parser.add_argument("--fname-survivors",
                        default="output_mergers_survivors.dat",
                        type=str, help="output_survivors file")
    parser.add_argument("--fname-quiescence",
                        default="output_mergers_quiescence.dat",
                        type=str, help="output_quiescence file")
    parser.add_argument("--fname-log",
                        default="mcfacts.log",
                        type=str, help="log file")
    opts = parser.parse_args()
    print(opts.fname_mergers)
    #assert os.path.isfile(opts.fname_mergers)
    #assert os.path.isfile(opts.fname_emris)
    #assert os.path.isfile(opts.fname_lvk)
    #assert os.path.isfile(opts.fname_survivors)
    #assert os.path.isfile(opts.fname_quiescence)
    return opts

def curve(x):
    #Curve fit by eye
    return (1.7*(np.exp(-(x/10.0)**2) + 0.2*(np.exp(-x/1500.0))**0.5)) 


def main():
    # plt.style.use('seaborn-v0_8-poster')
    
    # Load data from output files
    opts = arg()

    quiescence=[1,5,10,50,100,500,1000]
    merger_ratio = [1.98, 1.69, 0.58, 0.33, 0.39, 0.23, 0.32]

    plt.scatter(quiescence, merger_ratio,
                label=r'Label'
                )

    #plt.axvline(700, color='k', linestyle='--', zorder=0,
    #            label=f'Trap Radius = {trap_radius:.0f} ' + r'$R_g$')

    #plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Ng-1g/Ng-mg')
    plt.xlabel(r'Quiescence Time (Myr)')
    plt.xscale('log')
    #plt.yscale('log')

    x_curve = np.linspace(min(quiescence),max(quiescence),500)
    y_curve =curve(x_curve)
    plt.plot(x_curve,y_curve, color ='black', linestyle = '--')
    
    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.ylim(0, 2.1)
    plt.xlim(0.9,1100)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/quiescence_time_v_merger_ratio.png", format='png')
    plt.close()

######## Execution ########
if __name__ == "__main__":
    main()