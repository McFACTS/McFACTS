#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import os
# Grab those txt files
from mcfacts.vis import plotting
from mcfacts.vis import styles

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

#apj_col or apj_page
figsize = "apj_col"

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-directory",
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--fname-disk",
                        default="output_diskmasscycled.dat",
                        type=str, help="diskmasscycled.dat file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts

def main():
    opts = arg()

    data = np.loadtxt(opts.fname_disk, skiprows=1)

    # Get the average and std of mass gained/lost at each timestep

    mass_lost_avg = []
    mass_gain_avg = []
    mass_lost_std = []
    mass_gain_std = []

    for t in np.unique(data[:, 1]):
        mass_gain_avg.append(np.mean(data[:, 2][data[:, 1] == t]))
        mass_gain_std.append(np.mean(data[:, 2][data[:, 1] == t]))
        mass_lost_avg.append(np.mean(data[:, 3][data[:, 1] == t]))
        mass_lost_std.append(np.std(data[:, 3][data[:, 1] == t]))

    mass_gain_avg = np.array(mass_gain_avg)
    mass_gain_std = np.array(mass_gain_std)
    mass_lost_avg = np.array(mass_lost_avg)
    mass_lost_std = np.array(mass_lost_std)

    # Transform into Msun/yr. Hardcoding timestep as 10,000 years.
    timestep_division = 1e6
    mass_gain_avg_rate = mass_gain_avg / timestep_division
    mass_gain_std_rate = mass_gain_std / timestep_division
    mass_lost_avg_rate = mass_lost_avg / timestep_division
    mass_lost_std_rate = mass_lost_std / timestep_division

    # ========================================
    # Mass lost from the disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_division, -mass_lost_avg, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_division, -(mass_lost_avg - mass_lost_std), -(mass_lost_avg + mass_lost_std), alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [Myr]")
    plt.ylabel(r"$M_{\rm disk}$ lost [$M_\odot$]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_division + 1, 6))

    plt.savefig(opts.plots_directory + r"/mass_disk_lost.png",format="png")
    plt.close()

    # ========================================
    # Mass gained to the disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_division, mass_gain_avg, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_division, mass_gain_avg - mass_gain_std, mass_gain_avg + mass_gain_std, alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [Myr]")
    plt.ylabel(r"$M_{\rm disk}$ gained [$M_\odot$]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_division + 1, 6))

    plt.savefig(opts.plots_directory + r"/mass_disk_gain.png",format="png")
    plt.close()

    # ========================================
    # Mass accretion rate for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_division, mass_gain_avg_rate, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_division, mass_gain_avg_rate - mass_gain_std_rate, mass_gain_avg_rate + mass_gain_std_rate, alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [Myr]")
    plt.ylabel(r"$M_{\rm disk}$ accretion [$M_\odot$/yr]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_division + 1, 6))

    plt.savefig(opts.plots_directory + r"/mdot_disk_gain.png",format="png")
    plt.close()

    # ========================================
    # Mass loss rate for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_division, -mass_lost_avg_rate, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_division, -(mass_lost_avg_rate - mass_lost_std_rate), -(mass_lost_avg_rate + mass_lost_std_rate), alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [Myr]")
    plt.ylabel(r"$M_{\rm disk}$ los [$M_\odot$/yr]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_division + 1, 6))

    plt.savefig(opts.plots_directory + r"/mdot_disk_loss.png",format="png")
    plt.close()



######## Execution ########
if __name__ == "__main__":
    main()
