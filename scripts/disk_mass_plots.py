#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import os
from mcfacts.vis import styles
from mcfacts.vis import plotting
from mcfacts.outputs.ReadOutputs import ReadLog
from pagn import Sirko
import pagn.constants as ct
from astropy import units as u


def r_trap(trap_radius):
    return (r"$R_{\rm trap} = \qty{" + f"{trap_radius}" + r"}{\rg}$")


# Use the thesis plot style
plt.style.use("mcfacts.vis.kaila_thesis_figures")

figsize = 469.75502
fraction = 0.49

radius = r"Radius [\unit{\rg}]"
plot_format = "png"

# mcfacts pink
color = "#D47B7C"


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
    parser.add_argument("--fname-log",
                        default="mcfacts.log",
                        type=str, help="log file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    parser.add_argument("--time-cut",
                        default=-1,
                        type=int, help="max AGN lifetime to plot in Myr")
    parser.add_argument("--log-scale",
                        default=0,
                        type=int, help="put y-axes in linear (1) or logscale(0)")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts


def main():
    opts = arg()
    log_data = ReadLog(opts.fname_log)
    data = np.loadtxt(opts.fname_disk, skiprows=1)

    # Get disk Mdot
    disk = Sirko.SirkoAGN(Mbh=log_data["smbh_mass"]*ct.MSun,
                          alpha=log_data["disk_alpha_viscosity"],
                          le=log_data["disk_bh_eddington_ratio"],
                          eps = 0.1)
    mdot_disk = (disk.Mdot*u.kg/u.s).to("Msun/year").value

    # Cut to specified disk lifetime
    if opts.time_cut != -1:
        data = data[data[:, 1] <= (opts.time_cut * 1e6)]

    if opts.log_scale == 0:
        log = ""
    else:
        log = "_log"

    # Sort data by timestep for efficient grouping
    sort_indices = np.argsort(data[:, 1])
    sorted_data = data[sort_indices]

    # Find where timestep changes
    unique_timesteps, inverse_indices, counts = np.unique(sorted_data[:, 1],
                                                          return_inverse=True,
                                                          return_counts=True)

    # Calculate delta values
    mass_delta_values = sorted_data[:, 2] - sorted_data[:, 3]

    # Use reduceat to compute statistics for each group
    cumsum_indices = np.concatenate(([0], np.cumsum(counts[:-1])))

    # Mean calculations
    mass_gain_sums = np.add.reduceat(sorted_data[:, 2], cumsum_indices)
    mass_gain_avg = mass_gain_sums / counts

    mass_lost_sums = np.add.reduceat(sorted_data[:, 3], cumsum_indices)
    mass_lost_avg = mass_lost_sums / counts

    mass_delta_sums = np.add.reduceat(mass_delta_values, cumsum_indices)
    mass_delta_avg = mass_delta_sums / counts

    # Standard deviation calculations
    mass_gain_sq_sums = np.add.reduceat(sorted_data[:, 2]**2, cumsum_indices)
    mass_gain_std = np.sqrt(mass_gain_sq_sums / counts - mass_gain_avg**2)

    mass_lost_sq_sums = np.add.reduceat(sorted_data[:, 3]**2, cumsum_indices)
    mass_lost_std = np.sqrt(mass_lost_sq_sums / counts - mass_lost_avg**2)

    mass_delta_sq_sums = np.add.reduceat(mass_delta_values**2, cumsum_indices)
    mass_delta_std = np.sqrt(mass_delta_sq_sums / counts - mass_delta_avg**2)

    # Transform into Msun/yr. Hardcoding timestep as 10,000 years.
    timestep_duration_yr = 1e4
    mass_gain_avg_rate = mass_gain_avg / timestep_duration_yr
    mass_gain_std_rate = mass_gain_std / timestep_duration_yr
    mass_lost_avg_rate = mass_lost_avg / timestep_duration_yr
    mass_lost_std_rate = mass_lost_std / timestep_duration_yr
    mass_delta_avg_rate = mass_delta_avg / timestep_duration_yr
    mass_delta_std_rate = mass_delta_std / timestep_duration_yr

    print("AVERAGE mass_gain_avg_rate", np.mean(mass_gain_avg_rate))

    timestep_division = 1e6

    # ========================================
    # Mass lost from the disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize, fraction=fraction))

    if opts.log_scale == 1:
        mean_line = np.log10(mass_lost_avg)
        plt.ylabel(r"$\log M_{\rm disk,loss}/\unit{\msun}$")
    else:
        mean_line = mass_lost_avg
        std_under = mass_lost_avg - mass_lost_std
        std_over = mass_lost_avg + mass_lost_std
        plt.ylabel(r"$M_{\rm disk,loss}$ [\unit{\msun}]")
        plt.fill_between(np.unique(data[:, 1])/timestep_division, -(mass_lost_avg - mass_lost_std), -(mass_lost_avg + mass_lost_std), alpha=0.2, label='Standard deviation', color=color, zorder=0, edgecolor=None)

    plt.plot(np.unique(data[:, 1])/timestep_division, -mean_line, label='Mean value', color=color, zorder=10)

    plt.xlabel(r"Time [\unit{\myr}]")

    plt.xticks(np.linspace(0, data[:, 1].max()/timestep_division + 0.01, 6))

    plt.savefig(opts.plots_directory + f"/mass_disk_loss{log}." + plot_format)
    plt.close()

    # ========================================
    # Mass gained to the disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize, fraction=fraction))

    if opts.log_scale == 1:
        mean_line = np.log10(mass_gain_avg)
        plt.ylabel(r"$\log M_{\rm disk, gain} /\unit{\msun}$")
    else:
        mean_line = mass_gain_avg
        std_under = mass_gain_avg - mass_gain_std
        std_over = mass_gain_avg + mass_gain_std
        plt.ylabel(r"$M_{\rm disk, gain}$ [\unit{\msun}]")
        plt.fill_between(np.unique(data[:, 1])/timestep_division, std_under, std_over, alpha=0.2, label='Standard deviation', color=color, edgecolor=None)

    plt.plot(np.unique(data[:, 1])/timestep_division, mean_line, label='Mean value', color=color)

    plt.xlabel(r"Time [\unit{\myr}]")

    plt.xticks(np.linspace(0, data[:, 1].max()/timestep_division + 0.01, 6))

    plt.savefig(opts.plots_directory + f"/mass_disk_gain{log}." + plot_format)
    plt.close()

    # ========================================
    # Mass accretion rate for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize, fraction=fraction))

    if opts.log_scale == 1:
        mean_line = np.log10(mass_gain_avg_rate)
        plt.ylabel(r"$\log \dot{M}_{\rm disk, gain} /\unit{\msun \per \year}$")
    else:
        mean_line = mass_gain_avg_rate
        std_under = mass_gain_avg_rate - mass_gain_std_rate
        std_over = mass_gain_avg_rate + mass_gain_std_rate
        plt.ylabel(r"$\dot{M}_{\rm disk, gain}$ [\unit{\msun \per \year}]")
        plt.fill_between(np.unique(data[:, 1])/timestep_division, std_under, std_over, alpha=0.2, label='Standard deviation', color=color, edgecolor=None)

    plt.plot(np.unique(data[:, 1])/timestep_division, mean_line, label='Mean value', color=color)

    plt.xlabel(r"Time [\unit{\myr}]")

    plt.xticks(np.linspace(0, data[:, 1].max()/timestep_division + 0.01, 6))

    plt.savefig(opts.plots_directory + f"/mdot_disk_gain{log}." + plot_format)
    plt.close()

    # ========================================
    # Mass loss rate for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize, fraction=fraction))

    if opts.log_scale == 1:
        mean_line = np.log10(mass_lost_avg_rate)
        plt.ylabel(r"$\log \dot{M}_{\rm disk, loss} /\unit{\msun \per \year}$")
    else:
        mean_line = mass_lost_avg_rate
        std_under = mass_lost_avg_rate - mass_lost_std_rate
        std_over = mass_lost_avg_rate + mass_lost_std_rate
        plt.ylabel(r"$\dot{M}_{\rm disk, loss}$ [\unit{\msun \per \year}]")
        plt.fill_between(np.unique(data[:, 1])/timestep_division, -std_under, -std_over, alpha=0.2, label='Standard deviation', color=color, edgecolor=None)

    plt.plot(np.unique(data[:, 1])/timestep_division, -mean_line, label='Mean value', color=color)

    plt.xlabel(r"Time [\unit{\myr}]")

    plt.xticks(np.linspace(0, data[:, 1].max()/timestep_division + 0.01, 6))

    plt.savefig(opts.plots_directory + f"/mdot_disk_loss{log}." + plot_format)
    plt.close()

    # ========================================
    # Delta mass for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize, fraction=fraction))

    if opts.log_scale == 1:
        mean_line = np.log10(np.abs(mass_delta_avg))
        plt.ylabel(r"$\log \Delta M_{\rm disk} /\unit{\msun}$")
    else:
        mean_line = mass_delta_avg
        std_under = mass_delta_avg - mass_delta_std
        std_over = mass_delta_avg + mass_delta_std
        plt.ylabel(r"$\Delta M_{\rm disk}$ [\unit{\msun}]")
        plt.fill_between(np.unique(data[:, 1])/timestep_division, std_under, std_over, alpha=0.2, label='Standard deviation', color=color, edgecolor=None)

    plt.plot(np.unique(data[:, 1])/timestep_division, mean_line, label="Mean value", color=color)

    plt.xlabel(r"Time [\unit{\myr}]")

    plt.savefig(opts.plots_directory + f"/mass_disk_delta{log}." + plot_format)
    plt.close()

    # ========================================
    # Delta mdot for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize, fraction=(2/3)))

    if opts.log_scale == 1:
        mean_line = np.log10(mass_delta_avg_rate)
        plt.ylabel(r"$\log \dot{M}_{\rm disk,total} /\unit{\msun \per \year}$")
    else:
        mean_line = -mass_delta_avg_rate
        std_under = -(mass_delta_avg_rate) - (mass_delta_std_rate)
        std_over = -(mass_delta_avg_rate) + (mass_delta_std_rate)
        plt.ylabel(r"$\dot{M}$ [\unit{\msun \per \year}]")
        plt.fill_between(np.unique(data[:, 1])/timestep_division, std_under, std_over, alpha=0.2, color=color, edgecolor=None)

    plt.axhline(mdot_disk, color='k', linestyle="--", label=r"$\dot{M}_{\rm disk}$")
    plt.plot(np.unique(data[:, 1])/timestep_division, mean_line, label=r"$\dot{M}_{\rm stars}$", color=color)

    plt.xlabel(r"Time [\unit{\myr}]")
    plt.legend()

    plt.savefig(opts.plots_directory + f"/mdot_disk_delta{log}." + plot_format)
    plt.close()


######## Execution ########
if __name__ == "__main__":
    main()
