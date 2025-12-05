#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
# Grab those txt files
from mcfacts.vis import plotting
from mcfacts.vis import styles
from mcfacts.outputs.ReadOutputs import ReadLog
from mcfacts.physics.stellar_interpolation import interp_star_params


def r_trap(trap_radius):
    return (r"$R_{\rm trap} = \qty{" + f"{trap_radius}" + r"}{\rg}$")


# Use the thesis plot style
plt.style.use("mcfacts.vis.kaila_thesis_figures")

figsize = 469.75502
fraction = 0.49

radius = r"Radius [\unit{\rg}]"
plot_format = "png"

gen_labels = ['1g', '2g', r'$\geq$3g']
gen_colors = [styles.color_gen1, styles.color_gen2, styles.color_genX]

color_merge = "#4C3329"
color_explode = "#4A8CB0"


######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-directory",
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--fname-stars-final",
                        default="output_stars_population.dat",
                        type=str, help="output_stars file")
    parser.add_argument("--fname-stars-merge",
                        default="output_stars_merged.dat",
                        type=str, help="output merged stars file")
    parser.add_argument("--fname-stars-explode",
                        default="output_stars_exploded.dat",
                        type=str, help="output exploded stars file")
    parser.add_argument("--fname-stars-immortal",
                        default="output_stars_immortal.dat",
                        type=str, help="output immortal stars file")
    parser.add_argument("--fname-log",
                        default="mcfacts.log",
                        type=str, help="log file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts


def make_gen_masks_single(table, col):
    """Create masks for retrieving different sets of a stellar population

    Parameters
    ----------
    table : numpy.ndarray
        Data
    col : int
        Column index where generation information is stored
    """

    # Column with generation data
    gen_obj = table[:, col]

    # Masks for hierarchical generations
    # g1 : 1g objects
    # g2 : 2g objects
    # g3 : >=3g
    # Pipe operator (|) = logical OR. (&)= logical AND.

    g1_mask = (gen_obj == 1)
    g2_mask = (gen_obj == 2)
    gX_mask = (gen_obj >= 3)

    return (g1_mask, g2_mask, gX_mask)


def make_gen_masks_binary(table, col1, col2):
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


def make_immortal_mask(table, col, immortal_mass):
    """Create mask for retrieving immortal stars

    Parameters
    ----------
    table : numpy.ndarray
        Data
    col : int
        Column index where mass information is stored
    """

    # Column with mass info
    mass_obj = table[:, col]

    mask = (mass_obj == immortal_mass)

    return mask


def get_count_by_param(table, param_arr):
    """Get number of objects per param value (usually timestep or galaxy)

    Parameters
    ----------
    table : numpy.ndarray
        Data
    param_arr : numpy.ndarray
        All possible values of the parameter to count by
        (timesteps 0 to N, galaxies 0 to M, etc.)
    """
    params, counts = np.unique(table, return_counts=True)
    counts_arr = np.zeros(param_arr.shape)
    counts_arr[np.searchsorted(param_arr, params)] = counts
    return counts_arr


def main():
    opts = arg()
    log_data = ReadLog(opts.fname_log)
    immortal_mass = log_data["disk_star_initial_mass_cutoff"]
    trap_radius = log_data["disk_radius_trap"]
    disk_radius_outer = log_data["disk_radius_outer"]
    timestep = log_data["timestep_duration_yr"]
    timestep_arr = np.arange(0, log_data["timestep_duration_yr"] * log_data["timestep_num"], log_data["timestep_duration_yr"])

    stars_final = np.loadtxt(opts.fname_stars_final, skiprows=2)
    stars_merge = np.loadtxt(opts.fname_stars_merge, skiprows=1)
    stars_explode = np.loadtxt(opts.fname_stars_explode, skiprows=1)
    stars_immortal = np.loadtxt(opts.fname_stars_immortal, skiprows=1)

    # Take out stars that were already immortal (merged w immortal star)
    stars_immortal = stars_immortal[stars_immortal[:, 13] != immortal_mass]

    _, stars_merge_logL, stars_merge_logT = interp_star_params(stars_merge[:, 3])
    _, stars_explode_logL, stars_explode_lotT = interp_star_params(stars_explode[:, 3])

    # Get stars that become immortal through merger
    stars_merge_immortal = stars_merge[(stars_merge[:, 3] == log_data["disk_star_initial_mass_cutoff"]) &
                                       (stars_merge[:, 8] != log_data["disk_star_initial_mass_cutoff"]) &
                                       (stars_merge[:, 9] != log_data["disk_star_initial_mass_cutoff"])]

    final_g1_mask, final_g2_mask, final_gX_mask = make_gen_masks_single(stars_final, 6)
    explode_g1_mask, explode_g2_mask, explode_gX_mask = make_gen_masks_single(stars_explode, 6)
    merge_g1_mask, merge_g2_mask, merge_gX_mask = make_gen_masks_binary(stars_merge, 10, 11)
    immortal_g1_mask, immortal_g2_mask, immortal_gX_mask = make_gen_masks_single(stars_immortal, 6)
    merge_imm_g1_mask, merge_imm_g2_mask, merge_imm_gX_mask = make_gen_masks_single(stars_merge_immortal, 6)

    final_immortal_mask = make_immortal_mask(stars_final, 3, immortal_mass)
    explode_immortal_mask = make_immortal_mask(stars_explode, 3, immortal_mass)
    merge_immortal_mask = make_immortal_mask(stars_merge, 3, immortal_mass)

    # Ensure no union between sets
    assert all(final_g1_mask & final_g2_mask) == 0
    assert all(final_g1_mask & final_gX_mask) == 0
    assert all(final_g2_mask & final_gX_mask) == 0
    assert all(explode_g1_mask & explode_g2_mask) == 0
    assert all(explode_g1_mask & explode_gX_mask) == 0
    assert all(explode_g2_mask & explode_gX_mask) == 0
    assert all(merge_g1_mask & merge_g2_mask) == 0
    assert all(merge_g1_mask & merge_gX_mask) == 0
    assert all(merge_g2_mask & merge_gX_mask) == 0
    assert all(immortal_g1_mask & immortal_g2_mask) == 0
    assert all(immortal_g1_mask & immortal_gX_mask) == 0
    assert all(immortal_g2_mask & immortal_gX_mask) == 0

    # Ensure no elements are missed
    assert all(final_g1_mask | final_g2_mask | final_gX_mask) == 1
    assert all(explode_g1_mask | explode_g2_mask | explode_gX_mask) == 1
    assert all(merge_g1_mask | merge_g2_mask | merge_gX_mask) == 1
    assert all(immortal_g1_mask | immortal_g2_mask | immortal_gX_mask) == 1

    # ========================================
    # Bounds for rate plots for slides
    # ========================================

    # immortal stars
    gen1_time = stars_immortal[:, 1][immortal_g1_mask]
    gen2_time = stars_immortal[:, 1][immortal_g2_mask]
    genX_time = stars_immortal[:, 1][immortal_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))
    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])
    ax.set_yscale("log")

    rate_imm_ymin, rate_imm_ymax = ax.get_ylim()
    plt.close()

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_merge[:, 1][merge_g1_mask]
    gen2_time = stars_merge[:, 1][merge_g2_mask]
    genX_time = stars_merge[:, 1][merge_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_yscale("log")
    rate_merge_ymin, rate_merge_ymax = ax.get_ylim()
    plt.close()

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_explode[:, 1][explode_g1_mask]
    gen2_time = stars_explode[:, 1][explode_g2_mask]
    genX_time = stars_explode[:, 1][explode_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_yscale("log")
    rate_exp_ymin, rate_exp_ymax = ax.get_ylim()
    plt.close()

    rate_ymin = np.min([rate_imm_ymin, rate_merge_ymin, rate_exp_ymin])
    rate_ymax = np.max([rate_imm_ymax, rate_merge_ymax, rate_exp_ymax])


    # ========================================
    # Final state radius distribution for immortal stars
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction))

    # Separate generational subpopulations
    final_imm_gen1_orb_a = stars_final[:, 2][final_g1_mask & final_immortal_mask]
    final_imm_gen2_orb_a = stars_final[:, 2][final_g2_mask & final_immortal_mask]
    final_imm_genX_orb_a = stars_final[:, 2][final_gX_mask & final_immortal_mask]

    bins = np.logspace(np.log10(stars_final[:, 2][final_immortal_mask].min()),
                       np.log10(stars_final[:, 2][final_immortal_mask].max()), 50)

    hist_data = [final_imm_gen1_orb_a, final_imm_gen2_orb_a, final_imm_genX_orb_a]

    ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)
    ax.axvline(trap_radius, color="k", linestyle="--", label=r_trap(int(np.rint(trap_radius))))

    ax.legend(title=r"$t=\tau_{\rm AGN}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Number")
    ax.set_xlabel(r"$R_{\rm final}$ [\unit{\rg}]")

    imm_radius_dist_ymin, imm_radius_dist_ymax = ax.get_ylim()

    plt.savefig(opts.plots_directory + "/star_final_immortal_orba_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars radius distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_radius = stars_immortal[:, 2][immortal_g1_mask]
    gen2_radius = stars_immortal[:, 2][immortal_g2_mask]
    genX_radius = stars_immortal[:, 2][immortal_gX_mask]

    bins = np.logspace(np.log10(stars_immortal[:, 2].min()), np.log10(stars_immortal[:, 2].max()), 50)
    hist_data = [gen1_radius, gen2_radius, genX_radius]

    ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)
    ax.axvline(trap_radius, color="k", linestyle="--", label=r_trap(int(np.rint(trap_radius))))

    ax.set_xlabel(radius)
    ax.set_ylabel("Number")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(imm_radius_dist_ymin, imm_radius_dist_ymax)
    ax.legend()

    plt.savefig(opts.plots_directory + "/star_immortal_radius_dist." + plot_format)

    # ========================================
    # Immortal stars time distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_immortal[:, 1][immortal_g1_mask]
    gen2_time = stars_immortal[:, 1][immortal_g2_mask]
    genX_time = stars_immortal[:, 1][immortal_gX_mask]

    #bins = np.linspace(0, (stars_final[:, 1].max() + timestep)/1e6, 50)
    rwidth = 0.8
    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)

    hist_gen1, bin_edges = np.histogram(gen1_time/1e6, bins=bins)
    hist_gen2, _ = np.histogram(gen2_time/1e6, bins=bins)
    hist_genX, _ = np.histogram(genX_time/1e6, bins=bins)

    hist_gen1 = hist_gen1/log_data["galaxy_num"]
    hist_gen2 = hist_gen2/log_data["galaxy_num"]
    hist_genX = hist_genX/log_data["galaxy_num"]

    bin_centers = bin_edges[:-1]  # Use left edges since align="left"
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin

    # Create the stacked bar plot
    ax.bar(bin_centers, hist_gen1, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[0], label=gen_labels[0])
    ax.bar(bin_centers, hist_gen2, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[1], bottom=hist_gen1, label=gen_labels[1])
    ax.bar(bin_centers, hist_genX, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[2], bottom=hist_gen1 + hist_gen2, label=gen_labels[2])

    #hist_data = [gen1_time/1e6, gen2_time/1e6, genX_time/1e6]

    #ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel("Number")
    ax.legend()
    #ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_immortal_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal accretion stars rate
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_immortal[:, 1][immortal_g1_mask]
    gen2_time = stars_immortal[:, 1][immortal_g2_mask]
    genX_time = stars_immortal[:, 1][immortal_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel(r"$\dot{N}$ [\unit{\per \year}]")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    #ax.set_ylim(1e-10, 1e-1)

    plt.savefig(opts.plots_directory + "/star_immortal_rate." + plot_format)

    ax.set_ylim(rate_ymin, rate_ymax)
    ax.set_ylabel(r"$\dot{N}_{\rm imm}$ [\unit{\per \year}]")
    plt.savefig(opts.plots_directory + "/star_immortal_rate_slides." + plot_format)

    plt.close()

    # ========================================
    # Immortal stars through merger time distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_merge_immortal[:, 1][merge_imm_g1_mask]
    gen2_time = stars_merge_immortal[:, 1][merge_imm_g2_mask]
    genX_time = stars_merge_immortal[:, 1][merge_imm_gX_mask]

    #bins = np.linspace(0, (stars_final[:, 1].max() + timestep)/1e6, 50)
    rwidth = 0.8
    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)

    hist_gen1, bin_edges = np.histogram(gen1_time/1e6, bins=bins)
    hist_gen2, _ = np.histogram(gen2_time/1e6, bins=bins)
    hist_genX, _ = np.histogram(genX_time/1e6, bins=bins)

    hist_gen1 = hist_gen1/log_data["galaxy_num"]
    hist_gen2 = hist_gen2/log_data["galaxy_num"]
    hist_genX = hist_genX/log_data["galaxy_num"]

    bin_centers = bin_edges[:-1]  # Use left edges since align="left"
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin

    # Create the stacked bar plot
    ax.bar(bin_centers, hist_gen1, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[0], label=gen_labels[0])
    ax.bar(bin_centers, hist_gen2, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[1], bottom=hist_gen1, label=gen_labels[1])
    ax.bar(bin_centers, hist_genX, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[2], bottom=hist_gen1 + hist_gen2, label=gen_labels[2])

    #hist_data = [gen1_time/1e6, gen2_time/1e6, genX_time/1e6]

    #ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel("Number")
    ax.legend()
    #ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_immortal_merged_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial mass distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_mass_initial = stars_immortal[:, 13][immortal_g1_mask]
    gen2_mass_initial = stars_immortal[:, 13][immortal_g2_mask]
    genX_mass_initial = stars_immortal[:, 13][immortal_gX_mask]

    bins = np.linspace(0, stars_immortal[stars_immortal[:, 13] != immortal_mass][:, 13].max(), 30)
    hist_data = [gen1_mass_initial, gen2_mass_initial, genX_mass_initial]

    ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"$M_{\rm initial}$ [\unit{\myr}]")
    ax.set_ylabel("Number")
    ax.legend()
    ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_immortal_mass_initial_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial radius distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_orb_a_initial = stars_immortal[:, 14][immortal_g1_mask]
    gen2_orb_a_initial = stars_immortal[:, 14][immortal_g2_mask]
    genX_orb_a_initial = stars_immortal[:, 14][immortal_gX_mask]

    bins = np.logspace(np.log10(stars_immortal[stars_immortal[:, 13] != immortal_mass][:, 14].min()), np.log10(stars_immortal[stars_immortal[:, 13] != immortal_mass][:, 14].max()), 50)
    hist_data = [gen1_orb_a_initial, gen2_orb_a_initial, genX_orb_a_initial]

    ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.axvline(trap_radius, color="k", linestyle="--", label=r_trap(int(np.rint(trap_radius))))

    ax.set_xlabel(r"$R_{\rm initial}$ [\unit{\rg}]")
    ax.set_ylabel("Number")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(imm_radius_dist_ymin, imm_radius_dist_ymax)
    ax.legend(loc="lower left")

    plt.savefig(opts.plots_directory + "/star_immortal_orba_initial_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial mass vs initial radius
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    ax.scatter(gen1_orb_a_initial, gen1_mass_initial,
               s=styles.markersize_gen1,
               marker=styles.marker_gen1,
               edgecolor=styles.color_gen1,
               facecolors="None",
               alpha=styles.markeralpha_gen1,
               label=gen_labels[0])

    ax.scatter(gen2_orb_a_initial, gen2_mass_initial,
               s=styles.markersize_gen2,
               marker=styles.marker_gen2,
               edgecolor=styles.color_gen2,
               facecolors="None",
               alpha=styles.markeralpha_gen2,
               label=gen_labels[1])

    ax.scatter(genX_orb_a_initial, genX_mass_initial,
               s=styles.markersize_genX,
               marker=styles.marker_genX,
               edgecolor=styles.color_genX,
               facecolors="None",
               alpha=styles.markeralpha_genX,
               label=gen_labels[2])

    ax.axvline(trap_radius, color="k", linestyle="--", label=r_trap(int(np.rint(trap_radius))))

    ax.set_xlabel(r"$R_{\rm initial}$ [\unit{\rg}]")
    ax.set_ylabel(r"$M_{\rm initial}$ [\unit{\msun}]")
    ax.legend(loc="upper left")
    ax.set_xscale("log")

    plt.savefig(opts.plots_directory + "/star_immortal_mass_orba_initial." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial radius vs current radius
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    ax.scatter(gen1_radius, gen1_orb_a_initial,
               s=styles.markersize_gen1,
               marker=styles.marker_gen1,
               edgecolor=styles.color_gen1,
               facecolors="None",
               alpha=styles.markeralpha_gen1,
               label=gen_labels[0])

    ax.scatter(gen2_radius, gen2_orb_a_initial,
               s=styles.markersize_gen2,
               marker=styles.marker_gen2,
               edgecolor=styles.color_gen2,
               facecolors="None",
               alpha=styles.markeralpha_gen2,
               label=gen_labels[1])

    ax.scatter(genX_radius, genX_orb_a_initial,
               s=styles.markersize_genX,
               marker=styles.marker_genX,
               edgecolor=styles.color_genX,
               facecolors="None",
               alpha=styles.markeralpha_genX,
               label=gen_labels[2])

    ax.axvline(trap_radius, color="k", linestyle="--", label=r_trap(int(np.rint(trap_radius))))
    ax.axhline(trap_radius, color="k", linestyle="--")

    ax.set_ylabel(r"$R_{\rm initial}$ [\unit{\rg}]")
    ax.set_xlabel(radius)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower right")

    plt.savefig(opts.plots_directory + "/star_immortal_orba_initial_current." + plot_format)
    plt.close()


    # ========================================
    #  Mass vs Radius for exploded stars
    # ========================================

    # Separate generational subpopulations
    ex_gen1_orb_a = stars_explode[:, 2][explode_g1_mask]
    ex_gen2_orb_a = stars_explode[:, 2][explode_g2_mask]
    ex_genX_orb_a = stars_explode[:, 2][explode_gX_mask]
    ex_gen1_mass = stars_explode[:, 3][explode_g1_mask]
    ex_gen2_mass = stars_explode[:, 3][explode_g2_mask]
    ex_genX_mass = stars_explode[:, 3][explode_gX_mask]

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    ax.scatter(ex_gen1_orb_a, ex_gen1_mass,
               s=styles.markersize_gen1,
               marker=styles.marker_gen1,
               edgecolor=styles.color_gen1,
               facecolors="none",
               alpha=styles.markeralpha_gen1,
               label=gen_labels[0]
               )

    ax.scatter(ex_gen2_orb_a, ex_gen2_mass,
               s=styles.markersize_gen2,
               marker=styles.marker_gen2,
               edgecolor=styles.color_gen2,
               facecolors="none",
               alpha=styles.markeralpha_gen2,
               label=gen_labels[1]
               )

    ax.scatter(ex_genX_orb_a, ex_genX_mass,
               s=styles.markersize_genX,
               marker=styles.marker_genX,
               edgecolor=styles.color_genX,
               facecolors="none",
               alpha=styles.markeralpha_genX,
               label=gen_labels[2]
               )

    ax.axvline(trap_radius, color='k', linestyle='--', zorder=0,
               label=r_trap(int(np.rint(trap_radius))))

    ax.set_ylabel(r'Mass [\unit{\msun}]')
    ax.set_xlabel(radius)
    ax.set_xscale('log')

    ax.legend(loc="upper left")

    ax.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_exploded_mass_v_radius." + plot_format)
    plt.close()

    # ========================================
    #  Mass vs Radius for merged stars
    # ========================================

    # Separate generational subpopulations
    merged_gen1_orb_a = stars_merge[:, 2][merge_g1_mask]
    merged_gen2_orb_a = stars_merge[:, 2][merge_g2_mask]
    merged_genX_orb_a = stars_merge[:, 2][merge_gX_mask]
    merged_gen1_mass = stars_merge[:, 3][merge_g1_mask]
    merged_gen2_mass = stars_merge[:, 3][merge_g2_mask]
    merged_genX_mass = stars_merge[:, 3][merge_gX_mask]

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    ax.scatter(merged_gen1_orb_a, merged_gen1_mass,
               s=styles.markersize_gen1,
               marker=styles.marker_gen1,
               edgecolor=styles.color_gen1,
               facecolors="none",
               alpha=styles.markeralpha_gen1,
               label=gen_labels[0]
               )

    ax.scatter(merged_gen2_orb_a, merged_gen2_mass,
               s=styles.markersize_gen2,
               marker=styles.marker_gen2,
               edgecolor=styles.color_gen2,
               facecolors="none",
               alpha=styles.markeralpha_gen2,
               label=gen_labels[1]
               )

    ax.scatter(merged_genX_orb_a, merged_genX_mass,
               s=styles.markersize_genX,
               marker=styles.marker_genX,
               edgecolor=styles.color_genX,
               facecolors="none",
               alpha=styles.markeralpha_genX,
               label=gen_labels[2]
               )

    ax.axvline(trap_radius, color='k', linestyle='--', zorder=0,
               label=r_trap(int(np.rint(trap_radius))))

    ax.set_ylabel(r'Mass [\unit{\msun}]')
    ax.set_xlabel(radius)
    ax.set_xscale('log')

    ax.legend(loc="upper left")

    ax.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_merged_mass_v_radius." + plot_format)
    plt.close()

    # ========================================
    # Exploded stars luminosity distribution
    # ========================================

    # Separate generational subpopulations
    gen1_lum = stars_explode_logL[explode_g1_mask]
    gen2_lum = stars_explode_logL[explode_g2_mask]
    genX_lum = stars_explode_logL[explode_gX_mask]

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    bins = np.linspace(stars_explode_logL.min(), stars_explode_logL.max(), 50)
    hist_data = [gen1_lum, gen2_lum, genX_lum]

    ax.hist(hist_data, bins=bins, align='left', color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"$\log L/L_\odot$")
    ax.set_ylabel("Number")
    ax.legend()
    ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_exploded_lum_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged stars luminosity distribution
    # ========================================

    # Separate generational subpopulations
    gen1_lum = stars_merge_logL[merge_g1_mask]
    gen2_lum = stars_merge_logL[merge_g2_mask]
    genX_lum = stars_merge_logL[merge_gX_mask]

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    bins = np.linspace(stars_merge_logL.min(), stars_merge_logL.max(), 50)
    hist_data = [gen1_lum, gen2_lum, genX_lum]

    ax.hist(hist_data, bins=bins, align='left', color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"$\log L/L_\odot$")
    ax.set_ylabel("Number")
    ax.legend()
    ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_merged_lum_dist." + plot_format)
    plt.close()

    # ========================================
    # Number of Mergers vs Mass
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    bins = np.linspace(stars_merge[:, 3].min(), stars_merge[:, 3].max(), 50)
    hist_data = [merged_gen1_mass, merged_gen2_mass, merged_genX_mass]

    ax.hist(hist_data, bins=bins, color=gen_colors, align="left", alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_ylabel("Number of mergers")
    ax.set_xlabel(r"Mass [\unit{\msun}]")
    ax.set_yscale("log")

    ax.legend()

    plt.savefig(opts.plots_directory + r"/star_merged_mass_dist." + plot_format)

    # ========================================
    # Merged stars m1 m2 2d histogram
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction))

    bins = np.linspace(0, 300, 15)
    _, _, _, cm = ax.hist2d(stars_merge[:, 8], stars_merge[:, 9], bins=bins, cmap="binary", norm="log")
    ax.set_aspect("equal")

    cbar = fig.colorbar(cm)
    cbar.set_label("Number")

    ax.set_xlabel(r"$M_1$ [\unit{\msun}]")
    ax.set_ylabel(r"$M_2$ [\unit{\msun}]")

    plt.savefig(opts.plots_directory + "/star_merged_m1m2_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged stars time distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction))

    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)
    rwidth = 0.8

    gen1_time = stars_merge[:, 1][merge_g1_mask]
    gen2_time = stars_merge[:, 1][merge_g2_mask]
    genX_time = stars_merge[:, 1][merge_gX_mask]

    hist_gen1, bin_edges = np.histogram(gen1_time/1e6, bins=bins)
    hist_gen2, _ = np.histogram(gen2_time/1e6, bins=bins)
    hist_genX, _ = np.histogram(genX_time/1e6, bins=bins)

    hist_gen1 = hist_gen1/log_data["galaxy_num"]
    hist_gen2 = hist_gen2/log_data["galaxy_num"]
    hist_genX = hist_genX/log_data["galaxy_num"]

    bin_centers = bin_edges[:-1]  # Use left edges since align="left"
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin

    # Create the stacked bar plot
    ax.bar(bin_centers, hist_gen1, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[0], label=gen_labels[0])
    ax.bar(bin_centers, hist_gen2, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[1], bottom=hist_gen1, label=gen_labels[1])
    ax.bar(bin_centers, hist_genX, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[2], bottom=hist_gen1 + hist_gen2, label=gen_labels[2])

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel("Number")

    ax.legend()

    plt.savefig(opts.plots_directory + "/star_merged_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Exploded stars time distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction))

    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)
    rwidth = 0.8

    gen1_time = stars_explode[:, 1][explode_g1_mask]
    gen2_time = stars_explode[:, 1][explode_g2_mask]
    genX_time = stars_explode[:, 1][explode_gX_mask]

    hist_gen1, bin_edges = np.histogram(gen1_time/1e6, bins=bins)
    hist_gen2, _ = np.histogram(gen2_time/1e6, bins=bins)
    hist_genX, _ = np.histogram(genX_time/1e6, bins=bins)

    hist_gen1 = hist_gen1/log_data["galaxy_num"]
    hist_gen2 = hist_gen2/log_data["galaxy_num"]
    hist_genX = hist_genX/log_data["galaxy_num"]

    bin_centers = bin_edges[:-1]  # Use left edges since align="left"
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin

    # Create the stacked bar plot
    ax.bar(bin_centers, hist_gen1, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[0], label=gen_labels[0])
    ax.bar(bin_centers, hist_gen2, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[1], bottom=hist_gen1, label=gen_labels[1])
    ax.bar(bin_centers, hist_genX, width=bin_width*rwidth, align='edge', alpha=0.9, color=gen_colors[2], bottom=hist_gen1 + hist_gen2, label=gen_labels[2])

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel("Number")

    ax.legend()

    plt.savefig(opts.plots_directory + "/star_exploded_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged and exploded stars time distribution
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction))

    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)
    rwidth = 0.8

    hist_merge, bin_edges = np.histogram(stars_merge[:, 1]/1e6, bins=bins)
    hist_explode, _ = np.histogram(stars_explode[:, 1]/1e6, bins=bins)

    hist_merge = hist_merge/log_data["galaxy_num"]
    hist_explode = hist_explode/log_data["galaxy_num"]

    bin_centers = bin_edges[:-1]  # Use left edges since align="left"
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin

    # Create the stacked bar plot
    ax.bar(bin_centers, hist_merge, width=bin_width*rwidth, alpha=0.9, align='edge', color=color_merge, label='Merged stars')
    ax.bar(bin_centers, hist_explode, width=bin_width*rwidth, alpha=0.9, align='edge', color=color_explode, bottom=hist_merge, label='Exploded stars')

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel("Number")

    ax.legend()

    plt.savefig(opts.plots_directory + "/star_merged_exploded_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged stars mass vs time
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction))

    merged_gen1_mass = stars_merge[:, 3][merge_g1_mask]
    merged_gen2_mass = stars_merge[:, 3][merge_g2_mask]
    merged_genX_mass = stars_merge[:, 3][merge_gX_mask]
    gen1_time = stars_merge[:, 1][merge_g1_mask]/1e6
    gen2_time = stars_merge[:, 1][merge_g2_mask]/1e6
    genX_time = stars_merge[:, 1][merge_gX_mask]/1e6

    ax.scatter(gen1_time, merged_gen1_mass,
               s=styles.markersize_gen1,
               marker=styles.marker_gen1,
               edgecolor=styles.color_gen1,
               facecolors="none",
               alpha=styles.markeralpha_gen1,
               label=gen_labels[0]
               )

    ax.scatter(gen2_time, merged_gen2_mass,
               s=styles.markersize_gen2,
               marker=styles.marker_gen2,
               edgecolor=styles.color_gen2,
               facecolors="none",
               alpha=styles.markeralpha_gen2,
               label=gen_labels[1]
               )

    ax.scatter(genX_time, merged_genX_mass,
               s=styles.markersize_genX,
               marker=styles.marker_genX,
               edgecolor=styles.color_genX,
               facecolors="none",
               alpha=styles.markeralpha_genX,
               label=gen_labels[2]
               )

    ax.legend(loc="upper left")

    ax.set_ylabel(r"Mass [\unit{\msun}]")
    ax.set_xlabel(r"Time [\unit{\myr}]")

    plt.savefig(opts.plots_directory + "/star_merged_mass_v_time." + plot_format)

    # ========================================
    # Exploded stars mass vs time
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction))

    explode_gen1_mass = stars_explode[:, 3][explode_g1_mask]
    explode_gen2_mass = stars_explode[:, 3][explode_g2_mask]
    explode_genX_mass = stars_explode[:, 3][explode_gX_mask]
    gen1_time = stars_explode[:, 1][explode_g1_mask]/1e6
    gen2_time = stars_explode[:, 1][explode_g2_mask]/1e6
    genX_time = stars_explode[:, 1][explode_gX_mask]/1e6

    ax.scatter(gen1_time, explode_gen1_mass,
               s=styles.markersize_gen1,
               marker=styles.marker_gen1,
               edgecolor=styles.color_gen1,
               facecolors="none",
               alpha=styles.markeralpha_gen1,
               label=gen_labels[0]
               )

    ax.scatter(gen2_time, explode_gen2_mass,
               s=styles.markersize_gen2,
               marker=styles.marker_gen2,
               edgecolor=styles.color_gen2,
               facecolors="none",
               alpha=styles.markeralpha_gen2,
               label=gen_labels[1]
               )

    ax.scatter(genX_time, explode_genX_mass,
               s=styles.markersize_genX,
               marker=styles.marker_genX,
               edgecolor=styles.color_genX,
               facecolors="none",
               alpha=styles.markeralpha_genX,
               label=gen_labels[2]
               )

    ax.legend(loc="upper left")

    ax.set_ylabel(r"Mass [\unit{\msun}]")
    ax.set_xlabel(r"Time [\unit{\myr}]")

    plt.savefig(opts.plots_directory + "/star_exploded_mass_v_time." + plot_format)

    # ========================================
    # Merged and exploded stars rate
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_merge[:, 1][merge_g1_mask]
    gen2_time = stars_merge[:, 1][merge_g2_mask]
    genX_time = stars_merge[:, 1][merge_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    # exploded stars
    gen1_time = stars_explode[:, 1][explode_g1_mask]
    gen2_time = stars_explode[:, 1][explode_g2_mask]
    genX_time = stars_explode[:, 1][explode_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    ax.legend(loc="lower right")

    ls = "-"#(0, (1, 1))
    lw = 0.7
    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total", linestyle=ls, linewidth=lw)
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0], linestyle=ls, linewidth=lw)
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1], linestyle=ls, linewidth=lw)
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2], linestyle=ls, linewidth=lw)


    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel(r"$\dot{N}$ [\unit{\per \year}]")
    ax.set_yscale("log")
    #ax.set_ylim(1e-10, 1e-1)

    plt.savefig(opts.plots_directory + "/star_merged_exploded_rate." + plot_format)
    plt.close()


    # ========================================
    # Merged stars rate
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_merge[:, 1][merge_g1_mask]
    gen2_time = stars_merge[:, 1][merge_g2_mask]
    genX_time = stars_merge[:, 1][merge_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel(r"$\dot{N}_{\rm merge}$ [\unit{\per \year}]")
    ax.set_yscale("log")
    #ax.set_ylim(1e-10, 1e-1)
    plt.legend(loc="lower right")

    plt.savefig(opts.plots_directory + "/star_merged_rate." + plot_format)
    ax.set_ylim(rate_ymin, rate_ymax)
    plt.savefig(opts.plots_directory + "/star_merged_rate_slides." + plot_format)

    plt.close()

    # ========================================
    # Exploded stars rate
    # ========================================

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_explode[:, 1][explode_g1_mask]
    gen2_time = stars_explode[:, 1][explode_g2_mask]
    genX_time = stars_explode[:, 1][explode_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_xlabel(r"Time [\unit{\myr}]")
    ax.set_ylabel(r"$\dot{N}_{\rm explode}$ [\unit{\per \year}]")
    ax.legend(loc="lower right")
    ax.set_yscale("log")
    #ax.set_ylim(1e-10, 1e-1)

    plt.savefig(opts.plots_directory + "/star_exploded_rate." + plot_format)
    ax.set_ylim(rate_ymin, rate_ymax)
    ax.legend(loc="upper center")
    plt.savefig(opts.plots_directory + "/star_exploded_rate_slides." + plot_format)

    plt.close()


######## Execution ########
if __name__ == "__main__":
    main()
