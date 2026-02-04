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
    return (r"$R_{\rm trap} =" + f"{trap_radius}" + r"\,\mathrm{r}_\mathrm{g}$")


# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

figsize = "apj_col"
fraction = 0.49

radius = r"Radius [$\mathrm{r}_\mathrm{g}$]"
plot_format = "png"

gen_labels = ['1g', '2g', r'$\geq$3g']
gen_colors = [styles.color_gen1, styles.color_gen2, styles.color_genX]

color_merge = "#4C3329"
color_disrupted = "#4A8CB0"


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
    parser.add_argument("--fname-stars-merged",
                        default="output_stars_merged.dat",
                        type=str, help="output merged stars file")
    parser.add_argument("--fname-stars-disrupted",
                        default="output_stars_disrupted.dat",
                        type=str, help="output disrupted stars file")
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
    stars_merged = np.loadtxt(opts.fname_stars_merged, skiprows=1)
    stars_disrupted = np.loadtxt(opts.fname_stars_disrupted, skiprows=1)
    stars_immortal = np.loadtxt(opts.fname_stars_immortal, skiprows=1)

    # Take out stars that were already immortal (merged w immortal star)
    stars_immortal = stars_immortal[stars_immortal[:, 13] != immortal_mass]

    _, stars_merged_logL, stars_merged_logT = interp_star_params(stars_merged[:, 3])
    _, stars_disrupted_logL, stars_disrupted_lotT = interp_star_params(stars_disrupted[:, 3])

    # Get stars that become immortal through merger
    stars_merged_immortal = stars_merged[(stars_merged[:, 3] == log_data["disk_star_initial_mass_cutoff"]) &
                                       (stars_merged[:, 8] != log_data["disk_star_initial_mass_cutoff"]) &
                                       (stars_merged[:, 9] != log_data["disk_star_initial_mass_cutoff"])]

    final_g1_mask, final_g2_mask, final_gX_mask = make_gen_masks_single(stars_final, 6)
    disrupted_g1_mask, disrupted_g2_mask, disrupted_gX_mask = make_gen_masks_single(stars_disrupted, 6)
    merged_g1_mask, merged_g2_mask, merged_gX_mask = make_gen_masks_binary(stars_merged, 10, 11)
    immortal_g1_mask, immortal_g2_mask, immortal_gX_mask = make_gen_masks_single(stars_immortal, 6)
    merged_imm_g1_mask, merged_imm_g2_mask, merged_imm_gX_mask = make_gen_masks_single(stars_merged_immortal, 6)

    final_immortal_mask = make_immortal_mask(stars_final, 3, immortal_mass)
    disrupted_immortal_mask = make_immortal_mask(stars_disrupted, 3, immortal_mass)
    merged_immortal_mask = make_immortal_mask(stars_merged, 3, immortal_mass)

    # Ensure no union between sets
    assert all(final_g1_mask & final_g2_mask) == 0
    assert all(final_g1_mask & final_gX_mask) == 0
    assert all(final_g2_mask & final_gX_mask) == 0
    assert all(disrupted_g1_mask & disrupted_g2_mask) == 0
    assert all(disrupted_g1_mask & disrupted_gX_mask) == 0
    assert all(disrupted_g2_mask & disrupted_gX_mask) == 0
    assert all(merged_g1_mask & merged_g2_mask) == 0
    assert all(merged_g1_mask & merged_gX_mask) == 0
    assert all(merged_g2_mask & merged_gX_mask) == 0
    assert all(immortal_g1_mask & immortal_g2_mask) == 0
    assert all(immortal_g1_mask & immortal_gX_mask) == 0
    assert all(immortal_g2_mask & immortal_gX_mask) == 0

    # Ensure no elements are missed
    assert all(final_g1_mask | final_g2_mask | final_gX_mask) == 1
    assert all(disrupted_g1_mask | disrupted_g2_mask | disrupted_gX_mask) == 1
    assert all(merged_g1_mask | merged_g2_mask | merged_gX_mask) == 1
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
    gen1_time = stars_merged[:, 1][merged_g1_mask]
    gen2_time = stars_merged[:, 1][merged_g2_mask]
    genX_time = stars_merged[:, 1][merged_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_yscale("log")
    rate_merged_ymin, rate_merged_ymax = ax.get_ylim()
    plt.close()

    fig, ax = plt.subplots(figsize=plotting.set_size(figsize, fraction=fraction))

    # Separate generational subpopulations
    gen1_time = stars_disrupted[:, 1][disrupted_g1_mask]
    gen2_time = stars_disrupted[:, 1][disrupted_g2_mask]
    genX_time = stars_disrupted[:, 1][disrupted_gX_mask]

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

    rate_ymin = np.min([rate_imm_ymin, rate_merged_ymin, rate_exp_ymin])
    rate_ymax = np.max([rate_imm_ymax, rate_merged_ymax, rate_exp_ymax])


    # ========================================
    # Final state radius distribution for immortal stars
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    # Separate generational subpopulations
    final_imm_gen1_orb_a = stars_final[:, 2][final_g1_mask & final_immortal_mask]
    final_imm_gen2_orb_a = stars_final[:, 2][final_g2_mask & final_immortal_mask]
    final_imm_genX_orb_a = stars_final[:, 2][final_gX_mask & final_immortal_mask]

    bins = np.logspace(np.log10(stars_final[:, 2][final_immortal_mask].min()),
                       np.log10(stars_final[:, 2][final_immortal_mask].max()), 50)

    hist_data = [final_imm_gen1_orb_a, final_imm_gen2_orb_a, final_imm_genX_orb_a]

    ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)
    ax.axvline(trap_radius, color="k", linestyle="--", label=r_trap(int(np.rint(trap_radius))))

    if figsize == 'apj_col':
        plt.legend(fontsize=6, title=r"$t=\tau_{\rm AGN}$")
    elif figsize == 'apj_page':
        plt.legend(title=r"$t=\tau_{\rm AGN}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Number")
    ax.set_xlabel(r"$R_{\rm final}$ [$\mathrm{r}_\mathrm{g}$]")

    imm_radius_dist_ymin, imm_radius_dist_ymax = ax.get_ylim()

    plt.savefig(opts.plots_directory + "/star_final_immortal_orba_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars radius distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

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
    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()
    plt.savefig(opts.plots_directory + "/star_immortal_radius_dist." + plot_format)

    # ========================================
    # Immortal stars time distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

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

    ax.set_xlabel(r"Time [Myr]")
    ax.set_ylabel("Number")
    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()

    plt.savefig(opts.plots_directory + "/star_immortal_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal accretion stars rate
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

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

    ax.set_xlabel(r"Time [Myr]")
    ax.set_ylabel(r"$\dot{N}_{\rm imm}$ [$\mathrm{yr}^{-1}$]")
    ax.legend(loc="upper right")
    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()
    ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_immortal_rate." + plot_format)

    plt.close()

    # ========================================
    # Immortal stars through merger time distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    # Separate generational subpopulations
    gen1_time = stars_merged_immortal[:, 1][merged_imm_g1_mask]
    gen2_time = stars_merged_immortal[:, 1][merged_imm_g2_mask]
    genX_time = stars_merged_immortal[:, 1][merged_imm_gX_mask]

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

    ax.set_xlabel(r"Time [\Myr]")
    ax.set_ylabel("Number")
    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()    #ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_immortal_merged_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial mass distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    # Separate generational subpopulations
    gen1_mass_initial = stars_immortal[:, 13][immortal_g1_mask]
    gen2_mass_initial = stars_immortal[:, 13][immortal_g2_mask]
    genX_mass_initial = stars_immortal[:, 13][immortal_gX_mask]

    bins = np.linspace(0, stars_immortal[stars_immortal[:, 13] != immortal_mass][:, 13].max(), 30)
    hist_data = [gen1_mass_initial, gen2_mass_initial, genX_mass_initial]

    ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"$M_{\rm initial}$ [Myr]")
    ax.set_ylabel("Number")
    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()    #ax.set_yscale("log")
    ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_immortal_mass_initial_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial radius distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    # Separate generational subpopulations
    gen1_orb_a_initial = stars_immortal[:, 14][immortal_g1_mask]
    gen2_orb_a_initial = stars_immortal[:, 14][immortal_g2_mask]
    genX_orb_a_initial = stars_immortal[:, 14][immortal_gX_mask]

    bins = np.logspace(np.log10(stars_immortal[stars_immortal[:, 13] != immortal_mass][:, 14].min()), np.log10(stars_immortal[stars_immortal[:, 13] != immortal_mass][:, 14].max()), 50)
    hist_data = [gen1_orb_a_initial, gen2_orb_a_initial, genX_orb_a_initial]

    ax.hist(hist_data, bins=bins, align="left", color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.axvline(trap_radius, color="k", linestyle="--", label=r_trap(int(np.rint(trap_radius))))

    ax.set_xlabel(r"$R_{\rm initial}$ [$\mathrm{r}_\mathrm{g}$]")
    ax.set_ylabel("Number")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(imm_radius_dist_ymin, imm_radius_dist_ymax)
    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="lower left")
    elif figsize == 'apj_page':
        ax.legend(loc="lower left")

    plt.savefig(opts.plots_directory + "/star_immortal_orba_initial_dist." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial mass vs initial radius
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

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

    ax.set_xlabel(r"$R_{\rm initial}$ [$\mathrm{r}_\mathrm{g}$]")
    ax.set_ylabel(r"$M_{\rm initial}$ [$M_\odot$]")
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="upper left")
    elif figsize == 'apj_page':
        ax.legend(loc="upper left")

    plt.savefig(opts.plots_directory + "/star_immortal_mass_orba_initial." + plot_format)
    plt.close()

    # ========================================
    # Immortal stars initial radius vs current radius
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

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

    ax.set_ylabel(r"$R_{\rm initial}$ [$\mathrm{r}_\mathrm{g}$]")
    ax.set_xlabel(radius)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower right")
    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="lower right")
    elif figsize == 'apj_page':
        ax.legend(loc="lower right")

    plt.savefig(opts.plots_directory + "/star_immortal_orba_initial_current." + plot_format)
    plt.close()


    # ========================================
    #  Mass vs Radius for disrupted stars
    # ========================================

    # Separate generational subpopulations
    ex_gen1_orb_a = stars_disrupted[:, 2][disrupted_g1_mask]
    ex_gen2_orb_a = stars_disrupted[:, 2][disrupted_g2_mask]
    ex_genX_orb_a = stars_disrupted[:, 2][disrupted_gX_mask]
    ex_gen1_mass = stars_disrupted[:, 3][disrupted_g1_mask]
    ex_gen2_mass = stars_disrupted[:, 3][disrupted_g2_mask]
    ex_genX_mass = stars_disrupted[:, 3][disrupted_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

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

    ax.set_ylabel(r'Mass [$M_\odot$]')
    ax.set_xlabel(radius)
    ax.set_xscale('log')

    ax.legend(loc="upper left")
    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="upper left")
    elif figsize == 'apj_page':
        ax.legend(loc="upper left")

    ax.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_disrupted_mass_v_radius." + plot_format)
    plt.close()

    # ========================================
    #  Mass vs Radius for merged stars
    # ========================================

    # Separate generational subpopulations
    merged_gen1_orb_a = stars_merged[:, 2][merged_g1_mask]
    merged_gen2_orb_a = stars_merged[:, 2][merged_g2_mask]
    merged_genX_orb_a = stars_merged[:, 2][merged_gX_mask]
    merged_gen1_mass = stars_merged[:, 3][merged_g1_mask]
    merged_gen2_mass = stars_merged[:, 3][merged_g2_mask]
    merged_genX_mass = stars_merged[:, 3][merged_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

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

    ax.set_ylabel(r'Mass [$M_\odot$]')
    ax.set_xlabel(radius)
    ax.set_xscale('log')

    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="upper left")
    elif figsize == 'apj_page':
        ax.legend(loc="upper left")

    ax.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_merged_mass_v_radius." + plot_format)
    plt.close()

    # ========================================
    # Disrupted stars luminosity distribution
    # ========================================

    # Separate generational subpopulations
    gen1_lum = stars_disrupted_logL[disrupted_g1_mask]
    gen2_lum = stars_disrupted_logL[disrupted_g2_mask]
    genX_lum = stars_disrupted_logL[disrupted_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    bins = np.linspace(stars_disrupted_logL.min(), stars_disrupted_logL.max(), 50)
    hist_data = [gen1_lum, gen2_lum, genX_lum]

    ax.hist(hist_data, bins=bins, align='left', color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"$\log L/L_\odot$")
    ax.set_ylabel("Number")
    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()
    ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_disrupted_lum_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged stars luminosity distribution
    # ========================================

    # Separate generational subpopulations
    gen1_lum = stars_merged_logL[merged_g1_mask]
    gen2_lum = stars_merged_logL[merged_g2_mask]
    genX_lum = stars_merged_logL[merged_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    bins = np.linspace(stars_merged_logL.min(), stars_merged_logL.max(), 50)
    hist_data = [gen1_lum, gen2_lum, genX_lum]

    ax.hist(hist_data, bins=bins, align='left', color=gen_colors, alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_xlabel(r"$\log L/L_\odot$")
    ax.set_ylabel("Number")
    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()
    ax.set_yscale("log")

    plt.savefig(opts.plots_directory + "/star_merged_lum_dist." + plot_format)
    plt.close()

    # ========================================
    # Number of Mergers vs Mass
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    bins = np.linspace(stars_merged[:, 3].min(), stars_merged[:, 3].max(), 50)
    hist_data = [merged_gen1_mass, merged_gen2_mass, merged_genX_mass]

    ax.hist(hist_data, bins=bins, color=gen_colors, align="left", alpha=0.9, rwidth=0.8, label=gen_labels, stacked=True)

    ax.set_ylabel("Number of mergers")
    ax.set_xlabel(r"Mass [$M_\odot$]")
    ax.set_yscale("log")

    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()

    plt.savefig(opts.plots_directory + r"/star_merged_mass_dist." + plot_format)

    # ========================================
    # Merged stars m1 m2 2d histogram
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    bins = np.linspace(0, 300, 15)
    _, _, _, cm = ax.hist2d(stars_merged[:, 8], stars_merged[:, 9], bins=bins, cmap="binary", norm="log")
    ax.set_aspect("equal")

    cbar = fig.colorbar(cm)
    cbar.set_label("Number")

    ax.set_xlabel(r"$M_1$ [$M_\odot$]")
    ax.set_ylabel(r"$M_2$ [$M_\odot$]")

    plt.savefig(opts.plots_directory + "/star_merged_m1m2_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged stars time distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)
    rwidth = 0.8

    gen1_time = stars_merged[:, 1][merged_g1_mask]
    gen2_time = stars_merged[:, 1][merged_g2_mask]
    genX_time = stars_merged[:, 1][merged_gX_mask]

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

    ax.set_xlabel(r"Time [Myr]")
    ax.set_ylabel("Number")

    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()

    plt.savefig(opts.plots_directory + "/star_merged_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Disrupted stars time distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)
    rwidth = 0.8

    gen1_time = stars_disrupted[:, 1][disrupted_g1_mask]
    gen2_time = stars_disrupted[:, 1][disrupted_g2_mask]
    genX_time = stars_disrupted[:, 1][disrupted_gX_mask]

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

    ax.set_xlabel(r"Time [Myr]")
    ax.set_ylabel("Number")

    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()

    plt.savefig(opts.plots_directory + "/star_disrupted_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged and disrupted stars time distribution
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    bins = np.arange(0, (log_data["timestep_num"] * log_data["timestep_duration_yr"] + log_data["timestep_duration_yr"])/1e6, (log_data["timestep_duration_yr"] * 2)/1e6)
    rwidth = 0.8

    hist_merge, bin_edges = np.histogram(stars_merged[:, 1]/1e6, bins=bins)
    hist_disrupted, _ = np.histogram(stars_disrupted[:, 1]/1e6, bins=bins)

    hist_merge = hist_merge/log_data["galaxy_num"]
    hist_disrupted = hist_disrupted/log_data["galaxy_num"]

    bin_centers = bin_edges[:-1]  # Use left edges since align="left"
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin

    # Create the stacked bar plot
    ax.bar(bin_centers, hist_merge, width=bin_width*rwidth, alpha=0.9, align='edge', color=color_merge, label='Merged stars')
    ax.bar(bin_centers, hist_disrupted, width=bin_width*rwidth, alpha=0.9, align='edge', color=color_disrupted, bottom=hist_merge, label='Disrupted stars')

    ax.set_xlabel(r"Time [Myr]")
    ax.set_ylabel("Number")

    if figsize == 'apj_col':
        ax.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax.legend()

    plt.savefig(opts.plots_directory + "/star_merged_disrupted_time_dist." + plot_format)
    plt.close()

    # ========================================
    # Merged stars mass vs time
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    merged_gen1_mass = stars_merged[:, 3][merged_g1_mask]
    merged_gen2_mass = stars_merged[:, 3][merged_g2_mask]
    merged_genX_mass = stars_merged[:, 3][merged_gX_mask]
    gen1_time = stars_merged[:, 1][merged_g1_mask]/1e6
    gen2_time = stars_merged[:, 1][merged_g2_mask]/1e6
    genX_time = stars_merged[:, 1][merged_gX_mask]/1e6

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

    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="upper left")
    elif figsize == 'apj_page':
        ax.legend(loc="upper left")

    ax.set_ylabel(r"Mass [$M_\odot$]")
    ax.set_xlabel(r"Time [Myr]")

    plt.savefig(opts.plots_directory + "/star_merged_mass_v_time." + plot_format)

    # ========================================
    # Disrupted stars mass vs time
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    disrupted_gen1_mass = stars_disrupted[:, 3][disrupted_g1_mask]
    disrupted_gen2_mass = stars_disrupted[:, 3][disrupted_g2_mask]
    disrupted_genX_mass = stars_disrupted[:, 3][disrupted_gX_mask]
    gen1_time = stars_disrupted[:, 1][disrupted_g1_mask]/1e6
    gen2_time = stars_disrupted[:, 1][disrupted_g2_mask]/1e6
    genX_time = stars_disrupted[:, 1][disrupted_gX_mask]/1e6

    ax.scatter(gen1_time, disrupted_gen1_mass,
               s=styles.markersize_gen1,
               marker=styles.marker_gen1,
               edgecolor=styles.color_gen1,
               facecolors="none",
               alpha=styles.markeralpha_gen1,
               label=gen_labels[0]
               )

    ax.scatter(gen2_time, disrupted_gen2_mass,
               s=styles.markersize_gen2,
               marker=styles.marker_gen2,
               edgecolor=styles.color_gen2,
               facecolors="none",
               alpha=styles.markeralpha_gen2,
               label=gen_labels[1]
               )

    ax.scatter(genX_time, disrupted_genX_mass,
               s=styles.markersize_genX,
               marker=styles.marker_genX,
               edgecolor=styles.color_genX,
               facecolors="none",
               alpha=styles.markeralpha_genX,
               label=gen_labels[2]
               )

    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="upper left")
    elif figsize == 'apj_page':
        ax.legend(loc="upper left")

    ax.set_ylabel(r"Mass [$M_\odot$]")
    ax.set_xlabel(r"Time [Myr]")

    plt.savefig(opts.plots_directory + "/star_disrupted_mass_v_time." + plot_format)

    # ========================================
    # Merged stars rate
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    # Separate generational subpopulations
    gen1_time = stars_merged[:, 1][merged_g1_mask]
    gen2_time = stars_merged[:, 1][merged_g2_mask]
    genX_time = stars_merged[:, 1][merged_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_xlabel(r"Time [Myr]")
    ax.set_ylabel(r"$\dot{N}_{\rm merged}$ [$\mathrm{yr}^{-1}$]")
    ax.set_yscale("log")
    #ax.set_ylim(1e-10, 1e-1)
    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="lower right")
    elif figsize == 'apj_page':
        ax.legend(loc="lower right")

    plt.savefig(opts.plots_directory + "/star_merged_rate." + plot_format)
    ax.set_ylim(rate_ymin, rate_ymax)
    plt.savefig(opts.plots_directory + "/star_merged_rate_slides." + plot_format)

    plt.close()

    # ========================================
    # Disrupted stars rate
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax = fig.add_subplot(111)

    # Separate generational subpopulations
    gen1_time = stars_disrupted[:, 1][disrupted_g1_mask]
    gen2_time = stars_disrupted[:, 1][disrupted_g2_mask]
    genX_time = stars_disrupted[:, 1][disrupted_gX_mask]

    counts_gen1 = get_count_by_param(gen1_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_gen2 = get_count_by_param(gen2_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]
    counts_genX = get_count_by_param(genX_time, timestep_arr)/log_data["galaxy_num"]/log_data["timestep_duration_yr"]

    ax.plot(timestep_arr/1e6, counts_gen1 + counts_gen2 + counts_genX, color="k", label="Total")
    ax.plot(timestep_arr/1e6, counts_gen1, color=gen_colors[0], label=gen_labels[0])
    ax.plot(timestep_arr/1e6, counts_gen2, color=gen_colors[1], label=gen_labels[1])
    ax.plot(timestep_arr/1e6, counts_genX, color=gen_colors[2], label=gen_labels[2])

    ax.set_xlabel(r"Time [Myr")
    ax.set_ylabel(r"$\dot{N}_{\rm disrupted}$ [$\mathrm{yr}^{-1}$]")
    if figsize == 'apj_col':
        ax.legend(fontsize=6, loc="lower right")
    elif figsize == 'apj_page':
        ax.legend(loc="lower right")
    ax.set_yscale("log")
    #ax.set_ylim(1e-10, 1e-1)

    plt.savefig(opts.plots_directory + "/star_disrupted_rate." + plot_format)

    plt.close()


######## Execution ########
if __name__ == "__main__":
    main()