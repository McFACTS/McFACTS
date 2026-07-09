import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mcfacts.vis import styles
plt.style.use("mcfacts.vis.kaila_thesis_figures")
import glob as g
from mcfacts.outputs.ReadOutputs import ReadLog


def r_trap(trap_radius):
    return (r"$R_{\rm trap} = \qty{" + f"{trap_radius}" + r"}{\rg}$")


radius = r"Radius [\unit{\rg}]"


######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath-snapshots",
                        default="gal000",
                        type=str, help="path to galaxy")
    parser.add_argument("--fname-log",
                        default="mcfacts.log",
                        type=str, help="log file")
    parser.add_argument("--fname-stars-merged",
                        default="output_stars_merged.dat",
                        type=str, help="output merged stars file")
    parser.add_argument("--fname-stars-disrupted",
                        default="output_stars_disrupted.dat",
                        type=str, help="output disrupted stars file")
    parser.add_argument("--fname-stars-unbound",
                        default="output_stars_unbound.dat",
                        type=str, help="output unbound stars file")
    parser.add_argument("--fname-bh-unbound",
                        default="output_mergers_unbound.dat",
                        type=str, help="output unbound bh file")
    parser.add_argument("--fname-emri",
                        default="output_mergers_emris.dat",
                        type=str, help="emri file")
    parser.add_argument("--fname-star-tde",
                        default="output_tdes.dat",
                        type=str, help="output star tde file")
    parser.add_argument("--fname-star-plunge",
                        default="output_stars_plunge.dat",
                        type=str, help="stars plunge file")
    parser.add_argument("--num-timesteps",
                        default=50,
                        type=int, help="number of timesteps")
    parser.add_argument("--timestep-duration-yr",
                        default=10000,
                        type=int, help="timestep length in  years")
    parser.add_argument("--plots-directory",
                        default="gal000",
                        type=str, help="directory to save plots")
    parser.add_argument("--plot-objects",
                        default=0,
                        type=int, help="0: plot stars + BH, 1: plot stars, 2: plot BHs")
    opts = parser.parse_args()
    print(opts.fpath_snapshots)
    assert os.path.isdir(opts.fpath_snapshots)
    assert os.path.isdir(opts.plots_directory)
    return opts


immortal_star_cutoff = 298
# Higher zorder means on top
star_zorder = 5
bh_zorder = 10
bbh_zorder = 15

color_bh = "lightskyblue"
color_bbh = "#005f73"
color_star = "#DA627D"
color_immortal ="#901343" # '#450920'
color_event = "k"
plt.rcParams.update({'axes.labelsize': 20,
                     "xtick.labelsize": 20,
                     "ytick.labelsize": 20,
                     "legend.fontsize": 12,
                     "font.size": 15,
                     "xtick.major.pad": 6})


def plotting(plot_objects, stars_orba, stars_mass, mask_immortal, starsin_orba, starsin_mass, starsretro_orba, starsretro_mass,
             bh_orba, bh_mass, bhin_orba, bhin_mass, bhretro_orba, bhretro_mass, bbh_orba, bbh_mass,
             bbh_merged_orba, bbh_merged_mass, star_merged_orba, star_merged_mass, star_disrupted_orba, star_disrupted_mass,
             bh_unbound_orba, bh_unbound_mass, star_unbound_orb_a, star_unbound_mass,
             emri_orba, emri_mass, star_plunge_orba, star_plunge_mass,
             bh_unbound_flag,
             timestep, mask_label, nomask_label, bh_label, bbh_label, bbh_merged_label, star_merged_label, star_disrupted_label,
             bh_unbound_label, star_unbound_label, emri_label, star_plunge_label, save_name,
             trap_radius, disk_radius_outer):
    fig = plt.figure()
    fig.set_figwidth(8)
    if (plot_objects == 0) or (plot_objects == 1):
        plt.scatter(stars_orba[~mask_immortal], stars_mass[~mask_immortal], marker="o", edgecolor=color_star, facecolor='None', zorder=star_zorder)
        plt.scatter(stars_orba[mask_immortal], stars_mass[mask_immortal], marker="o", edgecolor=color_immortal, facecolor='None', zorder=star_zorder)
        plt.scatter(starsin_orba, starsin_mass, marker="o", edgecolor=color_star, facecolor='None', zorder=star_zorder)
        plt.scatter(starsretro_orba, starsretro_mass, marker="o", edgecolor=color_star, facecolor='None', zorder=star_zorder)
        plt.scatter(star_merged_orba, star_merged_mass, marker="d", edgecolor=color_event, facecolor="None", zorder=star_zorder)
        plt.scatter(star_disrupted_orba, star_disrupted_mass, marker="X", edgecolor=color_event, facecolor="None", zorder=star_zorder)
        plt.scatter(star_unbound_orb_a, star_unbound_mass, marker=">", edgecolor=color_event, facecolor="None", zorder=star_zorder)
        plt.scatter(star_plunge_orba, star_plunge_mass, marker="2", color=color_event, zorder=star_zorder)

        plt.scatter(0, -10, label=nomask_label, color=color_star)
        plt.scatter(0, -10, label=mask_label, color=color_immortal)
        plt.scatter(0, -10, marker="d", label=star_merged_label, edgecolor=color_event, facecolor='None')
        plt.scatter(0, -10, marker="X", label=star_disrupted_label, edgecolor=color_event, facecolor='None')
        plt.scatter(0, -10, marker=">", label=star_unbound_label, edgecolor=color_event, facecolor='None')
        plt.scatter(0, -10, marker="2", label=star_plunge_label, color=color_event)

    if (plot_objects == 0) or (plot_objects == 2):
        plt.scatter(bh_orba, bh_mass, marker="o", edgecolor=color_bh, facecolor="None", zorder=bh_zorder)
        plt.scatter(bhin_orba, bhin_mass, marker="o", edgecolor=color_bh, facecolor="None", zorder=bh_zorder)
        plt.scatter(bhretro_orba, bhretro_mass, marker="o", edgecolor=color_bh, facecolor="None", zorder=bh_zorder)
        plt.scatter(bbh_orba, bbh_mass, marker="o", edgecolors=color_bbh, facecolor="None", zorder=bbh_zorder)
        plt.scatter(bbh_merged_orba, bbh_merged_mass, marker="D", edgecolor=color_event, facecolor=color_event, zorder=bbh_zorder)
        if bh_unbound_flag is True:
            plt.scatter(bh_unbound_orba, bh_unbound_mass, marker="^", edgecolor=color_event, facecolor="None", zorder=bh_zorder)
        #plt.scatter(emri_orba, emri_mass, marker="1", color=color_event, zorder=bh_zorder)

        plt.scatter(0, -10, label=bh_label, color=color_bh)
        plt.scatter(0, -10, label=bbh_label, color=color_bbh)
        plt.scatter(0, -10, marker="D", label=bbh_merged_label, edgecolor=color_event, facecolor='None')
        if bh_unbound_flag is True:
            plt.scatter(0, -10, marker="^", label=bh_unbound_label, edgecolor=color_event, facecolor='None')
        #plt.scatter(0, -10, marker="1", label=emri_label, color=color_event)

    plt.axvline(trap_radius, color='dimgrey', linestyle="--", label=r_trap(int(np.rint(trap_radius))), zorder=0)

    plt.legend(title=rf"{(int(timestep))/1e2} Myr", loc="upper left", frameon=False)
    plt.ylabel(r"Mass [\unit{\msun}]", labelpad=8)
    plt.xlabel(radius)
    plt.ylim(-5, 320)
    plt.xlim(50, disk_radius_outer)

    plt.xscale("log")
    plt.savefig(save_name + "_log.png", dpi=300)

    plt.close()


def load_data(fname, orba_idx, mass_idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = np.loadtxt(fname)
    if data.size == 0:
        orba = None
        mass = None
    elif len(data.shape) == 1:
        orba = np.array([data[orba_idx]])
        mass = np.array([data[mass_idx]])
    elif len(data.shape) == 2:
        orba = data[:, orba_idx]
        mass = data[:, mass_idx]
    else:
        raise IndexError("Data array not of correct shape")

    # if a binary, add the component masses together
    if isinstance(mass_idx, (list, np.ndarray)) and len(mass_idx) == 2:
        try:
            mass = np.sum(mass, axis=1)
        except np.exceptions.AxisError:
            mass = np.sum(mass)

    return orba, mass


def generate_plots(plot_objects, fpath, num_timesteps, timestep_duration_yr,
                   bbh_merged_data, star_disrupted_data, star_merged_data,
                   bh_unbound_data, star_unbound_data,
                   emri_data, star_plunge_data, bh_unbound_flag,
                   trap_radius, disk_radius_outer):

    star_pro_orba, star_pro_mass = None, None
    star_inner_orba, star_inner_mass = None, None
    star_retro_orba, star_retro_mass = None, None
    mask_immortal = None
    bh_pro_orba, bh_pro_mass = None, None
    bh_inner_orba, bh_inner_mass = None, None
    bh_retro_orba, bh_retro_mass = None, None
    bh_binary_orba, bh_binary_mass = None, None
    bbh_merged_orba, bbh_merged_mass = None, None
    star_merged_orba, star_merged_mass = None, None
    star_disrupted_orba, star_disrupted_mass = None, None
    bh_unbound_orba, bh_unbound_mass = None, None
    star_unbound_orba, star_unbound_mass = None, None
    emri_orba, emri_mass = None, None
    star_plunge_orba, star_plunge_mass = None, None

    for i in range(0, num_timesteps):

        if (plot_objects == 0) or (plot_objects == 1):
            star_pro_orba, star_pro_mass = load_data(fpath + f"/output_stars_single_pro_{i}.dat", 1, 2)
            star_inner_orba, star_inner_mass = load_data(fpath + f"/output_stars_single_inner_disk_{i}.dat", 1, 2)
            star_retro_orba, star_retro_mass = load_data(fpath + f"/output_stars_single_retro_{i}.dat", 1, 2)
            star_merged_orba = star_merged_data[star_merged_data[:, 1] == i * timestep_duration_yr][:, 2]
            star_merged_mass = star_merged_data[star_merged_data[:, 1] == i * timestep_duration_yr][:, 3]
            star_disrupted_orba = star_disrupted_data[star_disrupted_data[:, 1] == i * timestep_duration_yr][:, 2]
            star_disrupted_mass = star_disrupted_data[star_disrupted_data[:, 1] == i * timestep_duration_yr][:, 3]
            star_unbound_orba = star_unbound_data[star_unbound_data[:, 1] == i * timestep_duration_yr][:, 2]
            star_unbound_mass = star_unbound_data[star_unbound_data[:, 1] == i * timestep_duration_yr][:, 3]
            star_plunge_orba = star_plunge_data[star_plunge_data[:, 1] == i * timestep_duration_yr][:, 2]
            star_plunge_mass = star_plunge_data[star_plunge_data[:, 1] == i * timestep_duration_yr][:, 3]
            mask_immortal = star_pro_mass == immortal_star_cutoff

        if (plot_objects == 0) or (plot_objects == 2):
            bh_pro_orba, bh_pro_mass = load_data(fpath + f"/output_bh_single_pro_{i}.dat", 1, 2)
            bh_inner_orba, bh_inner_mass = load_data(fpath + f"/output_bh_single_inner_disk_{i}.dat", 1, 2)
            bh_retro_orba, bh_retro_mass = load_data(fpath + f"/output_bh_single_retro_{i}.dat", 1, 2)
            bh_binary_orba, bh_binary_mass = load_data(fpath + f"/output_bh_binary_{i}.dat", 9, [2, 3])
            if bh_unbound_flag is True:
                bh_unbound_orba = bh_unbound_data[bh_unbound_data[:, 1] == i * timestep_duration_yr][:, 2]
                bh_unbound_mass = bh_unbound_data[bh_unbound_data[:, 1] == i * timestep_duration_yr][:, 3]
            if len(bbh_merged_data.shape) > 1:
                bbh_merged_orba = bbh_merged_data[bbh_merged_data[:, 2] == i * timestep_duration_yr][:, 0]
                bbh_merged_mass = bbh_merged_data[bbh_merged_data[:, 2] == i * timestep_duration_yr][:, 1]
            else:
                bbh_merged_orba = bbh_merged_data[bbh_merged_data[2] == i * timestep_duration_yr][:, 0]
                bbh_merged_mass = bbh_merged_data[bbh_merged_data[2] == i * timestep_duration_yr][:, 1]
            emri_orba = emri_data[emri_data[:, 1] == i * timestep_duration_yr][:, 2]
            emri_mass = emri_data[emri_data[:, 1] == i * timestep_duration_yr][:, 3]

        plotting(plot_objects, star_pro_orba, star_pro_mass, mask_immortal, star_inner_orba, star_inner_mass, star_retro_orba, star_retro_mass,
                 bh_pro_orba, bh_pro_mass, bh_inner_orba, bh_inner_mass, bh_retro_orba, bh_retro_mass,
                 bh_binary_orba, bh_binary_mass,
                 bbh_merged_orba, bbh_merged_mass,
                 star_merged_orba, star_merged_mass,
                 star_disrupted_orba, star_disrupted_mass,
                 bh_unbound_orba, bh_unbound_mass,
                 star_unbound_orba, star_unbound_mass,
                 emri_orba, emri_mass,
                 star_plunge_orba, star_plunge_mass,
                 bh_unbound_flag,
                 f"{i:03d}",
                 r"Immortal star", "Single star", "Single BH", "BBH", "BBH merger", "Star merger",
                 "Star disruption", "Unbound BH", "Unbound star", "EMRI", "Star TDE/plunger",
                 fpath + f"movie_frames_orba_mass/orba_mass_movie_timestep_{i:03d}",
                 trap_radius, disk_radius_outer)


def main():

    opts = arg()
    log_data = ReadLog(opts.fname_log)

    # check if stars exist (if not, unbound BH file doesn't exist)
    bh_unbound_flag = False
    star_files = g.glob(opts.fpath_snapshots + "*stars*")
    if len(star_files) != 0:
        bh_unbound_flag = True
    gal_num = int("".join(filter(str.isdigit, [s for s in opts.fpath_snapshots.split("/") if "gal" in s][0])))
    bbh_merge, bh_unbound, emri = None, None, None
    star_disrupted, star_merged, star_unbound, star_plunge = None, None, None, None

    # plot stars
    if (opts.plot_objects == 0) or (opts.plot_objects == 1):
        # following cols are galaxy, time_sn, orb_a_star, mass_star
        star_disrupted = np.loadtxt(opts.fname_stars_disrupted, usecols=(0, 1, 2, 3))
        star_merged = np.loadtxt(opts.fname_stars_merged, usecols=(0, 1, 2, 3))
        star_unbound = np.loadtxt(opts.fname_stars_unbound, usecols=(0, 1, 2, 3))
        tde = np.loadtxt(opts.fname_star_tde, usecols=(0, 1, 2, 3))
        star_plunge = np.loadtxt(opts.fname_star_plunge, usecols=(0, 1, 2, 3))

        # Cut out other galaxies
        star_disrupted = star_disrupted[star_disrupted[:, 0] == gal_num]
        star_merged = star_merged[star_merged[:, 0] == gal_num]
        star_unbound = star_unbound[star_unbound[:, 0] == gal_num]
        tde = tde[tde[:, 0] == gal_num]
        star_plunge = star_plunge[star_plunge[:, 0] == gal_num]
        star_plunge = np.concatenate((star_plunge, tde))

    # plot black holes
    if (opts.plot_objects == 0) or (opts.plot_objects == 2):
        # Load BBH mergers, star mergers, star disruptions
        # BBH cols are bin_orb_a, mass_final, time_merged
        bbh_merged = np.loadtxt(opts.fpath_snapshots + "/output_mergers.dat", usecols=(1, 2, 14))
        # following cols are galaxy, time_sn, orb_a_star, mass_star
        if bh_unbound_flag is True:
            bh_unbound = np.loadtxt(opts.fname_bh_unbound, usecols=(0, 1, 2, 3))
            bh_unbound = bh_unbound[bh_unbound[:, 0] == gal_num]

        emri = np.loadtxt(opts.fname_emri, usecols=(0, 1, 2, 3))

        # Cut out other galaxies
        emri = emri[emri[:, 0] == gal_num]

    generate_plots(opts.plot_objects, opts.fpath_snapshots, opts.num_timesteps, opts.timestep_duration_yr,
                   bbh_merged, star_disrupted, star_merged, bh_unbound, star_unbound,
                   emri, star_plunge, bh_unbound_flag, log_data["disk_radius_trap"], log_data["disk_radius_outer"])


######## Execution ########
if __name__ == "__main__":
    main()
