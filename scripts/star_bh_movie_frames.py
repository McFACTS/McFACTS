import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath-snapshots",
                        default="gal000",
                        type=str, help="path to galaxy")
    parser.add_argument("--fname-stars-merge",
                        default="output_stars_merged.dat",
                        type=str, help="output merged stars file")
    parser.add_argument("--fname-stars-explode",
                        default="output_stars_exploded.dat",
                        type=str, help="output exploded stars file")
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
                        default=60,
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


def plotting(plot_objects, stars_orba, stars_mass, mask_immortal, starsin_orba, starsin_mass, starsretro_orba, starsretro_mass,
             bh_orba, bh_mass, bhin_orba, bhin_mass, bhretro_orba, bhretro_mass, bbh_orba, bbh_mass,
             bbh_merge_orba, bbh_merge_mass, star_merge_orba, star_merge_mass, star_explode_orba, star_explode_mass,
             bh_unbound_orba, bh_unbound_mass, star_unbound_orb_a, star_unbound_mass,
             emri_orba, emri_mass, star_plunge_orba, star_plunge_mass,
             timestep, mask_label, nomask_label, bh_label, bbh_label, bbh_merge_label, star_merge_label, star_explode_label,
             bh_unbound_label, star_unbound_label, emri_label, star_plunge_label, save_name):
    if (plot_objects == 0) or (plot_objects == 1):
        plt.scatter(stars_orba[~mask_immortal], stars_mass[~mask_immortal], marker="o", edgecolor='#DA627D', facecolor='None', zorder=star_zorder)
        plt.scatter(stars_orba[mask_immortal], stars_mass[mask_immortal], marker="o", edgecolor='#450920', facecolor='None', zorder=star_zorder)
        plt.scatter(starsin_orba, starsin_mass, marker="o", edgecolor='#DA627D', facecolor='None', zorder=star_zorder)
        plt.scatter(starsretro_orba, starsretro_mass, marker="o", edgecolor='#DA627D', facecolor='None', zorder=star_zorder)
        plt.scatter(star_merge_orba, star_merge_mass, marker="d", edgecolor="k", facecolor="None", zorder=star_zorder)
        plt.scatter(star_explode_orba, star_explode_mass, marker="X", edgecolor="k", facecolor="None", zorder=star_zorder)
        plt.scatter(star_unbound_orb_a, star_unbound_mass, marker=">", edgecolor="k", facecolor="None", zorder=star_zorder)
        plt.scatter(star_plunge_orba, star_plunge_mass, marker="2", color="k", zorder=star_zorder)
        plt.scatter(0, -10, label=nomask_label, color="#DA627D")
        plt.scatter(0, -10, label=mask_label, color="#450920")
        plt.scatter(0, -10, marker="d", label=star_merge_label, color="k")
        plt.scatter(0, -10, marker="X", label=star_explode_label, color="k")
        plt.scatter(0, -10, marker=">", label=star_unbound_label, color="k")
        plt.scatter(0, -10, marker="2", label=star_plunge_label, color="k")

    if (plot_objects == 0) or (plot_objects == 2):
        plt.scatter(bh_orba, bh_mass, marker="o", edgecolor="tab:blue", facecolor="None", zorder=bh_zorder)
        plt.scatter(bhin_orba, bhin_mass, marker="o", edgecolor="tab:blue", facecolor="None", zorder=bh_zorder)
        plt.scatter(bhretro_orba, bhretro_mass, marker="o", edgecolor="tab:blue", facecolor="None", zorder=bh_zorder)
        plt.scatter(bbh_orba, bbh_mass, marker="o", edgecolors="#005f73", facecolor="None", zorder=bbh_zorder)
        plt.scatter(bbh_merge_orba, bbh_merge_mass, marker="D", edgecolor="k", facecolor="None", zorder=bbh_zorder)
        plt.scatter(bh_unbound_orba, bh_unbound_mass, marker="^", edgecolor="k", facecolor="None", zorder=bh_zorder)
        plt.scatter(emri_orba, emri_mass, marker="1", color="k", zorder=bh_zorder)
        plt.scatter(0, -10, label=bh_label, color="tab:blue")
        plt.scatter(0, -10, label=bbh_label, color="#005f73")
        plt.scatter(0, -10, marker="D", label=bbh_merge_label, color="k")
        plt.scatter(0, -10, marker="^", label=bh_unbound_label, color="k")
        plt.scatter(0, -10, marker="1", label=emri_label, color="k")

    plt.vlines(700, -5, 500, colors='dimgrey', label="Trap radius", zorder=0)

    plt.legend(title=rf"{(int(timestep))/1e2} Myr", loc="upper left", frameon=False)
    plt.ylabel(r"Mass [$M_\odot$]")
    plt.xlabel(r"Semi-major axis [R$_{\mathrm {g, SMBH}}$]")
    plt.ylim(-5, 310)
    plt.xlim(20, 55000)

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
        orba = data[orba_idx]
        mass = data[mass_idx]
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
                   bbh_merge_data, star_explode_data, star_merge_data,
                   bh_unbound_data, star_unbound_data,
                   emri_data, star_plunge_data):

    star_pro_orba, star_pro_mass = None, None
    star_inner_orba, star_inner_mass = None, None
    star_retro_orba, star_retro_mass = None, None
    mask_immortal = None
    bh_pro_orba, bh_pro_mass = None, None
    bh_inner_orba, bh_inner_mass = None, None
    bh_retro_orba, bh_retro_mass = None, None
    bh_binary_orba, bh_binary_mass = None, None
    bbh_merge_orba, bbh_merge_mass = None, None
    star_merge_orba, star_merge_mass = None, None
    star_explode_orba, star_explode_mass = None, None
    bh_unbound_orba, bh_unbound_mass = None, None
    star_unbound_orba, star_unbound_mass = None, None
    emri_orba, emri_mass = None, None
    star_plunge_orba, star_plunge_mass = None, None

    for i in range(0, num_timesteps):

        if (plot_objects == 0) or (plot_objects == 1):
            star_pro_orba, star_pro_mass = load_data(fpath + f"/output_stars_single_pro_{i}.dat", 1, 2)
            star_inner_orba, star_inner_mass = load_data(fpath + f"/output_stars_single_inner_disk_{i}.dat", 1, 2)
            star_retro_orba, star_retro_mass = load_data(fpath + f"/output_stars_single_retro_{i}.dat", 1, 2)
            star_merge_orba = star_merge_data[star_merge_data[:, 1] == i * timestep_duration_yr][:, 2]
            star_merge_mass = star_merge_data[star_merge_data[:, 1] == i * timestep_duration_yr][:, 3]
            star_explode_orba = star_explode_data[star_explode_data[:, 1] == i * timestep_duration_yr][:, 2]
            star_explode_mass = star_explode_data[star_explode_data[:, 1] == i * timestep_duration_yr][:, 3]
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
            bh_unbound_orba = bh_unbound_data[bh_unbound_data[:, 1] == i * timestep_duration_yr][:, 2]
            bh_unbound_mass = bh_unbound_data[bh_unbound_data[:, 1] == i * timestep_duration_yr][:, 3]
            bbh_merge_orba = bbh_merge_data[bbh_merge_data[:, 2] == i * timestep_duration_yr][:, 0]
            bbh_merge_mass = bbh_merge_data[bbh_merge_data[:, 2] == i * timestep_duration_yr][:, 1]
            emri_orba = emri_data[emri_data[:, 1] == i * timestep_duration_yr][:, 2]
            emri_mass = emri_data[emri_data[:, 1] == i * timestep_duration_yr][:, 3]

        plotting(plot_objects, star_pro_orba, star_pro_mass, mask_immortal, star_inner_orba, star_inner_mass, star_retro_orba, star_retro_mass,
                 bh_pro_orba, bh_pro_mass, bh_inner_orba, bh_inner_mass, bh_retro_orba, bh_retro_mass,
                 bh_binary_orba, bh_binary_mass,
                 bbh_merge_orba, bbh_merge_mass,
                 star_merge_orba, star_merge_mass,
                 star_explode_orba, star_explode_mass,
                 bh_unbound_orba, bh_unbound_mass,
                 star_unbound_orba, star_unbound_mass,
                 emri_orba, emri_mass,
                 star_plunge_orba, star_plunge_mass,
                 f"{i:03d}",
                 r"Immortal star ($298$ $M_\odot$)", "Single star", "Single BH", "BBH", "BBH merger", "Star merger",
                 "Supernova", "Unbound BH", "Unbound star", "EMRI", "Star TDE/plunger",
                 fpath + f"/orba_mass_movie_timestep_{i:03d}")


def main():

    opts = arg()

    gal_num = int(opts.fpath_snapshots[len(opts.fpath_snapshots) - 4:-1])
    bbh_merge, bh_unbound, emri = None, None, None
    star_explode, star_merge, star_unbound, star_plunge = None, None, None, None

    # plot stars
    if (opts.plot_objects == 0) or (opts.plot_objects == 1):
        # following cols are galaxy, time_sn, orb_a_star, mass_star
        star_explode = np.loadtxt(opts.fname_stars_explode, usecols=(0, 1, 2, 3))
        star_merge = np.loadtxt(opts.fname_stars_merge, usecols=(0, 1, 2, 3))
        star_unbound = np.loadtxt(opts.fname_stars_unbound, usecols=(0, 1, 2, 3))
        tde = np.loadtxt(opts.fname_star_tde, usecols=(0, 1, 2, 3))
        star_plunge = np.loadtxt(opts.fname_star_plunge, usecols=(0, 1, 2, 3))

        # Cut out other galaxies
        star_explode = star_explode[star_explode[:, 0] == gal_num]
        star_merge = star_merge[star_merge[:, 0] == gal_num]
        star_unbound = star_unbound[star_unbound[:, 0] == gal_num]
        tde = tde[tde[:, 0] == gal_num]
        star_plunge = star_plunge[star_plunge[:, 0] == gal_num]
        star_plunge = np.concatenate((star_plunge, tde))

    # plot black holes
    if (opts.plot_objects == 0) or (opts.plot_objects == 2):
        # Load BBH mergers, star mergers, star explosions
        # BBH cols are bin_orb_a, mass_final, time_merged
        bbh_merge = np.loadtxt(opts.fpath_snapshots + "/output_mergers.dat", usecols=(1, 2, 14))
        # following cols are galaxy, time_sn, orb_a_star, mass_star
        bh_unbound = np.loadtxt(opts.fname_bh_unbound, usecols=(0, 1, 2, 3))
        emri = np.loadtxt(opts.fname_emri, usecols=(0, 1, 2, 3))

        # Cut out other galaxies
        bh_unbound = bh_unbound[bh_unbound[:, 0] == gal_num]
        emri = emri[emri[:, 0] == gal_num]

    generate_plots(opts.plot_objects, opts.fpath_snapshots, opts.num_timesteps, opts.timestep_duration_yr,
                   bbh_merge, star_explode, star_merge, bh_unbound, star_unbound,
                   emri, star_plunge)


######## Execution ########
if __name__ == "__main__":
    main()
