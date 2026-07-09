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


def plotting(plot_objects, star_orba, star_imm_orba, bh_orba, bh_binary_orba,
             bins, timestep, mask_label, nomask_label, bh_label, bbh_label,
             save_name, trap_radius, disk_outer_radius):
    fig = plt.figure()
    fig.set_figwidth(8)

    if (plot_objects == 0) or (plot_objects == 1):
        plt.hist([star_orba, star_imm_orba, bh_orba, bh_binary_orba], bins=bins, label=[nomask_label, mask_label, bh_label, bbh_label], color=[color_star, color_immortal, color_bh, color_bbh], stacked=True)

    if plot_objects == 1:
        plt.hist([star_orba, star_imm_orba], bins=bins, label=[nomask_label, mask_label], color=[color_star, color_immortal], stacked=True)

    if plot_objects == 2:
        plt.hist([bh_orba, bh_binary_orba], bins=bins, label=[bh_label, bbh_label], color=[color_bh, color_bbh], stacked=True)

    plt.axvline(trap_radius, color='dimgrey', linestyle="--", label=r_trap(int(np.rint(trap_radius))), zorder=0)

    plt.legend(title=rf"{(int(timestep))/1e2} Myr", loc="upper left", frameon=False)
    plt.ylabel(r"Number")
    plt.xlabel(radius)
    #plt.ylim(-5, 600)
    plt.xlim(50, disk_outer_radius)

    plt.xscale("log")
    plt.savefig(save_name + "_log.png", dpi=300)

    plt.close()


def load_data(fname, orba_idx, mass_idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = np.loadtxt(fname)
    if data.size == 0:
        orba = np.array([None])
        mass = np.array([None])
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


def generate_plots(plot_objects, fpath, num_timesteps, timestep_duration_yr, trap_radius, disk_outer_radius):

    bh_binary_orba, bh_binary_mass = None, None
    star_orba = None
    star_imm_orba = None
    bh_orba = None

    bins = np.logspace(np.log10(20), np.log10(50000), 50)

    for i in range(0, num_timesteps):

        if (plot_objects == 0) or (plot_objects == 1):
            star_pro_orba, star_pro_mass = load_data(fpath + f"/output_stars_single_pro_{i}.dat", 1, 2)
            star_inner_orba, star_inner_mass = load_data(fpath + f"/output_stars_single_inner_disk_{i}.dat", 1, 2)
            star_retro_orba, star_retro_mass = load_data(fpath + f"/output_stars_single_retro_{i}.dat", 1, 2)
            mask_immortal = star_pro_mass == immortal_star_cutoff
            star_orba = np.concatenate([star_pro_orba[~mask_immortal], star_inner_orba, star_retro_orba])
            star_imm_orba = star_pro_orba[mask_immortal]
            star_imm_orba = star_imm_orba[star_imm_orba != None]
            star_orba = star_orba[star_orba != None]

        if (plot_objects == 0) or (plot_objects == 2):
            bh_pro_orba, bh_pro_mass = load_data(fpath + f"/output_bh_single_pro_{i}.dat", 1, 2)
            bh_inner_orba, bh_inner_mass = load_data(fpath + f"/output_bh_single_inner_disk_{i}.dat", 1, 2)
            bh_retro_orba, bh_retro_mass = load_data(fpath + f"/output_bh_single_retro_{i}.dat", 1, 2)
            bh_binary_orba, bh_binary_mass = load_data(fpath + f"/output_bh_binary_{i}.dat", 9, [2, 3])
            bh_orba = np.concatenate([bh_pro_orba, bh_inner_orba, bh_retro_orba])
            bh_orba = bh_orba[bh_orba != None]
            bh_binary_orba = bh_binary_orba[bh_binary_orba != None]

        plotting(plot_objects, star_orba, star_imm_orba, bh_orba, bh_binary_orba,
                 bins, f"{i:03d}",
                 r"Immortal star", "Single star", "Single BH", "BBH",
                 fpath + f"movie_frames_orba_dist/orba_dist_movie_timestep_{i:03d}", trap_radius, disk_outer_radius)


def main():

    opts = arg()

    log_data = ReadLog(opts.fname_log)

    generate_plots(opts.plot_objects, opts.fpath_snapshots,
                   opts.num_timesteps, opts.timestep_duration_yr,
                   log_data["disk_radius_trap"], log_data["disk_radius_outer"])


######## Execution ########
if __name__ == "__main__":
    main()
