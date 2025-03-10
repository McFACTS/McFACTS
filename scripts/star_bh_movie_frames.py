import os
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
    parser.add_argument("--num-timesteps",
                        default=50,
                        type=int, help="number of timesteps")
    parser.add_argument("--plots-directory",
                        default="gal000",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.fpath_snapshots)
    assert os.path.isdir(opts.fpath_snapshots)
    assert os.path.isdir(opts.plots_directory)
    return opts



immortal_star_cutoff = 298

def plotting(stars_orba, stars_mass, mask_immortal, bh_orba, bh_mass, bbh_orba, bbh_mass, timestep,
             mask_label, nomask_label, bh_label, bbh_label, save_name):
    plt.scatter(stars_orba[~mask_immortal], stars_mass[~mask_immortal], marker="o", edgecolor='#DA627D', facecolor='None', zorder=10)
    plt.scatter(stars_orba[mask_immortal], stars_mass[mask_immortal], marker="o", edgecolor='#450920', facecolor='None', zorder=10)
    plt.scatter(bh_orba, bh_mass, marker="o", edgecolor="darkgoldenrod", facecolor="None", zorder=10)
    plt.scatter(bbh_orba, bbh_mass, marker="o", edgecolors="#005f73", facecolor="None", zorder=10)
    plt.scatter(0, -10, label=nomask_label, color="#DA627D")
    plt.scatter(0, -10, label=mask_label, color="#450920")
    plt.scatter(0, -10, label=bh_label, color="darkgoldenrod")
    plt.scatter(0, -10, label=bbh_label, color="#005f73")

    plt.vlines(700, -5, 500, colors='dimgrey', label="Trap radius", zorder=0)

    plt.legend(title=rf"{(int(timestep) + 1)/1e2} Myr", loc="upper right", frameon=False)
    plt.ylabel(r"Mass [$M_\odot$]")
    plt.xlabel(r"Semi-major axis [R$_{\mathrm {g, SMBH}}$]")
    plt.ylim(-5, 500)
    plt.xlim(70, 55000)

    plt.xscale("log")
    plt.savefig(save_name + "_log.png", dpi=300)

    plt.close()


def generate_plots(fpath, num_timesteps):
    bh_single_pro = []
    bh_binary = []
    star_pro = []
    for i in range(0, num_timesteps):

        bh_single_pro.append(fpath + f"/output_bh_single_pro_{i}.dat")
        bh_binary.append(fpath + f"/output_bh_binary_{i}.dat")

        star_pro.append(fpath + f"/output_stars_single_pro_{i}.dat")

    for star_stuff, bh_stuff, bbh_stuff in zip(enumerate(star_pro), enumerate(bh_single_pro), enumerate(bh_binary)):
        t_star, fname_star = star_stuff
        i_bh, fname_bh = bh_stuff
        i_bbh, fname_bbh = bbh_stuff
        t_star = f"{t_star:02d}"
        star = np.loadtxt(fname_star, skiprows=1)
        bh = np.loadtxt(fname_bh, skiprows=1)
        bbh = np.loadtxt(fname_bbh, skiprows=1)
        mask_immortal = star[:, 2] == immortal_star_cutoff
        if (bbh.size == 0):
            plotting(star[:, 1], star[:, 2], mask_immortal, bh[:, 1], bh[:, 2], None, None, t_star,
                     r"Immortal star ($298$ $M_\odot$)", "Single prograde star", "Single prograde BH", "BBH", fpath + f"/orba_mass_movie_timestep_{t_star}")
        elif (bbh.size == 22):
            plotting(star[:, 1], star[:, 2], mask_immortal, bh[:, 1], bh[:, 2], bbh[9], bbh[2] + bbh[3], t_star,
                     r"Immortal star ($298$ $M_\odot$)", "Single prograde star", "Single prograde BH", "BBH", fpath + f"/orba_mass_movie_timestep_{t_star}")
        else:
            plotting(star[:, 1], star[:, 2], mask_immortal, bh[:, 1], bh[:, 2], bbh[:, 9], bbh[:, 2] + bbh[:, 3], t_star,
                     r"Immortal star ($298$ $M_\odot$)", "Single prograde star", "Single prograde BH", "BBH", fpath + f"/orba_mass_movie_timestep_{t_star}")


def main():

    opts = arg()

    generate_plots(opts.fpath_snapshots, opts.num_timesteps)


######## Execution ########
if __name__ == "__main__":
    main()
