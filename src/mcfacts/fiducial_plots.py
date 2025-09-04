import matplotlib.ticker as mticker
import numpy as np
import os
from matplotlib import pyplot as plt

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.snapshot import TxtSnapshotHandler

from mcfacts.vis import styles
from mcfacts.vis import plotting

def make_gen_masks(gen_obj1, gen_obj2):
    """Create masks for retrieving different sets of a merged or binary population based on generation.
    """
    # Column of generation data

    # Masks for hierarchical generations
    # g1 : all 1g-1g objects
    # g2 : 2g-1g and 2g-2g objects
    # g3 : >=3g-Ng (first object at least 3rd gen; second object any gen)
    # Pipe operator (|) = logical OR. (&)= logical AND.
    g1_mask = (gen_obj1 == 1) & (gen_obj2 == 1)
    g2_mask = ((gen_obj1 == 2) | (gen_obj2 == 2)) & ((gen_obj1 <= 2) & (gen_obj2 <= 2))
    gX_mask = (gen_obj1 >= 3) | (gen_obj2 >= 3)

    return g1_mask, g2_mask, gX_mask


def num_mergers_vs_mass(settings, figsize, save_dir, merger_masks, mass_final):
    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks

    fig = plt.figure(figsize=plotting.set_size(figsize))
    counts, bins = np.histogram(mass_final)
    bins = np.arange(int(mass_final.min()), int(mass_final.max()) + 2, 1)

    hist_data = [mass_final[merger_g1_mask], mass_final[merger_g2_mask], mass_final[merger_gX_mask]]
    hist_label = ['1g-1g', '2g-1g or 2g-2g', r'$\geq$3g-Ng']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.xscale('log')
    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    svf_ax.tick_params(axis='x', direction='out', which='both')
    svf_ax.yaxis.grid(True, color='gray', ls='dashed')

    plt.xticks(np.geomspace(int(mass_final.min()), int(mass_final.max()), 5).astype(int))

    svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "merger_remnant_mass.png"), format='png', dpi=300)
    plt.close()

def merger_vs_radius(settings, figsize, save_dir, merger_masks, mass, orb_a):
    # for i in range(len(mergers[:, 1])):
    #     if mergers[i, 1] < 10.0:
    #         mergers[i, 1] = 10.0

    merger_g1_mask, merger_g2_mask, merger_gX_mask = merger_masks

    # Separate generational subpopulations
    gen1_orb_a = orb_a[merger_g1_mask]
    gen2_orb_a = orb_a[merger_g2_mask]
    genX_orb_a = orb_a[merger_gX_mask]
    gen1_mass = mass[merger_g1_mask]
    gen2_mass = mass[merger_g2_mask]
    genX_mass = mass[merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_orb_a, gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )

    plt.scatter(gen2_orb_a, gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )

    plt.scatter(genX_orb_a, genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g-Ng'
                )

    plt.axvline(settings.disk_radius_trap, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {settings.disk_radius_trap:.0f} ' + r'$R_g$')

    # plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Remnant Mass [$M_\odot$]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.ylim(15, 1000)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "merger_mass_v_radius.png"), format='png', dpi=300)
    plt.close()


def main():
    # TODO: Handle settings import through arguments
    settings = SettingsManager({
        "verbose": False,
        "override_files": True,
        "save_state": True,
        "save_each_timestep": True,
        "disk_inner_stable_circ_orb": 16
    })

    plt.style.use("mcfacts.vis.mcfacts_figures")

    figsize = "apj_col"

    snapshot_handler = TxtSnapshotHandler(settings)

    population_cabinet = snapshot_handler.load_cabinet("./runs", "population")

    mergers = population_cabinet["blackholes_merged"]
    mass_final = mergers["mass_final"]
    orb_a = mergers["orb_a"]

    merger_masks = (make_gen_masks(mergers["gen_1"], mergers["gen_2"])) # Man, I hate python

    plots_dir = "./runs/plots"

    num_mergers_vs_mass(settings, figsize, plots_dir, merger_masks, mass_final)
    merger_vs_radius(settings, figsize, plots_dir, merger_masks, mass_final, orb_a)


if __name__ == "__main__":
    main()