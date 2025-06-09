from mcfacts.setup import setupdiskstars
from mcfacts.setup import diskstars_hillspheremergers
from mcfacts.physics import stellar_interpolation
from mcfacts.objects.agnobject import AGNStar
import numpy as np


def init_single_stars(opts, disk_aspect_ratio, galaxy, id_start_val=None):

    # Generate initial number of stars
    star_num_initial = setupdiskstars.setup_disk_stars_num(
            opts.nsc_mass,
            opts.nsc_ratio_bh_num_star_num,
            opts.nsc_ratio_bh_mass_star_mass,
            opts.disk_star_scale_factor,
            opts.nsc_radius_outer,
            opts.nsc_density_index_outer,
            opts.smbh_mass,
            opts.disk_radius_outer,
            opts.disk_aspect_ratio_avg,
            opts.nsc_radius_crit,
            opts.nsc_density_index_inner,
        )

    if opts.flag_coalesce_initial_stars:
        print("num stars initial", star_num_initial)

        # Generate initial masses for the initial number of stars, pre-Hill sphere mergers
        masses_initial = setupdiskstars.setup_disk_stars_masses(star_num=star_num_initial,
                                                                disk_star_mass_min_init=opts.disk_star_mass_min_init,
                                                                disk_star_mass_max_init=opts.disk_star_mass_max_init,
                                                                nsc_imf_star_powerlaw_index=opts.nsc_imf_star_powerlaw_index)

        orbs_a_initial = setupdiskstars.setup_disk_stars_orb_a(star_num_initial, opts.disk_radius_outer, opts.disk_inner_stable_circ_orb)

        # Sort the mass and location arrays by the location array
        sort_idx = np.argsort(orbs_a_initial)
        orbs_a_initial_sorted = orbs_a_initial[sort_idx]
        masses_initial_sorted = masses_initial[sort_idx]
        masses_stars, orbs_a_stars = diskstars_hillspheremergers.hillsphere_mergers(n_stars=star_num_initial,
                                                                                    masses_initial_sorted=masses_initial_sorted,
                                                                                    orbs_a_initial_sorted=orbs_a_initial_sorted,
                                                                                    min_initial_star_mass=opts.disk_star_mass_min_init,
                                                                                    disk_radius_outer=opts.disk_radius_outer,
                                                                                    smbh_mass=opts.smbh_mass,
                                                                                    P_m=1.35,
                                                                                    P_r=1.)
    else:
        masses_stars = setupdiskstars.setup_disk_stars_masses(star_num=star_num_initial,
                                                              disk_star_mass_min_init=opts.disk_star_mass_min_init,
                                                              disk_star_mass_max_init=opts.disk_star_mass_max_init,
                                                              nsc_imf_star_powerlaw_index=opts.nsc_imf_star_powerlaw_index)
        orbs_a_stars = setupdiskstars.setup_disk_stars_orb_a(star_num_initial, opts.disk_radius_outer, opts.disk_inner_stable_circ_orb)

    star_num = len(masses_stars)

    if (opts.flag_initial_stars_BH_immortal == 0):
        # Stars over disk_star_initial_mass_cutoff will be held at disk_star_initial_mass_cutoff and be immortal
        masses_stars[masses_stars > opts.disk_star_initial_mass_cutoff] = opts.disk_star_initial_mass_cutoff

    star_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(star_num)
    star_orb_inc = setupdiskstars.setup_disk_stars_inc(star_num, orbs_a_stars, star_orb_ang_mom, disk_aspect_ratio)
    star_orb_arg_periapse = setupdiskstars.setup_disk_stars_arg_periapse(star_num)
    if opts.flag_orb_ecc_damping == 1:
        star_orb_ecc = setupdiskstars.setup_disk_stars_eccentricity_uniform(star_num, opts.disk_bh_orb_ecc_max_init)
    else:
        star_orb_ecc = setupdiskstars.setup_disk_stars_circularized(star_num, opts.disk_bh_pro_orb_ecc_crit)

    star_X, star_Y, star_Z = setupdiskstars.setup_disk_stars_comp(star_num=star_num,
                                                                  star_ZAMS_metallicity=opts.nsc_star_metallicity_z_init,
                                                                  star_ZAMS_helium=opts.nsc_star_metallicity_y_init)
    log_radius, log_luminosity, log_teff = stellar_interpolation.interp_star_params(masses_stars)

    stars = AGNStar(mass=masses_stars,
                    orb_a=orbs_a_stars,
                    orb_inc=star_orb_inc,
                    orb_ecc=star_orb_ecc,
                    orb_ang_mom=star_orb_ang_mom,
                    orb_arg_periapse=star_orb_arg_periapse,
                    log_radius=log_radius,
                    log_luminosity=log_luminosity,
                    log_teff=log_teff,
                    star_X=star_X,
                    star_Y=star_Y,
                    star_Z=star_Z,
                    galaxy=np.full(star_num, galaxy),
                    time_passed=np.zeros(star_num),
                    id_start_val=id_start_val,
                    star_num=star_num)

    return (stars, star_num)
