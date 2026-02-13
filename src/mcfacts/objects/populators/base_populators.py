import numpy as np
from numpy.random import Generator

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.modules import stellar_interpolation
from mcfacts.objects.galaxy import GalaxyPopulator
from mcfacts.utilities.random_state import uuid_provider
from mcfacts.objects.agn_object_array import AGNObjectArray, AGNBlackHoleArray, AGNStarArray
from mcfacts.setup import setupdiskblackholes, initializediskstars, setupdiskstars, diskstars_hillspheremergers


class SingleBlackHolePopulator(GalaxyPopulator):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__(settings.bh_array_name if name is None else name, settings)

    def populate(self, agn_disk: AGNDisk, random_generator: Generator) -> AGNObjectArray:
        sm = self.settings

        disk_bh_num = setupdiskblackholes.setup_disk_nbh(
            sm.nsc_mass,
            sm.nsc_ratio_bh_num_star_num,
            sm.nsc_ratio_bh_mass_star_mass,
            sm.nsc_radius_outer,
            sm.nsc_density_index_outer,
            sm.smbh_mass,
            sm.disk_radius_outer,
            sm.disk_aspect_ratio_avg,
            sm.nsc_radius_crit,
            sm.nsc_density_index_inner,
        )

        bh_orb_a_initial = setupdiskblackholes.setup_disk_blackholes_location_NSC_powerlaw(
            disk_bh_num, sm.disk_radius_outer, sm.disk_inner_stable_circ_orb,
            sm.smbh_mass, sm.nsc_radius_crit, sm.nsc_density_index_inner,
            sm.nsc_density_index_outer,
            random_generator,
            volume_scaling = True
        )

        bh_mass_initial = setupdiskblackholes.setup_disk_blackholes_masses(
            disk_bh_num,
            sm.nsc_imf_bh_mode,
            sm.nsc_imf_bh_mass_max,
            sm.nsc_imf_bh_powerlaw_index,
            sm.mass_pile_up,
            sm.nsc_imf_bh_method,
            random_generator
        )

        bh_spin_initial = setupdiskblackholes.setup_disk_blackholes_spins(
            disk_bh_num,
            sm.nsc_bh_spin_dist_mu,
            sm.nsc_bh_spin_dist_sigma,
            random_generator
        )

        bh_spin_angle_initial = setupdiskblackholes.setup_disk_blackholes_spin_angles(
            disk_bh_num,
            bh_spin_initial,
            random_generator
        )

        bh_orb_ang_mom_initial = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(
            disk_bh_num, random_generator
        )

        if sm.flag_orb_ecc_damping == 1:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(
                disk_bh_num,
                sm.disk_bh_orb_ecc_max_init,
                random_generator
            )
        else:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_circularized(
                disk_bh_num,
                sm.disk_bh_pro_orb_ecc_crit
            )

        bh_orb_inc_initial = setupdiskblackholes.setup_disk_blackholes_incl(
            disk_bh_num, bh_orb_a_initial,
            bh_orb_ang_mom_initial,
            agn_disk.disk_aspect_ratio,
            random_generator
        )

        bh_orb_arg_periapse_initial = setupdiskblackholes.setup_disk_blackholes_arg_periapse(
            disk_bh_num,
            random_generator
        )

        unique_ids = np.array([uuid_provider(random_generator) for _ in range(disk_bh_num)])

        return AGNBlackHoleArray(
            unique_id=unique_ids,
            mass=bh_mass_initial,
            spin=bh_spin_initial,
            spin_angle=bh_spin_angle_initial,
            orb_a=bh_orb_a_initial,
            orb_inc=bh_orb_inc_initial,
            orb_ecc=bh_orb_ecc_initial,
            orb_ang_mom=bh_orb_ang_mom_initial,
            orb_arg_periapse=bh_orb_arg_periapse_initial,
        )


class SingleStarPopulator(GalaxyPopulator):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__(settings.star_array_name if name is None else name, settings)

    def populate(self, agn_disk: AGNDisk, random_generator: Generator) -> AGNObjectArray:
        sm = self.settings

        # Initialize stars

        # Generate initial number of stars
        star_num_initial = setupdiskstars.setup_disk_stars_num(
            sm.nsc_mass,
            sm.nsc_ratio_bh_num_star_num,
            sm.nsc_ratio_bh_mass_star_mass,
            sm.disk_star_scale_factor,
            sm.nsc_radius_outer,
            sm.nsc_density_index_outer,
            sm.smbh_mass,
            sm.disk_radius_outer,
            sm.disk_aspect_ratio_avg,
            sm.nsc_radius_crit,
            sm.nsc_density_index_inner,
        )

        if sm.flag_coalesce_initial_stars:
            self.log(f"num stars initial {star_num_initial}")

            # Generate initial masses for the initial number of stars, pre-Hill sphere mergers
            masses_initial = setupdiskstars.setup_disk_stars_masses(star_num=star_num_initial,
                                                                    disk_star_mass_min_init=sm.disk_star_mass_min_init,
                                                                    disk_star_mass_max_init=sm.disk_star_mass_max_init,
                                                                    nsc_imf_star_powerlaw_index=sm.nsc_imf_star_powerlaw_index)

            orbs_a_initial = setupdiskstars.setup_disk_stars_orb_a(star_num_initial, sm.disk_radius_outer,
                                                                   sm.disk_inner_stable_circ_orb)

            # Sort the mass and location arrays by the location array
            sort_idx = np.argsort(orbs_a_initial)
            orbs_a_initial_sorted = orbs_a_initial[sort_idx]
            masses_initial_sorted = masses_initial[sort_idx]
            masses_stars, orbs_a_stars = diskstars_hillspheremergers.hillsphere_mergers(n_stars=star_num_initial,
                                                                                        masses_initial_sorted=masses_initial_sorted,
                                                                                        orbs_a_initial_sorted=orbs_a_initial_sorted,
                                                                                        min_initial_star_mass=sm.disk_star_mass_min_init,
                                                                                        disk_radius_outer=sm.disk_radius_outer,
                                                                                        smbh_mass=sm.smbh_mass,
                                                                                        P_m=1.35,
                                                                                        P_r=2.)
            self.log(f"After coalescing stars: max star mass is {np.round(masses_stars.max(), 2)} Msun\n\t{np.sum(masses_stars > sm.disk_star_initial_mass_cutoff)} stars over immortal limit ({sm.disk_star_initial_mass_cutoff} Msun)")

        else:
            masses_stars = setupdiskstars.setup_disk_stars_masses(star_num=star_num_initial,
                                                                  disk_star_mass_min_init=sm.disk_star_mass_min_init,
                                                                  disk_star_mass_max_init=sm.disk_star_mass_max_init,
                                                                  nsc_imf_star_powerlaw_index=sm.nsc_imf_star_powerlaw_index,
                                                                  random=random_generator)
            orbs_a_stars = setupdiskstars.setup_disk_stars_orb_a(star_num_initial, sm.disk_radius_outer,
                                                                 sm.disk_inner_stable_circ_orb,
                                                                 random=random_generator)

        star_num = len(masses_stars)

        if sm.flag_initial_stars_BH_immortal == 0:
            # Stars over disk_star_initial_mass_cutoff will be held at disk_star_initial_mass_cutoff and be immortal
            masses_stars[masses_stars > sm.disk_star_initial_mass_cutoff] = sm.disk_star_initial_mass_cutoff

        star_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(star_num, random_generator)
        star_orb_inc = setupdiskstars.setup_disk_stars_inc(star_num, orbs_a_stars, star_orb_ang_mom, agn_disk.disk_aspect_ratio, random_generator)
        star_orb_arg_periapse = setupdiskstars.setup_disk_stars_arg_periapse(star_num, random_generator)

        if sm.flag_orb_ecc_damping == 1:
            star_orb_ecc = setupdiskstars.setup_disk_stars_eccentricity_uniform(star_num, sm.disk_bh_orb_ecc_max_init, random_generator)
        else:
            star_orb_ecc = setupdiskstars.setup_disk_stars_circularized(star_num, sm.disk_bh_pro_orb_ecc_crit)

        star_X, star_Y, star_Z = setupdiskstars.setup_disk_stars_comp(star_num=star_num,
                                                                      star_ZAMS_metallicity=sm.nsc_star_metallicity_z_init,
                                                                      star_ZAMS_helium=sm.nsc_star_metallicity_y_init)
        log_radius, log_luminosity, log_teff = stellar_interpolation.interp_star_params(masses_stars)

        unique_ids = np.array([uuid_provider(random_generator) for _ in range(star_num)])

        return AGNStarArray(
            unique_id=unique_ids,
            mass=masses_stars,
            orb_a=orbs_a_stars,
            orb_inc=star_orb_inc,
            orb_ecc=star_orb_ecc,
            orb_ang_mom=star_orb_ang_mom,
            orb_arg_periapse=star_orb_arg_periapse,
            log_radius=log_radius,
            log_luminosity=log_luminosity,
            log_teff=log_teff,
            star_x=star_X,
            star_y=star_Y,
            star_z=star_Z,
        )

        # TODO: Precalculate star info

        # Create arrays to keep track of initial star values
        stars_all_id_nums = stars.id_num
        stars_masses_initial = stars.mass
        stars_orb_a_initial = stars.orb_a

        # if opts.flag_add_stars:
        #     # Pre-calculating stars captured from NSC
        #     captured_star_mass_total = disk_capture_stars.stellar_mass_captured_nsc(time_final, opts.smbh_mass, opts.nsc_density_index_inner, opts.nsc_mass, opts.nsc_ratio_bh_num_star_num, opts.nsc_ratio_bh_mass_star_mass, disk_surface_density, opts.disk_star_mass_min_init, opts.disk_star_mass_max_init, opts.nsc_imf_star_powerlaw_index)
        #     captured_stars_masses = disk_capture_stars.setup_captured_stars_masses(captured_star_mass_total, opts.disk_star_mass_min_init, opts.disk_star_mass_max_init, opts.nsc_imf_star_powerlaw_index)
        #     captured_stars_orbs_a = disk_capture_stars.setup_captured_stars_orbs_a(len(captured_stars_masses), time_final, opts.smbh_mass, disk_surface_density, opts.disk_star_mass_min_init, opts.disk_star_mass_max_init, opts.nsc_imf_star_powerlaw_index)
        #     captured_stars = disk_capture_stars.distribute_captured_stars(captured_stars_masses, captured_stars_orbs_a, opts.timestep_num, opts.timestep_duration_yr)
        #     print(f"Capturing ~{int(np.rint(len(captured_stars_masses) / opts.timestep_num))} NSC stars per timestep / ~{int(np.rint(len(captured_stars_masses) / time_final * 1e5))} NSC stars per 0.1 Myr")
        # # Writing initial parameters to file
        # if opts.flag_add_stars:
        #     stars.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/initial_params_star.dat"))
        # blackholes.to_txt(os.path.join(opts.work_directory, f"gal{galaxy_zfilled_str}/initial_params_bh.dat"))
