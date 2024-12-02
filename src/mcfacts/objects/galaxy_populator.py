from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from mcfacts.inputs import ReadInputs
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.mcfacts_random_state import uuid_provider
from mcfacts.objects.agn_disk import AGNDisk
from mcfacts.objects.agn_object_array import AGNBlackHoleArray, AGNStarArray
from mcfacts.objects.agnobject import AGNStar
from mcfacts.objects.galaxy import AGNObjectArray
from mcfacts.setup import setupdiskblackholes, diskstars_hillspheremergers, setupdiskstars


class GalaxyPopulator(ABC):
    def __init__(self, name: str, settings: SettingsManager = SettingsManager()):
        self.name: str = name
        self.settings_manager: SettingsManager = settings

    @abstractmethod
    def populate(self, agn_disk: AGNDisk, random_generator: Generator) -> AGNObjectArray:
        return NotImplemented


class TestPopulator(GalaxyPopulator):

    def populate(self, agn_disk: AGNDisk, random_generator: Generator) -> AGNBlackHoleArray:
        return AGNBlackHoleArray()


class SingleBlackHolePopulator(GalaxyPopulator):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__(settings.bh_array_name if name is None else name, settings)

    def populate(self, agn_disk: AGNDisk, random_generator: Generator) -> AGNObjectArray:
        sm = self.settings_manager

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

        bh_orb_a_initial = setupdiskblackholes.setup_disk_blackholes_location(disk_bh_num, sm.disk_radius_outer, sm.disk_inner_stable_circ_orb)

        bh_mass_initial = setupdiskblackholes.setup_disk_blackholes_masses(disk_bh_num, sm.nsc_imf_bh_mode, sm.nsc_imf_bh_mass_max, sm.nsc_imf_bh_powerlaw_index, sm.mass_pile_up)
        bh_spin_initial = setupdiskblackholes.setup_disk_blackholes_spins(disk_bh_num, sm.nsc_bh_spin_dist_mu, sm.nsc_bh_spin_dist_sigma)
        bh_spin_angle_initial = setupdiskblackholes.setup_disk_blackholes_spin_angles(disk_bh_num, bh_spin_initial)
        bh_orb_ang_mom_initial = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(disk_bh_num)

        if sm.flag_orb_ecc_damping == 1:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(disk_bh_num, sm.disk_bh_orb_ecc_max_init)
        else:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_circularized(disk_bh_num, sm.disk_bh_pro_orb_ecc_crit)

        bh_orb_inc_initial = setupdiskblackholes.setup_disk_blackholes_incl(disk_bh_num, bh_orb_a_initial, bh_orb_ang_mom_initial, agn_disk.disk_aspect_ratio)
        bh_orb_arg_periapse_initial = setupdiskblackholes.setup_disk_blackholes_arg_periapse(disk_bh_num)

        bh_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(disk_bh_num)

        unique_ids = np.array([uuid_provider(random_generator) for _ in range(disk_bh_num)])

        return AGNBlackHoleArray(
            unique_id=unique_ids,
            mass=bh_mass_initial,
            spin=bh_spin_initial,
            spin_angle=bh_spin_angle_initial,
            orb_a=bh_orb_a_initial,
            orb_inc=bh_orb_inc_initial,
            orb_ecc=bh_orb_ecc_initial,
            orb_ang_mom=bh_orb_ang_mom,
            orb_arg_periapse=bh_orb_arg_periapse_initial,
        )


class SingleStarPopulator(GalaxyPopulator):
    def populate(self, agn_disk: AGNDisk, random_generator: Generator) -> AGNObjectArray:
        sm = self.settings_manager

        # Generate initial number of stars
        star_num_initial = setupdiskstars.setup_disk_stars_num(
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

        # giprint(star_num_initial) 152_248_329
        star_num_initial = 1_000_000

        # Generate initial masses for the initial number of stars, pre-Hill sphere mergers
        masses_initial = setupdiskstars.setup_disk_stars_masses(
            star_num=star_num_initial,
            disk_star_mass_min_init=sm.disk_star_mass_min_init,
            disk_star_mass_max_init=sm.disk_star_mass_max_init,
            nsc_imf_star_powerlaw_index=sm.nsc_imf_star_powerlaw_index
        )

        # Generating star locations in a x^2 distribution
        x_vals = random_generator.uniform(low=0.002, high=1, size=star_num_initial)

        r_locations_initial = np.sqrt(x_vals)
        r_locations_initial_scaled = r_locations_initial * sm.disk_radius_trap

        # Sort the mass and location arrays by the location array
        sort_idx = np.argsort(r_locations_initial_scaled)
        r_locations_initial_sorted = r_locations_initial_scaled[sort_idx]
        masses_initial_sorted = masses_initial[sort_idx]

        star_mass, star_orb_a = \
            diskstars_hillspheremergers.hillsphere_mergers(
                n_stars=star_num_initial,
                masses_initial_sorted=masses_initial_sorted,
                r_locations_initial_sorted=r_locations_initial_sorted,
                min_initial_star_mass=sm.disk_star_mass_min_init,
                R_disk=sm.disk_radius_trap,
                smbh_mass=sm.smbh_mass,
                P_m=1.35,
                P_r=1.
            )

        star_num = len(star_mass)

        # star_radius = setupdiskstars.setup_disk_stars_radius(masses_stars)
        star_spin = setupdiskstars.setup_disk_stars_spins(star_num, sm.nsc_star_spin_dist_mu, sm.nsc_star_spin_dist_sigma)
        star_spin_angle = setupdiskstars.setup_disk_stars_spin_angles(star_num, star_spin)
        star_orb_inc = setupdiskstars.setup_disk_stars_inclination(star_num)

        # star_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(rng,star_num)
        star_orb_arg_periapse = setupdiskstars.setup_disk_stars_arg_periapse(star_num)

        if sm.flag_orb_ecc_damping == 1:
            star_orb_ecc = setupdiskstars.setup_disk_stars_eccentricity_uniform(star_num)
        else:
            star_orb_ecc = setupdiskstars.setup_disk_stars_circularized(star_num, sm.disk_bh_pro_orb_ecc_crit)

        star_X, star_Y, star_Z = setupdiskstars.setup_disk_stars_comp(star_num=star_num, star_ZAMS_metallicity=sm.nsc_star_metallicity_z_init, star_ZAMS_helium=sm.nsc_star_metallicity_y_init)

        star_radius = setupdiskstars.setup_disk_stars_radius(star_mass)

        unique_ids = np.array([uuid_provider(random_generator) for _ in range(star_num_initial)])

        return AGNStarArray(
            unique_id=unique_ids,
            mass=star_mass,
            spin=star_spin,
            spin_angle=star_spin_angle,
            orb_a=star_orb_a,
            orb_inc=star_orb_inc,
            orb_ecc=star_orb_ecc,
            orb_arg_periapse=star_orb_arg_periapse,
            star_x=star_X,
            star_y=star_Y,
            star_z=star_Z,
            radius=star_radius
        )
