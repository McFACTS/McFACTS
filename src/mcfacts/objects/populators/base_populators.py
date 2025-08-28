import numpy as np
from numpy.random import Generator

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.objects.galaxy import GalaxyPopulator
from mcfacts.utilities.random_state import uuid_provider
from mcfacts.objects.agn_object_array import AGNObjectArray, AGNBlackHoleArray
from mcfacts.setup import setupdiskblackholes


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

        bh_orb_a_initial = setupdiskblackholes.setup_disk_blackholes_location_NSC_powerlaw(
            disk_bh_num, sm.disk_radius_outer, sm.disk_inner_stable_circ_orb,
            sm.smbh_mass, sm.nsc_radius_crit, sm.nsc_density_index_inner,
            sm.nsc_density_index_outer, volume_scaling=True)

        bh_mass_initial = setupdiskblackholes.setup_disk_blackholes_masses(
            disk_bh_num,
            sm.nsc_imf_bh_mode, sm.nsc_imf_bh_mass_max, sm.nsc_imf_bh_powerlaw_index, sm.mass_pile_up, sm.nsc_imf_bh_method)

        bh_spin_initial = setupdiskblackholes.setup_disk_blackholes_spins(
            disk_bh_num,
            sm.nsc_bh_spin_dist_mu, sm.nsc_bh_spin_dist_sigma)

        bh_spin_angle_initial = setupdiskblackholes.setup_disk_blackholes_spin_angles(
            disk_bh_num,
            bh_spin_initial)

        bh_orb_ang_mom_initial = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(
            disk_bh_num)

        if sm.flag_orb_ecc_damping == 1:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(disk_bh_num, sm.disk_bh_orb_ecc_max_init)
        else:
            bh_orb_ecc_initial = setupdiskblackholes.setup_disk_blackholes_circularized(disk_bh_num, sm.disk_bh_pro_orb_ecc_crit)

        bh_orb_inc_initial = setupdiskblackholes.setup_disk_blackholes_incl(disk_bh_num, bh_orb_a_initial, bh_orb_ang_mom_initial, agn_disk.disk_aspect_ratio)
        bh_orb_arg_periapse_initial = setupdiskblackholes.setup_disk_blackholes_arg_periapse(disk_bh_num)

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

