import numpy as np
from numpy.random import Generator

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.mcfacts_random_state import uuid_provider
from mcfacts.objects.agn_object_array import AGNBlackHoleArray
from mcfacts.objects.galaxy import FilingCabinet, AGNDisk
from mcfacts.objects.timeline import TimelineActor
from mcfacts.physics import feedback, migration, accretion, eccentricity, dynamics, disk_capture
from mcfacts.setup import setupdiskblackholes


class BlackHoleMigration(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__("Black Hole Migration" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        self.log("Logging from inside the timeline actor!")

        # Check to make sure the array exists in the filing cabinet
        if sm.bh_prograde_array_name not in filing_cabinet:
            return

        blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)

        # First if feedback present, find ratio of feedback heating torque to migration torque
        if sm.flag_thermal_feedback > 0:
            ratio_heat_mig_torques = feedback.feedback_bh_hankla(
                blackholes_pro.orb_a,
                agn_disk.disk_surface_density,
                agn_disk.disk_opacity,
                sm.disk_bh_eddington_ratio,
                sm.disk_alpha_viscosity,
                sm.disk_radius_outer
            )
        else:
            ratio_heat_mig_torques = np.ones(blackholes_pro.num)

        # Set empty variable, we'll fill it based on torque_prescription
        new_orb_a_bh = None

        # Choose your torque_prescription

        # Old is the original approximation used in v.0.1.0, based off (but not identical to Paardekooper 2010)-usually within factor [0.5-2]
        if sm.torque_prescription == 'old':
            # Old migration prescription
            new_orb_a_bh = migration.type1_migration_single(
                sm.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                blackholes_pro.orb_ecc,
                sm.disk_bh_pro_orb_ecc_crit,
                agn_disk.disk_surface_density,
                agn_disk.disk_aspect_ratio,
                ratio_heat_mig_torques,
                sm.disk_radius_trap,
                sm.disk_radius_outer,
                sm.timestep_duration_yr
            )

        # Normalized torque (multiplies torque coeff)
        if sm.torque_prescription == 'paardekooper' or sm.torque_prescription == 'jimenez_masset':
            normalized_torque_bh = migration.normalized_torque(
                sm.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                blackholes_pro.orb_ecc,
                sm.disk_bh_pro_orb_ecc_crit,
                agn_disk.disk_surface_density,
                agn_disk.disk_aspect_ratio
            )

            torque_bh = None
            disk_trap_radius = None
            disk_anti_trap_radius = None

            # Paardekooper torque coeff (default)
            if sm.torque_prescription == 'paardekooper':
                paardekooper_torque_coeff_bh = migration.paardekooper10_torque(
                    blackholes_pro.orb_a,
                    blackholes_pro.orb_ecc,
                    sm.disk_bh_pro_orb_ecc_crit,
                    agn_disk.disk_dlog10surfdens_dlog10R_func,
                    agn_disk.disk_dlog10temp_dlog10R_func
                )

                torque_bh = paardekooper_torque_coeff_bh * normalized_torque_bh

                disk_trap_radius = sm.disk_radius_trap
                disk_anti_trap_radius = sm.disk_radius_trap

            # Jimenez-Masset torque coeff (from Grishin+24)
            if sm.torque_prescription == 'jimenez_masset':
                jimenez_masset_torque_coeff_bh = migration.jimenezmasset17_torque(
                    sm.smbh_mass,
                    agn_disk.disk_surface_density,
                    agn_disk.disk_opacity,
                    agn_disk.disk_aspect_ratio,
                    agn_disk.temp_func,
                    blackholes_pro.orb_a,
                    blackholes_pro.orb_ecc,
                    sm.disk_bh_pro_orb_ecc_crit,
                    agn_disk.disk_dlog10surfdens_dlog10R_func,
                    agn_disk.disk_dlog10temp_dlog10R_func
                )

                # Thermal torque from JM17 (if flag_thermal_feedback off, this component is 0.)
                jimenez_masset_thermal_torque_coeff_bh = migration.jimenezmasset17_thermal_torque_coeff(
                    sm.smbh_mass,
                    agn_disk.disk_surface_density,
                    agn_disk.disk_opacity,
                    agn_disk.disk_aspect_ratio,
                    agn_disk.temp_func,
                    sm.disk_bh_eddington_ratio,
                    blackholes_pro.orb_a,
                    blackholes_pro.orb_ecc,
                    sm.disk_bh_pro_orb_ecc_crit,
                    blackholes_pro.mass,
                    sm.flag_thermal_feedback,
                    agn_disk.disk_dlog10pressure_dlog10R_func
                )

                if sm.flag_thermal_feedback == 1:
                    total_jimenez_masset_torque_bh = jimenez_masset_torque_coeff_bh + jimenez_masset_thermal_torque_coeff_bh
                else:
                    total_jimenez_masset_torque_bh = jimenez_masset_torque_coeff_bh

                torque_bh = total_jimenez_masset_torque_bh * normalized_torque_bh

                # Set up trap scaling as a function of mass for Jimenez-Masset (for SG-like disk)
                # No traps if M_smbh >10^8Msun (approx.)
                if sm.smbh_mass > 1.e8:
                    disk_trap_radius = sm.disk_inner_stable_circ_orb
                    disk_anti_trap_radius = sm.disk_inner_stable_circ_orb
                if sm.smbh_mass == 1.e8:
                    disk_trap_radius = sm.disk_radius_trap
                    disk_anti_trap_radius = sm.disk_radius_trap
                # Trap changes as a function of r_g if M_smbh <10^8Msun (default trap radius ~700r_g). Grishin+24
                if sm.smbh_mass < 1.e8 and sm.smbh_mass > 1.e6:
                    disk_trap_radius = sm.disk_radius_trap * (sm.smbh_mass / 1.e8) ** (-1.225)
                    disk_anti_trap_radius = sm.disk_radius_trap * (sm.smbh_mass / 1.e8) ** (0.099)
                # Trap location changes again at low SMBH mass (Grishin+24)
                if sm.smbh_mass < 1.e6:
                    disk_trap_radius = sm.disk_radius_trap * (sm.smbh_mass / 1.e8) ** (-0.97)
                    disk_anti_trap_radius = sm.disk_radius_trap * (sm.smbh_mass / 1.e8) ** (0.099)

            # Timescale on which migration happens based on overall torque
            torque_mig_timescales_bh = migration.torque_mig_timescale(
                sm.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                blackholes_pro.orb_ecc,
                sm.disk_bh_pro_orb_ecc_crit,
                torque_bh
            )

            # Calculate new bh_orbs_a using torque (here including details from Jimenez & Masset '17 & Grishin+'24)
            new_orb_a_bh = migration.type1_migration_distance(
                sm.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                blackholes_pro.orb_ecc,
                sm.disk_bh_pro_orb_ecc_crit,
                torque_mig_timescales_bh,
                ratio_heat_mig_torques,
                disk_trap_radius,
                disk_anti_trap_radius,
                sm.disk_radius_outer,
                sm.timestep_duration_yr,
                sm.flag_phenom_turb,
                sm.phenom_turb_centroid,
                sm.phenom_turb_std_dev,
                sm.nsc_imf_bh_mode,
                sm.torque_prescription
            )

        # TODO: Reconsider for physicality
        # I'm not sure why where have this filter/check if we overwrite everything right after?
        # Was this supposed to be some sort of edge detection?
        blackholes_pro.orb_a = np.where(blackholes_pro.orb_a > sm.disk_inner_stable_circ_orb, blackholes_pro.orb_a, 3 * sm.disk_inner_stable_circ_orb)

        blackholes_pro.orb_a = new_orb_a_bh

        blackholes_pro.consistency_check()

class BlackHoleAccretion(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__("Black Hole Accretion" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if sm.bh_prograde_array_name not in filing_cabinet:
            return

        blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)

        blackholes_pro.mass = accretion.change_bh_mass(
            blackholes_pro.mass,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_eddington_mass_growth_rate,
            sm.timestep_duration_yr
        )

        blackholes_pro.spin = accretion.change_bh_spin_magnitudes(
            blackholes_pro.spin,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_torque_condition,
            sm.timestep_duration_yr,
            blackholes_pro.orb_ecc,
            sm.disk_bh_pro_orb_ecc_crit,
        )

        blackholes_pro.spin_angle = accretion.change_bh_spin_angles(
            blackholes_pro.spin_angle,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_torque_condition,
            sm.disk_bh_spin_resolution_min,
            sm.timestep_duration_yr,
            blackholes_pro.orb_ecc,
            sm.disk_bh_pro_orb_ecc_crit
        )

        # TODO: Move based on cause leading to physical effect?
        # Is this dampening due to accretion, or should it be moved to a different module?
        blackholes_pro.orb_ecc = eccentricity.orbital_ecc_damping(
            sm.smbh_mass,
            blackholes_pro.orb_a,
            blackholes_pro.mass,
            agn_disk.disk_surface_density,
            agn_disk.disk_aspect_ratio,
            blackholes_pro.orb_ecc,
            sm.timestep_duration_yr,
            sm.disk_bh_pro_orb_ecc_crit,
        )

        blackholes_pro.consistency_check()


# class ProgradeBlackHoleDynamics(TimelineActor):
#     def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
#         super().__init__("Prograde Black Hole Dynamics" if name is None else name, settings)
#
#     def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
#         sm = self.settings
#
#         # Check to make sure the array exists in the filing cabinet
#         if sm.bh_prograde_array_name not in filing_cabinet:
#             return
#
#         blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)
#
#         # Migrate
#         # First if feedback present, find ratio of feedback heating torque to migration torque
#         if sm.flag_thermal_feedback > 0:
#             ratio_heat_mig_torques = feedback.feedback_bh_hankla(
#                 blackholes_pro.orb_a,
#                 agn_disk.disk_surface_density,
#                 agn_disk.disk_opacity,
#                 sm.disk_bh_eddington_ratio,
#                 sm.disk_alpha_viscosity,
#                 sm.disk_radius_outer)
#         else:
#             ratio_heat_mig_torques = np.ones(len(blackholes_pro.orb_a))
#
#         # Migrate as usual
#         blackholes_pro.orb_a = migration.type1_migration_single(
#             sm.smbh_mass,
#             blackholes_pro.orb_a,
#             blackholes_pro.mass,
#             blackholes_pro.orb_ecc,
#             sm.disk_bh_pro_orb_ecc_crit,
#             agn_disk.disk_surface_density,
#             agn_disk.disk_aspect_ratio,
#             ratio_heat_mig_torques,
#             sm.disk_radius_trap,
#             sm.disk_radius_outer,
#             sm.timestep_duration_yr
#         )
#
#         # Check for orb_a unphysical
#         bh_pro_id_num_unphysical_a = blackholes_pro.unique_id[blackholes_pro.orb_a == 0.0]
#         blackholes_pro.remove_all(bh_pro_id_num_unphysical_a)
#
#         # Accrete
#         blackholes_pro.mass = accretion.change_bh_mass(
#             blackholes_pro.mass,
#             sm.disk_bh_eddington_ratio,
#             sm.disk_bh_eddington_mass_growth_rate,
#             sm.timestep_duration_yr
#         )
#
#         # Spin up
#         blackholes_pro.spin = accretion.change_bh_spin_magnitudes(
#             blackholes_pro.spin,
#             sm.disk_bh_eddington_ratio,
#             sm.disk_bh_torque_condition,
#             sm.timestep_duration_yr,
#             blackholes_pro.orb_ecc,
#             sm.disk_bh_pro_orb_ecc_crit,
#         )
#
#         # Torque spin angle
#         blackholes_pro.spin_angle = accretion.change_bh_spin_angles(
#             blackholes_pro.spin_angle,
#             sm.disk_bh_eddington_ratio,
#             sm.disk_bh_torque_condition,
#             sm.disk_bh_spin_resolution_min,
#             sm.timestep_duration_yr,
#             blackholes_pro.orb_ecc,
#             sm.disk_bh_pro_orb_ecc_crit
#         )
#
#         # Damp BH orbital eccentricity
#         blackholes_pro.orb_ecc = eccentricity.orbital_ecc_damping(
#             sm.smbh_mass,
#             blackholes_pro.orb_a,
#             blackholes_pro.mass,
#             agn_disk.disk_surface_density,
#             agn_disk.disk_aspect_ratio,
#             blackholes_pro.orb_ecc,
#             sm.timestep_duration_yr,
#             sm.disk_bh_pro_orb_ecc_crit,
#         )
#
#         if sm.flag_dynamic_enc > 0:
#             blackholes_pro.orb_a, blackholes_pro.orb_ecc = dynamics.circular_singles_encounters_prograde(
#                 sm.smbh_mass,
#                 blackholes_pro.orb_a,
#                 blackholes_pro.mass,
#                 blackholes_pro.orb_ecc,
#                 sm.timestep_duration_yr,
#                 sm.disk_bh_pro_orb_ecc_crit,
#                 sm.delta_energy_strong,
#             )
#
#         # After this time period, was there a disk capture via orbital grind-down?
#         # To do: What eccentricity do we want the captured BH to have? Right now ecc=0.0? Should it be ecc<h at a?
#         # Assume 1st gen BH captured and orb ecc =0.0
#         # To do: Bias disk capture to more massive BH!
#         capture = time_passed % sm.capture_time_yr
#         if capture == 0:
#             disk_bh_num = 1
#
#             bh_orb_a_captured = setupdiskblackholes.setup_disk_blackholes_location(disk_bh_num, sm.disk_radius_capture_outer, sm.disk_inner_stable_circ_orb)
#             bh_mass_captured = setupdiskblackholes.setup_disk_blackholes_masses(disk_bh_num, sm.nsc_imf_bh_mode, sm.nsc_imf_bh_mass_max, sm.nsc_imf_bh_powerlaw_index, sm.mass_pile_up)
#             bh_spin_captured = setupdiskblackholes.setup_disk_blackholes_spins(disk_bh_num, sm.nsc_bh_spin_dist_mu, sm.nsc_bh_spin_dist_sigma)
#             bh_spin_angle_captured = setupdiskblackholes.setup_disk_blackholes_spin_angles(disk_bh_num, bh_spin_captured)
#
#             unique_ids = np.array([uuid_provider(random_generator) for _ in range(disk_bh_num)])
#
#             captured_prograde_bh = AGNBlackHoleArray(
#                 unique_id=unique_ids,
#                 mass=bh_mass_captured,
#                 spin=bh_spin_captured,
#                 spin_angle=bh_spin_angle_captured,
#                 orb_a=bh_orb_a_captured,
#                 orb_inc=np.full(disk_bh_num, 0.0),
#                 orb_ecc=np.full(disk_bh_num, 0.0),
#                 orb_ang_mom=np.ones(disk_bh_num),
#                 orb_arg_periapse=np.full(disk_bh_num, -1.5),
#                 gen=np.ones(disk_bh_num)
#             )
#
#             blackholes_pro.add_objects(captured_prograde_bh)
#
#         # Test if any BH or BBH are in the danger-zone (<mininum_safe_distance, default =50r_g) from SMBH.
#         # Potential EMRI/BBH EMRIs.
#         # Find prograde BH in inner disk. Define inner disk as <=50r_g.
#         # Since a 10Msun BH will decay into a 10^8Msun SMBH at 50R_g in ~38Myr and decay time propto a^4.
#         # e.g at 25R_g, decay time is only 2.3Myr.
#
#         # Remove the pros if it is in the inner disk and add it to the inner disk array.
#         pro_bh_inner_disk_ids = blackholes_pro.unique_id[blackholes_pro.orb_a < sm.inner_disk_outer_radius]
#
#         inner_pros = blackholes_pro.copy()
#         inner_pros.keep_only(pro_bh_inner_disk_ids)
#         filing_cabinet.create_or_append_array(sm.bh_inner_disk_array_name, inner_pros)
#
#         blackholes_pro.remove_all(pro_bh_inner_disk_ids)
#
#
# class RetrogradeBlackholeDynamics(TimelineActor):
#     def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
#         super().__init__("Retrograde Black Hole Dynamics" if name is None else name, settings)
#
#     def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
#         sm = self.settings
#
#         # Check to make sure the array exists in the filing cabinet
#         if sm.bh_retrograde_array_name not in filing_cabinet:
#             return
#
#         blackholes_retro = filing_cabinet.get_array(sm.bh_retrograde_array_name, AGNBlackHoleArray)
#
#         orb_ecc, orb_a, orb_inc = disk_capture.retro_bh_orb_disk_evolve(
#             sm.smbh_mass,
#             blackholes_retro.mass,
#             blackholes_retro.orb_a,
#             blackholes_retro.orb_ecc,
#             blackholes_retro.orb_inc,
#             blackholes_retro.orb_arg_periapse,
#             sm.disk_inner_stable_circ_orb,
#             agn_disk.disk_surface_density,
#             timestep_length
#         )
#
#         # Update retro black hole object array attributes
#         blackholes_retro.orb_ecc = np.array(orb_ecc)
#         blackholes_retro.orb_a = np.array(orb_a)
#         blackholes_retro.orb_inc = np.array(orb_inc)
#
#         # Check for unphysical orbital eccentricities
#         bh_retro_id_num_unphysical_ecc = blackholes_retro.unique_id[blackholes_retro.orb_ecc >= 1.]
#         blackholes_retro.remove_all(bh_retro_id_num_unphysical_ecc)
#
#         # Remove the retro if it is in the inner disk and add it to the inner disk array.
#         retro_bh_inner_disk_ids = blackholes_retro.unique_id[blackholes_retro.orb_a < sm.inner_disk_outer_radius]
#
#         inner_retros = blackholes_retro.copy()
#         inner_retros.keep_only(retro_bh_inner_disk_ids)
#         filing_cabinet.create_or_append_array(sm.bh_inner_disk_array_name, inner_retros)
#
#         blackholes_retro.remove_all(retro_bh_inner_disk_ids)
#
#         # Remove any retros that have flipped prograde and add them to the prograde black holes array
#         inc_threshhold = 5.0 * np.pi / 180.0
#
#         bh_id_num_flip_to_pro = blackholes_retro.unique_id[np.where((np.abs(blackholes_retro.orb_inc) <= inc_threshhold) | (blackholes_retro.orb_ecc == 0.0))]
#
#         blackholes_pro = blackholes_retro.copy()
#         blackholes_pro.keep_only(bh_id_num_flip_to_pro)
#         filing_cabinet.create_or_append_array(sm.bh_prograde_array_name, blackholes_pro)
#
#         blackholes_retro.remove_all(bh_id_num_flip_to_pro)