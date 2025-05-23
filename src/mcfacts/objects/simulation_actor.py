from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from numpy.random import Generator

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.mcfacts_random_state import uuid_provider
from mcfacts.objects.agn_disk import AGNDisk
from mcfacts.objects.agn_object_array import AGNBlackHoleArray, AGNBinaryBlackHoleArray, AGNMergedBlackHoleArray
from mcfacts.objects.filing_cabinet import FilingCabinet
from mcfacts.physics import migration, accretion, eccentricity, disk_capture, dynamics, gw, feedback
from mcfacts.physics.binary import evolve, merge, formation
from mcfacts.physics.gw import gw_strain_freq
from mcfacts.setup import setupdiskblackholes


class SimulationActor(ABC):
    """
    A base class representing a generic simulation actor in the simulation framework.

    Simulation actors define behaviors or actions to be performed during the simulation,
    and each actor is identified by a name and utilizes specific settings.

    Attributes:
        name (str): The name of the simulation actor.
        settings (SettingsManager): A manager for accessing configuration settings. Defaults to the base `SettingsManager`.

    Methods:
        perform(timestep: int, filing_cabinet: FilingCabinet, randomState: RandomState):
            Abstract method to be implemented by subclasses. Defines the behavior of the simulation actor for a given timestep.

        __str__():
            Returns a string representation of the simulation actor, including its name and type.
    """

    def __init__(self, name: str, settings: SettingsManager = SettingsManager()):
        """
        Initializes a SimulationActor instance.

        Args:
            name (str): The name of the simulation actor.
            settings (SettingsManager, optional): The settings manager to retrieve configurations for the actor.
                Defaults to an empty `SettingsManager`.
        """
        self.name: str = name
        self.settings: SettingsManager = settings

    @abstractmethod
    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        """
        Abstract method to define the behavior of the simulation actor during a timestep.

        Subclasses must implement this method to specify actions to be taken in the simulation.

        Args:
            timestep (int): The current simulation timestep.
            timestep_length (float): The duration of a timestep in years.
            time_passed (float): Time passed in years.
            filing_cabinet (FilingCabinet): A container for accessing and modifying simulation data arrays.
            agn_disk (AGNDisk): An object containing the generated properties of the AGN disk.
            random_generator (Generator): A random number generator for stochastic behaviors.

        Returns:
            NotImplemented: Must be overridden by subclasses.
        """
        return NotImplemented

    def __str__(self):
        """
        Returns a string representation of the simulation actor.

        The string includes the actor's name and its class type.

        Returns:
            str: A formatted string representation of the simulation actor.
        """
        return f"{self.name} ({type(self)})"


class ReclassifyDiskObjects(SimulationActor):
    """
    This simulation actor looks at the initial seeded array of objects and distributes them
    based on different classifications.

    Attributes:
        name (str): Optional. The name of the simulation actor. Defaults to "Reclassify Disk Objects".
        settings (SettingsManager): A manager for accessing configuration settings. Defaults to an empty `SettingsManager`.

    Methods:
        perform(timestep, filing_cabinet, randomState):
            Classifies black holes based on their semi-major axis and orbital angular momentum
            and stores the results in the filing cabinet.

    Notes:
        - Black holes with a semi-major axis smaller than `settings.inner_disk_outer_radius` are classified as inner disk black holes.
        - Prograde black holes are identified by a positive orbital angular momentum (`orb_ang_mom > 0`).
        - Retrograde black holes are identified by a negative orbital angular momentum (`orb_ang_mom < 0`).
        - At the end of this process, there should be no unclassified black holes left in the original array (`settings.bh_array_name`).
    """

    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        """
        Initializes the `ReclassifyDiskObjects` simulation actor.

        Args:
            name (str): Optional. The name of the actor. Defaults to "Reclassify Disk Objects".
            settings (SettingsManager): A settings manager instance. Defaults to base instance of `SettingsManager`.
        """
        super().__init__("Reclassify Disk Objects" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        """
        Performs the classification of objects into separate categories and updates the filing cabinet.

        Args:
            timestep (int): The current simulation timestep.
            timestep_length (float): The duration of a timestep in years.
            time_passed (float): Time passed in years.
            filing_cabinet (FilingCabinet): A container for managing and storing categorized black holes.
            agn_disk (AGNDisk): An object containing the generated properties of the AGN disk.
            random_generator (Generator): A random number generator for stochastic behaviors.

        Black Hole Classification Steps:
            1. Inner Disk Black Holes:
               - Black holes with a semi-major axis (`orb_a`) smaller than `settings.inner_disk_outer_radius`.
               - Moved to a separate array (`settings.bh_inner_disk_array_name`).

            2. Prograde Black Holes:
               - Black holes with a positive orbital angular momentum (`orb_ang_mom > 0`).
               - Moved to a separate array (`settings.bh_prograde_array_name`).

            3. Retrograde Black Holes:
               - Black holes with a negative orbital angular momentum (`orb_ang_mom < 0`).
               - Moved to a separate array (`settings.bh_retrograde_array_name`).

        Notes:
            - At the end of this method, the original black hole array (`settings.bh_array_name`) should be empty.
        """
        sm = self.settings

        # Black Holes, get array from filing cabinet or create an empty one if it doesn't

        blackholes = filing_cabinet.get_array(sm.bh_array_name, AGNBlackHoleArray, True)

        # Inner disk black holes

        inner_disk_blackholes = blackholes.copy()

        # Create a mask for all black holes with a semi-major axis smaller than the defined outer radius
        inner_disk_bh_ids = blackholes.unique_id[blackholes.orb_a < sm.inner_disk_outer_radius]

        # Apply the mask the two lists
        inner_disk_blackholes.keep_only(inner_disk_bh_ids)
        blackholes.remove_all(inner_disk_bh_ids)

        # Add inner disk black holes to separate filing cabinet array.
        filing_cabinet.create_or_append_array(sm.bh_inner_disk_array_name, inner_disk_blackholes)

        # Prograde black holes

        prograde_blackholes = blackholes.copy()

        # Find prograde BH orbiters. Identify BH with orb. ang mom > 0 (orb_ang_mom is only ever +1 or -1)
        prograde_bh_ids = blackholes.unique_id[blackholes.orb_ang_mom > 0]

        # Apply the mask
        prograde_blackholes.keep_only(prograde_bh_ids)
        blackholes.remove_all(prograde_bh_ids)

        # Add prograde disk black holes to separate filing cabinet array.
        filing_cabinet.create_or_append_array(sm.bh_prograde_array_name, prograde_blackholes)

        # Retrograde black holes

        retrograde_blackholes = blackholes.copy()

        # Find retrograde black holes
        retrograde_bh_ids = blackholes.unique_id[blackholes.orb_ang_mom < 0]

        # Apply the mask
        retrograde_blackholes.keep_only(retrograde_bh_ids)
        blackholes.remove_all(retrograde_bh_ids)

        # Add prograde disk black holes to separate filing cabinet array.
        filing_cabinet.create_or_append_array(sm.bh_retrograde_array_name, retrograde_blackholes)

        # At the end of this, we should have no black holes in the sm.bh_array_name array,
        # it can be used in the future to seed new black holes before classification is run.

        # TODO: Stars


class ProgradeBlackHoleDynamics(SimulationActor):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__("Prograde Black Hole Dynamics" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        # Check to make sure the array exists in the filing cabinet
        if sm.bh_prograde_array_name not in filing_cabinet:
            return

        blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)

        # Migrate
        # First if feedback present, find ratio of feedback heating torque to migration torque
        if sm.flag_thermal_feedback > 0:
            ratio_heat_mig_torques = feedback.feedback_bh_hankla(
                blackholes_pro.orb_a,
                agn_disk.disk_surface_density,
                agn_disk.disk_opacity,
                sm.disk_bh_eddington_ratio,
                sm.disk_alpha_viscosity,
                sm.disk_radius_outer)
        else:
            ratio_heat_mig_torques = np.ones(len(blackholes_pro.orb_a))

        # Migrate as usual
        blackholes_pro.orb_a = migration.type1_migration_single(
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

        # Check for orb_a unphysical
        bh_pro_id_num_unphysical_a = blackholes_pro.unique_id[blackholes_pro.orb_a == 0.0]
        blackholes_pro.remove_all(bh_pro_id_num_unphysical_a)

        # Accrete
        blackholes_pro.mass = accretion.change_bh_mass(
            blackholes_pro.mass,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_eddington_mass_growth_rate,
            sm.timestep_duration_yr
        )

        # Spin up
        blackholes_pro.spin = accretion.change_bh_spin_magnitudes(
            blackholes_pro.spin,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_torque_condition,
            sm.timestep_duration_yr,
            blackholes_pro.orb_ecc,
            sm.disk_bh_pro_orb_ecc_crit,
        )

        # Torque spin angle
        blackholes_pro.spin_angle = accretion.change_bh_spin_angles(
            blackholes_pro.spin_angle,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_torque_condition,
            sm.disk_bh_spin_resolution_min,
            sm.timestep_duration_yr,
            blackholes_pro.orb_ecc,
            sm.disk_bh_pro_orb_ecc_crit
        )

        # Damp BH orbital eccentricity
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

        if sm.flag_dynamic_enc > 0:
            blackholes_pro.orb_a, blackholes_pro.orb_ecc = dynamics.circular_singles_encounters_prograde(
                sm.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                blackholes_pro.orb_ecc,
                sm.timestep_duration_yr,
                sm.disk_bh_pro_orb_ecc_crit,
                sm.delta_energy_strong,
            )

        # After this time period, was there a disk capture via orbital grind-down?
        # To do: What eccentricity do we want the captured BH to have? Right now ecc=0.0? Should it be ecc<h at a?
        # Assume 1st gen BH captured and orb ecc =0.0
        # To do: Bias disk capture to more massive BH!
        capture = time_passed % sm.capture_time_yr
        if capture == 0:
            disk_bh_num = 1

            bh_orb_a_captured = setupdiskblackholes.setup_disk_blackholes_location(disk_bh_num, sm.disk_radius_capture_outer, sm.disk_inner_stable_circ_orb)
            bh_mass_captured = setupdiskblackholes.setup_disk_blackholes_masses(disk_bh_num, sm.nsc_imf_bh_mode, sm.nsc_imf_bh_mass_max, sm.nsc_imf_bh_powerlaw_index, sm.mass_pile_up)
            bh_spin_captured = setupdiskblackholes.setup_disk_blackholes_spins(disk_bh_num, sm.nsc_bh_spin_dist_mu, sm.nsc_bh_spin_dist_sigma)
            bh_spin_angle_captured = setupdiskblackholes.setup_disk_blackholes_spin_angles(disk_bh_num, bh_spin_captured)

            unique_ids = np.array([uuid_provider(random_generator) for _ in range(disk_bh_num)])

            captured_prograde_bh = AGNBlackHoleArray(
                unique_id=unique_ids,
                mass=bh_mass_captured,
                spin=bh_spin_captured,
                spin_angle=bh_spin_angle_captured,
                orb_a=bh_orb_a_captured,
                orb_inc=np.full(disk_bh_num, 0.0),
                orb_ecc=np.full(disk_bh_num, 0.0),
                orb_ang_mom=np.ones(disk_bh_num),
                orb_arg_periapse=np.full(disk_bh_num, -1.5),
                gen=np.ones(disk_bh_num)
            )

            blackholes_pro.add_objects(captured_prograde_bh)

        # Test if any BH or BBH are in the danger-zone (<mininum_safe_distance, default =50r_g) from SMBH.
        # Potential EMRI/BBH EMRIs.
        # Find prograde BH in inner disk. Define inner disk as <=50r_g.
        # Since a 10Msun BH will decay into a 10^8Msun SMBH at 50R_g in ~38Myr and decay time propto a^4.
        # e.g at 25R_g, decay time is only 2.3Myr.

        # Remove the pros if it is in the inner disk and add it to the inner disk array.
        pro_bh_inner_disk_ids = blackholes_pro.unique_id[blackholes_pro.orb_a < sm.inner_disk_outer_radius]

        inner_pros = blackholes_pro.copy()
        inner_pros.keep_only(pro_bh_inner_disk_ids)
        filing_cabinet.create_or_append_array(sm.bh_inner_disk_array_name, inner_pros)

        blackholes_pro.remove_all(pro_bh_inner_disk_ids)


class RetrogradeBlackholeDynamics(SimulationActor):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__("Retrograde Black Hole Dynamics" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        # Check to make sure the array exists in the filing cabinet
        if sm.bh_retrograde_array_name not in filing_cabinet:
            return

        blackholes_retro = filing_cabinet.get_array(sm.bh_retrograde_array_name, AGNBlackHoleArray)

        orb_ecc, orb_a, orb_inc = disk_capture.retro_bh_orb_disk_evolve(
            sm.smbh_mass,
            blackholes_retro.mass,
            blackholes_retro.orb_a,
            blackholes_retro.orb_ecc,
            blackholes_retro.orb_inc,
            blackholes_retro.orb_arg_periapse,
            sm.disk_inner_stable_circ_orb,
            agn_disk.disk_surface_density,
            timestep_length
        )

        # Update retro black hole object array attributes
        blackholes_retro.orb_ecc = np.array(orb_ecc)
        blackholes_retro.orb_a = np.array(orb_a)
        blackholes_retro.orb_inc = np.array(orb_inc)

        # Check for unphysical orbital eccentricities
        bh_retro_id_num_unphysical_ecc = blackholes_retro.unique_id[blackholes_retro.orb_ecc >= 1.]
        blackholes_retro.remove_all(bh_retro_id_num_unphysical_ecc)

        # Remove the retro if it is in the inner disk and add it to the inner disk array.
        retro_bh_inner_disk_ids = blackholes_retro.unique_id[blackholes_retro.orb_a < sm.inner_disk_outer_radius]

        inner_retros = blackholes_retro.copy()
        inner_retros.keep_only(retro_bh_inner_disk_ids)
        filing_cabinet.create_or_append_array(sm.bh_inner_disk_array_name, inner_retros)

        blackholes_retro.remove_all(retro_bh_inner_disk_ids)

        # Remove any retros that have flipped prograde and add them to the prograde black holes array
        inc_threshhold = 5.0 * np.pi / 180.0

        bh_id_num_flip_to_pro = blackholes_retro.unique_id[np.where((np.abs(blackholes_retro.orb_inc) <= inc_threshhold) | (blackholes_retro.orb_ecc == 0.0))]

        blackholes_pro = blackholes_retro.copy()
        blackholes_pro.keep_only(bh_id_num_flip_to_pro)
        filing_cabinet.create_or_append_array(sm.bh_prograde_array_name, blackholes_pro)

        blackholes_retro.remove_all(bh_id_num_flip_to_pro)


class BinaryBlackHoleDynamics(SimulationActor):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__("Binary Black Hole Dynamics" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        # Check to make sure the array exists in the filing cabinet
        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

        # Check if array is empty
        if len(blackholes_binary) == 0:
            return

        # First check that binaries are real. Discard any columns where the location or the mass is 0.
        # SF: I believe this step is handling an error checking thing that may have been
        #     set up in the previous timeloop if e.g. a binary either merged or was ionized?
        #     Please explain what this is and how it works right here?
        bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary=blackholes_binary)
        # One of the key parameter (mass or location is zero). Not real. Delete binary. Remove column at index = ionization_flag
        blackholes_binary.remove_all(bh_binary_id_num_unphysical)

        # Check for bin_ecc unphysical
        bh_binary_id_num_unphysical_ecc = blackholes_binary.unique_id[blackholes_binary.bin_ecc >= 1.]
        # The binary has unphysical eccentricity. Delete
        blackholes_binary.remove_all(bh_binary_id_num_unphysical_ecc)

        # If there are binaries, evolve them
        # Damp binary orbital eccentricity
        eccentricity.orbital_bin_ecc_damping(
            sm.smbh_mass,
            blackholes_binary,
            agn_disk.disk_surface_density,
            agn_disk.disk_aspect_ratio,
            sm.timestep_duration_yr,
            sm.disk_bh_pro_orb_ecc_crit
        )

        if sm.flag_dynamic_enc > 0 and sm.bh_prograde_array_name in filing_cabinet:
            blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)

            # Harden/soften binaries via dynamical encounters
            # Harden binaries due to encounters with circular singletons (e.g. Leigh et al. 2018)
            # FIX THIS: RETURN perturbed circ singles (orb_a, orb_ecc)
            dynamics.circular_binaries_encounters_circ_prograde(
                sm.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                blackholes_pro.orb_ecc,
                sm.timestep_duration_yr,
                sm.disk_bh_pro_orb_ecc_crit,
                sm.delta_energy_strong,
                blackholes_binary,
            )

            # Soften/ ionize binaries due to encounters with eccentric singletons
            # Return 3 things: perturbed biary_bh_array, disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc
            dynamics.circular_binaries_encounters_ecc_prograde(
                sm.smbh_mass,
                blackholes_pro.orb_a,
                blackholes_pro.mass,
                blackholes_pro.orb_ecc,
                sm.timestep_duration_yr,
                sm.disk_bh_pro_orb_ecc_crit,
                sm.delta_energy_strong,
                blackholes_binary,
            )

        # Check for bin_ecc unphysical
        # We need a second check here
        bh_binary_id_num_unphysical_ecc = blackholes_binary.unique_id[blackholes_binary.bin_ecc >= 1.]
        blackholes_binary.remove_all(bh_binary_id_num_unphysical_ecc)

        time_gw_normalization = merge.normalize_tgw(sm.smbh_mass, sm.inner_disk_outer_radius)
        # print("Scale of t_gw (yrs)=", time_gw_normalization)

        # Harden binaries via gas
        # Choose between Baruteau et al. 2011 gas hardening, or gas hardening from LANL simulations. To do: include dynamical hardening/softening from encounters
        evolve.bin_harden_baruteau(
            blackholes_binary,
            sm.smbh_mass,
            sm.timestep_duration_yr,
            time_gw_normalization,
            time_passed,
        )

        # Check closeness of binary. Are black holes at merger condition separation
        evolve.bin_contact_check(blackholes_binary, sm.smbh_mass)

        # Accrete gas onto binary components
        evolve.change_bin_mass(
            blackholes_binary,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_eddington_mass_growth_rate,
            sm.timestep_duration_yr,
        )

        # Spin up binary components
        evolve.change_bin_spin_magnitudes(
            blackholes_binary,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_torque_condition,
            sm.timestep_duration_yr,
        )

        # Torque angle of binary spin components
        evolve.change_bin_spin_angles(
            blackholes_binary,
            sm.disk_bh_eddington_ratio,
            sm.disk_bh_torque_condition,
            sm.disk_bh_spin_resolution_min,
            sm.timestep_duration_yr,
        )

        if sm.flag_dynamic_enc > 0:
            # Spheroid encounters
            # FIX THIS: Replace nsc_imf_bh below with nsc_imf_stars_ since pulling from stellar MF
            dynamics.bin_spheroid_encounter(
                sm.smbh_mass,
                sm.timestep_duration_yr,
                blackholes_binary,
                time_passed,
                sm.nsc_imf_bh_powerlaw_index,
                sm.delta_energy_strong,
                sm.nsc_spheroid_normalization,
            )

        if sm.flag_dynamic_enc > 0:
            # Recapture bins out of disk plane.
            # FIX THIS: Replace this with orb_inc_damping but for binary bhbh OBJECTS (KN)
            dynamics.bin_recapture(
                blackholes_binary,
                sm.timestep_duration_yr
            )

        # Migrate binaries
        # First if feedback present, find ratio of feedback heating torque to migration torque
        if sm.flag_thermal_feedback > 0:
            ratio_heat_mig_torques_bin_com = evolve.bin_com_feedback_hankla(
                blackholes_binary,
                agn_disk.disk_surface_density,
                agn_disk.disk_opacity,
                sm.disk_bh_eddington_ratio,
                sm.disk_alpha_viscosity,
                sm.disk_radius_outer
            )
        else:
            ratio_heat_mig_torques_bin_com = np.ones(blackholes_binary.num)

        # Migrate binaries center of mass
        migration.type1_migration_binary(
            sm.smbh_mass, blackholes_binary,
            sm.disk_bh_pro_orb_ecc_crit,
            agn_disk.disk_surface_density, agn_disk.disk_aspect_ratio, ratio_heat_mig_torques_bin_com,
            sm.disk_radius_trap, sm.disk_radius_outer, sm.timestep_duration_yr)

        # Test to see if any binaries separation is O(1r_g)
        # If so, track them for GW freq, strain.
        # Minimum BBH separation (in units of r_g)
        # If there are binaries AND if any separations are < min_bbh_gw_separation

        bh_binary_id_num_gw = blackholes_binary.unique_id[(blackholes_binary.bin_sep < sm.min_bbh_gw_separation) & (blackholes_binary.bin_sep > 0)]

        if bh_binary_id_num_gw.size > 0:
            bbh_gw_freq = filing_cabinet.get_value("bbh_gw_freq", default_value=9.e-7 * np.ones(bh_binary_id_num_gw.size))

            old_bbh_gw_freq = deepcopy(bbh_gw_freq)

            bbh_gw_strain, bbh_gw_freq = gw.bbh_gw_params(
                blackholes_binary,
                bh_binary_id_num_gw,
                sm.smbh_mass,
                sm.timestep_duration_yr,
                old_bbh_gw_freq,
                sm.agn_redshift
            )

            filing_cabinet.set_value("bbh_gw_freq", bbh_gw_freq)

            blackholes_binary_gw = blackholes_binary.copy()
            blackholes_binary_gw.keep_only(bh_binary_id_num_gw)

            blackholes_binary.remove_all(bh_binary_id_num_gw)

            filing_cabinet.create_or_append_array(sm.bbh_gw_array_name, blackholes_binary_gw)

        # Evolve GW frequency and strain
        gw.evolve_gw(
            blackholes_binary,
            sm.smbh_mass,
            sm.agn_redshift
        )

        ionized_binary_ids = evolve.bin_ionization_check(blackholes_binary, sm.smbh_mass)
        if ionized_binary_ids.size > 0:

            ionized_blackholes = AGNBlackHoleArray(
                unique_id=np.concatenate([
                    blackholes_binary.get_attribute("unique_id_1", ionized_binary_ids),
                    blackholes_binary.get_attribute("unique_id_2", ionized_binary_ids)]),

                mass=np.concatenate([
                    blackholes_binary.get_attribute("mass_1", ionized_binary_ids),
                    blackholes_binary.get_attribute("mass_2", ionized_binary_ids)]),

                spin=np.concatenate([
                    blackholes_binary.get_attribute("spin_1", ionized_binary_ids),
                    blackholes_binary.get_attribute("spin_2", ionized_binary_ids)]),

                spin_angle=np.concatenate([
                    blackholes_binary.get_attribute("spin_angle_1", ionized_binary_ids),
                    blackholes_binary.get_attribute("spin_angle_2", ionized_binary_ids)]),

                orb_a=np.concatenate([
                    blackholes_binary.get_attribute("orb_a_1", ionized_binary_ids),
                    blackholes_binary.get_attribute("orb_a_2", ionized_binary_ids)]),

                gen=np.concatenate([
                    blackholes_binary.get_attribute("gen_1", ionized_binary_ids),
                    blackholes_binary.get_attribute("gen_2", ionized_binary_ids)]),

                orb_ecc=np.full(ionized_binary_ids.size * 2, 0.01),
                orb_inc=np.full(ionized_binary_ids.size * 2, 0.0),
                orb_ang_mom=np.ones(ionized_binary_ids.size * 2),
                orb_arg_periapse=np.full(ionized_binary_ids.size * 2, -1.5),
                gw_freq=np.full(ionized_binary_ids.size * 2, -1.5),
                gw_strain=np.full(ionized_binary_ids.size * 2, -1.5),
            )

            filing_cabinet.create_or_append_array(sm.bh_prograde_array_name, ionized_blackholes)
            blackholes_binary.remove_all(ionized_binary_ids)

            # Consistency Checks

            # Check for primary masses of zero
            zero_mass_binary_ids = blackholes_binary.unique_id[blackholes_binary.mass_1 == 0]
            blackholes_binary.remove_all(zero_mass_binary_ids)

            # Check for NaNs in flag_merging
            nan_flag_ids = blackholes_binary.unique_id[np.isnan(blackholes_binary.flag_merging)]
            blackholes_binary.remove_all(nan_flag_ids)

            # If one of the key parameter (mass or location) is zero, the binary isn't real.
            bh_binary_id_num_unphysical = evolve.bin_reality_check(blackholes_binary=blackholes_binary)
            blackholes_binary.remove_all(bh_binary_id_num_unphysical)

            merging_blackhole_ids = blackholes_binary.unique_id[np.nonzero(blackholes_binary.flag_merging)]

            if sm.verbose:
                print(f"Number of Binary Black Hole mergers {len(merging_blackhole_ids)}")
                print(f"Number of Binary Black Holes removed due to failed consistency checks: {len(zero_mass_binary_ids) + len(nan_flag_ids) + len(bh_binary_id_num_unphysical)}")

            if merging_blackhole_ids.size > 0:
                bh_mass_merged = merge.merged_mass(
                    blackholes_binary.get_attribute("mass_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("mass_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_2", merging_blackhole_ids)
                )

                bh_spin_merged = merge.merged_spin(
                    blackholes_binary.get_attribute("mass_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("mass_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_2", merging_blackhole_ids)
                )

                bh_chi_eff_merged = merge.chi_effective(
                    blackholes_binary.get_attribute("mass_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("mass_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_angle_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_angle_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("bin_orb_ang_mom", merging_blackhole_ids)
                )

                bh_chi_p_merged = merge.chi_p(
                    blackholes_binary.get_attribute("mass_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("mass_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_angle_1", merging_blackhole_ids),
                    blackholes_binary.get_attribute("spin_angle_2", merging_blackhole_ids),
                    blackholes_binary.get_attribute("bin_orb_inc", merging_blackhole_ids)
                )

                merged_binaries = AGNMergedBlackHoleArray(
                    unique_id_1=blackholes_binary.get_attribute("unique_id_1", merging_blackhole_ids),
                    unique_id_2=blackholes_binary.get_attribute("unique_id_2", merging_blackhole_ids),
                    mass_1=blackholes_binary.get_attribute("mass_1", merging_blackhole_ids),
                    mass_2=blackholes_binary.get_attribute("mass_2", merging_blackhole_ids),
                    spin_1=blackholes_binary.get_attribute("spin_1", merging_blackhole_ids),
                    spin_2=blackholes_binary.get_attribute("spin_2", merging_blackhole_ids),
                    orb_a_1=blackholes_binary.get_attribute("orb_a_1", merging_blackhole_ids),
                    orb_a_2=blackholes_binary.get_attribute("orb_a_2", merging_blackhole_ids),
                    spin_angle_1=blackholes_binary.get_attribute("spin_angle_1", merging_blackhole_ids),
                    spin_angle_2=blackholes_binary.get_attribute("spin_angle_2", merging_blackhole_ids),
                    gen_1=blackholes_binary.get_attribute("gen_1", merging_blackhole_ids),
                    gen_2=blackholes_binary.get_attribute("gen_2", merging_blackhole_ids),
                    bin_sep=blackholes_binary.get_attribute("bin_sep", merging_blackhole_ids),
                    bin_ecc=blackholes_binary.get_attribute("bin_ecc", merging_blackhole_ids),
                    bin_orb_a=blackholes_binary.get_attribute("bin_orb_a", merging_blackhole_ids),
                    bin_orb_ang_mom=blackholes_binary.get_attribute("bin_orb_ang_mom", merging_blackhole_ids),
                    bin_orb_inc=blackholes_binary.get_attribute("bin_orb_inc", merging_blackhole_ids),
                    bin_orb_ecc=blackholes_binary.get_attribute("bin_orb_ecc", merging_blackhole_ids),
                    time_to_merger_gw=blackholes_binary.get_attribute("time_to_merger_gw", merging_blackhole_ids),
                    flag_merging=blackholes_binary.get_attribute("flag_merging", merging_blackhole_ids),
                    chi_eff=bh_chi_eff_merged,
                    chi_p=bh_chi_p_merged,
                    time_merged=np.full(len(merging_blackhole_ids), time_passed),
                    mass_final=bh_mass_merged,
                    spin_final=bh_spin_merged,
                    spin_angle_final=np.zeros(len(merging_blackhole_ids)),
                )

                filing_cabinet.create_or_append_array(sm.bbh_merged_array_name, merged_binaries)
                blackholes_binary.remove_all(merging_blackhole_ids)

                next_generation = np.maximum(merged_binaries.get_attribute("gen_1", merging_blackhole_ids), merged_binaries.get_attribute("gen_2", merging_blackhole_ids)) + 1.0

                new_blackholes = AGNBlackHoleArray(
                    unique_id=np.array([uuid_provider(random_generator) for _ in range(merging_blackhole_ids.size)]),
                    mass=merged_binaries.get_attribute("mass_final", merging_blackhole_ids),
                    orb_a=merged_binaries.get_attribute("bin_orb_a", merging_blackhole_ids),
                    spin=merged_binaries.get_attribute("spin_final", merging_blackhole_ids),
                    spin_angle=np.zeros(merging_blackhole_ids.size),
                    orb_inc=np.zeros(merging_blackhole_ids.size),
                    orb_ang_mom=np.ones(merging_blackhole_ids.size),  # All new BH are prograde, so don't ad them to the unsorted array
                    orb_arg_periapse=np.full(merging_blackhole_ids.size, -1.5),
                    orb_ecc=np.full(merging_blackhole_ids.size, 0.01),
                    gen=next_generation
                )

                filing_cabinet.create_or_append_array(sm.bh_prograde_array_name, new_blackholes)


class BinaryBlackHoleFormation(SimulationActor):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__("Binary Black Hole Formation" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        # Check to make sure the array exists in the filing cabinet
        if sm.bh_prograde_array_name not in filing_cabinet:
            return

        blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)

        encounter_indices = formation.binary_check(
            blackholes_pro.orb_a,
            blackholes_pro.mass,
            sm.smbh_mass,
            blackholes_pro.orb_ecc,
            sm.disk_bh_pro_orb_ecc_crit
        )

        if len(encounter_indices) == 0:
            if sm.verbose:
                print("No binaries formed")

            return

        primary_ids = np.array([blackholes_pro.unique_id[index] for index in encounter_indices[0]])
        secondary_ids = np.array([blackholes_pro.unique_id[index] for index in encounter_indices[1]])

        mass_1 = blackholes_pro.get_attribute("mass", primary_ids)
        mass_2 = blackholes_pro.get_attribute("mass", secondary_ids)

        orb_a_1 = blackholes_pro.get_attribute("orb_a", primary_ids)
        orb_a_2 = blackholes_pro.get_attribute("orb_a", secondary_ids)

        bin_sep = np.abs(orb_a_1 - orb_a_2)
        bin_orb_a = orb_a_1 + ((bin_sep * mass_2) / (mass_1 + mass_2))

        bin_orb_ang_mom = np.full(len(primary_ids), 1)

        if sm.fraction_bin_retro > 0:
            bin_orb_ang_mom = [random_generator.choice(a=[1, -1], p=[1 - sm.fraction_bin_retro, sm.fraction_bin_retro]) for _ in range(primary_ids.size)]

        gw_strain, gw_freq = gw_strain_freq(mass_1=mass_1, mass_2=mass_2, obj_sep=bin_sep, timestep_duration_yr=-1,
                                            old_gw_freq=-1, smbh_mass=sm.smbh_mass, agn_redshift=sm.agn_redshift,
                                            flag_include_old_gw_freq=0)

        new_binaries = AGNBinaryBlackHoleArray(
            unique_id_1=primary_ids,
            unique_id_2=secondary_ids,
            orb_a_1=orb_a_1,
            orb_a_2=orb_a_2,
            mass_1=mass_1,
            mass_2=mass_2,
            spin_1=blackholes_pro.get_attribute("spin", primary_ids),
            spin_2=blackholes_pro.get_attribute("spin", secondary_ids),
            spin_angle_1=blackholes_pro.get_attribute("spin_angle", primary_ids),
            spin_angle_2=blackholes_pro.get_attribute("spin_angle", secondary_ids),
            bin_sep=bin_sep,
            bin_orb_a=bin_orb_a,
            time_to_merger_gw=np.zeros(primary_ids.size),
            flag_merging=np.zeros(primary_ids.size),
            time_merged=np.zeros(primary_ids.size),
            bin_ecc=np.array([random_generator.uniform() for _ in range(primary_ids.size)]),
            gen_1=blackholes_pro.get_attribute("gen", primary_ids),
            gen_2=blackholes_pro.get_attribute("gen", secondary_ids),
            bin_orb_ang_mom=bin_orb_ang_mom,
            bin_orb_inc=np.zeros(primary_ids.size),
            bin_orb_ecc=np.full(primary_ids.size, sm.initial_binary_orbital_ecc),
            gw_freq=gw_freq,
            gw_strain=gw_strain,
        )

        filing_cabinet.create_or_append_array(sm.bbh_array_name, new_binaries)

        blackholes_pro.remove_all(primary_ids)
        blackholes_pro.remove_all(secondary_ids)

        if sm.verbose:
            print(f"Binaries created: {primary_ids.size}")


class BreakupBinaryBlackHoles(SimulationActor):
    def __init__(self, name: str = None, settings: SettingsManager = SettingsManager()):
        super().__init__("Breakup Binary Black Holes" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

        # Check if array is empty
        if len(blackholes_binary) == 0:
            return

        binary_ids = blackholes_binary.unique_id

        blackholes = AGNBlackHoleArray(
            unique_id=np.concatenate([
                blackholes_binary.get_attribute("unique_id_1", binary_ids),
                blackholes_binary.get_attribute("unique_id_2", binary_ids)]),

            mass=np.concatenate([
                blackholes_binary.get_attribute("mass_1", binary_ids),
                blackholes_binary.get_attribute("mass_2", binary_ids)]),

            spin=np.concatenate([
                blackholes_binary.get_attribute("spin_1", binary_ids),
                blackholes_binary.get_attribute("spin_2", binary_ids)]),

            spin_angle=np.concatenate([
                blackholes_binary.get_attribute("spin_angle_1", binary_ids),
                blackholes_binary.get_attribute("spin_angle_2", binary_ids)]),

            orb_a=np.concatenate([
                blackholes_binary.get_attribute("orb_a_1", binary_ids),
                blackholes_binary.get_attribute("orb_a_2", binary_ids)]),

            gen=np.concatenate([
                blackholes_binary.get_attribute("gen_1", binary_ids),
                blackholes_binary.get_attribute("gen_2", binary_ids)]),

            orb_ecc=np.full(binary_ids.size * 2, 0.01),
            orb_inc=np.full(binary_ids.size * 2, 0.0),
            orb_ang_mom=np.ones(binary_ids.size * 2),
            orb_arg_periapse=np.full(binary_ids.size * 2, -1.5),
            gw_freq=np.full(binary_ids.size * 2, -1.5),
            gw_strain=np.full(binary_ids.size * 2, -1.5),
        )

        filing_cabinet.create_or_append_array(sm.bh_prograde_array_name, blackholes)
        blackholes_binary.remove_all(binary_ids)
