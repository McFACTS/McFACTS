import numpy as np
from numpy.random import Generator

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.objects.agn_object_array import AGNBlackHoleArray, FilingCabinet
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities import checks


class InitialObjectReclassification(TimelineActor):
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

    def __init__(self, name: str = None, settings: SettingsManager = None):
        """
        Initializes the `ReclassifyDiskObjects` simulation actor.

        Args:
            name (str): Optional. The name of the actor. Defaults to "Reclassify Disk Objects".
            settings (SettingsManager): A settings manager instance. Defaults to base instance of `SettingsManager`.
        """
        super().__init__("Initial Object Reclassification" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet,
                agn_disk: AGNDisk, random_generator: Generator):
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

        self.log("Objects Reclassified")

        # At the end of this, we should have no black holes in the sm.bh_array_name array,
        # it can be used in the future to seed new black holes before classification is run.

        # TODO: Stars


class InnerDiskFilter(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Inner Disk Filter" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet,
                agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        # TODO: For some reason, I was only sorting out retrograde inner disk orbiters? Things break if we don't sort out prograde orbiters as well.
        if sm.bh_retrograde_array_name in filing_cabinet:
            blackholes_retro = filing_cabinet.get_array(sm.bh_retrograde_array_name, AGNBlackHoleArray)

            bh_id_num_retro_inner_disk = blackholes_retro.id_num[blackholes_retro.orb_a < sm.inner_disk_outer_radius]

            if bh_id_num_retro_inner_disk.size > 0:
                # Add BH to inner_disk_arrays
                blackholes_inner_disk = blackholes_retro.copy()
                blackholes_inner_disk.keep_only(bh_id_num_retro_inner_disk)

                # Remove from blackholes_retro and update filing_cabinet
                filing_cabinet.create_or_append_array(sm.bh_inner_disk_array_name, blackholes_inner_disk)
                blackholes_retro.remove_all(bh_id_num_retro_inner_disk)

        if sm.bh_prograde_array_name in filing_cabinet:
            blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)

            bh_id_num_pro_inner_disk = blackholes_pro.id_num[blackholes_pro.orb_a < sm.inner_disk_outer_radius]

            if bh_id_num_pro_inner_disk.size > 0:
                # Add BH to inner_disk_arrays
                blackholes_inner_disk = blackholes_pro.copy()
                blackholes_inner_disk.keep_only(bh_id_num_pro_inner_disk)

                # Remove from blackholes_prograde and update filing_cabinet
                filing_cabinet.create_or_append_array(sm.bh_inner_disk_array_name, blackholes_inner_disk)
                blackholes_pro.remove_all(bh_id_num_pro_inner_disk)

        if sm.bh_inner_disk_array_name in filing_cabinet:
            lower_disk_function_bound = filing_cabinet.get_value("lower_disk_function_bound",
                                                                 checks.find_function_lower_bounds(agn_disk.disk_density))

            blackholes_inner = filing_cabinet.get_array(sm.bh_inner_disk_array_name, AGNBlackHoleArray)

            bh_id_num_gw_only = blackholes_inner.id_num[blackholes_inner.orb_a <= lower_disk_function_bound]

            if bh_id_num_gw_only.size > 0:
                blackholes_gw_only = blackholes_inner.copy()
                blackholes_gw_only.keep_only(bh_id_num_gw_only)

                filing_cabinet.create_or_append_array(sm.bh_inner_gw_array_name, blackholes_gw_only)
                blackholes_inner.remove_all(bh_id_num_gw_only)


class FlipRetroProFilter(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Flip Retrograde Prograde Filter" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet,
                agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if sm.bh_retrograde_array_name not in filing_cabinet:
            return

        if sm.bh_prograde_array_name not in filing_cabinet:
            return

        blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)
        blackholes_retro = filing_cabinet.get_array(sm.bh_retrograde_array_name, AGNBlackHoleArray)

        inc_threshhold = 5.0 * np.pi / 180.0

        bh_id_num_flip_to_pro = blackholes_retro.id_num[np.where((np.abs(blackholes_retro.orb_inc) <= inc_threshhold) | (blackholes_retro.orb_ecc == 0.0))]

        blackholes_flipped = blackholes_retro.copy()
        blackholes_flipped.keep_only(bh_id_num_flip_to_pro)
        blackholes_flipped.orb_ang_mom = np.ones(bh_id_num_flip_to_pro.size)

        blackholes_pro.add_objects(blackholes_flipped)
        blackholes_retro.remove_all(bh_id_num_flip_to_pro)
