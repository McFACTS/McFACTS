from numpy.random import Generator

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities.checks import binary_reality_check


class SingleBlackHoleRealityCheck(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Single Black Hole Reality Check" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet,
                agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        # Prograde black hole hyperbolic check
        if sm.bh_prograde_array_name in filing_cabinet:
            blackholes_pro = filing_cabinet.get_array(sm.bh_prograde_array_name, AGNBlackHoleArray)

            bh_pro_id_ecc_hyperbolic = blackholes_pro.unique_id[blackholes_pro.orb_ecc >= 1.]
            if len(bh_pro_id_ecc_hyperbolic) > 0:
                self.log(f"{len(bh_pro_id_ecc_hyperbolic)} prograde bh have an orb_ecc > 1.0 and will be removed.")
                blackholes_pro.remove_all(bh_pro_id_ecc_hyperbolic)

            ejected_ids = blackholes_pro.unique_id[blackholes_pro.orb_a >= sm.disk_radius_outer]
            if len(ejected_ids) > 0:
                self.log(f"{len(ejected_ids)} prograde bh have an orb_a > disk_outer_radius and will be ejected.")

                ejected = blackholes_pro.copy()
                ejected.keep_only(ejected_ids)

                blackholes_pro.remove_all(ejected_ids)

                filing_cabinet.create_or_append_array(sm.bh_ejected_array_name, ejected)

            blackholes_pro.consistency_check()

        # Retrograde black hole hyperbolic check
        if sm.bh_retrograde_array_name in filing_cabinet:
            blackholes_retro = filing_cabinet.get_array(sm.bh_retrograde_array_name, AGNBlackHoleArray)
            bh_retro_id_ecc_hyperbolic = blackholes_retro.unique_id[blackholes_retro.orb_ecc >= 1.]

            if len(bh_retro_id_ecc_hyperbolic) > 0:
                self.log(f"{len(bh_retro_id_ecc_hyperbolic)} retrograde bh have an orb_ecc > 1.0 and will be removed.")
                blackholes_retro.remove_all(bh_retro_id_ecc_hyperbolic)


class BinaryBlackHoleRealityCheck(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Binary Black Hole Reality Check" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet,
                agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        binary_reality_check(sm, filing_cabinet, self.log)
