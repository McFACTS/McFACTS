from numpy.random import Generator

from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBlackHoleArray, AGNBinaryBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.physics import evolve


def bin_reality_check(bin_mass_1, bin_mass_2, bin_orb_a_1, bin_orb_a_2, bin_ecc, bin_id_num):
    """Tests if binaries are real (location and mass do not equal 0)

    This function tests to see if the binary is real. If location = 0 or mass = 0 *and* any other element is NON-ZERO then discard this binary element.
    Returns ID numbers of fake binaries.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters

    Returns
    -------
    id_nums or bh_bin_id_num_fakes : numpy.ndarray
        ID numbers of fake binaries with :obj:`float` type
    """
    bh_bin_id_num_fakes = np.array([])

    mass_1_id_num = bin_id_num[bin_mass_1 == 0]
    mass_2_id_num = bin_id_num[bin_mass_2 == 0]
    orb_a_1_id_num = bin_id_num[bin_orb_a_1 == 0]
    orb_a_2_id_num = bin_id_num[bin_orb_a_2 == 0]
    bin_ecc_id_num = bin_id_num[bin_ecc >= 1]

    id_nums = np.concatenate([mass_1_id_num, mass_2_id_num,
                              orb_a_1_id_num, orb_a_2_id_num, bin_ecc_id_num])

    if id_nums.size > 0:
        return (id_nums)
    else:
        return (bh_bin_id_num_fakes)


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

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

        # First check that binaries are real (mass and location are not zero)
        bh_binary_id_num_unphysical = evolve.bin_reality_check(
            blackholes_binary.mass_1,
            blackholes_binary.mass_2,
            blackholes_binary.orb_a_1,
            blackholes_binary.orb_a_2,
            blackholes_binary.bin_ecc,
            blackholes_binary.id_num
        )
        blackholes_binary.remove_all(bh_binary_id_num_unphysical)

        # Check for binaries with hyperbolic eccentricity (ejected from disk)
        bh_binary_id_num_ecc_hyperbolic = blackholes_binary.id_num[blackholes_binary.bin_orb_ecc >= 1.]
        blackholes_binary.remove_all(bh_binary_id_num_ecc_hyperbolic)
