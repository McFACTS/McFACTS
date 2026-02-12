import os
import shutil

from tqdm.auto import tqdm

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.modules.accretion import ProgradeBlackHoleAccretion, BinaryBlackHoleAccretion, ProgradeBlackHoleBondi
from mcfacts.modules.damping import ProgradeBlackHoleDamping, BinaryBlackHoleDamping
from mcfacts.modules.disk_capture import EvolveRetrogradeBlackHoles, RecaptureBinaryBlackHoles, \
    CaptureNSCProgradeBlackHoles
from mcfacts.modules.dynamics import SingleBlackHoleDynamics, BinaryBlackHoleDynamics
from mcfacts.modules.formation import BinaryBlackHoleFormation
from mcfacts.modules.gas_hardening import BinaryBlackHoleGasHardening
from mcfacts.modules.gw import BinaryBlackHoleEvolveGW, InnerBlackHoleDynamics
from mcfacts.modules.merge import ProcessBinaryBlackHoleMergers, ProcessEMRIMergers
from mcfacts.modules.migration import ProgradeBlackHoleMigration, BinaryBlackHoleMigration
from mcfacts.objects.actors import InitialObjectReclassification, InnerDiskFilter, FlipRetroProFilter
from mcfacts.objects.actors.reality_checks import SingleBlackHoleRealityCheck, BinaryBlackHoleRealityCheck
from mcfacts.objects.agn_object_array import *
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.populators import SingleBlackHolePopulator
from mcfacts.objects.snapshot import TxtSnapshotHandler
from mcfacts.objects.timeline import SimulationTimeline

def main(settings: SettingsManager):
    if settings.overwrite_files == False and os.path.isdir(settings.output_dir):
        assert False, f"Output directory {settings.output_dir} already exist. Set --overwrite_files=True to clear the directory."

    if settings.overwrite_files and os.path.isdir(settings.output_dir):
        shutil.rmtree(settings.output_dir)

    agn_disk = AGNDisk(settings)
    snapshot_handler = TxtSnapshotHandler(settings = settings)

    snapshot_handler.save_settings("./runs", "settings", settings)

    population_cabinet = FilingCabinet()

    pbar = tqdm(total=settings.galaxy_num, position=0, leave=True)

    for galaxy_id in range(settings.galaxy_num,):
        pbar.set_description(f"Running Galaxy {galaxy_id}")
        pbar.update(1)

        galaxy_seed = settings.seed - galaxy_id

        # Create instance of galaxy
        galaxy = Galaxy(seed=galaxy_seed, runs_folder="./runs", galaxy_id=str(galaxy_id), settings=settings)

        # Create instance of populators
        single_bh_populator = SingleBlackHolePopulator()
        # single_star_populator = SingleStarPopulator("single_stars")
        galaxy.populate([single_bh_populator], agn_disk)

        # Create timeline to classify objects created during population
        pre_timeline = SimulationTimeline("Reclassification", timesteps=1, timestep_length=0)
        pre_timeline.add_timeline_actor(InitialObjectReclassification())
        pre_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())
        galaxy.run(pre_timeline, agn_disk)

        # Create timeline to run main simulation
        active_phase_timeline = SimulationTimeline("Active Timeline",
                                               timesteps=settings.active_timestep_num,
                                               timestep_length=galaxy.settings.active_timestep_duration_yr)

        # Retrograde evolution and capture
        active_phase_timeline.add_timeline_actor(EvolveRetrogradeBlackHoles())
        active_phase_timeline.add_timeline_actor(FlipRetroProFilter())

        # NSC Capture
        active_phase_timeline.add_timeline_actor(CaptureNSCProgradeBlackHoles())
        active_phase_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())

        active_phase_timeline.add_timeline_actor(InnerDiskFilter())

        prograde_array = galaxy.settings.bh_prograde_array_name
        innerdisk_array = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name

        # Migration: Single BH, Binary BH
        active_phase_timeline.add_timeline_actors([
            ProgradeBlackHoleMigration(target_array=innerdisk_array),
            ProgradeBlackHoleMigration(target_array=prograde_array),
            BinaryBlackHoleMigration()
        ])

        # Accretion: Single BH, Binary BH,
        if settings.flag_enable_bondi:
            accretion = [
                ProgradeBlackHoleBondi(target_array=innerdisk_array),
                ProgradeBlackHoleBondi(target_array=prograde_array),
            ]
        else:
            accretion = [
                ProgradeBlackHoleAccretion(target_array=innerdisk_array),
                ProgradeBlackHoleAccretion(target_array=prograde_array),
            ]

        active_phase_timeline.add_timeline_actors(accretion)
        active_phase_timeline.add_timeline_actor(BinaryBlackHoleAccretion(reality_merge_checks=True),)

        # Gas Dynamics: Single BH, Binary BH
        active_phase_timeline.add_timeline_actors([
            ProgradeBlackHoleDamping(target_array=innerdisk_array),
            ProgradeBlackHoleDamping(target_array=prograde_array),
            BinaryBlackHoleDamping(),
            BinaryBlackHoleGasHardening(reality_merge_checks=True)
        ])

        # Dynamical Encounters: Single BH, Binary BH
        active_phase_timeline.add_timeline_actors([
            SingleBlackHoleDynamics(target_array=innerdisk_array),
            InnerBlackHoleDynamics(target_array=innerdisk_array),
            InnerBlackHoleDynamics(target_array=inner_gw_only_array),
            SingleBlackHoleDynamics(target_array=prograde_array),
            BinaryBlackHoleDynamics(reality_merge_checks=True)
        ])

        # Misc Evolution: Binary BH
        active_phase_timeline.add_timeline_actors([
            RecaptureBinaryBlackHoles(),
            BinaryBlackHoleRealityCheck(),
            BinaryBlackHoleEvolveGW(),
            ProcessBinaryBlackHoleMergers(),
            BinaryBlackHoleFormation()
        ])

        # EMRI Dynamics: Single BH (inner disk)
        active_phase_timeline.add_timeline_actor(ProcessEMRIMergers())

        galaxy.run(active_phase_timeline, agn_disk)

        population_cabinet.ignore_check_array("blackholes_merged")
        population_cabinet.ignore_check_array("blackholes_lvk")

        bbh_merged_array = galaxy.settings.bbh_merged_array_name
        bbh_lvk_array = galaxy.settings.bbh_gw_array_name
        emri_merged_array = galaxy.settings.emri_array_name

        if bbh_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_merged",galaxy.filing_cabinet.get_array(bbh_merged_array))

        if bbh_lvk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_lvk", galaxy.filing_cabinet.get_array(bbh_lvk_array))

        if innerdisk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(innerdisk_array))

        if inner_gw_only_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(inner_gw_only_array))

        if emri_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(emri_merged_array))

    pbar.close()

    snapshot_handler.save_cabinet("./runs", "population", population_cabinet)

if __name__ == "__main__":
    main(SettingsManager())
