"""
simulation.py contains the canonical simulation supported by the McFACTS collaboration.
"""

import os
import shutil

from tqdm.auto import tqdm

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.modules.accretion import ProgradeBlackHoleAccretion, BinaryBlackHoleAccretion, ProgradeBlackHoleBondi
from mcfacts.modules.damping import ProgradeBlackHoleDamping, BinaryBlackHoleDamping
from mcfacts.modules.disk_capture import EvolveRetrogradeBlackHoles, RecaptureBinaryBlackHoles, \
    CaptureNSCProgradeBlackHoles
from mcfacts.modules.dynamics import SingleBlackHoleDynamics, BinaryBlackHoleDynamics, BinaryBlackHoleIonization, \
    BinaryBlackHoleSpheroidDynamics, BinaryBlackHoleEccDynamics
from mcfacts.modules.formation import BinaryBlackHoleFormation
from mcfacts.modules.gas_hardening import BinaryBlackHoleGasHardening
from mcfacts.modules.gw import BinaryBlackHoleEvolveGW, InnerBlackHoleDynamics
from mcfacts.modules.merge import ProcessBinaryBlackHoleMergers, ProcessEMRIMergers
from mcfacts.modules.migration import ProgradeBlackHoleMigration, BinaryBlackHoleMigration
from mcfacts.objects.actors import InitialBlackHoleReclassification, InnerDiskFilter, FlipRetroProFilter, \
    InitialStarReclassification
from mcfacts.objects.actors.reality_checks import SingleBlackHoleRealityCheck, BinaryBlackHoleRealityCheck
from mcfacts.objects.agn_object_array import *
from mcfacts.objects.disk import AGNDisk
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.populators import SingleBlackHolePopulator, SingleStarPopulator
from mcfacts.objects.snapshot import TxtSnapshotHandler, IniSnapshotHandler
from mcfacts.objects.timeline import SimulationTimeline


#### Methods
def main(settings: SettingsManager):
    # Check for existing output files and overwrite flags
    # TODO: These checks probably should be done via the snapshot handler
    if settings.overwrite_files == False and os.path.isdir(settings.output_dir):
        assert False, f"Output directory {settings.output_dir} already exist. Set --overwrite_files=True to clear the directory."

    if settings.overwrite_files and os.path.isdir(settings.output_dir):
        shutil.rmtree(settings.output_dir)

    # Create the IO handlers and save the current settings
    snapshot_handler = TxtSnapshotHandler(settings = settings)

    ini_handler = IniSnapshotHandler(settings=settings)
    ini_handler.save_settings("./runs", "settings", settings)

    # Load disk model and setup empty filing cabinet for result populations
    agn_disk = AGNDisk(settings)
    population_cabinet = FilingCabinet()

    pbar = tqdm(total=settings.galaxy_num, position=0, leave=True)

    for galaxy_id in range(settings.galaxy_num):
        pbar.set_description(f"Running Galaxy {galaxy_id}")
        pbar.update(1)

        # The Galaxy class creates a random generated based on this seed,
        # Philox output for n and n+-1 seeds are uncorrelated
        galaxy_seed = settings.seed - galaxy_id

        # Create instance of galaxy
        galaxy = Galaxy(seed=galaxy_seed, runs_folder="./runs", galaxy_id=str(galaxy_id), settings=settings)

        # Create instance of populators
        single_bh_populator = SingleBlackHolePopulator()
        single_star_populator = SingleStarPopulator()
        galaxy.populate([single_bh_populator, single_star_populator], agn_disk)

        # Create timeline to classify objects created during population
        pre_timeline = SimulationTimeline("Reclassification", timesteps=1, timestep_length=0)

        # Run stars reclassification first, since it can convert stars to bh under certain conditions
        pre_timeline.add_timeline_actor(InitialStarReclassification())
        pre_timeline.add_timeline_actor(InitialBlackHoleReclassification())
        pre_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())

        galaxy.run(pre_timeline, agn_disk)

        # Create timeline to run main simulation
        active_phase_timeline = SimulationTimeline("Active Timeline",
                                               timesteps=settings.active_timestep_num,
                                               timestep_length=galaxy.settings.active_timestep_duration_yr)

        # Initial check to make sure our single black holes are real
        active_phase_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())

        # Get names of different singleton arrays we run through the same module
        prograde_array = galaxy.settings.bh_prograde_array_name
        innerdisk_array = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name

        # Single Object Physics
        active_phase_timeline.add_timeline_actors([
            ProgradeBlackHoleMigration(target_array=innerdisk_array),
            ProgradeBlackHoleMigration(target_array=prograde_array),
            SingleBlackHoleRealityCheck(),

            ProgradeBlackHoleAccretion(target_array=innerdisk_array),
            ProgradeBlackHoleAccretion(target_array=prograde_array),
            ProgradeBlackHoleDamping(target_array=innerdisk_array),
            ProgradeBlackHoleDamping(target_array=prograde_array),

            EvolveRetrogradeBlackHoles(),
            SingleBlackHoleRealityCheck(),

            InnerBlackHoleDynamics(target_array=innerdisk_array),
            InnerBlackHoleDynamics(target_array=inner_gw_only_array),
            SingleBlackHoleDynamics(target_array=innerdisk_array),
            SingleBlackHoleDynamics(target_array=prograde_array),
        ])

        # Binary Object Physics
        active_phase_timeline.add_timeline_actors([
            BinaryBlackHoleDamping(),

            BinaryBlackHoleDynamics(reality_merge_checks=False),
            ProcessBinaryBlackHoleMergers(),

            BinaryBlackHoleEccDynamics(reality_merge_checks=False),
            ProcessBinaryBlackHoleMergers(),

            BinaryBlackHoleGasHardening(reality_merge_checks=False),
            ProcessBinaryBlackHoleMergers(),

            BinaryBlackHoleAccretion(reality_merge_checks=False),
            ProcessBinaryBlackHoleMergers(),

            BinaryBlackHoleSpheroidDynamics(reality_merge_checks=False),
            ProcessBinaryBlackHoleMergers(),

            RecaptureBinaryBlackHoles(),
            BinaryBlackHoleMigration(),
            BinaryBlackHoleRealityCheck(),

            BinaryBlackHoleEvolveGW(),

            BinaryBlackHoleIonization(),
            ProcessBinaryBlackHoleMergers(),

            BinaryBlackHoleFormation()
        ])

        # Create new prograde black holes
        active_phase_timeline.add_timeline_actor(CaptureNSCProgradeBlackHoles())

        # Population Filters
        active_phase_timeline.add_timeline_actor(InnerDiskFilter())
        active_phase_timeline.add_timeline_actor(FlipRetroProFilter())

        # Handle EMRI Dynamics
        active_phase_timeline.add_timeline_actor(ProcessEMRIMergers())

        # Rub the active timeline
        galaxy.run(active_phase_timeline, agn_disk)

        # Ignore consistency checks on these arrays since they are allowed to have duplicates
        population_cabinet.ignore_consistency_check("blackholes_merged")
        population_cabinet.ignore_consistency_check("blackholes_lvk")

        # Grab array names from settings manager
        bbh_merged_array = galaxy.settings.bbh_merged_array_name
        bbh_lvk_array = galaxy.settings.bbh_gw_array_name
        emri_merged_array = galaxy.settings.emri_array_name
        bh_ejected_array = galaxy.settings.bh_ejected_array_name

        # Sort objects into the final population cabinet containing results from all galaxies
        if bh_ejected_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_ejected", galaxy.filing_cabinet.get_array(bh_ejected_array))

        if bbh_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_merged", galaxy.filing_cabinet.get_array(bbh_merged_array))

        if bbh_lvk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_lvk", galaxy.filing_cabinet.get_array(bbh_lvk_array))

        if innerdisk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(innerdisk_array))

        if inner_gw_only_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(inner_gw_only_array))

        if emri_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(emri_merged_array))

    pbar.close()

    # Save the entire population cabinet
    snapshot_handler.save_cabinet("./runs", "population", population_cabinet)


if __name__ == "__main__":
    main(SettingsManager())
