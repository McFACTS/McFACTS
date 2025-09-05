import sys
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.objects.actors import InitialObjectReclassification, InnerDiskFilter, FlipRetroProFilter
from mcfacts.objects.agn_object_array import *
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.populators import SingleBlackHolePopulator
from mcfacts.objects.snapshot import TxtSnapshotHandler
from mcfacts.objects.timeline import SimulationTimeline
from mcfacts.modules.accretion import ProgradeBlackHoleAccretion, BinaryBlackHoleAccretion
from mcfacts.modules.damping import ProgradeBlackHoleDamping, BinaryBlackHoleDamping
from mcfacts.modules.disk_capture import EvolveRetrogradeBlackHoles, RecaptureBinaryBlackHoles, CaptureNSCProgradeBlackHoles
from mcfacts.modules.dynamics import InnerBlackHoleDynamics, SingleBlackHoleDynamics, BinaryBlackHoleDynamics
from mcfacts.modules.gas_hardening import BinaryBlackHoleGasHardening
from mcfacts.modules.formation import BinaryBlackHoleFormation
from mcfacts.modules.merge import ProcessBinaryBlackHoleMergers, ProcessEMRIMergers
from mcfacts.modules.migration import ProgradeBlackHoleMigration
from mcfacts.objects.actors.reality_checks import SingleBlackHoleRealityCheck, BinaryBlackHoleRealityCheck
from mcfacts.modules.gw import BinaryBlackHoleEvolveGW

settings = SettingsManager({
    "verbose": False,
    "override_files": True,
    "save_state": True,
    "save_each_timestep": True
})

def args():
    # TODO: Handle argument inputs, support legacy .ini files with SnapshotHandler framework
    pass

def main():
    population_cabinet = FilingCabinet()

    agn_disk = AGNDisk(settings)
    snapshot_handler = TxtSnapshotHandler(settings)

    n_galaxy = 100

    pbar = tqdm(total=n_galaxy, position=0, leave=True)

    for galaxy_id in range(n_galaxy):
        pbar.set_description(f"Running Galaxy {galaxy_id}")
        pbar.update(1)

        galaxy_seed = 223849053863469657747974663531730220530 - galaxy_id

        # Create instance of galaxy
        galaxy = Galaxy(seed=galaxy_seed, runs_folder="./runs", galaxy_id=galaxy_id, settings=settings)

        # Create instance of populators
        single_bh_populator = SingleBlackHolePopulator()
        #single_star_populator = SingleStarPopulator("single_stars")
        galaxy.populate([single_bh_populator], agn_disk)

        # Create timeline to classify objects created during population
        pre_timeline = SimulationTimeline("Reclassification", timesteps=1, timestep_length=0)
        pre_timeline.add_timeline_actor(InitialObjectReclassification())
        pre_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())
        galaxy.run(pre_timeline, agn_disk)

        #settings=SettingsManager({**settings.settings_overrides, "verbose": True})
        # Create timeline to run main simulation
        dynamics_timeline = SimulationTimeline("Dynamics", timesteps=60, timestep_length=galaxy.settings.timestep_duration_yr)

        dynamics_timeline.add_timeline_actor(EvolveRetrogradeBlackHoles())
        dynamics_timeline.add_timeline_actor(FlipRetroProFilter())
        dynamics_timeline.add_timeline_actor(InnerDiskFilter())
        dynamics_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())

        dynamics_timeline.add_timeline_actor(ProcessEMRIMergers())

        prograde_array = galaxy.settings.bh_prograde_array_name
        innerdisk_array = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name

        innerdisk_dynamics = [
            ProgradeBlackHoleMigration(target_array=innerdisk_array),
            ProgradeBlackHoleAccretion(target_array=innerdisk_array),
            ProgradeBlackHoleDamping(target_array=innerdisk_array),
            SingleBlackHoleDynamics(target_array=innerdisk_array),
            InnerBlackHoleDynamics(target_array=innerdisk_array),
            InnerBlackHoleDynamics(target_array=inner_gw_only_array),
            SingleBlackHoleRealityCheck(),
        ]

        dynamics_timeline.add_timeline_actors(innerdisk_dynamics)

        singleton_dynamics = [
            ProgradeBlackHoleMigration(target_array=prograde_array),
            ProgradeBlackHoleAccretion(target_array=prograde_array),
            ProgradeBlackHoleDamping(target_array=prograde_array),
            SingleBlackHoleDynamics(target_array=prograde_array),
            CaptureNSCProgradeBlackHoles(),
            SingleBlackHoleRealityCheck()
        ]

        dynamics_timeline.add_timeline_actors(singleton_dynamics)

        binary_dynamics = [
            BinaryBlackHoleDamping(),
            BinaryBlackHoleDynamics(reality_merge_checks=True),
            BinaryBlackHoleAccretion(reality_merge_checks=True),
            BinaryBlackHoleGasHardening(reality_merge_checks=True),
            RecaptureBinaryBlackHoles(),
            BinaryBlackHoleEvolveGW(),
            ProcessBinaryBlackHoleMergers(),
            BinaryBlackHoleRealityCheck()
        ]

        dynamics_timeline.add_timeline_actors(binary_dynamics)

        dynamics_timeline.add_timeline_actor(BinaryBlackHoleFormation())

        galaxy.run(dynamics_timeline, agn_disk)

        # Create timeline to cleanup the final bits and bobs at the end of a
        # cleanup_timeline = SimulationTimeline("Cleanup", timesteps=1, timestep_length=0)
        # cleanup_timeline.add_timeline_actor(BreakupBinaryBlackHoles())
        # galaxy.run(cleanup_timeline, agn_disk)

        if "blackholes_merged" in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_merged", galaxy.filing_cabinet.get_array("blackholes_merged"))

    pbar.close()

    snapshot_handler.save_cabinet("./runs", "population", population_cabinet)

if __name__ == "__main__":
    main()