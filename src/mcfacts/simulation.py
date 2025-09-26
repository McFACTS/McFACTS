import warnings

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
from mcfacts.modules.migration import ProgradeBlackHoleMigration
from mcfacts.objects.actors import InitialObjectReclassification, InnerDiskFilter, FlipRetroProFilter
from mcfacts.objects.actors.reality_checks import SingleBlackHoleRealityCheck, BinaryBlackHoleRealityCheck
from mcfacts.objects.agn_object_array import *
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.populators import SingleBlackHolePopulator
from mcfacts.objects.snapshot import TxtSnapshotHandler
from mcfacts.objects.timeline import SimulationTimeline

settings = SettingsManager({
    "verbose": False,
    "override_files": True,
    "save_state": True,
    "save_each_timestep": True,
    "flag_use_pagn": True,
    "stalling_separation": 0.0
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

    # i = 0

    for galaxy_id in range(n_galaxy):
        pbar.set_description(f"Running Galaxy {galaxy_id}")
        pbar.update(1)

        # i = i +1
        #
        # if i != 93:
        #     continue

        galaxy_seed = 223849053863469657747974663531730220530 - galaxy_id

        # Create instance of galaxy
        galaxy = Galaxy(seed=galaxy_seed, runs_folder="./runs", galaxy_id=galaxy_id, settings=settings)

        # Create instance of populators
        single_bh_populator = SingleBlackHolePopulator()
        # single_star_populator = SingleStarPopulator("single_stars")
        galaxy.populate([single_bh_populator], agn_disk)

        # Create timeline to classify objects created during population
        pre_timeline = SimulationTimeline("Reclassification", timesteps=1, timestep_length=0)
        pre_timeline.add_timeline_actor(InitialObjectReclassification())
        pre_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())
        galaxy.run(pre_timeline, agn_disk)

        # settings=SettingsManager({**settings.settings_overrides, "verbose": True})
        # Create timeline to run main simulation
        dynamics_timeline = SimulationTimeline("Dynamics", timesteps=50,
                                               timestep_length=galaxy.settings.timestep_duration_yr)

        dynamics_timeline.add_timeline_actor(EvolveRetrogradeBlackHoles())

        dynamics_timeline.add_timeline_actor(InnerDiskFilter())
        dynamics_timeline.add_timeline_actor(SingleBlackHoleRealityCheck())

        dynamics_timeline.add_timeline_actor(ProcessEMRIMergers())

        prograde_array = galaxy.settings.bh_prograde_array_name
        innerdisk_array = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name

        innerdisk_dynamics = [
            ProgradeBlackHoleMigration(target_array=innerdisk_array),
            #ProgradeBlackHoleBondi(target_array=innerdisk_array),
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
            #ProgradeBlackHoleBondi(target_array=prograde_array),
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
        dynamics_timeline.add_timeline_actor(FlipRetroProFilter())

        galaxy.run(dynamics_timeline, agn_disk)

        population_cabinet.ignore_check_array("blackholes_merged")
        population_cabinet.ignore_check_array("blackholes_lvk")

        bbh_merged_array = galaxy.settings.bbh_merged_array_name
        bbh_lvk_array = galaxy.settings.bbh_gw_array_name

        if bbh_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_merged",galaxy.filing_cabinet.get_array(bbh_merged_array))

        if bbh_lvk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_lvk", galaxy.filing_cabinet.get_array(bbh_lvk_array))

        if innerdisk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(innerdisk_array))

        if inner_gw_only_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array("blackholes_emri", galaxy.filing_cabinet.get_array(inner_gw_only_array))

    pbar.close()

    snapshot_handler.save_cabinet("./runs", "population", population_cabinet)


if __name__ == "__main__":
    main()
