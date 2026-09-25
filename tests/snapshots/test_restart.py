#!/usr/bin/env python3
"""Test the AGNDisk object"""
######## Imports ########
#### Standard ####
import collections
import itertools
import subprocess
import tempfile
import time
import os
from os.path import isfile, isdir, join

#### Third Party ####
import numpy as np

#### Vera ####
from xdata import Database

#### Local ####
from mcfacts.inputs.settings_manager import SettingsManager, DEFAULT_SETTINGS
from mcfacts.objects.agn_object_array import FilingCabinet
from mcfacts.objects.disk import AGNDisk
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.populators import SingleBlackHolePopulator, SingleStarPopulator
from mcfacts.objects.snapshot import TxtSnapshotHandler, IniSnapshotHandler
from mcfacts.objects.snapshot import HDF5SnapshotHandler
from mcfacts.simulation import run_galaxy

######## Setup ########
# Taken from <https://stackoverflow.com/a/9098295/4761692>
def named_product(**items):
    Options = collections.namedtuple('Options', items.keys())
    return itertools.starmap(Options, itertools.product(*items.values()))

def agn_objects_are_equal(A, B):
    """Check to see if one AGN object array is equal to another"""
    # Check the population objects
    for name, agn_object_array in A.items():
        if name not in B:
            return False
        if isinstance(agn_object_array, dict):
            for key, value in agn_object_array.items():
                if key not in B[name]:
                    return False
                if not np.all(value == B[name][key]):
                    return False
        else:
            for key, value in agn_object_array.get_super_dict().items():
                if key not in B[name]:
                    return False
                if not np.all(value == B[name][key]):
                    return False
    return True

######## HDF5 settings ########
HDF5_SNAPSHOT_MODES = ["columns", "compound"]
HDF5_SNAPSHOT_COMPRESSIONS = ["none", "gzip"]
TEST_HDF5_SETTING_REALS = named_product(
    mode = HDF5_SNAPSHOT_MODES,
    compression = HDF5_SNAPSHOT_COMPRESSIONS,
)

######## Tests ########
def test_run_galaxy():
    """ Test running a galaxy without main """
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as wkdir:
        # Define some settings
        live = SettingsManager()
        live.set_preprocessing("output_dir", wkdir)
        live.set_preprocessing("save_state", True)
        live.set_preprocessing("save_each_timestep", True)
        assert live.output_dir == wkdir, \
            "Failed to setup output directory"
        assert live.save_state
        assert live.save_each_timestep

        # Start timer
        tic = time.perf_counter()
        # Create the IO handlers and save the current settings
        txt_handler = TxtSnapshotHandler(settings = live)

        # Load disk model and setup empty filing cabinet for result populations
        agn_disk = AGNDisk(live)
        population_cabinet = FilingCabinet()

        # Create instance of galaxy
        galaxy = Galaxy(
            seed=42,
            runs_folder=live.output_dir,
            galaxy_id="0",
            settings=live,
        )

        ## Populate galaxy with a new population ##
        # Create instance of populators
        single_bh_populator = SingleBlackHolePopulator(settings=live)
        single_star_populator = SingleStarPopulator(settings=live)
        galaxy.populate([single_bh_populator, single_star_populator], agn_disk)

        ## Run the galaxy ##
        run_galaxy(live, galaxy, agn_disk=agn_disk)

        ## Manage simulation outputs ##
        # Ignore consistency checks on these arrays since they are allowed to have duplicates
        population_cabinet.ignore_consistency_check("blackholes_merged")
        population_cabinet.ignore_consistency_check("blackholes_lvk")

        # Grab array names from settings manager
        prograde_array = galaxy.settings.bh_prograde_array_name
        innerdisk_array = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name
        bbh_merged_array = galaxy.settings.bbh_merged_array_name
        bbh_lvk_array = galaxy.settings.bbh_gw_array_name
        emri_merged_array = galaxy.settings.emri_array_name
        bh_ejected_array = galaxy.settings.bh_ejected_array_name

        # Sort objects into the final population cabinet containing results from all galaxies
        if bh_ejected_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_ejected",
                galaxy.filing_cabinet.get_array(bh_ejected_array),
            )

        if bbh_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_merged",
                galaxy.filing_cabinet.get_array(bbh_merged_array),
            )

        if bbh_lvk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_lvk",
                galaxy.filing_cabinet.get_array(bbh_lvk_array),
            )

        if innerdisk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(innerdisk_array),
            )

        if inner_gw_only_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(inner_gw_only_array),
            )

        if emri_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(emri_merged_array),
            )

        # Save the entire population cabinet
        txt_handler.save_cabinet(
            live.output_dir,
            "population",
            population_cabinet,
        )
        # End timer
        toc = time.perf_counter()
        txt_time = toc - tic
        # Get size of directory
        du_output = subprocess.run(
            ["du", "-sh", wkdir],
            capture_output=True,
            text=True,
        ).stdout
        txt_size = du_output.split("\t")[0]

        # Get an unrelated TxtSnapshotHandler
        txt_loader = TxtSnapshotHandler(settings = \
            {key: value for key, value in live.settings_finals.items()})
        # Load some AGN objects
        txt_agn_pop_objs = txt_loader.load_cabinet(
            live.output_dir,
            "population",
        )[0]
        # Check the population objects
        assert agn_objects_are_equal(
            population_cabinet.agn_objects,
            txt_agn_pop_objs,
        )
        # Load the final state of the galaxy
        txt_gal00_s02_objs = txt_loader.load_cabinet(
            f"{wkdir}/gal00",
            "gal00_s02",
        )[0]
        ## HDF5SnapshotHandler ##
        live.set_preprocessing("settings_snapshot", "hdf5")
        live.set_preprocessing("cabinet_snapshot", "hdf5")
        live.set_preprocessing("hdf5_snapshot_file", "live.hdf5")
        live.set_preprocessing("hdf5_snapshot_label", "live")
        live.set_preprocessing("hdf5_snapshot_mode", "compound")
        live.set_preprocessing("hdf5_snapshot_compression", "gzip")
        # Create the IO handlers and save the current settings
        hdf_handler = live.new_cabinet_snapshot()
        assert isinstance(hdf_handler, HDF5SnapshotHandler)
        # Start hdf timer
        tic = time.perf_counter()
        # Load disk model and setup empty filing cabinet for result populations
        agn_disk = AGNDisk(live)
        population_cabinet = FilingCabinet()

        # Create instance of galaxy
        galaxy = Galaxy(
            seed=42,
            runs_folder=live.output_dir,
            galaxy_id="0",
            settings=live,
        )

        ## Populate galaxy with a new population ##
        # Create instance of populators
        single_bh_populator = SingleBlackHolePopulator()
        single_star_populator = SingleStarPopulator()
        galaxy.populate([single_bh_populator, single_star_populator], agn_disk)

        ## Run the galaxy ##
        run_galaxy(live, galaxy, agn_disk=agn_disk)

        ## Manage simulation outputs ##
        # Ignore consistency checks on these arrays since they are allowed to have duplicates
        population_cabinet.ignore_consistency_check("blackholes_merged")
        population_cabinet.ignore_consistency_check("blackholes_lvk")

        # Grab array names from settings manager
        prograde_array = galaxy.settings.bh_prograde_array_name
        innerdisk_array = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name
        bbh_merged_array = galaxy.settings.bbh_merged_array_name
        bbh_lvk_array = galaxy.settings.bbh_gw_array_name
        emri_merged_array = galaxy.settings.emri_array_name
        bh_ejected_array = galaxy.settings.bh_ejected_array_name

        # Sort objects into the final population cabinet containing results from all galaxies
        if bh_ejected_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_ejected",
                galaxy.filing_cabinet.get_array(bh_ejected_array),
            )

        if bbh_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_merged",
                galaxy.filing_cabinet.get_array(bbh_merged_array),
            )

        if bbh_lvk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_lvk",
                galaxy.filing_cabinet.get_array(bbh_lvk_array),
            )

        if innerdisk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(innerdisk_array),
            )

        if inner_gw_only_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(inner_gw_only_array),
            )

        if emri_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(emri_merged_array),
            )

        # Save the entire population cabinet
        hdf_handler.save_cabinet(
            live.output_dir,
            live.hdf5_snapshot_file,
            population_cabinet,
            addr=f"{hdf_handler.label}/population",
        )
        # End timer
        toc = time.perf_counter()
        hdf_time = toc - tic
        # Get an unrelated HDF5SnapshotHandler
        hdf_loader = HDF5SnapshotHandler(
            settings = SettingsManager(
                {key: value for key, value in live.settings_finals.items()}
            )
        )
        # Load some AGN objects
        hdf_agn_pop_objs = hdf_loader.load_cabinet(
            live.output_dir,
            live.hdf5_snapshot_file,
            addr=f"{hdf_handler.label}/population",
        )[0]
        # Check the population objects
        assert agn_objects_are_equal(
            population_cabinet.agn_objects,
            hdf_agn_pop_objs,
        )
        # print("Not dead yet!")
        # Loop
        #os.system(f"h5ls -r {wkdir}/live.hdf5")
        # Feels good to be able to use this
        db = Database(join(wkdir, live.hdf5_snapshot_file), "live/gal00")

        # Loop things
        for name in db.list_items():
            # Construct the full address
            addr = f"live/gal00/{name}"
            # Get parts of the string
            parts = name.split("_")
            #print(len(parts), name, addr, parts)
            hdf_agn_objs, hdf_all_else = hdf_loader.load_cabinet(
                wkdir,
                live.hdf5_snapshot_file,
                addr = addr,
            )
            # Find state snapshots
            if len(parts) == 2:
                # Identify state
                state = parts[1]
                # Load some AGN objects
                txt_agn_objs, txt_all_else = txt_loader.load_cabinet(
                    live.output_dir,
                    name,
                )
                assert agn_objects_are_equal(
                    txt_agn_objs,
                    hdf_agn_objs,
                )
                for key in txt_all_else:
                    assert key in hdf_all_else
                    assert txt_all_else[key] == hdf_all_else[key]
            # Find timestep snapshots
            elif len(parts) == 5:
                # Identify state
                prev_state = parts[1]
                next_state = parts[3]
                tmpdir = join(
                    wkdir,
                    parts[0],
                    f"{parts[0]}_{prev_state}_to_{next_state}",
                )
                # Load some AGN objects
                txt_agn_objs, txt_all_else = txt_loader.load_cabinet(
                    tmpdir,
                    name,
                )
                try:
                    assert agn_objects_are_equal(
                        hdf_agn_objs,
                        txt_agn_objs,
                    )
                except AssertionError:
                    for item, agn_object_array in hdf_agn_objs.items():
                        if item not in txt_agn_objs:
                            # Initialize empty
                            empty = True
                            for key, value in agn_object_array.items():
                                if np.size(value) > 0:
                                    empty = False
                            if empty:
                                continue
                            print(f"{item} not in txt_agn_objs for {name}")
                        for key, value in agn_object_array.items():
                            if key not in txt_agn_objs[item]:
                                print(f"{key} not in txt_agn_objs {item} for {name}")
                            if not np.all(value == txt_agn_objs[item][key]):
                                print(f"Unequal;")
                                print(
                                    f"HDF5 type: {type(value)}; "
                                    f"shape: {np.shape(value)}; "
                                    f"value: {value}"
                                )
                                print(
                                    f"Txt type: {type(txt_agn_objs[item][key])}; "
                                    f"shape: {np.shape(txt_agn_objs[item][key])}; "
                                    f"value: {txt_agn_objs[item][key]}"
                                )
                for key in txt_all_else:
                    assert key in hdf_all_else
                    assert txt_all_else[key] == hdf_all_else[key]
            # Die
            else:
                raise RuntimeError(f"Unaccounted group: {addr}")

        ## Report ##
        print(f"TxtSnapshotHandler  Size: {txt_size}")
        hdf_size = os.path.getsize(join(wkdir, live.hdf5_snapshot_file)) \
            / 1_000_000
        print(f"HDF5SnapshotHandler Size: {hdf_size} MB")
        print(f"TxtSnapshotHandler  Time: {txt_time:.3f} s")
        print(f"HDF5SnapshotHandler Time: {hdf_time:.3f} s")

def test_compression():
    """ Test running a galaxy with hdf5 compression"""
    # Loop settings
    for mode, compression in TEST_HDF5_SETTING_REALS:
      # Create a temporary workspace
      with tempfile.TemporaryDirectory() as wkdir:
        # Define some settings
        live = SettingsManager()
        live.set_preprocessing("output_dir", wkdir)
        live.set_preprocessing("active_timestep_num", 10)
        live.set_preprocessing("save_state", True)
        live.set_preprocessing("save_each_timestep", True)
        live.set_preprocessing("settings_snapshot", "hdf5")
        live.set_preprocessing("cabinet_snapshot", "hdf5")
        live.set_preprocessing("hdf5_snapshot_file", "live.hdf5")
        live.set_preprocessing("hdf5_snapshot_label", "live")
        live.set_preprocessing("hdf5_snapshot_mode", mode)
        live.set_preprocessing("hdf5_snapshot_compression", compression)

        ## HDF5SnapshotHandler ##
        # Get new settings_handler
        settings_handler = live.new_settings_snapshot()
        settings_handler.save_settings(wkdir, "settings.hdf5", live)
        # Create the IO handlers and save the current settings
        hdf_handler = live.new_cabinet_snapshot()
        assert isinstance(hdf_handler, HDF5SnapshotHandler)
        # Start hdf timer
        tic = time.perf_counter()
        # Get size of directory
        du_output = subprocess.run(
            ["du", "-sh", wkdir],
            capture_output=True,
            text=True,
        ).stdout
        txt_size = du_output.split("\t")[0]

        # Load disk model and setup empty filing cabinet for result populations
        agn_disk = AGNDisk(live)
        population_cabinet = FilingCabinet()

        # Create instance of galaxy
        galaxy = Galaxy(
            seed=42,
            runs_folder=live.output_dir,
            galaxy_id="0",
            settings=live,
        )

        ## Populate galaxy with a new population ##
        # Create instance of populators
        single_bh_populator = SingleBlackHolePopulator()
        single_star_populator = SingleStarPopulator()
        galaxy.populate([single_bh_populator, single_star_populator], agn_disk)

        ## Run the galaxy ##
        run_galaxy(live, galaxy, agn_disk=agn_disk)

        ## Manage simulation outputs ##
        # Ignore consistency checks on these arrays (duplicates allowed)
        population_cabinet.ignore_consistency_check("blackholes_merged")
        population_cabinet.ignore_consistency_check("blackholes_lvk")

        # Grab array names from settings manager
        prograde_array = galaxy.settings.bh_prograde_array_name
        innerdisk_array = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name
        bbh_merged_array = galaxy.settings.bbh_merged_array_name
        bbh_lvk_array = galaxy.settings.bbh_gw_array_name
        emri_merged_array = galaxy.settings.emri_array_name
        bh_ejected_array = galaxy.settings.bh_ejected_array_name

        # Sort objects into the final population with results from all galaxies
        if bh_ejected_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_ejected",
                galaxy.filing_cabinet.get_array(bh_ejected_array),
            )

        if bbh_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_merged",
                galaxy.filing_cabinet.get_array(bbh_merged_array),
            )

        if bbh_lvk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_lvk",
                galaxy.filing_cabinet.get_array(bbh_lvk_array),
            )

        if innerdisk_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(innerdisk_array),
            )

        if inner_gw_only_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(inner_gw_only_array),
            )

        if emri_merged_array in galaxy.filing_cabinet:
            population_cabinet.create_or_append_array(
                "blackholes_emri",
                galaxy.filing_cabinet.get_array(emri_merged_array),
            )

        # Save the entire population cabinet
        hdf_handler.save_cabinet(
            live.output_dir,
            "population.hdf5",
            population_cabinet,
            addr=f"{hdf_handler.label}/population",
        )
        # End timer
        toc = time.perf_counter()
        hdf_time = toc - tic
        # Get an unrelated HDF5SnapshotHandler
        hdf_loader = HDF5SnapshotHandler(
            settings = SettingsManager(
                {key: value for key, value in live.settings_finals.items()}
            )
        )
        # Load some AGN objects
        hdf_agn_pop_objs = hdf_loader.load_cabinet(
            live.output_dir,
            "population.hdf5",
            addr=f"{hdf_handler.label}/population",
        )[0]
        # Check the population objects
        assert agn_objects_are_equal(
            population_cabinet.agn_objects,
            hdf_agn_pop_objs,
        )
        ## Report ##
        print(
            "HDF5SnapshotHandler test ("
            f"mode: {mode}; compression: {compression}; "
            f"timesteps: {live.active_timestep_num})"
        )
        settings_size = os.path.getsize(join(wkdir, "settings.hdf5")) / int(1e6)
        pop_size = os.path.getsize(join(wkdir, "population.hdf5")) / int(1e6)
        states_size = os.path.getsize(join(wkdir, live.hdf5_snapshot_file)) \
            / int(1e6)
        print(f"Time: {hdf_time:.3} s!")
        print(f"Settings    size: {settings_size} MB")
        print(f"Population  size: {pop_size} MB")
        print(f"States      size: {states_size} MB")


######## Main ########
def main():
    test_compression()
    test_run_galaxy()
    return

######## Execution ########
if __name__ == "__main__":
    main()
