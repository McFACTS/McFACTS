#!/usr/bin/env python3
"""Test the AGNDisk object"""
######## Imports ########
#### Standard ####
import tempfile
import os

#### Third Party ####
import numpy as np

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

def agn_objects_are_equal(A, B):
    """Check to see if one AGN object array is equal to another"""
    # Check the population objects
    for name, agn_object_array in A.items():
        if name not in B:
            return False
        for key, value in agn_object_array.get_super_dict().items():
            if key not in B[name]:
                return False
            if not np.all(value == B[name][key]):
                return False
    return True

######## Tests ########
def test_run_galaxy():
    """ Test running a galaxy without main """
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as wkdir:
        # Define some settings
        live = SettingsManager()
        live.set_preprocessing("output_dir", wkdir)
        live.set_preprocessing("save_state", True)
        #live.set_preprocessing("save_each_timestep", True)
        assert live.output_dir == wkdir, \
            "Failed to setup output directory"
        assert live.save_state
        #assert live.save_each_timestep

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
        txt_handler.save_cabinet(
            live.output_dir,
            "population",
            population_cabinet,
        )
        print(os.listdir(wkdir))
        # Get an unrelated TxtSnapshotHandler
        txt_loader = TxtSnapshotHandler(settings = \
            {key: value for key, value in live.settings_finals.items()})
        # Load some AGN objects
        txt_agn_pop_objs = txt_loader.load_cabinet(
            live.output_dir,
            "population",
        )
        # Check the population objects
        assert agn_objects_are_equal(
            population_cabinet.agn_objects,
            txt_agn_pop_objs,
        )
        # Load the final state of the galaxy
        txt_gal00_s02_objs = txt_loader.load_cabinet(
            f"{wkdir}/gal00",
            "gal00_s02",
        )
        # Was used for testing TxtSnapshotHandler.load/save cabinet
        """
        for item in os.listdir(f"{wkdir}/gal00"):
            kind = "directory" if os.path.isdir(f"{wkdir}/gal00/{item}") else \
                "file"
            print(item, kind)
            if kind == "directory":
                print(os.listdir(f"{wkdir}/gal00/{item}"))

        for name, arr in galaxy.filing_cabinet.agn_objects.items():
            print(name)
            print(name in txt_gal00_s02_objs)
        raise Exception
        if False:
            print(name in txt_gal00_s02_objs)
            galaxy.filing_cabinet.agn_objects
            for key, value in arr.items():
                print(key)
                print(key in galaxy.filing_cabinet.agn_objects[name])
        # Check the final state of the galaxy
        assert agn_objects_are_equal(
            galaxy.filing_cabinet.agn_objects,
            txt_gal00_s02_objs,
        )
        """
        ## HDF5SnapshotHandler ##
        # Create the IO handlers and save the current settings
        hdf_handler = HDF5SnapshotHandler(settings = live)

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
            "live.hdf5",
            population_cabinet,
            addr=f"{hdf_handler.label}/population",
        )
        print(os.listdir(wkdir))
        # Get an unrelated TxtSnapshotHandler
        hdf_loader = HDF5SnapshotHandler(settings = \
            {key: value for key, value in live.settings_finals.items()})
        # Load some AGN objects
        hdf_agn_pop_objs = hdf_loader.load_cabinet(
            live.output_dir,
            "live.hdf5",
            addr=f"{hdf_handler.label}/population",
        )
        # Check the population objects
        assert agn_objects_are_equal(
            population_cabinet.agn_objects,
            hdf_agn_pop_objs,
        )
        print("Not dead yet!")
        # Load the final state of the galaxy
        hdf_gal00_s02_objs = hdf_loader.load_cabinet(
            f"{wkdir}",
            "live.hdf5",
            addr=f"{hdf_handler.label}/gal00/s02",
        )

######## Main ########
def main():
    test_run_galaxy()
    return

######## Execution ########
if __name__ == "__main__":
    main()
