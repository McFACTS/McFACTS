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
from mcfacts.simulation import run_galaxy

######## Setup ########


######## Tests ########
def test_run_galaxy():
    """ Test running a galaxy without main """
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as wkdir:
        # Define some settings
        live = SettingsManager()
        live.set_preprocessing("output_dir", wkdir)
        assert live.output_dir == wkdir, \
            "Failed to setup output directory"

        # Create the IO handlers and save the current settings
        txt_handler = TxtSnapshotHandler(settings = live)

        # Load disk model and setup empty filing cabinet for result populations
        agn_disk = AGNDisk(live)
        population_cabinet = FilingCabinet()

        # Create instance of galaxy
        galaxy = Galaxy(
            seed=42,
            runs_folder=live.output_dir,
            galaxy_id=0,
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
        print(os.listdir(wkdir))



######## Main ########
def main():
    test_run_galaxy()
    return

######## Execution ########
if __name__ == "__main__":
    main()
