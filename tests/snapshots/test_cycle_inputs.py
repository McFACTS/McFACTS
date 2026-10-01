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
from mcfacts.objects.snapshot import TxtSnapshotHandler, IniSnapshotHandler

######## Setup ########

######## Functions ########
def non_default_settings(settings):
    # Initialize output
    out = []
    # Loop through the settings in DEFAULT_SETTINGS
    for prop in DEFAULT_SETTINGS:
        # Skip settings_file
        if prop.name == "settings_file":
            continue
        # Find any that don't match
        if prop.value != getattr(settings, prop.name):
            out.append(prop.name)
    return out

def settings_equal(A, B):
    # Loop through the settings in DEFAULT_SETTINGS
    for prop in DEFAULT_SETTINGS:
        # Skip settings_file
        if prop.name == "settings_file":
            continue
        # Find any that don't match
        if getattr(A, prop.name) != getattr(B, prop.name):
            return False
    return True

######## Tests ########
def test_cycle_inputs():
    """ Test that saving and loading inputs persists settings.
    """
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as wkdir:
        # Define some settings
        live = SettingsManager()
        # Check that all settings are default
        assert len(non_default_settings(live)) == 0, \
            f"{len(non_default_settings)} settings are not default values!"
        # Set a setting
        live.set_preprocessing("seed", 42)
        assert live.seed == 42
        # Check that one setting is not default
        assert len(non_default_settings(live)) == 1, \
            f"Failed to set seed"

        ## IniSnapshotHandler ##
        # Create an ini-handler
        ini_handler = IniSnapshotHandler(settings=live)
        ini_handler.save_settings(wkdir, "seedset.ini", live)
        # Create an unrelated handler and load the inifile
        ini_loader = IniSnapshotHandler()
        loaded = ini_loader.load_settings(wkdir, "seedset.ini")
        # Check that seed is correct
        assert loaded.seed == 42, \
            "Serialization failed for IniSnapshotHandler!"
        # Check that loaded == live
        assert settings_equal(live, loaded), \
            "Serialization failed for IniSnapshotHandler!"
        assert len(non_default_settings(loaded)) == 1, \
            "Serialization failed for IniSnapshotHandler!"
        live.set_preprocessing("seed", 9001)
        assert live.seed == 9001
        assert not settings_equal(live, loaded), "Equivalence check failed!"

        ## TxtSnapshotHandler ##
        txt_handler = TxtSnapshotHandler(settings=live)
        txt_handler.save_settings(wkdir, "seedset.txt", live)
        # Create an unrelated handler and load the txt snapshot
        txt_loader = TxtSnapshotHandler()
        loaded = txt_handler.load_settings(wkdir, "seedset.txt")
        assert loaded.seed == 9001, \
            "Serialization failed for TxtSnapshotHandler"
        assert settings_equal(live, loaded), \
            "Serialization failed for TxtSnapshotHandler"
        assert len(non_default_settings(loaded)) == 1, \
            "Serialization failed for TxtSnapshotHandler!"

######## Main ########
def main():
    test_cycle_inputs()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
