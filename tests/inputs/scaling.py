#!/usr/bin/env python3
"""Test the AGNDisk object"""
######## Imports ########
#### Standard ####
import tempfile
import os

#### Third Party ####
import numpy as np

#### Local ####
from mcfacts.inputs.scaling import setup_scaling, disk_truncation
from mcfacts.inputs.settings_manager import SettingsManager, DEFAULT_SETTINGS
from mcfacts.objects.disk import AGNDisk
from mcfacts.objects.snapshot import IniSnapshotHandler

######## Setup ########

######## Tests ########

def test_noop():
    """Test that calling setup_scaling does nothing if scaling is off"""
    # Define some settings
    live = SettingsManager()
    # Check they're all defaults
    for prop in DEFAULT_SETTINGS:
        assert getattr(live, prop.name) == prop.value, \
            f"{prop.name} has value {getattr(live, prop.name)}, " \
            f"but should be default ({prop.value})!"
    # Check that the defaults are sane
    assert not live.flag_use_scaling, \
        f"The default settings now scale the disk in preprocessing. " \
        f"If this is intended, remove this assert."
    assert live.disk_truncation == "none", \
        f"The default settings now truncate the disk in preprocessing. " \
        f"If this is intended, remove this assert."
    assert live.disk_radius_max_pc == 0., \
        f"The default settings now enforce a maximum disk size in parsecs. " \
        f"If this is intended, remove this assert."
    # Scale them
    setup_scaling(live)
    # Check they're STILL defaults
    for prop in DEFAULT_SETTINGS:
        assert getattr(live, prop.name) == prop.value, \
            f"{prop.name} has value {getattr(live, prop.name)}, " \
            f"but should be default ({prop.value})!"
            
def test_max_pc():
    """Test the max_pc argument -- has nothing to do with the
    disk_truncation argument.
    """
    # Define some settings
    live = SettingsManager()
    # Turn off truncation and max pc
    live.set_preprocessing("flag_use_scaling",      False)
    live.set_preprocessing("disk_truncation",       "none")
    live.set_preprocessing("disk_radius_max_pc",    0.)
    live.set_preprocessing("flag_use_pagn",         False)
    # Construct a disk
    untruncated_disk = AGNDisk(live)
    # Get the outer disk radius
    untruncated_disk_radius_outer = untruncated_disk.disk_radius_outer
    # get the radius in pc
    untruncated_radius_pc = untruncated_disk.pc_dist(
        untruncated_disk.smbh_mass,
        untruncated_disk_radius_outer,
    )
    # Set disk_radius_max_pc to half of outer_disk_radius in parsecs
    live.set_preprocessing("disk_radius_max_pc", 0.5 * untruncated_radius_pc)
    truncated_disk = AGNDisk(live)
    assert truncated_disk.disk_radius_outer < untruncated_disk_radius_outer
    assert np.isclose(
        truncated_disk.disk_radius_outer,
        0.5 * untruncated_disk_radius_outer,
    )
    # Set the max disk radius (in parsecs) to something larger than the disk
    live.set_preprocessing("disk_radius_max_pc", 2.0 * untruncated_radius_pc)
    test_disk = AGNDisk(live)
    assert np.isclose(
        untruncated_disk_radius_outer,
        test_disk.disk_radius_outer,
    )
    # Set the max disk radius (in parsecs), and we really mean it this time
    live.set_preprocessing("disk_radius_max_pc", -2.0 * untruncated_radius_pc)
    extended_disk = AGNDisk(live)
    assert extended_disk.disk_radius_outer > untruncated_disk_radius_outer
    assert np.isclose(
        extended_disk.disk_radius_outer,
        2.0 * untruncated_disk_radius_outer,
    )

def test_disk_truncation():
    """Test the opacity disk truncation option"""
    # Define some settings
    live = SettingsManager()
    # before
    radius_before = 5.e4
    # Turn off truncation and max pc
    live.set_preprocessing("flag_use_scaling",      False)
    live.set_preprocessing("disk_truncation",       "opacity-equals-inner-disk")
    live.set_preprocessing("disk_radius_max_pc",    0.)
    live.set_preprocessing("disk_model_name",       "sirko_goodman")
    live.set_preprocessing("smbh_mass",             1.e8)
    live.set_preprocessing("disk_radius_outer",     radius_before)
    live.set_preprocessing("flag_use_pagn",         True)
    # Run the truncator
    mydisk = disk_truncation(live, return_disk=True)
    assert live.disk_radius_outer < radius_before
    # TODO note: I have yet to test making the disk radius smaller
    # This is why AGNDisk is not returned by the setup_scaling function
    assert mydisk.disk_radius_outer == radius_before
    # Test priority
    live.set_preprocessing(
        "disk_radius_max_pc",
        0.5 * mydisk.pc_dist(mydisk.smbh_mass, mydisk.disk_radius_outer),
    )
    truncated_disk = AGNDisk(live)
    assert np.isclose(
        0.5 * mydisk.disk_radius_outer,
        truncated_disk.disk_radius_outer,
    )

def test_serialization():
    """\
    Test that a preprocessed disk can be saved, and a loaded disk keeps changes
    """
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as wkdir:
        # Define some settings
        live = SettingsManager()
        # Set scaling to True
        live.set_preprocessing("flag_use_scaling",      True)
        # Mess with nuclear star cluster mass
        nsc_fudge = 1.e12
        live.set_preprocessing("nsc_mass",              nsc_fudge)
        # Make sure nsc scaling is on
        live.set_preprocessing("scale_nsc_mass",        "neumayer-early")
        # Apply scaling
        setup_scaling(live)
        # Check that nsc_mass is now less than the given value
        assert live.nsc_mass < nsc_fudge
        # Save the file
        ini_handler = IniSnapshotHandler(settings=live)
        ini_handler.save_settings(wkdir, "scaled_nsc.ini", live)
        # Create an unrelated handler and load the inifile
        ini_loader = IniSnapshotHandler()
        loaded = ini_loader.load_settings(wkdir, "scaled_nsc.ini")
        # First check that live actually noted changes
        assert live.flag_use_scaling, \
            f"Even before i/o, set_preprocessing is not respected"
        # Second, test to see if flag_use_scaling is True
        # This tests if settings manually passed to set_preprocessing 
        #  were saved.
        assert loaded.flag_use_scaling, \
            f"set_preprocessing was respected by live SettingsManager, but " \
            f"does not persist upon save / load."
        # Third, test that loaded.nsc_mass is not nsc_fudge
        assert loaded.nsc_mass != nsc_fudge, \
            f"The loaded nsc mass is not default and not scaled."
        # Fourth test equivalence of nsc_mass
        assert loaded.nsc_mass == live.nsc_mass, \
            f"The loaded nsc mass is different from the live nsc mass"
        # Fifth, check serialization
        for prop in DEFAULT_SETTINGS:
            assert getattr(live, prop.name) == getattr(loaded, prop.name), \
                f"live setting {prop.name} has value " \
                f"{getattr(live, prop.name)}, while loaded setting has " \
                f" value {getattr(loaded, prop.name)}!"
            

######## Main ########
def main():
    test_noop()
    test_max_pc()
    test_disk_truncation()
    test_serialization()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
