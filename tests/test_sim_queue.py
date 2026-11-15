#!/usr/bin/env python3
"""Test the mcfacts.objects.sim_queue module"""
######## Imports ########
#### Standard ####
import tempfile
#### Local ####
# Yeah, I know this is bad, but this is the module we are testing
from mcfacts.objects.sim_queue import *

######## Setup ########
# Number of Galaxies to run
N_GAL = 3

def queue_worthy_settings():
    """Return some example settings compatible with sim_queue"""
    # Define some settings
    settings = SettingsManager()
    # These are overwritten by the queue
    settings.set_preprocessing("show_timeline_progress",    True)
    settings.set_preprocessing("save_state",                True)
    settings.set_preprocessing("cabinet_snapshot",          "hdf5")
    settings.set_preprocessing("settings_snapshot",         "hdf5")
    settings.set_preprocessing("hdf5_snapshot_retries",     2*N_GAL)
    settings.set_preprocessing("hdf5_snapshot_sleep",       1.0)
    settings.set_preprocessing("hdf5_snapshot_mode",        "compound")
    settings.set_preprocessing("hdf5_snapshot_compression", "none")
    # This just makes testing it faster. :P
    settings.set_preprocessing("flag_use_pagn",             False)
    settings.set_preprocessing("galaxy_num",                N_GAL)
    settings.set_preprocessing("seed",                      9001)
    return settings

######## Tests ########
def test_last_line():
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as wkdir:
        fname = f"{wkdir}/some_file.txt"
        with open(fname, 'w') as F:
            F.write("Hello World!\n")
            F.write("Goodbye World!\n")
        assert "Hello World!" not in last_line(fname)
        assert "Goodbye World!" in last_line(fname)

def test_run_simulation():
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as wkdir:
        # Define log file
        fname_log = f"{wkdir}/testing.log"
        # Define some settings
        live = queue_worthy_settings()
        # Set the output directory
        live.set_preprocessing("output_dir", wkdir)
        live.set_preprocessing("overwrite_files", False)
        live.set_preprocessing("hdf5_snapshot_file", "test_simulation.hdf5")
        live.set_preprocessing("hdf5_snapshot_label", "testme")
        print("Running simulation")
        run_simulation(live, fname_log)
        print("Done!")

######## Main ########
def main():
    test_last_line()
    test_run_simulation()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
