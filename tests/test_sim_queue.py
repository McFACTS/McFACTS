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

BATCH_TEXT = """#hdf5_snapshot_label stellar_mass scale_nsc_mass
batch/early_010 1.77827941e+10 neumayer-early
batch/late_010 1.77827941e+10 neumayer-late
"""

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
    settings.set_preprocessing("flag_use_pagn",             True)
    settings.set_preprocessing("galaxy_num",                N_GAL)
    settings.set_preprocessing("seed",                      9001)
    # Unnecessary, but helpful
    settings.set_preprocessing("verbose",                   False)
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
        # Run the simulation once
        run_simulation(live, fname_log)
        # Check that there is a datafile
        assert os.path.isfile(os.path.join(wkdir, live.hdf5_snapshot_file))
        # Open a database pointer
        db = Database(os.path.join(wkdir, live.hdf5_snapshot_file))
        # Check that the run exists
        assert db.exists(live.hdf5_snapshot_label, kind='group')
        # Update the database pointer
        db = Database(
            os.path.join(wkdir, live.hdf5_snapshot_file),
            group=live.hdf5_snapshot_label,
        )
        # Get the attributes
        attr_dict = db.attr_dict(".")
        # Check the settings
        for key, value in attr_dict.items():
            # Check that the key exists
            assert key in live.settings_finals
            # Check that the settings are correct
            assert SETTING_TYPES[key](value) == getattr(live, key)
        # There should be a population group
        subgroups = db.list_items()
        assert "population" in subgroups
        # There should also be galaxy groups
        galaxy_groups = {}
        for i in np.arange(N_GAL):
            found = False
            for sg in subgroups:
                if not sg.startswith("gal"):
                    continue
                if int(sg.lstrip("gal")) == i:
                    found = True
                    galaxy_groups[i] = sg
            assert found
        # Loop the galaxy groups
        for galaxy_id, gal_label in galaxy_groups.items():
            # Get the galaxy attrs
            galaxy_attrs = db.attr_dict(gal_label)
            # Check success
            assert "success" in galaxy_attrs
            assert galaxy_attrs["success"]
            # Look at what's in the subgroup
            states = db.list_items(gal_label)
            assert len(states) > 0

        ## Check that rerunning a completed run successfully does nothing.
        # Run the simulation again, but catch stdout and stderr
        with open(fname_log, 'a') as LogFile:
          with redirect_stdout(LogFile), redirect_stderr(LogFile):
            try:
                run_simulation(live, fname_log)
            except Exception:
                traceback.print_exc(file=LogFile)
                raise
        chk = f"No need to rerun"
        found = False
        with open(fname_log, 'r') as LogFile:
            for line in LogFile:
                if chk in line:
                    found = True
                    break
        assert found

        ## Sabotage galaxy success
        # Loop the galaxy groups
        for galaxy_id, gal_label in galaxy_groups.items():
            # Tell them they failed
            db.attr_set(
                gal_label,
                "success",
                type(galaxy_attrs["success"])(False),
            )
        # Clear the logfile
        os.remove(fname_log)
        assert not os.path.isfile(fname_log)

        # Run the simulation again
        with open(fname_log, 'a') as LogFile:
          with redirect_stdout(LogFile), redirect_stderr(LogFile):
            try:
                run_simulation(live, fname_log)
            except Exception:
                traceback.print_exc(file=LogFile)
                raise
        # Check that it didn't skip anything
        chk = f"No need to rerun"
        found = False
        with open(fname_log, 'r') as LogFile:
            for line in LogFile:
                if chk in line:
                    found = True
                    break
        assert not found

        # Check that the run succeeded this time
        for galaxy_id, gal_label in galaxy_groups.items():
            galaxy_attrs = db.attr_dict(gal_label)
            assert "success" in galaxy_attrs
            assert galaxy_attrs["success"]

        ## Sabotage galaxy seeds
        for galaxy_id, gal_label in galaxy_groups.items():
            # Lie about their seed
            db.attr_set(gal_label, "seed", galaxy_id)

        # Clear the logfile
        os.remove(fname_log)
        assert not os.path.isfile(fname_log)

        # Run the simulation again
        with open(fname_log, 'a') as LogFile:
          with redirect_stdout(LogFile), redirect_stderr(LogFile):
            try:
                run_simulation(live, fname_log)
            except Exception:
                traceback.print_exc(file=LogFile)
                raise
        # Check that it did skip things
        chk = f"No need to rerun"
        found = False
        with open(fname_log, 'r') as LogFile:
            for line in LogFile:
                if chk in line:
                    found = True
                    break
        assert found

        # Check that the run succeeded this time and has the seeds we assigned
        for galaxy_id, gal_label in galaxy_groups.items():
            galaxy_attrs = db.attr_dict(gal_label)
            assert "success" in galaxy_attrs
            assert galaxy_attrs["success"]
            assert "seed" in galaxy_attrs
            assert galaxy_attrs["seed"] == galaxy_id

        ## Bulldoze the seeds
        live.set_preprocessing("overwrite_files", True)
        # Clear the logfile
        os.remove(fname_log)
        assert not os.path.isfile(fname_log)

        # Run the simulation again
        with open(fname_log, 'a') as LogFile:
          with redirect_stdout(LogFile), redirect_stderr(LogFile):
            try:
                run_simulation(live, fname_log)
            except Exception:
                traceback.print_exc(file=LogFile)
                raise
        # Check that it did not skip things
        chk = f"No need to rerun"
        found = False
        with open(fname_log, 'r') as LogFile:
            for line in LogFile:
                if chk in line:
                    found = True
                    break
        assert not found

        # Check that the run succeeded this time and has the correct seeds
        for galaxy_id, gal_label in galaxy_groups.items():
            galaxy_attrs = db.attr_dict(gal_label)
            assert "success" in galaxy_attrs
            assert galaxy_attrs["success"]
            assert "seed" in galaxy_attrs
            assert galaxy_attrs["seed"] != galaxy_id

def test_batch():
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
        # Write the batch file
        batch_file = f"{wkdir}/batch.tsv"
        with open(batch_file, 'w') as F:
            print(BATCH_TEXT, file=F)
        # Make the queue object
        queue = SimulationQueue.from_settings(live, batch_file)
        # Run the queue, not parallel
        queue.run(max_workers=1)
        # Check it parallel (basically no-op)
        queue.run(max_workers=2)

######## Main ########
def main():
    test_last_line()
    test_run_simulation()
    test_batch()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
