"""Manage a queue of related McFACTS simulations"""
######## Imports ########
#### Standard Library ####
import argparse
import concurrent.futures
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import warnings
#### Third Party ####
import tqdm
import numpy as np
#### Vera ####
from xdata import Database, Connection
#### McFACTS ####
from mcfacts.inputs import setup_scaling
from mcfacts.inputs.settings_manager import SettingsProperty
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.inputs.settings_manager import DEFAULT_SETTINGS
from mcfacts.objects.agn_object_array import FilingCabinet
from mcfacts.objects.disk import AGNDisk
from mcfacts.objects.log import ContextLogFunction
from mcfacts.objects.galaxy import Galaxy
from mcfacts.objects.populators import SingleBlackHolePopulator, SingleStarPopulator
from mcfacts.objects.snapshot import IniSnapshotHandler
from mcfacts.objects.snapshot import HDF5SnapshotHandler
from mcfacts.simulation import run_galaxy

######## Setup ########
COLUMNS_FORBIDDEN = [
    "show_timeline_progress",
    "overwrite_files",
    "save_state",
    "output_dir",
    "settings_snapshot",
    "cabinet_snapshot",
    "hdf5_snapshot_file",
    "hdf5_snapshot_mode",
    "hdf5_snapshot_retries",
    "hdf5_snapshot_sleep",
]

COLUMNS_SCALING = [
    "disk_radius_outer",
    "stellar_mass",
    "smbh_mass",
    "nsc_mass",
    "inner_disk_outer_radius",
    "disk_radius_trap",
    "disk_radius_capture_outer",
    "capture_time_yr",
]

######## Functions ########

def last_line(fname):
    """Quickly read the last line of a file

    Thanks, StackOverflow!
    https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
    """
    with open(fname, 'rb') as F:
        try: # Catch OSError in case of a one line file
            F.seek(-2, os.SEEK_END)
            while F.read(1) != b'\n':
                F.seek(-2, os.SEEK_CUR)
        except OSError:
            F.seek(0)
        last_line = F.readline().decode().rstrip('\n')
    return last_line
    

def run_simulation(settings: SettingsManager, fname_log):
    """Populate a galaxy and run an active timeline

    Parameters
    ----------
    settings : SettingsManager
        The SettingsManager object for this run in particular
    fname_log : path_like
        Path to save outputs to

    Returns
    -------
    settings : SettingsManager
        I needed something besides None to distinguish successful runs
        from early exits, and this seemed like a useful one, anyway.
    """
    # Define log function
    log_fn = ContextLogFunction(
        fname_log,
        prefix="",
        catch_stderr=True,
    )

    ## Check for persistence ##
    # Check if this already ran
    db = Database(
        os.path.join(
            settings.output_dir,
            settings.hdf5_snapshot_file,
        ),
        retries=settings.hdf5_snapshot_retries,
        sleep=settings.hdf5_snapshot_sleep,
    )
    label_group_exists = db.exists(settings.hdf5_snapshot_label)
    if label_group_exists:
        # We are rerunning a simulation with the same name
        # setup_scaling is expensive (Have to make and throw away a disk
        # interpolator object). We don't have to do it again.
        prev_handler = HDF5SnapshotHandler()
        try:
            prev_settings = prev_handler.load_settings(
                settings.output_dir,
                settings.hdf5_snapshot_file,
                addr=settings.hdf5_snapshot_label,
            )
        except KeyError as exc:
            prev_settings = None
    else:
        prev_settings = None

    # Check if the settings are the same as the last run
    match = True
    if prev_settings is None:
        match = False
        print(f"match failure; case 1")
    else:
        for key, value in prev_settings.settings_finals.items():
            # We might have re-seeded the galaxy
            if key == "seed":
                continue
            # Scaling will change some clumns
            if key in COLUMNS_SCALING:
                continue
            if value != getattr(settings, key):
                match = False
                print(f"match failure; case 2; key: {key}")

    # Collision
    if not match and label_group_exists:
        # We have incompatible attributes with the provided settings
        # We need to crash or overwrite them
        # The way to reliably overwrite them is to delete the group
        if settings.overwrite_files:
            db.remove(settings.hdf5_snapshot_label)
            label_group_exists = False
        else:
            raise ValueError(
                f"Failed to resume a failed run!"
            )

    ## Apply scaling ##
    # Load the scaling columns
    if match and settings.flag_use_scaling:
        settings = settings.copy(
            {key:getattr(prev_settings, key) for key in COLUMNS_SCALING},
        )
    # Rerun setup_scaling manually
    else:
        setup_scaling(settings)

    # Save the settings if they're not saved
    if not label_group_exists:
        settings_snapshot_handler = settings.new_settings_snapshot()
        settings_snapshot_handler.save_settings(
            settings.output_dir,
            settings.hdf5_snapshot_file,
            settings,
            addr=settings_snapshot_handler.label,
        )
    # Create the IO handlers and save the current settings
    cabinet_snapshot_handler = settings.new_cabinet_snapshot()

    # Load the AGN disk
    agn_disk = AGNDisk(settings)
    # Create a cabinet
    population_cabinet = FilingCabinet()

    ## Loop galaxies to run them ##
    for galaxy_id in np.arange(settings.galaxy_num):
        # Seed is offset; trick for restarting runs
        galaxy_seed = settings.seed - galaxy_id
        # Tag for labelling galaxy
        galaxy_tag = f"{galaxy_id:03d}"
        # Address of galaxy
        galaxy_addr = "/".join([
            settings.hdf5_snapshot_label,
            f"gal{galaxy_tag}",
        ])
        # Check if the galaxy was already run
        galaxy_prev_exists = db.exists(galaxy_addr)
        if galaxy_prev_exists:
            galaxy_prev_seed = int(db.attr_value(galaxy_addr, 'seed'))
        # Check for a collision
        if galaxy_prev_exists and galaxy_prev_seed != galaxy_seed:
            if settings.overwrite_files:
                print(f"Removing failed galaxy: {galaxy_addr} (seed:{galaxy_seed})")
                db.remove(galaxy_addr)
                galaxy_prev_exists = False
        # Check if we've literally run this exact simulation before
        if galaxy_prev_exists:
            print(f"We have run {galaxy_addr} with seed {galaxy_seed} before! No need to rerun")
            continue
        # Create the galaxy group and assign it the seed attribute
        else:
            db.create_group(galaxy_addr)
            db.attr_set(galaxy_addr, "seed", galaxy_seed)
            db.attr_set(galaxy_addr, "success", False)

        # Define the galaxy object
        galaxy = Galaxy(
            seed = galaxy_seed,
            runs_folder=settings.output_dir,
            galaxy_id=galaxy_tag,
            settings=settings,
        )
        galaxy.parent_log_func = log_fn.spawn(f"(ID:{galaxy_id:03d}) ")

        ## Populate galaxy with a new population ##
        single_bh_populator = SingleBlackHolePopulator(settings=settings)
        single_star_populator = SingleStarPopulator(settings=settings)
        galaxy.populate(
            [single_bh_populator, single_star_populator],
            agn_disk,
            strict_fill=False,
        )

        ## Run the galaxy ##
        run_galaxy(settings, galaxy, agn_disk=agn_disk)
        db.attr_set(galaxy_addr, "success", True)

    ## Loop galaxies to append population ##
    # Open the connection in readonly mode
    with Connection(
            os.path.join(
                settings.output_dir,
                settings.hdf5_snapshot_file,
            ),
            mode='r',
            retries=settings.hdf5_snapshot_retries,
            sleep=settings.hdf5_snapshot_sleep,
    ) as conn:
        # initialize dtypes and arrays
        population_dtypes = {}
        population_arrays = {}
        ## Loop galaxies ##
        for galaxy_id in np.arange(settings.galaxy_num):
            # Tag for labelling galaxy
            galaxy_tag = f"{galaxy_id:03d}"
            # Address of galaxy
            galaxy_addr = "/".join([
                settings.hdf5_snapshot_label,
                f"gal{galaxy_tag}",
            ])
            # Check address is in file
            if not galaxy_addr in conn.file:
                raise KeyError(f"{galaxy_addr} not found!")
            if "success" not in conn.file[galaxy_addr].attrs:
                raise KeyError(f"{galaxy_addr} not initialized properly!")
            _seed = conn.file[galaxy_addr].attrs["seed"]
            _success = bool(conn.file[galaxy_addr].attrs["success"])
            if not _success:
                raise RuntimeError(
                    f"{galaxy_addr} with seed "
                    f"{conn.file[galaxy_addr].attrs['seed']} "
                    "has failed to land correctly. "
                    "This may cost them the race!"
                    # This is a sonic riders reference.
                    # I can take it out if it's distracting.
                )

            # Okay, we found it. What now?
            states = [key for key in conn.file[galaxy_addr] if len(key.split('_')) == 2]
            states.sort()
            state_addr = f"{galaxy_addr}/{states[-1]}"
            assert state_addr in conn.file
            # Loop through keys (E.g. blackholes_merged)
            for array_name in conn.file[state_addr]:
                # Get array address
                array_addr = f"{state_addr}/{array_name}/compound"
                if array_addr not in conn.file:
                    continue
                # Load the array
                tmp = conn.file[array_addr][...]
                # Get the datatype
                population_dtypes[array_name] = tmp.dtype
                # Append it
                if not array_name in population_arrays:
                    population_arrays[array_name] = []
                population_arrays[array_name].append(tmp)

    # Open the connection with writing priveleges
    with Connection(
            os.path.join(
                settings.output_dir,
                settings.hdf5_snapshot_file,
            ),
            mode='r+',
            retries=settings.hdf5_snapshot_retries,
            sleep=settings.hdf5_snapshot_sleep,
    ) as conn:
        # initialize dtypes and arrays
        # Save arrays
        pop_addr = f"{settings.hdf5_snapshot_label}/population"
        if pop_addr not in conn.file:
            conn.file.create_group(pop_addr)
        # Loop array names
        for array_name in population_dtypes:
            # Create intermediate array group
            tmp_addr = f"{pop_addr}/{array_name}"
            if tmp_addr not in conn.file:
                conn.file.create_group(tmp_addr)
            # Get array address
            array_addr = f"{pop_addr}/{array_name}/compound"
            if array_addr in conn.file:
                # Overwrite population
                if settings.overwrite_files:
                    # Careful here
                    del conn.file[array_addr]
                else:
                    warnings.warn(f"Skipping {array_name}; already exists!")
                    continue
            # Concatenate arrays
            array_value = np.concatenate(
                population_arrays[array_name],
            )
            # Write population output
            conn.file.create_dataset(
                array_addr,
                array_value.shape,
                dtype=population_dtypes[array_name],
                data=array_value,
                compression=settings.hdf5_snapshot_compression,
            )
                
    return settings

def run_process(settings: SettingsManager):
    """Wrapper for simulation calls

    Redirects outputs to log file

    Parameters
    ----------
    settings : SettingsManager
        The SettingsManager object for this run in particular
    """
    # Start logging
    fname_log = os.path.join(
        settings.output_dir,
        f"{settings.hdf5_snapshot_label}.log",
    )
    # Check if the file already exists
    if os.path.isfile(fname_log):
        os.remove(fname_log)
    # Check for directory
    if not os.path.isdir(Path(fname_log).parent):
        Path(fname_log).parent.mkdir()
    # Open the logging file
    with open(fname_log, 'a') as LogFile:
        with redirect_stdout(LogFile), redirect_stderr(LogFile):
            try:
                out = run_simulation(settings, fname_log)
                # This is how COSMIC simulations end, so this is a short homage
                print("All done friend!")
            except Exception:
                traceback.print_exc(file=LogFile)
                raise
    return out

######## Objects ########
class SimulationQueue(object):
    """A object which runs a set of McFACTS simulations"""

    def __init__(
            self,
            settings : SettingsManager,
            columns : dict,
        ):
        """Construct the SimulationQueue object

        Parameters
        ----------
        settings : SettingsManager
            The master SettingsManager that subsequent settings are copied from
        columns : dict
            A dictionary to override settings differently for each run
        """
        # Get number of sims
        self.n_sim = columns[list(columns.keys())[0]].shape[0]
        print(f"Preparing {self.n_sim} simulations!")

        ### Check settings ###
        ## Hard nos ##
        ## Autocorrect ##
        replace = {
            "show_timeline_progress"    : True,
            "save_state"                : True,
            "cabinet_snapshot"          : "hdf5",
            "settings_snapshot"         : "hdf5",
            "hdf5_snapshot_retries"     : 2*self.n_sim,
            "hdf5_snapshot_sleep"       : 1.0,
            "hdf5_snapshot_mode"        : "compound",
        }

        # Settings
        self.settings = settings.copy(replace)

        #### Check columns ###
        for key in columns:
            if key in COLUMNS_FORBIDDEN:
                raise ValueError(
                    f"{key} was specified separately for each simulation. "
                    "If this was on purpose, remove it from COLUMNS_FORBIDDEN."
                )
        # Get global RNG
        self.rng = np.random.Generator(
            np.random.Philox(
                np.random.SeedSequence(settings.seed)
            )
        )
        ## Modify Columns ##
        if "seed" not in columns:
            columns["seed"] = \
                self.rng.bit_generator.random_raw(size=self.n_sim) >> 1

        # Label
        if "hdf5_snapshot_label" not in columns:
            columns["hdf5_snapshot_label"] = np.asarray(
                [f"sim_{i:3d}" for i in range(self.n_sim)]
            )

        # Set columns
        self.columns = columns

        #### Setup directoy ####
        wkdir = Path(settings.output_dir)
        wkdir.mkdir(exist_ok=True)
        fname_db = wkdir / settings.hdf5_snapshot_file
        self.wkdir = wkdir
        self.fname_db = fname_db
        self.db = Database(fname_db)

    @classmethod
    def from_batch_file(cls, fname_settings, fname_batch):
        """Generate a SimulationQueue from a saved file

        Parameters
        ----------
        filename : path_like
            The location of the file
        """
        # Get path of settings
        settings = Path(fname_settings)
        if not settings.is_file():
            raise ValueError(f"No such file or directory: {settings}")
        # Get path of batch file
        batch = Path(fname_batch)
        if not batch.is_file():
            raise ValueError(f"No such file or directory: {batch}")
        # Load SettingsManager object
        ini_handler = IniSnapshotHandler()
        settings = ini_handler.load_settings(str(settings.parent), str(settings.stem))
        # Load header
        with open(batch, 'r') as F:
            header_line = F.readline().lstrip("#")
        # Get the fields
        header_fields = header_line.split()
        # Get header_dtype
        header_dtype = [(key, type(getattr(settings,key))) for key in header_fields]
        # Load batch
        batch_data = np.loadtxt(batch, dtype=str)
        batch_dict = {}
        for i, key in enumerate(header_fields):
            batch_dict[key] = batch_data[:,i].astype(type(getattr(settings,key)))

        return SimulationQueue(
            settings,
            batch_dict,
        )

    def run(self, max_workers : int = 1):
        """Run all the simulations"""
        # Initialize a place to put a hypothetical return value
        return_values = np.full((self.n_sim,), None, dtype=object)
        # TQDM
        with tqdm.tqdm(
                total = self.n_sim,
                desc = f"",
            ) as pbar:
            # Recite the incantation
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit processes
                future_to_simulation = {executor.submit(
                    run_process,
                    self.settings.copy({key:self.columns[key][i] for key in self.columns}),
                ): i for i in np.arange(self.n_sim)}
                # Catch processes
                for future in concurrent.futures.as_completed(future_to_simulation):
                    # Get the index
                    index = future_to_simulation[future]
                    # Check if the run was successful
                    try:
                        # Assign the value
                        return_values[index] = future.result()
                    except:
                        # Failed sims will return None, so we don't need
                        #  to flag them again somehow.
                        pass
                    finally:
                        # Update tqdm
                        pbar.update(1)
        return return_values

    def log_of_run(self, run):
        # Determine where the log file should be
        log = Path(self.settings.output_dir) / f"{run}.log"
        # Check if the log file exists
        if not log.is_file():
            log = None
        return log

    def did_run_fail(self, run):
        # Check if the run is in the database
        if not self.db.exists(run, kind="group"):
            return 1
        # Check if the run has a population group
        pop_addr = f"{run}/population"
        if not self.db.exists(pop_addr, kind="group"):
            return 2
        # Check if "blackholes_merged" is in the population group
        merged_addr = f"{pop_addr}/blackholes_merged"
        if not self.db.exists(merged_addr, kind="group"):
            return 0
        # Check if the compound array exists
        compound_addr = f"{merged_addr}/compound"
        if not self.db.exists(compound_addr):
            return 5
        # Run succeeded!
        return 0

    #def attrs_of_run(self, run):

    def report(self):
        """Report on the status of runs"""
        # Get the filename for the snapshot file
        fname_db = os.path.join(self.wkdir, self.settings.hdf5_snapshot_file)
        if not os.path.isfile(fname_db):
            print(f"No such file named '{fname_db}' exists yet.")
            return
        # Identify expected runs
        expected = self.columns["hdf5_snapshot_label"]
        # Open the database
        db = Database(fname_db)
        for run in expected:
            # Determine where the log file should be
            log = self.log_of_run(run)
            # Check if the run succeeded
            ret_code = self.did_run_fail(run)
            success = ret_code == 0
            # Check if the attrs are available
            if db.exists(run):
                attrs = db.attr_dict(run)
            else:
                attrs = {}
            print(" ".join([
                run,
                f"return_code={ret_code}",
                f"seed={attrs['seed'] if 'seed' in attrs else None}",
                f"log={log if log is not None else None}",
            ]))
            # Report errors
            if not success:
                # Get the last line
                if log is None:
                    last = None
                else:
                    last = last_line(log)
                print(last)

    def failed_galaxy_ids(self):
        """Return the galaxy_ids of each failed run"""
        

######## Argparse ########
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True,
        help="Location of a [settings.ini] file")
    parser.add_argument("--batch", required=True,
        help="Location of [batch.txt] file")
    parser.add_argument("--max-workers", default=1, type=int,
        help="How many concurrent processes to start?"
    )
    opts = parser.parse_args()
    return opts

######## Main ########
def main(
        settings : str,
        batch : str,
        max_workers : int,
    ):
    queue = SimulationQueue.from_batch_file(
        settings,
        batch,
    )
    tic = time.perf_counter()
    values = queue.run(max_workers=max_workers)
    toc = time.perf_counter()
    print(f"SimulationQueue ran in {toc-tic:.6f} seconds!")
    queue.report()
    return
    

######## Execution ########
if __name__ == "__main__":
    main(**vars(arg()))
