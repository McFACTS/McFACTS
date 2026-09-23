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
#### Third Party ####
import tqdm
import numpy as np
#### Vera ####
from xdata import Database
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
FORBIDDEN_COLUMNS = [
    "seed",
    "verbose",
    "show_timeline_progress",
    "overwrite_files",
    "save_state",
    "save_each_timestep",
    "output_dir",
    "settings_snapshot",
    "cabinet_snapshot",
    "hdf5_snapshot_file",
    "hdf5_snapshot_mode",
    "hdf5_snapshot_gzip",
    "hdf5_snapshot_retries",
    "hdf5_snapshot_sleep",
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
    

def run_process(settings: SettingsManager):
    # Start logging
    fname_log = os.path.join(
        settings.output_dir,
        f"{settings.hdf5_snapshot_label}.log",
    )
    # Check for directory
    if not os.path.isdir(Path(fname_log).parent):
        Path(fname_log).parent.mkdir()
    # Open the logging file
    with open(fname_log, 'a') as LogFile:
        with redirect_stdout(LogFile), redirect_stderr(LogFile):
            try:
                out = run_simulation(settings, fname_log)
            except Exception:
                traceback.print_exc(file=LogFile)
                raise
    return out

def run_simulation(settings: SettingsManager, fname_log=None):
    # Define log function
    log_fn = ContextLogFunction(
        fname_log,
        prefix="",
        catch_stderr=True,
    )
    # Enforce scaling
    if settings.flag_use_scaling:
        setup_scaling(settings)

    # Create the IO handlers and save the current settings
    cabinet_snapshot_handler = settings.new_cabinet_snapshot()
    settings_snapshot_handler = settings.new_settings_snapshot()
    settings_snapshot_handler.save_settings(
        settings.output_dir,
        settings.hdf5_snapshot_file,
        settings,
        addr=settings_snapshot_handler.label,
    )

    # Load the AGN disk
    agn_disk = AGNDisk(settings)
    # Create a cabinet
    population_cabinet = FilingCabinet()
    ## Loop galaxies ##
    for galaxy_id in np.arange(settings.galaxy_num):
        # Define the galaxy object
        galaxy = Galaxy(
            seed = settings.seed - galaxy_id,
            runs_folder=settings.output_dir,
            galaxy_id=f"{galaxy_id:03d}",
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

        ## Manage simulation outputs ##
        population_cabinet.ignore_consistency_check("blackholes_merged")
        population_cabinet.ignore_consistency_check("blackholes_lvk")

        # Grab array names from settings manager
        prograde_array      = galaxy.settings.bh_prograde_array_name
        innerdisk_array     = galaxy.settings.bh_inner_disk_array_name
        inner_gw_only_array = galaxy.settings.bh_inner_gw_array_name
        bbh_merged_array    = galaxy.settings.bbh_merged_array_name
        bbh_lvk_array       = galaxy.settings.bbh_gw_array_name
        emri_merged_array   = galaxy.settings.emri_array_name
        bh_ejected_array    = galaxy.settings.bh_ejected_array_name

        # Sort objects into the final population cabinet containing results from all galaxies
        # (shamelessly copy-pasted from simulation.py)
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
    cabinet_snapshot_handler.save_cabinet(
        settings.output_dir,
        settings.hdf5_snapshot_file,
        population_cabinet,
    )

    return settings

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
            "cabinet_snapshot"          : "hdf5",
            "settings_snapshot"         : "hdf5",
            "hdf5_snapshot_retries"     : 2*self.n_sim,
            "hdf5_snapshot_sleep"       : 1.0,
        }

        # Settings
        self.settings = settings.copy(replace)

        #### Check columns ###
        for key in columns:
            if key in FORBIDDEN_COLUMNS:
                raise ValueError(
                    f"{key} was specified separately for each simulation. "
                    "If this was on purpose, remove it from FORBIDDEN_COLUMNS."
                )
        # Get global RNG
        self.rng = np.random.Generator(
            np.random.Philox(
                np.random.SeedSequence(settings.seed)
            )
        )
        # Label
        if "hdf5_snapshot_label" not in columns:
            columns["hdf5_snapshot_label"] = np.asarray(
                [f"sim_{i:3d}" for i in range(self.n_sim)]
            )

        ## Modify Columns ##
        columns["seed"] = \
            self.rng.bit_generator.random_raw(size=self.n_sim) >> 1

        # Set columns
        self.columns = columns

        #### Setup directoy ####
        wkdir = Path(settings.output_dir)
        wkdir.mkdir(exist_ok=True)
        fname = wkdir / settings.hdf5_snapshot_file
        if fname.is_file():
            if settings.overwrite_files:
                fname.unlink()
            else:
                raise ValueError(f"{fname} already exists!")

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
    db = Database(
        queue.settings.output_dir + "/" + queue.settings.hdf5_snapshot_file
    )
    # Identify finished runs
    finished = [val.hdf5_snapshot_label if val is not None else None for val in values]
    print(finished)
    for item in queue.columns["hdf5_snapshot_label"]:
        # Check if this run finished
        success = item in finished
        # Determine where the log file should be
        log = Path(queue.settings.output_dir) / f"{item}.log"
        if not log.is_file():
            log = None
        # Try to get the seed
        try:
            seed = db.attr_value(item, "seed")
        except:
            seed = None
        # Print information
        print(f"{item}: success={success}; seed={seed}; log={log}")
        # Print error
        if not success and log is not None:
            print(last_line(log))
    return
    

######## Execution ########
if __name__ == "__main__":
    main(**vars(arg()))
