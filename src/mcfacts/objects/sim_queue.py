"""Manage a queue of related McFACTS simulations"""
######## Imports ########
#### Standard Library ####
import argparse
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
#### Third Party ####
import tqdm
import numpy as np
#### McFACTS ####
from mcfacts.inputs.settings_manager import SettingsProperty
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.inputs.settings_manager import DEFAULT_SETTINGS
from mcfacts.objects.snapshot import IniSnapshotHandler
from mcfacts.objects.snapshot import HDF5SnapshotHandler

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
def run_process(settings: SettingsManager):
    return settings.seed

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
        columns["seed"] = self.rng.bit_generator.random_raw(size=self.n_sim)

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

    def run(self, max_processes : int = 1):
        """Run all the simulations"""
        # Initialize a place to put a hypothetical return value
        return_values = np.empty((self.n_sim,), dtype=object)
        # TQDM
        with tqdm.tqdm(
                total = self.n_sim,
                desc = f"",
            ) as pbar:
            # Recite the incantation
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                # Submit processes
                future_to_simulation = {executor.submit(
                    run_process,
                    self.settings.copy({key:self.columns[key][i] for key in self.columns}),
                ): i for i in np.arange(self.n_sim)}
                # Catch processes
                for future in concurrent.futures.as_completed(future_to_simulation):
                    # Get the index
                    index = future_to_simulation[future]
                    # Assign the value
                    return_values[index] = future.result()
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
    parser.add_argument("--max-processes", default=1, type=int,
        help="How many concurrent processes to start?"
    )
    opts = parser.parse_args()
    return opts

######## Main ########
def main(
        settings : str,
        batch : str,
        max_processes : int,
    ):
    queue = SimulationQueue.from_batch_file(
        settings,
        batch,
    )
    values = queue.run(max_processes=max_processes)
    print(values)
    return
    

######## Execution ########
if __name__ == "__main__":
    main(**vars(arg()))
