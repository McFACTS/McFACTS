"""
mcfacts.py is main script that orchestrates the launching of simulations with provided settings.
"""

#### IMPORTS
import argparse
import cProfile
from pathlib import Path

from mcfacts import __version__
from mcfacts import fiducial_plots, simulation
from mcfacts.inputs import settings_manager
from mcfacts.inputs.scaling import setup_scaling
from mcfacts.inputs.settings_manager import SettingsManager, StaticSettingsProperty
from mcfacts.inputs.settings_manager import SETTING_TYPES
from mcfacts.objects.snapshot import TxtSnapshotHandler, IniSnapshotHandler
from mcfacts.objects.sim_queue import SimulationQueue
from mcfacts.utilities.unit_conversion import str2bool

#### SETUP
COMMANDS = ["run", "rp", "plot", "batch"]
CMD_MSG = "<" + " ".join(COMMANDS) + ">"

#### METHODS
def seed_settings_args(sub_parser: argparse.ArgumentParser):
    """
    This method compiles the settings for the simulation based on the defaults from SettingsManager and the
    overrides passed by the user.

    Parameters
    ----------
    sub_parser: argparse.ArgumentParser
        The sub parser representing the sub command to add the settings to.
    """
    # Create a specific flag for passing in a baked settings or ini file
    sub_parser.add_argument(
        "-s", "--settings", "--fname-ini", "--settings-file",
        dest="settings_file",
        help="Filename of settings file",
        default=None,
        type=str,
    )

    # Using the supplied file, instantiate and populate a new SettingsManager
    seed_settings_parse, unknown = sub_parser.parse_known_args()
    settings_file = seed_settings_parse.settings_file

    if settings_file is not None:
        suffix = Path(settings_file).suffix.lower()
        if suffix == ".ini":
            snapshot_handler = IniSnapshotHandler()
        elif suffix == ".txt":
            snapshot_handler = TxtSnapshotHandler()
        else:
            raise ValueError(
                f"Unsupported settings file format '{suffix}', "
                "expected .ini or .txt"
            )

        # Load settings
        loaded_settings = snapshot_handler.load_settings(
            str(Path(settings_file).parent),
            Path(settings_file).stem,
        )
    else:
        loaded_settings = SettingsManager()

    # These are things the user should not change at runtime
    static_settings = [prop.name for prop in settings_manager.DEFAULT_SETTINGS if isinstance(prop, StaticSettingsProperty)]

    # Parse through the loaded settings and create the
    #  corresponding arguments with the loaded settings as defaults
    for key, value in loaded_settings.settings_finals.items():
        # Skip things that should not be set at runtime or are irrelevant
        if key in static_settings:
            continue
        if key == "settings_file":
            continue

        # Skip arguments which were already defined
        if any(arg.dest == key for arg in sub_parser._actions):
            continue

        # Handle these explicitly
        if key == "overwrite_files":
            sub_parser.add_argument(
                f"--overwrite_files", "-o",
                default=value,
                type=str2bool,
                metavar=key,
                dest=key,
            )
            continue

        # Handle other settings
        if type(value) is bool:
            sub_parser.add_argument(
                f"--{key}",
                default=value,
                type=str2bool,
                metavar=key,
                dest=key,
            )
        else:
            sub_parser.add_argument(
                f"--{key}",
                default=value,
                type=type(value),
                metavar=key,
                dest=key,
           )


def run_simulation(settings, profiling=False, filename=None):
    """Avoid code duplication"""
    # Hotwire settings for scaling run
    if settings.flag_use_scaling:
        # This is done outside of simulation.py so that the live settings
        #   are recorded in the log.
        setup_scaling(settings)
    # Run with or without profiling
    if profiling:
        prof = cProfile.Profile()
        prof.runcall(simulation.main, settings)
        prof.dump_stats(filename)
    else:
        simulation.main(settings)

def main():
    """
    Main method that interprets user input and runs the simulation based on the provided
    sub command and settings.
    """
    ## Pre-process args ##
    # Create the preparser, and do not catch "help" here
    # This has to be an entirely separate parser if we want to catch subcommand
    pre_parser = argparse.ArgumentParser(add_help=False)
    # Add subcommand
    # nargs='?' makes it not crash when subcommand is not given.
    pre_parser.add_argument("subcommand", nargs='?')
    # GNU Coding Standards version syntax 
    pre_parser.add_argument(
        "--version", "-V",
        dest="print_version",
        action='store_true',
    )
    pre_parse_opts, _ = pre_parser.parse_known_args()

    # If the version flag was passed, print the version and quit
    if pre_parse_opts.print_version:
        print(f"McFACTS Version: {__version__}")
        return

    # Identify subcommand
    subcommand = pre_parse_opts.subcommand
    # If the subcommand is not present, this will be None.
    # If present, we can now read it.
    # We don't want to crash yet; we just don't want to load the same 
    #  arguments many times

    ## Create the real parser ##
    # Create instance of argument parser
    parser = argparse.ArgumentParser()
    
    ## Here is where we define subparsers
    sub_parsers = parser.add_subparsers(dest='subcommand',metavar=str(CMD_MSG))

    # Run command
    run_parser = sub_parsers.add_parser('run')
    # seed_settings_args is expensive if we don't use it.
    if subcommand == "run":
        seed_settings_args(run_parser)
        run_parser.add_argument(
            "--profile", dest="enable_profiling", action="store_true"
        )
        run_parser.add_argument(
            "--profile-out", dest="profiling_file", default="mcfacts.prof", type=str
        )

    # Plot command
    plot_parser = sub_parsers.add_parser('plot')
    # seed_settings_args is expensive if we don't use it.
    if subcommand == "plot": seed_settings_args(plot_parser)

    # Run and Plot (rp) command
    rp_parser = sub_parsers.add_parser('rp')
    # seed_settings_args is expensive if we don't use it.
    if subcommand == "rp":
        seed_settings_args(rp_parser)
        rp_parser.add_argument(
            "--profile",
            dest="enable_profiling",
            action="store_true",
        )
        rp_parser.add_argument(
            "--profile-out",
            dest="profiling_file",
            default="mcfacts.prof",
            type=str,
        )

    # Batch command
    batch_parser = sub_parsers.add_parser('batch')
    # seed_settings_args is expensive if we don't use it.
    if subcommand == "batch": seed_settings_args(batch_parser)

    # Additional arguments for the batch command
    batch_parser.add_argument(
        "--batch-file",
        type=str,
        required=True,
        help="Location of a SimulationQueue batch file.",
    )
    batch_parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel processes to launch at once.",
    )

    # Parse arguments (for real this time guys!)
    inputs = parser.parse_args()

    # Catch missing subcommand
    if inputs.subcommand is None:
        parser.print_help()
        return

    # Get just the settings which belong in a settings file:
    overrides = {key : value for key, value in vars(inputs).items() if key in SETTING_TYPES}
    # Create a settings manager including the file defaults and any CLI overrides.
    settings = SettingsManager(settings_overrides=overrides)

    if inputs.subcommand == "run":
        run_simulation(settings,inputs.enable_profiling,inputs.profiling_file)

    if inputs.subcommand == "plot":
        fiducial_plots.main(settings)
        return

    if inputs.subcommand == "rp":
        run_simulation(settings,inputs.enable_profiling,inputs.profiling_file)
        fiducial_plots.main(settings)

    if inputs.subcommand == "batch":
        # Define the queue
        sim_queue = SimulationQueue.from_settings(
            settings,
            inputs.batch_file,
        )
        # Run the queue
        sim_queue.run(max_workers=inputs.max_workers)
        # Reprot the status
        sim_queue.report()

if __name__ == "__main__":
    main()
