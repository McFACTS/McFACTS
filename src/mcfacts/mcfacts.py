"""
mcfacts.py is main script that orchestrates the launching of simulations with provided settings.
"""

#### IMPORTS
import argparse
import cProfile
from pathlib import Path

from mcfacts import fiducial_plots, simulation
from mcfacts.inputs import settings_manager
from mcfacts.inputs.settings_manager import SettingsManager, StaticSettingsProperty
from mcfacts.objects.snapshot import TxtSnapshotHandler, IniSnapshotHandler
from mcfacts.utilities.unit_conversion import str2bool


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
    # Create a default instance of SettingsManager to get the default location of the settings file
    initial_settings = SettingsManager()

    # Create a specific flag for passing in a baked settings or ini file
    sub_parser.add_argument("-s", "--settings", "--fname-ini", "--settings-file",
                            dest="settings_file",
                            help="Filename of settings file",
                            default=None, type=str)

    sub_parser.add_argument("--profile", dest="enable_profiling", action="store_true")
    sub_parser.add_argument("--profile-out", dest="profiling_file", default="mcfacts.prof", type=str)

    # Using the supplied file, instantiate and populate a new SettingsManager
    initial_parse, unknown = sub_parser.parse_known_args()
    settings_file = initial_parse.settings_file

    print(settings_file)

    if settings_file is not None:
        suffix = Path(settings_file).suffix.lower()
        if suffix == ".ini":
            snapshot_handler = IniSnapshotHandler()
        elif suffix == ".txt":
            snapshot_handler = TxtSnapshotHandler()
        else:
            raise ValueError(f"Unsupported settings file format '{suffix}', expected .ini or .txt")

        loaded_settings = snapshot_handler.load_settings(str(Path(settings_file).parent), Path(settings_file).stem)
    else:
        loaded_settings = initial_settings

    static_settings = [prop.name for prop in settings_manager.DEFAULT_SETTINGS if isinstance(prop, StaticSettingsProperty)]

    # Parse through the loaded settings and create the corresponding arguments with the loaded settings as defaults
    for key, value in loaded_settings.settings_finals.items():
        if key in static_settings:
            continue

        if key is "settings_file":
            continue

        options = []

        for action in sub_parser._actions:
            option_strings = action.option_strings

            for option in option_strings:
                options.append(option)

        alias = f"-{str(key)[0]}"

        if type(value) is bool:
            if alias in options:
                sub_parser.add_argument(f"--{key}",
                                    default=value, type=str2bool, metavar=key, dest=key)
            else:
                sub_parser.add_argument(alias, f"--{key}",
                                    default=value, type=str2bool, metavar=key, dest=key)
        else:
            if alias in options:
                sub_parser.add_argument(f"--{key}",
                                    default=value, type=type(value), metavar=key, dest=key)
            else:
                sub_parser.add_argument(alias, f"--{key}",
                                    default=value, type=type(value), metavar=key, dest=key)


def main():
    """
    Main method that interprets user input and runs the simulation based on the provided
    sub command and settings.
    """
    # Create instance of argument parser
    parser = argparse.ArgumentParser(allow_abbrev=False)

    sub_parsers = parser.add_subparsers(dest='subcommand')

    run_parser = sub_parsers.add_parser('run')
    seed_settings_args(run_parser)

    plot_parser = sub_parsers.add_parser('plot')
    seed_settings_args(plot_parser)

    rp_parser = sub_parsers.add_parser('rp')
    seed_settings_args(rp_parser)

    inputs = parser.parse_args()

    if inputs.subcommand is None:
        parser.print_help()
        return

    # With the parsed arguments, create a final settings manager including the file defaults and any CLI overrides.
    settings = SettingsManager(vars(inputs))

    if inputs.subcommand == "run":
        if inputs.enable_profiling:
            cProfile.runctx('simulation.main(settings)', globals(), locals(), filename=inputs.profiling_file)
        else:
            simulation.main(settings)
        return

    if inputs.subcommand == "plot":
        fiducial_plots.main(settings)
        return

    if inputs.subcommand == "rp":
        if inputs.enable_profiling:
            cProfile.runctx('simulation.main(settings)', globals(), locals(), filename=inputs.profiling_file)
        else:
            simulation.main(settings)

        fiducial_plots.main(settings)
        return


if __name__ == "__main__":
    main()