import argparse
import cProfile

from mcfacts import fiducial_plots, simulation
from mcfacts.inputs.settings_manager import SettingsManager

# Source - https://stackoverflow.com/a/43357954
# Posted by Maxim, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-24, License - CC BY-SA 4.0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_settings_args(sub_parser: argparse.ArgumentParser):
    # Create a default instance of SettingsManager to get the default location of the settings file
    initial_settings = SettingsManager()

    # Create a specific flag for passing in a baked settings or ini file
    sub_parser.add_argument("-s", "--settings", "--fname-ini",
                            dest="settings_file",
                            help="Filename of settings file",
                            default=initial_settings.settings_file, type=str)

    sub_parser.add_argument("--profile", dest="enable_profiling", action="store_true")
    sub_parser.add_argument("--profile-out", dest="profiling_file", default="mcfacts.prof", type=str)

    # Using the supplied file, instantiate and populate a new SettingsManager
    inital_parse, _ = sub_parser.parse_known_args(["-s", initial_settings.settings_file])
    settings_file = inital_parse.settings_file

    # TODO: handle settings file loading.
    loaded_settings = SettingsManager({
        "verbose": False,
        "override_files": True,
        "save_state": True,
        "save_each_timestep": True,
        "flag_use_pagn": True,
        "stalling_separation": 0.0
    })

    # Parse through the loaded settings and create the corresponding arguments with the loaded settings as defaults
    for key, value in loaded_settings.settings_finals.items():
        if key in loaded_settings.static_settings:
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
    # Create instance of argument parser
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='subcommand')

    run_parser = sub_parsers.add_parser('run')
    seed_settings_args(run_parser)

    plot_parser = sub_parsers.add_parser('plot')
    seed_settings_args(plot_parser)

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


if __name__ == "__main__":
    main()