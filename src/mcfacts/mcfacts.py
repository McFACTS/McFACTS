import argparse

from mcfacts import fiducial_plots, simulation
from mcfacts.inputs.settings_manager import SettingsManager


def main():
    # Create instance of argument parser
    parser = argparse.ArgumentParser()

    # Create a default instance of SettingsManager to get the default location of the settings file
    inital_settings = SettingsManager()

    # Create a specific flag for passing in a baked settings or ini file
    parser.add_argument("-s", "--settings", "--fname-ini",
                        dest="settings_file",
                        help="Filename of settings file",
                        default=inital_settings.settings_file, type=str)

    # Using the supplied file, intanciate and populate a new SettingsManager
    inital_parse, _ = parser.parse_known_args(["-s", inital_settings.settings_file])
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

        for action in parser._actions:
            option_strings = action.option_strings

            for option in option_strings:
                options.append(option)

        alias = f"-{str(key)[0]}"

        if alias in options:
            parser.add_argument(f"--{key}",
                                default=value, type=type(value), metavar=key)
        else:
            parser.add_argument(alias, f"--{key}",
                                default=value, type=type(value), metavar=key)

    inputs = parser.parse_args()

    # With the parsed arguments, create a final settings manager including the file defaults and any CLI overrides.
    settings = SettingsManager(vars(inputs))

    simulation.main(settings)

    fiducial_plots.main(settings)


if __name__ == "__main__":
    main()