#!/usr/bin/env python3
"""Test the mcfacts.mcfacts.seed_settings_args function"""
######## Imports ########
#### Standard ####
import argparse

#### Third Party ####
import numpy as np

#### Local ####
from mcfacts.mcfacts import seed_settings_args
from mcfacts.inputs.settings_manager import SettingsManager, DEFAULT_SETTINGS
from mcfacts.inputs.settings_manager import SETTING_TYPES
from mcfacts.inputs.settings_manager import StaticSettingsProperty
from mcfacts.utilities.unit_conversion import str2bool

######## Setup ########
# These are things the user should not change at runtime
STATIC_SETTINGS = [prop.name for prop in DEFAULT_SETTINGS if isinstance(prop, StaticSettingsProperty)]

def setup_subparser():
    # Get a parser
    parser = argparse.ArgumentParser()
    # Get an object for managing subparsers
    sub_parsers = parser.add_subparsers(dest='subcommand')
    # Get a sub_parser
    sub_parser = sub_parsers.add_parser('test')
    # Run it through seed_settings_args
    seed_settings_args(sub_parser)
    return parser, sub_parsers, sub_parser

######## Tests ########

def test_seed_settings_args():
    """Test the seed_settings_args function"""
    # First test that it doesn't crash
    p, s, sub1 = setup_subparser()
    # Get all the desitnations
    dest_index = {}
    for i, act in enumerate(sub1._actions):
        dest_index[act.dest] = i
    # Next, check its arguments
    for act in sub1._actions:
        if act.dest == "help":
            continue
        if act.dest not in SETTING_TYPES:
            raise KeyError(f"Missing from settings: {act.dest}")
        if act.dest in STATIC_SETTINGS:
            raise KeyError(f"sub_parser has arg: {act.dest}, which is static.")
    # Next, check the settings
    for key, dtype in SETTING_TYPES.items():
        # Skip static
        if key in STATIC_SETTINGS:
            continue
        # Check that there does exist an action like this
        if key not in dest_index:
            raise KeyError(f"Missing from sub_parser: {key}")
        # We are unconcerned with act_tokens at this stage.
        act_type = sub1._actions[dest_index[key]].type
        # Check for booleans
        if (dtype == bool) and (act_type is str2bool):
            continue
        # Next, check the type
        if dtype != act_type:
            raise TypeError(
                f"Type mismatch: {key} has type {act_type}, "
                f"but Setting has type {dtype}!"
            )

######## Main ########
def main():
    test_seed_settings_args()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
