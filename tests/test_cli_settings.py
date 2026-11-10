#!/usr/bin/env python3
"""Test that the mcfacts command line exposes the settings from SettingsManager"""
######## Imports ########
#### Standard Library ####
import argparse
import sys

#### Third Party ####
import pytest

#### Local ####
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.mcfacts import seed_settings_args

######## Setup ########

def parse(monkeypatch, *args):
    # seed_settings_args reads sys.argv itself to look for a settings file
    monkeypatch.setattr(sys, "argv", ["mcfacts", *args])

    parser = argparse.ArgumentParser(allow_abbrev=False)
    seed_settings_args(parser)

    return parser.parse_args(list(args))


# region Settings Arguments

# Check that a setting can be passed by its canonical name
def test_cli_canonical_name(monkeypatch):
    assert parse(monkeypatch, "--mu_circ_encounter", "0.5").mu_circ_encounter == 0.5


# Check that a setting can still be passed by its old (alias) name
def test_cli_alias_name(monkeypatch):
    assert parse(monkeypatch, "--mean_harden_energy_delta", "0.5").mu_circ_encounter == 0.5


# Check that every alias is accepted on the command line
@pytest.mark.parametrize("alias, name", SettingsManager().flat_aliases.items())
def test_cli_every_alias(monkeypatch, alias, name):
    default = getattr(SettingsManager(), name)
    assert getattr(parse(monkeypatch, f"--{alias}", str(default)), name) == default


# Check that the parsed arguments build a SettingsManager with the value given by alias
def test_cli_alias_reaches_settings_manager(monkeypatch):
    inputs = parse(monkeypatch, "--mean_harden_energy_delta", "0.5")
    assert SettingsManager(vars(inputs)).mu_circ_encounter == 0.5

# endregion
