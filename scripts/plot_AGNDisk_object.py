#!/usr/bin/env python3
"""Plot the properties of an AGNDisk object"""
######## Imports ########
#### Standard ####
import argparse
from importlib import resources as impresources
import os
import sys
import contextlib
from os.path import isdir, isfile
import itertools
import collections

#### Third Party ####
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt

#### Local ####
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.inputs.scaling import setup_scaling
from mcfacts.inputs.ReadInputs import INPUT_TYPES
from mcfacts.inputs.ReadInputs import ReadInputs_ini
from mcfacts.inputs.ReadInputs import load_disk_arrays
from mcfacts.inputs.ReadInputs import construct_disk_direct
from mcfacts.inputs.ReadInputs import construct_disk_pAGN
from mcfacts.inputs.ReadInputs import construct_disk_interp
from mcfacts.objects.snapshot import IniSnapshotHandler
from mcfacts.inputs.settings_manager import SettingsManager, DEFAULT_SETTINGS
from mcfacts.objects.disk import AGNDisk, AGNDiskInterp

######## Setup ########
# Taken from <https://stackoverflow.com/a/9098295/4761692>
def named_product(**items):
    Options = collections.namedtuple('Options', items.keys())
    return itertools.starmap(Options, itertools.product(*items.values()))

# Disk model names to try
DISK_MODEL_NAMES = [
    "sirko_goodman",
    "thompson_etal",
]

#### Defaults ####
DEFAULT_SMBH_MASS               = None
DEFAULT_DISK_MODEL_NAME         = None
DEFAULT_DISK_RADIUS_OUTER       = None
DEFAULT_DISK_ALPHA_VISCOSITY    = None
DEFAULT_DISK_BH_EDDINGTON_RATIO = None
for prop in DEFAULT_SETTINGS:
    if prop.name == "smbh_mass":
        DEFAULT_SMBH_MASS               = prop.value
    elif prop.name == "disk_model_name":
        DEFAULT_DISK_MODEL_NAME         = prop.value
    elif prop.name == "disk_radius_outer":
        DEFAULT_DISK_RADIUS_OUTER       = prop.value
    elif prop.name == "disk_alpha_viscosity":
        DEFAULT_DISK_ALPHA_VISCOSITY    = prop.value
    elif prop.name == "disk_bh_eddington_ratio":
        DEFAULT_DISK_BH_EDDINGTON_RATIO = prop.value
assert DEFAULT_SMBH_MASS                is not None
assert DEFAULT_DISK_MODEL_NAME          is not None
assert DEFAULT_DISK_RADIUS_OUTER        is not None
assert DEFAULT_DISK_ALPHA_VISCOSITY     is not None
assert DEFAULT_DISK_BH_EDDINGTON_RATIO  is not None

######## Arg ########
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pAGN", action='store_true',
        help="Use pAGN to setup a disk model.",
    )
    parser.add_argument("--truncate-opacity", action='store_true',
        help="Flag: Truncate the disk where opacity drops off.")
    parser.add_argument("--settings", type=str, default=None,
        help="Settings file to import.",
    )
    parser.add_argument("--smbh_mass", type=float, 
        default=DEFAULT_SMBH_MASS,
        help="SMBH mass (solar masses)",
    )
    parser.add_argument("--disk_model_name", type=str,
        default=DEFAULT_DISK_MODEL_NAME,
        help="Name of disk model (sirko_goodman/thompson_etal)",
    )
    parser.add_argument("--disk_radius_outer", type=float,
        default=DEFAULT_DISK_RADIUS_OUTER,
        help="Outer disk radius (R_g)",
    )
    parser.add_argument("--disk_alpha_viscosity", type=float,
        default=DEFAULT_DISK_ALPHA_VISCOSITY,
        help="Shakura-Sunyaev viscoisty parameter",
    )
    parser.add_argument("--disk_bh_eddington_ratio", type=float,
        default=DEFAULT_DISK_BH_EDDINGTON_RATIO,
        help="Fraction of eddington accretion onto SMBH.",
    )
    opts = parser.parse_args()
    return opts


######## Tests ########
def plot_disk_object(
        settings = None,
        pAGN = False,
        truncate_opacity = False,
        smbh_mass = DEFAULT_SMBH_MASS,
        disk_model_name = DEFAULT_DISK_MODEL_NAME,
        disk_radius_outer = DEFAULT_DISK_RADIUS_OUTER,
        disk_alpha_viscosity = DEFAULT_DISK_ALPHA_VISCOSITY,
        disk_bh_eddington_ratio = DEFAULT_DISK_BH_EDDINGTON_RATIO,
    ):
    """Plot disk properties for the AGNDisk object

    Parameters
    ----------
    pAGN : bool
        Flag (use pAGN or not)
    smbh_mass : float
        SMBH mass in solar masses
    disk_model_name : str
        Name of disk model (sirko_goodman/thompson_etal)
    disk_radius_outer : float
        Outer disk radius (R_g)
    disk_alpha_viscosity : float
        Shakura-Sunyaev viscosity parameter
    disk_bh_eddington_ratio : float
        Fraction of Eddington accretion onto SMBH
    """
    #### Handle inputs ####
    # Initialize a new SettingsManager object
    if settings is None:
        settings_path = None
        # Overwrite settings as provided
        overrides = {
            "flag_use_pagn"             : pAGN,
            "disk_truncation"           : \
                "opacity-equals-inner-disk" if truncate_opacity else "none",
            "smbh_mass"                 : smbh_mass,
            "disk_model_name"           : disk_model_name,
            "disk_radius_outer"         : disk_radius_outer,
            "disk_alpha_viscosity"      : disk_alpha_viscosity,
            "disk_bh_eddington_ratio"   : disk_bh_eddington_ratio,
        }
        settings = SettingsManager(settings_overrides=overrides)
        setup_scaling(settings)
    elif isinstance(settings, SettingsManager):
        pass
    else:
        settings_path = Path(settings)
        if not settings_path.is_file():
            raise FileNotFoundError(f"No such file: {settings_path}")
        par = settings_path.parent
        ini_loader = IniSnapshotHandler()
        settings = ini_loader.load(par, settings_path.stem)
        setup_scaling(settings)
    # Make disk
    agn_disk = AGNDisk(settings)
    # Evaluate estimates for each quantity
    training_loc = {
    "surface_density_loglog"            : \
        agn_disk._surface_density_loglog.x_train_unstacked[0] / np.log(10),
    "aspect_ratio_loglog"               : \
        agn_disk._aspect_ratio_loglog.x_train_unstacked[0] / np.log(10),
    "opacity_loglog"                    : \
        agn_disk._opacity_loglog.x_train_unstacked[0] / np.log(10),
    "sound_speed_loglog"                : \
        agn_disk._sound_speed_loglog.x_train_unstacked[0] / np.log(10),
    "density_loglog"                    : \
        agn_disk._density_loglog.x_train_unstacked[0] / np.log(10),
    "omega_loglog"                      : \
        agn_disk._omega_loglog.x_train_unstacked[0] / np.log(10),
    "temperature_loglog"                : \
        agn_disk._temperature_loglog.x_train_unstacked[0] / np.log(10),
    "pressure_grad_linear"              : \
        agn_disk._pressure_grad_linear.x_train_unstacked[0],
    "dlog10_surface_density_dlog10R"    : \
        agn_disk._dlog10_surface_density_dlog10R.x_train_unstacked[0],
    "dlog10_temp_dlog10R"               : \
        agn_disk._dlog10_temp_dlog10R.x_train_unstacked[0],
    "dlog10_midplane_pressure_dlog10R"  : \
        agn_disk._dlog10_midplane_pressure_dlog10R.x_train_unstacked[0],
    }
    training_val = {}
    for item in training_loc:
        training_val[item] = getattr(agn_disk, f"_{item}").y_train
        if "loglog" in item:
            training_val[item] /= np.log(10)

    # Make plots
    plt.style.use('bmh')
    for item in training_loc:
        fig, ax = plt.subplots()
        fig.suptitle(item)
        if "linear" in item:
            ax.semilogx(training_loc[item], training_val[item])
        else:
            ax.plot(training_loc[item], training_val[item])
        plt.show()
        plt.close()


######## Execution ########
if __name__ == "__main__":
    plot_disk_object(**arg().__dict__)
