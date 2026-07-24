#!/usr/bin/env python
'''Plotting script for mcfacts quantities'''
######## Imports ########
#### Standard ####
import argparse
import os

#### Third Party ####
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

#### Local ####
from mcfacts.inputs.scaling import setup_scaling, disk_truncation
from mcfacts.inputs.settings_manager import SettingsManager, DEFAULT_SETTINGS
from mcfacts.objects.disk import AGNDisk
from mcfacts.objects.snapshot import IniSnapshotHandler
from mcfacts.utilities.unit_conversion import r_g_from_units
from mcfacts.vis import data, plotting, styles

######## Setup ########
## Set plot style ##
plt.style.use('bmh')
#plt.style.use("mcfacts.vis.mcfacts_figures")
#size = "apj_col"

######## Argparse ########
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stellar-mass-min", default=1e9, type=float,
        help="Minimum stellar mass for scaling test")
    parser.add_argument("--stellar-mass-max", default=1e13, type=float,
        help="Maximum stellar mass for scaling test")
    parser.add_argument("--stellar-mass-bins", default=33, type=int,
        help="How many stellar mass bins for scaling test?")
    parser.add_argument("--flag-use-pagn", '-p', action='store_true',
        help="Use pAGN disk?")
    parser.add_argument("--truncate-disk-opacity", '-t', action='store_true',
        help="Truncate disk based on opacity? (requires pagn)")
    parser.add_argument("--disk-radius-max-pc", default=0., type=float,
        help="Disk max parsec radius.")
    parser.add_argument("--disk-model-name", default="sirko_goodman",
        type=str, help="Which kind of disk are we using?")
    parser.add_argument("--output", "--output-directory", default="./",
        type=str, help="Output directory")
    parser.add_argument("--label", default="disk", type=str,
        help="Base of fname for plots")
    opts = parser.parse_args()
    return opts

def disk_single(
        base : SettingsManager,
        stellar_mass,
        scale_nsc_mass,
    ):
    # Create new settingsmanager
    new = SettingsManager()
    # Universal things for current scaling
    new.set_preprocessing("flag_use_scaling",       True)
    new.set_preprocessing("scale_smbh_mass",       "schramm-silverman")
    new.set_preprocessing("scale_inner_disk",      "decay-time")
    new.set_preprocessing("scale_trap",            "sqrt-smbh")
    new.set_preprocessing("scale_capture_radius",  "sqrt-smbh")
    new.set_preprocessing("scale_capture_time",    "hubble")
    # Specific to how script was called
    new.set_preprocessing("disk_model_name",       base.disk_model_name)
    new.set_preprocessing("disk_radius_max_pc",    base.disk_radius_max_pc)
    new.set_preprocessing("flag_use_pagn",         base.flag_use_pagn)
    new.set_preprocessing("disk_truncation",       base.disk_truncation)
    # Specific to this disk
    new.set_preprocessing("stellar_mass",           float(stellar_mass))
    new.set_preprocessing("scale_nsc_mass",         scale_nsc_mass)
    # Scale disk
    setup_scaling(new)
    return new

def make_disks(
        stellar_mass_min = 1e9,
        stellar_mass_max = 1e13,
        stellar_mass_bins = 33,
        flag_use_pagn = False,
        truncate_disk_opacity = False,
        disk_radius_max_pc = 0.,
        disk_model_name = "sirko_goodman",
    ):
    # Check pAGN and truncation arguments
    if truncate_disk_opacity and not flag_use_pagn:
        raise NotImplementedError(
            f"Truncating based on opacity is currently "
            f"only supported through pAGN"
        )
    # Generate the stellar mass linspace
    mstar_arr = np.logspace(
        np.log10(stellar_mass_min),
        np.log10(stellar_mass_max),
        stellar_mass_bins,
    )
    # Load a settings manager object
    settings = SettingsManager()
    # set universal settings manually
    settings.set_preprocessing("disk_model_name",       disk_model_name)
    settings.set_preprocessing("disk_radius_max_pc",    disk_radius_max_pc)
    settings.set_preprocessing("disk_radius_max_pc",    disk_radius_max_pc)
    settings.set_preprocessing("flag_use_pagn",         flag_use_pagn)
    # Set truncation
    if truncate_disk_opacity:
        settings.set_preprocessing(
            "disk_truncation",
            "opacity-equals-inner-disk"
        )
    else:
        settings.set_preprocessing(
            "disk_truncation",
            "none"
        )
    
    ### Make disks ###
    early_disks = {}
    late_disks = {}
    ## Make early disks ##
    for i, mstar in enumerate(mstar_arr):
        new = disk_single(settings, mstar, "neumayer-early")
        early_disks[i] = AGNDisk(new)
    ## Make late disks ##
    for i, mstar in enumerate(mstar_arr):
        new = disk_single(settings, mstar, "neumayer-late")
        late_disks[i] = AGNDisk(new)

    return early_disks, late_disks


######## Plots ########
def plot_nsc_mass_vs_smbh_mass(early, late, fname_out, title=None):
    '''Plot smbh mass'''
    # Initialize plot
    fig, ax = plt.subplots()
    # Grab data
    early_smbh = [early[key].smbh_mass for key in early]
    early_nsc = [early[key].settings.nsc_mass for key in early]
    late_smbh = [late[key].smbh_mass for key in late]
    late_nsc = [late[key].settings.nsc_mass for key in late]
    # Plot things
    ax.scatter(np.log10(early_smbh), np.log10(early_nsc), label="early")
    ax.scatter(np.log10(late_smbh), np.log10(late_nsc), label="late")
    ax.set_xlabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{SMBH}}) [\mathrm{M}_{\odot}]$")
    ax.set_ylabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{NSC}}) [\mathrm{M}_{\odot}]$")
    # legend
    plt.legend()
    # Check title
    ax.set_title(title)
    # Tight layout
    plt.tight_layout()
    # savefig
    plt.savefig(fname_out)
    # Close plt
    plt.close()

def plot_disk_radii(early, fname_out, title=None):
    '''Plot smbh mass'''
    # Initialize plot
    fig, ax = plt.subplots(dpi=200)
    #fig, ax = plt.subplots(figsize=plotting.set_size(size),dpi=200)
    # Grab data
    early_smbh = np.asarray(
        [early[key].smbh_mass for key in early]
    )
    early_outer = np.asarray(
        [early[key].disk_radius_outer for key in early]
    )
    early_inner = np.asarray(
        [early[key].settings.inner_disk_outer_radius for key in early]
    )
    early_isco = np.asarray(
        [early[key].settings.disk_inner_stable_circ_orb for key in early]
    )
    early_trap = np.asarray(
        [early[key].settings.disk_radius_trap for key in early]
    )
    parsec = r_g_from_units(early_smbh, 1. * u.pc)

    # Fix isco
    isco_mask = early_inner < early_isco
    early_inner[isco_mask] = early_isco[isco_mask]
    # Plot things
    ax.scatter(
        np.log10(early_smbh),
        np.log10(early_outer),
        label="disk_radius_outer"
    )
    ax.scatter(
        np.log10(early_smbh),
        np.log10(early_inner),
        label="inner_disk_outer_radius"
    )
    ax.scatter(
        np.log10(early_smbh),
        np.log10(early_trap),
        label="disk_radius_trap"
    )
    ax.scatter(
        np.log10(early_smbh),
        np.log10(early_isco),
        label="disk_inner_stable_circ_orb"
    )
    ax.plot(
        np.log10(early_smbh),
        np.log10(parsec),
        label="1 pc",
        linestyle="dotted",
        color='black',
    )
    ax.set_xlabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{SMBH}}) [\mathrm{M}_{\odot}]$",fontsize=20)
    ax.set_ylabel(r"$\mathrm{log}_{10}(\mathrm{R}_{\mathrm{g}})$",fontsize=20)
    # legend
    #plt.legend(fontsize=12)
    plt.legend(prop={"size":14})
    # Check title
    ax.set_title(title)
    # Tight layout
    plt.tight_layout()
    # savefig
    plt.savefig(fname_out)
    # Close plt
    plt.close()
    
######## main ########
def main(
        stellar_mass_min = 1e9,
        stellar_mass_max = 1e13,
        stellar_mass_bins = 33,
        flag_use_pagn = False,
        truncate_disk_opacity = False,
        disk_radius_max_pc = 0.,
        disk_model_name = "sirko_goodman",
        output = "./",
        label = "disk",
    ):
    # Assert the directory makes sense
    assert os.path.isdir(output), f"No such directory {output}"
    # Generate all of the disks we will use to make the plot
    early, late = make_disks(
        stellar_mass_min=stellar_mass_min,
        stellar_mass_max=stellar_mass_max,
        stellar_mass_bins=stellar_mass_bins,
        flag_use_pagn=flag_use_pagn,
        truncate_disk_opacity=truncate_disk_opacity,
        disk_radius_max_pc=disk_radius_max_pc,
        disk_model_name = disk_model_name,
    )
    # Plot nsc_mass
    plot_nsc_mass_vs_smbh_mass(
        early,
        late,
        os.path.join(output, f"{label}_nsc_scaling.png"),
        title=label,
    )
    # Plot disk radii
    plot_disk_radii(
        early,
        os.path.join(output, f"{label}_radius_scaling.png"),
        title=label,
    )

######## Execution ########
if __name__ == "__main__":
    main(**arg().__dict__)
