"""Test mcfacts.inputs.ReadInputs.py functions

Test various things from ReadInputs.py
"""
######## Imports ########
#### Standard ####
from importlib import resources as impresources
import os
from os.path import isdir, isfile
import itertools
import collections

#### Third Party ####
import numpy as np

#### Local ####
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.inputs.ReadInputs import INPUT_TYPES
from mcfacts.inputs.ReadInputs import ReadInputs_ini
from mcfacts.inputs.ReadInputs import load_disk_arrays
from mcfacts.inputs.ReadInputs import construct_disk_direct
from mcfacts.inputs.ReadInputs import construct_disk_pAGN
from mcfacts.inputs.ReadInputs import construct_disk_interp

######## Setup ########
# Disk model names to try
DISK_MODEL_NAMES = [
    "sirko_goodman",
    "thompson_etal",
]

# SMBH masses to try
#SMBH_MASSES = np.asarray([1e6, 1e7, 1e8, 1e9,])
SMBH_MASSES = np.asarray([1e8,])
# disk_alpha_viscosities to try
DISK_ALPHA_VISCOSITIES = np.asarray([0.1, 0.5])
# disk_bh_eddington_ratios to try
DISK_BH_EDDINGTON_RATIOS = np.asarray([0.5,0.1])
# pAGN flag
FLAG_USE_PAGN = np.asarray([True, False])

######## Functions ########

# Taken from <https://stackoverflow.com/a/9098295/4761692>
def named_product(**items):
    Options = collections.namedtuple('Options', items.keys())
    return itertools.starmap(Options, itertools.product(*items.values()))

######## Tests ########
def test_input_types(verbose=True):
    """test the INPUT_TYPES dictionary

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing INPUT_TYPES")
    # Check type
    assert isinstance(INPUT_TYPES, dict), "INPUT_TYPES is not a dict"
    # Check key/value pairs
    for key in INPUT_TYPES:
        # Assign value from dict
        value = INPUT_TYPES[key]
        # Check that it is a class
        assert "class" in str(value)
        if verbose:
            print("  INPUT_TYPES[%s] = %s"%(str(key), str(value)))
    if verbose:
        print("  pass!")

def test_ReadInputs_ini(verbose=True):
    """test ReadInputs_ini function and surrounding data

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing ReadInputs_ini")
    # Check that the data folder exists
    data_folder = impresources.files(mcfacts_input_data)
    assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
    # Find model_choice.ini
    fname_ini = data_folder / "model_choice.ini"
    assert isfile(fname_ini), "Cannot find %s"%fname_ini
    # Get input variables
    input_variables = ReadInputs_ini(fname_ini, verbose=verbose)
    # Check that this returns a dictionary
    assert isinstance(input_variables, dict), \
        "ReadInputs_ini returned %s"%(str(type(input_variables)))
    # Loop the input variables
    for key in input_variables:
        # Find key/value pairs
        value = input_variables[key]
        # Check that key is in INPUT_TYPES
        assert key in INPUT_TYPES, \
            "%s is not defined in ReadInputs.INPUT_TYPES"%(key)
        # check the type of value is correct
        assert isinstance(value, INPUT_TYPES[key]), \
            "%s is not a %s (ReadInputs.INPUT_TYPES"%(key,str(INPUT_TYPES[key]))
        # Print key/value pair
        if verbose:
            print("  %s = %s (type: %s)"%(key,str(value),str(INPUT_TYPES[key])))
    if verbose:
        print("  pass!")

def test_load_disk_arrays(verbose=True):
    """test mcfacts.inputs.ReadInputs.load_disk_arrays

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing load_disk_arrays")
    # Check that the data folder exists
    data_folder = impresources.files(mcfacts_input_data)
    assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
    # Identify some files that should exist in the data folder
    fname_thompson_surface_density      = data_folder / "thompson_etal_surface_density.txt"
    fname_thompson_aspect_ratio         = data_folder / "thompson_etal_aspect_ratio.txt"
    fname_sirko_goodman_surface_density = data_folder / "sirko_goodman_surface_density.txt"
    fname_sirko_goodman_aspect_ratio    = data_folder / "sirko_goodman_aspect_ratio.txt"
    # Check things that should exist in the data folder
    assert isfile(fname_thompson_surface_density), \
        "Cannot find %s"%(fname_thompson_surface_density)
    assert isfile(fname_thompson_aspect_ratio), \
        "Cannot find %s"%(fname_thompson_aspect_ratio)
    assert isfile(fname_sirko_goodman_surface_density), \
        "Cannot find %s"%(fname_sirko_goodman_surface_density)
    assert isfile(fname_sirko_goodman_aspect_ratio), \
        "Cannot find %s"%(fname_sirko_goodman_aspect_ratio)
    # Find the default inifile
    fname_default_ini = data_folder / "model_choice.ini"
    assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
    # Get input variables
    input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
    # We only want disk_radius_outer
    disk_radius_outer = input_variables["disk_radius_outer"]
    # Loop disk models
    for disk_model_name in DISK_MODEL_NAMES:
        # Load the disk arrays
        trunc_surf_density_data, trunc_aspect_ratio_data, \
                trunc_opacity_data, trunc_sound_speed_data, \
                trunc_density_data, trunc_omega_data, \
                trunc_pressure_data, trunc_temperature_data = \
            load_disk_arrays(disk_model_name, disk_radius_outer)
        # Check the arrays
        assert isinstance(trunc_surf_density_data, np.ndarray), \
            "load_disk_arrays returned trunc_surf_density_data as type %s"%(
                type(trunc_surf_density_data)
            )
        assert isinstance(trunc_aspect_ratio_data, np.ndarray), \
            "load_disk_arrays returned trunc_aspect_ratio_data as type %s"%(
                type(trunc_aspect_ratio_data)
            )
        assert isinstance(trunc_opacity_data, np.ndarray), \
            "load_disk_arrays returned trunc_opacity_data as type %s"%(
                type(trunc_opacity_data)
            )
        assert isinstance(trunc_sound_speed_data, np.ndarray), \
            "load_disk_arrays returned trunc_sound_speed_data as type %s"%(
                type(trunc_sound_speed_data)
            )
        assert isinstance(trunc_density_data, np.ndarray), \
            "load_disk_arrays returned trunc_density_data as type %s"%(
                type(trunc_density_data)
            )
        assert isinstance(trunc_omega_data, np.ndarray), \
            "load_disk_arrays returned trunc_omega_data as type %s"%(
                type(trunc_omega_data)
            )
        assert isinstance(trunc_pressure_data, np.ndarray), \
            "load_disk_arrays returned trunc_pressure_data as type %s"%(
                type(trunc_pressure_data)
            )
        assert isinstance(trunc_temperature_data, np.ndarray), \
            "load_disk_arrays returned trunc_temperature_data as type %s"%(
                type(trunc_temperature_data)
            )
        # Check that arrays are one-dimensional
        assert len(trunc_surf_density_data.shape) == 2, \
            "trunc_surf_density_data.shape = %s"%(str(trunc_surf_density_data.shape))
        assert len(trunc_aspect_ratio_data.shape) == 2, \
            "trunc_aspect_ratio_data.shape = %s"%(str(trunc_aspect_ratio_data.shape))
        assert len(trunc_opacity_data.shape) == 2, \
            "trunc_opacity_data.shape = %s"%(str(trunc_opacity_data.shape))
        assert len(trunc_sound_speed_data.shape) == 2, \
            "trunc_sound_speed_data.shape = %s"%(str(trunc_sound_speed_data.shape))
        assert len(trunc_density_data.shape) == 2, \
            "trunc_density_data.shape = %s"%(str(trunc_density_data.shape))
        assert len(trunc_omega_data.shape) == 2, \
            "trunc_omega_data.shape = %s"%(str(trunc_omega_data.shape))
        assert len(trunc_pressure_data.shape) == 2, \
            "trunc_pressure_data.shape = %s"%(str(trunc_pressure_data.shape))
        assert len(trunc_temperature_data.shape) == 2, \
            "trunc_temperature_data.shape = %s"%(str(trunc_temperature_data.shape))
    if verbose:
        print("  pass!")

def test_construct_disk_direct(verbose=True):
    """test mcfacts.inputs.ReadInputs.construct_disk_direct

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing construct_disk_direct")
    # Check that the data folder exists
    data_folder = impresources.files(mcfacts_input_data)
    assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
    # Find the default inifile
    fname_default_ini = data_folder / "model_choice.ini"
    assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
    # Get input variables
    input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
    # We only want disk_radius_outer
    disk_radius_outer = input_variables["disk_radius_outer"]
    # Loop disk models
    for disk_model_name in DISK_MODEL_NAMES:
        # Load the disk arrays
        trunc_surf_density_data, trunc_aspect_ratio_data, \
                trunc_opacity_data, trunc_sound_speed_data, \
                trunc_density_data, trunc_omega_data, \
                trunc_pressure_data, trunc_temperature_data = \
            load_disk_arrays(disk_model_name, disk_radius_outer)
        # Construct disk
        disk_surf_dens_func, disk_aspect_ratio_func, \
                disk_opacity_func, sound_speed_func, \
                disk_density_func, disk_pressure_grad_func, \
                disk_omega_func, disk_surf_dens_func_log, \
                temp_func, \
                surf_dens_log10_derivative_func, \
                temp_log10_derivative_func, \
                pressure_log10_derivative_func, \
                disk_model_properties = \
            construct_disk_direct(disk_model_name, disk_radius_outer)
        # Evaluate estimates for each quantity
        surface_density_estimate = disk_surf_dens_func(trunc_surf_density_data[0])
        assert np.allclose(surface_density_estimate, trunc_surf_density_data[1]), \
            "NumPy allclose failed for %s surface_density interpolation"%(disk_model_name)
        aspect_ratio_estimate = disk_aspect_ratio_func(trunc_aspect_ratio_data[0])
        assert np.allclose(aspect_ratio_estimate, trunc_aspect_ratio_data[1]), \
            "NumPy allclose failed for %s aspect_ratio interpolation"%(disk_model_name)
        opacity_estimate = disk_opacity_func(trunc_opacity_data[0])
        assert np.allclose(opacity_estimate, trunc_opacity_data[1]), \
            "NumPy allclose failed for %s opacity interpolation"%(disk_model_name)
        sound_speed_estimate = sound_speed_func(trunc_sound_speed_data[0])
        assert np.allclose(sound_speed_estimate, trunc_sound_speed_data[1]), \
            "NumPy allclose failed for %s sound_speed interpolation"%(disk_model_name)
        density_estimate = disk_density_func(trunc_density_data[0])
        assert np.allclose(density_estimate, trunc_density_data[1]), \
            "NumPy allclose failed for %s density interpolation"%(disk_model_name)
        omega_estimate = disk_omega_func(trunc_omega_data[0])
        assert np.allclose(omega_estimate, trunc_omega_data[1]), \
            "NumPy allclose failed for %s omega interpolation"%(disk_model_name)
        pressure_estimate = disk_pressure_grad_func(trunc_pressure_data[0])
        assert np.allclose(pressure_estimate, trunc_pressure_data[1]), \
            "NumPy allclose failed for %s pressure interpolation"%(disk_model_name)
        temperature_estimate = temp_func(trunc_temperature_data[0])
        assert np.allclose(temperature_estimate, trunc_temperature_data[1]), \
            "NumPy allclose failed for %s temperature interpolation"%(disk_model_name)
        # TODO test log10 derivatives

    if verbose:
        print("  pass!")

def test_construct_disk_pAGN(verbose=True):
    """test mcfacts.inputs.ReadInputs.construct_disk_pAGN

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing construct_disk_pAGN")
    # Check that the data folder exists
    data_folder = impresources.files(mcfacts_input_data)
    assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
    # Find the default inifile
    fname_default_ini = data_folder / "model_choice.ini"
    assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
    # Get input variables
    input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
    # We only want disk_radius_outer
    disk_radius_outer = input_variables["disk_radius_outer"]
    # Construct productspace
    test_product_space = named_product(
        disk_model_name         = DISK_MODEL_NAMES,
        smbh_mass               = SMBH_MASSES,
        disk_alpha_viscosity    = DISK_ALPHA_VISCOSITIES,
        disk_bh_eddington_ratio = DISK_BH_EDDINGTON_RATIOS,
    )
    # Loop tests
    for test_config in test_product_space:
        # Run pAGN
        disk_surf_dens_func, disk_aspect_ratio_func, \
                disk_opacity_func, sound_speed_func, \
                disk_density_func, disk_pressure_grad_func, \
                disk_omega_func, disk_surf_dens_func_log, \
                temp_func, surf_dens_log10_derivative_func, \
                temp_log10_derivative_func, pressure_log10_derivative_func, \
                disk_model_properties, bonus_structures = \
            construct_disk_pAGN(
                test_config.disk_model_name,
                test_config.smbh_mass,
                disk_radius_outer,
                test_config.disk_alpha_viscosity,
                test_config.disk_bh_eddington_ratio,
            )
    if verbose:
        print("  pass!")

def test_construct_disk_interp(
    verbose=True,
    ):
    """Test mcfacts.inputs.ReadInputs.construct_disk_interp

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing construct_disk_interp")
    # Check that the data folder exists
    data_folder = impresources.files(mcfacts_input_data)
    assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
    # Find the default inifile
    fname_default_ini = data_folder / "model_choice.ini"
    assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
    # Get input variables
    input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
    # We want a few things
    smbh_mass = input_variables["smbh_mass"]
    disk_radius_outer = input_variables["disk_radius_outer"]
    disk_alpha_viscosity = input_variables["disk_alpha_viscosity"]
    disk_bh_eddington_ratio = input_variables["disk_bh_eddington_ratio"]
    disk_radius_max_pc = input_variables["disk_radius_max_pc"]
    # Construct productspace
    test_product_space = named_product(
        disk_model_name = DISK_MODEL_NAMES,
        flag_use_pagn   = FLAG_USE_PAGN,
    )
    # Loop tests
    for test_config in test_product_space:
        # Run function
        disk_surf_dens_func, disk_aspect_ratio_func, \
                disk_opacity_func, sound_speed_func, \
                disk_density_func, disk_pressure_grad_func, \
                disk_omega_func, disk_surf_dens_func_log, \
                temp_func, surf_dens_log10_derivative_func, \
                temp_log10_derivative_func, pressure_log10_derivative_func = \
            construct_disk_interp(
                smbh_mass,
                disk_radius_outer,
                test_config.disk_model_name,
                disk_alpha_viscosity,
                disk_bh_eddington_ratio,
                disk_radius_max_pc=disk_radius_max_pc,
                flag_use_pagn=test_config.flag_use_pagn,
                verbose=verbose,
            )
    if verbose:
        print("  pass!")

######## Main ########
def main():
    test_input_types()
    test_ReadInputs_ini()
    test_load_disk_arrays()
    test_construct_disk_direct()
    test_construct_disk_pAGN()
    test_construct_disk_interp()

######## Execution ########
if __name__ == "__main__":
    main()
