#!/usr/bin/env python3
"""Test the AGNDisk object"""
######## Imports ########
#### Standard ####
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

#### Local ####
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.inputs.ReadInputs import INPUT_TYPES
from mcfacts.inputs.ReadInputs import ReadInputs_ini
from mcfacts.inputs.ReadInputs import load_disk_arrays
from mcfacts.inputs.ReadInputs import construct_disk_direct
from mcfacts.inputs.ReadInputs import construct_disk_pAGN
from mcfacts.inputs.ReadInputs import construct_disk_interp
from mcfacts.inputs.settings_manager import SettingsManager
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

# SMBH masses to try
SMBH_MASSES = np.asarray([1e8,])
# disk_alpha_viscosities to try
DISK_ALPHA_VISCOSITIES = np.asarray([0.01,])
# disk_bh_eddington_ratios to try
DISK_BH_EDDINGTON_RATIOS = np.asarray([1.0,])

######## Tests ########

def test_construct_disk_object(verbose=True):
    """test mcfacts.objects.disk.AGNDiskInterp

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing AGNDiskInterp object")
    # Check that the data folder exists
    data_folder = impresources.files(mcfacts_input_data)
    assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
    # Find the default inifile
    fname_default_ini = data_folder / "mcfacts_default.ini"
    assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
    # Get input variables
    input_variables = ReadInputs_ini(fname_default_ini, verbose=False)
    # We only want disk_radius_outer
    disk_radius_outer = input_variables["disk_radius_outer"]
    # Loop disk models
    for disk_model_name in DISK_MODEL_NAMES:
        if verbose:
            print(disk_model_name)
        disko = AGNDiskInterp.from_importlib(disk_model_name, disk_radius_outer)
        # Load the disk arrays
        trunc_surf_density_data, trunc_aspect_ratio_data, \
                trunc_opacity_data, trunc_sound_speed_data, \
                trunc_density_data, trunc_omega_data, \
                trunc_pressure_data, trunc_temperature_data = \
            load_disk_arrays(disk_model_name, disk_radius_outer)
        # Evaluate estimates for each quantity
        surface_density_estimate = disko.surface_density(trunc_surf_density_data[0])
        assert np.allclose(surface_density_estimate, trunc_surf_density_data[1]), \
            "NumPy allclose failed for %s surface_density interpolation"%(disk_model_name)
        aspect_ratio_estimate = disko.aspect_ratio(trunc_aspect_ratio_data[0])
        assert np.allclose(aspect_ratio_estimate, trunc_aspect_ratio_data[1]), \
            "NumPy allclose failed for %s aspect_ratio interpolation"%(disk_model_name)
        opacity_estimate = disko.opacity(trunc_opacity_data[0])
        assert np.allclose(opacity_estimate, trunc_opacity_data[1]), \
            "NumPy allclose failed for %s opacity interpolation"%(disk_model_name)
        sound_speed_estimate = disko.sound_speed(trunc_sound_speed_data[0])
        assert np.allclose(sound_speed_estimate, trunc_sound_speed_data[1]), \
            "NumPy allclose failed for %s sound_speed interpolation"%(disk_model_name)
        density_estimate = disko.density(trunc_density_data[0])
        assert np.allclose(density_estimate, trunc_density_data[1]), \
            "NumPy allclose failed for %s density interpolation"%(disk_model_name)
        omega_estimate = disko.omega(trunc_omega_data[0])
        assert np.allclose(omega_estimate, trunc_omega_data[1]), \
            "NumPy allclose failed for %s omega interpolation"%(disk_model_name)
        pressure_estimate = disko.pressure_gradient(trunc_pressure_data[0])
        assert np.allclose(pressure_estimate, trunc_pressure_data[1]), \
            "NumPy allclose failed for %s pressure interpolation"%(disk_model_name)
        temperature_estimate = disko.temperature(trunc_temperature_data[0])
        assert np.allclose(temperature_estimate, trunc_temperature_data[1]), \
            "NumPy allclose failed for %s temperature interpolation"%(disk_model_name)
        # Identify midplane pressure
        midplane_pressure = (trunc_sound_speed_data[1] ** 2) / trunc_density_data[1]
        dPdR_estimator = CubicSpline(
            np.log10(trunc_density_data[0]),
            np.log10(midplane_pressure),
        ).derivative()
        dPdR_estimate_scipy = dPdR_estimator(
            np.log10(trunc_density_data[0])
        )
        dPdR_estimate_disk = disko.dlog10_midplane_pressure_dlog10R(
            np.log10(trunc_density_data[0])
        )
        assert np.allclose(
            dPdR_estimate_scipy,
            dPdR_estimate_disk,
        )
        # Create surface density log derivative interpolator object
        dlog10_surface_density_dlog10R_estimator = CubicSpline(
            np.log10(trunc_surf_density_data[0]),
            np.log10(trunc_surf_density_data[1]),
        ).derivative()
        dlog10_surface_density_dlog10R_estimate_scipy = \
            dlog10_surface_density_dlog10R_estimator(
                np.log10(trunc_surf_density_data[0])
            )
        dlog10_surface_density_dlog10R_estimate_disk = \
            disko.dlog10_surface_density_dlog10R(
                np.log10(trunc_surf_density_data[0])
            )
        assert np.allclose(
            dlog10_surface_density_dlog10R_estimate_scipy,
            dlog10_surface_density_dlog10R_estimate_disk,
        )

        # Create temperature log derivative interpolator object
        dlog10_temp_dlog10R_estimator = CubicSpline(
            np.log10(trunc_temperature_data[0]),
            np.log10(trunc_temperature_data[1]),
        ).derivative()
        dlog10_temp_dlog10R_estimate_scipy = \
            dlog10_temp_dlog10R_estimator(
                np.log10(trunc_temperature_data[0])
            )
        dlog10_temp_dlog10R_estimate_disk = \
            disko.dlog10_temperature_dlog10R(
                np.log10(trunc_temperature_data[0])
            )
        assert np.allclose(
            dlog10_temp_dlog10R_estimate_scipy,
            dlog10_temp_dlog10R_estimate_disk,
        )
        # Test aliases
        surface_density_estimate = disko.disk_surface_density(trunc_surf_density_data[0])
        assert np.allclose(surface_density_estimate, trunc_surf_density_data[1]), \
            "NumPy allclose failed for %s surface_density interpolation"%(disk_model_name)
        aspect_ratio_estimate = disko.disk_aspect_ratio(trunc_aspect_ratio_data[0])
        assert np.allclose(aspect_ratio_estimate, trunc_aspect_ratio_data[1]), \
            "NumPy allclose failed for %s aspect_ratio interpolation"%(disk_model_name)
        opacity_estimate = disko.disk_opacity(trunc_opacity_data[0])
        assert np.allclose(opacity_estimate, trunc_opacity_data[1]), \
            "NumPy allclose failed for %s opacity interpolation"%(disk_model_name)
        sound_speed_estimate = disko.disk_sound_speed(trunc_sound_speed_data[0])
        assert np.allclose(sound_speed_estimate, trunc_sound_speed_data[1]), \
            "NumPy allclose failed for %s sound_speed interpolation"%(disk_model_name)
        density_estimate = disko.disk_density(trunc_density_data[0])
        assert np.allclose(density_estimate, trunc_density_data[1]), \
            "NumPy allclose failed for %s density interpolation"%(disk_model_name)
        omega_estimate = disko.disk_omega(trunc_omega_data[0])
        assert np.allclose(omega_estimate, trunc_omega_data[1]), \
            "NumPy allclose failed for %s omega interpolation"%(disk_model_name)
        pressure_estimate = disko.disk_pressure_grad(trunc_pressure_data[0])
        assert np.allclose(pressure_estimate, trunc_pressure_data[1]), \
            "NumPy allclose failed for %s pressure interpolation"%(disk_model_name)
        temperature_estimate = disko.temp_func(trunc_temperature_data[0])
        assert np.allclose(temperature_estimate, trunc_temperature_data[1]), \
            "NumPy allclose failed for %s temperature interpolation"%(disk_model_name)
        # TODO test log10 derivatives

    if verbose:
        print("  pass!")

def test_pagn_disk_object(verbose=True):
    """test mcfacts.objects.disk.AGNDiskInterp

    Parameters
    ----------
    verbose : bool
        Verbose output
    """
    if verbose:
        print("Testing AGNDiskInterp object")
    # Check that the data folder exists
    data_folder = impresources.files(mcfacts_input_data)
    assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
    # Find the default inifile
    fname_default_ini = data_folder / "mcfacts_default.ini"
    assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
    # Get input variables
    input_variables = ReadInputs_ini(fname_default_ini, verbose=False)
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
        # Get disk model
        disko = AGNDiskInterp.from_pagn(
            test_config.disk_model_name,
            test_config.smbh_mass,
            disk_radius_outer,
            test_config.disk_alpha_viscosity,
            test_config.disk_bh_eddington_ratio,
        )
        # Run pAGN
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                (
                    disk_surf_dens_func,
                    disk_aspect_ratio_func,
                    disk_opacity_func,
                    sound_speed_func,
                    disk_density_func,
                    disk_pressure_grad_func,
                    disk_omega_func,
                    disk_surf_dens_func_log,
                    temp_func,
                    surf_dens_log10_derivative_func,
                    temp_log10_derivative_func,
                    pressure_log10_derivative_func,
                    disk_model_properties,
                    bonus_structures
                ) = construct_disk_pAGN(
                    test_config.disk_model_name,
                    test_config.smbh_mass,
                    disk_radius_outer,
                    test_config.disk_alpha_viscosity,
                    test_config.disk_bh_eddington_ratio,
                )
        # Evaluate estimates for each quantity
        surface_density_loc = np.exp(disko._surface_density_loglog.x_train_unstacked[0])
        aspect_ratio_loc    = np.exp(disko._aspect_ratio_loglog.x_train_unstacked[0])
        opacity_loc         = np.exp(disko._opacity_loglog.x_train_unstacked[0])
        sound_speed_loc     = np.exp(disko._sound_speed_loglog.x_train_unstacked[0])
        density_loc         = np.exp(disko._density_loglog.x_train_unstacked[0])
        omega_loc           = np.exp(disko._omega_loglog.x_train_unstacked[0])
        temperature_loc     = np.exp(disko._temperature_loglog.x_train_unstacked[0])
        pressure_loc        = disko._pressure_grad_linear.x_train_unstacked[0]
        dSigmadR_loc        = disko._dlog10_surface_density_dlog10R.x_train_unstacked[0]
        dTdR_loc            = disko._dlog10_temp_dlog10R.x_train_unstacked[0]
        dPdR_loc            = disko._dlog10_midplane_pressure_dlog10R.x_train_unstacked[0]

        assert np.allclose(
            disk_surf_dens_func(    surface_density_loc),
            disko.surface_density(  surface_density_loc),
        ), "NumPy allclose failed for %s surface_density interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            disk_aspect_ratio_func( aspect_ratio_loc),
            disko.aspect_ratio(     aspect_ratio_loc),
        ), "NumPy allclose failed for %s aspect_ratio interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            disk_opacity_func(  opacity_loc),
            disko.opacity(      opacity_loc),
        ), "NumPy allclose failed for %s opacity interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            sound_speed_func(   sound_speed_loc),
            disko.sound_speed(  sound_speed_loc),
        ), "NumPy allclose failed for %s sound_speed interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            disk_density_func(  density_loc),
            disko.density(      density_loc),
        ), "NumPy allclose failed for %s density interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            disk_omega_func(    omega_loc),
            disko.omega(        omega_loc),
        ), "NumPy allclose failed for %s omega interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            temp_func(          temperature_loc),
            disko.temperature(  temperature_loc),
        ), "NumPy allclose failed for %s temperature interpolation"%(test_config.disk_model_name)
        # Identify midplane pressure
        assert np.allclose(
            disk_pressure_grad_func(    pressure_loc[1:]),
            disko.pressure_gradient(    pressure_loc)[1:],
        ), "NumPy allclose failed for %s pressure interpolation"%(test_config.disk_model_name)
        # dlog10_dR stuff
        assert np.allclose(
            surf_dens_log10_derivative_func(dSigmadR_loc),
            disko.dlog10_surface_density_dlog10R(dSigmadR_loc),
        ), "NumPy allclose failed for %s dlog10Sigmadlog10R interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            temp_log10_derivative_func(dTdR_loc), 
            disko.dlog10_temperature_dlog10R(dTdR_loc),
        ), "NumPy allclose failed for %s dlog10Tdlog10R interpolation"%(test_config.disk_model_name)
        assert np.allclose(
            pressure_log10_derivative_func(dPdR_loc), 
            disko.dlog10_midplane_pressure_dlog10R(dPdR_loc)
        ), "NumPy allclose failed for %s dlog10Pdlog10R interpolation"%(test_config.disk_model_name)

    if verbose:
        print("  pass!")
    # Loop disk models
    for disk_model_name in DISK_MODEL_NAMES:
        if verbose:
            print(disk_model_name)

def test_from_settings():
    """Test calling AGNDisk objects with the settings dictionary"""
    settings = SettingsManager()
    # Test 1: SG no pAGN
    settings.disk_model_name = "sirko_goodman"
    settings.smbh_mass = 1e8
    settings.disk_radius_outer = 5e4
    settings.flag_use_pagn = False
    cached_disk = AGNDisk(settings)
    assert cached_disk.smbh_mass == settings.smbh_mass
    assert cached_disk.disk_radius_outer == settings.disk_radius_outer
    # Make sure it runs
    surface_density_loc = np.exp(cached_disk._surface_density_loglog.x_train_unstacked[0])
    _ = cached_disk.surface_density(surface_density_loc),
    # Test 2: Changing things doesn't change the disk
    settings.disk_radius_outer = 5.1e4
    settings.smbh_mass = 1.1e8
    assert cached_disk.smbh_mass != settings.smbh_mass
    assert cached_disk.disk_radius_outer != settings.disk_radius_outer
    # Test 3: SG pAGN
    settings.disk_model_name = "sirko_goodman"
    settings.smbh_mass = 1e8
    settings.disk_radius_outer = 5e4
    settings.flag_use_pagn = True
    pagn_disk = AGNDisk(settings)
    model = pagn_disk.pagn_model
    bonus = pagn_disk.pagn_bonus_structures
    # Test 4: 1 pc
    settings.flag_use_pagn = False
    settings.disk_radius_max_pc = -1.0
    cached_disk = AGNDisk(settings)
    parsec = cached_disk.disk_radius_outer
    assert parsec != pagn_disk.disk_radius_outer
    # Test 5: half max
    settings.disk_radius_max_pc = settings.disk_radius_outer / (2 * parsec)
    cached_disk = AGNDisk(settings)
    assert cached_disk.disk_radius_outer < settings.disk_radius_outer

######## Main ########
def main():
    test_construct_disk_object()
    test_pagn_disk_object()
    test_from_settings()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
