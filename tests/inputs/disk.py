#!/usr/bin/env python3
"""Test the AGNDisk object"""
######## Imports ########
#### Standard ####
from importlib import resources as impresources
import os
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
from mcfacts.objects.disk import AGNDisk, AGNDiskInterp

######## Setup ########
# Disk model names to try
DISK_MODEL_NAMES = [
    "sirko_goodman",
    "thompson_etal",
]

# SMBH masses to try
SMBH_MASSES = np.asarray([1e8,])

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
        dlog10_surface_density_log10R_estimator = CubicSpline(
            np.log10(trunc_surf_density_data[0]),
            np.log10(trunc_surf_density_data[1]),
        ).derivative()
        dlog10_surface_density_log10R_estimate_scipy = \
            dlog10_surface_density_log10R_estimator(
                np.log10(trunc_surf_density_data[0])
            )
        dlog10_surface_density_log10R_estimate_disk = \
            disko.dlog10_surface_density_dlog10R(
                np.log10(trunc_surf_density_data[0])
            )
        assert np.allclose(
            dlog10_surface_density_log10R_estimate_scipy,
            dlog10_surface_density_log10R_estimate_disk,
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
######## Main ########
def main():
    test_construct_disk_object()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
