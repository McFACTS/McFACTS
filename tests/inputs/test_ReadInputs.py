# """Test mcfacts.inputs.ReadInputs.py functions
#
# Test various things from ReadInputs.py
# """
# ######## Imports ########
# #### Standard ####
# from importlib import resources as impresources
# import os
# from os.path import isdir, isfile
# import itertools
# import collections
#
# #### Third Party ####
# import numpy as np
#
# #### Local ####
# from mcfacts.inputs import data as mcfacts_input_data
# from mcfacts.inputs.ReadInputs import INPUT_TYPES
# from mcfacts.inputs.ReadInputs import ReadInputs_ini
# from mcfacts.inputs.ReadInputs import load_disk_arrays
# from mcfacts.inputs.ReadInputs import construct_disk_direct
# from mcfacts.inputs.ReadInputs import construct_disk_pAGN
# from mcfacts.inputs.ReadInputs import construct_disk_interp
#
# ######## Setup ########
# # Disk model names to try
# DISK_MODEL_NAMES = [
#     "sirko_goodman",
#     "thompson_etal",
# ]
#
# # SMBH masses to try
# #SMBH_MASSES = np.asarray([1e6, 1e7, 1e8, 1e9,])
# SMBH_MASSES = np.asarray([1e8,])
# # disk_alpha_viscosities to try
# DISK_ALPHA_VISCOSITIES = np.asarray([0.1, 0.5])
# # disk_bh_eddington_ratios to try
# DISK_BH_EDDINGTON_RATIOS = np.asarray([0.5,0.1])
# # pAGN flag
# FLAG_USE_PAGN = np.asarray([True, False])
#
# ######## Functions ########
#
# # Taken from <https://stackoverflow.com/a/9098295/4761692>
# def named_product(**items):
#     Options = collections.namedtuple('Options', items.keys())
#     return itertools.starmap(Options, itertools.product(*items.values()))
#
# ######## Tests ########
# def test_input_types(verbose=True):
#     """test the INPUT_TYPES dictionary
#
#     Parameters
#     ----------
#     verbose : bool
#         Verbose output
#     """
#     if verbose:
#         print("Testing INPUT_TYPES")
#     # Check type
#     assert isinstance(INPUT_TYPES, dict), "INPUT_TYPES is not a dict"
#     # Check key/value pairs
#     for key in INPUT_TYPES:
#         # Assign value from dict
#         value = INPUT_TYPES[key]
#         # Check that it is a class
#         assert "class" in str(value)
#         if verbose:
#             print("  INPUT_TYPES[%s] = %s"%(str(key), str(value)))
#     if verbose:
#         print("  pass!")
#
# def test_ReadInputs_ini(verbose=True):
#     """test ReadInputs_ini function and surrounding data
#
#     Parameters
#     ----------
#     verbose : bool
#         Verbose output
#     """
#     if verbose:
#         print("Testing ReadInputs_ini")
#     # Check that the data folder exists
#     data_folder = impresources.files(mcfacts_input_data)
#     assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
#     # Find model_choice.ini
#     fname_ini = data_folder / "model_choice.ini"
#     assert isfile(fname_ini), "Cannot find %s"%fname_ini
#     # Get input variables
#     input_variables = ReadInputs_ini(fname_ini, verbose=verbose)
#     # Check that this returns a dictionary
#     assert isinstance(input_variables, dict), \
#         "ReadInputs_ini returned %s"%(str(type(input_variables)))
#     # Loop the input variables
#     for key in input_variables:
#         # Find key/value pairs
#         value = input_variables[key]
#         # Check that key is in INPUT_TYPES
#         assert key in INPUT_TYPES, \
#             "%s is not defined in ReadInputs.INPUT_TYPES"%(key)
#         # check the type of value is correct
#         assert isinstance(value, INPUT_TYPES[key]), \
#             "%s is not a %s (ReadInputs.INPUT_TYPES"%(key,str(INPUT_TYPES[key]))
#         # Print key/value pair
#         if verbose:
#             print("  %s = %s (type: %s)"%(key,str(value),str(INPUT_TYPES[key])))
#     if verbose:
#         print("  pass!")
#
# def test_load_disk_arrays(verbose=True):
#     """test mcfacts.inputs.ReadInputs.load_disk_arrays
#
#     Parameters
#     ----------
#     verbose : bool
#         Verbose output
#     """
#     if verbose:
#         print("Testing load_disk_arrays")
#     # Check that the data folder exists
#     data_folder = impresources.files(mcfacts_input_data)
#     assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
#     # Identify some files that should exist in the data folder
#     fname_thompson_surface_density      = data_folder / "thompson_etal_surface_density.txt"
#     fname_thompson_aspect_ratio         = data_folder / "thompson_etal_aspect_ratio.txt"
#     fname_sirko_goodman_surface_density = data_folder / "sirko_goodman_surface_density.txt"
#     fname_sirko_goodman_aspect_ratio    = data_folder / "sirko_goodman_aspect_ratio.txt"
#     # Check things that should exist in the data folder
#     assert isfile(fname_thompson_surface_density), \
#         "Cannot find %s"%(fname_thompson_surface_density)
#     assert isfile(fname_thompson_aspect_ratio), \
#         "Cannot find %s"%(fname_thompson_aspect_ratio)
#     assert isfile(fname_sirko_goodman_surface_density), \
#         "Cannot find %s"%(fname_sirko_goodman_surface_density)
#     assert isfile(fname_sirko_goodman_aspect_ratio), \
#         "Cannot find %s"%(fname_sirko_goodman_aspect_ratio)
#     # Find the default inifile
#     fname_default_ini = data_folder / "model_choice.ini"
#     assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
#     # Get input variables
#     input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
#     # We only want disk_radius_outer
#     disk_radius_outer = input_variables["disk_radius_outer"]
#     # Loop disk models
#     for disk_model_name in DISK_MODEL_NAMES:
#         # Load the disk arrays
#         truncated_disk_radii, truncated_surface_densities, truncated_aspect_ratios = \
#             load_disk_arrays(disk_model_name, disk_radius_outer)
#         # Check the arrays
#         assert isinstance(truncated_disk_radii, np.ndarray), \
#             "load_disk_arrays returned truncated_disk_radii as type %s"%(
#                 type(truncated_disk_radii)
#             )
#         assert isinstance(truncated_surface_densities, np.ndarray), \
#             "load_disk_arrays returned truncated_surface_densities as type %s"%(
#                 type(truncated_surface_densities)
#             )
#         assert isinstance(truncated_aspect_ratios, np.ndarray), \
#             "load_disk_arrays returned truncated_aspect_ratios as type %s"%(
#                 type(truncated_aspect_ratios)
#             )
#         # Check that arrays are one-dimensional
#         assert len(truncated_disk_radii.shape) == 1, \
#             "truncated_disk_radii.shape = %s"%(str(truncated_disk_radii.shape))
#         assert len(truncated_surface_densities.shape) == 1, \
#             "truncated_surface_densities.shape = %s"%(str(truncated_surface_densities.shape))
#         assert len(truncated_aspect_ratios.shape) == 1, \
#             "truncated_aspect_ratios.shape = %s"%(str(truncated_aspect_ratios.shape))
#         # Check that arrays have the same length
#         assert truncated_disk_radii.size == truncated_surface_densities.size, \
#             "truncated_disk_radii and truncated_surface_densities have different length"
#         assert truncated_disk_radii.size == truncated_aspect_ratios.size, \
#             "truncated_disk_radiis and truncated_aspect_ratios have different length"
#     if verbose:
#         print("  pass!")
#
# def test_construct_disk_direct(verbose=True):
#     """test mcfacts.inputs.ReadInputs.construct_disk_direct
#
#     Parameters
#     ----------
#     verbose : bool
#         Verbose output
#     """
#     if verbose:
#         print("Testing construct_disk_direct")
#     # Check that the data folder exists
#     data_folder = impresources.files(mcfacts_input_data)
#     assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
#     # Find the default inifile
#     fname_default_ini = data_folder / "model_choice.ini"
#     assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
#     # Get input variables
#     input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
#     # We only want disk_radius_outer
#     disk_radius_outer = input_variables["disk_radius_outer"]
#     # Loop disk models
#     for disk_model_name in DISK_MODEL_NAMES:
#         # Load the disk arrays
#         truncated_disk_radii, truncated_surface_densities, truncated_aspect_ratios = \
#             load_disk_arrays(disk_model_name, disk_radius_outer)
#         # Construct disk
#         disk_surf_dens_func, disk_aspect_ratio_func, disk_model_properties = \
#             construct_disk_direct(disk_model_name, disk_radius_outer)
#         # Evaluate estimates for each quantity
#         surface_density_estimate = disk_surf_dens_func(truncated_disk_radii)
#         aspect_ratio_estimate = disk_aspect_ratio_func(truncated_disk_radii)
#         # Check that they're close
#         assert np.allclose(surface_density_estimate, truncated_surface_densities), \
#             "NumPy allclose failed for %s surface_density interpolation"%(disk_model_name)
#         assert np.allclose(aspect_ratio_estimate, truncated_aspect_ratios), \
#             "NumPy allclose failed for %s aspect_ratio interpolation"%(disk_model_name)
#     if verbose:
#         print("  pass!")
#
# def test_construct_disk_pAGN(verbose=True):
#     """test mcfacts.inputs.ReadInputs.construct_disk_pAGN
#
#     Parameters
#     ----------
#     verbose : bool
#         Verbose output
#     """
#     if verbose:
#         print("Testing construct_disk_pAGN")
#     # Check that the data folder exists
#     data_folder = impresources.files(mcfacts_input_data)
#     assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
#     # Find the default inifile
#     fname_default_ini = data_folder / "model_choice.ini"
#     assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
#     # Get input variables
#     input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
#     # We only want disk_radius_outer
#     disk_radius_outer = input_variables["disk_radius_outer"]
#     # Construct productspace
#     test_product_space = named_product(
#         disk_model_name         = DISK_MODEL_NAMES,
#         smbh_mass               = SMBH_MASSES,
#         disk_alpha_viscosity    = DISK_ALPHA_VISCOSITIES,
#         disk_bh_eddington_ratio = DISK_BH_EDDINGTON_RATIOS,
#     )
#     # Loop tests
#     for test_config in test_product_space:
#         # Run pAGN
#         disk_surf_dens_func, disk_aspect_ratio_func, disk_model_properties, bonus_structures = \
#             construct_disk_pAGN(
#                 test_config.disk_model_name,
#                 test_config.smbh_mass,
#                 disk_radius_outer,
#                 test_config.disk_alpha_viscosity,
#                 test_config.disk_bh_eddington_ratio,
#             )
#     if verbose:
#         print("  pass!")
#
# def test_construct_disk_interp(
#     verbose=True,
#     ):
#     """Test mcfacts.inputs.ReadInputs.construct_disk_interp
#
#     Parameters
#     ----------
#     verbose : bool
#         Verbose output
#     """
#     if verbose:
#         print("Testing construct_disk_interp")
#     # Check that the data folder exists
#     data_folder = impresources.files(mcfacts_input_data)
#     assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
#     # Find the default inifile
#     fname_default_ini = data_folder / "model_choice.ini"
#     assert isfile(fname_default_ini), "Cannot find %s"%(fname_default_ini)
#     # Get input variables
#     input_variables = ReadInputs_ini(fname_default_ini, verbose=verbose)
#     # We want a few things
#     smbh_mass = input_variables["smbh_mass"]
#     disk_radius_outer = input_variables["disk_radius_outer"]
#     disk_alpha_viscosity = input_variables["disk_alpha_viscosity"]
#     disk_bh_eddington_ratio = input_variables["disk_bh_eddington_ratio"]
#     disk_radius_max_pc = input_variables["disk_radius_max_pc"]
#     # Construct productspace
#     test_product_space = named_product(
#         disk_model_name = DISK_MODEL_NAMES,
#         flag_use_pagn   = FLAG_USE_PAGN,
#     )
#     # Loop tests
#     for test_config in test_product_space:
#         # Run function
#         disk_surf_dens_func, disk_aspect_ratio_func = construct_disk_interp(
#             smbh_mass,
#             disk_radius_outer,
#             test_config.disk_model_name,
#             disk_alpha_viscosity,
#             disk_bh_eddington_ratio,
#             disk_radius_max_pc=disk_radius_max_pc,
#             flag_use_pagn=test_config.flag_use_pagn,
#             verbose=verbose,
#         )
#     if verbose:
#         print("  pass!")
#
# ######## Main ########
# def main():
#     test_input_types()
#     test_ReadInputs_ini()
#     test_load_disk_arrays()
#     test_construct_disk_direct()
#     test_construct_disk_pAGN()
#     test_construct_disk_interp()
#
# ######## Execution ########
# if __name__ == "__main__":
#     main()
