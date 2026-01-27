# read from tau_inputs.csv

import numpy as np
import pandas as pd
import scipy
import ast
from importlib import resources as impresources
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.physics.disk_capture import tau_ecc_dyn, tau_ecc_dyn_optimized, tau_inc_dyn, tau_inc_dyn_optimized

# parse array out of the csv file
def parse_array(cell):
    if isinstance(cell, str) and cell.startswith('['):
        cleaned = cell.strip('[]')
        return np.fromstring(cleaned, sep=' ')
    return cell

# parse scalar value out of csv file
def parse_value(cell):
    if isinstance(cell, str):
        # check if it's an array
        if '[' in cell:
            cleaned = cell.replace('[', '').replace(']', '')
            return np.fromstring(cleaned, sep=' ')
        
        # try to extract number from string with units (e.g., "147662503805.01248 m")
        try:
            # split by whitespace and take the first part (the number)
            numeric_part = cell.split()[0]
            return float(numeric_part)
        except (ValueError, IndexError):
            return cell
    return cell

# load 0 and 1 to construct disk_surf_desn_func_log
spline = pd.read_csv('tests/tau_spline.csv', header=None)

spline_parsed = spline.map(parse_array)
first_row = spline_parsed.iloc[0]
x_data = first_row[0]  # first array
y_data = first_row[1]  # second array

# construct the density function using the spline
disk_surf_dens_func_log = scipy.interpolate.CubicSpline(
    np.log(x_data), np.log(y_data))
disk_surf_density_func = lambda x, f=disk_surf_dens_func_log: np.exp(f(np.log(x)))

# now read from tau_inputs.csv and run
inputs = pd.read_csv("tests/tau_inputs.csv", header=None)

for _, row in inputs.iterrows():
    smbh_mass = row[0] # scalar
    disk_bh_retro_orbs_a = parse_array(row[1]) 
    disk_bh_retro_masses = parse_array(row[2]) 
    disk_bh_retro_arg_periapse = parse_array(row[3])
    disk_bh_retro_orbs_ecc = parse_array(row[4])
    disk_bh_retro_orbs_inc = parse_array(row[5])
    # r_g_in_meters = parse_value(row[7])
    r_g_in_meters = None

    tau_e_orig, tau_a_orig = tau_ecc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
                                               disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc,
                                               disk_surf_density_func, r_g_in_meters)
    tau_e_opt, tau_a_opt = tau_ecc_dyn_optimized(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
                                               disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc,
                                               disk_surf_density_func, r_g_in_meters)

    assert(np.allclose(tau_e_orig, tau_e_opt, rtol=1e9))
    assert(np.allclose(tau_a_orig, tau_a_opt, rtol=1e9))

    tau_inc_orig = tau_inc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
                                  disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc,
                                  disk_bh_retro_orbs_inc, disk_surf_density_func, r_g_in_meters)
    tau_inc_opt = tau_inc_dyn_optimized(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
                                  disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc,
                                  disk_bh_retro_orbs_inc, disk_surf_density_func, r_g_in_meters)

    assert(np.allclose(tau_inc_orig, tau_inc_opt, rtol=1e9))


