import numpy as np
import pandas as pd
import ast
from importlib import resources as impresources
from mcfacts.inputs import ReadInputs
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.physics.lum import jet_luminosity, jet_luminosity_opt

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


# defaults
# 100000000.0 50000.0 sirko_goodman 0.01 1.0 0.0 0 False

# constructing lambdas from defaults
disk_surface_density, disk_aspect_ratio, disk_opacity, disk_sound_speed, disk_density, disk_pressure_grad, disk_omega, disk_surface_density_log, temp_func, disk_dlog10surfdens_dlog10R_func, disk_dlog10temp_dlog10R_func, disk_dlog10pressure_dlog10R_func = ReadInputs.construct_disk_interp(100000000.0,
                                 50000.0,
                                 "sirko_goodman",
                                 0.01,
                                 1.0, 
                                 disk_radius_max_pc=0.0,
                                 flag_use_pagn=0,
                                 verbose=False
                                 )

def jet_lum(
    bh_mass_merged,
    bin_orbs_a, 
    disk_density,
    bh_spin_merged,
    bh_v_kick,
    disk_sound_speed
):
    original = bh_lum_jet = jet_luminosity(
        bh_mass_merged,
        bin_orbs_a, 
        disk_density,
        bh_spin_merged,
        bh_v_kick,
        disk_sound_speed)
    optimized = bh_lum_jet = jet_luminosity_opt(
        bh_mass_merged,
        bin_orbs_a, 
        disk_density,
        bh_spin_merged,
        bh_v_kick,
        disk_sound_speed)

    assert(np.allclose(original, optimized, rtol=1e-9))

def test_jet_lum():
    # now read from jet_inputs.csv and run
    inputs = pd.read_csv("tests/jet_inputs.csv", header=None)

    for _, row in inputs.iterrows():
        bh_mass_merged = parse_array(row[0]) # scalar
        bin_orbs_a = parse_array(row[1]) 
        bh_spin_merged = parse_array(row[2]) 
        bh_v_kick = parse_array(row[3])

        jet_lum(
            bh_mass_merged,
            bin_orbs_a, 
            disk_density,
            bh_spin_merged,
            bh_v_kick,
            disk_sound_speed
        )

if __name__ == "__main__":
    test_jet_lum()
