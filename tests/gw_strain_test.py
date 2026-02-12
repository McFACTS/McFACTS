import numpy as np
import pandas as pd
import ast
from importlib import resources as impresources
from mcfacts.inputs import ReadInputs
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.physics.gw import gw_strain_freq, gw_strain_freq_optimized

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

def gw_strain(
    mass_1, 
    mass_2, 
    obj_sep, 
    timestep_duration_yr, 
    old_gw_freq, 
    smbh_mass, 
    agn_redshift, 
    flag_include_old_gw_freq):

    original = gw_strain_freq(
        mass_1, 
        mass_2, 
        obj_sep, 
        timestep_duration_yr, 
        old_gw_freq, 
        smbh_mass, 
        agn_redshift, 
        flag_include_old_gw_freq
    )

    optimized = gw_strain_freq_optimized(
        mass_1, 
        mass_2, 
        obj_sep, 
        timestep_duration_yr, 
        old_gw_freq, 
        smbh_mass, 
        agn_redshift, 
        flag_include_old_gw_freq
    )

    assert(np.allclose(original, optimized, rtol=1e-8))

def test_gw_strain():
    # now read from jet_inputs.csv and run
    inputs = pd.read_csv("tests/gw_strain_inputs.csv", header=None)

    for _, row in inputs.iterrows():
        mass_1 = parse_array(row[0])
        mass_2 = parse_array(row[1]) 
        obj_sep = parse_array(row[2])
        timestep_duration_yr = parse_value(row[3])
        old_gw_freq = parse_value(row[4])
        smbh_mass = parse_value(row[5])
        agn_redshift = parse_value(row[6])
        flag_include_old_gw_freq = parse_value(row[7])

        gw_strain(
            mass_1, 
            mass_2, 
            obj_sep, 
            timestep_duration_yr, 
            old_gw_freq, 
            smbh_mass, 
            agn_redshift, 
            flag_include_old_gw_freq
        )

if __name__ == "__main__":
    test_gw_strain()


