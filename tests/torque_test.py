import numpy as np
import pandas as pd
import ast
from importlib import resources as impresources
from mcfacts.inputs import ReadInputs
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.physics.migration import torque_mig_timescale, torque_mig_timescale_optimized
import astropy.units as u

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


def torque(
    smbh_mass,
    orbs_a,
    masses,
    orbs_ecc,
    orb_ecc_crit,
    migration_torque,
    r_g_in_meters
):
    original = torque_mig_timescale(
        smbh_mass,
        orbs_a,
        masses,
        orbs_ecc,
        orb_ecc_crit,
        migration_torque,
        r_g_in_meters=r_g_in_meters
    )
    optimized = torque_mig_timescale_optimized(
        smbh_mass,
        orbs_a,
        masses,
        orbs_ecc,
        orb_ecc_crit,
        migration_torque,
        r_g_in_meters=r_g_in_meters
    )

    assert(np.allclose(original, optimized, rtol=1e-6))

def test_torque():
    # now read from torque_inputs.csv and run
    inputs = pd.read_csv("tests/torque_inputs.csv", header=None)

    for _, row in inputs.iterrows():
        smbh_mass = parse_value(row[0])
        orbs_a = parse_array(row[1])
        masses = parse_array(row[2])
        orbs_ecc = parse_array(row[3]),
        orb_ecc_crit = parse_value(row[4]),
        migration_torque = parse_array(row[5])
        r_g_in_meters = parse_value(row[6])

        torque(
            smbh_mass,
            orbs_a,
            masses,
            orbs_ecc[0],
            orb_ecc_crit[0],
            migration_torque,
            r_g_in_meters * u.m
        )

if __name__ == "__main__":
    test_torque()
