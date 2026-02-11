import numpy as np
import pandas as pd
import ast
from importlib import resources as impresources
from mcfacts.inputs import ReadInputs
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.setup.setupdiskblackholes import setup_disk_blackholes_location_NSC_powerlaw, setup_disk_blackholes_location_NSC_powerlaw_optimized
from mcfacts.mcfacts_random_state import rng

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


def powerlaw(
    disk_bh_num,
    disk_radius_outer,
    disk_inner_stable_circ_orb,
    smbh_mass,
    nsc_radius_crit,
    nsc_density_index_inner,
    nsc_density_index_outer,
):
    # setting seed
    rng.seed(seed=483445)
    original = setup_disk_blackholes_location_NSC_powerlaw(
        disk_bh_num,
        disk_radius_outer,
        disk_inner_stable_circ_orb,
        smbh_mass,
        nsc_radius_crit,
        nsc_density_index_inner,
        nsc_density_index_outer,
        volume_scaling=True
    )
    # setting seed
    rng.seed(seed=483445)
    optimized = setup_disk_blackholes_location_NSC_powerlaw_optimized(
        disk_bh_num,
        disk_radius_outer,
        disk_inner_stable_circ_orb,
        smbh_mass,
        nsc_radius_crit,
        nsc_density_index_inner,
        nsc_density_index_outer,
        volume_scaling=True
    )

    assert(np.allclose(original, optimized, rtol=1e-9))

def test_powerlaw():
    # now read from jet_inputs.csv and run
    inputs = pd.read_csv("tests/powerlaw_inputs.csv", header=None)

    for _, row in inputs.iterrows():
        disk_bh_num = row[0] # scalar
        disk_radius_outer = row[1]
        disk_inner_stable_circ_orb = row[2] 
        smbh_mass = row[3]
        nsc_radius_crit = row[4]
        nsc_density_index_inner = row[5]
        nsc_density_index_outer = row[6]

        powerlaw(
            disk_bh_num,
            disk_radius_outer,
            disk_inner_stable_circ_orb,
            smbh_mass,
            nsc_radius_crit,
            nsc_density_index_inner,
            nsc_density_index_outer,
        )

if __name__ == "__main__":
    test_powerlaw()
