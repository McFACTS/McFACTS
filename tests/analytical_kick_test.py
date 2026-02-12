import numpy as np
import pandas as pd
import ast
from mcfacts.physics.analytical_velocity import analytical_kick_velocity, analytical_kick_velocity_opt
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

def analytical_velocity(
    mass_1,
    mass_2,
    spin_1,
    spin_2,
    spin_angle_1,
    spin_angle_2
):

    rng.seed(seed=483445)
    original = analytical_kick_velocity(
        mass_1,
        mass_2,
        spin_1,
        spin_2,
        spin_angle_1,
        spin_angle_2
    )

    rng.seed(seed=483445)
    optimized = analytical_kick_velocity_opt(
        mass_1,
        mass_2,
        spin_1,
        spin_2,
        spin_angle_1,
        spin_angle_2
    )

    assert(np.allclose(original, optimized, rtol=1e-5))

def test_analytical_velocity():
    # now read from jet_inputs.csv and run
    inputs = pd.read_csv("tests/analytical_inputs.csv", header=None)

    for _, row in inputs.iterrows():
        mass_1 = parse_array(row[0])
        mass_2 = parse_array(row[1])
        spin_1 = parse_array(row[2])
        spin_2 = parse_array(row[3])
        spin_angle_1 = parse_array(row[4])
        spin_angle_2 = parse_array(row[5])

        analytical_velocity(
            mass_1,
            mass_2,
            spin_1,
            spin_2,
            spin_angle_1,
            spin_angle_2
        )

if __name__ == "__main__":
    test_analytical_velocity()


