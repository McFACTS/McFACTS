import time
import numpy as np

import astropy.units as u
import astropy.constants as const

from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.dynamics import circular_singles_encounters_prograde, circular_singles_encounters_prograde_sweep
    
AXIS_TOLERANCE =  0.0000001
ECCENTRICITY_TOLERANCE = 0.0001 # the eccentricities are much less reliable than the axes

def generate_data(N: int, circ_proportion: float, rng: np.random.Generator):
    """Generates random mock data for the simulation functions."""
    # Mock physical constants
    mock_params = {
        "smbh_mass": 1.0e8,
        "timestep_duration_yr": 1.0e4,
        "disk_bh_pro_orb_ecc_crit": 0.1,
        "delta_energy_strong": 0.11, # setting it to greater than disk_bh_pro... to avoid bug
        "disk_radius_outer": 20000.0
    }

    # Generate random BH properties
    num_circ = int(N * circ_proportion)
    num_ecc = N - num_circ

    # Generate eccentricities to match the desired C/E proportion
    ecc_crit = mock_params["disk_bh_pro_orb_ecc_crit"]
    e_circ = rng.uniform(0, ecc_crit, size=num_circ)
    e_ecc = rng.uniform(ecc_crit * 1.01, 0.8, size=num_ecc) # Ensure e > e_crit
    
    # Combine and shuffle to avoid any ordering bias
    all_eccs = np.concatenate((e_circ, e_ecc))
    rng.shuffle(all_eccs)
    
    mock_params["disk_bh_pro_orbs_ecc"] = all_eccs
    mock_params["disk_bh_pro_masses"] = rng.uniform(5, 50, size=N)
    # Ensure semi-major axis is always positive and within the disk
    mock_params["disk_bh_pro_orbs_a"] = rng.uniform(
        100, mock_params["disk_radius_outer"] * 0.95, size=N
    )
    
    return mock_params


def run_benchmark(N: int, circ_proportion: float):
    """Runs a single benchmark test for a given N and C/E proportion."""
    print(f"--- Testing: N={N}, C/E Proportion={circ_proportion:.2f} ---")
    
    # Use a fixed seed for the main data generation
    data_rng = np.random.default_rng(seed=123)
    data = generate_data(N, circ_proportion, data_rng)

    # We need separate, identically-seeded RNGs for the functions themselves
    # to ensure they use the same random numbers internally for the test.
    rng1 = np.random.default_rng(seed=456)
    rng2 = np.random.default_rng(seed=456)

    # --- Run Original Function ---
    # Important: Copy data as the functions modify arrays in-place
    data_for_orig = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    start_time = time.perf_counter()
    a_orig, ecc_orig = circular_singles_encounters_prograde(**data_for_orig, rng_here=rng1)
    time_orig = time.perf_counter() - start_time
    print(f"Original took:   {time_orig:.4f} seconds")

    # --- Run Optimized Function ---
    data_for_opt = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    start_time = time.perf_counter()
    a_opt, ecc_opt = circular_singles_encounters_prograde_sweep(**data_for_opt, rng_here=rng2)
    time_opt = time.perf_counter() - start_time
    print(f"Optimized took:  {time_opt:.4f} seconds")

    # --- Correctness ---
    correct_a = np.allclose(a_orig, a_opt, rtol = AXIS_TOLERANCE)
    correct_ecc = np.allclose(ecc_orig, ecc_opt, rtol = ECCENTRICITY_TOLERANCE) # the eccentricities are much less reliable than the axes

    assert correct_a, \
        "The returned semi-major axes were not within specified tolerances"
    
    assert correct_ecc, \
        "The returned eccentricities were not within specified tolerances"

    # --- Speedup ---
    # we should at least see parity, and never see a considerable slowdown
    # otherwise, we haven't set the length check correctly and we're using an ill-suited algorithm

    speedup = time_orig / time_opt if time_opt > 0 else float('inf')
    assert speedup > 0.80, \
        "We see a considerable slowdown"

def test_circular_singles_encounters_parity():
    # Define the set of test cases to run
    test_cases = [
        (10, 0.5),
        (20, 0.5),
        (30, 0.5),
        (40, 0.5),
        (50, 0.5),
        (100, 0.5),
        (250, 0.5),   
        (500, 0.5),
        (1000, 0.5),
        (10, 0.1),
        (20, 0.1),
        (30, 0.1),
        (40, 0.1),
        (50, 0.1),
        (100, 0.1),
        (250, 0.1),   
        (500, 0.1),
        (1000, 0.1),
        (10, 0.9),
        (20, 0.9),
        (30, 0.9),
        (40, 0.9),
        (50, 0.9),
        (100, 0.9),
        (250, 0.9),   
        (500, 0.9),
        (1000, 0.9),
    ]

    for N, prop in test_cases:
        run_benchmark(N, prop)

######## Main ########
def main():
    test_circular_singles_encounters_parity()

######## Execution ########
if __name__ == "__main__":
    main()
