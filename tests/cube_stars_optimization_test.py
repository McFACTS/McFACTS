import time
import sys
import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy

SINGLETON_PROP = 0.15
AXIS_TOLERANCE =  0.0000001
ECCENTRICITY_TOLERANCE = 0.0001 # the eccentricities are much less reliable than the axes
SPEED_TOLERANCE = 0.80

from mcfacts.physics.dynamics import circular_singles_encounters_prograde_stars

def generate_data_stars(N: int, circ_proportion: float, singleton_proportion: float, rng: np.random.Generator):
    """Generates random mock data for the star simulation functions."""
    # Mock physical constants
    mock_params = {
        "smbh_mass": 1.0e8,
        "timestep_duration_yr": 1.0e4,
        "disk_star_pro_orb_ecc_crit": 0.1,
        "rstar_rhill_exponent": 2, # 2 is the default
        "delta_energy_strong_mu": 0.11,
        "delta_energy_strong_sigma": 0.05,
        "disk_radius_outer": 20000.0
    }

    # Generate random BH properties
    num_circ = int(N * circ_proportion)
    num_ecc = N - num_circ

    # Generate eccentricities to match the desired C/E proportion
    ecc_crit = mock_params["disk_star_pro_orb_ecc_crit"]
    e_circ = rng.uniform(0, ecc_crit, size=num_circ)
    e_ecc = rng.uniform(ecc_crit * 1.01, 0.8, size=num_ecc) # Ensure e > e_crit
    
    # Combine and shuffle to avoid any ordering bias
    all_eccs = np.concatenate((e_circ, e_ecc))
    rng.shuffle(all_eccs)
    
    mock_params["disk_star_pro_orbs_ecc"] = all_eccs
    mock_params["disk_star_pro_masses"] = rng.uniform(5, 50, size=N)
    # Ensure semi-major axis is always positive and within the disk
    mock_params["disk_star_pro_orbs_a"] = rng.uniform(
        100, mock_params["disk_radius_outer"] * 0.95, size=N
    )
    mock_params["disk_star_pro_radius"] = rng.uniform(1, 20, size = N)
    # mocking a certain proportion of these stars as singletons, non-repeating
    mock_params["disk_star_pro_id_nums"] = np.arange(0, N)
    
    return mock_params

def run_benchmark_stars(N: int, circ_proportion: float):
    """Runs a single benchmark test for a given N and C/E proportion."""
    print(f"--- Testing: N={N}, C/E Proportion={circ_proportion:.2f} ---")

    data_rng = np.random.default_rng(seed=123)
    data = generate_data_stars(N, circ_proportion, singleton_proportion = SINGLETON_PROP, rng = data_rng)

    # We need separate, identically-seeded RNGs for the functions themselves
    # to ensure they use the same random numbers internally for the test.
    rng1 = np.random.default_rng(seed=456)
    rng2 = np.random.default_rng(seed=456)
    

    # --- Run Original Function ---
    # Important: Copy data as the functions modify arrays in-place
    data_for_orig = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    start_time = time.perf_counter()
    a_orig, ecc_orig, id_nums_touch_orig, id_nums_unbound_orig, id_nums_flipped_rotation_orig = circular_singles_encounters_prograde_stars(**data_for_orig, rng_here=rng1, cardano=False)
    time_orig = time.perf_counter() - start_time
    print(f"Original took:   {time_orig:.4f} seconds")

    # --- Run Optimized Function ---
    data_for_opt = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    start_time = time.perf_counter()
    a_opt, ecc_opt, id_nums_touch_opt, id_nums_unbound_opt, id_nums_flipped_rotation_opt = circular_singles_encounters_prograde_stars(**data_for_opt, rng_here=rng2, cardano = True)
    time_opt = time.perf_counter() - start_time
    print(f"Optimized took:  {time_opt:.4f} seconds")

    # --- Correctness ---
    correct_a = np.allclose(a_orig, a_opt, rtol = AXIS_TOLERANCE)
    correct_ecc = np.allclose(ecc_orig, ecc_opt, rtol = ECCENTRICITY_TOLERANCE) # the eccentricities are much less reliable than the axes
    correct_touch = np.all(id_nums_touch_orig == id_nums_touch_opt)
    correct_unbound = np.all(id_nums_unbound_orig == id_nums_unbound_opt)
    correct_flipped_rotation = np.all(id_nums_flipped_rotation_orig == id_nums_flipped_rotation_opt)

    assert correct_a, \
        "The returned semi-major axes were not within specified tolerances"

    assert correct_ecc, \
        "The returned eccentricities were not within specified tolerances"

    assert correct_touch, \
        "The returned touch ids were not identical"

    assert correct_unbound, \
        "The returned unbound ids were not identical"

    assert correct_flipped_rotation, \
        "The returned flipped rotation ids were not identical"


    speedup = time_orig / time_opt if time_opt > 0 else float('inf')
    print(f"🚀 Speedup: {speedup:.2f}x")

    # --- Speedup ---
    # we should at least see parity, and never see a considerable slowdown
    # otherwise, we haven't set the length check correctly and we're using an ill-suited algorithm

    # speedup = time_orig / time_opt if time_opt > 0 else float('inf')
    # assert speedup > SPEED_TOLERANCE, \
    #     "We see a considerable slowdown"



if __name__ == "__main__":
    # Define the set of test cases to run
    test_cases = [
        # Test scaling with N at a 50/50 C/E split
        (10, 0.5),
        (20, 0.5),
        (30, 0.5),
        (40, 0.5),
        (50, 0.5),
        (100, 0.5),
        (250, 0.5),   
        (500, 0.5),
        (1000, 0.5),
        (2000, 0.5),
        (4000, 0.5),  
        (8000, 0.5),
        # (10000, 0.5),  
        # (15000, 0.5),
        # (20000, 0.5),
        # (25000, 0.5),
        # (30000, 0.5),
        # (40000, 0.5),
        # (60000, 0.5),
        # (80000, 0.5),
        # (100000, 0.5),
        # (160000, 0.5),
        # The C*E product is largest at 0.5, so these should be faster for the original
    ]

    for N, prop in test_cases:
        run_benchmark_stars(N, prop)
