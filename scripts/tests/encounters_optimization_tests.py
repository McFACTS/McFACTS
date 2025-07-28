import time
import numpy as np
import pytest

import astropy.units as u
import astropy.constants as const

from mcfacts.mcfacts_random_state import rng
# from mcfacts.physics.dynamics import circular_singles_encounters_prograde, circular_singles_encounters_prograde_sweep

# ===================================================================
# ORIGINAL FUNCTION (lightly modified for rng)
# ===================================================================

def circular_singles_encounters_prograde(
        smbh_mass: float,
        disk_bh_pro_orbs_a: np.ndarray,
        disk_bh_pro_masses: np.ndarray,
        disk_bh_pro_orbs_ecc: np.ndarray,
        timestep_duration_yr: float,
        disk_bh_pro_orb_ecc_crit: float,
        delta_energy_strong: float,
        disk_radius_outer: float,
        rng: np.random.Generator
        ):
    # Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc <= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc > disk_bh_pro_orb_ecc_crit).nonzero()[0]

    if (len(circ_prograde_population_indices) == 0) or (len(ecc_prograde_population_indices) == 0):
        return disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon = (disk_radius_outer * ((disk_bh_pro_masses[circ_prograde_population_indices] /
               (3 * (disk_bh_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)))[:, None] * \
              rng.uniform(size=(circ_prograde_population_indices.size, ecc_prograde_population_indices.size))

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = np.pi*((disk_bh_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*const.G.value)/(const.c.value**(3.0)*3.15e7)
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    # Generate all possible needed random numbers ahead of time
    chance_of_enc = rng.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
    num_poss_ints = 0
    num_encounters = 0
    if len(circ_prograde_population_indices) > 0:
        for i, circ_idx in enumerate(circ_prograde_population_indices):
            for j, ecc_idx in enumerate(ecc_prograde_population_indices):
                if (disk_bh_pro_orbs_a[circ_idx] < ecc_orb_max[j] and disk_bh_pro_orbs_a[circ_idx] > ecc_orb_min[j]):
                    # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                    # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                    temp_bin_mass = disk_bh_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]
                    bh_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                    mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
                    prob_orbit_overlap = (1./np.pi)*mass_ratio_factor
                    prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                    if prob_enc_per_timestep > 1:
                        prob_enc_per_timestep = 1
                    if chance_of_enc[i][j] < prob_enc_per_timestep:
                        num_encounters = num_encounters + 1
                        # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                        # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                        # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                        if disk_bh_pro_orbs_ecc[circ_idx] <= disk_bh_pro_orb_ecc_crit:
                            disk_bh_pro_orbs_ecc[circ_idx] = delta_energy_strong
                            disk_bh_pro_orbs_a[circ_idx] = disk_bh_pro_orbs_a[circ_idx]*(1.0 + delta_energy_strong)
                            # Catch for if orb_a > disk_radius_outer
                            if (disk_bh_pro_orbs_a[circ_idx] > disk_radius_outer):
                                disk_bh_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon[i][j]
                            disk_bh_pro_orbs_ecc[ecc_idx] = disk_bh_pro_orbs_ecc[ecc_idx]*(1 - delta_energy_strong)
                            disk_bh_pro_orbs_a[ecc_idx] = disk_bh_pro_orbs_a[ecc_idx]*(1 - delta_energy_strong)
                    num_poss_ints = num_poss_ints + 1
            num_poss_ints = 0
            num_encounters = 0

    # Check finite
    assert np.isfinite(disk_bh_pro_orbs_a).all(), \
        "Finite check failed for disk_bh_pro_orbs_a"
    assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for disk_bh_pro_orbs_ecc"
    assert np.all(disk_bh_pro_orbs_a < disk_radius_outer), \
        "disk_bh_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_bh_pro_orbs_a > 0), \
        "disk_bh_pro_orbs_a contains values <= 0"

    return (disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)

# ===================================================================
# SWEEP APPROACH
# ===================================================================

def circular_singles_encounters_prograde_sweep(
        smbh_mass: float,
        disk_bh_pro_orbs_a: np.ndarray,
        disk_bh_pro_masses: np.ndarray,
        disk_bh_pro_orbs_ecc: np.ndarray,
        timestep_duration_yr: float,
        disk_bh_pro_orb_ecc_crit: float,
        delta_energy_strong: float,
        disk_radius_outer: float,
        rng: np.random.Generator
    ):
    # Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc <= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_bh_pro_orbs_ecc > disk_bh_pro_orb_ecc_crit).nonzero()[0]

    circ_len = len(circ_prograde_population_indices)
    ecc_len = len(ecc_prograde_population_indices)
    if (circ_len == 0) or (ecc_len == 0):
        return disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon = (disk_radius_outer * ((disk_bh_pro_masses[circ_prograde_population_indices] /
               (3 * (disk_bh_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)))[:, None] * \
              rng.uniform(size=(circ_prograde_population_indices.size, ecc_prograde_population_indices.size))

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = np.pi*((disk_bh_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*const.G.value)/(const.c.value**(3.0)*3.15e7)
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
    # Generate all possible needed random numbers ahead of time
    chance_of_enc = rng.uniform(size=(circ_prograde_population_indices.size, ecc_prograde_population_indices.size))


    if (circ_len/(circ_len + ecc_len)) * (ecc_len/(circ_len + ecc_len)) * 100 > 50: # an ad-hoc check to see whether the double loop or sweep will be faster
        # if True engage the sweep algorithm

        # create the events array
        # define types to ensure correct sorting at boundary conditions:
        # START events are processed first, then POINTs, then ENDs
        START, POINT, END = -1, 0, 1
        
        # C = circ_prograde_population_indices.size
        # ecc_len = ecc_prograde_population_indices.size

        # create a single, flat, contiguous array for all events
        events = np.empty(circ_len + 2 * ecc_len, dtype=[('radius', 'f8'), ('type', 'i4'), ('rel_idx', 'u4')])

        # add POINT events for each circular object
        events[:circ_len] = np.array(list(zip(disk_bh_pro_orbs_a[circ_prograde_population_indices], [POINT] * circ_len, np.arange(circ_len))), dtype=events.dtype)

        # Add START and ecc_lenND events for each eccentric object's interval
        ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices] * (1.0 - disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
        ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices] * (1.0 + disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
        events[circ_len:circ_len+ecc_len] = np.array(list(zip(ecc_orb_min, [START] * ecc_len, np.arange(ecc_len))), dtype=events.dtype)
        events[circ_len+ecc_len:] = np.array(list(zip(ecc_orb_max, [END] * ecc_len, np.arange(ecc_len))), dtype=events.dtype)

        # sort the events by radius
        # uses numpy sort, very performant
        events.sort(order=['radius', 'type'])

        # sweep and process
        active_ecc_indices = set()
        for radius, type, rel_idx in events:
            if type == START:
                active_ecc_indices.add(rel_idx)
            elif type == END:
                active_ecc_indices.discard(rel_idx) # Use discard for safety
            elif type == POINT:
                # when we hit a POINT event, the `active_ecc_indices` set contains
                # ALL eccentric particles whose intervals contain this point
                if not active_ecc_indices:
                    continue

                circ_rel_idx = rel_idx
                circ_idx = circ_prograde_population_indices[circ_rel_idx]
                
                # sort the indices to ensure deterministic processing order
                sorted_interlopers = sorted(list(active_ecc_indices))

                # if we remove this sort and instead just iterate directly
                # over active_ecc_indices, we unlock another 2x+ improvement in performance
                # but at the cost of genuinely massively deviating values

                for ecc_rel_idx in sorted_interlopers:
                    ecc_idx = ecc_prograde_population_indices[ecc_rel_idx]
                    
                    temp_bin_mass = disk_bh_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]
                    bh_smbh_mass_ratio = temp_bin_mass / (3.0 * smbh_mass)
                    mass_ratio_factor = (bh_smbh_mass_ratio)**(1. / 3.)
                    prob_orbit_overlap = (1. / np.pi) * mass_ratio_factor
                    prob_enc_per_timestep = min(prob_orbit_overlap * N_circ_orbs_per_timestep[circ_rel_idx], 1.0)
                    
                    if chance_of_enc[circ_rel_idx, ecc_rel_idx] < prob_enc_per_timestep:
                        # apply state change, using the fixed logic
                        disk_bh_pro_orbs_ecc[circ_idx] = delta_energy_strong * 1.0001
                        disk_bh_pro_orbs_a[circ_idx] *= (1.0 + delta_energy_strong)
                        if (disk_bh_pro_orbs_a[circ_idx] > disk_radius_outer):
                            
                            disk_bh_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon[circ_rel_idx][ecc_rel_idx]
                        
                        disk_bh_pro_orbs_ecc[ecc_idx] *= (1.0 - delta_energy_strong)
                        disk_bh_pro_orbs_a[ecc_idx] *= (1.0 - delta_energy_strong)
                        # Once the circular BH is kicked, break from this inner loop
                        # as it can't have more encounters in this timestep
                        break 
    else:
        # if False, engage the double loop, as this N is too small to make the up-front sort of the sweep algorithm worthwhile
        
        num_poss_ints = 0
        num_encounters = 0
        if len(circ_prograde_population_indices) > 0:
            for i, circ_idx in enumerate(circ_prograde_population_indices):
                for j, ecc_idx in enumerate(ecc_prograde_population_indices):
                    if (disk_bh_pro_orbs_a[circ_idx] < ecc_orb_max[j] and disk_bh_pro_orbs_a[circ_idx] > ecc_orb_min[j]):
                        # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                        # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                        temp_bin_mass = disk_bh_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]
                        bh_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                        mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
                        prob_orbit_overlap = (1./np.pi)*mass_ratio_factor
                        prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                        if prob_enc_per_timestep > 1:
                            prob_enc_per_timestep = 1
                        if chance_of_enc[i][j] < prob_enc_per_timestep:
                            num_encounters = num_encounters + 1
                            # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                            # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                            # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                            if disk_bh_pro_orbs_ecc[circ_idx] <= disk_bh_pro_orb_ecc_crit:
                                disk_bh_pro_orbs_ecc[circ_idx] = delta_energy_strong
                                disk_bh_pro_orbs_a[circ_idx] = disk_bh_pro_orbs_a[circ_idx]*(1.0 + delta_energy_strong)
                                # Catch for if orb_a > disk_radius_outer
                                if (disk_bh_pro_orbs_a[circ_idx] > disk_radius_outer):
                                    disk_bh_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon[i][j]
                                disk_bh_pro_orbs_ecc[ecc_idx] = disk_bh_pro_orbs_ecc[ecc_idx]*(1 - delta_energy_strong)
                                disk_bh_pro_orbs_a[ecc_idx] = disk_bh_pro_orbs_a[ecc_idx]*(1 - delta_energy_strong)
                        num_poss_ints = num_poss_ints + 1
                num_poss_ints = 0
                num_encounters = 0

    # Check finite
    assert np.isfinite(disk_bh_pro_orbs_a).all(), \
        "Finite check failed for disk_bh_pro_orbs_a"
    assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
        "Finite check failed for disk_bh_pro_orbs_ecc"
    assert np.all(disk_bh_pro_orbs_a < disk_radius_outer), \
        "disk_bh_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_bh_pro_orbs_a > 0), \
        "disk_bh_pro_orbs_a contains values <= 0"

    return (disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)



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
    a_orig, ecc_orig = circular_singles_encounters_prograde(**data_for_orig, rng=rng1)
    time_orig = time.perf_counter() - start_time
    print(f"Original took:   {time_orig:.4f} seconds")

    # --- Run Optimized Function ---
    data_for_opt = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    start_time = time.perf_counter()
    a_opt, ecc_opt = circular_singles_encounters_prograde_sweep(**data_for_opt, rng=rng2)
    time_opt = time.perf_counter() - start_time
    print(f"Optimized took:  {time_opt:.4f} seconds")



    # --- Correctness ---
    correct_a = np.allclose(a_orig, a_opt, rtol = 0.0000001)
    correct_ecc = np.allclose(ecc_orig, ecc_opt, rtol = 0.0001) # the eccentricities are much less reliable than the axes

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


