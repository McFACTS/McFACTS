#!/usr/bin/env python3
"""Test the SettingsHandler object"""
######## Imports ########
#### Standard Library ####
import concurrent.futures
from copy import deepcopy
import multiprocessing as mp
import pickle
import time

#### Third Party ####
import tqdm
import numpy as np

#### Local ####
from mcfacts.inputs.settings_manager import SettingsManager, DEFAULT_SETTINGS

######## Setup ########
N_PROC = 10

def monkey_hit_head(settings):
    assert isinstance(settings, SettingsManager)
    rng = np.random.Generator(np.random.Philox(seed=settings.seed))
    time.sleep(rng.random())
    return rng.choice([True, False])

######## Tests ########

def test_default_settings():
    """Test that calling setup_scaling does nothing if scaling is off"""
    # Define some settings
    live = SettingsManager()
    # Check they're all defaults
    for prop in DEFAULT_SETTINGS:
        assert getattr(live, prop.name) == prop.value, \
            f"{prop.name} has value {getattr(live, prop.name)}, " \
            f"but should be default ({prop.value})!"

def test_atomic_settings():
    # Define some settings
    live = SettingsManager()
    # Loop
    for key, value in live.settings_finals.items():
        assert deepcopy(value) == value

def test_pickle():
    # Define some settings
    live = SettingsManager()
    # Pickle them
    jar = pickle.loads(pickle.dumps(live))
    # Check they are equal
    assert live == jar
    # Check they are not the same
    assert live is not jar

def test_copy():
    # Define some settings
    live = SettingsManager()
    # Make a copy
    copy = live.copy()
    # Check they are equal
    assert live == copy
    # Check they are not the same
    assert live is not copy
    # Make another copy
    copy = live.copy({"seed":42})
    # Check that they are not equal
    assert live != copy

def test_process_pool():
    # Define some settings
    live = SettingsManager()
    # Initialize monkey pool
    monkeys = np.empty((N_PROC,), dtype=bool)
    # Try tqdm
    with tqdm.tqdm(
        total = N_PROC, desc=f"{N_PROC} Litte monkeys jumping on the bed"
        ) as pbar:
        # Recite the incantation
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROC) as executor:
            # Submit processes
            future_to_monkey = {executor.submit(
                monkey_hit_head,
                live.copy({"seed":live.seed-int(monkey)})
            ): monkey for monkey in np.arange(N_PROC)}
            # Catch processes
            for future in concurrent.futures.as_completed(future_to_monkey):
                # Get the index 
                monkey_see = future_to_monkey[future]
                # Assign the value
                monkeys[monkey_see] = future.result() # Monkey do
                # Update tqdm as processes complete
                pbar.update(1)
    # Assert Monkeys
    print(f"{monkeys.sum()} fell off and bumped their head!")
    assert monkeys.sum() == 8, "Default seed has probably changed."
                    
            
######## Main ########
def main():
    test_default_settings()
    test_atomic_settings()
    test_pickle()
    test_copy()
    test_process_pool()
    return 

######## Execution ########
if __name__ == "__main__":
    main()
