"""
Controls the global random state used throughout McFACTS simulations.
"""
import numpy as np

default_seed = 1
# Initialize random number generator
rng = np.random.RandomState(seed=default_seed)


def reset_random(seed):
    """
    Set the seed of the random number generator

    Parameters
    ----------
    seed : int
        seed value to set

    Returns
    -------
    rng : numpy random number generator
        rng set with input value seed
    """
    rng.seed(seed)
