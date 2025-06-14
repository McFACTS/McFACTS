"""
Controls the global random state used throughout McFACTS simulations.
"""
import uuid

import numpy as np
from numpy.random import Generator

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


def uuid_provider(randomGenerator: Generator) -> uuid.UUID:
    """
    Generates a random UUID (version 4) using a specified random number generator.

    Args:
        randomGenerator (Generator): A random number generator from numpy used to generate random bytes.

    Returns:
        uuid.UUID: A randomly generated UUID (version 4) based on random bytes provided by the generator.
    """
    return uuid.UUID(bytes=randomGenerator.bytes(16), version=4)
