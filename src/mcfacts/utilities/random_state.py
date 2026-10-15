"""
Utilities utilizing randomization throughout McFACTS modules.
"""
import uuid
from contextlib import contextmanager

import numpy as np
from numpy.random import Generator


def uuid_provider(random_generator: Generator) -> uuid.UUID:
    """
    Generates a random UUID (version 4) using a specified random number generator.

    Args:
        random_generator (Generator): A random number generator from numpy used to generate random bytes.

    Returns:
        uuid.UUID: A randomly generated UUID (version 4) based on random bytes provided by the generator.
    """
    return uuid.UUID(bytes=random_generator.bytes(16), version=4)


@contextmanager
def global_numpy_random_from(random_generator: Generator):
    """
    Temporarily set the global random functions in numpy (np.random.uniform, etc.) to use a defined Generator.

    Use this to wrap third-party code that make calls to np.random, so all random draws come from the
    same stream as the rest of the simulation.

    The previous global state is restored on exit.

    Args:
        random_generator (Generator): A random number generator from numpy whose bit generator is shared.
    """
    previous_bit_generator = np.random.get_bit_generator()

    np.random.set_bit_generator(random_generator.bit_generator)

    try:
        yield
    finally:
        np.random.set_bit_generator(previous_bit_generator)
