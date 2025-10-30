"""
Utilities for random used throughout McFACTS simulations.
"""
import uuid

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
