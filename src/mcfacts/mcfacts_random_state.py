"""
Controls the global random state used throughout McFACTS simulations.
"""
import numpy as np
from numpy.random import Generator

default_seed = 1

rng = Generator(np.random.Philox(seed=default_seed))


def reset_random(seed) -> Generator:
    return Generator(np.random.Philox(seed=seed))


def call_count() -> int:
    bit_generator = rng.bit_generator

    if not isinstance(bit_generator, np.random._philox.Philox):
        return -1

    buffer_pos = bit_generator.state["buffer_pos"]
    counter = bit_generator.state["state"]["counter"]

    if counter[0] == 0:
        return 0

    return max(int(counter[0]) - 1, 0) * 4 + buffer_pos
