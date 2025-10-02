"""
Controls the global random state used throughout McFACTS simulations.
"""
import sys

import numpy as np
from numpy.random import Generator

this = sys.modules[__name__]

this.default_seed = 1

this.rng = Generator(np.random.Philox(seed=this.default_seed))


def reset_random(seed) -> Generator:
    this.rng.bit_generator.state = Generator(np.random.Philox(seed=seed)).bit_generator.state

    return this.rng


def call_count() -> int:
    bit_generator = this.rng.bit_generator

    if not isinstance(bit_generator, np.random._philox.Philox):
        return -1

    buffer_pos = bit_generator.state["buffer_pos"]
    counter = bit_generator.state["state"]["counter"]

    if counter[0] == 0:
        return 0

    return max(int(counter[0]) - 1, 0) * 4 + buffer_pos