"""
Module for calculating the orbital and binary eccentricity damping.
"""

import numpy as np
from mcfacts.mcfacts_random_state import rng





def ionized_orb_ecc(num_bh, orb_ecc_max):
    """Calculate new eccentricity for each component of an ionized binary.

    Parameters
    ----------
    num_bh : int
        Number of BHs (num of ionized binaries * 2)
    orb_ecc_max : float
        Maximum allowed orb_ecc
    """
    orb_eccs = rng.uniform(low=0.0, high=orb_ecc_max, size=num_bh)

    return (orb_eccs)
