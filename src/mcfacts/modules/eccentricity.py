"""
Module for calculating the orbital and binary eccentricity damping.
"""
from numpy.random import Generator


def ionized_orb_ecc(num_bh, orb_ecc_max, random: Generator):
    """Calculate new eccentricity for each component of an ionized binary.

    Parameters
    ----------
    num_bh : int
        Number of BHs (num of ionized binaries * 2)
    orb_ecc_max : float
        Maximum allowed orb_ecc
    random : Generator
        Generator used to generate random numbers
    """
    orb_eccs = random.uniform(low=0.0, high=orb_ecc_max, size=num_bh)

    return (orb_eccs)
