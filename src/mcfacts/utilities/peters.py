"""
Orbital decay and gravitational wave emission calculations based on Peters & Mathews.

This module provides utility functions for computing the effects of gravitational
radiation on binary systems, following:

- Peters, P.C. & Mathews, J. (1963), Phys. Rev. 131, 435.
- Peters, P.C. (1964), Phys. Rev. 136, B1224.
"""

import numpy as np
from astropy import constants as const, units as u


def time_of_orbital_shrinkage(mass_1, mass_2, sep_initial, sep_final):
    """Calculates the GW time for orbital shrinkage

    Calculate the time it takes for two orbiting masses
    to shrink from an initial separation to a final separation (Peters)

    Parameters
    ----------
    mass_1 : astropy.units.quantity.Quantity
        Mass of object 1
    mass_2 : astropy.units.quantity.Quantity
        Mass of object 2
    sep_initial : astropy.units.quantity.Quantity
        Initial separation of two bodies
    sep_final : astropy.units.quantity.Quantity
        Final separation of two bodies

    Returns
    -------
    time_of_shrinkage : astropy.units.quantity.Quantity
        Time [s] of orbital shrinkage
    """
    # Calculate c and G in SI
    c = const.c.to('m/s').value
    G = const.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_initial = sep_initial.to('m').value
    sep_final = sep_final.to('m').value
    # Set up the constant as a single float
    const_G_c = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const_G_c * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate the time
    time_of_shrinkage = ((sep_initial ** 4) - (sep_final ** 4)) / 4 / beta_arr
    # Assign units
    time_of_shrinkage = time_of_shrinkage * u.s

    assert np.all(time_of_shrinkage >= 0), \
        "time_of_shrinkage contains values < 0"

    return time_of_shrinkage
