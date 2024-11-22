"""Module for handling simple GR orbital evolution (Peters 1964)

Contain functions for orbital evolution and converting between
    units of r_g and SI units

This module is its own module because it does not import other parts
    of mcfacts, in order to avoid circular imports
"""
######## Imports ########

import numpy as np
import scipy

from astropy import units as astropy_units
from astropy import constants as astropy_constants

######## Functions ########


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
    c = astropy_constants.c.to('m/s').value
    G = astropy_constants.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_initial = sep_initial.to('m').value
    sep_final = sep_final.to('m').value
    # Set up the constant as a single float
    const = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate the time
    time_of_shrinkage = ((sep_initial ** 4) - (sep_final ** 4)) / 4 / beta_arr
    # Assign units
    time_of_shrinkage = time_of_shrinkage * astropy_units.s
    return time_of_shrinkage


def orbital_separation_evolve(mass_1, mass_2, sep_initial, evolve_time):
    """Calculates the final separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy.units.quantity.Quantity
        Mass of object 1
    mass_2 : astropy.units.quantity.Quantity
        Mass of object 2
    sep_initial : astropy.units.quantity.Quantity
        Initial separation of two bodies
    evolve_time : astropy.units.quantity.Quantity
        Time to evolve GW orbit

    Returns
    -------
    sep_final : astropy.units.quantity.Quantity
        Final separation [m] of two bodies
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s').value
    G = astropy_constants.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_initial = sep_initial.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_initial ** 4) - (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_final = np.zeros_like(sep_initial)
    sep_final[quantity > 0] = np.sqrt(np.sqrt(quantity[quantity > 0]))
    return sep_final * astropy_units.m


def orbital_separation_evolve_reverse(mass_1, mass_2, sep_final, evolve_time):
    """Calculates the initial separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy.units.quantity.Quantity
        Mass of object 1
    mass_2 : astropy.units.quantity.Quantity
        Mass of object 2
    sep_final : astropy.units.quantity.Quantity
        Final separation of two bodies
    evolve_time : astropy.units.quantity.Quantity
        Time to evolve GW orbit

    Returns
    -------
    sep_initial : astropy.units.quantity.Quantity
        Initial separation [m] of two bodies
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s').value
    G = astropy_constants.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_final = sep_final.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_final ** 4) + (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_initial = np.sqrt(np.sqrt(quantity))
    return sep_initial * astropy_units.m


def si_from_r_g(smbh_mass, distance_rg):
    """Calculate the SI distance from r_g

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    distance_rg : array_like
        Distances [r_{g,SMBH}]

    Returns
    -------
    distance : numpy.ndarray
        Distance in SI with :obj:`astropy.units.quantity.Quantity` type
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s')
    G = astropy_constants.G.to('m^3/(kg s^2)')
    # Assign units to smbh mass
    if hasattr(smbh_mass, 'unit'):
        smbh_mass = smbh_mass.to('solMass')
    else:
        smbh_mass = smbh_mass * astropy_units.solMass
    # convert smbh mass to kg
    smbh_mass = smbh_mass.to('kg')
    # Calculate r_g in SI
    r_g = G*smbh_mass/(c ** 2)
    # Calculate distance
    distance = distance_rg * r_g
    return distance


def r_g_from_units(smbh_mass, distance):
    """Calculate the SI distance from r_g

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    distance_rg : astropy.units.quantity.Quantity
        Distances

    Returns
    -------
    distance_rg : numpy.ndarray
        Distances [r_g]
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s')
    G = astropy_constants.G.to('m^3/(kg s^2)')
    # Assign units to smbh mass
    if hasattr(smbh_mass, 'unit'):
        smbh_mass = smbh_mass.to('solMass')
    else:
        smbh_mass = smbh_mass * astropy_units.solMass
    # convert smbh mass to kg
    smbh_mass = smbh_mass.to('kg')
    # Calculate r_g in SI
    r_g = G*smbh_mass/(c ** 2)
    # Calculate distance
    distance_rg = distance / r_g
    return distance_rg
