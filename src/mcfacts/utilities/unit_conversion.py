import astropy.units
import numpy as np
from astropy import constants as const, units as u

def si_from_r_g(smbh_mass, distance_rg, r_g_defined=None):
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

    # if r_g_defined is not None, we just calculate the
    # distance from the provided value

    if r_g_defined is not None:
        r_g = r_g_defined
    else:
        # Calculate c and G in SI
        c = const.c.to('m/s')
        G = const.G.to('m^3/(kg s^2)')

        # Assign units to smbh mass
        if hasattr(smbh_mass, 'unit'):
            smbh_mass = smbh_mass.to('solMass')
        else:
            smbh_mass = smbh_mass * u.solMass

        # convert smbh mass to kg
        smbh_mass = smbh_mass.to('kg')

        # Calculate r_g in SI
        r_g = G*smbh_mass/(c ** 2)

    # Calculate distance
    distance = (distance_rg * r_g).to("meter")

    assert np.isfinite(distance).all(), \
        "Finite check failure: distance"
    assert np.all(distance >= 0).all(), \
        "distance contains values < 0"

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
    c = const.c.to('m/s')
    G = const.G.to('m^3/(kg s^2)')
    # Assign units to smbh mass
    if hasattr(smbh_mass, 'unit'):
        smbh_mass = smbh_mass.to('solMass')
    else:
        smbh_mass = smbh_mass * u.solMass
    # convert smbh mass to kg
    smbh_mass = smbh_mass.to('kg')
    # Calculate r_g in SI
    r_g = G*smbh_mass/(c ** 2)
    # Calculate distance
    distance_rg = distance.to("meter") / r_g

    # Check to make sure units are okay.
    assert u.dimensionless_unscaled == distance_rg.unit, \
        "distance_rg is not dimensionless. Check your input is a astropy Quantity, not an astropy Unit."
    assert np.isfinite(distance_rg).all(), \
        "Finite check failure: distance_rg"
    assert np.all(distance_rg > 0), \
        "Finite check failure: distance_rg"

    return distance_rg


def r_schwarzschild_of_m(mass):
    """Calculate the Schwarzschild radius from the mass of the object.

    Parameters
    ----------
    mass : numpy.ndarray or float
        Mass [Msun] of the object(s)

    Returns
    -------
    r_sch : numpy.ndarray
        Schwarzschild radius [m] with `astropy.units.quantity.Quantity`
    """

    # Assign units to mass
    if hasattr(mass, 'unit'):
        mass = mass.to('solMass')
    else:
        mass = mass * u.solMass

    r_sch = (2. * const.G * mass / (const.c ** 2)).to("meter")

    assert np.isfinite(r_sch).all(), \
        "Finite check failure: r_sch"
    assert np.all(r_sch > 0).all(), \
        "r_sch contains values <= 0"

    return (r_sch)


def initialize_r_g(smbh_mass):
    """Initilializes the r_g value in meters

    This function precomputes the r_g value which would otherwise be
    calculated anew with each call to si_from_r_g, using the input
    SMBH_MASS value.

    It mutates the input_variables dictionary in place, adding a
    r_g_in_meters key to it containing the r_g value in meters.

    Parameters
    ----------
    input_variables : dict
        Dictionary of input variables
    """
    # pre-calculating r_g from the provided smbh_mass
    c = const.c.to('m/s')
    G = const.G.to('m^3/(kg s^2)')

    # Assign units to smbh mass
    if hasattr(smbh_mass, 'unit'):
        smbh_mass = smbh_mass.to('solMass')
    else:
        smbh_mass = smbh_mass * u.solMass

    # convert smbh mass to kg
    smbh_mass = smbh_mass.to('kg')
    # Calculate r_g in SI
    r_g_in_meters = G * smbh_mass / (c ** 2)

    # adding r_g_in_meters to dictionary
    return r_g_in_meters