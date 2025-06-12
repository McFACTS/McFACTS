import numpy as np
from astropy import constants as const, units as u


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
    assert np.all(distance > 0).all(), \
        "distance contains values <= 0"

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
