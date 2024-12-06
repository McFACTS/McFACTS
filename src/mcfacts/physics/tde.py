"""
Module for TDE specific calculations.
"""

import numpy as np
from astropy import units as astropy_units
from mcfacts.physics.point_masses import si_from_r_g


def check_tde_or_flip(star_retro_id_num, star_retro_mass, star_retro_log_radius, star_retro_orb_ecc, star_retro_orb_a, smbh_mass):
    """Retrograde stars that flip to prograde are TDEs if they are inside the disk's tidal disruption radius and have sufficiently high eccentricity.

    Parameters
    ----------
    star_retro_id_num : numpy.ndarray
        ID numbers of retrograde stars that may flip to prograde
    star_retro_mass : numpy.ndarray
        Star mass [Msun] with :obj:`float` type
    star_retro_log_radius : numpy.ndarray
        Star log radius/Rsun with :obj:`float` type
    star_retro_orb_ecc : numpy.ndarray
        Star orbital eccentricity
    star_retro_orb_a : numpy.ndarray
        Star semi-major axis wrt SMBH with :obj:`float` type
    smbh_mass : float
        Mass [Msun] of the SMBH

    Returns
    -------
    tde_id_num : numpy.ndarray
        ID numbers of stars that will become TDEs
    flip_id_num : numpy.ndarray
        ID numbers of stars that will flip to prograde
    """

    # Convert everything to units
    star_mass = star_retro_mass * astropy_units.Msun
    star_radius = (10 ** star_retro_log_radius) * astropy_units.Rsun
    star_orb_a = (si_from_r_g(smbh_mass, star_retro_orb_a)).to("meter")
    smbh_mass_units = smbh_mass * astropy_units.Msun

    # Tidal disruption radius of the disk is R_star * (M_smbh / M_star)^1/3
    disk_radius_tidal_disruption = (star_radius * ((smbh_mass_units / star_mass) ** (1./3.))).to("meter")

    # Stars are TDEs if they are inside the TD radius and have eccentricity >= 0.8
    tde_mask = ((star_orb_a * (1 - star_retro_orb_ecc)) <= disk_radius_tidal_disruption) & (star_retro_orb_ecc >= 0.8)

    # Stars that don't become TDEs will flip to prograde
    tde_id_num = star_retro_id_num[tde_mask]
    flip_id_num = star_retro_id_num[~tde_mask]

    return (tde_id_num, flip_id_num)
