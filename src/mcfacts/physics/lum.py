"""
Module for calculating luminosities produced by merger remnant interacting with gas via ram-pressure stripping or jet formation.
"""
import numpy as np
from astropy import units as u
from astropy import constants as ct
from mcfacts.physics.point_masses import si_from_r_g

def shock_luminosity(smbh_mass,
        mass_final,
        bin_orb_a,
        disk_aspect_ratio,
        disk_density,
        v_kick):
    """
    Estimate the shock luminosity from the interaction between a merger remnant 
    and gas within its Hill sphere.

    Based on McKernan et al. (2019) (arXiv:1907.03746v2), this function computes:
    - The Hill radius of the remnant system.
    - The gas volume inside the Hill sphere, accounting for the vertical extent of the disk.
    - The mass of gas available for interaction.
    - The shock energy and characteristic timescale over which energy is radiated.

    The shock luminosity is given by:
        L_shock ≈ E / t,
    where
        E = 1e47 erg * (M_gas / M_sun) * (v_kick / 200 km/s)^2
        t ~ R_Hill / v_kick

    Parameters:
    ----------
    smbh_mass : float
        Mass of the supermassive black hole (in solar masses).
    mass_final : numpy.ndarray
        Final mass of the binary black hole remnant (in solar masses).
    bin_orb_a : numpy.ndarray
        Orbital separation between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_aspect_ratio : callable
        Function that returns the aspect ratio (height/radius) of the disk at a given radius.
    disk_density : callable
        Function that returns the gas density at a given radius (in kg m^-3).
    v_kick : numpy.ndarray
        Kick velocity imparted to the remnant (in km/s).

    Returns:
    -------
    Lshock : float
        Estimated shock luminosity (in erg/s).
    """
    r_hill_rg = bin_orb_a * ((mass_final / smbh_mass) / 3)**(1/3) 
    r_hill_m = si_from_r_g(smbh_mass, r_hill_rg)

    r_hill_rg_scale = 10**3 * ((65 / 10**9) / 3)**(1/3) 

    disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a
    disk_height_m = si_from_r_g(smbh_mass, disk_height_rg)

    v_hill = (4 / 3) * np.pi * r_hill_m**3  
    v_hill_gas = abs(v_hill - (2 / 3) * np.pi * ((r_hill_m - disk_height_m)**2) * (3 * r_hill_m - (r_hill_m - disk_height_m)))
    
    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)

    msolar = ct.M_sun
    r_hill_mass = (disk_density_si * v_hill_gas) 

    v_kick = v_kick  * (u.km / u.s)
    v_kick_scale = 100. * (u.km / u.s)
    E = 1e47 * (r_hill_mass / msolar) * (v_kick / v_kick_scale)**2  # Energy of the shock
    time = 1.577e7 * (r_hill_rg / 3 * r_hill_rg_scale) / (v_kick / v_kick_scale)  # Timescale for energy dissipation
    Lshock = E / time  # Shock luminosity
    return Lshock.value

def jet_luminosity(mass_final,
        bin_orb_a,
        disk_density,
        spin_final,
        v_kick):
    """
    Estimate the jet luminosity produced by Bondi-Hoyle-Lyttleton (BHL) accretion.

    Based on Graham et al. (2020), the luminosity goes as:
        L_BHL ≈ 2.5e45 erg/s * (η / 0.1) * (M / 100 M_sun)^2 * (v / 200 km/s)^-3 * (rho / 1e-9 g/cm^3)

    Parameters:
    ----------
    mass_final : numpy.ndarray
        mass of remnant post-merger (mass loss accounted for via Tichy & Maronetti 08)
    bin_orb_a : numpy.ndarray
        Orbital separation between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_density : callable
        Function that returns the gas density at a given radius (in kg m^-3).
    v_kick : numpy.ndarray
        Kick velocity imparted to the remnant (in km/s).

    Returns:
    -------
    LBHL : numpy.ndarray
        Estimated jet (Bondi-Hoyle) luminosity (in erg/s).
    """

    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)

    v_kick = v_kick * (u.km / u.s)
    v_kick_scale = 200. * (u.km / u.s)

    mass_scale = 100
    density_scale = 1e-7 * (u.kg / u.m**3)
    eta = spin_final**2

    Ljet = 2.5e45 * (eta / 0.1) * (mass_final / mass_scale)**2 * (v_kick / v_kick_scale)**-3 * (disk_density_si / density_scale)  # Jet luminosity
    return Ljet.value