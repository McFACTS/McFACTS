"""
Module for calculating bolometric luminosities from a merger remnant interacting with gas via shocks or jets.
"""
import numpy as np
from astropy import units as u
from astropy import constants as ct
from mcfacts.physics.point_masses import si_from_r_g
from mcfast import shock_luminosity_helper, jet_luminosity_helper

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
    - The local height of the disk.
    - The gas volume inside the Hill sphere.
    - The mass of gas inside the remnant's Hill sphere.
    - The energy and timescale over which energy is dissipated into the disk.

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
        Distance between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_aspect_ratio : callable
        Function that returns the aspect ratio (height/radius) of the disk at a given radius.
    disk_density : callable
        Function that returns the gas density at a given radius (in [kg m**-3]).
    v_kick : numpy.ndarray
        Kick velocity imparted to the remnant (in [km s**-1]).

    Returns:
    -------
    L_shock : float
        Shock luminosity (in [erg s**-1]).
    """
    # get the Hill radius in [R_g] and convert to [m]
    r_hill_rg = bin_orb_a * ((mass_final / smbh_mass) / 3)**(1/3) 
    r_hill_m = si_from_r_g(smbh_mass, r_hill_rg)
    r_hill_m = r_hill_m.value

    # initalize scaling value for Hill radius from McKernan et al. (2019)
    r_hill_rg_scale = 10**3 * ((65 / 10**9) / 3)**(1/3) 

    # get the height of the disk in [R_g] and convert to [m]
    disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a
    disk_height_m = si_from_r_g(smbh_mass, disk_height_rg)
    disk_height_m = disk_height_m.value

    # compute the volume of the Hill sphere in [m**3]
    v_hill = (4 / 3) * np.pi * r_hill_m**3  
    # compute the volume of the gas contained within the hill sphere, from McKernan et al. (2019) in [m**3]
    v_hill_gas = abs(v_hill - (2 / 3) * np.pi * ((r_hill_m - disk_height_m)**2) * (3 * r_hill_m - (r_hill_m - disk_height_m)))
    
    # get the local disk density [kg] / [m**3]
    disk_density_si = disk_density(bin_orb_a)

    # use the disk density and volume of the gas within the Hill sphere to get the mass of the gas contained in the Hill sphere in [kg]
    r_hill_mass = (disk_density_si * v_hill_gas) 
    # initalize the scaling value for the gas contained within the Hill sphere from McKernan et al. (2019)
    r_hill_mass_scale = ct.M_sun.value

    # initalize the scaling value for the kick velocity of the remnant black hole from McKernan et al. (2019)
    v_kick_scale = 100.

    # calculate the energy dissipated into the disk as in McKernan et al. (2019)
    E = 1e47 * (r_hill_mass / r_hill_mass_scale) * (v_kick / v_kick_scale)**2  # energy
    # calculate the time scale for energy dissipation as in McKernan et al. (2019)
    time = 1.577e7 * (r_hill_rg / 3 * r_hill_rg_scale) / (v_kick / v_kick_scale) 
    # calculate the shock luminosity as the energy dissipated into the disk overtime, as in McKernan et al. (2019)
    L_shock = E / time
    return L_shock

def shock_luminosity_opt(smbh_mass,
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
    - The local height of the disk.
    - The gas volume inside the Hill sphere.
    - The mass of gas inside the remnant's Hill sphere.
    - The energy and timescale over which energy is dissipated into the disk.

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
        Distance between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_aspect_ratio : callable
        Function that returns the aspect ratio (height/radius) of the disk at a given radius.
    disk_density : callable
        Function that returns the gas density at a given radius (in [kg m**-3]).
    v_kick : numpy.ndarray
        Kick velocity imparted to the remnant (in [km s**-1]).

    Returns:
    -------
    L_shock : float
        Shock luminosity (in [erg s**-1]).
    """

    disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a
    disk_density_si = disk_density(bin_orb_a)

    L_shock = shock_luminosity_helper(smbh_mass, mass_final, bin_orb_a, disk_height_rg, disk_density_si, v_kick)

    return L_shock

def jet_luminosity(mass_final,
        bin_orb_a,
        disk_density,
        spin_final,
        v_kick,
        disk_sound_speed):
    """
    Estimate the jet luminosity produced by Bondi-Hoyle-Lyttleton (BHL) accretion.

    Based on Graham et al. (2020), the luminosity goes as:
        L_BHL ≈ 2.5e45 erg s **-1 * (eta / 0.1) * (M / 100 M_sun)**2 * (v_kick / 200 km/s)**-3 * (rho / 1e-9 g/cm^3)
    where eta is the radiation efficiency, which is well modeled as eta ~ a**2, 
    where a is the spin of the remnant BH (Tagawa et al. (2023)), M is the mass of the remnant black hole,
    v_kick is the kick velocity imparted to the remannt upon merger, and rho is the local gas density
    of the AGN accretion disk.

    Parameters:
    ----------
    mass_final : numpy.ndarray
        mass of remnant post-merger (mass loss accounted for via Tichy & Maronetti 08)
    bin_orb_a : numpy.ndarray
        Distance between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_density : callable
        Function that returns the gas density at a given radius (in [kg m**-3]).
    spin_final : numpy.ndarray
        Spin of the remnant black hole. Unitless.
    v_kick : numpy.ndarray
        Kick velocity imparted to the remnant (in [km s**-1]).
    disk_sound_speed : callable
        Function that returns the disk sound speed at a given radius (in [m s**-1]).

    Returns:
    -------
    LBHL : numpy.ndarray
        Estimated jet luminosity (in [erg s**-1]).
    """
    #print(migration_velocity)
    # get the local disk density and convert from [kg m**-3] to [g cm**-3]
    disk_density_cgs = disk_density(bin_orb_a) * 10**-3

    # get the local sound speed of the disk (in [m s**-1])
    sound_speed = disk_sound_speed(bin_orb_a) 
    # get the relative velocity of the remnant and migration velocity [cm s**-1]
    v_rel = (v_kick * 10**3 * 10**2)

    # convert the mass of the remnant black hole from [Msun] to [g]
    mass_final_g = mass_final * 1.98841e+33 

    # calculate Bondi accretion, convert sound speed from m / s to cm / s
    mdot_bondi = 4 * np.pi * (ct.G.cgs.value ** 2) * (mass_final_g ** 2) * disk_density_cgs * (v_rel**2 + (sound_speed * 10**2)**2)**-(3/2)

    kappa = 0.1
    # calculate the jet luminosity as in Kim & Most 2025
    L_jet = (0.1) * (kappa / 0.1) * (0.9 / spin_final)**2 * mdot_bondi * ct.c.cgs.value**2
    return L_jet


def jet_luminosity_opt(mass_final,
        bin_orb_a,
        disk_density,
        spin_final,
        v_kick,
        disk_sound_speed):
    """
    Estimate the jet luminosity produced by Bondi-Hoyle-Lyttleton (BHL) accretion.

    Based on Graham et al. (2020), the luminosity goes as:
        L_BHL ≈ 2.5e45 erg s **-1 * (eta / 0.1) * (M / 100 M_sun)**2 * (v_kick / 200 km/s)**-3 * (rho / 1e-9 g/cm^3)
    where eta is the radiation efficiency, which is well modeled as eta ~ a**2, 
    where a is the spin of the remnant BH (Tagawa et al. (2023)), M is the mass of the remnant black hole,
    v_kick is the kick velocity imparted to the remannt upon merger, and rho is the local gas density
    of the AGN accretion disk.

    Parameters:
    ----------
    mass_final : numpy.ndarray
        mass of remnant post-merger (mass loss accounted for via Tichy & Maronetti 08)
    bin_orb_a : numpy.ndarray
        Distance between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_density : callable
        Function that returns the gas density at a given radius (in [kg m**-3]).
    spin_final : numpy.ndarray
        Spin of the remnant black hole. Unitless.
    v_kick : numpy.ndarray
        Kick velocity imparted to the remnant (in [km s**-1]).
    disk_sound_speed : callable
        Function that returns the disk sound speed at a given radius (in [m s**-1]).

    Returns:
    -------
    LBHL : numpy.ndarray
        Estimated jet luminosity (in [erg s**-1]).
    """
    #print(migration_velocity)
    # get the local disk density and convert from [kg m**-3] to [g cm**-3]
    disk_density_cgs = disk_density(bin_orb_a) * 10**-3

    # get the local sound speed of the disk (in [m s**-1])
    sound_speed = disk_sound_speed(bin_orb_a) 

    L_jet = jet_luminosity_helper(mass_final, disk_density_cgs, spin_final, v_kick, sound_speed)

    return L_jet
