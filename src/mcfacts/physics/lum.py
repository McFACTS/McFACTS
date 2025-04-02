"""
Module for calculating luminosities produced by merger remnant interacting with gas via ram-pressure stripping or jet formation.
"""

import numpy as np
from astropy import units as u
from astropy import constants as const
from mcfacts.physics.point_masses import si_from_r_g

def shock_luminosity(smbh_mass, mass_final, bin_orb_a, disk_aspect_ratio, disk_density, vk):
    """
    Calculate the gas volume, Hill sphere volume, and Hill radius. McKernan et al. 2019

    R_hill ≡ bin_orb_a*(q/3)**1/3)

    where q = (remnant mass/smbh mass)

    V_gas = 4/3*pi*R_hill**3 - 2/3*pi*(R_hill - height)**2 * [3*R_hill - (R_hill - height)]
    McKernan et al 2019 eqn 4.

    Calculate the shock luminosity based on the gas mass, Hill radius, and velocity.

    E = 10e46 erg/s * (M_hill / 1 M_solar) * (v_kick / 200 km/s)
    t = O(12 mon [seconds]) * (R_hill / v_kick)

    shock_luminsoity = E/t [erg/s]

    Parameters:
    - smbh_mass: Mass of the SMBH
    - mass_final: Mass of the BBH (solar masses).
    - bin_orb_a: semi major axis between binary and smbh.
    - disk_aspect_ratio: aspect ratio of the disk at merger.
    - disk_density: Gas density (kg m^-3).
    - vk: Characteristic velocity (km/s).

    Returns:
    - Lshock: Shock luminosity (erg s^-1).
    """

    r_hill_rg = bin_orb_a * ((mass_final / smbh_mass) / 3)**(1/3) 
    r_hill_m = si_from_r_g(smbh_mass, r_hill_rg)

    r_hill_m = r_hill_m.to('cm').value

    disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a
    disk_height_m = si_from_r_g(smbh_mass, disk_height_rg)
    disk_height_m = disk_height_m.to('cm').value

    v_hill = (4 / 3) * np.pi * r_hill_m**3  
    v_hill_gas = abs(v_hill - (2 / 3) * np.pi * ((r_hill_m - disk_height_m)**2) * (3 * r_hill_m - (r_hill_m - disk_height_m)))
    
    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)
    disk_density_cgs = disk_density_si.to(u.g / u.cm**3)

    disk_density_cgs = disk_density_cgs.value
    msolar = const.M_sun.to('g').value

    r_hill_mass = (disk_density_cgs * v_hill_gas) / msolar

    v_kick_scale = 200. * (u.km / u.s)
    v_kick_scale = v_kick_scale.value
    E = 10**46 * (r_hill_mass / 1) * (vk / v_kick_scale)**2  # Energy of the shock
    time = 31556952.0 * ((r_hill_rg / 3) / (vk / v_kick_scale))  # Timescale for energy dissipation
    Lshock = E / time  # Shock luminosity
    
    return Lshock

def jet_luminosity(mass_final, bin_orb_a, disk_density, vk):
    """
    Calculate Bondi-Hoyle-Lyttleton luminosity based on the mass, density, and velocity. add eqn. please insert paper.

    Graham et al. 2020 eqn 5 ():
    L_BHL = 2.5e45 * (eta / 0.1) * (M_BBH / 100 M_solar)**2 * (v_kick / 200 km/s)**-3 * (rho / 10e-10 g/cm**-3)
https://file+.vscode-resource.vscode-cdn.net/Users/emilymcpike/McFACTS/runs/time_vs_jet_lum.png?version%3D1740528569713
    Parameters:
    - mass_final: Mass of the BBH (solar masses).
    - bin_orb_a: semi major axis of binary at merger
    - disk_density: Gas density (kg m^3).
    - vk: kick velocity (km/s).

    Returns:
    - LBHL: Jet luminosity (erg s^-1).
    """
    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)

    disk_density_cgs = disk_density_si.to(u.g / u.cm**3)
    disk_density_cgs = disk_density_cgs.value
    # eta depends on spin t.f. isco, assuming eddington accretion... .06-.42 and so 0.1 is a good O approx.
    # but.. bondi is greater mass accretion rate, t.f. L per mass acrreted will be less becayse so much shit 
    # is trying to get in in so little space for light to escape
    v_kick_scale = 200. * (u.km / u.s)
    v_kick_scale = v_kick_scale.value
    LBHL = 2.5e45 * (0.1 / 0.1) * (mass_final / 100)**2 * (vk / v_kick_scale)**-3 * (disk_density_cgs / 10e-10)  # Jet luminosity
    return LBHL


def AGN_lum(temp_func, smbh_mass, bin_orb_a):
    T = temp_func(bin_orb_a)
    
    r = [1]  # r_isco
    sorted_bin_orb_a = np.sort(bin_orb_a)  
    r.extend(sorted_bin_orb_a)  
    r.append(2)  # r_outer

    lum_agn = []
    
    for i in range(len(r) - 1):  # Loop up to the second-to-last element
        dr = r[i + 1] - r[i]  # Compute radial difference
        area = 2 * np.pi * r[i] * dr  # Compute area of annulus
        L_i = area * (const.sigma_sb.value * T[i]**4)  # Compute luminosity for each annulus
        lum_agn.append(L_i)  # Append to the luminosity list
    
    return lum_agn
    
    #* emitting area, annulus, 2pir*dr... add up whole disk from isco -> outer edge. div 
    # area = 2pir*dr
    # array from isco to outer edge. divide into sensible chunks, at interior edge its x1, emission from 0th component to 1
    # component is dr, do 2pirdr
    #print(L)

