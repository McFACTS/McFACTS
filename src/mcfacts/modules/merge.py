"""
Module for calculating the final variables of a merging binary.
"""
import uuid

import warnings
import numpy as np
from astropy import constants as const
from astropy import units as u
from numpy.random import Generator

from mcfacts.external.sxs import fit_modeler, evolve_binary
from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBinaryBlackHoleArray, AGNBlackHoleArray, AGNMergedBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities import unit_conversion, checks
from mcfacts.utilities.random_state import uuid_provider, rng
from mcfacts.utilities.unit_conversion import si_from_r_g


def analytical_kick_velocity(
        mass_1,
        mass_2,
        spin_1,
        spin_2,
        spin_angle_1,
        spin_angle_2):
    """
    Compute the analytical gravitational wave recoil (kick) velocity for merging black hole binaries
    as in Akiba et al. 2024 (arXiv:2410.19881).

    Parameters
    ----------
    mass_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    mass_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    spin_1 : numpy.ndarray
        Spin magnitude [unitless] of object 1 with :obj:`float` type
    spin_2 : numpy.ndarray
        Spin magnitude [unitless] of object 2 with :obj:`float` type
    spin_angle_1 : numpy.ndarray
        Spin angle [radian] of object 1 with :obj:`float` type
    spin_angle_2 : numpy.ndarray
        Spin angle [radian] of object 2 with :obj:`float` type

    Returns
    -------
    v_kick : np.ndarray
        Kick velocity [km/s] of the remnant BH with :obj:`float` type
    """
    # As in Akiba et al 2024 Appendix A, mass_2 should be the more massive BH in the binary.
    mask = mass_1 <= mass_2

    m_1_new = np.where(mask, mass_1, mass_2) * u.solMass
    m_2_new = np.where(mask, mass_2, mass_1)* u.solMass
    spin_1_new = np.where(mask, spin_1, spin_2)
    spin_2_new = np.where(mask, spin_2, spin_1)
    spin_angle_1_new = np.where(mask, spin_angle_1, spin_angle_2)
    spin_angle_2_new = np.where(mask, spin_angle_2, spin_angle_1)

    # "perp" and "par" refer to components perpendicular and parallel to the orbital angular momentum axis, respectively.
    # Orbital angular momentum axis of binary is aligned with the disk angualr momentum.
    # Find the perp and par components of spin:
    spin_1_par = spin_1_new * np.cos(spin_angle_1_new)
    spin_1_perp = spin_1_new * np.sin(spin_angle_1_new)
    spin_2_par = spin_2_new * np.cos(spin_angle_2_new)
    spin_2_perp = spin_2_new * np.sin(spin_angle_2_new)

    # Find the mass ratio q and asymmetric mass ratio eta
    # as defined in Akiba et al. 2024 Appendix A:
    q = m_1_new / m_2_new
    eta = q / (1 + q)**2

    # Use Akiba et al. 2024 eqn A5:
    S = (2 * (spin_1_new + q**2 * spin_2_new)) / (1 + q)**2

    # As defined in Akiba et al. 2024 Appendix A:
    xi = np.radians(145)
    A = 1.2e4 * u.km / u.s
    B = -0.93
    H = 6.9e3 * u.km / u.s
    V_11, V_A, V_B, V_C = 3678 * u.km / u.s, 2481 * u.km / u.s, 1793* u.km / u.s, 1507 * u.km / u.s
    angle = rng.uniform(0.0, 2*np.pi, size=len(mass_1))

    # Use Akiba et al. 2024 eqn A2:
    v_m = A * eta**2 * np.sqrt(1 - 4 * eta) * (1 + B * eta)

    # Use Akiba et al. 2024 eqn A3:
    v_perp = (H * eta**2 / (1 + q)) * (spin_2_par - q * spin_1_par)

    # Use Akiba et al. 2024 eqn A4:
    v_par = ((16 * eta**2) / (1 + q)) * (V_11 + (V_A * S) + (V_B * S**2) + (V_C * S**3)) * \
            np.abs(spin_2_perp - q * spin_1_perp) * np.cos(angle)

    # Use Akiba et al. 2024 eqn A1:
    v_kick = np.sqrt((v_m + v_perp * np.cos(xi))**2 +
                     (v_perp * np.sin(xi))**2 +
                     v_par**2)
    v_kick = np.array(v_kick.value)
    assert np.all(v_kick > 0), \
        "v_kick has values <= 0"
    assert np.isfinite(v_kick).all(), \
        "Finite check failure: v_kick"
    return v_kick


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
    r_hill_rg = bin_orb_a * ((mass_final / smbh_mass) / 3) ** (1 / 3)
    r_hill_m = unit_conversion.si_from_r_g(smbh_mass, r_hill_rg)
    r_hill_cm = r_hill_m.cgs

    disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a
    disk_height_m = unit_conversion.si_from_r_g(smbh_mass, disk_height_rg)
    disk_height_cm = disk_height_m.cgs

    v_hill = (4 / 3) * np.pi * r_hill_cm ** 3
    v_hill_gas = abs(v_hill - (2 / 3) * np.pi * ((r_hill_cm - disk_height_cm) ** 2) * (3 * r_hill_cm - (r_hill_cm - disk_height_cm)))

    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m ** 3)
    disk_density_cgs = disk_density_si.cgs

    msolar = const.M_sun.cgs

    r_hill_mass = (disk_density_cgs * v_hill_gas) / msolar

    # for scaling:
    rg = const.G.cgs * smbh_mass * const.M_sun.cgs / const.c.cgs

    v_kick = v_kick * (u.km / u.s)
    v_kick_scale = 200. * (u.km / u.s)
    E = 10 ** 46 * (r_hill_mass / 1 * rg) * (v_kick / v_kick_scale) ** 2  # Energy of the shock
    time = 31556952.0 * ((r_hill_rg / 3 * rg) / (v_kick / v_kick_scale))  # Timescale for energy dissipation
    Lshock = E / time  # Shock luminosity
    return Lshock.value


def jet_luminosity(mass_final,
                   bin_orb_a,
                   disk_density,
                   disk_aspect_ratio,
                   smbh_mass,
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

    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m ** 3)
    disk_density_cgs = disk_density_si.cgs

    v_kick = v_kick * (u.km / u.s)
    v_kick_scale = 200. * (u.km / u.s)

    eta = spin_final ** 2

    Ljet = 2.5e45 * (eta / 0.1) * (mass_final / 100 * u.M_sun) ** 2 * (v_kick / v_kick_scale) ** -3 * (disk_density_cgs / 10e-10 * (u.g / u.cm * 3))  # Jet luminosity
    return Ljet.value


def chi_effective(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2, bin_ang_mom):
    """Calculates the effective spin :math:`\\chi_{\rm eff}` associated with a merger.

    The measured effective spin of a merger is calculated as

    .. math:: \\chi_{\rm eff}=\frac{m_1*\\chi_1*\\cos(\theta_1) + m_2*\\chi_2*\\cos(\theta_2)}/{m_{\rm bin}}

    Parameters
    ----------
    masses_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    masses_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    spins_1 : numpy.ndarray
        Spin magnitude [unitless] of object 1 with :obj:`float` type
    spins_2 : numpy.ndarray
        Spin magnitude [unitless] of object 2 with :obj:`float` type
    spin_angles_1 : numpy.ndarray
        Spin angle [radian] of object 1 with :obj:`float` type
    spin_angles_2 : numpy.ndarray
        Spin angle [radian] of object 2 with :obj:`float` type
    bin_ang_mom : int/ndarray
        Magnitude of the binary's mutual angular momentum. If 1, the binary
        is prograde (aligned with disk angular momentum). If -1, the binary
        is retrograde (anti-aligned with disk angular momentum).

    Returns
    -------
    chi_eff : numpy.ndarray
        The effective spin value [unitless] for these object(s) with :obj:`float` type
    """

    total_masses = masses_1 + masses_2
    spins_1 = np.abs(spins_1)
    spins_2 = np.abs(spins_2)

    spin_angles_1[bin_ang_mom < 0] = np.pi - spin_angles_1[bin_ang_mom < 0]
    spin_angles_2[bin_ang_mom < 0] = np.pi - spin_angles_2[bin_ang_mom < 0]

    spin_factors_1 = (masses_1 / total_masses) * spins_1 * np.cos(spin_angles_1)
    spin_factors_2 = (masses_2 / total_masses) * spins_2 * np.cos(spin_angles_2)

    chi_eff = spin_factors_1 + spin_factors_2

    assert np.isfinite(chi_eff).all(), \
        "Finite check failure: chi_eff"

    return (chi_eff)


def chi_p(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2, bin_orbs_inc):
    """Calculates the precessing spin component :math:`\chi_p` associated with a merger.

    chi_p = max[spin_1_perp, (q(4q+3)/(4+3q))* spin_2_perp]

    where

    spin_1_perp = spin_1 * sin(spin_angle_1)
    spin_2_perp = spin_2 * sin(spin_angle_2)

    are perpendicular to `spin_1

    and :math:`q=M_2/M_1` where :math:`M_2 < M_1`.

    Parameters
    ----------
    masses_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    masses_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    spins_1 : numpy.ndarray
        Spin magnitude [unitless] of object 1 with :obj:`float` type
    spins_2 : numpy.ndarray
        Spin magnitude [unitless] of object 2 with :obj:`float` type
    spin_angles_1 : numpy.ndarray
        Spin angle [radian] of object 1 with :obj:`float` type
    spin_angles_2 : numpy.ndarray
        Spin angle [radian] of object 2 with :obj:`float` type
    bin_orbs_inc : numpy.ndarray
        Angle of inclination [radian] of the binary with respect to the disk.

    Returns
    -------
    chi_p : numpy.ndarray
        Precessing spin component for these objects
    """

    # If mass_1 is the dominant binary component
    # Define default mass ratio of 1.0, otherwise choose based on masses
    mass_ratios = np.ones(masses_1.size)

    # Define spin angle to include binary inclination wrt disk (units of radians)
    spin_angles_1 = spin_angles_1 + bin_orbs_inc
    spin_angles_2 = spin_angles_2 + bin_orbs_inc

    # Make sure angles are < pi radians
    spin_angles_1_diffs = spin_angles_1 - np.pi
    spin_angles_2_diffs = spin_angles_2 - np.pi

    spin_angles_1[spin_angles_1_diffs > 0] = spin_angles_1[spin_angles_1_diffs > 0] - spin_angles_1_diffs[spin_angles_1_diffs > 0]
    spin_angles_2[spin_angles_2_diffs > 0] = spin_angles_2[spin_angles_2_diffs > 0] - spin_angles_2_diffs[spin_angles_2_diffs > 0]

    # Define default spins
    spins_1_perp = np.abs(spins_1) * np.sin(spin_angles_1)
    spins_2_perp = np.abs(spins_2) * np.sin(spin_angles_2)

    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_2 > masses_1] = masses_1[masses_2 > masses_1] / masses_2[masses_2 > masses_1]

    spins_1_perp[masses_2 > masses_1] = np.abs(spins_2[masses_2 > masses_1]) * np.sin(spin_angles_2[masses_2 > masses_1])

    spins_2_perp[masses_2 > masses_1] = np.abs(spins_1[masses_2 > masses_1]) * np.sin(spin_angles_1[masses_2 > masses_1])

    mass_ratio_factors = mass_ratios * ((4.0 * mass_ratios) + 3.0) / (4.0 + (3.0 * mass_ratios))

    # Assume spins_1_perp is dominant source of chi_p
    chi_p = spins_1_perp
    # If not then change chi_p definition and output
    chi_p[chi_p < (mass_ratio_factors * spins_2_perp)] = mass_ratio_factors[chi_p < (mass_ratio_factors * spins_2_perp)] * spins_2_perp[chi_p < (mass_ratio_factors * spins_2_perp)]

    assert np.isfinite(chi_p).all(), \
        "Finite check failure: chi_p"

    return (chi_p)


def merged_mass(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2):
    """Calculates the final mass of a merged binary.

    Using approximations from Tichy \\& Maronetti (2008) where

    .. math::
        `m_final = (M_1+M_2) * [1.0 - 0.2{nu} - 0.208{nu}^2(a_1z + a_2z)]`
    where nu is the symmetric mass ratio or `nu = q/((1+q)^2)`

    Parameters
    ----------
    masses_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    masses_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    spins_1 : numpy.ndarray
        Spin magnitude [unitless] of object 1 with :obj:`float` type
    spins_2 : numpy.ndarray
        Spin magnitude [unitless] of object 2 with :obj:`float` type
    spin_angles_1 : numpy.ndarray
        spin angle of m1 (wrt disk angular momentum) in radians
    spin_angles_2 : numpy.ndarray
        spin angle of m2 (wrt disk angular momentum) in radians

    Returns
    -------
    merged_masses: numpy.ndarray
        Final mass [M_sun] of merger remnant with :obj:`float` type
    """

    mass_ratios = np.ones(masses_1.size)
    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_1 < masses_2] = masses_1[masses_1 < masses_2] / masses_2[masses_1 < masses_2]

    # Setting random array of phi angles for each of the progenitors

    # Spin components of each of the progenitors
    spin_1_z = spins_1 * np.cos(spin_angles_1)
    spin_2_z = spins_2 * np.cos(spin_angles_2)

    total_masses = masses_1 + masses_2
    total_spins = spin_1_z + spin_2_z
    nu_factors = (1.0 + mass_ratios) ** 2.0
    nu = mass_ratios / nu_factors
    nu_squared = nu * nu

    mass_factors = 1.0 - (0.2 * nu) - (0.208 * nu_squared * total_spins)
    merged_masses = total_masses*mass_factors

    assert np.all(merged_masses > 0), \
        "merged_mass has values <= 0"

    return (merged_masses)


def merged_spin(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2):
    """Calculates the spin magnitude of a merged binary.

    Only depends on M1,M2,a1,a2,theta1,theta2 and the binary angular momentum around its center of mass.
    Using approximations from Tichy \\& Maronetti (2008) where
    .. math:: `az_final = 0.686 * (5.04{nu} - 4.16{nu}^2) + 0.3995[(a1_z / (0.632 + q)^2) + (a2_z / (0.632 + (1/q))^2)] + (0.128{nu}^2)[(a_x + b_x)^2 + (a_y + b_y)^2 - (a_z + b_z)^2]`
    where `q=m_2/m_1 ; nu=q/((1+q)^2)` \n
    a_x == x component of the magnitude of the spin (spin_1) ; a_y == y component of the magnitude of the spin (spin_1) ; a_z == z component of the magnitude of the spin (spin_1) \n
    and b_x == x component of the magnitude of the spin (spin_2) ; b_y == y component of the magnitude of the spin (spin_2) ; b_z == z component of the magnitude of the spin (spin_2)


    Parameters
    ----------
    masses_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    masses_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    spins_1 : numpy.ndarray
        Spin magnitude [unitless] of object 1 with :obj:`float` type
    spins_2 : numpy.ndarray
        Spin magnitude [unitless] of object 2 with :obj:`float` type
    spin_angles_1 : numpy.ndarray
        Spin angle of m1 (wrt disk angular momentum) in radians with :obj:`float` type
    spin_angles_2 : numpy.nbdarray
        Spin angle of m2 (wrt disk angular momentum) in radians with :obj:`float` type

    Returns
    -------
    merged_spins : numpy array
        Final spin magnitude [unitless] of merger remnant with :obj:`float` type
    """

    # Setting random array of phi angles for each of the progenitors
    phi_1_rand = rng.uniform(0, 2 * np.pi, len(spins_1))
    phi_2_rand = rng.uniform(0, 2 * np.pi, len(spins_1))

    # Rearranging spins, spin angles, and mass ratios to align with each other
    spins_1_new = np.ones(spins_1.size)
    spins_1_new[masses_1 > masses_2] = spins_2[masses_1 > masses_2]
    spins_1_new[masses_1 < masses_2] = spins_1[masses_1 < masses_2]

    spins_2_new = np.ones(spins_2.size)
    spins_2_new[masses_1 < masses_2] = spins_2[masses_1 < masses_2]
    spins_2_new[masses_1 > masses_2] = spins_1[masses_1 > masses_2]

    spin_angles_1_new = np.ones(spin_angles_1.size)
    spin_angles_1_new[masses_1 > masses_2] = spin_angles_2[masses_1 > masses_2]
    spin_angles_1_new[masses_1 < masses_2] = spin_angles_1[masses_1 < masses_2]

    spin_angles_2_new = np.ones(spin_angles_2.size)
    spin_angles_2_new[masses_1 < masses_2] = spin_angles_2[masses_1 < masses_2]
    spin_angles_2_new[masses_1 > masses_2] = spin_angles_1[masses_1 > masses_2]

    mass_ratios = np.ones(masses_1.size)
    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_1 < masses_2] = masses_1[masses_1 < masses_2] / masses_2[masses_1 < masses_2]

    mass_ratios_inv = 1.0 / mass_ratios

    # Spin components of each of the progenitors
    # progenitor a == object 1
    # progenitor b == object 2
    a_x = spins_1_new * np.sin(spin_angles_1_new) * np.cos(phi_1_rand)
    a_y = spins_1_new * np.sin(spin_angles_1_new) * np.sin(phi_1_rand)
    a_z = spins_1_new * np.cos(spin_angles_1_new)

    b_x = spins_2_new * np.sin(spin_angles_2_new) * np.cos(phi_2_rand)
    b_y = spins_2_new * np.sin(spin_angles_2_new) * np.sin(phi_2_rand)
    b_z = spins_2_new * np.cos(spin_angles_2_new)

    nu_factors = (1.0 + mass_ratios) ** 2.0
    nu = mass_ratios / nu_factors
    nu_squared = nu * nu

    spin_factors_1xy = (0.329 + mass_ratios) ** 2.0
    spin_factors_2xy = (0.329 + mass_ratios_inv) ** 2.0

    spin_factors_1z = (0.632 + mass_ratios) ** 2.0
    spin_factors_2z = (0.632 + mass_ratios_inv) ** 2.0

    merged_spins_x = 0.33 * ((a_x/spin_factors_1xy) + (b_x/spin_factors_2xy)) - (0.112 * nu * (a_y+b_y))
    merged_spins_y = (0.112 * nu * (a_x+b_x)) + (0.33 * ((a_y/spin_factors_1xy) + (b_y/spin_factors_2xy)))
    merged_spins_z = 0.686 * ((5.04 * nu) - (4.16 * nu_squared)) + (0.3995 * ((a_z / spin_factors_1z) + (b_z / spin_factors_2z))) + ((0.128 * nu_squared)*((a_x+b_x)**2 + (a_y+b_y)**2 - (a_z+b_z)**2))

    merged_spins = np.sqrt(merged_spins_x**2.0 + merged_spins_y**2.0 + merged_spins_z**2.0)
    #merged_spin_angle = np.arccos( [insert z component of spin] /merged_spins)

    assert np.isfinite(merged_spins).all(), \
        "Finite check failure: merged_spins"

    return (merged_spins) #merged_spin_angle


def merged_orb_ecc(bin_orbs_a, v_kicks, smbh_mass):
    """Calculates orbital eccentricity of a merged binary.

    Parameters
    ----------
    bin_orbs_a : numpy.ndarray
        Location of binary [r_{g,SMBH}] wrt to the SMBH with :obj:`float` type
    v_kicks : numpy.ndarray
        Kick velocity [km/s] with :obj:`float` type
    smbh_mass : float
        Mass [Msun] of the SMBH

    Returns
    -------
    merged_ecc : numpy.ndarray
        Orbital eccentricity of merged binary with :obj:`float` type
    """
    smbh_mass_units = smbh_mass * u.solMass
    orbs_a_units = unit_conversion.si_from_r_g(smbh_mass * u.solMass, bin_orbs_a).to("meter")

    v_kep = ((np.sqrt(const.G * smbh_mass_units / orbs_a_units)).to("km/s")).value

    merged_ecc = v_kicks/v_kep

    return (merged_ecc)

def merge_blackholes_precession(
    mass_1,
    mass_2,
    chi_1,
    chi_2,
    theta1,
    theta2,
    bin_sep_r_g,
    bin_ecc,
    smbh_mass,
    ):
    """Use Davide Gerosa's precession package to calculate merged properties

    https://pure-oai.bham.ac.uk/ws/files/61360327/1605.01067.pdf

    Parameters
    ----------
    mass_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    mass_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    chi_1 : numpy.ndarray
        Spin magnitude [unitless] of object 1 with :obj:`float` type
    chi_2 : numpy.ndarray
        Spin magnitude [unitless] of object 2 with :obj:`float` type
    theta1 : numpy.ndarray
        Spin angle [radian] of object 1 with :obj:`float` type
    theta2 : numpy.ndarray
        Spin angle [radian] of object 2 with :obj:`float` type
    bin_sep : numpy.ndarray
        Binary separation (R_g) with :obj:`float` type
    bin_ecc : numpy.ndarray
        Binary eccentricity with :obj:`float` type
    smbh_mass : float
        Mass of supermassive black hole (solMass)

    Returns
    -------
    bh_mass_merged : np.ndarray
        Merged mass of remnant
    bh_spin_merged : np.ndarray
        spin magnitudes of merged remnant
    bh_v_kick : np.ndarray
        Kick velocity of merged remnant (km/s)
    """
    #### Setup ####
    # Import Davide's precession package
    import precession
    # We need theta1, theta2, deltaphi, q, chi1, and chi2.
    # mass_1 and mass_2 need to be consistent with LVK
    mass_switch = mass_1 < mass_2
    if any(mass_switch):
        # Initialize temporary things
        _mass_1 = mass_1.copy()
        _mass_2 = mass_2.copy()
        _chi_1 = chi_1.copy()
        _chi_2 = chi_2.copy()
        _theta1 = theta1.copy()
        _theta2 = theta2.copy()
        # Make switches
        _mass_1[mass_switch] = mass_2[mass_switch]
        _mass_2[mass_switch] = mass_1[mass_switch]
        _chi_1[mass_switch] = chi_2[mass_switch]
        _chi_2[mass_switch] = chi_1[mass_switch]
        _theta1[mass_switch] = theta2[mass_switch]
        _theta2[mass_switch] = theta1[mass_switch]
        # Apply switches
        mass_1 = _mass_1
        mass_2 = _mass_2
        chi_1 = _chi_1
        chi_2 = _chi_2
        theta1 = _theta1
        theta2 = _theta2
    # check reasonable thetas
    theta1[theta1 < 1e-3] = 1e-3
    theta2[theta2 < 1e-3] = 1e-3
    # The chis shouldn't be negative
    # TODO assert positive spin
    chi_1 = np.abs(chi_1)
    chi_2 = np.abs(chi_2)


    # Estimate q
    mass_ratio = mass_2/ mass_1
    # Draw random deltaphi
    deltaphi = rng.uniform(low=0.,high=2*np.pi,size=mass_ratio.size)
    # Get binary separation
    bin_sep_si = si_from_r_g(smbh_mass, bin_sep_r_g)
    orbital_period_si = np.sqrt(
        (4 * np.pi**2 * bin_sep_si**3) / \
        (const.G * (mass_1 * u.solMass + mass_2 * u.solMass))
    ).si
    f_GW_si = 2/orbital_period_si
    bin_sep_M = (bin_sep_si * const.c**2 / const.G) / \
        ((mass_1 + mass_2) * u.solMass)
    bin_sep_M = bin_sep_M.si
    bin_sep_M = bin_sep_M.value

    ## Identify the separation at the beginning of the inspiral (20 Hz GW)
    # Estimate orbital period of 20 Hz (GW) / 10 Hz (orb)
    orbital_period_inspiral = 1 / (10 * u.Hz)
    # Identify the separation at 20 Hz, GW
    separation_inspiral = ((orbital_period_inspiral**2 * \
        ((mass_1 + mass_2) * u.solMass) * const.G / \
        (4 * np.pi**2))**(1/3)).si
    # Estimate the separation in units of M, at 20 Hz
    critical_separation = ((separation_inspiral * const.c**2 / const.G) / \
        ((mass_1 + mass_2) * u.solMass)).si.value

    # Check for unphysical spins
    chi_eff = precession.eval_chieff(
        theta1,
        theta2,
        mass_ratio,
        chi_1,
        chi_2,
    )

    #### Inspiral ####
    for i in range(mass_1.size):
        # Check separation
        if bin_sep_M[i] < critical_separation[i]:
            continue
        elif bin_sep_M[i] > 1000:
            bin_sep_M[i] = 1000
        # Check angle
        if (theta1[i] == 0) and (theta2[i] == 0):
            # Not precessing
            continue
        # Check chi effective limits
        chi_eff_minus, chi_eff_plus = precession.chiefflimits(
            q=mass_ratio[i],
            chi1=chi_1[i],
            chi2=chi_2[i],
        )
        _chi_eff_minus = min(chi_eff_minus,chi_eff_plus)
        _chi_eff_plus = max(chi_eff_minus,chi_eff_plus)
        chi_eff_minus, chi_eff_plus = _chi_eff_minus, _chi_eff_plus
        if (chi_eff[i] > chi_eff_plus) or (chi_eff[i] < chi_eff_minus):
            print(f"chi_eff[i]: {chi_eff[i]}")
            print(f"chi_eff_minus: {chi_eff_minus}")
            print(f"chi_eff_plus: {chi_eff_plus}")
            print(f"mass_1[i]: {mass_1[i]}")
            print(f"mass_2[i]: {mass_2[i]}")
            print(f"theta1[i]: {theta1[i]}")
            print(f"theta2[i]: {theta2[i]}")
            print(f"deltaphi[i]: {deltaphi[i]}")
            print(f"mass_ratio[i]: {mass_ratio[i]}")
            print(f"chi_1[i]: {chi_1[i]}")
            print(f"chi_2[i]: {chi_2[i]}")
            warnings.warn(f"Nonphysical chi effective: {chi_eff}")
            #raise ValueError(f"Nonphysical chi effective: {chi_eff}")
            if chi_eff[i] > chi_eff_plus:
                chi_eff[i] = chi_eff_plus
            elif chi_eff[i] < chi_eff_minus:
                chi_eff[i] = chi_eff_minus
            else:
                raise ValueError(f"Nonphysical chi effective: {chi_eff}")
        # Evolve binary
        try:
            evolve_outputs = precession.inspiral_precav(
                r=[bin_sep_M[i],critical_separation[i]],
                theta1=theta1[i],
                theta2=theta2[i],
                deltaphi=deltaphi[i],
                q=mass_ratio[i],
                chi1=chi_1[i],
                chi2=chi_2[i],
            )
        except OverflowError:
            # This is an internal precession error
            # Binary is not evolved.
            continue
        if np.isnan(evolve_outputs["deltaphi"][0,-1]):
            # If the spins are not finite, do not evolve binary
            continue
        # Update quantities
        bin_sep_M[i] = critical_separation[i]
        #chi_1[i] = evolve_outputs["chi1"][0,-1]
        #chi_2[i] = evolve_outputs["chi2"][0,-1]
        theta1[i] = evolve_outputs["theta1"][0,-1]
        theta2[i] = evolve_outputs["theta2"][0,-1]
        deltaphi[i] = evolve_outputs["deltaphi"][0,-1]
        #print(evolve_outputs)
        #raise Exception

    #### Merger ####
    bh_mass_merged = precession.remnantmass(
        theta1,
        theta2,
        mass_ratio,
        chi_1,
        chi_2,
    ) * (mass_1 + mass_2)
    bh_v_kick = precession.remnantkick(
        theta1,
        theta2,
        deltaphi,
        mass_ratio,
        chi_1,
        chi_2,
        kms=True
    )
    bh_spin_merged = precession.remnantspin(
        theta1,
        theta2,
        deltaphi,
        mass_ratio,
        chi_1,
        chi_2,
    )
    bh_thetaL = precession.reminantspindirection(
        theta1,
        theta2,
        deltaphi,
        bin_sep_M,
        mass_ratio,
        chi_1,
        chi_2,
    )
    if not np.all(np.isfinite(bh_spin_merged)):
        print(f"chi_eff: {chi_eff}")
        print(f"chi_eff_minus: {chi_eff_minus}")
        print(f"chi_eff_plus: {chi_eff_plus}")
        print(f"mass_1: {mass_1}")
        print(f"mass_2: {mass_2}")
        print(f"theta1: {theta1}")
        print(f"theta2: {theta2}")
        print(f"deltaphi: {deltaphi}")
        print(f"mass_ratio: {mass_ratio}")
        print(f"chi_1: {chi_1}")
        print(f"chi_2: {chi_2}")
        raise ValueError(f"spins are not finite: {bh_spin_merged}")
# bh_thetaL is the angle between the
    #  spin of the remnant and the binary angular momentum
    # Somebody should check if there's something else we should do
    #  to estimate the angle which is actually calculated here.
    bh_spin_angle_merged = bh_thetaL

    return bh_mass_merged, bh_spin_merged, bh_spin_angle_merged, bh_v_kick, \
        mass_1, mass_2, chi_1, chi_2

def merge_blackholes(blackholes_binary, blackholes_pro, blackholes_merged, bh_binary_id_num_merger,
                     smbh_mass, flag_use_surrogate, disk_aspect_ratio, disk_density, time_passed, galaxy):
    """Calculates parameters for merged BHs and adds them to blackholes_pro and blackholes_merged

    This function calculates the new parameters for merged BHs and adds them to the
    blackholes_pro and blackholes_merged objects. It does NOT delete them from blackholes_binary
    or update the filing_cabinet with the new information.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    blackholes_pro : AGNBlackHole
        Prograde black holes
    blackholes_merged : AGNMergedBlackHole
        Merged black holes
    bh_binary_id_num_merger : np.ndarray
        Array of BH ID numbers to be merged
    smbh_mass : float
        Mass [Msun] of SMBH
    flag_use_surrogate : int
        Flag to use surrogate model for kick calculations
    disk_aspect_ratio : function
        Disk aspect ratio at specified rg
    disk_density : function
        Disk density at specified rg
    time_passed : float
        Current timestep [yr] in disk
    galaxy : int
        Current galaxy iteration
    """

    bh_mass_merged = merged_mass(
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
    )

    bh_chi_eff_merged = chi_effective(
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_ang_mom")
    )

    bh_chi_p_merged = chi_p(
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_inc")
    )

    if flag_use_surrogate == 0:
        bh_spin_merged = merged_spin(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")
        )
        bh_spin_merged = checks.spin_check(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_2"),
            bh_spin_merged
        )
        bh_v_kick = analytical_kick_velocity(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")
        )

        bh_mass_1_20Hz = np.zeros(bh_binary_id_num_merger.size)
        bh_mass_2_20Hz = np.zeros(bh_binary_id_num_merger.size)
        bh_spin_1_20Hz = np.zeros(bh_binary_id_num_merger.size)
        bh_spin_2_20Hz = np.zeros(bh_binary_id_num_merger.size)
        bh_spin_angle_merged = np.zeros(bh_binary_id_num_merger.size)

    elif flag_use_surrogate == 1:
        #bh_v_kick = 200 #evolve_binary.velocity()
        surrogate = fit_modeler.GPRFitters.read_from_file(f"../src/mcfacts/inputs/data/surrogate.joblib")
        bh_mass_merged, bh_spin_merged, bh_spin_angle_merged, bh_v_kick, bh_mass_1_20Hz, bh_mass_2_20Hz, bh_spin_1_20Hz, bh_spin_2_20Hz = evolve_binary.surrogate(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
            len(blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")), # phi_1 - randomly set in the function file
            len(blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")), # phi_2 - randomly set in the function file
            1000, # binary seperation - in units of mass_1+mass_2 - shawn need to optimize seperation to speed up processing time
            [0, 0, 1], # binary inclination - cartesian coords
            len(blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")), # bin_phase - randomly set in the function file
            # the following three None values are any correction needed to the values
            None, # bin_orb_a
            None, # mass_smbh
            None, # spin_smbh
            surrogate
        )
    elif flag_use_surrogate == -1:
        # Call Davide's code
        bh_mass_merged, bh_spin_merged, bh_spin_angle_merged, bh_v_kick, bh_mass_1_20Hz, bh_mass_2_20Hz, bh_spin_1_20Hz, bh_spin_2_20Hz = merge_blackholes_precession(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_sep"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_ecc"),
            smbh_mass,
        )
    else:
        raise ValueError(f"Invalid option: flag_use_surrogate = {flag_use_surrogate}")

    bh_lum_shock = shock_luminosity(
        smbh_mass,
        bh_mass_merged,
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
        disk_aspect_ratio,
        disk_density,
        bh_v_kick)

    bh_lum_jet = jet_luminosity(
        bh_mass_merged,
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
        disk_density,
        disk_aspect_ratio,
        smbh_mass,
        bh_spin_merged,
        bh_v_kick)

    bh_orb_ecc_merged = merged_orb_ecc(blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
                                             np.full(bh_binary_id_num_merger.size, bh_v_kick),
                                             smbh_mass)

    # Append new merged BH to arrays of single BH locations, masses, spins, spin angles & gens
    blackholes_merged.add_blackholes(new_id_num=bh_binary_id_num_merger,
                                     new_galaxy=np.full(bh_binary_id_num_merger.size, galaxy),
                                     new_bin_orb_a=blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
                                     new_mass_final=bh_mass_merged,
                                     new_spin_final=bh_spin_merged,
                                     new_spin_angle_final=bh_spin_angle_merged,
                                     new_mass_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
                                     new_mass_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
                                     new_spin_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
                                     new_spin_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
                                     new_spin_angle_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
                                     new_spin_angle_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
                                     new_gen_1=blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_1"),
                                     new_gen_2=blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_2"),
                                     new_chi_eff=bh_chi_eff_merged,
                                     new_chi_p=bh_chi_p_merged,
                                     new_v_kick=bh_v_kick,
                                     new_mass_1_20Hz=bh_mass_1_20Hz,
                                     new_mass_2_20Hz=bh_mass_2_20Hz,
                                     new_spin_1_20Hz=bh_spin_1_20Hz,
                                     new_spin_2_20Hz=bh_spin_2_20Hz,
                                     new_lum_shock=bh_lum_shock,
                                     new_lum_jet=bh_lum_jet,
                                     new_time_merged=np.full(bh_binary_id_num_merger.size, time_passed))

    # New bh generation is max of generations involved in merger plus 1
    blackholes_pro.add_blackholes(new_mass=blackholes_merged.at_id_num(bh_binary_id_num_merger, "mass_final"),
                                  new_orb_a=blackholes_merged.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
                                  new_spin=blackholes_merged.at_id_num(bh_binary_id_num_merger, "spin_final"),
                                  new_spin_angle=np.zeros(bh_binary_id_num_merger.size),
                                  new_orb_inc=np.zeros(bh_binary_id_num_merger.size),
                                  new_orb_ang_mom=np.ones(bh_binary_id_num_merger.size),
                                  new_orb_ecc=bh_orb_ecc_merged,
                                  new_gen=np.maximum(blackholes_merged.at_id_num(bh_binary_id_num_merger, "gen_1"),
                                                     blackholes_merged.at_id_num(bh_binary_id_num_merger, "gen_2")) + 1.0,
                                  new_orb_arg_periapse=np.full(bh_binary_id_num_merger.size, -1.5),
                                  new_galaxy=np.full(bh_binary_id_num_merger.size, galaxy),
                                  new_time_passed=np.full(bh_binary_id_num_merger.size, time_passed),
                                  new_id_num=bh_binary_id_num_merger)

    return (blackholes_merged, blackholes_pro)


class ProcessesBinaryBlackHoleMergers(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Binary Black Hole Merging" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

        checks.binary_reality_check(sm, filing_cabinet, self.log)
        checks.flag_binary_mergers(sm, filing_cabinet)

        bh_binary_id_num_merger = blackholes_binary.id_num[blackholes_binary.flag_merging < 0]

        self.log("Merger ID Numbers")
        self.log(bh_binary_id_num_merger)

        bh_mass_merged = merged_mass(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")
        )

        bh_spin_merged = merged_spin(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")
        )

        bh_chi_eff_merged = chi_effective(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_ang_mom")
        )

        bh_chi_p_merged = chi_p(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2"),
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_inc")
        )

        if sm.flag_use_surrogate == 0:
            bh_v_kick = analytical_kick_velocity(
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_1"),
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "mass_2"),
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_1"),
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_2"),
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_1"),
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "spin_angle_2")
            )
        else:
            bh_v_kick = 200.

        bh_lum_shock = shock_luminosity(
            sm.smbh_mass,
            bh_mass_merged,
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
            agn_disk.disk_aspect_ratio,
            agn_disk.disk_density,
            bh_v_kick)

        bh_lum_jet = jet_luminosity(
            bh_mass_merged,
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
            agn_disk.disk_density,
            agn_disk.disk_aspect_ratio,
            sm.smbh_mass,
            bh_spin_merged,
            bh_v_kick)

        bh_orb_ecc_merged = merged_orb_ecc(
            blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
            np.full(bh_binary_id_num_merger.size, bh_v_kick),
            sm.smbh_mass)

        blackholes_merged = blackholes_binary.copy()
        blackholes_merged.keep_only(bh_binary_id_num_merger)

        blackholes_merged = AGNMergedBlackHoleArray(
            **blackholes_merged.get_super_dict(),
            mass_final=bh_mass_merged,
            spin_final=bh_spin_merged,
            spin_angle_final=np.zeros(bh_binary_id_num_merger.size, dtype=np.float64),
            chi_eff=bh_chi_eff_merged,
            chi_p=bh_chi_p_merged,
            lum_shock=np.array(bh_lum_shock, dtype=np.float64),
            lum_jet=np.array(bh_lum_jet, dtype=np.float64)
        )

        next_generation = np.maximum(
            blackholes_merged.at_id_num(bh_binary_id_num_merger, "gen_1"),
            blackholes_merged.at_id_num(bh_binary_id_num_merger, "gen_2")
        ) + int(1)

        new_blackholes = AGNBlackHoleArray(
            unique_id=np.array([uuid_provider(random_generator) for _ in range(bh_binary_id_num_merger.size)], dtype=uuid.UUID),
            mass=blackholes_merged.at_id_num(bh_binary_id_num_merger, "mass_final"),
            orb_a=blackholes_merged.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
            spin=blackholes_merged.at_id_num(bh_binary_id_num_merger, "spin_final"),
            spin_angle=np.zeros(bh_binary_id_num_merger.size, dtype=np.float64),
            orb_inc=np.zeros(bh_binary_id_num_merger.size, dtype=np.float64),
            orb_ang_mom=np.ones(bh_binary_id_num_merger.size, dtype=np.float64),
            orb_arg_periapse=np.full(bh_binary_id_num_merger.size, -1.5),
            orb_ecc=bh_orb_ecc_merged,
            gen=next_generation,
        )

        self.log(f"Number of mergers {len(blackholes_merged)}")

        # All new BH are prograde, so don't add them to the unsorted array
        filing_cabinet.create_or_append_array(sm.bh_prograde_array_name, new_blackholes)
        filing_cabinet.create_or_append_array(sm.bbh_merged_array_name, blackholes_merged)

        blackholes_binary.remove_all(bh_binary_id_num_merger)
