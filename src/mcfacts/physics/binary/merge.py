"""
Module for calculating the final variables of a merging binary.
"""
import numpy as np
from astropy import units as u

from mcfacts.physics.point_masses import time_of_orbital_shrinkage, si_from_r_g


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

    return (chi_p)


def normalize_tgw(smbh_mass, inner_disk_outer_radius):
    """Normalizes Gravitational wave timescale.

    Calculate the normalization for timescale of a merger (in yrs) due to GW emission.
    From Peters(1964):

    .. math:: t_{gw} \approx (5/256)* c^5/G^3 *a_b^4/(M_{b}^{2}mu_{b})
    assuming ecc=0.0.

    For a_b in units of r_g=GM_smbh/c^2 we find

    .. math:: t_{gw}=(5/256)*(G/c^3)*(a/r_g)^{4} *(M_s^4)/(M_b^{2}mu_b)

    Put bin_mass_ref in units of 10Msun (is a reference mass).
    reduced_mass in units of 2.5Msun.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    inner_disk_outer_radius : float
        Outer radius of the inner disk [r_g]

    Returns
    -------
    time_gw_normalization : float
        Normalization to gravitational wave timescale [s]
    """

    bin_mass_ref = 10.0
    '''
    G = const.G
    c = const.c
    mass_sun = const.M_sun
    year = 3.1536e7
    reduced_mass = 2.5
    norm = (5.0/256.0)*(G/(c**(3)))*(smbh_mass**(4))*mass_sun/((bin_mass_ref**(2))*reduced_mass)
    time_gw_normalization = norm/year
    '''
    time_gw_normalization = time_of_orbital_shrinkage(
        smbh_mass * u.solMass,
        bin_mass_ref * u.solMass,
        si_from_r_g(smbh_mass * u.solMass, inner_disk_outer_radius),
        0 * u.m,
    )
    return time_gw_normalization.si.value


def merged_mass(masses_1, masses_2, spins_1, spins_2):
    """Calculates the final mass of a merged binary.

    Using approximations from Tichy \\& Maronetti (2008) where
    m_final=(M_1+M_2)*[1.0-0.2\nu-0.208\nu^2(a_1+a_2)]
    where nu is the symmetric mass ratio or nu=q/((1+q)^2)

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

    Returns
    -------
    merged_masses: numpy.ndarray
        Final mass [M_sun] of merger remnant with :obj:`float` type
    """

    mass_ratios = np.ones(masses_1.size)
    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_1 < masses_2] = masses_1[masses_1 < masses_2] / masses_2[masses_1 < masses_2]

    total_masses = masses_1 + masses_2
    total_spins = spins_1 + spins_2
    nu_factors = (1.0 + mass_ratios) ** 2.0
    nu = mass_ratios / nu_factors
    nu_squared = nu * nu

    mass_factors = 1.0 - (0.2 * nu) - (0.208 * nu_squared * total_spins)
    merged_masses = total_masses*mass_factors

    return (merged_masses)


def merged_spin(masses_1, masses_2, spins_1, spins_2):
    """Calculates the spin magnitude of a merged binary.

    Only depends on M1,M2,a1,a2 and the binary ang mom around its center of mass.
    Using approximations from Tichy \\& Maronetti(2008) where
    :math:`a_{final}=0.686(5.04\nu-4.16\nu^2) +0.4[a_{1}/((0.632+1/q)^2)+ a_2/((0.632+q)^2)]`
    where q=m_2/m_1 and nu=q/((1+q)^2)

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

    Returns
    -------
    merged_spins : numpy array
        Final spin magnitude [unitless] of merger remnant with :obj:`float` type
    """

    mass_ratios = np.ones(masses_1.size)
    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_1 < masses_2] = masses_1[masses_1 < masses_2] / masses_2[masses_1 < masses_2]

    mass_ratios_inv = 1.0 / mass_ratios

    nu_factors = (1.0 + mass_ratios) ** 2.0
    nu = mass_ratios / nu_factors
    nu_squared = nu * nu

    spin_factors_1 = (0.632 + mass_ratios_inv) ** 2.0
    spin_factors_2 = (0.632 + mass_ratios) ** 2.0

    merged_spins = 0.686 * ((5.04 * nu) - (4.16 * nu_squared)) + (0.4 * ((spins_1 / spin_factors_1) + (spins_2 / spin_factors_2)))

    return (merged_spins)
