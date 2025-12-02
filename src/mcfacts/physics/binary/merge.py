"""
Module for calculating the final variables of a merging binary.
"""
import warnings
import numpy as np
from scipy.stats import truncnorm
from astropy import units as u
from astropy import constants as const
from mcfacts.mcfacts_random_state import rng
from mcfacts.physics import analytical_velocity, lum
from mcfacts.external.sxs import evolve_binary
from mcfacts.external.sxs import fit_modeler
from mcfacts.physics.point_masses import si_from_r_g
#from mcfacts.inputs import data

from mcfacts.physics.point_masses import time_of_orbital_shrinkage, si_from_r_g


def chi_effective(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2, bin_ang_mom):
    """Calculates the effective spin :math:`\\chi_{\\rm eff}` associated with a merger.

    The measured effective spin of a merger is calculated as

    .. math:: \\chi_{\\rm eff}=\\frac{m_1\\chi_1\\cos(\\theta_1) + m_2\\chi_2\\cos(\\theta_2)}{m_{\\rm bin}} L_{\\rm bin}

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

    :code:`chi_p = max[spin_1_perp, (q(4q+3)/(4+3q)) * spin_2_perp]` where

    :code:`spin_1_perp = spin_1 * sin(spin_angle_1)` and

    :code:`spin_2_perp = spin_2 * sin(spin_angle_2)`

    are perpendicular to :code:`spin_1` and :code:`spin_2`, respectively,

    and :code:`q = mass_2 / mass_1` where :code:`mass_2 < mass_1`.

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

    # ====== DISSCUSSION NEEDED BEFORE CHANGES - WHY ADDING IN BIN_ORB_INC ======
    # Define spin angle to include binary inclination wrt disk (units of radians)
    spin_angles_1 = spin_angles_1 + bin_orbs_inc
    spin_angles_2 = spin_angles_2 + bin_orbs_inc

    # Make sure angles are < pi radians
    spin_angles_1_diffs = spin_angles_1 - np.pi
    spin_angles_2_diffs = spin_angles_2 - np.pi

    spin_angles_1[spin_angles_1_diffs > 0] = spin_angles_1[spin_angles_1_diffs > 0] - spin_angles_1_diffs[spin_angles_1_diffs > 0]
    spin_angles_2[spin_angles_2_diffs > 0] = spin_angles_2[spin_angles_2_diffs > 0] - spin_angles_2_diffs[spin_angles_2_diffs > 0]
    
    # temporary solution to the spin_angle values being set to a negative value
    spin_angles_1[spin_angles_1 < 0] = 0
    spin_angles_2[spin_angles_2 < 0] = 0

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
    #if any(chi_p < 0):
    #    print('mass1 :', masses_1)
    #    print('mass2 :', masses_2)
    #    print('spin1 :', spins_1)
    #    print('spin2 :', spins_2)
    #    print('spin_angle1 :', spin_angles_1)
    #    print('spin_angle2 :', spin_angles_2)
    #    print('bin_orb_inc :', bin_orbs_inc)
    #    raise ValueError("We have negative chi_p for some reason! EEK!")
    assert all(chi_p >= 0), \
        "We have negative chi_p for some reason! EEK!"

    return (chi_p)


def normalize_tgw(smbh_mass, inner_disk_outer_radius):
    """Normalizes Gravitational wave timescale.

    Calculate the normalization for timescale of a merger (in s) due to GW emission.
    From Peters (1964):

    .. math:: t_{\\rm gw} \\approx \\frac{5}{256} \\frac{c^5}{G^3} \\frac{a_b^4}{M_b^2\\mu_b}

    assuming eccentricity :math:`e=0.0`.

    For :math:`a_b` in units of :math:`r_g=GM_{\\rm SMBH}/c^2` we find

    .. math:: t_{\\rm gw}=\\frac{5}{256} \\frac{G}{c^3} \\left(\\frac{a_b}{r_g}\\right)^{4} \\frac{M_s^4}{M_b^2\\mu_b}

    Put :code:`bin_mass_ref` in units of :math:`10\\,M_\\odot` (is a reference mass).
    :code:`reduced_mass` in units of :math:`2.5\\,M_\\odot`.

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


def merged_mass(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2):
    """Calculates the final mass of a merged binary.

    Using approximations from Tichy \\& Maronetti (2008) where

    .. math:: m_{\\rm final}=(M_1+M_2)[1.0-0.2\\nu-0.208\\nu^2(a_1+a_2)]
    where :math:`\\nu` is the symmetric mass ratio or :math:`\\nu=q/((1+q)^2)`
    and :math:`q=M_2/M_1` with :math:`M_2<M_1`

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

    Only depends on :math:`M_1,M_2,a_1,a_2` and the binary ang mom around its center of mass.
    Using approximations from Tichy \\& Maronetti (2008)

    .. math::

        a_{\\rm final}=0.686(5.04\\nu-4.16\\nu^2) +0.4[a_1/(0.632+1/q)^2+ a_2/(0.632+q)^2]

    where :math:`q=M_2/M_1` and :math:`nu=q/((1+q)^2)`.

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

def generate_truncated_normal(mean=0, std=1, lower=0.75, upper=0.85, size=10):
    """ Random Guassian distribution generator 
    
    Parameters
    ----------
        mean : int
            Peak value of the distribution
        std : int
            Standard deiviation of the distriubtion
        lower : float 
            Lower bound of the distribution
        upper : float
            Upper bound of the distribution
        size : int
            Number of bins based on sample size
            
    Returns
    -------
        spin_dist : numpy.ndarray
            Random distribution of the spins within a set range
    """
    
    a = (lower - mean) / std
    b = (upper - mean) / std
    spin_dist = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    return spin_dist

def spin_check(gen_1, gen_2, spin_merged):
    """ Since the Tichy and Marronetti '08 perscription generates spin values outside of the expected range for higher mass ratio objects this file checks spin values after merger and if the magnitude is too low, this function resets it to a random distribution between a set range in order to generate results similiar to that of the NRsurrogate model.

    Parameters
    ----------        
        gen_1 : numpy.ndarray
            generation of m1 (before merger) (1=natal BH that has never been in a prior merger)
        gen_2 : numpy.darray
            generation of m2 (before merger) (1=natal BH that has never been in a prior merger)
        spin_merged : numpy.darray
            Final spin magnitude [unitless] of merger remnant with :obj:`float` type

    Returns
    -------
    merged_spins : numpy array
        Final spin magnitude [unitless] of merger remnant with :obj:`float` type
    """
    
    new_spin_merged = []
    
    for i in range(len(spin_merged)):
        # Sorting first gen objects and keeping their parameters
        if (gen_1[i] == 1.) & (gen_2[i] == 1.):
            #print('gen 1', spin_merged[i])
            new_spin_merged.append(spin_merged[i])
        # Sorting 2nd gen objects and updating their spins as needed otherwise keeping them the same
        # If spins < 0.75, they are reset to a randomly selected gaussian distribution between 0.75 - 0.85
        elif ((gen_1[i] == 2.) | (gen_2[i] == 2.)) & ((gen_1[i] <= 2.) & (gen_2[i] <= 2.)):
            #print('gen 2', spin_merged[i])
            if spin_merged[i] < 0.75:
                spin_plus_noise = generate_truncated_normal(mean=0, std=1, lower=0.75, upper=0.85, size=1)
                #print('gen 2 plus noise', spin_plus_noise)
                new_spin_merged.append(float(spin_plus_noise))
            else:
                #print('gen 2', spin_merged[i])
                new_spin_merged.append(spin_merged[i])
        # Sorting 3+ gen objects and updating their spins as needed otherwise keeping them the same
        # If spins < 0.85, they are reset to a randomly selected gaussian distribution between 0.85 - 0.95
        elif (gen_1[i] >= 3.) | (gen_2[i] >= 3.):
            #print('gen x', spin_merged[i])
            if spin_merged[i] < 0.85:
                spin_plus_noise = generate_truncated_normal(mean=0, std=1, lower=0.85, upper=0.95, size=1)
                #print('gen 3+ plus noise', spin_plus_noise)
                new_spin_merged.append(float(spin_plus_noise))
            else:
                #print('gen 3+', spin_merged[i])
                new_spin_merged.append(spin_merged[i])
                
    return np.array(new_spin_merged)
    
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
    orbs_a_units = si_from_r_g(smbh_mass * u.solMass, bin_orbs_a).to("meter")

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
    bh_thetaL = precession.remnantspindirection(
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
                     smbh_mass, flag_use_surrogate, flag_use_spin_check, disk_aspect_ratio, disk_density, disk_sound_speed, time_passed, galaxy):
    """Calculates parameters for merged BHs and adds them to :code:`blackholes_pro` and :code:`blackholes_merged`

    This function calculates the new parameters for merged BHs and adds them to the
    :code:`blackholes_pro` and :code:`blackholes_merged` objects. It does NOT delete them from :code:`blackholes_binary`
    or update the :code:`filing_cabinet` with the new information.

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
    flag_use_spin_check : int 
        Flag to apply spin_check filter to spin results
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
        if flag_use_spin_check == 1:
            bh_spin_merged = spin_check(
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_1"),
                blackholes_binary.at_id_num(bh_binary_id_num_merger, "gen_2"),
                bh_spin_merged
            )
        else:
            bh_spin_merged = bh_spin_merged
        bh_v_kick = analytical_velocity.analytical_kick_velocity(
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
        bh_mass_merged, bh_kick_comp_merged, bh_spin_merged, bh_spin_angle_merged, bh_v_kick, bh_mass_1_20Hz, bh_mass_2_20Hz, bh_spin_1_20Hz, bh_spin_2_20Hz = evolve_binary.surrogate(
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

    bh_lum_shock = lum.shock_luminosity(
        smbh_mass,
        bh_mass_merged,
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
        disk_aspect_ratio,
        disk_density,
        bh_v_kick)

    bh_lum_jet = lum.jet_luminosity(
        bh_mass_merged,
        blackholes_binary.at_id_num(bh_binary_id_num_merger, "bin_orb_a"),
        disk_density,
        bh_spin_merged,
        bh_v_kick,
        disk_sound_speed)

    # ====== Varun here is the function you're changing for the components. Replace the bh_v_kick --> bh_kick_comp_merged ======
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
    #raise Exception
    return (blackholes_merged, blackholes_pro)
