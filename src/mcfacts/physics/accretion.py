"""
Module for calculating change of mass, spin magnitude, and spin angle due to accretion.
"""

import numpy as np
import astropy.constants as const
import astropy.units as u
from mcfacts.physics.point_masses import si_from_r_g


def change_star_mass(disk_star_pro_masses,
                     disk_star_pro_orbs_a,
                     disk_star_pro_eccs,
                     disk_star_luminosity_factor,
                     disk_star_initial_mass_cutoff,
                     smbh_mass,
                     disk_sound_speed,
                     disk_density,
                     timestep_duration_yr):
    """Adds mass according to Fabj+2024 accretion rate

    Takes initial star masses at start of timestep and adds mass according to Fabj+2024.

    Parameters
    ----------
    disk_star_pro_masses : numpy.ndarray
        Initial masses [M_sun] of stars in prograde orbits around SMBH with :obj:`float` type.
    disk_star_eddington_ratio : float
        Accretion rate of fully embedded stars [Eddington accretion rate].
        1.0=embedded star accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    mdisk_star_eddington_mass_growth_rate : float
        Fractional rate of mass growth AT Eddington accretion rate per year (fixed at 2.3e-8 in mcfacts_sim) [yr^{-1}]
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    disk_star_pro_new_masses : numpy.ndarray
        Masses [M_sun] of stars after accreting at prescribed rate for one timestep [M_sun] with :obj:`float` type

    Notes
    -----
    Calculate Bondi radius: R_B = (2 G M_*)/(c_s **2) and Hill radius: R_Hill \\approx a(1-e)(M_*/(3(M_* + M_SMBH)))^(1/3).
    Accretion rate is Mdot = (pi/f) * rho * c_s * min[R_B, R_Hill]**2
    with f ~ 4 as luminosity dependent factor that accounts for the decrease of the accretion rate onto the star as it
    approaches the Eddington luminosity (see Cantiello+2021), rho as the disk density, and c_s as the sound speed.
    """

    # Put things in SI units
    star_masses_si = disk_star_pro_masses * u.solMass
    disk_sound_speed_si = disk_sound_speed(disk_star_pro_orbs_a) * u.meter/u.second
    disk_density_si = disk_density(disk_star_pro_orbs_a) * (u.kg / (u.m ** 3))
    timestep_duration_yr_si = timestep_duration_yr * u.year

    # Calculate Bondi and Hill radii
    r_bondi = (2 * const.G.to("m^3 / kg s^2") * star_masses_si / (disk_sound_speed_si ** 2)).to("meter")
    r_hill_rg = (disk_star_pro_orbs_a * (1 - disk_star_pro_eccs) * ((disk_star_pro_masses / (3 * (disk_star_pro_masses + smbh_mass))) ** (1./3.)))
    r_hill_m = si_from_r_g(smbh_mass, r_hill_rg)

    # Determine which is smaller for each star
    min_radius = np.minimum(r_bondi, r_hill_m)

    # Calculate the mass accretion rate
    mdot = ((np.pi / disk_star_luminosity_factor) * disk_density_si * disk_sound_speed_si * (min_radius ** 2)).to("kg/yr")

    # Accrete mass onto stars
    disk_star_pro_new_masses = ((star_masses_si + mdot * timestep_duration_yr_si).to("Msun")).value

    # Stars can't accrete over disk_star_initial_mass_cutoff
    disk_star_pro_new_masses[disk_star_pro_new_masses > disk_star_initial_mass_cutoff] = disk_star_initial_mass_cutoff

    return disk_star_pro_new_masses


def change_bh_mass(disk_bh_pro_masses, disk_bh_eddington_ratio, disk_bh_eddington_mass_growth_rate, timestep_duration_yr):
    """Adds mass according to chosen BH mass accretion prescription

    Takes initial BH masses at start of timestep and adds mass according to
    chosen BH mass accretion prescription

    Parameters
    ----------
    disk_bh_pro_masses : numpy.ndarray
        Initial masses [M_sun] of black holes in prograde orbits around SMBH :obj:`float` type
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    mdisk_bh_eddington_mass_growth_rate : float
        Fractional rate of mass growth [yr^{-1}] AT Eddington accretion rate per year (fixed at 2.3e-8 in mcfacts_sim)
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    disk_bh_pro_new_masses : numpy.ndarray
        Masses [M_sun] of black holes after accreting at prescribed rate for one timestep with :obj:`float` type
    """
    # Mass grows exponentially for length of timestep:
    disk_bh_pro_new_masses = disk_bh_pro_masses*np.exp(disk_bh_eddington_mass_growth_rate*disk_bh_eddington_ratio*timestep_duration_yr)

    return disk_bh_pro_new_masses


def change_bh_spin_magnitudes(disk_bh_pro_spins,
                              disk_bh_eddington_ratio,
                              disk_bh_torque_condition,
                              timestep_duration_yr,
                              disk_bh_pro_orbs_ecc,
                              disk_bh_pro_orbs_ecc_crit):
    """Updates the spin magnitude of the embedded black holes based on their accreted mass in this timestep.

    Parameters
    ----------
    disk_bh_pro_spins : numpy.ndarray
        Initial spins [unitless] of black holes in prograde orbits around SMBH
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    disk_bh_torque_condition : float
        Fraction of initial mass required to be accreted before BH spin is torqued fully into
        alignment with the AGN disk. We don't know for sure but Bogdanovic et al. says
        between 0.01=1% and 0.1=10% is what is required
        User chosen input set by input file
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of BH in prograde orbits around SMBH with :obj:`float` type
    disk_bh_pro_orbs_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which prograde accretion
        (& migration & binary formation) occurs
    Returns
    -------
    disk_bh_pro_spins_new : numpy.ndarray
        Spin magnitudes [unitless] of black holes after accreting at prescribed rate for one timestep with :obj:`float` type
    """
    # A retrograde BH a=-1 will spin down to a=0 when it accretes a factor sqrt(3/2)=1.22 in mass (Bardeen 1970).
    # Since M_edd/t = 2.3 e-8 M0/yr or 2.3e-4M0/10kyr then M(t)=M0*exp((M_edd/t)*f_edd*time)
    # so M(t)~1.2=M0*exp(0.2) so in 10^7yr, spin should go a=-1 to a=0. Or delta a ~ 10^-3 every 10^4yr.

    normalized_Eddington_ratio = disk_bh_eddington_ratio/1.0
    normalized_timestep = timestep_duration_yr/1.e4
    normalized_spin_torque_condition = disk_bh_torque_condition/0.1

    # Magnitude of spin iteration per normalized timestep
    spin_iteration = (1.e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)

    disk_bh_pro_spins_new = disk_bh_pro_spins

    # Singleton BH with orb_ecc > orb_ecc_crit will spin down bc accrete retrograde
    indices_bh_spin_down = np.asarray(disk_bh_pro_orbs_ecc > disk_bh_pro_orbs_ecc_crit).nonzero()[0]
    # Singleton BH with orb ecc < disk_star_pro_orbs_ecc_crit will spin up b/c accrete prograde
    indices_bh_spin_up = np.asarray(disk_bh_pro_orbs_ecc <= disk_bh_pro_orbs_ecc_crit).nonzero()[0]

    # disk_bh_pro_spins_new[prograde_orb_ang_mom_indices]=disk_bh_pro_spins_new[prograde_orb_ang_mom_indices]+(4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    disk_bh_pro_spins_new[indices_bh_spin_up] = disk_bh_pro_spins[indices_bh_spin_up] + spin_iteration
    # Spin down BH with orb ecc > disk_bh_pro_orbs_ecc_crit
    disk_bh_pro_spins_new[indices_bh_spin_down] = disk_bh_pro_spins[indices_bh_spin_down] - spin_iteration
    # Housekeeping: Max possible spins. Do not spin above or below these values
    disk_bh_pro_spin_max = 0.98
    disk_bh_pro_spin_min = -0.98

    disk_bh_pro_spins_new[disk_bh_pro_spins_new < disk_bh_pro_spin_min] = disk_bh_pro_spin_min
    disk_bh_pro_spins_new[disk_bh_pro_spins_new > disk_bh_pro_spin_max] = disk_bh_pro_spin_max

    return disk_bh_pro_spins_new


def change_bh_spin_angles(disk_bh_pro_spin_angles,
                          disk_bh_eddington_ratio,
                          disk_bh_torque_condition,
                          disk_bh_spin_minimum_resolution,
                          timestep_duration_yr,
                          disk_bh_pro_orbs_ecc,
                          disk_bh_pro_orbs_ecc_crit):
    """Updates the spin angles of the embedded black holes based on their accreted mass in this timestep.

    Parameters
    ----------
    disk_bh_pro_spin_angles : numpy.ndarray
        Initial spin angles [radian] of black holes in prograde orbits around SMBH with :obj:`float` type
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    disk_bh_torque_condition : float
        Fraction of initial mass required to be accreted before BH spin is torqued fully into
        alignment with the AGN disk. We don't know for sure but Bogdanovic et al. says
        between 0.01=1% and 0.1=10% is what is required
        User chosen input set by input file
    disk_bh_spin_minimum_resolution : float
        Minimum resolution of spin change followed by code [unitless]
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_bh_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of BH in prograde orbits around SMBH [unitless] with :obj:`float` type
    disk_bh_orbs_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which prograde accretion (& migration & binary formation) occurs
    Returns
    -------
    disk_bh_pro_spin_new : numpy.ndarray
        Spin angles [radian] of black holes after accreting at prescribed rate for one timestep with :obj:`float` type
    """

    # Calculate change in spin angle due to accretion during timestep
    normalized_Eddington_ratio = disk_bh_eddington_ratio/1.0
    normalized_timestep = timestep_duration_yr/1.e4
    normalized_spin_torque_condition = disk_bh_torque_condition/0.1

    spin_torque_iteration = (6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)

    # Assume same angles as before to start
    disk_bh_spin_angles_new = disk_bh_pro_spin_angles

    # Singleton BH with orb_ecc > orb_ecc_crit will spin down bc accrete retrograde
    indices_bh_spin_down = np.asarray(disk_bh_pro_orbs_ecc > disk_bh_pro_orbs_ecc_crit).nonzero()[0]
    # Singleton BH with orb ecc < disk_star_pro_orbs_ecc_crit will spin up b/c accrete prograde
    indices_bh_spin_up = np.asarray(disk_bh_pro_orbs_ecc <= disk_bh_pro_orbs_ecc_crit).nonzero()[0]


    # Spin up BH are torqued towards zero (ie alignment with disk, so decrease mag of spin angle)
    disk_bh_spin_angles_new[indices_bh_spin_up] = disk_bh_pro_spin_angles[indices_bh_spin_up] - spin_torque_iteration
    # Spin down BH with orb ecc > disk_bh_pro_orbs_ecc_crit are torqued toward anti-alignment with disk, incr mag of spin angle.
    disk_bh_spin_angles_new[indices_bh_spin_down] = disk_bh_pro_spin_angles[indices_bh_spin_down] + spin_torque_iteration
    # print(disk_bh_spin_angles_new[indices_bh_spin_down])

    # Housekeeping
    # Max bh spin angle in rads (pi rads = anti-alignment). Do not grow bh spin angle < 0 or > bh_max_spin_angle
    bh_max_spin_angle = 3.10
    disk_bh_spin_angles_new[disk_bh_spin_angles_new < disk_bh_spin_minimum_resolution] = 0.0
    disk_bh_spin_angles_new[disk_bh_spin_angles_new > bh_max_spin_angle] = bh_max_spin_angle

    return disk_bh_spin_angles_new
