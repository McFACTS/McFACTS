"""
Module for calculating the gw strain and freq of a binary and handling simple GR orbital evolution (Peters 1964)

Contain functions for orbital evolution and converting between
    units of r_g and SI units
"""
import numpy as np
from astropy import units as u, constants as const
from astropy.units import cds
from numpy.random import Generator

from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBinaryBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities import unit_conversion, peters


def gw_strain_freq(mass_1, mass_2, obj_sep, timestep_duration_yr, old_gw_freq, smbh_mass, agn_redshift, flag_include_old_gw_freq=1):
    """Calculates GW strain [unitless] and frequency [Hz]

    This function takes in two masses, their separation, the previous frequency, and the redshift and
    calculates the new GW strain (unitless) and frequency (Hz).

    Parameters
    ----------
    mass_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    mass_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    obj_sep : numpy.ndarray
        Separation between both objects [r_{g,SMBH}] with :obj:`float` type
    timestep_duration_yr : float, or -1 if not given
        Current timestep [yr]
    old_gw_freq : numpy.ndarray, or -1 if not given
        Previous GW frequency [Hz] with :obj:`float` type
    smbh_mass : float
        Mass [M_sun] of the SMBH
    agn_redshift : float
        Redshift [unitless] of the SMBH
    flag_include_old_gw_freq : int
        Flag indicating if old_gw_freq should be included in calculations
        if not, we use the hardcoded value (see note below)
        0 if no, 1 if yes

    Returns
    -------
    char_strain : numpy.ndarray
        Characteristic strain [unitless] with :obj:`float` type
    nu_gw : numpy.ndarray
        GW frequency [Hz] with :obj:`float` type

    Notes
    -----
    Note from Saavik about hardcoding strain_factor to 4e3 if nu_gw > 1e-6:
    basically we are implicitly assuming if the frequency is low enough the source is monochromatic
    in LISA over the course of 1yr, so that's where those values come from... and we do need to make
    a decision about that... and that's an ok decision for now. But if someone were to be considering
    a different observatory they might not like that decision?

    """

    redshift_d_obs_dict = {0.1: 421 * u.Mpc,
                           0.5: 1909 * u.Mpc}

    timestep_units = (timestep_duration_yr*u.year).to(u.second)

    # 1rg =1AU=1.5e11m for 1e8Msun
    rg = 1.5e11 * (smbh_mass/1.e8) * u.meter
    mass_1 = (mass_1 * cds.Msun).to(u.kg)
    mass_2 = (mass_2 * cds.Msun).to(u.kg)
    mass_total = mass_1 + mass_2
    bin_sep = obj_sep * rg

    mass_chirp = ((mass_1 * mass_2) ** (3./5.)) / (mass_total ** (1./5.))
    rg_chirp = ((const.G * mass_chirp) / (const.c ** 2.0)).to(u.meter)

    # If separation is less than rg_chirp then cap separation at rg_chirp.
    bin_sep[bin_sep < rg_chirp] = rg_chirp[bin_sep < rg_chirp]

    nu_gw = (1.0/np.pi) * np.sqrt(mass_total * const.G / (bin_sep ** 3.0))
    nu_gw = nu_gw.to(u.Hz)

    # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
    # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
    # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
    d_obs = redshift_d_obs_dict[agn_redshift].to(u.meter)
    strain = (4/d_obs) * rg_chirp * ((np.pi * nu_gw * rg_chirp / const.c) ** (2./3.))

    # But power builds up in band over multiple cycles!
    # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
    strain_factor = np.ones(len(nu_gw))
    #count LISA-LIGO binaries
    gwb = np.count_nonzero (nu_gw > 1.e-6 *u.Hz)
    # of which count tight binaries (nu_gw >2.e-3 Hz)
    tight = nu_gw >2.e-3 *u.Hz
    # char amplitude = strain_factor*h0
    #                = sqrt(N/8)*h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
    strain_factor[nu_gw < (1e-6) * u.Hz] = np.sqrt(nu_gw[nu_gw < (1e-6) * u.Hz] * np.pi * (1e7) / 8)

    # For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt).
    # char amplitude = strain_factor*h0
    #                = sqrt(freq^2/(dfreq/dt)/8)*h0
    if (flag_include_old_gw_freq == 1):
        delta_nu = np.abs(old_gw_freq - nu_gw)
        delta_nu_delta_timestep = delta_nu/timestep_units
        nu_squared = (nu_gw * nu_gw)
        nu_factor = (nu_gw)**(-5./6.)
        strain_factor[~tight] = np.sqrt((nu_squared[~tight] / delta_nu_delta_timestep[~tight]) / 8.)
        # If BBH merges within next timestep then nu_gw>2e-3 Hz
        #  N = (1/8pi)sqrt(5/96)(1/pi)^1/3 (c/r_g_chirp)^5/6 freq^-5/6
        #for i in range (0,tight):
        #if nu_gw.any(> 2.e-3 * u.Hz) :
        #print ('tight', tight)
        if np.any(tight):
            num_factor = np.sqrt(5./96.)*(1/(8*np.pi))*(1/np.pi)**(1./3.)
            #rg_chirp = ((const.G * mass_chirp) / (const.c ** 2.0)).to(u.meter)

            strain_factor[tight] = num_factor * ((const.c/rg_chirp[tight])**(5./6.)) * (nu_gw[tight])**(-5./6.)
    

    # Condition from evolve_gw
    elif (flag_include_old_gw_freq == 0):
        strain_factor[nu_gw > (1e-6) * u.Hz] = 4.e3
    char_strain = strain_factor*strain

    assert np.isfinite(char_strain.value).all(), \
        "Finite check failure: char_strain.value"
    assert np.isfinite(nu_gw.value).all(), \
        "Finite check failure: nu_gw.value"

    return (char_strain.value, nu_gw.value)


def evolve_gw(bin_mass_1, bin_mass_2, bin_sep, smbh_mass, agn_redshift):
    """Wrapper function to calculate GW strain [unitless] and frequency [Hz] for BBH with no previous GW frequency

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    smbh_mass : float
        Mass [M_sun] of the SMBH
    agn_redshift : float
        Redshift [unitless] of the SMBH

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        BBH with GW strain [unitless] and frequency [Hz] updated
    """

    char_strain, nu_gw = gw_strain_freq(mass_1=bin_mass_1,
                                        mass_2=bin_mass_2,
                                        obj_sep=bin_sep,
                                        timestep_duration_yr=-1,
                                        old_gw_freq=-1,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=0)

    return (nu_gw, char_strain)


def bbh_gw_params(bin_mass_1, bin_mass_2, bin_sep, smbh_mass, timestep_duration_yr, old_bbh_freq, agn_redshift):
    """Wrapper function to calculate GW strain and frequency for BBH at the end of each timestep

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    bh_binary_id_num_gw : numpy.ndarray
        ID numbers of binaries with separations below :math:`mathtt_{min_bbh_gw_separation}` with :obj:`float` type
    smbh_mass : float
        Mass [M_sun] of the SMBH
    timestep_duration_yr : float
        Length of timestep [yr]
    old_bbh_freq : numpy.ndarray
        Previous GW frequency [Hz] with :obj:`float` type
    agn_redshift : float
        Redshift [unitless] of the AGN, used to set d_obs

    Returns
    -------
    char_strain : numpy.ndarray
        Characteristic strain [unitless] with :obj:`float` type
    nu_gw : numpy.ndarray
        GW frequency [Hz] with :obj:`float` type
    """

    num_tracked = bin_mass_1.size

    old_bbh_freq = old_bbh_freq * u.Hz

    # while (num_tracked > len(old_bbh_freq)):
    #     old_bbh_freq = np.append(old_bbh_freq, (9.e-7) * u.Hz)
    #
    # while (num_tracked < len(old_bbh_freq)):
    #     old_bbh_freq = np.delete(old_bbh_freq, 0)

    char_strain, nu_gw = gw_strain_freq(mass_1=bin_mass_1,
                                        mass_2=bin_mass_2,
                                        obj_sep=bin_sep,
                                        timestep_duration_yr=timestep_duration_yr,
                                        old_gw_freq=old_bbh_freq,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=1)

    return (char_strain, nu_gw)


def orbital_separation_evolve(mass_1, mass_2, sep_initial, evolve_time):
    """Calculates the final separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy.units.quantity.Quantity
        Mass of object 1
    mass_2 : astropy.units.quantity.Quantity
        Mass of object 2
    sep_initial : astropy.units.quantity.Quantity
        Initial separation of two bodies
    evolve_time : astropy.units.quantity.Quantity
        Time to evolve GW orbit

    Returns
    -------
    sep_final : astropy.units.quantity.Quantity
        Final separation [m] of two bodies
    """
    # Calculate c and G in SI
    c = const.c.to('m/s').value
    G = const.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_initial = sep_initial.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const_g_c = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const_g_c * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_initial ** 4) - (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_final = np.zeros_like(sep_initial)
    sep_final[quantity > 0] = np.sqrt(np.sqrt(quantity[quantity > 0]))

    assert np.isfinite(sep_final).all(), \
        "Finite check failure: sep_final"
    assert np.all(sep_final > 0), \
        "sep_final contains values <= 0"

    return sep_final * u.m


def orbital_separation_evolve_reverse(mass_1, mass_2, sep_final, evolve_time):
    """Calculates the initial separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy.units.quantity.Quantity
        Mass of object 1
    mass_2 : astropy.units.quantity.Quantity
        Mass of object 2
    sep_final : astropy.units.quantity.Quantity
        Final separation of two bodies
    evolve_time : astropy.units.quantity.Quantity
        Time to evolve GW orbit

    Returns
    -------
    sep_initial : astropy.units.quantity.Quantity
        Initial separation [m] of two bodies
    """
    # Calculate c and G in SI
    c = const.c.to('m/s').value
    G = const.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_final = sep_final.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const_g_c = ((64 / 5) * (G ** 3)) * (c ** -5)
    # Calculate the beta array
    beta_arr = const_g_c * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_final ** 4) + (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_initial = np.sqrt(np.sqrt(quantity))

    assert np.isfinite(sep_initial).all(), \
        "Finite check failure: sep_initial"
    assert np.all(sep_initial > 0), \
        "sep_initial contains values <= 0"

    return sep_initial * u.m


def evolve_emri_gw(blackholes_inner_disk, timestep_duration_yr, old_gw_freq, smbh_mass, agn_redshift):
    """Evaluates the EMRI gravitational wave frequency and strain at the end of each timestep_duration_yr

    Parameters
    ----------
    blackholes_inner_disk : AGNBlackHole
        Parameters of black holes in the inner disk
    timestep_duration_yr : float
        Length of timestep [yr]
    old_gw_freq : numpy.ndarray
        Previous GW frequency [Hz] with :obj:`float` type
    smbh_mass : float
        Mass [M_sun] of the SMBH
    agn_redshift : float
        Redshift [unitless] of the AGN
    """

    old_gw_freq = old_gw_freq * u.Hz

    # If number of EMRIs has grown since last timestep_duration_yr, add a new component to old_gw_freq to carry out dnu/dt calculation
    # while (blackholes_inner_disk.num < len(old_gw_freq)):
    #     old_gw_freq = np.delete(old_gw_freq, 0)
    # while blackholes_inner_disk.num > len(old_gw_freq):
    #     old_gw_freq = np.append(old_gw_freq, (9.e-7) * u.Hz)

    char_strain, nu_gw = gw_strain_freq(mass_1=smbh_mass,
                                        mass_2=blackholes_inner_disk.mass,
                                        obj_sep=blackholes_inner_disk.orb_a,
                                        timestep_duration_yr=timestep_duration_yr,
                                        old_gw_freq=old_gw_freq,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=1)

    return (char_strain, nu_gw)


def normalize_tgw(smbh_mass, inner_disk_outer_radius):
    """Normalizes Gravitational wave timescale.

    Calculate the normalization for timescale of a merger (in s) due to GW emission.
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
    time_gw_normalization = peters.time_of_orbital_shrinkage(
        smbh_mass * u.solMass,
        bin_mass_ref * u.solMass,
        unit_conversion.si_from_r_g(smbh_mass * u.solMass, inner_disk_outer_radius),
        0 * u.m,
    )
    return time_gw_normalization.si.value


class BinaryBlackHoleEvolveGW(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Binary Black Hole Evolve GW" if name is None else name, settings)

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator) -> None:
        sm = self.settings

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBinaryBlackHoleArray)

        blackholes_binary.gw_freq, blackholes_binary.gw_strain = evolve_gw(
            blackholes_binary.mass_1,
            blackholes_binary.mass_2,
            blackholes_binary.bin_sep,
            sm.smbh_mass,
            sm.agn_redshift
        )

        blackholes_binary.consistency_check()



