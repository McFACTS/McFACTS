"""
Module for calculating the gw strain and freq of a binary.
"""
import numpy as np
from astropy import units as u, constants as const
from astropy.units import cds


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
    flag_include_old_gw_freq : boolean
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
        strain_factor[nu_gw > (1e-6) * u.Hz] = np.sqrt((nu_squared[nu_gw > (1e-6) * u.Hz] / delta_nu_delta_timestep[nu_gw > (1e-6) * u.Hz]) / 8.)
    # Condition from evolve_gw
    elif (flag_include_old_gw_freq == 0):
        strain_factor[nu_gw > (1e-6) * u.Hz] = 4.e3
    char_strain = strain_factor*strain

    return (char_strain.value, nu_gw.value)


def evolve_gw(blackholes_binary, smbh_mass, agn_redshift):
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

    char_strain, nu_gw = gw_strain_freq(mass_1=blackholes_binary.mass_1,
                                        mass_2=blackholes_binary.mass_2,
                                        obj_sep=blackholes_binary.bin_sep,
                                        timestep_duration_yr=-1,
                                        old_gw_freq=-1,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=0)

    # Update binaries
    blackholes_binary.gw_freq = nu_gw
    blackholes_binary.gw_strain = char_strain

    return (blackholes_binary)


def bbh_gw_params(blackholes_binary, bh_binary_id_num_gw, smbh_mass, timestep_duration_yr, old_bbh_freq, agn_redshift):
    """Wrapper function to calculate GW strain and frequency for BBH at the end of each timestep

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    bh_binary_id_num_gw : numpy.ndarray
        ID numbers of binaries with separations below :math:`\\mathtt{min_bbh_gw_separation}` with :obj:`float` type
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

    num_tracked = bh_binary_id_num_gw.size

    old_bbh_freq = old_bbh_freq * u.Hz

    while (num_tracked > len(old_bbh_freq)):
        old_bbh_freq = np.append(old_bbh_freq, (9.e-7) * u.Hz)

    while (num_tracked < len(old_bbh_freq)):
        old_bbh_freq = np.delete(old_bbh_freq, 0)

    char_strain, nu_gw = gw_strain_freq(mass_1=blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_1"),
                                        mass_2=blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_2"),
                                        obj_sep=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_sep"),
                                        timestep_duration_yr=timestep_duration_yr,
                                        old_gw_freq=old_bbh_freq,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=1)

    return (char_strain, nu_gw)
