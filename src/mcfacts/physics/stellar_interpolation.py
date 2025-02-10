"""
Module for interpolating stellar radius, luminosity, and effective temperature from a grid.
"""

import numpy as np
import astropy.constants as const

from mcfacts.inputs import data as mcfacts_input_data
from importlib import resources as impresources

# This is definitely the wrong and lazy way to go about things
# 0: mass
# 1: log(R)
# 2: log(L)
# 3: log(Teff)
fname_interp_data = impresources.files(mcfacts_input_data) / "stellar_grid/stellar_grid.txt"
interpolation_data = np.loadtxt(fname_interp_data)
interpolation_masses = interpolation_data[:, 0]


def interpolate_values(mhigh_value, mlow_value, ratio):
    """Interpolate between two values

    Parameters
    ----------
    mhigh_value : float
        Value associated with the higher mass grid star
    mlow_value : float
        Value associated with the lower mass grid star
    ratio : numpy.ndarray
        Ratio between the star mass and the grid star masses np.log10(mass / low_mass) / (np.log10(high_mass / low_mass)) with :obj:`float` type

    Returns
    -------
    new_values : numpy.ndarray
        New interpolated values with :obj:`float` type
    """
    # Amount to adjust the lower grid value by
    diffs = np.abs(mhigh_value - mlow_value)*ratio

    # Set up array for new values
    new_values = np.full(len(ratio), -100.5)

    # If value associated with low mass grid star is greater than value associated with high mass grid star we subtract the diff
    # and vice versa
    if ((mlow_value - mhigh_value) > 0):
        new_values = mlow_value - diffs
    elif ((mlow_value - mhigh_value) < 0):
        new_values = mlow_value + diffs
    else:
        raise SyntaxError("mlow_value == mhigh_value")

    return (new_values)


def interp_star_params(disk_star_masses):
    """Interpolate star radii, luminosity, and effective temperature in logspace

    Parameters
    ----------
    disk_star_masses : numpy.ndarray
        Masses of stars to be interpolated with :obj:`float` type

    Returns
    -------
    new_logR, new_logL, new_logTeff : numpy.ndarray
        Arrays of new interpolated radii, luminosities, and effective temperatures
    """

    # Set up arrays for new values
    new_logR = np.full(len(disk_star_masses), -100.5)
    new_logL = np.full(len(disk_star_masses), -100.5)
    new_logTeff = np.full(len(disk_star_masses), -100.5)

    # Interpolates values for stars with masses between the grid (0.8Msun to 298.838913Msun [grid is not exact])
    for i in range(0, len(interpolation_masses) - 1):
        mass_range_idx = np.asarray((disk_star_masses > interpolation_masses[i]) & (disk_star_masses <= interpolation_masses[i + 1])).nonzero()[0]

        if (len(mass_range_idx) > 0):

            ratio = np.log10(disk_star_masses[mass_range_idx] / interpolation_masses[i]) / np.log10(interpolation_masses[i + 1] / interpolation_masses[i])

            new_logR[mass_range_idx] = interpolate_values(interpolation_data[i + 1][1], interpolation_data[i][1], ratio)
            new_logL[mass_range_idx] = interpolate_values(interpolation_data[i + 1][2], interpolation_data[i][2], ratio)
            new_logTeff[mass_range_idx] = interpolate_values(interpolation_data[i + 1][3], interpolation_data[i][3], ratio)

    # Using homology relations for stars with masses <= 0.8Msun
    # From K&W, mu relations go away because chemical comp is the same
    # Eqn 20.20: L/L' = (M/M')^3 (mu/mu')^4 --> logL = log10((M/M')^3 * L')
    # Eqn 20.21: R/R' = (M/M')^z1 (mu/mu')^z2 --> logR = log10((M/M')^z1 * R')
    # Eqn 20.22: Teff^4 = L/(4 pi sigma_sb R^2)
    # z1 ~ 0.43 for nu = 4.
    # X = 0.7064, Y = 0.2735, and Z = 0.02

    #star_X = 0.7064
    #star_Y = 0.2735

    #mean_mol_weight = 4./(6. * star_X + star_Y + 2.)

    mass_mask = (disk_star_masses <= interpolation_masses.min()) | (disk_star_masses > interpolation_masses.max())

    if (np.sum(mass_mask) > 0):
        z1 = 0.43

        new_logL[mass_mask] = np.log10(((disk_star_masses[mass_mask] / interpolation_masses.min()) ** 3.) * (10 ** interpolation_data[0][2]))
        new_logR[mass_mask] = np.log10(((disk_star_masses[mass_mask] / interpolation_masses.min()) ** z1) * (10 ** interpolation_data[0][1]))
        L_units = (10 ** new_logL[mass_mask]) * const.L_sun
        R_units = (10 ** new_logR[mass_mask]) * const.R_sun
        lowmass_Teff = ((L_units / (4. * np.pi * const.sigma_sb * (R_units ** 2))) ** (1./4.)).to("Kelvin")
        new_logTeff[mass_mask] = np.log10(lowmass_Teff.value)

    logl_mask = new_logL < -25
    logr_mask = new_logR < -25
    logt_mask = new_logTeff < -25
    if (np.sum(logl_mask) > 0 | np.sum(logr_mask) > 0 | np.sum(logt_mask) > 0):
        raise ValueError("Interpolated values are not being set properly!")

    return (new_logR, new_logL, new_logTeff)
