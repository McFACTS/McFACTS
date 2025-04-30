"""Unit Test Variable Provider
==========

This module provides a central location for unit tests to retrieve variables and known values.
"""
from enum import Enum

import numpy as np
import pytest
import scipy
import astropy.units as u

from mcfacts.inputs.ReadInputs import load_disk_arrays, construct_disk_direct

TEST_SEED = 314159

class InputParameterSet(Enum):
    """Input Parameter Set"""
    BASE = 1
    SINGLETON = 2
    BINARY = 3
    DYNAMICS = 4


INPUT_PARAMETERS = {
    "smbh_mass": {
        InputParameterSet.BASE: [1E5, 1E6, 1E7, 3E7, 1E8, 1E9],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "disk_radius_outer": {
        InputParameterSet.BASE: [50000],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "nsc_radius_crit": {
        InputParameterSet.BASE: [0.25],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "nsc_density_index_inner": {
        InputParameterSet.BASE: [1.75],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "nsc_density_index_outer": {
        InputParameterSet.BASE: [2.5],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_masses": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [3., 5., 10., 20., 35., 40., 50., 70., 100.],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_spins": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: np.append(np.linspace(-.98, .98, 10), 0.),
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_spin_angles": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: np.linspace(0., np.pi, 10),
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_spin_resolution": { # Convention
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [0.02],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_torque_condition": { # 0.01 to 0.1 AND test bad case > 0.1
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: np.linspace(0.01, 0.1, 10),
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_eccentricity": { # 0. to 1. - 1.0E-8
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [0, .1, .3, .5, .7, .9, .999, .99999, .9999999, 1 - 1e-9],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_inclination": { # -2pi to 2pi: 0, pi ,-pi
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [0, np.pi, -np.pi, np.pi/2, -np.pi/2],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_semi_major_axis_inner": { # 0 - 50 Rg
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: np.linspace(19, 50, 10),
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_semi_major_axis_outer": { # 50 - 10E6 Rg
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: np.linspace(50, 1E6, 20),
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_critical_eccentricity": { # 0.01, 0.05, 0.001
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [0.01, 0.05, 0.001],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_eddington_ratio": { # set at 1.0: testing 0.1 to 10
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: np.append(np.linspace(0.1, 10, 10), 1.0),
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_eddington_ratio_mass_growth_rate": { # Convention: 2.3e-8 NOTE:
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [2.3e-8],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_angular_momenta": { # +1 and -1 only these: test 0, should break
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [1, -1],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_argument_periapse": { # 0, pi, pi/2
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [0, np.pi, np.pi/2],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },

    "star_masses": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "star_spins": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "star_torque_condition": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "star_orbital_eccentricity": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "star_orbital_inclination": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "star_orbital_semi_major_axis": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "star_orbital_critical_eccentricity": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "star_eddington_ratio": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },

    "timestep_duration_yr": { # 1E4, 1E2, 1E3, 1E4, 1E5
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [1E4, 1E2, 1E3, 1E4, 1E5],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "simulation_time_passed": { # X timestep_duration_yr: test 0, 1, 2, 10, 50, 100
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [0, 1, 2, 10, 50, 100],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "gw_normalization_time": { # Based on SMBH mass
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "redshift": { # 0, .1, .5, 1, 2
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [0, .1, .5, 1, 2],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    }
}

INPUT_PARAMETER_UNITS = {
    "bh_masses": u.Msun,
    "timestep_duration_yr": u.year
}


def get_binary_array(parameter_set):
    """Generate test binary array for a given parameter set

    Parameters
    ----------
    parameter_set : string
        parameter set name given by `INPUT_PARAMETER_SETS`
    """
    # TODO: Generate binary array based on parameter set

    return []


def get_surface_density_func(disk_model_name):
    """Generate test surface density function for a given disk model name

        Parameters
        ----------
        disk_model_name : string
            Can be one of the follow options
            - sirko_goodman: from Sirko & Goodman 2003
            - thompson_etal: from Thompson, Quataert & Murray 2005
    """

    disk_model_radii, surface_densities, aspect_ratios, opacities = load_disk_arrays(disk_model_name, 50000)
    surf_dens_func_log = scipy.interpolate.CubicSpline(np.log(disk_model_radii), np.log(surface_densities))

    return lambda x, f=surf_dens_func_log: np.exp(f(x))


def get_disk_opacity_func(disk_model_name):
    """Generate test disk opacity function for a given disk model name

        Parameters
        ----------
        disk_model_name : string
            Can be one of the follow options
            - sirko_goodman: from Sirko & Goodman 2003
            - thompson_etal: from Thompson, Quataert & Murray 2005
    """

    disk_model_radii, surface_densities, aspect_ratios, opacities = load_disk_arrays(disk_model_name, 50000)
    disk_opacity_func_log = scipy.interpolate.CubicSpline(np.log(disk_model_radii), np.log(opacities))

    return lambda x, f=disk_opacity_func_log: np.exp(f(np.log(x)))


# if __name__ == "__main__":
#     print(np.array(INPUT_PARAMETERS["timestep_duration_yr"]) * INPUT_PARAMETER_UNITS["timestep_duration_yr"])
