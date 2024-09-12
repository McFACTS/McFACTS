"""Unit Test Variable Provider
==========

This module provides a central location for unit tests to retrieve variables and known values.
"""
from enum import Enum

import numpy as np
import scipy
from astropy import units as u

TEST_SEED = 314159


class InputParameterSet(Enum):
    """Input Parameter Set"""
    BASE = 1
    SINGLETON = 2
    BINARY = 3
    DYNAMICS = 4


INPUT_PARAMETERS = {
    "smbh_mass": {
        InputParameterSet.BASE: [10E5, 10E6, 10E7, 10E8],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },

    "bh_masses": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_spins": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_spin_angles": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_spin_resolution": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_torque_condition": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_eccentricity": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_inclination": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_semi_major_axis": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_orbital_critical_eccentricity": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_eddington_ratio": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_eddington_ratio_mass_growth_rate": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_angular_momenta": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "bh_argument_periapse": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
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

    "timestep_duration_yr": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "simulation_time_passed": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },
    "gw_normalization_time": {
        InputParameterSet.BASE: [],
        InputParameterSet.SINGLETON: [],
        InputParameterSet.BINARY: [],
        InputParameterSet.DYNAMICS: []
    },

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


def get_surface_density_func(parameter_set):
    """Generate test surface density function for a given parameter set

        Parameters
        ----------
        parameter_set : string
            parameter set name given by `INPUT_PARAMETER_SETS`
    """

    disk_model_radius_array = INPUT_PARAMETERS["disk_model_radius"][parameter_set]
    surface_density_array = INPUT_PARAMETERS["disk_surface_density"][parameter_set]

    surf_dens_func_log = scipy.interpolate.UnivariateSpline(disk_model_radius_array, np.log(surface_density_array))

    return lambda x, f=surf_dens_func_log: np.exp(f(x))


if __name__ == "__main__":
    print(np.array(INPUT_PARAMETERS["timestep_duration_yr"]) * INPUT_PARAMETER_UNITS["timestep_duration_yr"])
