"""Handling of the McFACTS AGNDisk object"""
######## Imports ########
#### Standard Library ####
from abc import ABC, abstractmethod
#### Third Party ####
import numpy as np
import pagn.constants as pagn_ct
#### Local ####
from mcfacts.objects.cache import readonly_cached_property
from mcfacts.objects.interp import CubicSpline, dCubicSpline
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.inputs.settings_manager import SettingsManager
import mcfacts.external.DiskModelsPAGN as pagn_dm

######## Disk object ########
class AGNDisk(ABC):
    """An object for managing the AGN Disk model
    """
    def __init__(self, settings : SettingsManager):
        if not isinstance(settings, SettingsManager):
            raise TypeError(f"argument settings is not a {SettingsManager}")
        self.settings = settings

    @abstractmethod
    def disk_surface_density(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_aspect_ratio(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_opacity(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_sound_speed(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_density(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_pressure_grad(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_omega(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_surface_density_log(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def temp_func(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_dlog10surfdens_dlog10R_func(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_dlog10temp_dlog10R_func(self, orb_a):
        raise NotImplementedError()

    @abstractmethod
    def disk_dlog10pressure_dlog10R_func(self, orb_a):
        raise NotImplementedError()

    @staticmethod
    def new(settings : SettingsManager):
        return NotImplementedError()

######## Subclasses ########
class AGNDiskInterp(AGNDisk):
    """Generic AGN Disk Interpolation"""
    def __init__(
        self,
        surface_density_data,
        aspect_ratio_data,
        opacity_data,
        sound_speed_data,
        density_data,
        omega_data,
        pressure_data,
        temperature_data
    ):
        """Initialization for a direct interpolation disk
        """
        ## Generate the CubicSplines
        # Create surface density interpolator object
        self._surface_density_loglog = CubicSpline(
            np.log(surface_density_data[0]),
            np.log(surface_density_data[1]),
        )
        # Create aspect ratio interpolator object
        self._aspect_ratio_loglog = CubicSpline(
            np.log(aspect_ratio_data[0]),
            np.log(aspect_ratio_data[1]),
        )
        # Create opacity interpolator object
        self._opacity_loglog = CubicSpline(
            np.log(opacity_data[0]),
            np.log(opacity_data[1])
        )
        # Create sound speed interpolator object
        self._sound_speed_loglog = CubicSpline(
            np.log(sound_speed_data[0]),
            np.log(sound_speed_data[1]),
        )
        # Create density interpolator object
        self._density_loglog = CubicSpline(
            np.log(density_data[0]),
            np.log(density_data[1]),
        )
        # Create omega interpolator object
        self._omega_loglog = CubicSpline(
            np.log(omega_data[0]),
            np.log(omega_data[1]),
        )
        # Create pressure gradient interpolator object
        self._pressure_grad_linear = CubicSpline(
            pressure_data[0],
            pressure_data[1],
        )
        # Create temperature interpolator object
        self._temperature_loglog = CubicSpline(
            np.log(temperature_data[0]),
            np.log(temperature_data[1]),
        )
        # Create surface density log derivative interpolator object
        self._dlog10_surface_density_log10R = dCubicSpline(
            np.log10(surface_density_data[0]),
            np.log10(surface_density_data[1]),
        )
        # Create temperature log derivative interpolator object
        self._dlog10_temp_dlog10R = dCubicSpline(
            np.log10(temperature_data[0]),
            np.log10(temperature_data[1]),
        )
        # Identify midplane pressure
        midplane_pressure = (sound_speed_data[1] ** 2) / density_data[1]
        # Create pressure log derivative interpolator object
        self._dlog10_midplane_pressure_dlog10R = dCubicSpline(
            np.log10(density_data[0]),
            np.log10(midplane_pressure)
        )

        ## Boundary Shenanigans ##

    ### Methods ###
    def surface_density(self, orb_a):
        return np.exp(self._surface_density_loglog(np.log(orb_a)))
    def disk_surface_density(self, orb_a):
        return self.surface_density(orb_a)
    def disk_surface_density_log(self, orb_a):
        return self._surface_density_loglog(orb_a)

    def aspect_ratio(self, orb_a):
        return np.exp(self._aspect_ratio_loglog(np.log(orb_a)))
    def disk_aspect_ratio(self, orb_a):
        return self.aspect_ratio(orb_a)

    def opacity(self, orb_a):
        return np.exp(self._opacity_loglog(np.log(orb_a)))
    def disk_opacity(self, orb_a):
        return self.opacity(orb_a)

    def sound_speed(self, orb_a):
        return np.exp(self._sound_speed_loglog(np.log(orb_a)))
    def disk_sound_speed(self, orb_a):
        return self.sound_speed(orb_a)

    def density(self, orb_a):
        return np.exp(self._density_loglog(np.log(orb_a)))
    def disk_density(self, orb_a):
        return self.density(orb_a)

    def omega(self, orb_a):
        return np.exp(self._omega_loglog(np.log(orb_a)))
    def disk_omega(self, orb_a):
        return self.omega(orb_a)

    def pressure_gradient(self, orb_a):
        return self._pressure_grad_linear(orb_a)
    def disk_pressure_grad(self, orb_a):
        return self.pressure_gradient(orb_a)

    def temperature(self, orb_a):
        return np.exp(self._temperature_loglog(np.log(orb_a)))
    def temp_func(self, orb_a):
        return self.temperature(orb_a)

    def dlog10_surface_density_dlog10R(self, orb_a):
        return self._dlog10_surface_density_log10R(orb_a)
    def disk_dlog10surfdens_dlog10R_func(self, orb_a):
        return self.dlog10_surface_density_dlog10R(orb_a)

    def dlog10_temperature_dlog10R(self, orb_a):
        return self._dlog10_temp_dlog10R(orb_a)
    def disk_dlog10temp_dlog10R_func(self, orb_a):
        return self.dlog10_temperature_dlog10R(orb_a)

    def dlog10_midplane_pressure_dlog10R(self, orb_a):
        return self._dlog10_midplane_pressure_dlog10R(orb_a)
    def disk_dlog10pressure_dlog10R_func(self, orb_a):
        return self.dlog10_midplane_pressure_dlog10R(orb_a)

    @classmethod
    def from_importlib(cls, disk_model_name, disk_radius_outer):
        """Load a disk from data stored in the mcfacts repository"""
        from mcfacts.inputs.ReadInputs import load_disk_arrays
        # Load the disk arrays
        trunc_surf_density_data, trunc_aspect_ratio_data, \
                trunc_opacity_data, trunc_sound_speed_data, \
                trunc_density_data, trunc_omega_data, \
                trunc_pressure_data, trunc_temperature_data = \
            load_disk_arrays(disk_model_name, disk_radius_outer)
        # Construct disk
        disk = AGNDiskInterp(
            trunc_surf_density_data,
            trunc_aspect_ratio_data,
            trunc_opacity_data,
            trunc_sound_speed_data,
            trunc_density_data,
            trunc_omega_data,
            trunc_pressure_data,
            trunc_temperature_data,
        )
        # Return disk
        return disk
        

