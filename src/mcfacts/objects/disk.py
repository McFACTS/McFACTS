"""Handling of the McFACTS AGNDisk object"""
######## Imports ########
#### Standard Library ####
import contextlib
import os
import sys
#### Third Party ####
import numpy as np
from astropy import units as u
from astropy import constants as ct
import pagn.constants as pagn_ct
#### Local ####
from mcfacts.objects.cache import readonly_cached_property
from mcfacts.objects.interp import CubicSpline, dCubicSpline
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.utilities.unit_conversion import si_from_r_g_optimized
from mcfacts.utilities.unit_conversion import r_g_from_units
import mcfacts.external.DiskModelsPAGN as pagn_dm

######## Without settings, it's easier to test ########
class AGNDiskInterp(object):
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
        temperature_data,
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
        self._dlog10_surface_density_dlog10R = dCubicSpline(
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

        # A pAGN model can be stored here
        self._pagn_model = None
        self._pagn_bonus_structures = None

        ## Boundary Shenanigans ##

    ### Properties ###
    @property
    def pagn_model(self):
        if self._pagn_model is not None:
            return self._pagn_model
        else:
            raise RuntimeError( 
                f"{self.__class__} not initialized with pAGN model")

    @pagn_model.setter
    def pagn_model(self, value):
        self._pagn_model = value

    @property
    def pagn_bonus_structures(self):
        if self._pagn_bonus_structures is not None:
            return self._pagn_bonus_structures
        elif self._pagn_model is not None:
            raise RuntimeError(
                f"Found pagn_model, but not pagn_bonus_structures"
            )
        else:
            raise RuntimeError( 
                f"{self.__class__} not initialized with pAGN model")

    @pagn_bonus_structures.setter
    def pagn_bonus_structures(self, value):
        self._pagn_bonus_structures = value

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
        return self._dlog10_surface_density_dlog10R(orb_a)
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
    
    @staticmethod
    def importlib_disk_arrays(disk_model_name, disk_radius_outer):
        from mcfacts.inputs.ReadInputs import load_disk_arrays
        return load_disk_arrays(disk_model_name, disk_radius_outer)

    @classmethod
    def from_importlib(cls, disk_model_name, disk_radius_outer):
        """Load a disk from data stored in the mcfacts repository"""
        # Construct disk
        return cls(*cls.importlib_disk_arrays(disk_model_name,disk_radius_outer))
        
    @staticmethod
    def run_pagn_model(
        disk_model_name,
        smbh_mass,
        disk_radius_outer,
        disk_alpha_viscosity,
        disk_bh_eddington_ratio,
        rad_efficiency=0.1,
    ):
        # instead, populate with pagn
        if "sirko" in disk_model_name:
            pagn_name = "Sirko"
            base_args = {
                'Mbh': smbh_mass*pagn_ct.MSun,
                'alpha': disk_alpha_viscosity, 
                'le': disk_bh_eddington_ratio,
                'eps': rad_efficiency
            }
        elif 'thompson' in disk_model_name:
            pagn_name = 'Thompson'
            base_args = {
                'Mbh': smbh_mass*pagn_ct.MSun,
                'm': disk_alpha_viscosity, 
            }
                #'epsilon': rad_efficiency
                #'le': disk_bh_eddington_ratio,\
            Rg = smbh_mass * ct.M_sun * ct.G / (ct.c**2)
            # pAGN TQM disk models exclude `Rout`, so feed pAGN a slightly
            # larger value (+1%) than the user set for `disk_radius_outer`
            base_args['Rout'] = 1.01 * disk_radius_outer * Rg.to('m').value
        else:
            raise RuntimeError("unknown disk model: %s"%(disk_model_name))

        # note Rin default is 3 Rs

        # Run pAGN; I am tired of seeing this
        # Use ReadInputs.construct_disk_pAGN if you want to see output
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                result = pagn_dm.AGNGasDiskModel(disk_type=pagn_name, **base_args)
        return result

    @classmethod
    def from_pagn(cls,*args,**kwargs):
        """Construct a pAGN disk model and interpolate the outputs"""
        pagn_model = cls.run_pagn_model(*args,**kwargs)
        interp_data, bonus_structures = pagn_model.return_disk_surf_data()
        disk = cls(*interp_data)
        disk.pagn_bonus_structures = bonus_structures
        disk.pagn_model = pagn_model
        return disk

######## Disk object ########
class AGNDisk(AGNDiskInterp):
    """Having tested all fo the stuff we can do with it without settings,
        let's now give it settings.
    """
    def __init__(self, settings: SettingsManager):
        # carry a reference to settings
        self.settings = settings
        # Cache the smbh mass at the time of creation
        _ = self.smbh_mass
        if settings.flag_use_pagn:
            pagn_model = super().run_pagn_model(
                settings.disk_model_name,
                self.smbh_mass,
                self.disk_radius_outer,
                settings.disk_alpha_viscosity,
                settings.disk_bh_eddington_ratio,
                rad_efficiency=0.1, # TODO This should really be a setting
            )
            interp_data, bonus = pagn_model.return_disk_surf_data()
            super().__init__(*interp_data)
            self.pagn_model = pagn_model
            self.pagn_bonus_structures = bonus
        else:
            super().__init__(
                *self.importlib_disk_arrays(
                    settings.disk_model_name,
                    self.disk_radius_outer,
                )
            )

    @readonly_cached_property
    def smbh_mass(self):
        return self.settings.smbh_mass

    @staticmethod
    def pc_dist(smbh_mass, num_r_g):
        return si_from_r_g_optimized(smbh_mass, num_r_g).to(u.pc).value.item()

    @readonly_cached_property
    def disk_radius_outer(self):
        disk_radius_outer_pc = self.pc_dist(
            self.smbh_mass, self.settings.disk_radius_outer)

        # Check disk_radius_max_pc argument
        if self.settings.disk_radius_max_pc == 0.:
            # Case 1: disk_radius_max_pc is disabled
            return self.settings.disk_radius_outer
        elif self.settings.disk_radius_max_pc < 0.:
            # Case 2: disk_radius_max_pc is negative
            # Always assign disk_radius_outer to given distance in parsecs
            return r_g_from_units(
                self.smbh_mass,
                -1. * self.settings.disk_radius_max_pc * u.pc,
            )
        else:
            # Case 3: disk_radius_max_pc is positive
            # Cap disk_radius_outer at given value
            if disk_radius_outer_pc > self.settings.disk_radius_max_pc:
                # calculate scale factor
                disk_radius_scale = self.settings.disk_radius_max_pc / disk_radius_outer_pc
                # Adjust disk_radius_outer as needed
                return self.settings.disk_radius_outer * disk_radius_scale
            else:
                return self.settings.disk_radius_outer
