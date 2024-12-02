from typing import Callable

from mcfacts.inputs import ReadInputs
from mcfacts.inputs.settings_manager import SettingsManager

class AGNDisk:
    def __init__(self, settings: SettingsManager):
        # TODO: More advanced handling?

        disk_surface_density, disk_aspect_ratio, disk_opacity = \
            ReadInputs.construct_disk_interp(settings.smbh_mass,
                                             settings.disk_radius_outer,
                                             settings.disk_model_name,
                                             settings.disk_alpha_viscosity,
                                             settings.disk_bh_eddington_ratio,
                                             disk_radius_max_pc=settings.disk_radius_max_pc,
                                             flag_use_pagn=settings.flag_use_pagn,
                                             verbose=settings.verbose
                                             )

        self.disk_surface_density = disk_surface_density
        self.disk_aspect_ratio = disk_aspect_ratio
        self.disk_opacity = disk_opacity