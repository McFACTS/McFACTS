"""Modify a SettingsManager to participate in scaled runs"""

######## Imports ########
#### Standard Library ####
#### Third Party ####
import numpy as np
#### Local ####
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.disk import AGNDisk

######## Functions ########
def disk_truncation(settings: SettingsManager): 
    """Truncate the disk based on properties of the SettingsManager"""
    # Check that the user asked for this
    if settings.disk_truncation == "none":
        return
    # Setup an AGNDisk
    TempDisk = AGNDisk(settings)

    if settings.disk_truncation == "opacity-equals-inner-disk":
        # Make sure pAGN is enabled
        if not settings.flag_use_pagn:
            raise NotImplementedError
        # Load R and tauV
        pagn_R = TempDisk.pagn_bonus_structures["R"]
        pagn_tauV = TempDisk.pagn_bonus_structures["tauV"]
        # Find where tauV is greater than its initial value
        tau_drop_mask = (pagn_tauV < pagn_tauV[0]) & (np.log10(pagn_R) > 3)
        # Find the drop index
        tau_drop_index = np.argmax(tau_drop_mask)
        # Find the drop radius
        tau_drop_radius = pagn_R[tau_drop_index]
        # Update settings
        print(settings.disk_radius_outer)
        settings.disk_radius_outer = tau_drop_radius
        print(settings.disk_radius_outer)

def setup_scaling(settings: SettingsManager):
    """Scale the AGN Disk according to settings"""
    # truncate disk
    disk_truncation(settings)
    return None
