"""Modify a SettingsManager to participate in scaled runs"""

######## Imports ########
#### Third Party ####
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

def setup_scaling(settings: SettingsManager):
    """Scale the AGN Disk according to settings"""
    return None
