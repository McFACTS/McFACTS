"""
Defines a collection of global variables used throughout McFACTS.


Global Variables:
- M_SUN_KG (float): number of Kg per Msun
- SEC_IN_YR (float): number of Seconds per Year
"""

#### IMPORTS
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

#### BEGIN GLOBAL VARIABLES
SEC_IN_YR = u.yr.to(u.s)
M_SUN_KG = u.M_sun.to(u.kg)
HUBBLE_TIME_IN_S = cosmo.age(0.).to(u.s)
