"""
Defines a collection of global variables used throughout McFACTS.


Global Variables:
- M_SUN_KG (float): number of Kg per Msun
- SEC_IN_YR (float): number of Seconds per Year
- G_SI (float): gravitational constant [m^3 kg^-1 s^-2]
- C_SI (float): speed of light [m/s]
"""

#### IMPORTS
from astropy import constants as const
from astropy import units as u

#### BEGIN GLOBAL VARIABLES
SEC_IN_YR = u.yr.to(u.s)
M_SUN_KG = u.M_sun.to(u.kg)
G_SI = const.G.si.value
C_SI = const.c.si.value
