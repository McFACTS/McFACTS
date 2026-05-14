"""
Defines a collection of global variables used throughout McFACTS.


Global Variables:
- mass_per_msun (float): number of Kg per Msun

"""

from astropy import constants as const
from astropy import units as u

# mass_per_msun = 1.99e30

SEC_IN_YR = u.yr.to(u.s)
M_SUN_KG = u.M_sun.to(u.kg)