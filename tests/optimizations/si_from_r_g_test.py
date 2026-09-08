import numpy as np

import astropy.units as u
from mcfacts.physics.point_masses import si_from_r_g, si_from_r_g_optimized
import itertools

# at present, the original si_from_r_g function cannot take in length or mass units as variants for the distance_r_g argument, it needs to be a scalar
units = [1, u.g, u.kg, u.solMass, u.jupiterMass, u.earthMass]

# and the array case
vals = [1.0, 5435.46345, 48.5, 137.0, 0.00001, np.array([1.0, 5435.46345, 48.5, 137.0, 0.00001])]

smbh_mass = 1.0

def test_si_from_rg():
    for val in vals:
        for unit in units:
            # notable difference: r_g_from units returns unitless values, while optimized 
            # returns values in meters. Since the vast majority of uses of this function return

            orig = si_from_r_g(smbh_mass * unit, val)
            opt = si_from_r_g_optimized(smbh_mass * unit, val)

            assert(np.allclose(orig, opt, rtol=1e-6))


