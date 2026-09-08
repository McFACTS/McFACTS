import numpy as np
import astropy.units as u
from mcfacts.physics.point_masses import r_g_from_units, r_g_from_units_optimized


units = [u.m, u.Rsun]

# and the array case
vals = [1.0, 5435.46345, 48.5, 137.0, 0.00001, -1.0, -32432.6, np.array([0.0, 1.0, 5435.46345, 48.5, 137.0, 0.00001, -1.0, -32432.6])]

smbh_mass = 1.0

def test_rg_units():
    for val, unit in zip(vals, units):
        # notable difference: r_g_from units returns unitless values, while optimized 
        # returns values in meters. Since the vast majority of uses of this function return

        orig = r_g_from_units(smbh_mass, val * unit) * u.m
        opt = r_g_from_units_optimized(smbh_mass, val * unit)

        assert(np.allclose(orig, opt, rtol=1e-6))


