import numpy as np
import astropy.units as u
from mcfacts.physics.point_masses import r_schwarzschild_of_m, r_g_from_units_optimized, r_schwarzschild_of_m_optimized

units = [u.g, u.kg, u.solMass, u.jupiterMass, u.earthMass]

# and the array case
vals = [np.array([1.0, 5435.46345, 48.5, 137.0, 0.00001])]

def test_r_schwarzschild():
    for val, unit in zip(vals, units):
        # notable difference: r_g_from units returns unitless values, while optimized 
        # returns values in meters. Since the vast majority of uses of this function return

        orig = r_schwarzschild_of_m(val * unit)
        opt = r_schwarzschild_of_m_optimized(val * unit)

        assert(np.allclose(orig, opt, rtol=1e-9))


