import numpy as np
import scipy

import mcfacts.physics.disk_capture
from mcfacts.physics.disk_capture import crude_retro_evol
#from mcfacts.physics.migration.type1 import retro_mig
#from mcfacts.physics.eccentricity import retro_ecc
#from mcfacts.physics.disk_capture import capture_inc_damp

if __name__ == "__main__":
    # Just want to test out my new crappy retro evolution module full of hardcoded awfulness

    # But I need to import a surface density profile and set it up as a function
    infile = "../inputs/data/sirko_goodman_surface_density.txt" # this is violence, sorry
    surface_density_file = open(infile, 'r')
    density_list = []
    radius_list = []
    for line in surface_density_file:
        line = line.strip()
        # If it is NOT a comment line
        if (line.startswith('#') == 0):
            columns = line.split()
            #If radius is less than disk outer radius
            #if columns[1] < disk_outer_radius:
            density_list.append(float(columns[0]))
            radius_list.append(float(columns[1]))
    # close file
    surface_density_file.close()

    # re-cast from lists to arrays
    surface_density_array = np.array(density_list)
    disk_model_radius_array = np.array(radius_list)

    # create function
    surf_dens_func_log = scipy.interpolate.UnivariateSpline(
        disk_model_radius_array, np.log(surface_density_array))
    surf_dens_func = lambda x, f=surf_dens_func_log: np.exp(f(x))

    mass_smbh = 1.0e8
    retrograde_bh_locations = 8.002169443665509789e+03 * np.ones(7)
    retrograde_bh_masses = 1.085329144860416584e+01 * np.ones(7)
    retrograde_bh_orb_ecc = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 2.274490742165892199e-01])
    retrograde_bh_orb_inc = 3.143573527032012649e+00 * np.ones(7)
    retro_arg_periapse = 0.0 * np.pi * np.ones(7)
    timestep = 1e4

    thing1, thing2, thing3 = mcfacts.physics.disk_capture.retro_bh_orb_disk_evolve(
        mass_smbh,
        retrograde_bh_masses,
        retrograde_bh_locations,
        retrograde_bh_orb_ecc,
        retrograde_bh_orb_inc,
        retro_arg_periapse,
        surf_dens_func,
        timestep)
    
    print("crude retro output")
    print(thing1)
    print(thing2)
    print(thing3)