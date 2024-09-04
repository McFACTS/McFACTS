import numpy as np


def setup_disk_blackholes_location(rng, n_bh, disk_outer_radius):
    #Return an array of BH locations distributed randomly uniformly in disk
    isco_radius = 6.0
    integer_nbh = int(n_bh)
    bh_initial_locations = disk_outer_radius*rng.random(integer_nbh)
    sma_too_small = np.where(bh_initial_locations<isco_radius)
    bh_initial_locations[sma_too_small] = isco_radius
    return bh_initial_locations

def setup_prior_blackholes_indices(rng, prograde_n_bh, prior_bh_locations):
    #Return an array of indices which allow us to read prior BH properties & replace prograde BH with these.
    integer_nbh = int(prograde_n_bh)
    len_prior_locations = (prior_bh_locations.size)-1
    bh_indices = np.rint(len_prior_locations*rng.random(integer_nbh))
    return bh_indices

def setup_disk_blackholes_masses(rng, n_bh,mode_mbh_init,max_initial_bh_mass,mbh_powerlaw_index):
    #Return an array of BH initial masses for a given powerlaw index and max mass
    integer_nbh = int(n_bh)
    bh_initial_masses = (rng.pareto(mbh_powerlaw_index,integer_nbh)+1)*mode_mbh_init
    #impose maximum mass condition (default upper bound is 40Msun)
    #Set a critical mass limit (say 35Msun) (in units of Msun)
    critical_bh_mass = 35.0
    mass_diff = max_initial_bh_mass - critical_bh_mass
    bh_initial_masses[bh_initial_masses > max_initial_bh_mass] = critical_bh_mass + np.rint(mass_diff*rng.random())
    return bh_initial_masses


def setup_disk_blackholes_spins(rng, n_bh, mu_spin_distribution, sigma_spin_distribution):
    #Return an array of BH initial spin magnitudes for a given mode and sigma of a distribution
    integer_nbh = int(n_bh)
    bh_initial_spins = rng.normal(mu_spin_distribution, sigma_spin_distribution, integer_nbh)
    return bh_initial_spins


def setup_disk_blackholes_spin_angles(rng, n_bh, bh_initial_spins):
    #Return an array of BH initial spin angles (in radians).
    #Positive (negative) spin magnitudes have spin angles [0,1.57]([1.5701,3.14])rads
    #All BH spin angles drawn from [0,1.57]rads and +1.57rads to negative spin indices
    integer_nbh = int(n_bh)
    bh_initial_spin_indices = np.array(bh_initial_spins)
    negative_spin_indices = np.where(bh_initial_spin_indices < 0.)
    bh_initial_spin_angles = rng.uniform(0.,1.57,integer_nbh)
    bh_initial_spin_angles[negative_spin_indices] = bh_initial_spin_angles[negative_spin_indices] + 1.57
    return bh_initial_spin_angles


def setup_disk_blackholes_orb_ang_mom(rng, n_bh):
    #Return an array of BH initial orbital angular momentum.
    #Assume either fully prograde (+1) or retrograde (-1)
    integer_nbh = int(n_bh)
    random_uniform_number = rng.random((integer_nbh,))
    bh_initial_orb_ang_mom = (2.0*np.around(random_uniform_number)) - 1.0
    return bh_initial_orb_ang_mom

def setup_disk_blackholes_eccentricity_thermal(rng, n_bh):
    # Return an array of BH orbital eccentricities
    # For a thermal initial distribution of eccentricities, select from a uniform distribution in e^2.
    # Thus (e=0.7)^2 is 0.49 (half the eccentricities are <0.7). 
    # And (e=0.9)^2=0.81 (about 1/5th eccentricities are >0.9)
    # So rnd= draw from a uniform [0,1] distribution, allows ecc=sqrt(rnd) for thermal distribution.
    # Thermal distribution in limit of equipartition of energy after multiple dynamical encounters
    integer_nbh = int(n_bh)
    random_uniform_number = rng.random((integer_nbh,))
    bh_initial_orb_ecc = np.sqrt(random_uniform_number)
    return bh_initial_orb_ecc

def setup_disk_blackholes_eccentricity_uniform(rng, n_bh, max_initial_eccentricity):
    # Return an array of BH orbital eccentricities
    # For a uniform initial distribution of eccentricities, select from a uniform distribution in e.
    # Thus half the eccentricities are <0.5
    # And about 1/10th eccentricities are >0.9
    # So rnd = draw from a uniform [0,1] distribution, allows ecc = rnd for uniform distribution
    # Most real clusters/binaries lie between thermal & uniform (e.g. Geller et al. 2019, ApJ, 872, 165)
    # Cap of max_initial_eccentricity allows for previous recent episode of AGN where some relaxation has happened.
    integer_nbh = int(n_bh)
    random_uniform_number = rng.random((integer_nbh,))
    bh_initial_orb_ecc = random_uniform_number*max_initial_eccentricity
    return bh_initial_orb_ecc

def setup_disk_blackholes_eccentricity_uniform_modified(rng, mod_factor, n_bh):
    # Return an array of BH orbital eccentricities
    # For a uniform initial distribution of eccentricities, select from a uniform distribution in e.
    # Thus half the eccentricities are <0.5
    # And about 1/10th eccentricities are >0.9
    # So rnd = draw from a uniform [0,1] distribution, allows ecc = rnd for uniform distribution
    # Most real clusters/binaries lie between thermal & uniform (e.g. Geller et al. 2019, ApJ, 872, 165)
    integer_nbh = int(n_bh)
    random_uniform_number = rng.random((integer_nbh,))
    bh_initial_orb_ecc = mod_factor*random_uniform_number
    return bh_initial_orb_ecc

def setup_disk_blackholes_inclination(rng, n_bh):
    # Return an array of BH orbital inclinations
    # Return an initial distribution of inclination angles that are 0.0
    #
    # To do: initialize inclinations so random draw with i <h (so will need to input bh_locations and disk_aspect_ratio)
    # and then damp inclination.
    # To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    # Then damp (i,e) as appropriate
    integer_nbh = int(n_bh)
    # For now, inclinations are zeros
    bh_initial_orb_incl = np.zeros((integer_nbh,),dtype = float)
    return bh_initial_orb_incl

def setup_disk_blackholes_incl(rng, n_bh, bh_location, ang_mom_idx, aspect_ratio_func):
    # Return an array of BH orbital inclinations
    # initial distribution is not 0.0
    integer_nbh = int(n_bh)
    # what is the max height at the orbiter location that keeps it in the disk
    max_height = bh_location * aspect_ratio_func(bh_location)
    # reflect that height to get the min
    min_height = -max_height
    random_uniform_number = rng.random((integer_nbh,))
    # pick the actual height between the min and max, then reset zero point
    height_range = max_height - min_height
    actual_height_range = height_range * random_uniform_number
    actual_height = actual_height_range + min_height
    # inclination is arctan of height over radius, modulo pro or retrograde
    bh_initial_orb_incl = np.arctan(actual_height/bh_location)
    # for retrogrades, add 180 degrees
    bh_initial_orb_incl[ang_mom_idx < 0.0] = bh_initial_orb_incl[ang_mom_idx < 0.0] + np.pi

    return bh_initial_orb_incl

def setup_disk_blackholes_circularized(rng, n_bh,crit_ecc):
    # Return an array of BH orbital inclinations
    # Return an initial distribution of inclination angles that are 0.0
    #
    # To do: initialize inclinations so random draw with i <h (so will need to input bh_locations and disk_aspect_ratio)
    # and then damp inclination.
    # To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    # Then damp (i,e) as appropriate
    integer_nbh = int(n_bh)
    # For now, inclinations are zeros
    #bh_initial_orb_ecc = crit_ecc*np.ones((integer_nbh,),dtype = float)
    #Try zero eccentricities
    bh_initial_orb_ecc = crit_ecc*np.zeros((integer_nbh,),dtype = float)
    return bh_initial_orb_ecc

def setup_disk_blackholes_arg_periapse(rng, n_bh):
    # Return an array of BH arguments of periapse
    # Should be fine to do a uniform distribution between 0-2pi
    # Someday we should probably pick all the orbital variables
    #   to be consistent with a thermal distribution or something
    #   otherwise physically well-motivated...
    #
    # Hahahaha, actually, no, for now everything will be at 0.0 or pi/2
    # For now, bc crude retro evol only treats arg periapse = (0.0, pi) or pi/2
    # and intermediate values are fucking up evolution
    # (and pi is symmetric w/0, pi/2 symmetric w/ 3pi/2 so doesn't matter)
    # when code can handle arbitrary arg of periapse, uncomment relevant line

    integer_nbh = int(n_bh)
    random_uniform_number = rng.random((integer_nbh,))
    #bh_initial_orb_arg_periapse = 2.0*np.pi*random_uniform_number
    bh_initial_orb_arg_periapse = 0.5 * np.pi * np.around(random_uniform_number)

    return bh_initial_orb_arg_periapse

def setup_disk_nbh(M_nsc,nbh_nstar_ratio,mbh_mstar_ratio,r_nsc_out,nsc_index_outer,mass_smbh,disk_outer_radius,h_disk_average,r_nsc_crit,nsc_index_inner):
    # Return the integer number of BH in the AGN disk as calculated from NSC inputs assuming isotropic distribution of NSC orbits
    # To do: Calculate when R_disk_outer is not equal to the r_nsc_crit
    # To do: Calculate when disky NSC population of BH in plane/out of plane.
    # Housekeeping:
    # Convert outer disk radius in r_g to units of pc. 1r_g =1AU (M_smbh/10^8Msun) and 1pc =2e5AU =2e5 r_g(M/10^8Msun)^-1
    pc_dist = 2.e5*((mass_smbh/1.e8)**(-1.0))
    critical_disk_radius_pc = disk_outer_radius/pc_dist
    #Total average mass of BH in NSC
    M_bh_nsc = M_nsc * nbh_nstar_ratio * mbh_mstar_ratio
    #print("M_bh_nsc",M_bh_nsc)
    #Total number of BH in NSC
    N_bh_nsc = M_bh_nsc / mbh_mstar_ratio
    #print("N_bh_nsc",N_bh_nsc)
    #Relative volumes:
    #   of central 1 pc^3 to size of NSC
    relative_volumes_at1pc = (1.0/r_nsc_out)**(3.0)
    #   of r_nsc_crit^3 to size of NSC
    relative_volumes_at_r_nsc_crit = (r_nsc_crit/r_nsc_out)**(3.0)
    #print(relative_volumes_at1pc)
    #Total number of BH 
    #   at R<1pc (should be about 10^4 for Milky Way parameters; 3x10^7Msun, 5pc, r^-5/2 in outskirts)
    N_bh_nsc_pc = N_bh_nsc * relative_volumes_at1pc * (1.0/r_nsc_out)**(-nsc_index_outer)
    #   at r_nsc_crit
    N_bh_nsc_crit = N_bh_nsc * relative_volumes_at_r_nsc_crit * (r_nsc_crit/r_nsc_out)**(-nsc_index_outer)
    #print("Normalized N_bh at 1pc",N_bh_nsc_pc)
    
    #Calculate Total number of BH in volume R < disk_outer_radius, assuming disk_outer_radius<=1pc.
    
    if critical_disk_radius_pc >= r_nsc_crit:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/1.0)**(3.0)
        Nbh_disk_volume = N_bh_nsc_pc * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/1.0)**(-nsc_index_outer))          
    else:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/r_nsc_crit)**(3.0)
        Nbh_disk_volume = N_bh_nsc_crit * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/r_nsc_crit)**(-nsc_index_inner))
     
    # Total number of BH in disk
    Nbh_disk_total = np.rint(Nbh_disk_volume * h_disk_average)
    #print("Nbh_disk_total",Nbh_disk_total)  
    return np.int64(Nbh_disk_total)

