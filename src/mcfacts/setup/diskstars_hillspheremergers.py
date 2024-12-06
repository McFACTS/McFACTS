import numpy as np


def mass_threshold_merger(disk_star_num, disk_stars_mass_min, smbh_mass, P_m, P_r, disk_stars_orb_a, disk_radius_outer):
    """
    disk_star_num : int
        number of stars in the initial draw
    disk_stars_mass_min : float
        minimum mass [M_sun] considered
    smbh_mass : float
        mass of SMBH, M_sun
    P_m : float
        exponent for mass cdf, assuming it is in the form P(> m_min) = (m_min/m)^P_m
    P_r : float
        exponent for disk orb_aation cdf, assuming the form P(r) = (r_orb_aation/disk_radius_outer)^P_r
    r_orb_aation : numpy array
        semi-major axis of stellar orbit around SMBH, R_sun (for now?)
    disk_radius_outer : float
        Outer radius [r_{g,SMBH}] of disk

    Returns:
    mass_threshold : numpy.ndarray
        minimum mass [M_sun] for stars to not merge

    Notes
    -----
    Eqn is
    (M_min^(P_m/(P_m - 1/3))) / ((3*M_smbh)^(1/3(P_m - 1/3))) * (N_star * P_r)^(1/(P_m - 1/3)) * (orb_a/R)^(P_r/(P_m - 1/3))
    """
    exp1_top = P_m/(P_m - (1./3.))
    exp1_bottom = 1./(3*(P_m - (1./3.)))
    frac1 = ((disk_stars_mass_min**exp1_top))/((3*smbh_mass)**exp1_bottom)

    exp2 = 1./(P_m - (1./3.))
    frac2 = (disk_star_num*P_r)**exp2

    exp3 = P_r/(P_m - (1./3.))
    frac3 = (disk_stars_orb_a/disk_radius_outer)**exp3

    mass_threshold = frac1*frac2*frac3

    return (mass_threshold)


def hill_sphere_orb_a(orbs_a_sorted, mass_threshold, smbh_mass, disk_radius_outer):
    """Find location of edge of Hill sphere for stars

    Parameters
    ----------
    orbs_a_sorted : numpy.ndarray
        Locations [r_{g,SMBH}] of stars in the disk with :obj:`float` type
    mass_threshold : numpy.ndarray
        Minimum mass [M_sun] to not merge, as a function of orbs_a_sorted with :obj:`float` type
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_radius_outer : float
        Outer radius [r_{g,SMBH}] of the disk

    Returns
    -------
    delta_orbs_a : numpy.ndarray
        Locations [r_{g,SMBH}] of the Hill spheres for stars in the disk
    """

    # Set up array for locations of Hill spheres in the disk
    delta_orbs_a = []
    # We start at 10 r_{g,SMBH} as minimum orb_a
    #delta_orbs_a.append(10)
    # orb_a is radius of Hill sphere
    #orb_a = delta_orbs_a[0]*((mass_threshold[0]/(3.*smbh_mass))**(1./3.)) + delta_orbs_a[0]
    # Hill sphere radius for the star closest to the SMBH
    orb_a = orbs_a_sorted[0]*((mass_threshold[0]/(3.*smbh_mass))**(1./3.)) + orbs_a_sorted[0]
    delta_orbs_a.append(orb_a)
    while orb_a < disk_radius_outer:
        # Find the next star closest to the edge of the Hill sphere radius
        idx = (np.abs(orb_a - orbs_a_sorted)).argmin()
        # Calculate its Hill sphere radius and add it to orb_a counter
        orb_a += orbs_a_sorted[idx]*((mass_threshold[idx]/(3.*smbh_mass))**(1./3.))
        delta_orbs_a.append(orb_a)
    delta_orbs_a = np.array(delta_orbs_a)
    return (delta_orbs_a)


def hillsphere_mergers(n_stars, masses_initial_sorted, orbs_a_initial_sorted, min_initial_star_mass, disk_radius_outer, smbh_mass, P_m, P_r):
    # P_m and P_r need to be added to opts

    # Get the minimum mass for stars to not merge for every orb_a
    mass_threshold = mass_threshold_merger(disk_star_num=n_stars,
                                           disk_stars_mass_min=min_initial_star_mass,
                                           smbh_mass=smbh_mass,
                                           P_m=P_m,
                                           P_r=P_r,
                                           disk_stars_orb_a=orbs_a_initial_sorted,
                                           disk_radius_outer=disk_radius_outer)

    # Get locations of Hill spheres for each threshold mass
    delta_orbs_a = hill_sphere_orb_a(orbs_a_initial_sorted, mass_threshold, smbh_mass, disk_radius_outer)

    # Set up arrays for final masses and orb_a
    new_masses = []
    new_orbs_a = []
    for idx in range(len(delta_orbs_a)-1):
        if (len(orbs_a_initial_sorted[(orbs_a_initial_sorted > delta_orbs_a[idx]) & (orbs_a_initial_sorted < delta_orbs_a[idx+1])]) > 0):
            mass_range = masses_initial_sorted[(orbs_a_initial_sorted > delta_orbs_a[idx]) & (orbs_a_initial_sorted < delta_orbs_a[idx+1])]
            r_range = orbs_a_initial_sorted[(orbs_a_initial_sorted > delta_orbs_a[idx]) & (orbs_a_initial_sorted < delta_orbs_a[idx+1])]
            mt_temp = mass_threshold_merger(disk_star_num=n_stars,
                                            disk_stars_mass_min=min_initial_star_mass,
                                            smbh_mass=1.e8,
                                            P_m=1.35,
                                            P_r=1.,
                                            disk_stars_orb_a=r_range,
                                            disk_radius_outer=disk_radius_outer)
            mass_range_merge = mass_range[mass_range <= mt_temp]

            mass_range_static = mass_range[mass_range > mt_temp]
            r_range_static = r_range[mass_range > mt_temp]

            if (len(mass_range_static) > 0):
                for m, r in zip(mass_range_static, r_range_static):
                    new_masses.append(m)
                    new_orbs_a.append(r)

            if (len(mass_range_merge) > 0):
                new_star_mass = np.sum(mass_range)
                new_star_r = np.average(r_range, weights=mass_range)
                new_masses.append(new_star_mass)
                new_orbs_a.append(new_star_r)

    new_masses = np.array(new_masses)
    new_orbs_a = np.array(new_orbs_a)

    return (new_masses, new_orbs_a)
