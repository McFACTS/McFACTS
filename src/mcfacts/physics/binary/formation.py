"""
Module for handling the formation of binaries.
"""
import numpy as np

from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.gw import gw_strain_freq
from mcfacts.physics.point_masses import si_from_r_g

def close_encounters_check(id_nums,
                           filing_cabinet,
                           smbh_mass,
                           disk_bh_pro_orb_ecc_crit):
    """Calculates which prograde objects will have close encounters in this timestep.

    Takes as inputs the singleton objects locations,masses & orbital eccentricities,
    and takes the candidate encounter population from objects with orbital eccentricities
    damped to < orb_ecc_crit. Among this damped population, checks if their
    separations are less than the mutual Hill sphere of any 2 adjacent objects. If this
    is the case, determine the smallest separation pairs (in units of their mutual
    Hill sphere) to form a set of actual encounters (this module does handle cases where
    3 or more bodies *might* form some set of binaries which would be mutually exclusive;
    however it does not handle or even flag the implied triple system dynamics).
    Returns a 2xN array of the relevant indices, for further handling to form actual
    encounters & assign additional parameters (e.g. angular momentum of the binary).

    Parameters
    ----------
    id_nums : float array
        ID numbers of relevant objects (single, prograde, outer disk)
    filing_cabinet: AGNFilingCabinet
        filing cabinet holding parameters of objects in the disk
    smbh_mass : float
        Mass [M_sun] of the SMBH
    disk_bh_pro_orb_ecc_crit : float
        Critical eccentricity [unitless] allowing bin formation and migration

    Returns
    -------
    encounter_id_nums : [2,N] int array
        array of ID numbers corresponding to objects that will have a close encounter,
        it has a length of the number of encounters to form (N) and a width of 2.
    """

    # First check for objects with sufficiently damped orbital eccentricity
    # (orb_ecc<=orb_ecc_crit (usually 0.01)).
    # This population is the sub-set of prograde objects that CAN interact.

    # Objects need to have orb_ecc <= ecc_crit
    id_nums_can_encounter = id_nums[filing_cabinet.at_id_num(id_nums, "orb_ecc") <= disk_bh_pro_orb_ecc_crit]
    # If nothing can form a binary, end the function
    if (id_nums_can_encounter.size == 0):
        encounter_id_nums = np.array([])
        return (encounter_id_nums)
    fc_can_encounter = filing_cabinet.copy()
    # Remove objects that are too eccentric
    fc_can_encounter.keep_id_num(id_nums_can_encounter)
    # Sort all arrays by orb_a
    fc_can_encounter.sort(sort_attr="orb_a")

    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    # Note length of separations is 1 less than disk_bh_pro_orbs_a
    # This is the set of separations between the sorted candidate BH
    separations = np.diff(fc_can_encounter.orb_a)

    if (len(separations) > 0):
        R_Hill_possible_encounter = (fc_can_encounter.orb_a[:-1] + separations/2.0) * \
            pow(((fc_can_encounter.mass[:-1] + fc_can_encounter.mass[1:]) /
                 (smbh_mass * 3.0)), (1.0/3.0))

        # compare separations to mutual Hill spheres - negative values mean possible binary formation
        minimum_formation_criteria = separations - R_Hill_possible_encounter

        # collect indices of possible real binaries (where separation is less than mutual Hill sphere)
        idx_poss_encounter = np.asarray(minimum_formation_criteria < 0).nonzero()[0]

        # If just one binary
        if (idx_poss_encounter.size == 1):
            final_encounter_indices = np.array([[idx_poss_encounter[0]], [idx_poss_encounter[0] + 1]])
            encounter_id_nums = fc_can_encounter.id_num[final_encounter_indices]

        elif (idx_poss_encounter.size > 1):
            # Check to see if repeat binaries among the set of binaries formed (e.g., (1,2)(2,3))
            # If repeats, only form a binary from the pair with the smallest fractional Hill sphere separation
            # Compute separation/R_Hill for all
            sequences_to_test = separations[idx_poss_encounter] / R_Hill_possible_encounter[idx_poss_encounter]

            # Get sorted index of sep/R_Hill for all possible binaries that need checking
            idx_sort_sequences = np.argsort(sequences_to_test)

            # Assume the smallest sep/R_Hill should form a binary
            if (len(idx_sort_sequences) > 0):
                # Index of smallest sorted fractional Hill radius binary so far
                checked_encounter_index = np.array([idx_poss_encounter[idx_sort_sequences[0]]])
            else:
                checked_encounter_index = []

            for idx_seq in idx_sort_sequences:
                # If we haven't already counted it
                if (idx_poss_encounter[idx_seq] not in checked_encounter_index):
                    # And it isn't the implicit partner of something we've already counted
                    if (idx_poss_encounter[idx_seq] not in checked_encounter_index + 1):
                        # And the implicit partner of this thing isn't already counted
                        if (idx_poss_encounter[idx_seq] + 1 not in checked_encounter_index):
                            # And the implicit partner of this thing isn't already an implicit partner we've counted
                            if (idx_poss_encounter[idx_seq] + 1 not in checked_encounter_index + 1):
                                # Then you can count it as a real binary
                                checked_encounter_index = np.append(checked_encounter_index, idx_poss_encounter[idx_seq])
            final_encounter_indices = np.array([checked_encounter_index, checked_encounter_index + 1])
            encounter_id_nums = fc_can_encounter.id_num[final_encounter_indices]

        else:
            # No binaries from candidates this timestep
            encounter_id_nums = np.array([])
    else:
        # No candidates for binarity testing yet
        encounter_id_nums = np.array([])

    return (encounter_id_nums)

def mass_hill_sphere(id_nums, blackholes_pro, smbh_mass, disk_surf_density_func, disk_aspect_ratio_func, timestep):    
    """Calculate the mass in the Hill sphere of the putative binary formed in close encounters. 
    Compare to the mass of the binary (M_bbh = M_1 + M_2).
    Returns ratio of mass of gas in Hill sphere to mass of binary.
    Whitehead et al. (2025) show that 
    for M_hill =(0.01,0.025,0.063,0.2)M_bbh, then 
    prob of BBH formation =(0.05,0.2,0.5,1) respectively.
    Mass of Hill sphere = (4/3) pi R_Hill^3 * rho in units kg
    where R_Hill = a(q/3)^1/3 with a= center of mass of BBH in units of rg_SMBH, q=M_bbh/M_smbh
    and rho = density of disk model at a (center of mass) in units kg/m^3.

    NB: Since (rho) drops as a^-3 in SG outside ~10^3R_s(e.g. 10^-9g/cc at 10^3R_s drops to 10^-12g/cc at 10^4R_s, eg check Fig 2 in SG)
    this means Mass of Hill sphere =(4/3) pi a^3 (q/3) rho (a^-3) or
    Mass of Hill sphere/ Mass of BBH = Constant outside ~10^3r_g which is where most of our binaries form
    This constant is 0.44185... for default, ie >0.2 so binaries within a mutual Hill sphere *always* form by Whitehead+25 if >10^3r_g
    
    Condition A from Whitehead+25 (eqn 39) is (E_gas/E_hill) = -12.28 (M_hill/M_bbh)^0.8
    Can rearrange this to (M_hill/M_bbh) = ((-1/12.28)*(E_gas/E_hill))^1.25
    Assume a BBH is formed if E_gas<-2E_hill or (M_hill/M_bbh)> (2/12.28)^1.25=0.103
    Assume a BBH is formed at 50% odds if -E_hill < E_gas < -2E_hill or (1/6.14)^1.25<(M_hill/M_bbh)<(2/12.28)^1.25
    or 0.0435 <(M_hill/M_bbh)<0.103 
    and assume BBH is not formed if E_gas >- E_hill or (M_hill/M_bbh)<0.0435
    """
    
    number_of_poss_encs = id_nums.shape[1]
    number_of_bbh_forming = id_nums
    orb_a_1 = np.zeros(number_of_poss_encs)
    orb_a_2 = np.zeros(number_of_poss_encs)
    mass_1 = np.zeros(number_of_poss_encs)
    mass_2 = np.zeros(number_of_poss_encs)
    da = np.zeros(number_of_poss_encs)
    com = np.zeros(number_of_poss_encs)
    density =np.zeros(number_of_poss_encs)
    mbin =np.zeros(number_of_poss_encs)
    mass_ratio = np.zeros(number_of_poss_encs)
    hill_sphere_mass = np.zeros(number_of_poss_encs)
    hill_sphere_radius = np.zeros(number_of_poss_encs)
    hill_sphere_mass_solar = np.zeros(number_of_poss_encs)
    ratio_m_hill_bbh = np.zeros(number_of_poss_encs)
    orb_a_in_meters = np.zeros(number_of_poss_encs)
    t_orb_1 = np.zeros(number_of_poss_encs)
    t_orb_2 = np.zeros(number_of_poss_encs)
    dt = np.zeros(number_of_poss_encs)
    num_inters = np.zeros(number_of_poss_encs)
    #ids_remove = np.nan(number_of_poss_encs)

    counter_prob_less_half = 0
    max_com = 600
    min_com = 1000
    #Probability of BBH formation in Whitehead+25 when -2E_hill<E_gas<-Ehill is 50%
    prob_bbh = 0.5
    prob_no_bbh = 0.5
    #Number of columns to be removed (usually 1 per timestep but occasionally 2!)
    counter_column = []

    for i in range(number_of_poss_encs):
        id_num_1 = id_nums[0, i]
        id_num_2 = id_nums[1, i]

        mass_1[i] = blackholes_pro.at_id_num(id_num_1, "mass")
        mass_2[i] = blackholes_pro.at_id_num(id_num_2, "mass")
        orb_a_1[i] = blackholes_pro.at_id_num(id_num_1, "orb_a")
        orb_a_2[i] = blackholes_pro.at_id_num(id_num_2, "orb_a")
        # Mbin
        mbin[i] = mass_1[i] + mass_2[i]
        # orb_a should be sorted, so orb_a_2 is always > orb_a_1, else need an np.abs() on line below
        da[i] = orb_a_2[i] - orb_a_1[i]
        #Center of mass is orb_a_2[i] -da[i]*mass_1/(mass_1+mass_2)
        com[i] = orb_a_2[i] - da[i]*mass_1[i]/(mass_1[i] + mass_2[i])
        # Convert com[i] to meters
        orb_a_in_meters[i] = si_from_r_g(smbh_mass, com[i]).to("m").value
        #T_orb = pi yrs (a/10^3r_g)^3/2 for default
        t_orb_1[i] = np.pi*(orb_a_1[i]/1000.)**(3./2.)
        t_orb_2[i] = np.pi*(orb_a_2[i]/1000.)**(3./2.)
        # dt = Number of orbits required to make up phase difference per orbit (t_orb_a2>t_orb_a1)
        # E.g. if t_orb2/t_orb1 = 1.005 then the slower orbiter adds .005 phase per orbit
        # Therefore in 200 orbits, phases match again.
        # E.g. if period ~1yr then in 1 timestep (10kyrs, default), expect 1 interaction/200yrs or ~10000/200=50 interactions per default timestep
        # So if half probability of BBH formation per interaction, then probability of NOT forming a BBH over multiple encounters is ~(0.5)^50=9.e-16
        dt[i] = 1/((t_orb_2[i]/t_orb_1[i])-1)
        num_inters[i] = timestep/(t_orb_2[i]*dt[i])
        disk_height_in_meters = orb_a_in_meters[i] * disk_aspect_ratio_func(com[i])
        # Density at com is 
        density[i] = disk_surf_density_func(com[i])/disk_height_in_meters
        mass_ratio[i] = (mass_1[i]+mass_2[i])/smbh_mass
        # Hill sphere radius = R_h = com(i)*(q/3)^(1/3)
        hill_sphere_radius[i] = orb_a_in_meters[i]*((mass_ratio[i]/3.)**(1./3.))
        # Hill sphere mass = ((4./3.) pi R_hill^3 * density)
        hill_sphere_mass[i] = (4./3.)*np.pi*(hill_sphere_radius[i]**(3.0))*density[i]
        #hill_sphere_mass_in_solar = hill_sphere_mass/2.e30
        hill_sphere_mass_solar[i] = hill_sphere_mass[i]/(2.e30)
        #ratio of hill sphere mass solar/ bbh.
        ratio_m_hill_bbh[i] = hill_sphere_mass_solar[i]/mbin[i]
        if ratio_m_hill_bbh[i] < 0.103 and ratio_m_hill_bbh[i] > 0.0435:
            #print("com,density,hillmass_solar,mbin,ratio", com[i], density[i], da[i], t_orb_1[i], t_orb_2[i], dt[i] , mbin[i], ratio_m_hill_bbh[i])
            counter_prob_less_half = counter_prob_less_half + 1
            if com[i] > max_com:
                max_com = com[i]
            if com[i] < min_com:
                min_com = com[i]

            #Probability that no BBH forms is (1-prob_bbh)^(num_inters) 
            prob_no_bbh = (1-prob_bbh)**(num_inters[i])
            random_uniform_number = rng.uniform()
            if random_uniform_number < prob_no_bbh:
                #print("NO BBH forms!",prob_no_bbh,random_uniform_number)
                print("Multiple interactions failed")
                #print("number_of_bbh_forming at i",number_of_bbh_forming[:,i],number_of_bbh_forming.shape[1])
                counter_column.append(i)
                #number_of_bbh_forming = np.delete(number_of_bbh_forming, i, axis = 1)
                #print("post:",number_of_bbh_forming.shape[1])
            
        if ratio_m_hill_bbh[i] <0.0435:
            #print("com,da,t1,t2,dt,mbin,ratio",com[i],da[i],t_orb_1[i],t_orb_2[i],dt[i],timestep, num_inters[i], mbin[i],ratio_m_hill_bbh[i])
            #print("Weak gas")
            if number_of_bbh_forming.shape[0] > 0:
                counter_column.append(i)
                #print("counter_column",counter_column)
                #print("number_of_bbh_forming",number_of_bbh_forming)
                #print("number_of_bbh_forming at i",number_of_bbh_forming[:,i],number_of_bbh_forming.shape[1])
                #number_of_bbh_forming = np.delete(number_of_bbh_forming, i, axis = 1)
                #print("post:",number_of_bbh_forming.shape[1])
            #if number_of_bbh_forming.shape[1] == 0:
                #print("Special!",number_of_bbh_forming.shape[0],number_of_bbh_forming.shape[1])
                #number_of_bbh_forming = np.delete(number_of_bbh_forming, 0, axis = 1)
                
            #prob_no_bbh = 1
    
    #End for loop
    
    if len(counter_column) > 0:        
        #print("counter_column",counter_column)
        #print("number of BBH forming",number_of_bbh_forming)
        for j in range(len(counter_column)): 
            id_num_1 = number_of_bbh_forming[0, counter_column[0] ]
            id_num_2 = number_of_bbh_forming[1, counter_column[0]]
            #print("id_num_1,id_num_2",id_num_1,id_num_2)
            mass_1[j] = blackholes_pro.at_id_num(id_num_1, "mass")
            mass_2[j] = blackholes_pro.at_id_num(id_num_2, "mass")
            orb_a_1[j] = blackholes_pro.at_id_num(id_num_1, "orb_a")
            orb_a_2[j] = blackholes_pro.at_id_num(id_num_2, "orb_a")
            # Mbin
            mbin[j] = mass_1[j] + mass_2[j]
            # orb_a should be sorted, so orb_a_2 is always > orb_a_1, else need an np.abs() on line below
            da[j] = orb_a_2[j] - orb_a_1[j]
            #Center of mass is orb_a_2[i] -da[i]*mass_1/(mass_1+mass_2)
            com[j] = orb_a_2[j] - da[j]*mass_1[j]/(mass_1[j] + mass_2[j])
            #print("com[j]",com[j])
            number_of_bbh_forming = np.delete(number_of_bbh_forming, counter_column[0], axis=1)

        #print("number of BBH forming",number_of_bbh_forming)

    return counter_prob_less_half, max_com, min_com,number_of_bbh_forming

def divide_types_encounters(id_nums, encounter_categories, filing_cabinet):
    """Divide ID numbers of close encounter objects by their type.

    Takes in the (2,N) array of ID numbers of objects that will encounter each other.
    Tests each pair to see if it is a BH-BH, BH-star, or star-star pair and returns 3
    arrays with ID numbers for objects in each category.

    Parameters
    ----------
    id_nums : numpy.ndarray
        ID numbers of objects that will encounter each other
    encounter_categories : list or numpy.ndarray
        List of types of encounters
        E.g., [[0, 0], [0, 1], [1, 1]] will return 3 arrays with ID numbers for pairs
        with BH-BH, BH-star, and star-star types.
    filing_cabinet : AGNFilingCabinet
        Filing cabinet holding information for all objects in the disk

    Returns
    -------
    results : dict
        Dictionary where the key is the pairing and the value is the (2, N) array of ID numbers
    """
    if isinstance(encounter_categories, list):
        if (isinstance(encounter_categories[0], list) | isinstance(encounter_categories[0], np.ndarray)):
            pairs_list = np.array(encounter_categories)
        else:
            raise AttributeError("encounter_categories must be a list or array of N (2,) arrays")
    elif isinstance(encounter_categories, np.ndarray):
        if (isinstance(encounter_categories[0], list) | isinstance(encounter_categories[0], np.ndarray)):
            pairs_list = encounter_categories
        else:
            raise AttributeError("encounter_categories must be a list or array of N (2,) arrays")
    else:
        raise AttributeError("encounter_categories must be a list or array of N (2,) arrays")

    # Get the categories for the objects at the specified ID numbers
    obj_categories = np.vstack((filing_cabinet.at_id_num(id_nums[0], "category"),
                                filing_cabinet.at_id_num(id_nums[1], "category")))

    results = []
    for pair in pairs_list:
        # Compare categories of obj pairs to desired pair (e.g., [0, 0])
        # We use np.sort so that [0, 1] and [1, 0] both come out as True
        compare_arr = (np.sort(obj_categories.T) == np.full(obj_categories.T.shape, np.sort(pair)))
        # Get indices of where both are True
        pair_idx = np.nonzero(np.sum(compare_arr, axis=1) == 2)[0]
        # Get ID numbers
        pair_id_nums = id_nums[:, pair_idx]
        results.append(pair_id_nums)

    return (tuple(results))


def add_to_binary_obj(blackholes_binary, blackholes_pro, bh_pro_id_num_binary, id_start_val, fraction_bin_retro, smbh_mass, agn_redshift, disk_bh_pro_orb_ecc_crit):
    """Create new BH binaries with appropriate parameters.

    We take the semi-maj axis, masses, spins, spin angles and generations
    from the relevant singletons, found in hillsphere.binary_check2, and sort
    those parameters into disk_bins_bhbh. We then ADD additional parameters
    relevant only for binaries, including semi-major axis of the binary,
    semi-major axis of the orbit of the center of mass of the binary around
    the SMBH, a flag to permit or suppress retrograde binaries, eventually
    eccentricity and inclination.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes
    blackholes_pro : AGNBlackHole
        Prograde black holes
    bh_pro_id_num_binary : numpy.ndarray
        ID numbers for the prograde blackholes that will form binaries with :obj:`int` type
    id_start_val : int
        Starting value for the ID numbers (add 1 to ensure it's unique)
    fraction_bin_retro : float
        Fraction of binaries which form retrograde (wrt to the disk gas)
        around their own center of mass.
        = 0.0 turns all retrograde BBH at formation into prograde BBH.
        = 0.5 half of the binaries will be retrograde
        = 1.0 all binaries will be retrograde.
    smbh_mass : float
        Mass [M_sun] of the SMBH
    agn_redshift : float
        Redshift [unitless] of the AGN, used to set d_obs

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole object with new binaries added
    id_nums : numpy.ndarray
        ID numbers of the new binary black holes with :obj:`int` type
    """

    bin_num = bh_pro_id_num_binary.shape[1]
    id_nums = np.arange(id_start_val+1, id_start_val + 1 + bin_num, 1)
    orb_a_1 = np.zeros(bin_num)
    orb_a_2 = np.zeros(bin_num)
    mass_1 = np.zeros(bin_num)
    mass_2 = np.zeros(bin_num)
    spin_1 = np.zeros(bin_num)
    spin_2 = np.zeros(bin_num)
    spin_angle_1 = np.zeros(bin_num)
    spin_angle_2 = np.zeros(bin_num)
    bin_sep = np.zeros(bin_num)
    bin_orb_a = np.zeros(bin_num)
    time_to_merger_gw = np.zeros(bin_num)
    flag_merging = np.zeros(bin_num)
    time_merged = np.zeros(bin_num)
    # Set up binary eccentricity around its own center of mass.
    # Draw uniform value btwn [0,1]
    bin_ecc = rng.uniform(size=bin_num)
    gen_1 = np.zeros(bin_num)
    gen_2 = np.zeros(bin_num)
    bin_orb_ang_mom = np.zeros(bin_num)
    # Set up binary inclination (in units radians). Will want this
    # to be pi radians if retrograde.
    bin_orb_inc = np.zeros(bin_num)
    # Set up binary orbital eccentricity of com around SMBH.
    # Assume initially v.small (e~0.01 = disk_bh_pro_orb_ecc_crit)
    bin_orb_ecc = np.full(bin_num, disk_bh_pro_orb_ecc_crit)
    galaxy = np.zeros(bin_num)

    for i in range(bin_num):
        id_num_1 = bh_pro_id_num_binary[0, i]
        id_num_2 = bh_pro_id_num_binary[1, i]

        mass_1[i] = blackholes_pro.at_id_num(id_num_1, "mass")
        mass_2[i] = blackholes_pro.at_id_num(id_num_2, "mass")
        orb_a_1[i] = blackholes_pro.at_id_num(id_num_1, "orb_a")
        orb_a_2[i] = blackholes_pro.at_id_num(id_num_2, "orb_a")
        spin_1[i] = blackholes_pro.at_id_num(id_num_1, "spin")
        spin_2[i] = blackholes_pro.at_id_num(id_num_2, "spin")
        spin_angle_1[i] = blackholes_pro.at_id_num(id_num_1, "spin_angle")
        spin_angle_2[i] = blackholes_pro.at_id_num(id_num_2, "spin_angle")
        gen_1[i] = blackholes_pro.at_id_num(id_num_1, "gen")
        gen_2[i] = blackholes_pro.at_id_num(id_num_1, "gen")
        bin_sep[i] = np.abs(orb_a_1[i] - orb_a_2[i])
        galaxy[i] = blackholes_pro.at_id_num(id_num_1, "galaxy")

        # Binary c.o.m.= location_1 + separation*M_2/(M_1+M_2)
        bin_orb_a[i] = orb_a_1[i] + ((bin_sep[i] * mass_2[i]) / (mass_1[i] + mass_2[i]))

        gen_1[i] = blackholes_pro.at_id_num(id_num_1, "gen")
        gen_2[i] = blackholes_pro.at_id_num(id_num_2, "gen")

        # Set up bin orb. ang. mom.
        # (randomly +1 (pro) or -1(retrograde))
        # random number between [0,1]
        # If fraction_bin_retro =0, range = [0,1] and set L_bbh = +1.
        if fraction_bin_retro == 0:
            bin_orb_ang_mom[i] = 1.
        else:
            # return a 1 or -1 in the ratio 
            # (1-fraction_bin_retro: fraction_bin_retro)
            bin_orb_ang_mom[i] = rng.choice(a=[1, -1], p=[1-fraction_bin_retro, fraction_bin_retro])

    gw_strain, gw_freq = gw_strain_freq(mass_1=mass_1, mass_2=mass_2, obj_sep=bin_sep, timestep_duration_yr=-1,
                                        old_gw_freq=-1, smbh_mass=smbh_mass, agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=0)

    blackholes_binary.add_binaries(new_orb_a_1=orb_a_1,
                                   new_orb_a_2=orb_a_2,
                                   new_mass_1=mass_1,
                                   new_mass_2=mass_2,
                                   new_spin_1=spin_1,
                                   new_spin_2=spin_2,
                                   new_spin_angle_1=spin_angle_1,
                                   new_spin_angle_2=spin_angle_2,
                                   new_bin_sep=bin_sep,
                                   new_bin_orb_a=bin_orb_a,
                                   new_time_to_merger_gw=time_to_merger_gw,
                                   new_flag_merging=flag_merging,
                                   new_time_merged=time_merged,
                                   new_bin_ecc=bin_ecc,
                                   new_gen_1=gen_1,
                                   new_gen_2=gen_2,
                                   new_bin_orb_ang_mom=bin_orb_ang_mom,
                                   new_bin_orb_inc=bin_orb_inc,
                                   new_bin_orb_ecc=bin_orb_ecc,
                                   new_gw_freq=gw_freq,
                                   new_gw_strain=gw_strain,
                                   new_id_num=id_nums,
                                   new_galaxy=galaxy)

    assert np.all(blackholes_binary.bin_sep >= 0), \
        "blackholes_binary.bin_sep has values <0"
    assert np.all(blackholes_binary.orb_a_1 > 0), \
        "blackholes_binary.orb_a_1 has values <=0"
    assert np.all(blackholes_binary.orb_a_2 > 0), \
        "blackholes_binary.orb_a_2 has values <= 0"
    assert np.all(blackholes_binary.mass_1 > 0), \
        "blackholes_binary.mass_1 has values <= 0"
    assert np.all(blackholes_binary.mass_2 > 0), \
        "blackholes_binary.mass_2 has values <= 0"
    assert np.all(blackholes_binary.bin_orb_a > 0), \
        "blackholes_binary.bin_orb_a has values <= 0"

    return (blackholes_binary, id_nums)
