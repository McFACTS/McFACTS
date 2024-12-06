"""
Module for handling the formation of binaries.
"""
import numpy as np

from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.gw import gw_strain_freq


def binary_check(
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        smbh_mass,
        disk_bh_pro_orbs_ecc,
        disk_bh_pro_orb_ecc_crit
        ):
    """Calculates which prograde BH will form binaries in this timestep.

    Takes as inputs the singleton BH locations,masses & orbital eccentricities,
    and takes the candidate binary population from BH with orbital eccentricities
    damped to < orb_ecc_crit. Among this damped population, checks if their
    separations are less than the mutual Hill sphere of any 2 adjacent BH. If this
    is the case, determine the smallest separation pairs (in units of their mutual
    Hill sphere) to form a set of actual binaries (this module does handle cases where
    3 or more bodies *might* form some set of binaries which would be mutually exclusive;
    however it does not handle or even flag the implied triple system dynamics).
    Returns a 2xN array of the relevant binary indices, for further handling to form actual
    binaries & assign additional parameters (e.g. angular momentum of the binary).

    Parameters
    ----------
    disk_bh_pro_orbs_a : float array
        Semi-major axes around the SMBH [r_{g,SMBH}] of prograde singleton BH at start of timestep
    disk_bh_pro_masses : float array
        Initial masses [M_sun] of bh in prograde orbits around SMBH
    smbh_mass : float
        Mass [M_sun] of the SMBH
    disk_bh_pro_orbs_ecc : float array
        Orbital ecc [unitless] of singleton BH after damping during timestep
    disk_bh_pro_orb_ecc_crit : float
        Critical eccentricity [unitless] allowing bin formation and migration

    Returns
    -------
    disk_bin_bhbh_pro_indices : [2,N] int array
        array of indices corresponding to locations in disk_bh_pro_orbs_a,
        disk_bh_pro_masses, etc. which corresponds to binaries that form in
        this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.

    Notes
    -----
    Internal variable names not standardized. Fix later.
    """

    # First check for BH with sufficiently damped orbital eccentricity
    # (orb_ecc<=orb_ecc_crit (usually 0.01)).
    # This population is the sub-set of prograde BH from which we CAN form
    # binaries.

    # Singleton BH with orb ecc < e_crit (candidates for binary formation)
    indices_bh_can_form_bins = np.asarray(disk_bh_pro_orbs_ecc <= disk_bh_pro_orb_ecc_crit).nonzero()[0]
    # Indices of those candidates for binary formation
    allowed_to_form_bins = np.array(indices_bh_can_form_bins)
    # Sort the location of the candidates
    sorted_bh_locations = np.sort(disk_bh_pro_orbs_a[allowed_to_form_bins])
    # Sort the indices of all singleton BH (the superset)
    sorted_bh_location_indices_superset = np.argsort(disk_bh_pro_orbs_a)
    # Set the condition for membership in candidate array to be searched/tested
    condition = np.isin(sorted_bh_location_indices_superset, allowed_to_form_bins)
    # Here is the subset of indices that can be tested for binarity
    subset = np.extract(condition, sorted_bh_location_indices_superset)

    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    # Note length of separations is 1 less than disk_bh_pro_orbs_a
    # This is the set of separations between the sorted candidate BH
    separations = np.diff(sorted_bh_locations)

    # Now compute mutual hill spheres of all possible candidate binaries if can test
    if len(separations) > 0:
        R_Hill_possible_binaries = (sorted_bh_locations[:-1] + separations/2.0) * \
            pow(((disk_bh_pro_masses[subset[:-1]] + \
                  disk_bh_pro_masses[subset[1:]]) / \
                    (smbh_mass * 3.0)), (1.0/3.0))
        # compare separations to mutual Hill spheres - negative values mean possible binary formation
        minimum_formation_criteria = separations - R_Hill_possible_binaries

        # collect indices of possible real binaries (where separation is less than mutual Hill sphere)
        index_formation_criteria = np.where(minimum_formation_criteria < 0)

        #Here's the index of the array of candidates
        test_idx = index_formation_criteria[0]

        #If we actually have any candidates this time step
        if np.size(test_idx) > 0:
            # Start with real index (from full singleton array) of 1st candidate binary component (implicit + 1 partner since separations are ordered )
            bin_indices = np.array([subset[test_idx[0]],subset[test_idx[0]+1]])
            # If only 1 binary this timestep, return this binary!
            disk_bin_bhbh_pro_indices = np.array([subset[test_idx],subset[test_idx+1]])

            for i in range(len(test_idx)):
                # If more than 1 binary
                if i > 0:
                    # append nth binary indices formed this timestep
                    bin_indices = np.append(bin_indices, [subset[test_idx[i]],subset[test_idx[i]+1]])

                    #Check to see if repeat binaries among the set of binaries formed (e.g. (1,2)(2,3) )
                    #If repeats, only form a binary from the pair with smallest fractional Hill sphere separation

                    # Compute separation/R_Hill for all
                    sequences_to_test = (separations[test_idx])/(R_Hill_possible_binaries[test_idx])
                    # sort sep/R_Hill for all 'binaries' that need checking & store indices
                    sorted_sequences = np.sort(sequences_to_test)
                    #Sort the indices for the test
                    sorted_sequences_indices = np.argsort(sequences_to_test)

                    # Assume the smallest sep/R_Hill should form a binary, so
                    if len(sorted_sequences) > 0:
                        # Index of smallest sorted fractional Hill radius binary so far
                        checked_binary_index = np.array([test_idx[sorted_sequences_indices[0]]])
                    else:
                        checked_binary_index = []
                    for j in range(len(sorted_sequences)):
                        # if we haven't already counted it
                        if (test_idx[sorted_sequences_indices[j]] not in checked_binary_index):
                            # and it isn't the implicit partner of something we've already counted
                            if (test_idx[sorted_sequences_indices[j]] not in checked_binary_index+1):
                                # and the implicit partner of this thing isn't already counted
                                if (test_idx[sorted_sequences_indices[j]]+1 not in checked_binary_index):
                                    # and the implicit partner of this thing isn't already an implicit partner we've counted
                                    if (test_idx[sorted_sequences_indices[j]]+1 not in checked_binary_index+1):
                                        # then you can count it as a real binary
                                        checked_binary_index = np.append(checked_binary_index, test_idx[sorted_sequences_indices[j]])
                    disk_bin_bhbh_pro_indices = np.array([subset[checked_binary_index],subset[checked_binary_index+1]])

        else:
            # No binaries from candidates this time step
            disk_bin_bhbh_pro_indices = []

    else:
        # No candidate for binarity testing yet
        disk_bin_bhbh_pro_indices = []

    return disk_bin_bhbh_pro_indices


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

    return (blackholes_binary, id_nums)
