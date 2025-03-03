"""
Module for calculating the final variables of two merging stars.
"""

import numpy as np

from mcfacts.physics import stellar_interpolation


def add_merged_stars(id_nums_mergers, stars, id_start_val, disk_bh_pro_orb_ecc_crit, disk_star_initial_mass_cutoff):

    merge_num = id_nums_mergers.shape[1]
    id_nums = np.arange(id_start_val + 1, id_start_val + 1 + merge_num, 1)
    # Merged mass is just masses added together. Any over disk_star_initial_mass_cutoff get set to disk_star_initial_mass_cutoff
    masses = stars.at_id_num(id_nums_mergers[0], "mass") + stars.at_id_num(id_nums_mergers[1], "mass")
    # New orb_a is the center of mass of the two stars
    orbs_a = ((stars.at_id_num(id_nums_mergers[0], "mass") * stars.at_id_num(id_nums_mergers[0], "orb_a")) +
              (stars.at_id_num(id_nums_mergers[1], "mass") * stars.at_id_num(id_nums_mergers[1], "orb_a"))) / masses
    # After doing the weighted average for orb_a we then cut off stars with mass > disk_star_initial_mass_cutoff
    masses[masses > disk_star_initial_mass_cutoff] = disk_star_initial_mass_cutoff
    # New gen is the maximum between the pair's gen plus one
    gens = np.maximum(stars.at_id_num(id_nums_mergers[0], "gen"), stars.at_id_num(id_nums_mergers[1], "gen")) + 1.0
    # Radius, luminosity, Teff are all interpolated based on the new mass
    logR, logL, logTeff = stellar_interpolation.interp_star_params(masses)
    # No metallicity evolution
    star_X = stars.at_id_num(id_nums_mergers[0], "star_X")
    star_Y = stars.at_id_num(id_nums_mergers[0], "star_Y")
    star_Z = stars.at_id_num(id_nums_mergers[0], "star_Z")
    # orb_ang_mom is +1 because two prograde stars merge
    orb_ang_mom = np.ones(merge_num)
    # orb_inc is zero
    orbs_inc = np.zeros(merge_num)
    # orb_ecc is initially very small
    orbs_ecc = np.full(merge_num, disk_bh_pro_orb_ecc_crit)
    # Assume orb_arg_periapse is same as before
    orb_arg_periapse = stars.at_id_num(id_nums_mergers[0], "orb_arg_periapse")
    # galaxy is same as before
    galaxy = stars.at_id_num(id_nums_mergers[0], "galaxy")
    # time passed is same as before
    time_passed = stars.at_id_num(id_nums_mergers[0], "time_passed")

    # Add to stars object
    stars.add_stars(new_id_num=id_nums,
                    new_mass=masses,
                    new_orb_a=orbs_a,
                    new_log_radius=logR,
                    new_log_teff=logTeff,
                    new_log_luminosity=logL,
                    new_X=star_X,
                    new_Y=star_Y,
                    new_Z=star_Z,
                    new_orb_ang_mom=orb_ang_mom,
                    new_orb_ecc=orbs_ecc,
                    new_orb_inc=orbs_inc,
                    new_orb_arg_periapse=orb_arg_periapse,
                    new_gen=gens,
                    new_galaxy=galaxy,
                    new_time_passed=time_passed
                    )

    # Delete id nums from stars object
    stars.remove_id_num(id_num_remove=id_nums_mergers.flatten())

    return (stars, id_nums)
