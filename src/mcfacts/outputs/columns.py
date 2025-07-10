"""File for defining columns for outputfiles in mcfacts_sim.py

With them here, they can be imported by people using the data,
    to properly interpret mcfacts outputs.
"""

# columns to write for incremental data files
merger_cols = ["galaxy", "bin_orb_a", "mass_final", "chi_eff", "spin_final",
    "spin_angle_final", "mass_1", "mass_2", "spin_1", "spin_2",
    "spin_angle_1", "spin_angle_2", "gen_1", "gen_2", "time_merged",
]
binary_cols = ["orb_a_1", "orb_a_2", "mass_1", "mass_2", "spin_1", "spin_2",
    "spin_angle_1", "spin_angle_2", "bin_sep", "bin_orb_a",
    "time_to_merger_gw", "flag_merging", "time_merged", "bin_ecc",
    "gen_1", "gen_2", "bin_orb_ang_mom", "bin_orb_inc", "bin_orb_ecc", 
    "gw_freq", "gw_strain", "id_num",
]

# Define columns to write
emri_cols = [
    "galaxy", "time_passed", "orb_a", "mass", "orb_ecc", "gw_strain",
    "gw_freq", "id_num",
]
bh_surviving_cols = [
    "galaxy", "orb_a", "mass", "spin", "spin_angle", "gen", "id_num"
]
bh_cols = [
    "galaxy", "time_passed", "orb_a", "mass", "orb_ecc", "spin",
    "spin_angle", "orb_inc", "orb_ang_mom", "gen", "id_num"
    ]
population_cols = [
    "galaxy", "bin_orb_a", "mass_final", "chi_eff", "spin_final",
    "spin_angle_final", "mass_1", "mass_2", "spin_1", "spin_2",
    "spin_angle_1", "spin_angle_2", "gen_1", "gen_2", "time_merged",
    "chi_p", "v_kick", "lum_shock", "lum_jet", "id_num",
]
binary_gw_cols = [
    "galaxy", "time_merged", "bin_sep", "mass_total", "bin_ecc", 
    "gw_strain", "gw_freq", "gen_1", "gen_2", "id_num",
]
stars_cols = [
    "galaxy", "time_passed", "orb_a", "mass", "orb_ecc", "log_radius", 
    "gen", "id_num", "log_teff", "log_luminosity", "star_X", "star_Y", "star_Z",
]
stars_explode_cols = [
    "galaxy", "time_sn", "orb_a_star", "mass_star", "orb_ecc_star",
    "star_log_radius", "gen_star", "id_num_star", "orb_inc_star",
    "orb_a_bh", "mass_bh", "orb_ecc_bh", "gen_bh", "id_num_bh", "orb_inc_bh",
]
tde_cols = [
    "galaxy", "time_passed", "orb_a", "mass", "orb_ecc", "log_radius", "gen", 
    "id_num", "log_teff", "log_luminosity", "star_X", "star_Y", "star_Z",
]
stars_merge_cols = [
    "galaxy", "time_merged","orb_a_final", "mass_final", "orb_ecc", 
    "log_radius_final", "gen_final", "id_num", "mass_1", "mass_2",
    "gen_1", "gen_2"
]
