"""
Module for evolving the state of a binary.
"""
import numpy as np
import scipy


def change_bin_mass(blackholes_binary, disk_bh_eddington_ratio,
                    disk_bh_eddington_mass_growth_rate, timestep_duration_yr):
    """Add mass to binary components according to chosen BH mass accretion prescription

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes in prograde orbits around SMBH
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    mdisk_bh_eddington_mass_growth_rate : float
        Fractional rate of mass growth [yr^{-1}] AT Eddington accretion rate per year (fixed at 2.3e-8 in mcfacts_sim)
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes with updated masses after accreting at prescribed rate for one timestep
    """

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)

    # If all BH have merged then nothing to do
    if (idx_non_mergers[0].shape[0] == 0):
        return (blackholes_binary)

    mass_growth_factor = np.exp(disk_bh_eddington_mass_growth_rate * disk_bh_eddington_ratio * timestep_duration_yr)

    mass_1_before = blackholes_binary.mass_1[idx_non_mergers]
    mass_2_before = blackholes_binary.mass_2[idx_non_mergers]

    blackholes_binary.mass_1[idx_non_mergers] = mass_1_before * mass_growth_factor
    blackholes_binary.mass_2[idx_non_mergers] = mass_2_before * mass_growth_factor

    return (blackholes_binary)


def change_bin_spin_magnitudes(blackholes_binary, disk_bh_eddington_ratio,
                               disk_bh_torque_condition, timestep_duration_yr):
    """Add spin according to chosen BH torque prescription

    Given initial binary black hole spins at start of timestep_duration_yr, add spin according to
    chosen BH torque prescription. If spin is greater than max allowed spin, spin is set to max value.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes in prograde orbits around SMBH
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    disk_bh_torque_condition : float
        Fraction of initial mass required to be accreted before BH spin is torqued fully into
        alignment with the AGN disk. We don't know for sure but Bogdanovic et al. says
        between 0.01=1% and 0.1=10% is what is required
        User chosen input set by input file
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes with updated spins after spinning up at prescribed rate for one timestep
    """

    disk_bh_eddington_ratio_normalized = disk_bh_eddington_ratio/1.0  # does nothing?
    timestep_duration_yr_normalized = timestep_duration_yr/1.e4  # yrs to yr/10k?
    disk_bh_torque_condition_normalized = disk_bh_torque_condition/0.1  # what does this do?

    # Set max allowed spin
    max_allowed_spin = 0.98

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)

    # If all BH have merged then nothing to do
    if (idx_non_mergers[0].shape[0] == 0):
        return (blackholes_binary)

    spin_change_factor = 4.4e-3 * disk_bh_eddington_ratio_normalized * disk_bh_torque_condition_normalized * timestep_duration_yr_normalized

    spin_1_before = blackholes_binary.spin_1[idx_non_mergers]
    spin_2_before = blackholes_binary.spin_2[idx_non_mergers]

    spin_1_after = spin_1_before + spin_change_factor
    spin_2_after = spin_2_before + spin_change_factor

    spin_1_after[spin_1_after > max_allowed_spin] = max_allowed_spin
    spin_2_after[spin_2_after > max_allowed_spin] = max_allowed_spin

    blackholes_binary.spin_1[idx_non_mergers] = spin_1_after
    blackholes_binary.spin_2[idx_non_mergers] = spin_2_after

    return (blackholes_binary)


def change_bin_spin_angles(blackholes_binary, disk_bh_eddington_ratio,
                           disk_bh_torque_condition, spin_minimum_resolution,
                           timestep_duration_yr):
    """Subtract spin angle according to chosen BH torque prescription

    Given initial binary black hole spin angles at start of timestep, subtract spin angle
    according to chosen BH torque prescription. If spin angle is less than spin minimum
    resolution, spin angle is set to 0.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        binary black holes in prograde orbits around the SMBH
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    disk_bh_torque_condition : float
        Fraction of initial mass required to be accreted before BH spin is torqued fully into
        alignment with the AGN disk. We don't know for sure but Bogdanovic et al. says
        between 0.01=1% and 0.1=10% is what is required
        User chosen input set by input file
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes with updated spin angles after subtracting angle at prescribed rate for one timestep
    """
    disk_bh_eddington_ratio_normalized = disk_bh_eddington_ratio/1.0  # does nothing?
    timestep_duration_yr_normalized = timestep_duration_yr/1.e4  # yrs to yr/10k?
    disk_bh_torque_condition_normalized = disk_bh_torque_condition/0.1  # what does this do?

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)

    # If all BH have merged then nothing to do
    if (idx_non_mergers[0].shape[0] == 0):
        return (blackholes_binary)

    spin_angle_change_factor = 6.98e-3 * disk_bh_eddington_ratio_normalized * disk_bh_torque_condition_normalized * timestep_duration_yr_normalized

    spin_angle_1_before = blackholes_binary.spin_angle_1[idx_non_mergers]
    spin_angle_2_before = blackholes_binary.spin_angle_2[idx_non_mergers]

    spin_angle_1_after = spin_angle_1_before - spin_angle_change_factor
    spin_angle_2_after = spin_angle_2_before - spin_angle_change_factor

    spin_angle_1_after[spin_angle_1_after < spin_minimum_resolution] = 0.0
    spin_angle_2_after[spin_angle_2_after < spin_minimum_resolution] = 0.0

    blackholes_binary.spin_angle_1[idx_non_mergers] = spin_angle_1_after
    blackholes_binary.spin_angle_2[idx_non_mergers] = spin_angle_2_after

    return (blackholes_binary)


def bin_com_feedback_hankla(blackholes_binary, disk_surface_density, disk_opacity_func, disk_bh_eddington_ratio, disk_alpha_viscosity, disk_radius_outer):
    """Calculates ratio of heating torque to migration torque using Eqn. 28 in Hankla, Jiang & Armitage (2020)

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black holes
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_opacity_model : lambda
        Opacity as a function of radius
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    disk_alpha_viscosity : float
        Disk gas viscocity [units??] alpha parameter
    disk_radius_outer : float
            Outer radius [r_{g,SMBH}] of the disk

    Returns
    -------
    ratio_feedback_to_mig : float array
        Ratio of feedback torque to migration torque [unitless]

    Notes
    -----
    This feedback model uses Eqn. 28 in Hankla, Jiang & Armitage (2020)
    which yields the ratio of heating torque to migration torque.
    Heating torque is directed outwards. 
    So, Ratio <1, slows the inward migration of an object. Ratio>1 sends the object migrating outwards.
    The direction & magnitude of migration (effected by feedback) will be executed in type1.py.

    The ratio of torque due to heating to Type 1 migration torque is calculated as
    R   = Gamma_heat/Gamma_mig 
        ~ 0.07 (speed of light/ Keplerian vel.)(Eddington ratio)(1/optical depth)(1/alpha)^3/2
    where Eddington ratio can be >=1 or <1 as needed,
    optical depth (tau) = Sigma* kappa
    alpha = disk_alpha_viscosity (e.g. alpha = 0.01 in Sirko & Goodman 2003)
    kappa = 10^0.76 cm^2 g^-1=5.75 cm^2/g = 0.575 m^2/kg for most of Sirko & Goodman disk model (see Fig. 1 & sec 2)
    but e.g. electron scattering opacity is 0.4 cm^2/g
    So tau = Sigma*0.575 where Sigma is in kg/m^2.
    Since v_kep = c/sqrt(a(r_g)) then
    R   ~ 0.07 (a(r_g))^{1/2}(Edd_ratio) (1/tau) (1/alpha)^3/2
    So if assume a=10^3r_g, Sigma=7.e6kg/m^2, alpha=0.01, tau=0.575*Sigma (SG03 disk model), Edd_ratio=1, 
    R   ~5.5e-4 (a/10^3r_g)^(1/2) (Sigma/7.e6) v.small modification to in-migration at a=10^3r_g
        ~0.243 (R/10^4r_g)^(1/2) (Sigma/5.e5)  comparable.
        >1 (a/2x10^4r_g)^(1/2)(Sigma/) migration is *outward* at >=20,000r_g in SG03
        >10 (a/7x10^4r_g)^(1/2)(Sigma/) migration outwards starts to runaway in SG03
    """

    # Making sure that surface density is a float or a function (from old function)
    if not isinstance(disk_surface_density, float):
        disk_surface_density_at_location = disk_surface_density(blackholes_binary.bin_orb_a)
    else:
        raise AttributeError("disk_surface_density is a float")

    # Define kappa (or set up a function to call).
    disk_opacity = disk_opacity_func(blackholes_binary.bin_orb_a)

    ratio_heat_mig_torques_bin_com = 0.07 * (1 / disk_opacity) * (disk_alpha_viscosity ** -1.5) * disk_bh_eddington_ratio * np.sqrt(blackholes_binary.bin_orb_a) / disk_surface_density_at_location

    ratio_heat_mig_torques_bin_com[blackholes_binary.bin_orb_a > disk_radius_outer] = 1.0

    return (ratio_heat_mig_torques_bin_com)


def bin_ionization_check(blackholes_binary, smbh_mass):
    """Tests whether binary has been ionized beyond some limit

    This function tests whether a binary has been softened beyond some limit.
    Returns ID numbers of binaries to be ionized.
    The limit is set to some fraction of the binary Hill sphere, frac_R_hill

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    smbh_mass : float
        Mass [M_sun] of the SMBH

    Returns
    -------
    bh_id_nums : numpy.ndarray
        ID numbers of binaries to be removed from binary array

    Notes
    -----
    Default is frac_R_hill = 1.0 (ie binary is ionized at the Hill sphere). 
    Change frac_R_hill if you're testing binary formation at >R_hill.

    R_hill = a_com*(M_bin/3M_smbh)^1/3

    where a_com is the radial disk location of the binary center of mass,
    M_bin = M_1 + M_2 is the binary mass
    M_smbh is the SMBH mass (given by smbh_mass) 

    Condition is:
    if bin_separation > frac_R_hill*R_hill:
        Ionize binary.
        Remove binary from blackholes_binary!
        Add two new singletons to the singleton arrays.
    """

    # Remove returning -1 if that's not how it's supposed to work
    # Define ionization threshold as a fraction of Hill sphere radius
    # Default is 1.0, change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    # bin_orb_a is in units of r_g of the SMBH = GM_smbh/c^2
    mass_ratio = blackholes_binary.mass_total/smbh_mass
    hill_sphere = blackholes_binary.bin_orb_a * ((mass_ratio / 3) ** (1. / 3.))

    bh_id_nums = blackholes_binary.id_num[np.where(blackholes_binary.bin_sep > (frac_rhill * hill_sphere))[0]]

    return (bh_id_nums)


def bin_contact_check(blackholes_binary, smbh_mass):
    """Tests if binary separation has shrunk so that binary is touching

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    smbh_mass : float
        Mass [M_sun] of the SMBH

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Returns modified blackholes_binary with updated bin_sep and flag_merging.

    Notes
    -----
    Touching condition is where binary separation is <= R_schw(M_1) + R_schw(M_2)
                                                      = 2(R_g(M_1) + R_g(M_2))
                                                      = 2G(M_1+M_2) / c^{2}

    Since binary separation is in units of r_g (GM_smbh/c^2) then condition is simply:
        binary_separation <= 2M_bin/M_smbh
    """

    mass_binary = blackholes_binary.mass_1 + blackholes_binary.mass_2

    # We assume bh are not spinning when in contact. TODO: Consider spin in future.
    contact_condition = 2 * (mass_binary / smbh_mass)
    mask_condition = (blackholes_binary.bin_sep <= contact_condition)

    # If binary separation <= contact condition, set binary separation to contact condition
    blackholes_binary.bin_sep[mask_condition] = contact_condition[mask_condition]
    blackholes_binary.flag_merging[mask_condition] = -2

    return (blackholes_binary)


def bin_reality_check(blackholes_binary):
    """Tests if binaries are real (location and mass do not equal 0)

    This function tests to see if the binary is real. If location = 0 or mass = 0 *and* any other element is NON-ZERO then discard this binary element.
    Returns ID numbers of fake binaries.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters

    Returns
    -------
    id_nums or bh_bin_id_num_fakes : numpy.ndarray
        ID numbers of fake binaries with :obj:`float` type
    """
    bh_bin_id_num_fakes = np.array([])

    mass_1_id_num = blackholes_binary.id_num[blackholes_binary.mass_1 == 0]
    mass_2_id_num = blackholes_binary.id_num[blackholes_binary.mass_2 == 0]
    orb_a_1_id_num = blackholes_binary.id_num[blackholes_binary.orb_a_1 == 0]
    orb_a_2_id_num = blackholes_binary.id_num[blackholes_binary.orb_a_2 == 0]
    bin_ecc_id_num = blackholes_binary.id_num[blackholes_binary.bin_ecc >= 1]

    id_nums = np.concatenate([mass_1_id_num, mass_2_id_num,
                             orb_a_1_id_num, orb_a_2_id_num, bin_ecc_id_num])

    if id_nums.size > 0:
        return (id_nums)
    else:
        return (bh_bin_id_num_fakes)


def bin_harden_baruteau(blackholes_binary, smbh_mass, timestep_duration_yr,
                        time_gw_normalization, time_passed):
    """Harden black hole binaries using Baruteau+11 prescription

    Use Baruteau+11 prescription to harden a pre-existing binary.
    For every 1000 orbits of binary around its center of mass, the
    separation (between binary components) is halved.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters
    smbh_mass : float
        Mass [M_sun] of the SMBH
    timestep_duration_yr : float
        Length of timestep [yr]
    time_gw_normalization : float
        A normalization for GW decay timescale, set by `smbh_mass` & normalized for
        a binary total mass of 10 solar masses.
    bin_index : int
        Count of number of binaries
    time_passed : float
        Time elapsed [yr] since beginning of simulation.

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Black hole binaries with time_to_merger_gw, bin_sep, flag_merging, and time_merged updated
    """

    # 1. Find active binaries
    # 2. Find number of binary orbits around its center of mass within the timestep
    # 3. For every 10^3 orbits, halve the binary separation.

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)[0]

    # If all binaries have merged then nothing to do
    if (idx_non_mergers.shape[0] == 0):
        return blackholes_binary

    # Set up variables
    mass_binary = blackholes_binary.mass_1[idx_non_mergers] + blackholes_binary.mass_2[idx_non_mergers]
    mass_reduced = (blackholes_binary.mass_1[idx_non_mergers] * blackholes_binary.mass_2[idx_non_mergers]) / mass_binary
    bin_sep = blackholes_binary.bin_sep[idx_non_mergers]
    bin_orb_ecc = blackholes_binary.bin_ecc[idx_non_mergers]

    # Find eccentricity factor (1-e_b^2)^7/2
    ecc_factor_1 = ((1 - (bin_orb_ecc ** 2.0)) ** 3.5)
    # and eccentricity factor [1+(73/24)e_b^2+(37/96)e_b^4]
    ecc_factor_2 = 1 + ((73/24) * (bin_orb_ecc ** 2.0)) + ((37/96) * (bin_orb_ecc ** 4.0))
    # overall ecc factor = ecc_factor_1/ecc_factor_2
    ecc_factor = ecc_factor_1/ecc_factor_2

    # Binary period = 2pi*sqrt((delta_r)^3/GM_bin)
    # or T_orb = 10^7s*(1r_g/m_smmbh=10^8Msun)^(3/2) *(M_bin/10Msun)^(-1/2) = 0.32yrs
    bin_period = 0.32 * (bin_sep ** 1.5) * ((smbh_mass/1.e8) ** 1.5) * ((mass_binary/10.0) ** -0.5)

    # Find how many binary orbits in timestep. Binary separation is halved for every 10^3 orbits.
    num_orbits_in_timestep = np.zeros(len(bin_period))
    num_orbits_in_timestep[bin_period > 0] = timestep_duration_yr / bin_period[bin_period > 0]
    scaled_num_orbits = num_orbits_in_timestep / 1000.0

    # Timescale for binary merger via GW emission alone, scaled to bin parameters
    time_to_merger_gw = time_gw_normalization * ((bin_sep ** 4.0)) * ((mass_binary/10.0) ** -2) * ((mass_reduced / 2.5) ** -1.0) * ecc_factor
    # Finite check
    assert np.isfinite(time_to_merger_gw).all(),\
        "Finite check failure: time_to_merger_gw"
    blackholes_binary.time_to_merger_gw[idx_non_mergers] = time_to_merger_gw

    # Binary will not merge in this timestep
    # new bin_sep according to Baruteu+11 prescription
    bin_sep[time_to_merger_gw > timestep_duration_yr] = bin_sep[time_to_merger_gw > timestep_duration_yr] * (0.5 ** scaled_num_orbits[time_to_merger_gw > timestep_duration_yr])
    blackholes_binary.bin_sep[idx_non_mergers[time_to_merger_gw > timestep_duration_yr]] = bin_sep[time_to_merger_gw > timestep_duration_yr]
    # Finite check
    assert np.isfinite(blackholes_binary.bin_sep).all(),\
        "Finite check failure: blackholes_binary.bin_sep"

    # Otherwise binary will merge in this timestep
    # Update flag_merging to -2 and time_merged to current time
    blackholes_binary.flag_merging[idx_non_mergers[time_to_merger_gw <= timestep_duration_yr]] = -2
    blackholes_binary.time_merged[idx_non_mergers[time_to_merger_gw <= timestep_duration_yr]] = time_passed
    # Finite check
    assert np.isfinite(blackholes_binary.flag_merging).all(),\
        "Finite check failure: blackholes_binary.flag_merging"
    # Finite check
    assert np.isfinite(blackholes_binary.time_merged).all(),\
        "Finite check failure: blackholes_binary.time_merged"

    return (blackholes_binary)
