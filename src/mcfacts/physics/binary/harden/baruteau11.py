import numpy as np


def bin_harden_baruteau(binary_bh_array, smbh_mass, timestep_duration_yr, norm_tgw, time_passed_yr):
    """Harden black hole binaries using Baruteau+11 prescription

    Parameters
    ----------
    disk_bins_bhbh : ndarray
        Array of binary black holes in the disk
    smbh_mass : ndarray
        Mass of supermassive black hole
    timestep_duration_yr : float
        Length of timestep in years
    norm_tgw : float
        Normalization factor for gravitational wave energy timescale
    time_passed_yr : float
        Time elapsed since beginning of active phase

    Returns
    -------
    array
        Updated array of black hole binaries in the disk
    """
    # Use Baruteau+11 prescription to harden a pre-existing binary.
    # Scaling is: for every 1000 orbits of binary around its center of mass,
    # separation (between binary components) is halved.

    # bin_array is array of binaries.
    # integer_nbinprop is int(number of properties of each bin. E.g. 13: mass_1,mass_2 etc.)
    # mass_smbh is mass of SMBH
    # timestep is time interval in simulation (usually 10kyr)
    # norm_tgw is a normalization for GW decay timescale, set by mass_smbh & normalized to M_bin=10Msun
    # bin_index is the count of binaries in bin_array.
    # Could read this from e.g. number of non-zero elements in row1 (or whatever) of bin_array
    # Now added ecc_factor = (1-e_b^2)^7/2* 1/(1+(73/24)e_b^2 + (37/96)e_b^4)
    # where e_b = eccentricity of binary around it's own center of mass.
    # number_of_bin_properties = index i (row) of bin_array[i,j]
    # bin_index = index j (column) of bin_array[i,j].



    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    # Find number of binary orbits around its center of mass.
    # For every 10^3 orbits, halve the binary separation.
    # 
    for j in range(0, bin_index):
        if disk_bins_bhbh[11,j] < 0:
            # do nothing - merger happened!
            continue
        else:
            temp_bh_loc_1 = disk_bins_bhbh[0,j]
            temp_bh_loc_2 = disk_bins_bhbh[1,j]
            temp_bh_mass_1 = disk_bins_bhbh[2,j]
            temp_bh_mass_2 = disk_bins_bhbh[3,j]
            temp_bh_spin_1 = disk_bins_bhbh[4,j]
            temp_bh_spin_2 = disk_bins_bhbh[5,j]
            temp_bh_spin_angle_1 = disk_bins_bhbh[6,j]
            temp_bh_spin_angle_2 = disk_bins_bhbh[7,j]
            temp_bin_separation = disk_bins_bhbh[8,j]
            temp_bin_ecc_around_com = disk_bins_bhbh[13,j]
            temp_bin_mass = temp_bh_mass_1 + temp_bh_mass_2
            temp_bin_reduced_mass = (temp_bh_mass_1*temp_bh_mass_2)/temp_bin_mass
            #Find eccentricity factor (1-e_b^2)^7/2
            ecc_factor_1 = (1 - (temp_bin_ecc_around_com**(2.0)))**(3.5)
            # and eccentricity factor [1+(73/24)e_b^2+(37/96)e_b^4]
            ecc_factor_2 = [1+((73/24)*(temp_bin_ecc_around_com**2.0))+((37/96)*(temp_bin_ecc_around_com**4.0))]
            #overall ecc factor = ecc_factor_1/ecc_factor_2
            ecc_factor = ecc_factor_1/ecc_factor_2
            # Binary period = 2pi*sqrt((delta_r)^3/GM_bin)
            # or T_orb = 10^7s*(1r_g/m_smmbh=10^8Msun)^(3/2) *(M_bin/10Msun)^(-1/2) = 0.32yrs
            temp_bin_period = 0.32*((temp_bin_separation)**(1.5))*((smbh_mass/1.e8)**(1.5))*(temp_bin_mass/10.0)**(-0.5)
            #Find how many binary orbits in timestep. Binary separation is halved for every 10^3 orbits.
            if temp_bin_period > 0:
                temp_num_orbits_in_timestep = timestep_duration_yr/temp_bin_period
            else:
                temp_num_orbits_in_timestep = 0

            scaled_num_orbits=temp_num_orbits_in_timestep/1000.0
            #Timescale for binary merger via GW emission alone, scaled to bin parameters
            temp_bin_t_gw = norm_tgw*((temp_bin_separation)**(4.0))*((temp_bin_mass/10.0)**(-2))*((temp_bin_reduced_mass/2.5)**(-1.0))*ecc_factor
            disk_bins_bhbh[10,j]=temp_bin_t_gw

            if temp_bin_t_gw > timestep_duration_yr:
                #Binary will not merge in this timestep.
                #Write new bin_separation according to Baruteau+11 prescription
                new_temp_bin_separation = temp_bin_separation*((0.5)**(scaled_num_orbits))
                disk_bins_bhbh[8,j] = new_temp_bin_separation
            else:
                #Binary will merge in this timestep.
                #Return a merger in bin_array! A negative flag on this line indicates merger.
                disk_bins_bhbh[11,j] = -2
                disk_bins_bhbh[12,j] = time_passed_yr

    return disk_bins_bhbh
