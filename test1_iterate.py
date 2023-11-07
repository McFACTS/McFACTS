from cgi import print_arguments
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

from inputs import ReadInputs

from setup import setupdiskblackholes
from physics.migration.type1 import type1
from physics.accretion.eddington import changebhmass
from physics.accretion.torque import changebh
#from physics.feedback import hankla22
#from physics.dynamics import wang22
from physics.binary.formation import hillsphere
from physics.binary.formation import add_new_binary
#from physics.binary.formation import secunda20
from physics.binary.evolve import evolve
from physics.binary.harden import baruteau11
from physics.binary.merge import tichy08
from physics.binary.merge import chieff
from physics.binary.merge import tgw
#from tests import tests
from outputs import mergerfile

def main():
    """
    """

    #1. Test a merger by calling modules
    print("Test merger using Tichy & Marionetti (2008)")
    
    # Pre-merger binary parameters
    mass_1 = 10.0
    mass_2 = 15.0
    spin_1 = 0.1
    spin_2 = 0.7
    angle_1 = 1.80
    angle_2 = 0.7
    bin_ang_mom = 1.0

    # Expected Tichy & Marionetti post-merger values
    expected_mass = 23.560384
    expected_spin = 0.8402299374639024
    expected_chi = 0.31214563487176167
    
    # Calculate post-merger quantities
    out_mass = tichy08.merged_mass(mass_1, mass_2, spin_1, spin_2)
    out_spin = tichy08.merged_spin(mass_1, mass_2, spin_1, spin_2, bin_ang_mom)
    out_chi = chieff.chi_effective(mass_1, mass_2, spin_1, spin_2, angle_1, angle_2, bin_ang_mom)
    calculated = [out_mass, out_spin, out_chi]
        
    # Test reference quantities for final mass, final spin, and chi
    reference = [expected_mass, expected_spin, expected_chi]
    # Did it pass?
    if calculated == reference:
        print(" Merger test passed!")
    else:
        print(" Merger test failed!")
        print("  Initial mass_1, mass_2 =", mass_1, mass_2)
        print("  Initial spin_1, spin_2 =", spin_1, spin_2)
        print("  Initial spin angle_1, angle_2 =", angle_1, angle_2)
        print("  Binary angular momentum =", bin_ang_mom)
        print("  Final mass =", out_mass)
        print("  Expected mass =", expected_mass)
        print("  Final spin =", out_spin)
        print("  Expected spin =", expected_spin)
        print("  Final chi =", out_chi)
        print("  Expected chi =", expected_chi)
    
    #Output should always be constant: 23.560384 0.8402299374639024 0.31214563487176167
    #test_merger=tests.test_merger()

    # Setting up automated input parameters
    # see ReadInputs.py for documentation of variable names/types/etc.

    mass_smbh, trap_radius, n_bh, mode_mbh_init, max_initial_bh_mass, \
         mbh_powerlaw_index, mu_spin_distribution, sigma_spin_distribution, \
             spin_torque_condition, frac_Eddington_ratio, max_initial_eccentricity, \
                 timestep, number_of_timesteps, disk_model_radius_array, disk_inner_radius,\
                     disk_outer_radius, surface_density_array, aspect_ratio_array, retro, \
                        = ReadInputs.ReadInputs_ini()
    
    # BARRY: WHAT FRESH HELL IS THIS? THERE ARE INTEGER NUMBER OF BH!
    # All of them should be in integer usage and will prob improve speed/memory usage
    # Is there some reason I should not change this for all of them forever?
    integer_nbh = int(n_bh)

    # set the number of times to run the model to generate population statistics
    number_of_iterations = 3
    
    # Containers to hold data from each iteration. `pop_` prefix refers the total population
    pop_bh_initial_masses = []
    pop_prograde_bh_locations = []
    pop_prograde_bh_masses = []
    pop_prograde_bh_spins = []
    pop_prograde_bh_spin_angles = []
    pop_merged_bh_array = []
    pop_binary_bh_array = []
    pop_number_of_mergers = []



    for iteration in range(number_of_iterations):

        print("Generate initial BH parameter arrays")
        bh_initial_locations = setupdiskblackholes.setup_disk_blackholes_location(n_bh, disk_outer_radius)
        bh_initial_masses = setupdiskblackholes.setup_disk_blackholes_masses(n_bh, mode_mbh_init, max_initial_bh_mass, mbh_powerlaw_index)
        bh_initial_spins = setupdiskblackholes.setup_disk_blackholes_spins(n_bh, mu_spin_distribution, sigma_spin_distribution)
        bh_initial_spin_angles = setupdiskblackholes.setup_disk_blackholes_spin_angles(n_bh, bh_initial_spins)
        bh_initial_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(n_bh)
        #bh_initial_generations = np.ones((integer_nbh,), dtype=int)
        bh_initial_generations = np.ones((integer_nbh,), dtype=int)

        #3.a Test migration of prograde BH
        #Disk surface density (assume constant for test)
        #BARRY: yeah we got fancy options now, let's use them???
        disk_surface_density = 1.e5
        #Housekeeping: Set up time
        initial_time = 0.0
        final_time = timestep * number_of_timesteps
        print("Migrate BH in disk")
        #Find prograde BH orbiters. Identify BH with orb. ang mom =+1
        bh_orb_ang_mom_indices = np.array(bh_initial_orb_ang_mom)
        prograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == 1)
        #retrograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == -1)
        prograde_bh_locations = bh_initial_locations[prograde_orb_ang_mom_indices]
        sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
        print("Sorted prograde BH locations:")
        print(sorted_prograde_bh_locations)

        #b. Test accretion onto prograde BH
        #Fractional rate of mass growth per year at the Eddington rate(2.3e-8/yr)
        # BARRY: we discussed this thing before, but I lost my notes is this a
        # housekeeping thing or an input choice?
        mass_growth_Edd_rate = 2.3e-8
        #Use masses of prograde BH only
        prograde_bh_masses = bh_initial_masses[prograde_orb_ang_mom_indices]
        print("Prograde BH initial masses")
        print(prograde_bh_masses)

        #c. Test spin change and spin angle torquing
        #Housekeeping: minimum spin angle resolution (ie less than this value gets fixed to zero) e.g 0.02 rad=1deg
        spin_minimum_resolution = 0.02
        #Torque prograde orbiting BH only
        print("Prograde BH initial spins")
        prograde_bh_spins = bh_initial_spins[prograde_orb_ang_mom_indices]
        print(prograde_bh_spins)
        print("Prograde BH initial spin angles")
        prograde_bh_spin_angles = bh_initial_spin_angles[prograde_orb_ang_mom_indices]
        print(prograde_bh_spin_angles)
        print("Prograde BH initial generations")
        prograde_bh_generations = bh_initial_generations[prograde_orb_ang_mom_indices]

        #4 Test Binary formation
        #Number of binary properties that we want to record (e.g. R1,R2,M1,M2,a1,a2,theta1,theta2,sep,com,t_gw,merger_flag,time of merger, gen_1,gen_2, bin_ang_mom)
        number_of_bin_properties = 17.0
        integer_nbinprop = int(number_of_bin_properties)
        bin_index = 0
        int_bin_index = int(bin_index)
        test_bin_number = 12.0
        integer_test_bin_number = int(test_bin_number)
        number_of_mergers = 0
        int_num_mergers = int(number_of_mergers)

        #Set up empty initial Binary array
        #Initially all zeros, then add binaries plus details as appropriate
        binary_bh_array = np.zeros((integer_nbinprop, integer_test_bin_number))
        #Set up normalization for t_gw
        norm_t_gw = tgw.normalize_tgw(mass_smbh)
        print("Scale of t_gw (yrs) =", norm_t_gw)
        
        # Set up merger array (identical to binary array)
        #number_of_merger_properties = 16.0
        num_of_mergers = 4.0
        #int_merg_props = int(number_of_merger_properties)
        #int_n_merg = int(num_of_mergers)
        merger_array = np.zeros((integer_nbinprop, integer_test_bin_number))

        #Set up output array (mergerfile)
        nprop_mergers = 14.0
        integer_nprop_merge = int(nprop_mergers)
        merged_bh_array = np.zeros((integer_nprop_merge, integer_test_bin_number))
        #Start Loop of Timesteps
        print("Start Loop!")
        time_passed = initial_time
        print("Initial Time(yrs) =", time_passed)
        while time_passed < final_time:
            #Migrate
            prograde_bh_locations = type1.dr_migration(prograde_bh_locations, prograde_bh_masses, disk_surface_density, timestep)
            #Accrete
            prograde_bh_masses = changebhmass.change_mass(prograde_bh_masses, frac_Eddington_ratio, mass_growth_Edd_rate, timestep)
            #Spin up    
            prograde_bh_spins = changebh.change_spin_magnitudes(prograde_bh_spins, frac_Eddington_ratio, spin_torque_condition, timestep)
            #Torque spin angle
            prograde_bh_spin_angles = changebh.change_spin_angles(prograde_bh_spin_angles, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep)
            #Calculate size of Hill sphere
            bh_hill_sphere = hillsphere.calculate_hill_sphere(prograde_bh_locations, prograde_bh_masses, mass_smbh)
            #Test for encounters within Hill sphere
            print("Time passed", time_passed)
            print("Number of binaries =", bin_index)
            #If binary exists, harden it. Add a thing here.
            if bin_index > 0:
                #Evolve binaries. 
                #Migrate binaries
                binary_bh_array = evolve.com_migration(binary_bh_array, disk_surface_density, timestep, integer_nbinprop, bin_index)
                #Accrete gas onto binaries
                binary_bh_array = evolve.change_bin_mass(binary_bh_array, frac_Eddington_ratio, mass_growth_Edd_rate, timestep, integer_nbinprop, bin_index)
                #Spin up binary components
                binary_bh_array = evolve.change_bin_spin_magnitudes(binary_bh_array, frac_Eddington_ratio, spin_torque_condition, timestep, integer_nbinprop, bin_index)
                #Torque binary spin components
                binary_bh_array = evolve.change_bin_spin_angles(binary_bh_array, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep, integer_nbinprop, bin_index)

                #Check and see if merger flagged (row 11, if negative)
                merger_flags = binary_bh_array[11,:]
                any_merger = np.count_nonzero(merger_flags) 
                print(merger_flags)
                merger_indices = np.where(merger_flags < 0.0)
                print(merger_indices)
                #print(binary_bh_array[:,merger_indices])
                if any_merger > 0:
                    print("Merger!")
                    #Calculate merger properties
                    mass_1 = binary_bh_array[2,merger_indices]
                    mass_2 = binary_bh_array[3,merger_indices]
                    spin_1 = binary_bh_array[4,merger_indices]
                    spin_2 = binary_bh_array[5,merger_indices]
                    angle_1 = binary_bh_array[6,merger_indices]
                    angle_2 = binary_bh_array[7,merger_indices]
                    bin_ang_mom = binary_bh_array[16,merger_indices]

                    merged_mass = tichy08.merged_mass(mass_1, mass_2, spin_1, spin_2)
                    merged_spin = tichy08.merged_spin(mass_1, mass_2, spin_1, spin_2, bin_ang_mom)
                    merged_chi_eff = chieff.chi_effective(mass_1, mass_2, spin_1, spin_2, angle_1, angle_2, bin_ang_mom)
                    merged_bh_array = mergerfile.merged_bh(merged_bh_array, binary_bh_array, merger_indices, merged_chi_eff, merged_mass, merged_spin, nprop_mergers, number_of_mergers)
                    
                    merger_array[:,merger_indices] = binary_bh_array[:,merger_indices]
                    #print(merger_array)
                    #Reset merger marker to zero
                    int_n_merge=int(number_of_mergers)
                    #Remove merged binary from binary array
                    binary_bh_array[:,merger_indices] = 0.0
                    binary_bh_array[11,int_n_merge] = 0
                    
                    #Reduce by 1 the number of binaries
                    bin_index = bin_index - 1
                    
                    #Find relevant properties of merged BH to add to single BH arrays
                    merged_bh_com = merged_bh_array[0,int_n_merge]
                    merged_mass = merged_bh_array[1,int_n_merge]
                    merged_spin = merged_bh_array[3,int_n_merge]
                    merged_spin_angle = merged_bh_array[4,int_n_merge]
                    #New bh generation is max of generations involved in merger plus 1
                    merged_bh_gen = np.maximum(merged_bh_array[11,int_n_merge], merged_bh_array[12,int_n_merge]) + 1.0 
                    print("Merger at =", merged_bh_com, merged_mass, merged_spin, merged_spin_angle, merged_bh_gen)
                    # Add to number of mergers
                    number_of_mergers = number_of_mergers + 1
                    # Append new merged BH to arrays of single BH locations, masses, spins, spin angles & gens
                    prograde_bh_locations = np.append(prograde_bh_locations, merged_bh_com)
                    prograde_bh_masses = np.append(prograde_bh_masses, merged_mass)
                    prograde_bh_spins = np.append(prograde_bh_spins, merged_spin)
                    prograde_bh_spin_angles = np.append(prograde_bh_spin_angles, merged_spin_angle)
                    prograde_bh_generations = np.append(prograde_bh_generations, merged_bh_gen)
                    sorted_prograde_bh_locations=np.sort(prograde_bh_locations)
                    print("New BH locations", sorted_prograde_bh_locations)
                    print("Merger Flag!")
                    print(number_of_mergers)
                    print("Time ", time_passed)
                    print(merger_array)
                else:                
                    # No merger
                    # Harden binary
                    binary_bh_array = baruteau11.bin_harden_baruteau(binary_bh_array, integer_nbinprop, mass_smbh, timestep, norm_t_gw, bin_index, time_passed)
                    print("Harden binary")
                    print("Time passed = ", time_passed)
                    print(binary_bh_array)
            else:
                
                # No Binaries present in bin_array. Nothing to do.
            


                #If a close encounter within mutual Hill sphere add a new Binary

                close_encounters = hillsphere.encounter_test(prograde_bh_locations, bh_hill_sphere)
                print(close_encounters)
                if len(close_encounters) > 0:
                    print("Make binary at time ", time_passed)
                    sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
                    sorted_prograde_bh_location_indices = np.argsort(prograde_bh_locations)
                    number_of_new_bins = (len(close_encounters)) // 2
                    binary_bh_array = add_new_binary.add_to_binary_array(binary_bh_array, prograde_bh_locations, prograde_bh_masses, prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_generations, close_encounters, bin_index, retro)
                    bin_index = bin_index + number_of_new_bins
                    bh_masses_by_sorted_location = prograde_bh_masses[sorted_prograde_bh_location_indices]
                    bh_spins_by_sorted_location = prograde_bh_spins[sorted_prograde_bh_location_indices]
                    bh_spin_angles_by_sorted_location = prograde_bh_spin_angles[sorted_prograde_bh_location_indices]
                    #Delete binary info from individual BH arrays
                    sorted_prograde_bh_locations = np.delete(sorted_prograde_bh_locations, close_encounters)
                    bh_masses_by_sorted_location = np.delete(bh_masses_by_sorted_location, close_encounters)
                    bh_spins_by_sorted_location = np.delete(bh_spins_by_sorted_location, close_encounters)
                    bh_spin_angles_by_sorted_location = np.delete(bh_spin_angles_by_sorted_location, close_encounters)
                    #Reset arrays
                    prograde_bh_locations = sorted_prograde_bh_locations
                    prograde_bh_masses = bh_masses_by_sorted_location
                    prograde_bh_spins = bh_spins_by_sorted_location
                    prograde_bh_spin_angles = bh_spin_angles_by_sorted_location

            #Iterate the time step
            #Empty close encounters
            empty = []
            close_encounters = np.array(empty)
            time_passed = time_passed + timestep
        #End Loop of Timesteps at Final Time, end all changes & print out results
        
        print("End Loop!")
        print("Final Time (yrs) = ", time_passed)
        print("BH locations at Final Time")
        print(prograde_bh_locations)
        print("Number of binaries = ", bin_index)
        print("Total number of mergers = ", number_of_mergers)
        print("Mergers")
        print(merged_bh_array)

        print('number of binaries:', bin_index)
        print('component masses')
        print(binary_bh_array[2:4,:bin_index])

        pop_bh_initial_masses.append(bh_initial_masses)
        pop_prograde_bh_locations.append(prograde_bh_locations)
        pop_prograde_bh_masses.append(prograde_bh_masses)
        pop_prograde_bh_spins.append(prograde_bh_spins)
        pop_prograde_bh_spin_angles.append(prograde_bh_spin_angles)
        pop_merged_bh_array.append(merged_bh_array)
        pop_binary_bh_array.append(binary_bh_array)
        pop_number_of_mergers.append(number_of_mergers)


    # Plot intial and final mass distributions
    numbins = 100
    plt.hist(bh_initial_masses, bins=numbins, align='left', label='Initial', color='grey', alpha=0.5)
    plt.hist(bh_masses_by_sorted_location, bins=numbins, align='left', label='Final', color='purple', alpha=0.5)
    plt.hist(binary_bh_array[2:4,:bin_index], bins=numbins, align='left', label='Final', color=['purple'], alpha=0.5)
    plt.title(f'Black Hole Mass Evolution Over {final_time:.1e} years\n'+
              f'Number of Mergers: {number_of_mergers}')
    plt.ylabel('Number')
    plt.xlabel(r'Mass ($M_\odot$)')
    plt.legend()
    plt.savefig("./mass_evolution.png", format='png')
    plt.close()
    

    print(len(pop_prograde_bh_locations))
    print(len(pop_bh_initial_masses))
    print(len(pop_prograde_bh_locations))
    print(len(pop_prograde_bh_masses))
    print(len(pop_prograde_bh_spins))
    print(len(pop_prograde_bh_spin_angles))
    print(len(pop_merged_bh_array))
    print(len(pop_binary_bh_array))
    print(len(pop_number_of_mergers))
    
    # Plot Inital and Final Positions as function of mass
    # plt.scatter(sorted_prograde_bh_locations, bh_masses_by_sorted_location)
    # plt.ylabel('Mass')
    # plt.xlabel('Radius')
    # plt.legend(title=r'$M_{\rm SMBH}$ ='+f'{mass_smbh:.0e}'+r' $M_\odot$', frameon=False)
    # plt.show()
    


if __name__ == "__main__":
    main()
