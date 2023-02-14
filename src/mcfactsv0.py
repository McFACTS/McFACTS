from cgi import print_arguments
import numpy as np
import matplotlib.pyplot as plt
# McFACTS v0.0 9/15/22 BMcK
# What does this code do?
# a.Set up a simple 1-d AGN disk model
# b.Insert stellar-origin BH at random locations in the disk
# c.Make their orbits prograde or retrograde. Make a note of which is which,
# since for now, only prograde orbiters migrate, accrete/torque into alignment
# d.Pick initial BH masses from a powerlaw distribution
# e.Pick initial BH spins from a narrow Gaussian distribution centered on zero. 
# Make a note of which spins are negative since initial spin angles 
# depend on sign of spin. Positive spin means spin angle [0,1.57]rad.
# Negative spin means spin angle [1.571,3.14]rad.
# f.Pick initial BH spin angle from a uniform random distribution:
#  [0,1.57]rad if positive spin, [1.571,3.14]rad if negative spin
# g. Iterate time in units of timestep
#  Per iteration, do the following to the prograde orbiting BH 
# move them inwards in the disk according to a Type I migration prescription
# Accrete mass according to a fractional Eddington prescription
# Torque spin angle and change spin according to gas mass accreted
# At end time print out final locations/masses etc. 
# Lots of comments throughout. Much of commentary can end up in a README for 
# the various definitions/functions etc. For now it's all here.


#PART 1 Inputs for Initial Conditions & Disk
#I'm putting in test initial values just to get this going. 
# TO DO: These should all go in an input file to be read in. 

# Read in inputs

# create a dictionary to store numerical variables in
input_variables = {}

# open the main input file for reading
model_inputs = open("model_choice.txt", 'r')
# go through the file line by line
for line in model_inputs:
    line = line.strip()
    # If it is NOT a comment line
    if (line.startswith('#') == 0):
        # split the line between the variable name and its value
        varname, varvalue = line.split("=")
        # remove whitespace
        varname = varname.strip()
        varvalue = varvalue.strip()
        # if the variable is the one that's a filename for the disk model, deal with it
        if (varname == 'disk_model_name'):
            disk_model_name = varvalue.strip("'")
        # otherwise, typecast to float and stick it in the dictionary
        else:
            input_variables[varname] = float(varvalue)
 
# close the file
model_inputs.close()

# Recast the inputs from the dictionary lookup to actual variable names
#   !!!is there a better (automated?) way to do this?
n_bh = input_variables['n_bh']
mode_mbh_init = input_variables['mode_mbh_init']
mbh_powerlaw_index = input_variables['mbh_powerlaw_index']
mu_spin_distribution = input_variables['mu_spin_distribution']
sigma_spin_distribution = input_variables['sigma_spin_distribution']
spin_torque_condition = input_variables['spin_torque_condition']
frac_Eddington_ratio = input_variables['frac_Eddington_ratio']
max_initial_eccentricity = input_variables['max_initial_eccentricity']
timestep = input_variables['timestep']
number_of_timesteps = input_variables['number_of_timesteps']

# open the disk model surface density file and read it in
# Note format is assumed to be comments with #
#   density in SI in first column
#   radius in r_g in second column
#   infile = model_surface_density.txt, where model is user choice
infile_suffix = '_surface_density.txt'
infile = disk_model_name+infile_suffix
surface_density_file = open(infile, 'r')
density_list = []
radius_list = []
for line in surface_density_file:
    line = line.strip()
    # If it is NOT a comment line
    if (line.startswith('#') == 0):
        columns = line.split()
        density_list.append(float(columns[0]))
        radius_list.append(float(columns[1]))
# close file
surface_density_file.close()

# re-cast from lists to arrays
surface_density_array = np.array(density_list)
disk_model_radius_array = np.array(radius_list)

# open the disk model aspect ratio file and read it in
# Note format is assumed to be comments with #
#   aspect ratio in first column
#   radius in r_g in second column must be identical to surface density file
#       (radius is actually ignored in this file!)
#   filename = model_aspect_ratio.txt, where model is user choice
infile_suffix = '_aspect_ratio.txt'
infile = disk_model_name+infile_suffix
aspect_ratio_file = open(infile, 'r')
aspect_ratio_list = []
for line in aspect_ratio_file:
    line = line.strip()
    # If it is NOT a comment line
    if (line.startswith('#') == 0):
        columns = line.split()
        aspect_ratio_list.append(float(columns[0]))
# close file
aspect_ratio_file.close()

# re-cast from lists to arrays
aspect_ratio_array = np.array(aspect_ratio_list)

# Housekeeping from input variables
disk_outer_radius = disk_model_radius_array[-1]
disk_inner_radius = disk_model_radius_array[0]
# these are bogus--right now just assuming constant so pull the first value
# !!! fix later
disk_aspect_ratio = aspect_ratio_array[0]
disk_surface_density = surface_density_array[0]

# Housekeeping initialization stuff -- do not change!

# What is the spin angle indicating alignment with AGN disk (should be zero rad)
aligned_spin_angle_radians=0.0
# Spin angle indicating anti-alignment with AGN disk (should be pi radians=180deg)
antialigned_spin_angle_radians=3.14
# minimum spin angle resolution (ie less than this value gets fixed to zero) e.g 0.02 rad=1deg
spin_minimum_resolution=0.02
# Fractional rate of mass growth per year at the Eddington rate(2.3e-8/yr)
mass_growth_Edd_rate=2.3e-8

# Binary properties
# Number of binary properties that we want to record (e.g. M_1,_2,a_1,2,theta_1,2,a_bin,a_com,t_gw,ecc,bin_ang_mom,generation)
number_of_binary_properties=13.0

# Start time (in years)
initial_time=0.

#PART 3 Set up Initial Black hole population
# TO DO: put all this in an initializing FUNCTION call.
# Or multiple initializing FUNCTION (def) calls. 
# So that user can change these, or other options included.
# E.g. initialize_locations (random uniform, random other distribution etc), 
# or E.g. initialize_masses (options for complicated powerlaw), etc.
#Now set up random locations,masses,spin mags,spin angles, orb. ang. mom
#E.g. Test draw 100 random radial locations from disk of size 1.e5r_g
bh_initial_locations=disk_outer_radius*np.random.random_sample((n_bh,))
#print(bh_initial_locations)
#draw masses from a powerlaw (pareto) distribution, say a=2,mode(mass)=10
bh_initial_masses=(np.random.pareto(mbh_powerlaw_index,n_bh)+1)*mode_mbh_init
#print(bh_initial_masses)
#draw spins from a narrow Gaussian with mean 0 and sigma=0.1 say
#make a note of the negative spins, since spin angle must be 
# [0,1.57]rads for positive spin and[1.571,3.14]rads for negative spin
bh_initial_spins=np.random.normal(mu_spin_distribution,sigma_spin_distribution,100)
bh_initial_spin_indices=np.array(bh_initial_spins)
negative_spin_indices=np.where(bh_initial_spin_indices < 0.)
#print(bh_initial_spins)
#draw spin angles randomly from [0,3.14]rads where 0rads 
# is fully aligned with AGN disk and 3.14 is fully anti-aligned
# Positive spins have spin angles [0,1.57]rads 
# Negative spins have spin angles [1.57,3.14]rads so 
#give all BH spin angles [0,1.57]rads and add 1.57rads to those with negative spin
bh_initial_spin_angles=np.random.uniform(0.,1.57,n_bh)
bh_initial_spin_angles[negative_spin_indices]=bh_initial_spin_angles[negative_spin_indices]+1.57
#print(bh_initial_spin_angle)
#draw orbital angular momentum as either -1(retrograde, against disk) or +1 (prograde, with disk)
#first draw a random no. from a uniform distribution [0,1] then round to nearest integer ie 0 or 1. Multiply by 2-> 0 or 2 and subtract 1 so -1 or +1 
random_uniform_number=np.random.random_sample((n_bh,))
bh_initial_orb_ang_mom=(2.0*np.around(random_uniform_number))-1.0
print(bh_initial_orb_ang_mom)
#Find the prograde BH in this array so can accrete/torque only these and not on the retrogrades (for now)
bh_orb_ang_mom_indices=np.array(bh_initial_orb_ang_mom)
prograde_orb_ang_mom_indices=np.where(bh_orb_ang_mom_indices == 1)
#print(prograde_orb_ang_mom_indices)


#PART 4: Set up Run (duration etc)
#TO DO: PUT this in in a length of run (or TOTAL TIME) for run function call
final_time=timestep*number_of_timesteps

#PART 5: DEFINITIONS/FUNCTIONS LIVE HERE

# 1. Define Migration Prescription
#Calculate migration timescale & update locations
#Assume inward migration at all disk locations: LATER put in out-migration interior to trap
#t_migration approx 38Myr (M_bh/5Msun)^-1 (Sigma/10^5kg/m^2)^-1(R/10^4r_g)^-1/2((h/r)/0.02)^2
# ie a 5Msun BH at 10^4r_g takes 38Myr to migrate to R=0 assuming these values don't change.
#So e.g. 10^4r_g/38Myr=10r_g/38kyr or 1r_g/3.8kyr. 
#So, in a timestep of 10kyr, assuming uniform values across *ENTIRE* disk for simplicity, 
# everything moves by 2.6r_g/10kyr 
#Set up function for change in radius of each BH according to a migration prescription.
 
def dr_migration(bh_locations,prograde_orb_ang_mom_indices,bh_masses,disk_surface_density,timestep):
    #TO DO: Add feedback prescription from Hankla et al. 2020 or include as modifying function
    #sg_norm is a normalization factor for the Sirko & Goodman (2003) disk model
    #38Myrs=3.8e7yrs is the time for a 5Msun BH to undergo Type I migration to 
    #the SMBH from 10^4r_g in that model.
    sg_norm=3.8e7
    #scaled_aspect=disk_aspect ratio scaled to 0.02 as a fiducial value. 
    #Everything will be scaled around that
    #scaled mass= BH mass/lower bound mass (e.g. 5Msun, upper end of lower mass gap)
    #scaled location= BH location scaled to 10^4r_g
    #scaled sigma= Disk surface density scaled to 10^5kg/m^2
    # Advantage of scaling like this is this is a useful fiducial for generic 
    # testing, but details will change with detailed disk model 
    # (ie location of BH)  
    scaled_mass=5.0
    scaled_aspect=0.02
    scaled_location=1.e4
    scaled_sigma=1.e5
    #Normalize the locations and BH masses for now to these scales 
    # for ease of computation
    normalized_locations=bh_locations/scaled_location
    normalized_masses=bh_masses/scaled_mass
    normalized_locations_sqrt=np.sqrt(normalized_locations)
    #Can normalize the aspect ratio and sigma to these scales when we 
    # implement the 1d disk model (interpolate over SG03)
    normalized_sigma=disk_surface_density/scaled_sigma
    normalized_aspect_ratio=disk_aspect_ratio/scaled_aspect
    normalized_aspect_ratio_squared=np.square(normalized_aspect_ratio)
    #So our fiducial timescale should now be 38Myrs as calcd below
    dt_mig=sg_norm*(normalized_aspect_ratio_squared)/((normalized_masses)*(normalized_locations_sqrt)*(normalized_sigma))
    #Effective fractional time of migration is timestep/dt_mig
    fractional_migration_timestep=timestep/dt_mig
    #Migration distance is then location of BH * fractional_migration_timestep
    migration_distance=bh_locations*fractional_migration_timestep
    #So new updated locations is bh_locations minus migration distance 
    #if migration is always inwards
    #BUT only do this for prograde BH.
    bh_new_locations=bh_locations
    bh_new_locations[prograde_orb_ang_mom_indices]=bh_locations[prograde_orb_ang_mom_indices]-migration_distance[prograde_orb_ang_mom_indices]
#    bh_new_locations=bh_locations-migration_distance
#return new BH locations after prograde orbiters have migrated over timestep
    return bh_new_locations

#2. Define mass change due to BH accretion modification per timestep
# Only prograde orbiters accrete. 
# TO DO: Accretion prescription for retrograde orbiters.
#At Eddington rate of accretion mass changes by (frac_Edd/1)*2.3e-8/yr
#where frac_Edd=1 is Eddington accretion rate, frac_Edd=2 is 2xEdd etc. 
#Mass_new=Mass_old*exp(2.3e-8*frac_Edd*time)
#E.g. if time=4e7yr=40Myr, 
# Mass_new=Mass_old*exp(2.3e-8*1*4.e7)=Mass_old*exp(0.92)=2.5Mass_old
def change_mass(bh_masses,prograde_orb_ang_mom_indices,frac_Eddington_ratio,mass_growth_Edd_rate,timestep):
    #bh_new_mass=bh_old_mass*exp(mass_growth_Edd_rate*frac_Edd_rate*timestep)
    #BUT only do this for the BH that are prograde (ie orb ang mom is +1). 
    # leave the retrograde BH alone (for now)
    bh_new_masses=bh_masses
    bh_new_masses[prograde_orb_ang_mom_indices]=bh_masses[prograde_orb_ang_mom_indices]*np.exp(mass_growth_Edd_rate*frac_Eddington_ratio*timestep)
#Return new updated masses after prograde orbiters have accreted over timestep
    return bh_new_masses

#4. Define BH spin magnitude change per timestep. Change the magnitude of spin
#Only prograde orbiters accrete mass per timestep
#TO DO: Spin change prescription for retrograde orbiters.
# Assume accretion at some fractional rate of Eddington.
# Note Bardeen (1970) shows that spin a=-1 changes to a=0 when the mass goes to
# sqrt(3/2)M_0=1.22M_0. 
# Assumption from Bogdanovic et al. 2007, you need 1-10% mass accretion 
# to torque a BH into alignment with a disk
#Since M(t)=M0*exp(2.3e-8*frac_Edd*time) and 
# exp(0.1)=1.11 and exp(0.01)=1.01,
# this implies if frac_Edd=1, time= 4.5e6yr (4.5e5yr)=4.5(0.5)Myr for alignment 
# at 10%(1%)mass accretion.
#So, if we assume a=-1 goes to a=+1 in 4.5Myr at 10%mass accretion at Eddington rate,
#this is a average rate of change of magnitude of 
# delta_a=+2/4.5Myr= 0.44/Myr=4.4e-4/kyr 
# or 4.4e-3/(frac_Eddington_ratio/1.0)(spin_torque_condition/0.1)(timestep/10kyr)
def change_spin_magnitudes(bh_spins,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,mass_growth_Edd_rate,timestep):
    bh_new_spins=bh_spins
    normalized_Eddington_ratio=frac_Eddington_ratio/1.0
    normalized_timestep=timestep/1.e4
    normalized_spin_torque_condition=spin_torque_condition/0.1
    bh_new_spins[prograde_orb_ang_mom_indices]=bh_new_spins[prograde_orb_ang_mom_indices]+(4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
# TO DO: Include a condition to keep a maximal (a=+0.98) spin BH at that value once it reaches it
#Return updated new spins    
    return bh_new_spins


#4. Define BH spin torque modification per timestep. Change the angle of spin.
#Only prograde orbiters are torqued (via accretion). 
# TO DO: Spin torque prescription for retrograde orbiters.
#Assume accretion happens in midplane of disk and 
#torques BH gradually towards alignment with disk angular momentum
# Note Bardeen (1970) shows that spin a=-1 changes to a=0 when the mass goes to
# sqrt(3/2)M_0=1.22M_0. 
#Assumption from Bogdanovic et al. 2007, you need 1-10% mass accretion 
# to torque a BH into alignment with a disk
#Since M(t)=M0*exp(2.3e-8*frac_Edd*time) and 
# exp(0.1)=1.11 and exp(0.01)=1.01,
# this implies if frac_Edd=1, time= 4.5e6yr (4.5e5yr) for alignment 
# at 10%(1%)mass accretion
# So if torque_alignment_condition=0.1(0.01) after 4.5e6(4.5e5)yr we expect
#spin angle of 3.14rad (180deg oriented to AGN disk orbital angular momentum)
# so in 4.5Myr, a BH at 3.14rad -> 0 rad, 
# or 3.14rad/(spin_torque_condition/0.1)4.5Myr=0.698rad/Myr=6.98e-4rad/kyr
# =6.98e-3 rad/(spin_torque_condition/0.1)(timestep/10kyr)
# TO DO: dynamics will be important as a randomizer *against* spin 
# alignment from orb. ang. momentum conservation
def change_spin_angles(bh_spin_angles,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,timestep):
    bh_new_spin_angles=bh_spin_angles
    normalized_Eddington_ratio=frac_Eddington_ratio/1.0
    normalized_timestep=timestep/1.e4
    normalized_spin_torque_condition=spin_torque_condition/0.1
    bh_new_spin_angles[prograde_orb_ang_mom_indices]=bh_new_spin_angles[prograde_orb_ang_mom_indices]-(6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    #TO DO: Include a condition to keep spin angle at or close to zero once it gets there
    #Return new spin angles
    return bh_new_spin_angles

#MAIN PROGRAM STARTS HERE. ALL FUNCTION CALLS ETC LIVE HERE
def main():
    #PRINT OUT SOME OF THE INITIAL QUANTITIES TO SEE CHANGES
    print("Initial BH Locations")
    print(bh_initial_locations)
    bh_locations=bh_initial_locations
    print("Initial BH masses")
    print(bh_initial_masses)
    bh_masses=bh_initial_masses
    print("Initial BH spin magnitudes")
    print(bh_initial_spins)
    bh_spins=bh_initial_spins
    print("Initial BH spin angles")
    print(bh_initial_spin_angles)
    bh_spin_angles=bh_initial_spin_angles

    #LOOP over TIME PASSED until FINAL TIME
    print("Start Loop!")
    time_passed=initial_time
    print("Initial Time(yrs)=",time_passed)
    while time_passed<final_time:
        bh_locations=dr_migration(bh_locations,prograde_orb_ang_mom_indices,bh_masses,disk_surface_density,timestep)
        bh_masses=change_mass(bh_masses,prograde_orb_ang_mom_indices,frac_Eddington_ratio,mass_growth_Edd_rate,timestep)
        bh_spins=change_spin_magnitudes(bh_spins,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,mass_growth_Edd_rate,timestep)
        bh_spin_angles=change_spin_angles(bh_spin_angles,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,timestep)
        #Iterate the time step
        time_passed=time_passed+timestep
        #Test looped output; show new locations, new masses etc.
        #if time_passed >=1.e4 and time_passed <1.2e4:
        #    print("Time is >=10kyr and <12kyr")
        #    print(dr_migration(bh_locations,prograde_orb_ang_mom_indices,bh_masses,disk_surface_density,timestep))
        #    print(change_mass(bh_masses,prograde_orb_ang_mom_indices,frac_Eddington_ratio,mass_growth_Edd_rate,timestep)) 
    final_bh_locations=bh_locations
    final_bh_masses=bh_masses
    final_bh_spins=bh_spins
    final_bh_spin_angles=bh_spin_angles
    print("End Loop!")
    print("Final Time(yrs)=",time_passed)
    print("BH locations at Final Time")
    print(final_bh_locations)
    print("Final BH masses")
    print(final_bh_masses)
    print("Final spins")
    print(final_bh_spins)
    print("Final BH spin angles")
    print(final_bh_spin_angles)


if __name__ == "__main__":
    main()
