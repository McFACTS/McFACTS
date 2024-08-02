import numpy as np
import scipy


def change_bin_mass(bin_array, frac_Eddington_ratio, mass_growth_Edd_rate, timestep, integer_nbinprop, bin_index):
    """_summary_

    Parameters
    ----------
    bin_array : _type_
        _description_
    frac_Eddington_ratio : _type_
        _description_
    mass_growth_Edd_rate : _type_
        _description_
    timestep : _type_
        _description_
    integer_nbinprop : _type_
        _description_
    bin_index : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    #Return new updated mass array due to accretion for prograde orbiting BH after timestep
    #Extract the binary locations and masses
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    for j in range(0, bindex): 
            if bin_array[11,j] < 0:
                #do nothing -merger happened!
                pass
            else:            
                #for i in range(0, integer_nbinprop):
                temp_bh_mass_1 = bin_array[2,j] 
                temp_bh_mass_2 = bin_array[3,j]
                mass_growth_factor = np.exp(mass_growth_Edd_rate*frac_Eddington_ratio*timestep)
                new_bh_mass_1 = temp_bh_mass_1*mass_growth_factor
                new_bh_mass_2 = temp_bh_mass_2*mass_growth_factor
                #Update new bh masses in bin_array
                bin_array[2,j] = new_bh_mass_1
                bin_array[3,j] = new_bh_mass_2

    return bin_array


def change_bin_spin_magnitudes(bin_array, frac_Eddington_ratio, spin_torque_condition, timestep, integer_nbinprop, bin_index):
    #def change_spin_magnitudes(bh_spins,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,mass_growth_Edd_rate,timestep):
    #bh_new_spins=bh_spins
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1
    #Extract the binary locations and spin magnitudes
    #max allowed spin
    max_allowed_spin=0.98
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    for j in range(0, bindex):
            if bin_array[11,j] < 0:
                #do nothing -merger happened!
                pass
            else:
                #for i in range(0, integer_nbinprop):
                temp_bh_spin_1 = bin_array[4,j] 
                temp_bh_spin_2 = bin_array[5,j]
                spin_change_factor = 4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep
                #print("Spin change factor", spin_change_factor)
                new_bh_spin_1 = temp_bh_spin_1 + spin_change_factor
                #print("Old spin1, new spin1 =",temp_bh_spin_1, new_bh_spin_1)
                new_bh_spin_2 = temp_bh_spin_2 + spin_change_factor
                if new_bh_spin_1 > max_allowed_spin:
                    new_bh_spin_1 = max_allowed_spin
                if new_bh_spin_2 > max_allowed_spin:
                    new_bh_spin_2 = max_allowed_spin
                #Update new bh masses in bin_array
                bin_array[4,j] = new_bh_spin_1
                bin_array[5,j] = new_bh_spin_2
                #print("EVOLVE SPINS",temp_bh_spin_1,temp_bh_spin_2,new_bh_spin_1,new_bh_spin_2)
    return bin_array


def change_bin_spin_angles(bin_array, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep, integer_nbinprop, bin_index):
    #Calculate change in spin angle due to accretion during timestep
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1

    #Extract the binary locations and spin magnitudes
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    for j in range(0, bindex):
            if bin_array[11,j] < 0:
                #do nothing -merger happened!
                pass
            else:
            #for i in range(0, integer_nbinprop):
                temp_bh_spin_angle_1 = bin_array[6,j] 
                temp_bh_spin_angle_2 = bin_array[7,j]
                #bh_new_spin_angles[prograde_orb_ang_mom_indices]=bh_new_spin_angles[prograde_orb_ang_mom_indices]-(6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
                spin_angle_change_factor = (6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
                new_bh_spin_angle_1 = temp_bh_spin_angle_1 - spin_angle_change_factor
                new_bh_spin_angle_2 = temp_bh_spin_angle_2 - spin_angle_change_factor
                if new_bh_spin_angle_1 < spin_minimum_resolution:
                    new_bh_spin_angle_1 = 0.0
                if new_bh_spin_angle_2 < spin_minimum_resolution:
                    new_bh_spin_angle_2 = 0.0
                bin_array[6,j] = new_bh_spin_angle_1
                bin_array[7,j] = new_bh_spin_angle_2
                #print("SPIN ANGLE EVOLVES, old1,old2, new1,new2",temp_bh_spin_angle_1,temp_bh_spin_angle_2,new_bh_spin_angle_1,new_bh_spin_angle_2)
    return bin_array

def com_feedback_hankla(bin_array, disk_surf_model, frac_Eddington_ratio, alpha):
    """_summary_
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
    alpha = disk viscosity parameter (e.g. alpha = 0.01 in Sirko & Goodman 2003)
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

    TO DO: Need alpha as an input for disk model (alpha=0.01 is SG03 default)
    TO (MAYBE) DO: kappa default as an input? Or kappa table? Or kappa user set?
    
    Parameters
    ----------
    
    bin_array : float array
        binary array. Row 9 is center of mass of binary BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated

    Returns
    -------
    ratio_feedback_to_mig : float array
        ratio of feedback torque to migration torque for each entry in prograde_bh_locations
    """
    #Extract the binary locations and masses
    #bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    

    temp_bin_com_locations = bin_array[9,:]
    
    #print("Bin com locations",temp_bin_com_locations)

    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_model, float):
        disk_surface_density = disk_surf_model
    else:
        disk_surface_density = disk_surf_model(temp_bin_com_locations)

    #Calculate ratio
    #
    #Define kappa (or set up a function to call). 
    #kappa = 10^0.76 cm^2/g = 10^(0.76) (10^-2m)^2/10^-3kg=10^(0.76-1)=10^(-0.24) m^2/kg to match units of Sigma
    kappa = 10**(-0.24)
    #Define alpha parameter for disk in Readinputs.py
    #alpha = 0.01

    Ratio_feedback_migration_torque_bin_com = 0.07 *(1/kappa)* ((alpha)**(-1.5))*frac_Eddington_ratio*np.sqrt(temp_bin_com_locations)/disk_surface_density

    #print("ratio", Ratio_feedback_migration_torque_bin_com)
    #print((1/kappa),((alpha)**(-1.5)),frac_Eddington_ratio)
    #print("Ratio", Ratio_feedback_migration_torque) 
    #print("BH locations", prograde_bh_locations) 

    return Ratio_feedback_migration_torque_bin_com  


def com_migration(bin_array, disk_surf_model, disk_aspect_ratio_model, timestep, integer_nbinprop, bin_index):
    """_summary_

    Parameters
    ----------
    bin_array : _type_
        _description_
    disk_surf_model : _type_
        _description_
    disk_aspect_ratio_model : _type_
        _description_
    timestep : _type_
        _description_
    integer_nbinprop : _type_
        _description_
    bin_index : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # !!!This should be re-written to do computations NOT in the loop 
    # (especially the surface density & aspect ratio)
    # Saavik has some worries about the functional forms working correctly
    # because it didn't actually fail when I didn't update them (I did singleton
    # migration first, tested, then did binary evolution, but it didn't fail
    # the first time... so that's a bit worrisome from a physics perspective)
    #Return updated locations of binary center of mass (com) and location 1,2 
    # based on Type 1 migration prescription
    #sg_norm is a normalization factor for the Sirko & Goodman (2003) disk model
    #38Myrs=3.8e7yrs is the time for a 5Msun BH to undergo Type I migration to
    #the SMBH from 10^4r_g in that model.
    sg_norm = 3.8e7
    #scaled mass= BH mass/lower bound mass (e.g. 5Msun, upper end of lower mass gap)
    scaled_mass = 5.0
    #scaled_aspect=disk_aspect ratio scaled to 0.02 as a fiducial value.
    scaled_aspect = 0.02
    #for test fixed disk aspect ratio
    #disk_aspect_ratio = 0.03
    #scaled location= BH location scaled to 10^4r_g
    scaled_location = 1.e4
    #scaled sigma= Disk surface density scaled to 10^5kg/m^2
    scaled_sigma = 1.e5
    
    #Extract the binary locations and masses
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    #If binary has been removed, it's all zeroes. So find first non-zero binary in array.
    #loc1_bins=bin_array[:,0]
    #live_loc1_bins=np.count_nonzero(loc1_bins) 
    #if live_loc1_bins > 0:
    #    bin_indices = np.where(live_loc1_bins > 0.0)

    for j in range(0,bindex):
            if bin_array[11,j] < 0:
                #do nothing -merger happened!
                pass
            else:
            #for i in range(0, integer_nbinprop):
                temp_bh_loc_1 = bin_array[0,j]
                temp_bh_loc_2 = bin_array[1,j]
                temp_bh_mass_1 = bin_array[2,j] 
                temp_bh_mass_2 = bin_array[3,j]
                temp_bin_sep = bin_array[8,j]
                temp_bin_com = bin_array[9,j]
                bin_mass = temp_bh_mass_1 + temp_bh_mass_2
                #Normalize the com location and BH masses 
                if isinstance(disk_surf_model, float):
                    disk_surface_density = disk_surf_model
                else:
                    disk_surface_density = disk_surf_model(temp_bin_com)
                if isinstance(disk_aspect_ratio_model, float):
                    disk_aspect_ratio = disk_aspect_ratio_model
                else:
                    disk_aspect_ratio = disk_aspect_ratio_model(temp_bin_com)
                normalized_com = temp_bin_com/scaled_location
                normalized_bin_mass = bin_mass/scaled_mass
                normalized_com_sqrt = np.sqrt(normalized_com)
                # Can normalize the aspect ratio and sigma to these scales when we
                # implement the 1d disk model (interpolate over SG03)
                normalized_sigma = disk_surface_density/scaled_sigma
                normalized_aspect_ratio = disk_aspect_ratio/scaled_aspect
                normalized_aspect_ratio_squared = np.square(normalized_aspect_ratio)
                #So our fiducial timescale should now be 38Myrs as calcd below
                dt_mig = sg_norm*(normalized_aspect_ratio_squared)/((normalized_bin_mass)*(normalized_com_sqrt)*(normalized_sigma))
                #Effective fractional time of migration is timestep/dt_mig
                fractional_migration_timestep = timestep/dt_mig
                #Migration distance is location of bin com * fractional_migration_timestep
                migration_distance = temp_bin_com*fractional_migration_timestep
                new_bin_com = temp_bin_com - migration_distance
                new_bh_loc1 = new_bin_com -(temp_bin_sep*temp_bh_mass_2/bin_mass)
                new_bh_loc2 = new_bh_loc1 + temp_bin_sep
                #Write new values of R1,R2,com to bin_array
                #bin_array[0,j] = new_bh_loc1
                #bin_array[1,j] = new_bh_loc2
                bin_array[9,j] = new_bin_com

    return bin_array

def bin_migration(mass_smbh, bin_array, disk_surf_model, disk_aspect_ratio_model, timestep, feedback_ratio, trap_radius, crit_ecc):
    """This function calculates how far the center of mass of a binary migrates in an AGN gas disk in a time
    of length timestep, assuming a gas disk surface density and aspect ratio profile, for
    objects of specified masses and starting locations, and returns their new locations
    after migration over one timestep.

    This function replaces com_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    bin_array : 2d float array (?)
        Bane of my existence, giant pain in the ass. All the binary parameters hacked into one.
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    timestep : float
        size of timestep in years
    feedback_ratio : float
        effect of feedback on Type I migration torque if feedback switch on
    trap_radius : float
        location of migration trap in units of r_g    

    Returns
    -------
    bin_array : 2d float array (?)
        Bane of my existence, giant pain in the ass. All the binary parameters hacked into one.
    """
    
    # get locations of center of mass of binary from bin_array
    bin_com = bin_array[9,:]
    # get masses of each binary by adding their component masses
    bin_mass = bin_array[2,:] + bin_array[3,:]
    # get orbital eccentricity of binary center of mass around SMBH
    bin_orb_ecc = bin_array[18,:]
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_model, float):
        disk_surface_density = disk_surf_model
    else:
        disk_surface_density = disk_surf_model(bin_com)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_model, float):
        disk_aspect_ratio = disk_aspect_ratio_model
    else:
        disk_aspect_ratio = disk_aspect_ratio_model(bin_com) 

    # compute migration timescale for each binary in seconds
    # eqn from Paardekooper 2014, rewritten for R in terms of r_g of SMBH = GM_SMBH/c^2
    # tau = (pi/2) h^2/(q_d*q) * (1/Omega)
    #   where h is aspect ratio, q is m/M_SMBH, q_d = pi R^2 disk_surface_density/M_SMBH
    #   and Omega is the Keplerian orbital frequency around the SMBH
    # here mass_smbh/prograde_bh_masses are both in M_sun, so units cancel
    # c, G and disk_surface_density in SI units
    tau_mig = ((disk_aspect_ratio**2)* scipy.constants.c/(3.0*scipy.constants.G) * (mass_smbh/bin_mass) / disk_surface_density) / np.sqrt(bin_com)
    # ratio of timestep to tau_mig (timestep in years so convert)
    dt = timestep * scipy.constants.year / tau_mig
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = bin_com * dt
    
    # Feedback provides a universal modification of migration distance
    # If feedback off, then feedback_ratio= ones and migration is unchanged
    # Construct empty array same size as prograde_bh_locations 

    bh_new_locations = np.empty_like(bin_com)

    # Find indices of objects where feedback ratio <1; these still migrate inwards, but more slowly
    # feedback ratio is a tuple, so need [0] part not [1] part (ie indices not details of array)
    index_inwards_modified = np.where(feedback_ratio < 1)[0]
    index_inwards_size = index_inwards_modified.size
    all_inwards_migrators = bin_com[index_inwards_modified]
    #print("all inwards migrators",all_inwards_migrators)

    #Given a population migrating inwards
    if index_inwards_size > 0: 
        for i in range(0,index_inwards_size):
                # Among all inwards migrators, find location in disk & compare to trap radius
                critical_distance = all_inwards_migrators[i]
                actual_index = index_inwards_modified[i]
                #If outside trap, migrates inwards
                if critical_distance > trap_radius:
                    bh_new_locations[actual_index] = bin_com[actual_index] - (migration_distance[actual_index]*(1-feedback_ratio[actual_index]))
                    #If inward migration takes object inside trap, fix at trap.
                    if bh_new_locations[actual_index] <= trap_radius:
                        bh_new_locations[actual_index] = trap_radius
                #If inside trap, migrates out
                if critical_distance < trap_radius:
                    #print("inside trap radius!")
                    bh_new_locations[actual_index] = bin_com[actual_index] + (migration_distance[actual_index]*(1-feedback_ratio[actual_index]))
                    #print("bh_inside_trap", bh_new_locations[actual_index])
                    #If outward migration takes object outside trap, fix at trap.
                    if bh_new_locations[actual_index] >= trap_radius:
                        bh_new_locations[actual_index] = trap_radius
                #If at trap, stays there
                if critical_distance == trap_radius:
                    #print("BH AT TRAP!")
                    #print(bh_new_locations[actual_index])
                    bh_new_locations[actual_index] = bin_com[actual_index]

    # Find indices of objects where feedback ratio >1; these migrate outwards. 
    # In Sirko & Goodman (2003) disk model this is well outside migration trap region.
    index_outwards_modified = np.where(feedback_ratio >1)[0]

    if index_outwards_modified.size > 0:
        bh_new_locations[index_outwards_modified] = bin_com[index_outwards_modified] +(migration_distance[index_outwards_modified]*(feedback_ratio[index_outwards_modified]-1))
    
    #Find indices where feedback ratio is identically 1; shouldn't happen (edge case) if feedback on, but == 1 if feedback off.
    index_unchanged = np.where(feedback_ratio == 1)[0]
    if index_unchanged.size > 0:
    # If BH location > trap radius, migrate inwards
        for i in range(0,index_unchanged.size):
            locn_index = index_unchanged[i]
            if bin_com[locn_index] > trap_radius:    
                bh_new_locations[locn_index] = bin_com[locn_index] - migration_distance[locn_index]
            # if new location is <= trap radius, set location to trap radius
                if bh_new_locations[locn_index] <= trap_radius:
                    bh_new_locations[locn_index] = trap_radius

        # If BH location < trap radius, migrate outwards
            if bin_com[locn_index] < trap_radius:
                bh_new_locations[locn_index] = bin_com[locn_index] + migration_distance[locn_index]
                #if new location is >= trap radius, set location to trap radius
                if bh_new_locations[locn_index] >= trap_radius:
                    bh_new_locations[locn_index] = trap_radius
    #print("bh new locations",np.sort(bh_new_locations))

    # new locations are original ones - distance traveled
    #bh_new_locations = prograde_bh_locations - migration_distance
    
    
    # new locations are original ones - distance traveled
    #bh_new_locations = bin_com - migration_distance
    # send locations back to bin_array and DONE!
   
    #Distance travelled per binary is old location of com minus new location of com. Is +ive(-ive) if migrating in(out)
    dist_travelled = bin_array[9,:] - bh_new_locations
    idx_nonzero_travel = np.where(dist_travelled !=0)
    num_of_bins = np.count_nonzero(bin_array[2,:])
    #for i in range(num_of_bins):
    #    print("dist travelled", bin_array[9,i], bh_new_locations[i], dist_travelled[i],bin_mass[i])
    
    # Update the binary center of mass in bin_array only if bin ecc is <= e_crit
    #First find num of bins by counting non zero M_1s.
    
    for i in range(num_of_bins):
        # If bin orb ecc < crit_ecc (circularized) then migrate
        if bin_array[18,i] <= crit_ecc:
            bin_array[9,i] = bh_new_locations[i]
        #If bin orb ecc not circularized, no migration    
        if bin_array[18,i] > crit_ecc:
            pass

    #Update location of BH 1 and BH 2
    #bin_array[0,:] = bin_array[0,:] - dist_travelled
    #bin_array[1,:] = bin_array[1,:] - dist_travelled
    #print("Loc 1",bin_array[0,:]," loc 2", bin_array[1,:])
    return bin_array

def evolve_gw(bin_array, bin_index, mass_smbh):
    """This function evaluates the binary gravitational wave frequency and strain at the end of each timestep
    Assume binary is located at z=0.1=422Mpc for now.
    """
    # Set up binary GW frequency
    # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
    
    for j in range(0,bin_index):
        temp_mass_1 = bin_array[2,j]
        temp_mass_2 = bin_array[3,j]
        temp_bin_mass = temp_mass_1 + temp_mass_2
        temp_bin_separation = bin_array[8,j]
        #1rg =1AU=1.5e11m for 1e8Msun
        rg = 1.5e11*(mass_smbh/1.e8)
        m_sun = 2.0e30
        temp_mass_1_kg = m_sun*temp_mass_1
        temp_mass_2_kg = m_sun*temp_mass_2
        temp_bin_mass_kg = m_sun*temp_bin_mass
        temp_bin_separation_meters = temp_bin_separation*rg
        
        # Set up binary strain of GW
        # h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
        # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
        
        m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/(temp_bin_mass_kg**(1/5))
        rg_chirp = (scipy.constants.G * m_chirp)/(scipy.constants.c**(2.0))
        # If separation is less than rg_chirp then cap separation at rg_chirp.
        if temp_bin_separation_meters < rg_chirp:
            temp_bin_separation_meters = rg_chirp

        nu_gw = (1.0/scipy.constants.pi)*np.sqrt(temp_bin_mass_kg*scipy.constants.G/(temp_bin_separation_meters**(3.0)))
        bin_array[19,j] = nu_gw

        # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc 
        # 1Mpc = 3.1e22m. 
        Mpc = 3.1e22
        # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
        # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
        d_obs = 421*Mpc
        strain = (4/d_obs)*rg_chirp*(np.pi*nu_gw*rg_chirp/scipy.constants.c)**(2/3)
        # But power builds up in band over multiple cycles! 
        # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
        strain_factor = 1
        if nu_gw < 10**(-6):
            strain_factor = np.sqrt(nu_gw*np.pi*(10**7)/8)

        if nu_gw > 10**(-6):
            strain_factor = 4.e3    
        # char amplitude = sqrt(N/8)h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
        # So in LISA band
        #For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt)
        bin_array[20,j] = strain_factor*strain
        #print("mbin(kg),sep(m),m_chirp(kg),rg_chirp,d_obs,nu,strain",temp_bin_mass_kg,temp_bin_separation_meters,m_chirp,rg_chirp,d_obs,nu_gw,strain)
    return bin_array

def bbh_gw_params(bin_array, bbh_gw_indices, mass_smbh, timestep, old_bbh_freq):
    """This function evaluates the binary gravitational wave frequency and strain at the end of each timestep
    Assume binary is located at z=0.1=422Mpc for now.
    """
    # Set up binary GW frequency
    # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)

    year =3.15e7
    timestep_secs = timestep*year
    # If there are BBH that meet the condition (ie if bbh_gw_indices exists, is not empty)
    if bbh_gw_indices:
        num_tracked = np.size(bbh_gw_indices,1)
        #num_tracked = len(bbh_gw_indices)    
        char_strain=np.zeros(num_tracked)
        nu_gw=np.zeros(num_tracked)
        #If number of BBH tracked has grown since last timestep, add a new component to old_gw_freq to carry out dnu/dt calculation
        while num_tracked > len(old_bbh_freq):
            old_bbh_freq = np.append(old_bbh_freq,9.e-7)
        #If number of BBH tracked has shrunk. Reduce old_bbh_freq to match size of num_tracked.
        while num_tracked < len(old_bbh_freq):
            old_bbh_freq = np.delete(old_bbh_freq,0)    

        for j in range(0,num_tracked):
            temp_mass_1 = bin_array[2,j]
            temp_mass_2 = bin_array[3,j]
            temp_bin_mass = temp_mass_1 + temp_mass_2
            temp_bin_separation = bin_array[8,j]
            #1rg =1AU=1.5e11m for 1e8Msun
            rg = 1.5e11*(mass_smbh/1.e8)
            m_sun = 2.0e30
            temp_mass_1_kg = m_sun*temp_mass_1
            temp_mass_2_kg = m_sun*temp_mass_2
            temp_bin_mass_kg = m_sun*temp_bin_mass
            temp_bin_separation_meters = temp_bin_separation*rg
        
            # Set up binary strain of GW
            # h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
            # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
        
            m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/(temp_bin_mass_kg**(1/5))
            rg_chirp = (scipy.constants.G * m_chirp)/(scipy.constants.c**(2.0))
            # If separation is less than rg_chirp then cap separation at rg_chirp.
            if temp_bin_separation_meters < rg_chirp:
                temp_bin_separation_meters = rg_chirp

            nu_gw[j] = (1.0/scipy.constants.pi)*np.sqrt(temp_bin_mass_kg*scipy.constants.G/(temp_bin_separation_meters**(3.0)))
            #bin_array[19,j] = nu_gw

            # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc 
            # 1Mpc = 3.1e22m. 
            Mpc = 3.1e22
            # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
            # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
            d_obs = 1909*Mpc
            strain = (4/d_obs)*rg_chirp*(np.pi*nu_gw[j]*rg_chirp/scipy.constants.c)**(2/3)
            # But power builds up in band over multiple cycles! 
            # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
            strain_factor = 1
            
            if nu_gw[j] < 10**(-6):
            # char amplitude = strain_factor*h0
            #                = sqrt(N/8)*h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
                strain_factor = np.sqrt(nu_gw[j]*np.pi*(10**7)/8)

            if nu_gw[j] > 10**(-6):
            #For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt).
            # char amplitude = strain_factor*h0
            #                = sqrt(freq^2/(dfreq/dt)/8)
                #print("old bbh freq",old_bbh_freq,nu_gw[j])
                dnu = np.abs(old_bbh_freq[j]-nu_gw[j])
                dnu_dt = dnu/timestep_secs
                nusq = nu_gw[j]*nu_gw[j]
                strain_factor = np.sqrt((nusq/dnu_dt)/8)
        
            char_strain[j] = strain_factor*strain
            #print("mbin(kg),sep(m),m_chirp(kg),rg_chirp,d_obs,nu,strain",temp_bin_mass_kg,temp_bin_separation_meters,m_chirp,rg_chirp,d_obs,nu_gw,strain)
    return char_strain,nu_gw

def evolve_emri_gw(inner_disk_locations,inner_disk_masses, mass_smbh,timestep,old_gw_freq):
    """This function evaluates the EMRI gravitational wave frequency and strain at the end of each timestep
    Assume binary is located at z=0.1=422Mpc for now. 
    z=0.5=1909 Mpc, using co-moving radial distance from https://www.astro.ucla.edu/~wright/CosmoCalc.html
    temp_emri_array is the value of the emri_array from the previous timestep.
    """
    # Set up binary GW frequency
    # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
    num_emris = np.size(inner_disk_locations)

    char_strain=np.zeros(num_emris)
    nu_gw=np.zeros(num_emris)
    
    m1 = mass_smbh
    
    #If number of EMRIs has grown since last timestep, add a new component to old_gw_freq to carry out dnu/dt calculation
    if num_emris > len(old_gw_freq):
        old_gw_freq = np.append(old_gw_freq,9.e-7)

    for i in range(0,num_emris):
        m2 = inner_disk_masses[i]
        temp_bin_mass = m1 + m2
        temp_bin_separation = inner_disk_locations[i]
        #1rg =1AU=1.5e11m for 1e8Msun
        rg = 1.5e11*(mass_smbh/1.e8)
        m_sun = 2.0e30
        temp_mass_1_kg = m_sun*m1
        temp_mass_2_kg = m_sun*m2
        temp_bin_mass_kg = m_sun*temp_bin_mass
        temp_bin_separation_meters = temp_bin_separation*rg
        #Year in seconds. Multiply to get timestep in seconds
        year = 3.15e7
        timestep_secs = year*timestep
        
        # Set up binary strain of GW
        # h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
        # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
        
        m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/(temp_bin_mass_kg**(1/5))
        rg_chirp = (scipy.constants.G * m_chirp)/(scipy.constants.c**(2.0))
        # If separation is less than rg_chirp then cap separation at rg_chirp.
        if temp_bin_separation_meters < rg_chirp:
            temp_bin_separation_meters = rg_chirp

        nu_gw[i] = (1.0/scipy.constants.pi)*np.sqrt(temp_bin_mass_kg*scipy.constants.G/(temp_bin_separation_meters**(3.0)))

        # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc 
        # 1Mpc = 3.1e22m. 
        Mpc = 3.1e22
        # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
        # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
        d_obs = 1909*Mpc
        strain = (4/d_obs)*rg_chirp*(np.pi*nu_gw[i]*rg_chirp/scipy.constants.c)**(2/3)
        # But power builds up in band over multiple cycles! 
        # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
        strain_factor = 1        

        if nu_gw[i] < 10**(-6):
            # char amplitude = strain_factor*h0
            #                = sqrt(N/8)*h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
            strain_factor = np.sqrt(nu_gw[i]*np.pi*(10**7)/8)

        if nu_gw[i] > 10**(-6):
            #For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt).
            # char amplitude = strain_factor*h0
            #                = sqrt(freq^2/(dfreq/dt)/8)
            dnu = np.abs(old_gw_freq[i]-nu_gw[i])
            dnu_dt = dnu/timestep_secs
            nusq = nu_gw[i]*nu_gw[i]
            strain_factor = np.sqrt((nusq/dnu_dt)/8)
        

        char_strain[i] = strain_factor*strain
        #print("mbin(kg),sep(m),m_chirp(kg),rg_chirp,d_obs,nu,strain",temp_bin_mass_kg,temp_bin_separation_meters,m_chirp,rg_chirp,d_obs,nu_gw,strain)
    return char_strain,nu_gw

def ionization_check(bin_array, bin_index, mass_smbh):
    """This function tests whether a binary has been softened beyond some limit.
        Returns index of binary to be ionized. Otherwise returns -1.
        The limit is set to some fraction of the binary Hill sphere, frac_R_hill

        Default is frac_R_hill =1.0 (ie binary is ionized at the Hill sphere). 
        Change frac_R_hill if you're testing binary formation at >R_hill.
        
        R_hill = a_com*(M_bin/3M_smbh)^1/3

        where a_com is the radial disk location of the binary center of mass (given by bin_array[9,*]),
        M_bin = M_1 + M_2 = bin_array[2,*]+bin_array[3,*] is the binary mass
        M_smbh is the SMBH mass (given by mass_smbh) 
        
        and binary separation is in bin_array[8,*].
        Condition is 
        if bin_separation > frac_R_hill*R_hill:
            Ionize binary. Return flag valued at index of binary in bin_array.
            Then in test1.py remove binary from bin_array! decrease bin_index by 1.
            Add two new singletons to the singleton arrays.


    """
    #Define Ionization threshold as fraction of Hill sphere radius
    #Default is 1.0. Change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    #Default return for the function is -1
    ionization_flag = -1.0
    for j in range(0,bin_index):
        # Read in binary masses (units of M_sun)
        temp_mass_1 = bin_array[2,j]
        temp_mass_2 = bin_array[3,j]
        # Set up binary mass (units of M_sun)
        temp_bin_mass = temp_mass_1 + temp_mass_2
        # Mass ratio of binary to SMBH (unitless)
        temp_mass_ratio = temp_bin_mass/mass_smbh
        # Read in binary separation (units of r_g of the SMBH =GM_smbh/c^2)
        temp_bin_separation = bin_array[8,j]
        # Read in binary com disk location ( units of r_g of the SMBH = GM_smbh/c^2)
        temp_bin_com_radius = bin_array[9,j]
        #Define binary Hill sphere (units of r_g of SMBH where 1r_g = GM_smbh/c^2 = 1AU for 10**8Msun SMBH
        temp_hill_sphere = temp_bin_com_radius*((temp_mass_ratio/3)**(1/3))

        if temp_bin_separation > frac_rhill*temp_hill_sphere:
            #Commented out for now
            # print("Ionize binary!", temp_bin_separation, frac_rhill*temp_hill_sphere)
            #Ionize binary!!!
            # print("Bin_array index",j)
            ionization_flag = j
            
    return ionization_flag

def contact_check(bin_array, bin_index, mass_smbh):
    """ This function tests to see if the binary separation has shrunk so that the binary is touching!
        Touching condition is where binary separation is <= R_g(M_chirp).
        Since binary separation is in units of r_g (GM_smbh/c^2) then condition is simply:
            binary_separation < M_chirp/M_smbh
        """
    for j in range(0,bin_index):
        #Read in mass 1, mass 2 (units of M_sun)
        temp_mass_1 = bin_array[2,j]
        temp_mass_2 = bin_array[3,j]
        #Total binary mass
        temp_bin_mass = temp_mass_1 + temp_mass_2
        #Binary separation in units of r_g=GM_smbh/c^2
        temp_bin_separation = bin_array[8,j]
        
        #Condition is if binary separation < R_g(M_chirp). 
        # Binary separation is in units of r_g(M_smbh) so 
        # condition is separation < R_g(M_chirp)/R_g(M_smbh) =M_chirp/M_smbh
        # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
        # M1,M2, M_smbh are all in units of M_sun
        m_chirp = ((temp_mass_1*temp_mass_2)**(3/5))/(temp_bin_mass**(1/5))
        condition = m_chirp/mass_smbh
        # If binary separation < merge condition, set binary separation to merge condition
        if temp_bin_separation < condition:
            bin_array[8,j] = condition
            bin_array[11,j] = -2
    return bin_array    

def ionization_check(bin_array, bin_index, mass_smbh):
    """This function tests whether a binary has been softened beyond some limit.
        Returns index of binary to be ionized. Otherwise returns -1.
        The limit is set to some fraction of the binary Hill sphere, frac_R_hill

        Default is frac_R_hill =1.0 (ie binary is ionized at the Hill sphere). 
        Change frac_R_hill if you're testing binary formation at >R_hill.
        
        R_hill = a_com*(M_bin/3M_smbh)^1/3

        where a_com is the radial disk location of the binary center of mass (given by bin_array[9,*]),
        M_bin = M_1 + M_2 = bin_array[2,*]+bin_array[3,*] is the binary mass
        M_smbh is the SMBH mass (given by mass_smbh) 
        
        and binary separation is in bin_array[8,*].
        Condition is 
        if bin_separation > frac_R_hill*R_hill:
            Ionize binary. Return flag valued at index of binary in bin_array.
            Then in test1.py remove binary from bin_array! decrease bin_index by 1.
            Add two new singletons to the singleton arrays.


    """
    #Define Ionization threshold as fraction of Hill sphere radius
    #Default is 1.0. Change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    #Default return for the function is -1
    ionization_flag = -1.0
    for j in range(0,bin_index):
        # Read in binary masses (units of M_sun)
        temp_mass_1 = bin_array[2,j]
        temp_mass_2 = bin_array[3,j]
        # Set up binary mass (units of M_sun)
        temp_bin_mass = temp_mass_1 + temp_mass_2
        # Mass ratio of binary to SMBH (unitless)
        temp_mass_ratio = temp_bin_mass/mass_smbh
        # Read in binary separation (units of r_g of the SMBH =GM_smbh/c^2)
        temp_bin_separation = bin_array[8,j]
        # Read in binary com disk location ( units of r_g of the SMBH = GM_smbh/c^2)
        temp_bin_com_radius = bin_array[9,j]
        #Define binary Hill sphere (units of r_g of SMBH where 1r_g = GM_smbh/c^2 = 1AU for 10**8Msun SMBH
        temp_hill_sphere = temp_bin_com_radius*((temp_mass_ratio/3)**(1/3))

        if temp_bin_separation > frac_rhill*temp_hill_sphere:
            #Comment out for now
            # print("Ionize binary!", temp_bin_separation, frac_rhill*temp_hill_sphere)
            #Ionize binary!!!
            # print("Bin_array index",j)
            ionization_flag = j
            
    return ionization_flag

def reality_check(bin_array, bin_index, nbin_properties):
    """ This function tests to see if the binary is real. If location = 0 or mass = 0 *and* any other element is NON-ZERO then discard this binary element.
        Returns flag, negative for default, if positive it is the index of the binary column to be deleted.
        """
    reality_flag = -2

    for j in range(0,bin_index):
        
        #Check other elements in bin_array are NON-ZERO
        for i in range(0,nbin_properties):
            #Read in mass 1, mass 2 (units of M_sun)
            temp_mass_1 = bin_array[2,j]
            temp_mass_2 = bin_array[3,j]
            #Read in location 1, location 2 (units of R_g (M_smbh))
            temp_location_1 = bin_array[0,j]
            temp_location_2 = bin_array[1,j]
            #If any element in binary other than the location or mass is non-zero
            if bin_array[i,j] > 0:
                #Check if any of locations or masses is zero
                if temp_mass_1 == 0 or temp_mass_2 == 0 or temp_location_1 == 0 or temp_location_2 == 0:
                    #Flag this binary    
                    reality_flag = j    
        
    return reality_flag        
