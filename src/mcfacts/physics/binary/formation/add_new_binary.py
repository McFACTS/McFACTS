import numpy as np

import scipy

def add_to_binary_array2(rng, bin_array, bh_locations, bh_masses, bh_spins, bh_spin_angles, bh_gens, close_encounters, bindex, retro, mass_smbh, verbose=False,):
    """This is where we add new binaries. We take the locations, masses, spins, spin angles and generations
    from the relevant singletons, found in hillsphere.binary_check, and sort those parameters into bin_array. 
    We then ADD additional parameters relevant only for binaries, including semi-major axis of the binary,
    semi-major axis of the orbit of the center of mass of the binary around the SMBH, a flag to permit or 
    suppress retrograde binaries, eventually eccentricity and inclination. There is also a verbose flag 
    that is by default set to False, to suppress overly detailed output.

    Parameters
    ----------
    bin_array : [17, bindex] mixed array
        binary black hole array, multi-dimensional;
            [0,j]: float
            location of object 1 at formation of binary (distance from SMBH in r_g)
            [1,j]: float
            location of object 2 at formation of binary (distance from SMBH in r_g)
            [2,j]: float
            mass of obj 1 at time t in units of solar masses
            [3,j]: float
            mass of obj 2 at time t in units of solar masses
            [4,j]: float
            dimensionless spin magnitude of obj 1 at time t
            [5,j]: float
            dimensionless spin magnitude of obj 2 at time t
            [6,j]: float
            spin angle of obj 1 wrt disk gas in radians at time t
            [7,j]: float
            spin angle of obj 2 wrt disk gas in radians at time t
            [8,j]: float
            binary semi-major axis around c.o.m. in units of r_g of SMBH
            [9,j]: float
            binary center of mass location wrt SMBH in r_g
            [10,j]: float
            time to merger through GW alone (not set here)
            [11,j]: int
            merger flag = -2 if merging this timestep, else = 0 (not set here)
            [12,j]: float
            time of merger if binary has already merged (not set here)
            [13,j]: float
            binary eccentricity around binary center of mass
            [14,j]: int
            generation of obj 1 (1=natal black hole, no prior mergers)
            [15,j]: int
            generation of obj 2 (1=natal black hole, no prior mergers)
            [16,j]: int
            binary angular momentum switch +1/-1 for pro/retrograde
            [17,j]: float
            binary orbital inclination
            [18,j]: float
            binary orbital eccentricity of binary center of mass around SMBH
    bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    bh_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    bh_spins : float array
        dimensionless spin magnitude of prograde singleton BH at start of timestep
    bh_spin_angles : float array
       spin angle of prograde singleton BH wrt the gas disk angular momentum in radians??? at start of time step
    bh_gens : int array
        generation of prograde singleton BH (1=no previous merger history)
    close_encounters : [2,N] int array
        array of indices corresponding to locations in prograde_bh_locations, prograde_bh_masses,
        prograde_bh_spins, prograde_bh_spin_angles, and prograde_bh_generations which corresponds
        to binaries that form in this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.
    bindex : int
        counter for length of bin_array before adding new binaries
    retro : int
        switch to inhibit retrograde binaries.
        retro = 0 turns all retrograde BBH at formation into prograde BBH. 
        retro = 1 keeps retrograde BBH once they form. Eventually will turn this into
        something physically motivated rather than a brute force switch.
        default : 0
    verbose : bool, optional
        how much output do you want? set True for debugging, by default False

    Returns
    -------
    bin_array : [17, bindex+N] float array
        as for input, but updated (thus longer), to include newly formed binaries
    """
    # find number of new binaries based on indices from hillsphere.binary_check 
    num_new_bins = np.shape(close_encounters)[1]  

    # If there are new binaries, actually form them!
    if num_new_bins > 0:
        # send close encounter indices to new array
        array_of_indices = close_encounters
        #print("Close encounters ", np.shape(close_encounters)[1], array_of_indices)
        bincount = 0
        # for all the new binaries that need to be created
        for j in range(bindex, bindex + num_new_bins):
            #print("num new bins", num_new_bins)
            # for each member of the binary
            for i in range(0,2):
                # pick the [N,0] or [N,1] index for member 1 and member 2 of binary N
                #if num_new_bins > 1:
                #print("array of indices comps",array_of_indices[0],array_of_indices[1])
                if num_new_bins == 1:
                    thing1 = array_of_indices[i]
                else:
                    #print("More than 1 BIN",array_of_indices[0][0],array_of_indices[1][0])
                    thing1 = array_of_indices[i][bincount]
                
                #print("thing1",thing1)
                bin_array[i,j] = bh_locations[thing1]
                bin_array[i+2,j] = bh_masses[thing1]
                bin_array[i+4,j] = bh_spins[thing1]
                bin_array[i+6,j] = bh_spin_angles[thing1]
                # For new binary create initial binary semi-major axis
                temp_loc_1 = bin_array[0,j]
                temp_loc_2 = bin_array[1,j]
                temp_bin_separation = np.abs(temp_loc_1 - temp_loc_2)
                bin_array[8,j] = temp_bin_separation
                # Binary c.o.m.= location_1 + separation*M_2/(M_1+M_2)
                temp_mass_1 = bin_array[2,j]
                temp_mass_2 = bin_array[3,j]
                temp_bin_mass = temp_mass_1 + temp_mass_2
                bin_array[9,j] = temp_loc_1 + (temp_bin_separation*temp_mass_2/temp_bin_mass)
                #Set up binary eccentricity around its own center of mass. Draw uniform value btwn [0,1]
                bin_array[13,j] = rng.random()
                # Set up binary member generations
                bin_array[i+14,j] = bh_gens[thing1]
                # Set up bin orb. ang. mom. (randomly +1 (pro) or -1(retrograde))
                # random number between [0,1]
                random_uniform_number = rng.random()
                # Generate a random number between [-retro,1-retro] where retro is the fraction of BBH that are retrograde.
                # For default retro =0.1, range is [-0.1,0.9] and 90% of BBH are prograde.
                bh_initial_orb_ang_mom = random_uniform_number - retro
                # If retro =0, range = [0,1] and set L_bbh = +1.
                if retro == 0:
                    bh_initial_orb_ang_mom = 1
                # If retro =0.1 (default), range = [-0.1,0.9].
                # If range <0, L_BBH = -1; if range >0, L_BBH = +1
                if retro > 0:
                    if bh_initial_orb_ang_mom < 0:
                        bh_initial_orb_ang_mom = -1
                    if bh_initial_orb_ang_mom > 0:
                        bh_initial_orb_ang_mom = 1

                bin_array[16,j] = bh_initial_orb_ang_mom                
                #Set up binary inclination (in units radians). Will want this to be pi radians if retrograde.
                bin_array[17,j] = 0
                # Set up binary orbital eccentricity of com around SMBH. Assume initially v.small (e~0.01)
                bin_array[18,j] = 0.01
                # Set up binary GW frequency
                # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
                #1rg =1AU=1.5e11m for 1e8Msun
                rg = 1.5e11*(mass_smbh/1.e8)
                temp_bin_separation_meters = temp_bin_separation*rg
                temp_mass_1_kg = 2.e30*temp_mass_1
                temp_mass_2_kg = 2.e30*temp_mass_2
                temp_bin_mass_kg = 2.e30*temp_bin_mass
                m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/(temp_bin_mass_kg**(1/5))
                rg_chirp = (scipy.constants.G * m_chirp)/(scipy.constants.c**(2.0))
                if temp_bin_separation_meters < rg_chirp:
                    temp_bin_separation_meters = rg_chirp
                    
                nu_gw = (1.0/scipy.constants.pi)*np.sqrt(temp_bin_mass_kg*scipy.constants.G/(temp_bin_separation_meters**(3.0)))
                bin_array[19,j] = nu_gw
                # Set up binary strain of GW
                # h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
                # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
                
                
                # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc 
                # 1Mpc = 3.1e22m. 
                Mpc = 3.1e22
                # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
                # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
                d_obs = 421*Mpc
                strain = (4/d_obs)*rg_chirp*(np.pi*nu_gw*rg_chirp/scipy.constants.c)**(2/3)
                bin_array[20,j] = strain
            bincount = bincount + 1
            #print("new binary",bin_array[:,j])
        if verbose:
            print("New Binary")
            print(bin_array)
        

    return bin_array
