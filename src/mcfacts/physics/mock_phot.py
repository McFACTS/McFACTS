#from pylab import *

import sys, os, time, string, math, subprocess
import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
#import scipy.interpolate as interpol
from astropy import constants as const
from astropy import units as u

#SI units
Msun=const.M_sun.value #kg per solar mass
Rsun=const.R_sun.value #meters per solar radius
G=const.G.value
c=const.c.value
sigma_SB=const.sigma_sb.value #stefan-boltzmann const
yr= (1 * u.yr).to(u.s).value #seconds per year
pc=const.pc.value #meters per parsec
AU=const.au.value #meters per AU
h=const.h.value #planck const
kB=const.k_B.value #boltzmann const
m_p=const.m_p.value #mass of proton/hydrogen atom
sigma_T=const.sigma_T.value #Thomson xsec
PI=np.pi
m_per_nm=1.0e-9

def find_B_lambda(temp_func, lam, bin_orb_a):
    #Planck function
    #BB intensity=2hc^2/lam^5 * 1/(exp(hc/lamkT)-1)
    I=(2*h*c**2/lam**5)/(math.e**(h*c/(lam*kB*temp_func(bin_orb_a)))-1)

    return I

def v_kep(M, r):
    #compute keplerian velocity given M, r
    #Inputs:M, r in SI
    #outputs: v in SI
    v=np.sqrt(G*M/r)

    return v

# we probably have something to calculate v_kep
# probs dont have Planck fxn

def find_spotTemp(epsilon,M_SMBH,dotm_edd,radius, disk_sound_speed, temp_func, r_in,model,r_alt,f_dep,M_bin,deltaM,spot_radius,R_Hill):
    #Spot temp varies as r from binary/merger due to orbital change
    #from evil supervillain mass loss
    #3 zones:
    # a) inner most zone is shocked, temp varies with r
    # b) middle zone (w/ most of mass) expands subsonically, ~constant T
    # c) outer zone is mass lost from RHill to SMBH Kep flow, subsonic, ~const T
    #inputs:
    #spot_radius in SI
    #R_Hill in SI
    #M_bin, M_SMBH, deltaM in Msun
    #radius in r_g_SMBH
    r_g_bin=(G*M_bin*Msun)/c**2
    r_g_SMBH=(G*M_SMBH*Msun)/c**2

    #get sound speed (SI)
    c_s= disk_sound_speed(radius, model) # mcfacts
    #first compute Mach number--need spot_radius in r_g_bin
    Mach=0.5/np.sqrt(spot_radius/r_g_bin)*(c/c_s)*(deltaM/M_bin)
    #and spot radius at which Mach number=1.0 (in r_g_bin)--for deltavee2
    spot_rad_Mach1=0.25*(c/c_s)**2*(deltaM/M_bin)**2
    #and local disk temp in absence of binary
    Tdisk=temp_func(epsilon,M_SMBH,dotm_edd,radius,r_in,model,r_alt,f_dep)
    #compute velocity change from Mach 1 region to exterior of spot (deltavee2)
    deltavee2=v_kep((M_bin*Msun),(spot_rad_Mach1*r_g_bin))-v_kep((M_bin*Msun),R_Hill)
    #compute velocity change from material that leaves R_Hill for disk flow
    deltavee3=v_kep((M_SMBH*Msun),(radius*r_g_SMBH))-v_kep((M_SMBH*Msun),((radius*r_g_SMBH)-R_Hill))

    #compute change of R_Hill--for some a, zone 3 is too thin to count
    #(unless subdivide spot into even smaller annuli)
    deltaR_Hill=R_Hill*(1.0-(1.0-deltaM/M_bin)**(1/3))

    if ((Mach>=1.0) and (spot_radius<(R_Hill-deltaR_Hill))):
        #if supersonic and not in zone 3 (the lost section of the Hill sphere)
        T=(5.0/16.0)*Mach**2*Tdisk
    elif ((Mach<1.0) and (spot_radius<(R_Hill-deltaR_Hill))):
        #if subsonic and not in zone 3
        T=(1.0/3.0)*(m_p/kB)*((deltaM/M_bin)*deltavee2)**2
    elif (spot_radius>=(R_Hill-deltaR_Hill)):
        #if in zone 3 (lost sec of Hill sphere)
        T=(1.0/3.0)*(m_p/kB)*deltavee3**2
        print(T)
    
    return T
    
# we have function to calculate sound speed MCFACTS

# we have temperature in MCFACTS


def get_filter_profile(path, filename):
    #assumes file contains ONLY wavelength and flux, in appropriate units, no headers, etc.
    
    #open the file for reading
    file1 = open(path+filename, 'r')

    #read in the data line by line, split into float lists of wavelength and transmission fraction
    lam1list=[]
    trans1list=[]
    for line in file1:
        line=line.strip()
        columns=line.split()
        lam1list.append(float(columns[0]))
        trans1list.append(float(columns[1]))

    #close file
    file1.close()

    #re-cast as arrays (from lists) for manipulation
    lam1 = np.array(lam1list)
    trans1 = np.array(trans1list)

    return lam1, trans1

def get_spotTeff(spot_temp, log_tau):
    #use very basic propagation formula from SG03 (eqn 4), given T_mid, tau

    #unlog
    tau=10.0**log_tau
    thing=(3.0*tau/8.0)+0.5+(1.0/(4.0*tau))
    Teff=spot_temp*(thing**-0.25)
    
    return Teff
# ask pagn whats T as function of radius for this disk
# mod to calculate spot temp

def spot(bin_orb_a, filter_filenames, mass_final, smbh_mass, spot_radius_in, 
         epsilon, find_area, dotm_edd, event_rad, radius_in, tempmod, R_alt,
         f_depress, M_bin, deltaM):
    spot_phot=np.zeros((len(bin_orb_a),len(filter_filenames)))
    #spot_phot=np.zeros((len(event_rad),len(filter_filenames)))
    #debugging
    max_spot_temp=0.0
    max_temp_area=0.0
    max_spot_radius=0.0
    #compute spot continuum
    #assume spot radius=R_Hill=a*(q/3)^1/3
    spot_F_lam_tot=np.zeros(len(filterlam[j]))


    R_Hill= bin_orb_a * ((mass_final / smbh_mass) / 3)**(1/3)  #event_rad[i]*r_g_SMBH*pow(M_bin/(3.0*M_SMBH),1.0/3.0)
    log_spot_radius=np.arange(np.log10(spot_radius_in*bin_orb_a),np.log10(R_Hill),0.01)
    spot_radius=10**log_spot_radius
    for k in range((len(spot_radius)-1)):
        spot_temp=find_spotTemp(epsilon,smbh_mass,dotm_edd,event_rad[i],radius_in,tempmod,R_alt,f_depress,M_bin,deltaM,spot_radius[k],R_Hill)
        spot_surf_temp=get_spotTeff(spot_temp, log_tau[i])
        spot_B_lambda=find_B_lambda(spot_surf_temp, filterlam[j])
        spot_area=find_area(spot_radius[k],spot_radius[k+1],1.0)
        #debugging
        if (spot_temp>=max_spot_temp):
            max_spot_temp=spot_temp
            max_temp_area=spot_area
            max_temp_B_lam=spot_B_lambda
        #compute flux
        spot_F_lam_ann=spot_area*spot_B_lambda
        spot_F_lam_tot=spot_F_lam_ann + spot_F_lam_tot
    #multiply SED by transmission profile and sum
    spot_phot[i][j]=sum(spot_F_lam_tot*filtertrans[j])
    #compute flux ratios
    spot_to_disk_fluxratio[i][j]=spot_phot[i][j]/disk_phot[j]
            