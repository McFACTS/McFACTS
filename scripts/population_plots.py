#!/usr/bin/env python3

######## Globals ########
COLUMN_NAMES = "iter CM M chi_eff a_tot spin_angle m1 m2 a1 a2 theta1 theta2 gen1 gen2 t_merge chi_p"

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import os
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data


######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname-emris",
        default="output_mergers_emris.dat",
        type=str, help="output_emris file")
    parser.add_argument("--fname-mergers",
        default="output_mergers_population.dat",
        type=str, help="output_mergers file")
    parser.add_argument("--plots-directory",
        default=".",
        type=str, help="directory to save plots")
    parser.add_argument("--fname-lvk",
        default="output_mergers_lvk.dat",
        type=str, help="output_lvk file")
    opts = parser.parse_args()
    print(opts.fname_mergers)
    assert os.path.isfile(opts.fname_mergers)
    assert os.path.isfile(opts.fname_emris)
    assert os.path.isfile(opts.fname_lvk)
    return opts

######## Main ########
def main():
    #plt.style.use('seaborn-v0_8-poster')

    # need section for loading data
    opts = arg()
    mergers = np.loadtxt(opts.fname_mergers, skiprows=2)
    emris = np.loadtxt(opts.fname_emris, skiprows=2)
    lvk = np.loadtxt(opts.fname_lvk,skiprows=2)

    # exclude all rows with NaNs or zeros in the final mass column
    mask = (np.isfinite(mergers[:,2])) & (mergers[:,2] != 0)
    mergers = mergers[mask]
    
    # plt.figure(figsize=(10,6))
    # plt.figure()

    # Plot intial and final mass distributions
    counts, bins = np.histogram(mergers[:,2])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(mergers[:,2].min()), int(mergers[:,2].max())+2, 1)
    plt.hist(mergers[:,2], bins=bins, align='left', color='rebeccapurple', alpha=0.9, rwidth=0.8)

    # plt.hist(bh_masses_by_sorted_location, bins=numbins, align='left', label='Final', color='purple', alpha=0.5)
    # plt.hist(binary_bh_array[2:4,:bin_index], bins=numbins, align='left', label='Final', color=['purple'], alpha=0.5)
    #plt.title(f'Black Hole Merger Remnant Masses\n'+
    #            f'Number of Mergers: {mergers.shape[0]}')
    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Mass ($M_\odot$)')

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory+"/merger_remnant_mass.png", format='png')
    plt.tight_layout()
    plt.close()


    # TQM has a trap at 245r_g, SG has a trap radius at 700r_g.
    #trap_radius = 245
    trap_radius = 700
    plt.figure()
    #plt.title('Migration Trap influence')
    for i in range(len(mergers[:,1])):
        if mergers[i,1] < 10.0:
            mergers[i,1] = 10.0

    plt.scatter(mergers[:,1], mergers[:,2], color='teal')
    plt.axvline(trap_radius, color='grey', linewidth=20, zorder=0, alpha=0.6, label=f'Radius = {trap_radius} '+r'$R_g$')
    #plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Mass ($M_\odot$)')
    plt.xlabel(r'Radius ($R_g$)')
    plt.xscale('log')
    plt.legend(frameon=False)
    plt.ylim(10,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig(opts.plots_directory+"/merger_mass_v_radius.png", format='png')
    plt.close()




    m1 = np.zeros(mergers.shape[0])
    m2 = np.zeros(mergers.shape[0])
    mass_ratio = np.zeros(mergers.shape[0])
    for i in range(mergers.shape[0]):
        if mergers[i,6] < mergers[i,7]:
            m1[i] = mergers[i,7]
            m2[i] = mergers[i,6]
            mass_ratio[i] = mergers[i,6] / mergers[i,7]
        else:
            mass_ratio[i] = mergers[i,7] / mergers[i,6]
            m1[i] = mergers[i,6]
            m2[i] = mergers[i,7]

    # (q,X_eff) Figure details here:
    # Want to highlight higher generation mergers on this plot
    chi_eff = mergers[:,3]
    gen1 = mergers[:,12]
    gen2 = mergers[:,13]
    chi_p = mergers[:,15]
    
    plt.ylim(0,1.05)
    plt.xlim(-1,1)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    # Pipe operator (|) = logical OR. (&)= logical AND.
    high_gen_chi_eff = np.where((gen1 > 1.0) | (gen2 > 1.0), chi_eff, np.nan)
    extreme_gen_chi_eff = np.where((gen1 > 2.0) | (gen2 > 2.0), chi_eff, np.nan)


    ax2.scatter(chi_eff,mass_ratio, color='darkgoldenrod')
    ax2.scatter(high_gen_chi_eff,mass_ratio, color='rebeccapurple',marker='+')
    ax2.scatter(extreme_gen_chi_eff,mass_ratio,color='red',marker='o')
    #plt.scatter(chi_eff, mass_ratio, color='darkgoldenrod')
    #plt.title("Mass Ratio vs. Effective Spin")
    plt.ylabel(r'$q = M_2 / M_1$ ($M_1 > M_2$)')
    plt.xlabel(r'$\chi_{\rm eff}$')
    plt.ylim(0,1.05)
    plt.xlim(-1,1)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./q_chi_eff.png", format='png')
    plt.close()

    #Figure of Disk radius vs Chi_p follows.
    # Can break out higher mass Chi_p events as test/illustration.
    #Set up default arrays for high mass BBH (>40Msun say) to overplot vs chi_p. 
    all_masses = mergers[:,2]
    all_locations = mergers[:,1]
    mass_bound = 40.0
    #print("All BBH merger masses:",all_masses)
    #print("All BBH merger locations:",all_locations)
    high_masses = np.where(all_masses > mass_bound, all_masses, np.nan)
    #print("High masses", high_masses)
    high_masses_locations = np.where(np.isfinite(high_masses),all_locations, np.nan )
    #print("High masses locations",high_masses_locations)
    plt.ylim(0,1)
    plt.xlim(0.,5.e4)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(all_locations,chi_p, color='darkgoldenrod')
    ax1.scatter(high_masses_locations,chi_p, color='rebeccapurple',marker='+')
    #plt.title("In-plane effective Spin vs. Merger radius")
    plt.ylabel(r'$\chi_{\rm p}$')
    plt.xlabel(r'Radius ($R_g$)')
    plt.xlim(0.,5.e4)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./r_chi_p.png", format='png')
    plt.close()
    
    # plt.figure()
    # index = 2
    # mode = 10
    # pareto = (np.random.pareto(index, 1000) + 1) * mode

    # x = np.linspace(1,100)
    # p = index*mode**index / x**(index+1)

    # # count, bins, _ = plt.hist(pareto, 100)
    # plt.plot(x, p)
    # plt.xlim(0,100)
    # plt.show()


    plt.figure()
    #plt.title("Time of Merger after AGN Onset")
    plt.scatter(mergers[:,14]/1e6, mergers[:,2], color='darkolivegreen')
    plt.xlabel('Time (Myr)')
    plt.ylabel(r'Mass ($M_\odot$)')
    # plt.xscale("log")
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig(opts.plots_directory+'/time_of_merger.png', format='png')
    plt.close()




    plt.figure()
    plt.scatter(m1, m2, color='k')
    plt.xlabel(r'$M_1$ ($M_\odot$)')
    plt.ylabel(r'$M_2$ ($M_\odot$)')
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig(opts.plots_directory+'/m1m2.png', format='png')

    #GW strain figure: 
    #make sure LISA.py and PhenomA.py in /vis directory
    #Using https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/blob/master/LISA.py 
    # and same location for PhenomA.py
    #Also make sure LIGO sensitivity curves are in /vis directory

    #--------------Here we're pulling the LIGO data from the released files and setting them to H1 and L1 values-------------
    # READ LIGO O3 Sensitivity data (from https://git.ligo.org/sensitivity-curves/o3-sensitivity-curves)
    H1 = impresources.files(data) / \
        'O3-H1-C01_CLEAN_SUB60HZ-1262197260.0_sensitivity_strain_asd.txt'  
    L1 = impresources.files(data) / \
        'O3-L1-C01_CLEAN_SUB60HZ-1262141640.0_sensitivity_strain_asd.txt'
    # Adjust sep according to your delimiter (e.g., '\t' for tab-delimited files)
    dfh1 = pd.read_csv(H1, sep='\t', header=None)  # Use header=None if the file doesn't contain header row
    dfl1 = pd.read_csv(L1, sep='\t', header=None)

    # Access columns as df[0], df[1], ...
    f_H1 = dfh1[0]
    h_H1 = dfh1[1]

    #H - hanford 
    #L - Ligvston

    # create LISA object
    lisa = li.LISA() 

    # Plot LISA's sensitivity curve
    #-------------- f is the frequency (x-axis) being created -------------
    #-------------- Sn is the sensitivity curve of LISA -------------
    f  = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
    Sn = lisa.Sn(f)


    fig, ax = plt.subplots(1, figsize=(8,6))
    #plt.tight_layout()

    ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
    ax.set_ylabel(r'${\rm h}_{\rm char}$', fontsize=20, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.set_xlim(1.0e-7, 1.0e+4)
    ax.set_ylim(1.0e-24, 1.0e-15)

    #----------Finding the rows in which EMRIs signals are either identical or zeroes and removing them----------
    identical_rows_emris = np.where( emris[:,5] == emris[:,6])
    zero_rows_emris = np.where(emris[:,6] == 0)    
    emris = np.delete(emris,identical_rows_emris,0)
    #emris = np.delete(emris,zero_rows_emris,0)
    emris[~np.isfinite(emris)] = 1.e-40

    #----------Finding the rows in which LVKs signals are either identical or zeroes and removing them----------
    identical_rows_lvk = np.where(lvk[:,5] == lvk[:,6])
    zero_rows_lvk = np.where(lvk[:,6] == 0)
    lvk = np.delete(lvk,identical_rows_lvk,0)
    #lvk = np.delete(lvk,zero_rows_lvk,0)
    lvk[~np.isfinite(lvk)] = 1.e-40

    #----------Setting the values for the EMRIs and LVKs signals and inverting them----------
    inv_freq_emris = 1/emris[:,6]
    inv_freq_lvk = 1/lvk[:,6]
    #ma_freq_emris = np.ma.where(freq_emris == 0)
    #ma_freq_lvk = np.ma.where(freq_lvk == 0)
    #indices_where_zeros_emris = np.where(freq_emris = 0.)
    #freq_emris = freq_emris[freq_emris !=0]
    #freq_lvk = freq_lvk[freq_lvk !=0]

    #inv_freq_emris = 1.0/ma_freq_emris
    #inv_freq_lvk = 1.0/ma_freq_lvk
    # timestep =1.e4yr
    timestep = 1.e4
    strain_per_freq_emris = emris[:,5]*inv_freq_emris/timestep
    strain_per_freq_lvk = lvk[:,5]*inv_freq_lvk/timestep
    ax.loglog(f, np.sqrt(f*Sn),label = 'LISA Sensitivity') # plot the characteristic strain
    ax.loglog(f_H1, h_H1,label = 'LIGO O3, H1 Sensitivity') # plot the characteristic strain
    ax.scatter(emris[:,6],strain_per_freq_emris)
    ax.scatter(lvk[:,6],strain_per_freq_lvk)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.loglog(f_L1, h_L1,label = 'LIGO O3, L1 Sensitivity') # plot the characteristic strain

    #ax.loglog(f_gw,h,color ='black', label='GW150914')

    ax.legend()
    ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
    ax.set_ylabel(r'h', fontsize=20, labelpad=10)
    plt.savefig('./gw_strain.png', format='png')
    plt.close()

######## Execution ########
if __name__ == "__main__":
    main()
