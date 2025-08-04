#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

#SI units
Msun=1.99e30 #kg per solar mass
Rsun=6.95e8 #meters per solar radius
G=6.67e-11
c=3e8
sigma_SB=5.7e-8 #stefan-boltzmann const
yr=3.15e7 #seconds per year
pc=3.086e16 #meters per parsec
AU=1.496e11 #meters per AU
h=6.626e-34 #planck const
kB=1.38e-23 #boltzmann const
m_p=1.67e-27 #mass of proton
sigma_T=6.65e-29 #Thomson xsec
PI=3.1415926
m_per_nm=1.0e-9

def main():
    """Dumb plotting routine for CQG paper, params for AGN fraction from Ueda++14
    """
    zc1_45=1.86
    zc1_44=1.85
    zc1_43=1.84
    p1_45=5.62
    p1_44=4.78
    p1_43=3.94
    
    # set redshift range & phi arrays
    z = np.arange(0.0, 2.0, 0.01)
    phi_45 = np.zeros(len(z))
    phi_44 = np.zeros(len(z))
    phi_43 = np.zeros(len(z))
    
    for i in range(len(z)):
        if (z[i]<=zc1_45):
                phi_45[i] = pow(1.0+z[i], p1_45)
        else:
                phi_45[i] = pow((1.0+zc1_45), p1_45) * pow(((1.0+z[i])/(1.0+zc1_45)), -1.5)
        if (z[i]<=zc1_44):
                phi_44[i] = pow(1.0+z[i], p1_44)
        else:
                phi_44[i] = pow((1.0+zc1_44), p1_44) * pow(((1.0+z[i])/(1.0+zc1_44)), -1.5)
        if (z[i]<=zc1_43):
                phi_43[i] = pow(1.0+z[i], p1_43)
        else:
                phi_43[i] = pow((1.0+zc1_43), p1_43) * pow(((1.0+z[i])/(1.0+zc1_43)), -1.5)

    # madau-dickinson SFR (eqn 15)
    psi = pow(1.0 + z, 2.7)

    # LIGO dependence per GWTC-3 rates & pops paper
    R_LIGO = pow(1.0 + z, 2.9)

    # Antonini, Silk & Barausse show M(NSC) increasing or decreasing by ~10x
    #   between z=0-2, for combined NSC+MBH systems so
    log_ASB_M_NSC_dec = 0.0 + (0.5 * z)
    log_ASB_M_NSC_inc = 1.0 - (0.5 * z)

    # Generozov++ shows little variation, up or down by factor of 2 at most
    log_gen_M_NSC_dec = 0.0 + (0.15 * z)
    log_gen_M_NSC_inc = 0.5 - (0.15 * z)

    #BEGIN DISPLAY INPUTS:
    #divide display box for graphs
    #params for axes
    #format=left, bottom, width, height
    #rect1=0.1,0.1,0.75,0.75
    
    #make figure
    fig=plt.figure()

    # create bottom figure
    ax1 = plt.subplot(2,1,2)
    #add axes & label them
    plt.xlim(0.0,2.0)
    plt.ylim(0.9,3.8)
    plt.ylabel(r"$log(\phi)$")
    plt.xlabel(r"$z$")
    
    stylecycle=['solid', 'dashed', 'dotted']
    
    plt.plot(z, np.log10(phi_45)+log_ASB_M_NSC_inc, color='k', ls=stylecycle[0], linewidth=2, label='45')
    plt.plot(z, np.log10(phi_44)+log_ASB_M_NSC_inc, color='k', ls=stylecycle[1], linewidth=2, label='44')
    plt.plot(z, np.log10(phi_43)+log_ASB_M_NSC_inc, color='k', ls=stylecycle[2], linewidth=2, label='43')
    plt.plot(z, np.log10(psi)+1.0, color='b', ls=stylecycle[0], linewidth=2, label='SFR')
    plt.plot(z, np.log10(R_LIGO)+1.0, color='r', ls=stylecycle[0], linewidth=2, label='LVK Rate')

    # create top figure
    ax2 = plt.subplot(2,1,1)

    plt.ylabel(r"$log(\phi)$")
    plt.xlim(0.0,2.0)
    plt.ylim(0.9,3.8)

    plt.plot(z, np.log10(phi_45)+1.0, color='k', ls=stylecycle[0], linewidth=2, label='45')
    plt.plot(z, np.log10(phi_44)+1.0, color='k', ls=stylecycle[1], linewidth=2, label='44')
    plt.plot(z, np.log10(phi_43)+1.0, color='k', ls=stylecycle[2], linewidth=2, label='43')
    plt.plot(z, np.log10(psi)+1.0, color='b', ls=stylecycle[0], linewidth=2, label='SFR')
    plt.plot(z, np.log10(R_LIGO)+1.0, color='r', ls=stylecycle[0], linewidth=2, label='LVK Rate')

    # create legend
    plt.legend(title=r'$log(L_{X})$', loc='upper left')

    # suppress the x axis labels for the top plot, and kill the whitespace
    ax2.set_xticklabels([])
    fig.subplots_adjust(hspace=0)
    
    plt.savefig('rates_vs_z.png')
    plt.savefig('rates_vs_z.pdf')



if __name__ == "__main__":
    main()
