from importlib import resources as impresources
from matplotlib import pyplot as plt
import numpy as np
from os.path import isfile, isdir, basename, expanduser, join, split
from mcfacts.inputs import data as mcfacts_input_data
from mcfacts.inputs.ReadInputs import ReadInputs_ini
from mcfacts.inputs.ReadInputs import load_disk_arrays, construct_disk_direct, construct_disk_pAGN
import argparse
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
from mcfacts.vis import data, plotting, styles

## Set plot style ##
plt.style.use('bmh')
plt.style.use("mcfacts.vis.mcfacts_figures")
size = "apj_col"


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname-ini", default="recipes/model_choice_old.ini")
    parser.add_argument("--outdir", default="./")
    opts = parser.parse_args()
    return opts


def ROS_plots():

    # Old global scope variables
    smbh_mass = 1e8
    disk_alpha_viscosity = 0.01
    disk_bh_eddington_ratio=0.5
    rad_efficiency = 0.1
    disk_model_name = 'sirko_goodman'
    disk_radius_outer = 50000

    # Begin
    disk_radius_arr, surface_density_arr, aspect_ratio_arr, opacity_arr = \
        load_disk_arrays(
            disk_model_name,
            disk_radius_outer,
            )
    surface_density_func, aspect_ratio_func, opacity_func, direct_model = \
        construct_disk_direct(
            disk_model_name,
            disk_radius_outer,
            )
    surface_density_func, aspect_ratio_func, opacity_func, pAGN_model, bonus_structures = \
        construct_disk_pAGN(
            disk_model_name,
            smbh_mass,
            disk_radius_outer,
            disk_alpha_viscosity,
            disk_bh_eddington_ratio,
            rad_efficiency,
            )
    rvals =        np.array( disk_radius_arr)
    indx_ok =  rvals < disk_radius_outer
    rvals = rvals[indx_ok]
    print('R', np.min(rvals), np.max(rvals))
    print('R_pagn', np.min(bonus_structures['R']), np.max(bonus_structures['R']))
    aspect1 = direct_model['h_over_r'](rvals)
    aspect2 = pAGN_model['h_over_r'](rvals)
    plt.scatter(np.log10(bonus_structures['R']), np.log10(bonus_structures['h_over_R']))
    plt.scatter(np.log10(disk_radius_arr), np.log10(aspect_ratio_arr))
    plt.plot(np.log10(rvals), np.log10(aspect1))
    plt.plot(np.log10(rvals), np.log10(aspect2))
    plt.savefig("aspect.png")
    plt.clf()
    plt.plot(np.log(rvals), (aspect1/aspect2))
    plt.savefig("aspect_ratio.png")



    plt.clf()
    Sigma1 = direct_model['Sigma'](rvals)
    Sigma2 = pAGN_model['Sigma'](rvals)
    print(Sigma1,Sigma2)
    plt.scatter(np.log10(rvals), np.log10(Sigma1))
    plt.plot(np.log10(rvals), np.log10(Sigma2))
    plt.savefig("Sigma.png")
    plt.clf()
    plt.plot(np.log(rvals), (Sigma1/Sigma2))
    plt.savefig("Sigma_ratio.png")

def pAGN_model_batch(
    disk_model_name="sirko_goodman",
    disk_radius_outer=50000,
    rad_efficiency=0.1,
    ):
    
    #disk_alpha_viscositys = np.asarray([0.1, 0.5])
    #edd_ratio = np.asarray([0.5,0.1])
    disk_alpha_viscositys = np.asarray([0.01,])
    edd_ratio = np.asarray([1.0])
    smbh_masses = 10**np.asarray([6,7,8,9])
    # Define dictionaries
    surface_density_dict = {}
    aspect_ratio_dict = {}
    opacity_dict = {}
    pAGN_model_dict = {}
    bonus_structures_dict = {}

    # Loop through pAGN models
    for _id_alpha, _alpha  in enumerate(disk_alpha_viscositys):
        for _id_edd,_edd in enumerate(edd_ratio):
            # Loop SMBH mass
            for _id_mass, _mass in enumerate(smbh_masses):
                # Identify pAGN model
                tag = "%s_%s_%s"%(str(_alpha),str(_edd),str(_mass))
                print("alpha:", _alpha)
                print("edd:", _edd)
                print("mass:",_mass)
                try:
                    # Generate model
                    surface_density_func, aspect_ratio_func, opacity_func, pAGN_model, bonus_structures = \
                        construct_disk_pAGN(
                            disk_model_name,
                            _mass,
                            disk_radius_outer,
                            _alpha,
                            _edd,
                            rad_efficiency,
                        )
                    # Add things to dictionary
                    surface_density_dict[tag] = surface_density_func
                    aspect_ratio_dict[tag] = aspect_ratio_func
                    opacity_dict[tag] = opacity_func
                    pAGN_model_dict[tag] = pAGN_model
                    bonus_structures_dict[tag] = bonus_structures
                except:
                    line = "Failed building pAGN disk model for "
                    line = line + "disk_alpha_viscosity = " + str(_alpha) + "; "
                    line = line + "edd = " + str(_edd) + "; "
                    line = line + "mass = " + str(_mass)
                    print(line)
            
    ### Density plot ###
    # Setup density plot
    fig, _axes = plt.subplots(
        figsize=(12,9),
        nrows=disk_alpha_viscositys.size,ncols=edd_ratio.size
        )
    plt.style.use('bmh')
    # Loop through pAGN models
    for _id_alpha, _alpha  in enumerate(disk_alpha_viscositys):
        for _id_edd,_edd in enumerate(edd_ratio):
            # Identify axis
            try:
                ax = _axes[_id_alpha, _id_edd]
            except:
                ax = _axes
            # Loop SMBH mass
            for _id_mass, _mass in enumerate(smbh_masses):
                # Identify pAGN model
                tag = "%s_%s_%s"%(str(_alpha),str(_edd),str(_mass))
                # Identify label
                if (_id_alpha == 0) and (_id_edd == 0):
                    label = "$M_{\mathrm{SMBH}} = 10^{%0.1f}$"%(np.log10(_mass))
                else:
                    label = None
                # Check if the pAGN model is really available
                if not (tag in surface_density_dict):
                    # If not, keep the legend consistent and move on
                    ax.plot([],[],label=label)
                    continue
                # Otherwise, plot things
                ax.plot(np.log10(bonus_structures_dict[tag]['R']),np.log10(bonus_structures_dict[tag]['rho']),label=label)

            # Vertical line for disk_radius_outer
            ylims = ax.get_ylim()
            ax.vlines(np.log10(disk_radius_outer), ylims[0],ylims[1], color='black', label="Truncate", linestyle='dashed')
            ax.set_ylim(ylims)

            # Generate a title
            title = r"$ \alpha = %0.3f; \mathrm{Edd} = %0.3f$"%(_alpha, _edd)
            ax.set_title(title)
            # Legend
            if (_id_alpha == 0) and (_id_edd == 0):
                ax.legend()

            # axis labels
            ax.set_xlabel(r"$\log_{10}(R)$")
            ax.set_ylabel(r"$\log_{10}(\rho)$")
    # show plots
    #plt.tight_layout()
    savename = "pAGN_%s_disk_rho.png"%(disk_model_name)
    fig.savefig(savename)
    #plt.show()
    plt.close()

    ### Temperature plot ###
    # Setup density plot
    fig, _axes = plt.subplots(
        figsize=(12,9),
        nrows=disk_alpha_viscositys.size,ncols=edd_ratio.size
        )
    plt.style.use('bmh')
    # Loop through pAGN models
    for _id_alpha, _alpha  in enumerate(disk_alpha_viscositys):
        for _id_edd,_edd in enumerate(edd_ratio):
            # Identify axis
            try:
                ax = _axes[_id_alpha, _id_edd]
            except:
                ax = _axes
            # Loop SMBH mass
            for _id_mass, _mass in enumerate(smbh_masses):
                # Identify pAGN model
                tag = "%s_%s_%s"%(str(_alpha),str(_edd),str(_mass))
                # Identify label
                if (_id_alpha == 0) and (_id_edd == 0):
                    label = "$M_{\mathrm{SMBH}} = 10^{%0.1f}$"%(np.log10(_mass))
                else:
                    label = None
                # Check if the pAGN model is really available
                if not (tag in surface_density_dict):
                    # If not, keep the legend consistent and move on
                    ax.plot([],[],label=label)
                    continue
                # Otherwise, plot things
                ax.plot(np.log10(bonus_structures_dict[tag]['R']),np.log10(bonus_structures_dict[tag]['T']),label=label)

            # Vertical line for disk_radius_outer
            ylims = ax.get_ylim()
            ax.vlines(np.log10(disk_radius_outer), ylims[0],ylims[1], color='black', label="Truncate", linestyle='dashed')
            ax.set_ylim(ylims)

            # Generate a title
            title = r"$ \alpha = %0.3f; \mathrm{Edd} = %0.3f$"%(_alpha, _edd)
            ax.set_title(title)
            # Legend
            if (_id_alpha == 0) and (_id_edd == 0):
                ax.legend()

            # axis labels
            ax.set_xlabel(r"$\log_{10}(R)$")
            ax.set_ylabel(r"$\log_{10}(T)$")
    # show plots
    #plt.tight_layout()
    savename = "pAGN_%s_disk_T.png"%(disk_model_name)
    fig.savefig(savename)
    #plt.show()
    plt.close()

    ### Opacity plot ###
    # Setup density plot
    fig, _axes = plt.subplots(
        figsize=(12,9),
        nrows=disk_alpha_viscositys.size,ncols=edd_ratio.size
        )
    plt.style.use('bmh')
    # Loop through pAGN models
    for _id_alpha, _alpha  in enumerate(disk_alpha_viscositys):
        for _id_edd,_edd in enumerate(edd_ratio):
            # Identify axis
            try:
                ax = _axes[_id_alpha, _id_edd]
            except:
                ax = _axes
            # Loop SMBH mass
            for _id_mass, _mass in enumerate(smbh_masses):
                # Identify pAGN model
                tag = "%s_%s_%s"%(str(_alpha),str(_edd),str(_mass))
                # Identify label
                if (_id_alpha == 0) and (_id_edd == 0):
                    label = "$M_{\mathrm{SMBH}} = 10^{%0.1f}$"%(np.log10(_mass))
                else:
                    label = None
                # Check if the pAGN model is really available
                if not (tag in surface_density_dict):
                    # If not, keep the legend consistent and move on
                    ax.plot([],[],label=label)
                    continue
                # Otherwise, plot things
                ax.plot(np.log10(bonus_structures_dict[tag]['R']),np.log10(bonus_structures_dict[tag]['tauV']),label=label)
                if (_id_alpha == 0) and (_id_edd == 0):
                    print(np.log10(bonus_structures_dict[tag]['tauV']))
                    tau_drop_mask = \
                        (np.log10(bonus_structures_dict[tag]['tauV']) < np.log10(bonus_structures_dict[tag]['tauV'][0])) & \
                        (np.log10(bonus_structures_dict[tag]['R']) > 3)
                    tau_drop_index = np.argmax(tau_drop_mask)
                    ylims = ax.get_ylim()
                    ax.vlines(np.log10(bonus_structures_dict[tag]['R'][tau_drop_index]),ylims[0],ylims[1],
                        linestyle="dashed",linewidth=1.5,color='black')
                    ax.set_ylim(ylims)
                    print(
                        np.log10(_mass),
                        np.log10(bonus_structures_dict[tag]['R'][tau_drop_index]),
                        np.log10(bonus_structures_dict[tag]['tauV'][tau_drop_index]),
                        )



            # Vertical line for disk_radius_outer
            ylims = ax.get_ylim()
            ax.vlines(np.log10(disk_radius_outer), ylims[0],ylims[1], color='black', label="Truncate", linestyle='solid')
            ax.set_ylim(ylims)

            # Generate a title
            title = r"$ \alpha = %0.3f; \mathrm{Edd} = %0.3f$"%(_alpha, _edd)
            ax.set_title(title)
            # Legend
            if (_id_alpha == 0) and (_id_edd == 0):
                ax.legend()

            # axis labels
            ax.set_xlabel(r"$\log_{10}(R)$")
            ax.set_ylabel(r"$\log_{10}(\tau_V)$")
    # show plots
    #plt.tight_layout()
    savename = "pAGN_%s_disk_tauV.png"%(disk_model_name)
    fig.savefig(savename)
    #plt.show()
    plt.close()

def plot_interpolators(fname_ini=None,output_directory="./"):
    ## Load input variables ##
    # Check if the user provided an  inifile
    if fname_ini is None:
        # Check that the data folder exists
        data_folder = impresources.files(mcfacts_input_data)
        assert isdir(data_folder), "Cannot find mcfacts.inputs.data folder"
        # Find model_choice.ini
        fname_ini = data_folder / "model_choice.ini"
    assert isfile(fname_ini), "Cannot find %s"%fname_ini
    # Get input variables
    input_variables = ReadInputs_ini(fname_ini)
    # Pull out disk_radius_outer
    disk_radius_outer = input_variables["disk_radius_outer"]

    # Assume pAGN off, and generate the interpolators
    direct_surf_dens_func, \
            direct_aspect_ratio_func, \
            direct_opacity_func, \
            sound_speed_func, \
            disk_density_func, \
            disk_pressure_grad_func, \
            disk_omega_func, \
            disk_surf_dens_func_log, \
            temp_func, \
            surf_dens_log10_derivative_func, \
            temp_log10_derivative_func, \
            pressure_log10_derivative_func, \
            direct_model_properties = \
        construct_disk_direct(
            input_variables["disk_model_name"],
            input_variables["disk_radius_outer"]
        )

    _alpha = input_variables["disk_alpha_viscosity"]
    _edd = input_variables["disk_bh_eddington_ratio"]
    #_edd = 0.5
    _mass = input_variables["smbh_mass"]
    # Identify pAGN model
    tag = "%s_%s_%s"%(str(_alpha),str(_edd),str(_mass))
    print("alpha:", _alpha)
    print("edd:", _edd)
    print("mass:",_mass)
    # Generate pAGN model
    pagn_surf_dens_func, \
            pagn_aspect_ratio_func, \
            pagn_opacity_func, \
            pagn_sound_speed_func, \
            pagn_disk_density_func, \
            pagn_disk_rpessure_grad_func, \
            pagn_disk_omega_func, \
            pagn_disk_surf_dens_func_log, \
            pagn_temp_func, \
            pagn_surf_dens_log10_derivative_func, \
            pagn_temp_log10_derivative_func, \
            pagn_pressure_log10_derivative_func, \
            pAGN_model, \
            bonus_structures = \
        construct_disk_pAGN(
            input_variables["disk_model_name"],
            _mass,
            disk_radius_outer,
            _alpha,
            _edd,
        )
            
    ### Density plot ###
    # Setup density plot
    fig, ax = plt.subplots(figsize=plotting.set_size(size))
    # Load pagn radius array
    pagn_disk_radius_array = bonus_structures['R']
    # Identify truncation mask for pAGN R
    pagn_disk_radius_mask = pagn_disk_radius_array < input_variables["disk_radius_outer"]
    # Identify the number of valid radius entrees
    pagn_disk_radius_length = np.sum(pagn_disk_radius_mask)
    # Identify log radius
    _ln_r = np.log(pagn_disk_radius_array)
    # Generate a test radius array
    disk_log_radius_test = np.linspace(
        _ln_r[pagn_disk_radius_mask][0],
        _ln_r[pagn_disk_radius_mask][-1],
        pagn_disk_radius_length * 10
    )
    # Generate a linear scale array
    disk_radius_test = np.exp(disk_log_radius_test)

    # Plot basic data
    ax.plot(
            np.log10(disk_radius_test),
            np.log10(direct_surf_dens_func(disk_radius_test)),
            label="McFacts disk interp",
    )
    # Plot pAGN interp
    ax.plot(
            np.log10(disk_radius_test),
            np.log10(pagn_surf_dens_func(disk_radius_test)),
            label="pAGN disk interp",
    )
    # Plot pAGN raw output
    ax.plot(
            np.log10(bonus_structures['R']),
            np.log10(bonus_structures['Sigma']),
            label="pAGN raw output",
            linestyle='dotted',
           )

    # Vertical line for disk_radius_outer
    ylims = ax.get_ylim()
    ax.vlines(
              np.log10(disk_radius_outer),
              ylims[0],ylims[1],
              color='black',
              label="disk_radius_outer",
              linestyle='dashed'
             )
    ax.set_ylim(ylims)

    # Set a legend
    ax.legend()

    # Limits
    ax.set_xlim([1.,6.])
    ax.set_ylim([2.,7.])
    # axis labels
    ax.set_xlabel(r"$\log_{10}(R)$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{Sigma})$")
    # show plots
    #plt.tight_layout()
    savename = f"{output_directory}/disk_interp_rho.png"
    fig.savefig(savename)
    #plt.show()
    plt.close()

def main():
    opts = arg()
    #ROS_plots()
    #pAGN_model_batch()
    plot_interpolators(opts.fname_ini,opts.outdir)

if __name__ == "__main__":
    main()
