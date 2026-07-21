"""Modify a SettingsManager to participate in scaled runs"""

######## Imports ########
#### Standard Library ####
#### Third Party ####
import numpy as np
#### Local ####
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.disk import AGNDisk

######## Functions ########
def disk_truncation(settings: SettingsManager): 
    """Truncate the disk based on properties of the SettingsManager"""
    # Check that the user asked for this
    if settings.disk_truncation == "none":
        return
    # Setup an AGNDisk
    TempDisk = AGNDisk(settings)

    if settings.disk_truncation == "opacity-equals-inner-disk":
        # Make sure pAGN is enabled
        if not settings.flag_use_pagn:
            raise NotImplementedError
        # Load R and tauV
        pagn_R = TempDisk.pagn_bonus_structures["R"]
        pagn_tauV = TempDisk.pagn_bonus_structures["tauV"]
        # Find where tauV is greater than its initial value
        tau_drop_mask = (pagn_tauV < pagn_tauV[0]) & (np.log10(pagn_R) > 3)
        # Find the drop index
        tau_drop_index = np.argmax(tau_drop_mask)
        # Find the drop radius
        tau_drop_radius = pagn_R[tau_drop_index]
        # Update settings
        print(settings.disk_radius_outer)
        settings.disk_radius_outer = tau_drop_radius
        print(settings.disk_radius_outer)

def stellar_mass_scaling(settings: SettingsManager):
    """Scale the disk by stellar mass relations"""

    ## SMBH mass relation ##
    #Schramm-Silverman
    if settings.scale_stellar_mass == "schramm-silverman":
        # Check if stellar mass was provided
        if settings.stellar_mass == 0.:
            # Stellar mass was not provided generate it.
            raise NotImplementedError
        else:
            # Stellar mass was provided. Overwrite smbh mass scaling
            raise NotImplementedError
    # None
    elif settings.scale_stellar_mass == "none":
        # The user clearly doesn't want us to overwrite the smbh or stellar mass
        pass
    else:
        raise ValueError(
            settings.scale_stellar_mass + " "
            "is not a valid option for scale_stellar_mass. "
            "Valid options include ['none', 'schramm-silverman']."
        )


    ## NSC mass relation ##
    if settings.scale_nsc_mass == "none":
        # The user clearly doesn't want us to overwrite the nsc mass
        pass
    elif settings.stellar_mass == 0.:
        raise ValueError(
            f"scale_nsc_mass is {settings.scale_nsc_mass}, but "
            "stellar_mass is zero. "
            "Set scale_nsc_mass to 'none' to keep NSC mass."
        )
    elif settings.scale_nsc_mass == "neumayer-early":
        raise NotImplementedError
    elif settings.scale_nsc_mass == "neumayer-late":
        raise NotImplementedError
    else:
        raise ValueError(
            settings.scale_nsc_mass + " "
            "is not a valid option for scale_nsc_mass. "
            "Valid options include ['none', 'neumayer-early', 'neumayer-late']."
        )

def setup_scaling(settings: SettingsManager):
    """Scale the AGN Disk according to settings"""
    # truncate disk
    disk_truncation(settings)
    # Scale masses
    stellar_mass_scaling(settings)
    return None
    # Rescale inner_disk_outer_radius
    # rescale 
    t_gw_inner_disk = time_of_orbital_shrinkage(
        smbh_mass_fiducial,
        test_mass,
        inner_disk_outer_radius_fiducial,
        0. * u.m,
    )
    # Find the new inner_disk_outer_radius
    new_inner_disk_outer_radius = orbital_separation_evolve_reverse(
        mcfacts_input_variables["smbh_mass"] * u.solMass,
        test_mass,
        0 * u.m,
        t_gw_inner_disk,
    )
    # Estimate in r_g
    new_inner_disk_outer_radius_r_g = r_g_from_units(
        mcfacts_input_variables["smbh_mass"] * u.solMass,
        new_inner_disk_outer_radius,
    )
    cmd=f"sed --in-place 's/inner_disk_outer_radius =.*/inner_disk_outer_radius = {new_inner_disk_outer_radius_r_g}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)

    # Estimate new trap radius
    new_trap_radius = mcfacts_input_variables["disk_radius_trap"] * np.sqrt(
        smbh_mass_fiducial /
        (mcfacts_input_variables["smbh_mass"] * u.solMass)
    ) 
    cmd=f"sed --in-place 's/disk_radius_trap =.*/disk_radius_trap = {new_trap_radius}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)
    
    # Estimate a new capture radius
    new_capture_radius = mcfacts_input_variables["disk_radius_capture_outer"] * np.sqrt(
        smbh_mass_fiducial /
        (mcfacts_input_variables["smbh_mass"] * u.solMass)
    )
    cmd=f"sed --in-place 's/disk_radius_capture_outer =.*/disk_radius_capture_outer = {new_capture_radius}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)
    # Estimate the agn lifetime
    agn_lifetime = mcfacts_input_variables["timestep_duration_yr"] * u.yr * \
        mcfacts_input_variables["timestep_num"]
    # Estimate the capture time
    t_capture = capture_time(
        mcfacts_input_variables["smbh_mass"] * u.solMass,
        mcfacts_input_variables["nsc_mass"] * u.solMass,
        agn_lifetime,
        disk_surf_dens_func,
        mcfacts_input_variables["nsc_ratio_bh_num_star_num"],
        mcfacts_input_variables["nsc_ratio_bh_mass_star_mass"],
    )
    print(f"smbh_mass: {mcfacts_input_variables['smbh_mass'] * u.solMass}")
    print(f"smbh_mass: {smbh_mass}")
    print(f"capture time: {t_capture}")
    # velocity dispersion of nsc
    cmd=f"sed --in-place 's/capture_time_yr =.*/capture_time_yr = {t_capture.to('yr').value}/' {fname_ini_local}"
    print(cmd)
    if not opts.print_only:
        os.system(cmd)

    # Open script
    mcfacts_script = fname_ini_local.rstrip("ini") + "sh"
    # Identify output file
    mcfacts_out = fname_ini_local.rstrip("ini") + "out"
    with open(mcfacts_script, 'w') as F:
        # Make all iterations
        cmd = "python3 %s --fname-ini %s --work-directory %s > %s\n"%(
            os.path.abspath(opts.mcfacts_exe), fname_ini_local, wkdir, mcfacts_out)
        print(cmd)
        F.writelines(cmd)
        # Make plots for all iterations
        cmd = "python3 %s --fname-mergers %s/output_mergers_population.dat --fname-nal %s --cdf bin_com chi_eff final_mass time_merge\n"%(
            opts.vera_plots_exe, wkdir, opts.fname_nal)
        print(cmd)
        F.writelines(cmd)
        # Make disk plots
        cmd = f"python3 {opts.plot_disk_exe} --fname-ini {fname_ini_local} --outdir {wkdir}\n"
        print(cmd)
        F.writelines(cmd)

    # Scrub runs
    if opts.scrub:
        cmd = "rm -rf %s/run*"%wkdir
        print(cmd)
        os.system(cmd)

