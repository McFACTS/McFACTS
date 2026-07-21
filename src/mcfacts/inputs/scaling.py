"""Modify a SettingsManager to participate in scaled runs"""

######## Imports ########
#### Standard Library ####
import warnings
#### Third Party ####
import numpy as np
from astropy import units as u
from astropy import constants as const
#### Local ####
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.disk import AGNDisk
from mcfacts.utilities import unit_conversion, peters
from mcfacts.modules.gw import orbital_separation_evolve_reverse

######## Setup ########
SMBH_MASS_FIDUCIAL = 1e8 # solar masses
TEST_MASS = 10 # solar masses
INNER_DISK_OUTER_RADIUS_FIDUCIAL = unit_conversion.si_from_r_g(
        SMBH_MASS_FIDUCIAL,
        50.,
) #50 r_g

######## Functions ########
def disk_truncation(
        settings: SettingsManager,
        return_disk = False,
    ): 
    """Truncate the disk based on properties of the SettingsManager"""
    # Check that the user asked for this
    if settings.disk_truncation.lower() == "none":
        if return_disk: return AGNDisk(settings)
        else: return
    
    if settings.verbose:
        print(f"Truncating disk!")
        print(f"settings.disk_truncation: {settings.disk_truncation}.")
        print(
            f"settings.disk_radius_outer (before): "
            f"{settings.disk_radius_outer} (r_g)."
        )

    # Setup an AGNDisk
    TempDisk = AGNDisk(settings)

    # Paper III truncation
    if settings.disk_truncation.lower() == "opacity-equals-inner-disk":
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
        settings.disk_radius_outer = tau_drop_radius
    else:
        raise ValueError(
            f"disk_truncation set to {settings.disk_truncation}; "
            "Invalid value."
        )

    # Update the user
    if settings.verbose:
        print(
            f"settings.disk_radius_outer (before): "
            f"{settings.disk_radius_outer} (r_g)."
        )

    if return_disk:
        return TempDisk

def scale_galaxy_mass(settings: SettingsManager):
    """Scale the disk by stellar mass relations

    Citations
    --------
        Schramm-Silverman:
        https://iopscience.iop.org/article/10.1088/0004-637X/767/1/13/pdf
        Neumayer Early (Eq. 1) / Late (Eq. 2):
        https://link.springer.com/article/10.1007/s00159-020-00125-0

    """
    if settings.verbose:
        print("Scaling galaxy mass!")
        print(f"settings.scale_smbh_mass: {settings.scale_smbh_mass}")
        print(f"settings.scale_nsc_mass: {settings.scale_nsc_mass}")
        print(f"settings.smbh_mass (before): {settings.smbh_mass:.3e}")
        print(f"settings.nsc_mass (before): {settings.nsc_mass:.3e}")
        print(f"settings.stellar_mass (before): {settings.stellar_mass:.3e}")

    ## SMBH mass relation ##
    # None
    if settings.scale_smbh_mass.lower() == "none":
        # The user clearly doesn't want us to overwrite the smbh or stellar mass
        pass
    #Schramm-Silverman
    elif settings.scale_smbh_mass.lower() == "schramm-silverman":
        # Check if stellar mass was provided
        if settings.stellar_mass == 0.:
            # Stellar mass was not provided generate it.
            settings.stellar_mass = (settings.smbh_mass / 7.066429e-05) ** (1/1.12)
        else:
            # Stellar mass was provided. Overwrite smbh mass scaling
            settings.smbh_mass = 7.066429e-05 * settings.stellar_mass**1.12
    else:
        raise ValueError(
            settings.scale_smbh_mass + " "
            "is not a valid option for scale_smbh_mass. "
            "Valid options include ['none', 'schramm-silverman']."
        )


    ## NSC mass relation ##
    if settings.scale_nsc_mass.lower() == "none":
        # The user clearly doesn't want us to overwrite the nsc mass
        pass
    elif settings.stellar_mass == 0.:
        raise ValueError(
            f"scale_nsc_mass is {settings.scale_nsc_mass}, but "
            "stellar_mass is zero. "
            "Set scale_nsc_mass to 'none' to keep NSC mass."
        )
    elif settings.scale_nsc_mass.lower() == "neumayer-early":
        settings.nsc_mass = 3235936.569296281 * (settings.stellar_mass * 1e-9)**0.48
    elif settings.scale_nsc_mass.lower() == "neumayer-late":
        if settings.stellar_mass < 1e9:
            warnings.warn(
                "Neumayer-late is a poor approximation for galaxies with "
                f"stellar mass below 10^9. "
                f"(current value: {settings.stellar_mass:.3e})."
            )
        settings.nsc_mass = 1348962.8825916534 * (settings.stellar_mass * 1e-9)**0.92
    else:
        raise ValueError(
            settings.scale_nsc_mass + " "
            "is not a valid option for scale_nsc_mass. "
            "Valid options include ['none', 'neumayer-early', 'neumayer-late']."
        )

    if settings.verbose:
        print(f"settings.smbh_mass (after): {settings.smbh_mass:.3e}")
        print(f"settings.nsc_mass (after): {settings.nsc_mass:.3e}")
        print(f"settings.stellar_mass (after): {settings.stellar_mass:.3e}")

def scale_inner_disk(settings: SettingsManager):
    """Scale the inner disk

    We decided to do this in paper 3. There's no other citation.
    """

    if settings.scale_inner_disk == "none":
        return

    if settings.verbose:
        print("Scaling the inner disk!")
        print(f"settings.scale_inner_disk: {settings.scale_inner_disk}")
        print(f"settings.inner_disk_outer_radius (before): {settings.inner_disk_outer_radius}")

    ## Paper III method: scale by constant decay time
    if settings.scale_inner_disk == "decay-time":
        t_gw_inner_disk = peters.time_of_orbital_shrinkage(
            SMBH_MASS_FIDUCIAL * u.solMass,
            TEST_MASS * u.solMass,
            INNER_DISK_OUTER_RADIUS_FIDUCIAL,
            0. * u.m,
        )
        # Find the new inner_disk_outer_radius
        new_inner_disk_outer_radius = orbital_separation_evolve_reverse(
            settings.smbh_mass * u.solMass,
            TEST_MASS * u.solMass,
            0 * u.m,
            t_gw_inner_disk,
        )
        # Estimate in r_g
        new_inner_disk_outer_radius_r_g = unit_conversion.r_g_from_units(
            settings.smbh_mass * u.solMass,
            new_inner_disk_outer_radius,
        )
        # re-assign
        settings.inner_disk_outer_radius = new_inner_disk_outer_radius_r_g
    else:
        raise ValueError(
            f"{settings.scale_inner_disk} is not a valid choice for "
            "scale_inner_disk. Please select from "
            "['none', 'decay-time']."
        )

    if settings.verbose:
        print(f"settings.inner_disk_outer_radius (before): {settings.inner_disk_outer_radius}")

def scale_trap(settings: SettingsManager):
    """Scale the trap radius

    We decided to do this in paper 3. There's no other citation.
    """
    if settings.scale_trap == "none":
        return

    if settings.verbose:
        print(f"Scaling the migration trap!")
        print(f"settings.scale_trap: {settings.scale_trap}")
        print(f"settings.disk_radius_trap (before): {settings.disk_radius_trap} (r_g)")

    if settings.scale_trap == "sqrt-smbh":
        # Estimate new trap radius
        settings.disk_radius_trap = settings.disk_radius_trap * np.sqrt(
            SMBH_MASS_FIDUCIAL / settings.smbh_mass
        )
    else:
        raise ValueError(
            f"{settings.scale_trap} is not a valid choice for "
            "scale_trap. Please select from "
            "['none', 'sqrt-smbh']."
        )
        SettingsProperty("scale_trap", "scale", "sqrt-smbh", str),
        SettingsProperty("scale_capture", "scale", "sqrt-smbh", str),
        SettingsProperty("scale_capture", "scale", "hubble", str),

    if settings.verbose:
        print(f"settings.disk_radius_trap (after): {settings.disk_radius_trap} (r_g)")

def WZL_2024_capture_time(
        agn_disk : AGNDisk,
        smbh_mass,
        nsc_mass,
        agn_lifetime,
        nsc_ratio_bh_num_star_num,
        nsc_ratio_bh_mass_star_mass,
        m_bh = 10 * u.solMass,
        verbose=True,
    ):
    """Wang, Zhu, and Lin (2024) capture times
    """
    if verbose:
        print(f"smbh_mass: {smbh_mass}")
        print(f"nsc_mass: {nsc_mass}")
        print(f"agn_lifetime: {agn_lifetime}")
        print(f"nsc_ratio_bh_num_star_num: {nsc_ratio_bh_num_star_num}")
        print(f"nsc_ratio_bh_mass_star_mass: {nsc_ratio_bh_mass_star_mass}")
        print(f"m_bh: {m_bh}")
    # velocity dispersion of nsc
    sig_nsc = (2.3*(u.km/u.s)) * \
        (smbh_mass / (1 * u.solMass))**(1./4.38)
    if verbose:
        print(f"sig_nsc: {sig_nsc}")
    # Calculate radius of influence
    r_infl = const.G * smbh_mass / sig_nsc**2
    r_infl = r_infl.to("pc")
    if verbose:
        print(f"r_infl: {r_infl}")
    # Calculate radius_of_influence in r_g
    r_infl_g = unit_conversion.r_g_from_units(
        smbh_mass,
        r_infl
    )
    if verbose:
        print(f"r_infl_g: {r_infl_g}")
    # Calculate orbital time at radius of influence
    p_orb_r_infl = 2 * np.pi * np.sqrt(r_infl**3 / (const.G * smbh_mass))
    p_orb_r_infl = p_orb_r_infl.si
    if verbose:
        print(f"p_orb_r_infl: {p_orb_r_infl}")
    # Calculate the surface density at the radius of influence
    Sigma_m = agn_disk.surface_density(r_infl_g) * u.kg / u.m**2
    if verbose:
        print(f"Sigma_m: {Sigma_m}")
    if not np.isfinite(Sigma_m):
        return cosmo.hubble_time
    # Total mass of BH in NSC
    total_mass_bh_in_nsc = nsc_mass * nsc_ratio_bh_num_star_num * nsc_ratio_bh_mass_star_mass
    if verbose:
        print(f"total_mass_bh_in_nsc: {total_mass_bh_in_nsc}")
    # Mass fraction of BH in NSC
    f_bh = total_mass_bh_in_nsc / nsc_mass
    if verbose:
        print(f"f_bh: {f_bh}")
    # Capture mass
    captured_mass = (2. * smbh_mass * f_bh * \
        (m_bh / smbh_mass) * \
        (Sigma_m *np.pi * r_infl**2 / smbh_mass) *\
        (agn_lifetime / p_orb_r_infl) \
    ).to("Msun")
    if verbose:
        print(f"captured_mass: {captured_mass}")
    # Calculate number of bh in agn lifetime
    n_bh_capture = (captured_mass / m_bh)
    if verbose:
        print(f"n_bh_capture: {n_bh_capture}")
    # Capture time
    t_capture = agn_lifetime / n_bh_capture
    if verbose:
        print(f"t_capture: {t_capture}")
        print(f"log_10(t_capture [yr]): {np.log10(t_capture.to('yr').value)}")
    return t_capture

def scale_capture(
        settings: SettingsManager,
        agn_disk: AGNDisk,
    ):
    """Scale the capture radius and timescale"""
    ## Scale capture radius ##
    if settings.verbose:
        print("Scaling capture radius!")
        print(
            f"settings.scale_capture_radius: {settings.scale_capture_radius}"
        )
        print(
            f"settings.disk_radius_capture_outer (before): "
            f"{settings.disk_radius_capture_outer} (r_g)"
        )
    # Check scaling method
    if settings.scale_capture_radius == "none":
        pass
    elif settings.scale_capture_radius == "sqrt-smbh":
        # Estimate a new capture radius
        settings.disk_radius_capture_outer = \
            settings.disk_radius_capture_outer * np.sqrt(
                SMBH_MASS_FIDUCIAL / settings.smbh_mass
            )
    else:
        raise ValueError(
            f"Invalid value for scale_capture_radius: "
            f"{settings.scale_capture_radius}:"
        )

    # Update user
    if settings.verbose:
        print(
            f"settings.disk_radius_capture_outer (after): "
            f"{settings.disk_radius_capture_outer} (r_g)"
        )

    ## Scale capture time ##
    if settings.verbose:
        print("Scaling capture time!")
        print(f"settings.scale_capture_time: {settings.scale_capture_time}")
        print(f"settings.capture_time_yr (before): {settings.capture_time_yr}")

    # Check method
    if settings.scale_capture_time == "none":
        pass
    elif settings.scale_capture_time == "hubble":
        settings.capture_time_yr = 1.38e10
    elif settings.scale_capture_time == "WZL+2024":
        pass
        # Estimate the agn lifetime
        agn_lifetime = settings.active_timestep_duration_yr * u.yr * \
            settings.active_timestep_num
        # Estimate the capture time
        t_capture = WZL_2024_capture_time(
            agn_disk,
            settings.smbh_mass * u.solMass,
            settings.nsc_mass * u.solMass,
            agn_lifetime,
            settings.nsc_ratio_bh_num_star_num,
            settings.nsc_ratio_bh_mass_star_mass,
            verbose=settings.verbose,
        )
        # Check units
        if hasattr(t_capture, 'unit'):
            t_capture = t_capture.to(u.yr).value
        # Update settings
        settings.capture_time_yr = t_capture
    else:
        raise ValueError(
            f"Invalid value for scale_capture_time: "
            f"{settings.scale_capture_time}"
        )

    # Update user
    if settings.verbose:
        print("Scaling capture time!")
        print(f"settings.scale_capture_time: {settings.scale_capture_time}")
        print(f"settings.capture_time_yr (before): {settings.capture_time_yr}")
        

def setup_scaling(
        settings: SettingsManager,
    ):
    """Scale the AGN Disk according to settings"""
    # Scale masses
    scale_galaxy_mass(settings)
    # truncate disk
    agn_disk = disk_truncation(settings, return_disk=True)
    # Scale inner disk
    scale_inner_disk(settings)
    # Scale migration trap
    scale_trap(settings)
    # Scale captures
    scale_capture(settings, agn_disk)
