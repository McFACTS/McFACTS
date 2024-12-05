"""
Module for calculating corrections to migration due to feedback models.
"""
import numpy as np


def feedback_bh_hankla(disk_bh_pro_orbs_a, disk_surf_density_func, disk_opacity_func, disk_bh_eddington_ratio, disk_alpha_viscosity, disk_radius_outer):
    """Calculate the ratio of radiative feedback torque to migration torque.

    This feedback model uses Eqn. 28 in Hankla, Jiang & Armitage (2020)
    which yields the ratio of heating torque to migration torque.
    Heating torque is directed outwards.
    So, Ratio < 1, slows the inward migration of an object. Ratio > 1 sends the object migrating outwards.
    The direction & magnitude of migration (effected by feedback) will be executed in type1.py.

    Parameters
    ----------
    disk_bh_pro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_opacity_model : lambda
        Opacity as a function of radius
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    disk_alpha_viscosity : float
        Disk gas viscocity [units??] alpha parameter
    disk_radius_outer : float
            Outer radius [r_{g,SMBH}] of the disk

    Returns
    -------
    ratio_feedback_migration_torque : numpy.ndarray
        Ratio of feedback torque to migration torque with :obj:`float` type

    Notes
    -----
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
    """

    # get disk surface density at black hole orbital semi-major axes
    disk_surface_density = disk_surf_density_func(disk_bh_pro_orbs_a)

    #Define kappa (or set up a function to call).
    disk_opacity = disk_opacity_func(disk_bh_pro_orbs_a)

    ratio_feedback_migration_torque = 0.07 * (1/disk_opacity) * (disk_alpha_viscosity ** -1.5) * \
                                      disk_bh_eddington_ratio * np.sqrt(disk_bh_pro_orbs_a) / disk_surface_density

    # set ratio = 1 (no migration) for black holes beyond the disk outer radius
    ratio_feedback_migration_torque[np.where(disk_bh_pro_orbs_a > disk_radius_outer)] = 1

    return ratio_feedback_migration_torque


def feedback_stars_hankla(disk_stars_pro_orbs_a, disk_surf_density_func, disk_opacity_func, disk_stars_eddington_ratio, disk_alpha_viscosity, disk_radius_outer):
    """Calculate the ratio of radiative feedback torque to migration torque.

    This feedback model uses Eqn. 28 in Hankla, Jiang & Armitage (2020)
    which yields the ratio of heating torque to migration torque.
    Heating torque is directed outwards.
    So, Ratio < 1, slows the inward migration of an object. Ratio > 1 sends the object
    migrating outwards.
    The direction & magnitude of migration (effected by feedback) will be executed in type1.py.

    Parameters
    ----------
    disk_bh_pro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_opacity_model : lambda
        Opacity as a function of radius
    disk_bh_eddington_ratio : float
        Accretion rate of fully embedded stellar mass black hole [Eddington accretion rate].
        1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
        User chosen input set by input file
    disk_alpha_viscosity : float
        Disk gas viscocity [units??] alpha parameter
    disk_radius_outer : float
            Outer radius [r_{g,SMBH}] of the disk


    Returns
    -------
    ratio_feedback_to_mig : numpy.ndarray
        Ratio of feedback torque to migration torque with :obj:`float` type

    Notes
    -----
    The ratio of torque due to heating to Type 1 migration torque is calculated as
    R   = Gamma_heat/Gamma_mig
        ~ 0.07 (speed of light/ Keplerian vel.)(Eddington ratio)(1/optical depth)(1/alpha)^3/2
    where Eddington ratio can be >=1 or <1 as needed,
    optical depth (tau) = Sigma* kappa
    alpha = disk viscosity parameter (e.g. alpha = 0.01 in Sirko & Goodman 2003)
    kappa = 10^0.76 cm^2 g^-1=5.75 cm^2/g = 0.575 m^2/kg for most of Sirko & Goodman
      disk model (see Fig. 1 & sec 2)
    but e.g. electron scattering opacity is 0.4 cm^2/g
    So tau = Sigma*0.575 where Sigma is in kg/m^2.
    Since v_kep = c/sqrt(a(r_g)) then
    R   ~ 0.07 (a(r_g))^{1/2}(Edd_ratio) (1/tau) (1/alpha)^3/2
    So if assume a=10^3r_g, Sigma=7.e6kg/m^2, alpha=0.01, tau=0.575*Sigma (SG03 disk model),
      Edd_ratio=1,
    R   ~5.5e-4 (a/10^3r_g)^(1/2) (Sigma/7.e6) v.small modification to in-migration at a=10^3r_g
        ~0.243 (R/10^4r_g)^(1/2) (Sigma/5.e5)  comparable.
        >1 (a/2x10^4r_g)^(1/2)(Sigma/) migration is *outward* at >=20,000r_g in SG03
        >10 (a/7x10^4r_g)^(1/2)(Sigma/) migration outwards starts to runaway in SG03
    """

    # get disk surface density at black hole orbital semi-major axes
    disk_surface_density = disk_surf_density_func(disk_stars_pro_orbs_a)

    #Define kappa (or set up a function to call).
    disk_opacity = disk_opacity_func(disk_stars_pro_orbs_a)

    ratio_feedback_migration_torque = 0.07 * (1/disk_opacity) * (disk_alpha_viscosity ** -1.5) * \
                                      disk_stars_eddington_ratio * np.sqrt(disk_stars_pro_orbs_a) / disk_surface_density

    # set ratio = 1 (no migration) for stars beyond the disk outer radius
    ratio_feedback_migration_torque[np.where(disk_stars_pro_orbs_a > disk_radius_outer)] = 1

    return ratio_feedback_migration_torque
