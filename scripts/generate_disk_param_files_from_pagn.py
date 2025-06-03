from pagn import Thompson
from pagn import Sirko
import pagn.constants as pagn_ct
from astropy import constants as ct
from astropy import units as u
from mcfacts.physics import point_masses
import pandas as pd
import scipy.interpolate
import numpy as np

"""Creates disk parameter files using pAGN for when running McFACTS with pAGN off

Each file is created for the Sirko-Goodman and TQM disks, using the default parameters in the ini files.
- Sound speed (c_s = H * Omega)
- Disk density (rho)
- Disk pressure gradient (dP/dR)
- Disk temperature (T)
- Disk Omega (omega)
"""


def func_to_vstack(func, radius):
    # Apply function to given radius
    param = func(radius)
    # Mask out NaN and inf values
    mask = np.isfinite(param)
    param = param[mask]
    radius = radius[mask]
    # Make vstack
    stack = np.vstack((param, radius)).T
    return (stack)


smbh_mass = 1.e8

#### -------------------- SIRKO GOODMAN DISK --------------------

# Sirko & Goodman 2003
sk_Mbh = smbh_mass*pagn_ct.MSun  # 10^8 solar mass SMBH
sk_le = 1.0  # disk_bh_eddington_ratio
sk_alpha = 0.01  # disk_alpha_viscosity
rad_efficiency = 0.1  # from ReadInputs

# Create disk model
sk = Sirko.SirkoAGN(Mbh=sk_Mbh, le=sk_le, alpha=sk_alpha, eps=rad_efficiency)
sk.solve_disk(N=1e4)  # 10^4 tends to be a sufficient resolution for most Mbh values

# Read in existing file to get radii
sk_aspect = pd.read_csv("src/mcfacts/inputs/data/sirko_goodman_aspect_ratio.txt",
                        names=['aspect_ratio','radius'], skiprows=1, sep=" ")

# Create interpolator for sound speed (c_s = H * Omega)
sk_R = sk.R / (sk.Rs / 2)
sk_cs = sk.h * sk.Omega
sk_ln_cs = np.log(sk_cs)
sk_cs_func_log = scipy.interpolate.CubicSpline(np.log(sk_R), sk_ln_cs, extrapolate=False)
sk_cs_func = lambda x, f=sk_cs_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
sk_data_cs = func_to_vstack(sk_cs_func, sk_aspect['radius'])

# Create interpolator for disk density (rho)
sk_ln_rho = np.log(sk.rho)
sk_rho_func_log = scipy.interpolate.CubicSpline(np.log(sk_R), sk_ln_rho, extrapolate=False)
sk_rho_func = lambda x, f=sk_rho_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
sk_data_rho = func_to_vstack(sk_rho_func, sk_aspect['radius'])

# Create interpolator for disk pressure gradient (dP/dR)
sk_pgas = sk.rho * sk.T * pagn_ct.Kb / pagn_ct.massU
sk_prad = sk.tauV * pagn_ct.sigmaSB * sk.Teff4 / (2 * pagn_ct.c)
sk_ptot = sk_pgas + sk_prad
sk_disk_pressure_grad_func_interp = scipy.interpolate.CubicSpline(
                                                        sk.R,
                                                        np.gradient(sk_ptot)/np.gradient(sk.R),
                                                        extrapolate=False)
sk_disk_pressure_grad_func = lambda x, f=sk_disk_pressure_grad_func_interp: f(point_masses.si_from_r_g(sk.Mbh * u.kg, x).value)
# Apply to radius and put in vstack form
sk_data_dpdr = func_to_vstack(sk_disk_pressure_grad_func, sk_aspect['radius'].values)

# Create interpolator for temperature
sk_temp_midplane = sk.T  # Disk midplane temp (K)
sk_ln_temp_midplane = np.log(sk_temp_midplane)  # ln midplane temp.
sk_temp_func_log = scipy.interpolate.CubicSpline(
                                                    np.log(sk_R),
                                                    sk_ln_temp_midplane,
                                                    extrapolate=False
                                                    )
sk_temp_func = lambda x, f=sk_temp_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
sk_data_temp = func_to_vstack(sk_temp_func, sk_aspect['radius'].values)

# Create interpolator for omega
sk_ln_omega = np.log(sk.Omega)
sk_disk_omega_func_log = scipy.interpolate.CubicSpline(
                                                    np.log(sk_R),
                                                    sk_ln_omega,
                                                    extrapolate=False
                                                    )
sk_disk_omega_func = lambda x, f=sk_disk_omega_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
sk_data_omega = func_to_vstack(sk_disk_omega_func, sk_aspect['radius'].values)


# Save to file
np.savetxt("src/mcfacts/inputs/data/sirko_goodman_sound_speed.txt", sk_data_cs, header="Sound speed [m/s] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/sirko_goodman_density.txt", sk_data_rho, header="Disk density [kg/m^3] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/sirko_goodman_pressure_gradient.txt", sk_data_dpdr, header="Disk pressure gradient [kg/m s^2 m] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/sirko_goodman_temperature.txt", sk_data_temp, header="Disk temperature [K] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/sirko_goodman_omega.txt", sk_data_omega, header="Disk omega [rad/s] radius [R_g=GM_SMBH/c^2]")


#### -------------------- TQM DISK --------------------

# Thompson et al. 2005
tqm_mbh = smbh_mass*pagn_ct.MSun
tqm_m = 0.01  # disk_alpha_viscosity
Rg = smbh_mass * ct.M_sun * ct.G / (ct.c**2)
disk_radius_outer = 50000.
tqm_Rout = disk_radius_outer * Rg.to('m').value

# Create disk model
tqm = Thompson.ThompsonAGN(Mbh=tqm_mbh, m=tqm_m, Rout=tqm_Rout)
tqm.solve_disk(N=1e4)

# Read in existing file to get radii
tqm_aspect = pd.read_csv("src/mcfacts/inputs/data/thompson_etal_aspect_ratio.txt",
                              names=['aspect_ratio', 'radius'], skiprows=1, sep=" ")

tqm_R = tqm.R / (tqm.Rs / 2)
tqm_cs = tqm.h * tqm.Omega
tqm_ln_cs = np.log(tqm_cs)
tqm_cs_func_log = scipy.interpolate.CubicSpline(np.log(tqm_R), tqm_ln_cs, extrapolate=False)
tqm_cs_func = lambda x, f=tqm_cs_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
tqm_data_cs = func_to_vstack(tqm_cs_func, tqm_aspect['radius'])

# Create interpolator for disk density (rho)
tqm_ln_rho = np.log(tqm.rho)
tqm_rho_func_log = scipy.interpolate.CubicSpline(np.log(tqm_R), tqm_ln_rho, extrapolate=False)
tqm_rho_func = lambda x, f=tqm_rho_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
tqm_data_rho = func_to_vstack(tqm_rho_func, tqm_aspect['radius'])

# Create interpolator for disk pressure gradient (dP/dR)
tqm_pgas = tqm.rho * tqm.T * pagn_ct.Kb / pagn_ct.massU
tqm_prad = tqm.tauV * pagn_ct.sigmaSB * tqm.Teff4 / (2 * pagn_ct.c)
tqm_ptot = tqm_pgas + tqm_prad
tqm_disk_pressure_grad_func_interp = scipy.interpolate.CubicSpline(
                                                        tqm.R,
                                                        np.gradient(tqm_ptot)/np.gradient(tqm.R),
                                                        extrapolate=False)
tqm_disk_pressure_grad_func = lambda x, f=tqm_disk_pressure_grad_func_interp: f(point_masses.si_from_r_g(tqm.Mbh * u.kg, x).value)
# Apply to radius and put in vstack form
tqm_data_dpdr = func_to_vstack(tqm_disk_pressure_grad_func, tqm_aspect['radius'].values)

# Create interpolator for temperature
tqm_temp_midplane = tqm.T  # Disk midplane temp (K)
tqm_ln_temp_midplane = np.log(tqm_temp_midplane)  # ln midplane temp.
tqm_temp_func_log = scipy.interpolate.CubicSpline(
                                                    np.log(tqm_R),
                                                    tqm_ln_temp_midplane,
                                                    extrapolate=False
                                                    )
tqm_temp_func = lambda x, f=tqm_temp_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
tqm_data_temp = func_to_vstack(tqm_temp_func, tqm_aspect['radius'])

# Create interpolator for omega
tqm_ln_omega = np.log(tqm.Omega)
tqm_disk_omega_func_log = scipy.interpolate.CubicSpline(
                                                    np.log(tqm_R),
                                                    tqm_ln_omega,
                                                    extrapolate=False
                                                    )
tqm_disk_omega_func = lambda x, f=tqm_disk_omega_func_log: np.exp(f(np.log(x)))
# Apply to radius and put in vstack form
tqm_data_omega = func_to_vstack(tqm_disk_omega_func, tqm_aspect['radius'])


# Save to file
np.savetxt("src/mcfacts/inputs/data/thompson_etal_sound_speed.txt", tqm_data_cs, header="Sound speed [m/s] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/thompson_etal_density.txt", tqm_data_rho, header="Disk density [kg/m^3] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/thompson_etal_pressure_gradient.txt", tqm_data_dpdr, header="Disk pressure gradient [kg/m s^2 m] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/thompson_etal_temperature.txt", tqm_data_temp, header="Disk temperature [K] radius [R_g=GM_SMBH/c^2]")
np.savetxt("src/mcfacts/inputs/data/thompson_etal_omega.txt", tqm_data_omega, header="Disk omega [rad/s] radius [R_g=GM_SMBH/c^2]")
