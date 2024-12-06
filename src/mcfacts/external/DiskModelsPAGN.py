"""
Interface with pAGN
"""
import numpy as np
import pagn.constants as ct

from pagn import Thompson
from pagn import Sirko
import scipy.interpolate


class AGNGasDiskModel(object):
    def __init__(self, disk_type="Sirko", **kwargs):
        self.disk_type = disk_type
        if self.disk_type == "Sirko":
            self.disk_model = Sirko.SirkoAGN(**kwargs)
        else:
            self.disk_model = Thompson.ThompsonAGN(**kwargs)
        self.disk_model.solve_disk(N=1e4)

    def save(self, filename):
        """Method to save key AGN model parameters to filename

        Parameters
        ----------
        obj: object
        Python object representing a solved AGN disk either from the Sirko & Goodman model
        or from the Thompson model

        """
        pgas = self.disk_model.rho * self.disk_model.T * ct.Kb / ct.massU
        prad = 4 * ct.sigmaSB * (self.disk_model.T ** 4) / (3 * ct.c)
        cs = np.sqrt((pgas + prad) / (self.disk_model.rho))
        omega = self.disk_model.Omega
        rho = self.disk_model.rho
        h = self.disk.model.h
        T = self.disk_model.T
        tauV = self.self.disk_model.tauV
        Q = self.disk_model.Q
        R = self.disk_model.R
        if hasattr(self.disk_model, "eta"):
            np.savetxt(filename, np.vstack((R/ct.pc, omega, T, rho, h, self.disk_model.eta, cs, tauV, Q)).T)
        else:
            np.savetxt(filename, np.vstack((R/ct.pc, omega, T, rho, h, cs, tauV, Q)).T)

    def return_disk_surf_model(self, flag_truncate_disk=False):
        """Generate disk surface model functions

        Interpolate and return disk surface model functions as a function of the disk radius.
          1) surface density (Sigma = 2 rho H) in  kg/m^2 given distance from SMBH in r_g = r_s/2
          2) aspect ratio (h/r)
          3) opacity (kappa = 2 * tau / Sigma) in m^2/kg

        Default pagn internal units are SI.

        Parameters
        ----------
        flag_truncate_disk : bool, optional
            If `True`, truncate these functions at the radius where star formation starts
            in the gas disk. If `False`, do not truncate. By default `False`.

        Returns
        -------
        surf_dens_func : lambda
            Surface density (radius)
        aspect_func : lambda
            Aspect ratio (radius)
        opacity_func : lambda
            Opacity (radius)
        bonus_structures : dict
            Other disk model things we may want, which are only available
            for pAGN models

        """

        # convert to R_g (=R/( M G/c^2) explicitly, using internal structures
        R = self.disk_model.R / (self.disk_model.Rs / 2)
        R_agn = self.disk_model.R_AGN / (self.disk_model.Rs / 2)
        Sigma = 2 * self.disk_model.h * self.disk_model.rho  # SI density
        kappa = 2 * self.disk_model.tauV / Sigma  # Opacity = 2*tau/Sigma

        if flag_truncate_disk: # truncate to gas part of disk (no SFR)
            R = R[:self.disk_model.isf]
            Sigma = Sigma[:self.disk_model.isf]
            kappa = kappa[:self.disk_model.isf]

        # Generate surface density (Sigma) interpolator function
        ln_Sigma = np.log(Sigma)  # log of SI density
        surf_dens_func_log = scipy.interpolate.CubicSpline(
                                                           np.log(R),
                                                           ln_Sigma,
                                                           extrapolate=False
                                                           )
        surf_dens_func = lambda x, f=surf_dens_func_log: np.exp(f(np.log(x)))

        # Generate aspect ratio (h/r) interpolator function
        ln_aspect_ratio = np.log(self.disk_model.h/self.disk_model.R)
        # if flag_truncate_disk: # truncate to gas part of disk (no SFR)
        #    ln_aspect_ratio = ln_aspect_ratio[:self.disk_model.isf]
        aspect_func_log = scipy.interpolate.CubicSpline(
                                                        np.log(R),
                                                        ln_aspect_ratio,
                                                        extrapolate=False
                                                        )
        aspect_func = lambda x, f=aspect_func_log: np.exp(f(np.log(x)))

        # Generate opacity (kappa) interpolator function
        ln_opacity = np.log(kappa)
        opacity_func_log = scipy.interpolate.CubicSpline(
                                                         np.log(R),
                                                         ln_opacity,
                                                         extrapolate=False
                                                         )
        opacity_func = lambda x, f=opacity_func_log: np.exp(f(np.log(x)))

        bonus_structures = {}
        bonus_structures['R_agn'] = R_agn
        bonus_structures['R'] = R
        bonus_structures['Sigma'] = Sigma
        bonus_structures['h_over_R'] = np.exp(ln_aspect_ratio)
        bonus_structures['kappa'] = kappa
        bonus_structures["rho"] = self.disk_model.rho
        bonus_structures["T"] = self.disk_model.T
        bonus_structures["tauV"] = self.disk_model.tauV

        return surf_dens_func, aspect_func, opacity_func, bonus_structures





###
### Migration torque code, from pagn repo
###

def gamma_0(q, hr, Sigma, r, Omega):
    """
    Method to find the normalization torque

    Parameters
    ----------
    q: float/array
        Float or array representing the mass ratio between the migrator and the central BH.
    hr: float/array
        Float or array representing the disk height to distance from central BH ratio.
    Sigma: float/array
        Float or array representing the disk surface density in kg/m^2
    r: float/array
        Float or array representing the distance from the central BH in m
    Omega: float/array
        Float or array representing the angular velocity at the migrator position in SI units.

    Returns
    -------
    gamma_0: float/array
        Float or array representing the single-arm migration torque on the migrator in kg m^2/ s^2.

    """
    gamma_0 = q*q*Sigma*r*r*r*r*Omega*Omega/(hr*hr)
    return gamma_0


def gamma_iso(dSigmadR, dTdR):
    """
    Method to find the locally isothermal torque.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    gamma_iso: float/array
        Float or array representing the locally isothermal torque on the migrator in kg m^2/ s^2.

    """
    alpha = - dSigmadR
    beta = - dTdR
    gamma_iso = - 0.85 - alpha - 0.9*beta
    return gamma_iso


def gamma_ad(dSigmadR, dTdR):
    """
    Method to find the adiabatic torque.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    gamma_ad: float/array
        Float or array representing the adabiatic torque on the migrator in kg m^2/ s^2.

    """
    alpha = - dSigmadR
    beta = - dTdR
    gamma = 5/3
    xi = beta - (gamma - 1)*alpha
    gamma_ad = - 0.85 - alpha - 1.7*beta + 7.9*xi/gamma
    return gamma_ad


def dSigmadR(obj):
    """
    Method that interpolates the surface density gradient of an AGN disk object.

    Parameters
    ----------
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.

    Returns
    -------
    dSigmadR: float/array
        Discrete array of the log surface density gradient.

    """
    Sigma = 2*obj.rho*obj.h  # discrete
    rlog10 = np.log10(obj.R)  # discrete
    Sigmalog10 = np.log10(Sigma)  # discrete
    Sigmalog10_spline = scipy.interpolate.UnivariateSpline(rlog10, Sigmalog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dSigmadR_spline = Sigmalog10_spline.derivative()
    dSigmadR = dSigmadR_spline(rlog10)
    return dSigmadR


def dTdR(obj):
    """
    Method that interpolates the thermal gradient of an AGN disk object.

    Parameters
    ----------
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.

    Returns
    -------
    dTdR: float/array
        Discrete array of the log thermal gradient.

    """
    rlog10 = np.log10(obj.R)  # descrete
    Tlog10 = np.log10(obj.T)  # descrete
    Tlog10_spline = scipy.interpolate.UnivariateSpline(rlog10, Tlog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dTdR_spline = Tlog10_spline.derivative()
    dTdR = dTdR_spline(rlog10)
    return dTdR


def dPdR(obj):
    """
    Method that interpolates the total pressure gradient of an AGN disk object.

    Parameters
    ----------
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.

    Returns
    -------
    dPdR: float/array
        Discrete array of the log total pressure gradient.

    """
    rlog10 = np.log10(obj.R)  # descrete
    pgas = obj.rho * obj.T * ct.Kb / ct.massU
    prad = obj.tauV*ct.sigmaSB*obj.Teff4/(2*ct.c)
    ptot = pgas + prad
    Plog10 = np.log10(ptot)  # descrete
    Plog10_spline = scipy.interpolate.UnivariateSpline(rlog10, Plog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dPdR_spline = Plog10_spline.derivative()
    dPdR = dPdR_spline(rlog10)
    return dPdR


def CI_p10(dSigmadR, dTdR):
    """
    Method to calculate torque coefficient for the Paardekooper et al. 2010 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    cI: float/array
        Paardekooper et al. 2010 migration torque coefficient
    """
    cI = -0.85 + 0.9*dTdR + dSigmadR
    return cI


def CI_jm17_tot(dSigmadR, dTdR, gamma, obj):
    """
    Method to calculate torque coefficient for the Jiménez and Masset 2017 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.
    gamma: float
        Adiabatic index
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.


    Returns
    -------
    cI: float/array
        Jiménez and Masset 2017 migration torque coefficient
    """
    cL = CL(dSigmadR, dTdR, gamma, obj)
    cI = cL + (0.46 + 0.96*dSigmadR - 1.8*dTdR)/gamma
    return cI


def CI_jm17_iso(dSigmadR, dTdR):
    """
    Method to calculate the locally isothermal torque coefficient for the Jiménez and Masset 2017 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    cI: float/array
        Jiménez and Masset 2017 migration locally isothermal torque coefficient
    """
    cI = -1.36 + 0.54*dSigmadR + 0.5*dTdR
    return cI


def CL(dSigmadR, dTdR, gamma, obj):
    """
    Method to calculate the Lindlblad torque for the Jiménez and Masset 2017 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.
    gamma: float
        Adiabatic index
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.


    Returns
    -------
    cL: float/array
        Jiménez and Masset 2017 Lindblad torque coefficient
    """
    xi = 16*gamma*(gamma - 1)*ct.sigmaSB*(obj.T*obj.T*obj.T*obj.T)\
         /(3*obj.kappa*obj.rho*obj.rho*obj.h*obj.h*obj.Omega*obj.Omega)
    x2_sqrt = np.sqrt(xi/(2*obj.h*obj.h*obj.Omega))
    fgamma = (x2_sqrt + 1/gamma)/(x2_sqrt+1)
    cL = (-2.34 - 0.1*dSigmadR + 1.5*dTdR)*fgamma
    return cL


def gamma_thermal(gamma, obj, q):
    """
    Method to calculate the thermal torque from the Masset 2017 equations, with decay and torque saturation.

    Parameters
    ----------
    gamma: float
        Adiabatic index
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.
    q: float/array
        Float or array representing the mass ratio between the migrator and the central BH.

    Returns
    -------
    g_thermal: float/array
        Masset 2017 migration total thermal torque.
    """
    xi = 16 * gamma * (gamma - 1) * ct.sigmaSB * (obj.T * obj.T * obj.T * obj.T) \
         / (3 * obj.kappa * obj.rho * obj.rho * obj.h * obj.h * obj.Omega * obj.Omega)
    mbh = obj.Mbh*q
    muth = xi * obj.cs / (ct.G * mbh)
    R_Bhalf = ct.G*mbh/obj.cs**2
    muth[obj.h < R_Bhalf] = (xi / (obj.cs*obj.h))[obj.h < R_Bhalf]

    Lc = 4*np.pi*ct.G*mbh*obj.rho*xi/gamma
    lam = np.sqrt(2*xi/(3*gamma*obj.Omega))

    dP = -dPdR(obj)
    xc = dP*obj.h*obj.h/(3*gamma*obj.R)

    kes = electron_scattering_opacity(X=0.7)
    L = 4 * np.pi * ct.G * ct.c * mbh / kes

    g_hot = 1.61*(gamma - 1)*xc*L/(Lc*gamma*lam)
    g_cold = -1.61*(gamma - 1)*xc/(gamma*lam)
    g_thermal = g_hot + g_cold
    g_thermal_new = g_hot*(4*muth/(1+4*muth)) + g_cold*(2*muth/(1+2*muth))
    g_thermal[muth < 1] = g_thermal_new[muth < 1]
    decay = 1 - np.exp(-lam*obj.tauV/obj.h)
    return g_thermal*decay
