# NRSurrogate model published by Varma+2019 and modified for use in McFACTS by Keefe Mitman

import joblib
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from uncertainties import ufloat


class GPRFitter(RegressorMixin, BaseEstimator):
    def __init__(self):
        return

    def __call__(self, xVals, return_std=False, raise_warning=True):
        if raise_warning:
            if np.any(xVals < self.minVals) or np.any(xVals > self.maxVals):
                warnings.warn(
                    "Parameters " + str(xVals) + " are outside of the fit range!"
                )

        if return_std:
            y, y_std = self.gpr.predict([xVals], return_std=True)
            y = y[0]
            y_std = y_std
            y += self.linear.predict([xVals])[0]

            return ufloat(y, y_std)
        else:
            y = self.gpr.predict([xVals], return_std=False)[0]
            y += self.linear.predict([xVals])[0][0]

            return y

    def read_from_file(filename):
        return joblib.load(filename)

    def write_to_file(self, filename):
        joblib.dump(self, filename)

    def fit(
        self,
        xVals,
        yVals,
        yVals_std=None,
        normalize_y=True,
        n_restarts_optimizer=10,
        **kwargs
    ):
        self.minVals = np.min(xVals, axis=0)
        self.maxVals = np.max(xVals, axis=0)

        if yVals_std is None:
            alpha = 1e-10
            sample_weight = np.ones(yVals.shape[0])
        else:
            alpha = yVals_std**2
            sample_weight = 1.0 / alpha

        self.linear_fit(xVals, yVals, sample_weight=sample_weight)

        self.gpr_fit(
            xVals,
            yVals,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer,
            **kwargs
        )

        return

    def linear_fit(self, xVals, yVals, sample_weight=None):
        self.linear = Pipeline(
            [("scaler", StandardScaler()), ("lin", LinearRegression())],
        )

        self.linear.fit(xVals, yVals, lin__sample_weight=sample_weight)

        return

    def gpr_fit(self, xVals, yVals, **kwargs):
        kernel = self.setup_kernel(xVals)

        self.gpr = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gpr", GaussianProcessRegressor(kernel=kernel, **kwargs)),
            ]
        )

        yVals_linear = self.linear.predict(xVals)

        self.gpr.fit(xVals, yVals - yVals_linear)

        return

    def setup_kernel(
        self,
        xVals,
        constKernel_init=1.0,
        constKernel_bounds=(1.0e-2, 1.0e2),
        noiseKernel_init=1e-4,
        noiseKernel_bounds=(1e-7, 1e-2),
        lengthScale_bounds_factors=(1e-2, 1e1),
    ):
        ndims = xVals.shape[1]

        lengthScale_init = np.asarray(
            [(np.max(xVals[:, i]) - np.min(xVals[:, i])) for i in range(ndims)]
        )
        lengthScale_bounds = np.asarray(
            [
                (
                    lengthScale_bounds_factors[0] * L_i,
                    lengthScale_bounds_factors[1] * L_i,
                )
                for L_i in lengthScale_init
            ]
        )

        mainKernel = RBF(
            length_scale=lengthScale_init, length_scale_bounds=lengthScale_bounds
        )

        constantKernel = ConstantKernel(
            constant_value=constKernel_init, constant_value_bounds=constKernel_bounds
        )

        whiteKernel = WhiteKernel(noiseKernel_init, noiseKernel_bounds)

        kernel = constantKernel * mainKernel + whiteKernel

        return kernel


class GPRFitters:
    def __init__(self, components):
        self.components = components
        return

    def __call__(self, xVals, return_std=False, raise_warning=True):
        ys = []
        for component in self.components:
            y = component(xVals, return_std=return_std, raise_warning=raise_warning)
            ys.append(y)

        return np.array(ys)

    def read_from_file(filename):
        return joblib.load(filename)

    def write_to_file(self, filename):
        joblib.dump(self, filename)
