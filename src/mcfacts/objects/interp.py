"""Handling of the McFACTS AGNDisk object"""
######## Imports ########
#### Standard Library ####
from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import cached_property
#### Third Party ####
import numpy as np
from scipy.interpolate import CubicSpline as ScipyCubicSpline
from scipy.interpolate import NearestNDInterpolator

######## Objects ########
#### Boundary Object ####
class Boundary(Enum):
    NAN_INC     = auto()
    NAN_EXC     = auto()
    EXTRAPOLATE = auto()
    FLAT        = auto()
    ZERO_INC    = auto()
    ZERO_EXC    = auto()
    ONE_INC     = auto()
    ONE_EXC     = auto()

#### Bounds Object ####
class Bounds(object):
    """What should be behaviour be at each boundary?"""
    def __init__(self, limits : np.ndarray):
        """Create an array of NAN boundaries (the default boundary behaviour)
        """
        self._values = np.empty_like(limits, dtype=object)
        self._values[...] = Boundary.NAN_INC

    def __setitem__(self, key, value : Boundary):
        """Set the behaviour of a boundary to an enum value
        """
        if not isinstance(value, Boundary):
            raise TypeError(f"value {value} is not of type {Boundary}")
        self._values[key] = value

    def __getitem__(self, key):
        """Return the boundary behaviour enum"""
        return self._values[key]

#### Abstract interpolator ####
class Interpolator(ABC):
    def __init__(
            self,
            ndim : int,
            x_train : np.ndarray,
            y_train : np.ndarray,
            y_err   : np.ndarray = None,
            **kwargs
        ):
        """A Generic interpolator class"""
        self._trained = False
        # Record inputs
        self._ndim = ndim
        self._x_train = x_train
        self._y_train = y_train
        self._y_err = y_err
        self.kwargs = kwargs
        # Generate boundary behaviour array
        self._bounds = Bounds(self.limits)
        # Generate readonly cached properties
        _ = self.limits
        _ = self.x_train
        _ = self.y_train
        _ = self.y_err

    @readonly_cached_property
    def ndim(self):
        return self._ndim

    @readonly_cached_property
    def x_train_unstacked(self):
        # One dimensional branch
        if self.ndim == 1:
            if len(np.shape(self._x_train)) == 1:
                return (self._x_train.copy(),)
            elif np.prod(self._x_train.shape) == self._x_train.size:
                return (self._x_train.flatten(),)
            else:
                raise ValueError(
                    f"Can't train {self.ndim} dim model "
                    f"with data of shape {self._x_train.shape}!"
                )
        # N-dimensional branch
        else:
            # Check inputs
            if len(self._x_train.shape) != 2:
                raise ValueError(
                    f"Training data for {self.ndim} interp should not have "
                    f"shape {self._x_train.shape}"
                )
            # Minimum number of points
            if np.prod(self._x_train.shape) <= self.ndim**2:
                raise ValueError(
                    f"You need at least {self.ndim + 1} points to interpolate "
                    f"in {self.ndim} dimensions. "
                    f"(x_train.shape = {self.x_train.shape})"
                )
            # Infer the correct dimensionality
            if self._x_train.shape[0] == self.ndim:
                return np.unstack(self._x_train, axis=0)
            else:
                return np.unstack(self._x_train, axis=1)

    @readonly_cached_property
    def x_train(self):
        return np.column_stack(self.x_train_unstacked)

    @readonly_cached_property
    def limits(self):
        return np.asarray(
            [(dim.min(), dim.max()) for dim in self.x_train_unstacked]
        )

    @property
    def bounds(self):
        return self._bounds

    @readonly_cached_property
    def y_train(self):
        return self._y_train.copy()

    @readonly_cached_property
    def y_err(self):
        if self._y_err is None:
            return None
        else:
            return self._y_err.copy()
    
    @cached_property
    @abstractmethod
    def interp(self):
        return None

    def __call__(self, x_eval):
        # Number of points is always number of rows
        if self.ndim == 1:
            npts = x_eval.size
            return self._call1d(x_eval, npts)
        else:
            npts, ndim = x_eval.shape
            if not ndim == self.ndim:
                raise ValueError(
                    f"Interp has ndim {self.ndim}, but evals have dim {ndim}!"
                )
            return self._callnd(x_eval, npts)
    
    def _call1d(self, x_eval, npts):
        # Reshape x_eval
        x_eval = x_eval.reshape((npts,))
        # Initialize output
        out = np.full(npts, np.nan)
        # Initialize mask
        inside = np.ones(npts, dtype=bool)

        # Check lower and upper boundary
        for b in [0, 1]:
            # Get boundary enum
            bound = self.bounds[0, b]
            # Check for extrapolate
            if bound is Boundary.EXTRAPOLATE:
                continue
            # Check for NAN_INC
            elif bound is Boundary.NAN_INC:
                if b == 0:
                    inside &= x_eval >= self.limits[0, 0]
                else:
                    inside &= x_eval <= self.limits[0, 1]
            elif bound is Boundary.NAN_EXC:
                if b == 0:
                    inside &= x_eval > self.limits[0, 0]
                else:
                    inside &= x_eval < self.limits[0, 1]
            elif bound is Boundary.FLAT:
                if b == 0:
                    match = x_eval >= self.limits[0, 0]
                    inside &= match
                    out[~match] = self.interp(np.asarray([self.limits[0,0],]))[0]
                else:
                    match = x_eval <= self.limits[0, 1]
                    inside &= match
                    out[~match] = self.interp(np.asarray([self.limits[0,1],]))[0]
            elif bound is Boundary.ONE_INC:
                if b == 0:
                    match = x_eval >= self.limits[0, 0]
                    inside &= match
                    out[~match] = 1.
                else:
                    match = x_eval <= self.limits[0, 1]
                    inside &= match
                    out[~match] = 1.
            elif bound is Boundary.ONE_EXC:
                if b == 0:
                    match = x_eval > self.limits[0, 0]
                    inside &= match
                    out[~match] = 1.
                else:
                    match = x_eval < self.limits[0, 1]
                    inside &= match
                    out[~match] = 1.
            elif bound is Boundary.ZERO_INC:
                if b == 0:
                    match = x_eval >= self.limits[0, 0]
                    inside &= match
                    out[~match] = 0.
                else:
                    match = x_eval <= self.limits[0, 1]
                    inside &= match
                    out[~match] = 0.
            elif bound is Boundary.ZERO_EXC:
                if b == 0:
                    match = x_eval > self.limits[0, 0]
                    inside &= match
                    out[~match] = 0.
                else:
                    match = x_eval < self.limits[0, 1]
                    inside &= match
                    out[~match] = 0.
            else: 
                raise ValueError(f"Somehow bound is not a {Boundary}")

        # Evaluate interpolant
        out[inside] = self.interp(x_eval[inside])
        return out

    def _callnd(self, x_eval, npts):
        # Copy inputs
        x_eval = x_eval.copy()
        # Initialize output
        out = np.full(npts, np.nan)
        # Initialize mask
        inside = np.ones(npts, dtype=bool)
        # Loop dimensions
        for dim in range(self.ndim):
            # Check lower and upper boundary
            for b in [0, 1]:
                # Get boundary enum
                bound = self.bounds[dim, b]
                # Check for extrapolate
                if bound is Boundary.EXTRAPOLATE:
                    continue
                # Check for NAN_INC
                elif bound is Boundary.NAN_INC:
                    if b == 0:
                        inside &= x_eval[:,dim] >= self.limits[dim, 0]
                    else:
                        inside &= x_eval[:,dim] <= self.limits[dim, 1]
                elif bound is Boundary.NAN_EXC:
                    if b == 0:
                        inside &= x_eval[:,dim] > self.limits[dim, 0]
                    else:
                        inside &= x_eval[:,dim] < self.limits[dim, 1]
                elif bound is Boundary.FLAT:
                    if b == 0:
                        match = x_eval[:,dim] < self.limits[dim, 0]
                        x_eval[match,dim] = self.limits[dim,0]
                    else:
                        match = x_eval[:,dim] > self.limits[dim, 1]
                        x_eval[match,dim] = self.limits[dim,1]
                elif bound is Boundary.ONE_INC:
                    if b == 0:
                        match = x_eval[:,dim] >= self.limits[dim, 0]
                        inside &= match
                        out[~match] = 1.
                    else:
                        match = x_eval[:,dim] <= self.limits[dim, 1]
                        inside &= match
                        out[~match] = 1.
                elif bound is Boundary.ONE_EXC:
                    if b == 0:
                        match = x_eval[:,dim] > self.limits[dim, 0]
                        inside &= match
                        out[~match] = 1.
                    else:
                        match = x_eval[:,dim] < self.limits[dim, 1]
                        inside &= match
                        out[~match] = 1.
                elif bound is Boundary.ZERO_INC:
                    if b == 0:
                        match = x_eval[:,dim] >= self.limits[dim, 0]
                        inside &= match
                        out[~match] = 0.
                    else:
                        match = x_eval[:,dim] <= self.limits[dim, 1]
                        inside &= match
                        out[~match] = 0.
                elif bound is Boundary.ZERO_EXC:
                    if b == 0:
                        match = x_eval[:,dim] > self.limits[dim, 0]
                        inside &= match
                        out[~match] = 0.
                    else:
                        match = x_eval[:,dim] < self.limits[dim, 1]
                        inside &= match
                        out[~match] = 0.
                else: 
                    raise ValueError(f"Somehow bound is not a {Boundary}")

        # Evaluate interpolant
        out[inside] = self.interp(x_eval[inside])
        return out

#### CubicSpline ####
class CubicSpline(Interpolator):
    """CubicSpline interpolator"""
    def __init__(self, x_train, y_train):
        super().__init__(1, x_train, y_train)
    @cached_property
    def interp(self):
        return ScipyCubicSpline(
            self.x_train_unstacked[0],
            self.y_train,
        )

class dCubicSpline(Interpolator):
    """CubicSpline interpolator"""
    def __init__(self, x_train, y_train):
        super().__init__(1, x_train, y_train)
    @cached_property
    def interp(self):
        return ScipyCubicSpline(
            self.x_train_unstacked[0],
            self.y_train,
        ).derivative()
        
#### NearestNeighbor ####
class NearestNeighbor(Interpolator):
    """Example ND interpolator for unit tests"""
    @cached_property
    def interp(self):
        return NearestNDInterpolator(
            self.x_train,
            self.y_train,
        )
