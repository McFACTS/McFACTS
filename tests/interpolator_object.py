#!/usr/bin/env python3
"""Test for Vera's new interpolator object"""

######## Imports ########
#### Standard Library ####
#### Third Party ####
import numpy as np
#### McFACTS ####
from mcfacts.objects.interp import Boundary, Bounds, Interpolator
from mcfacts.objects.interp import CubicSpline, NearestNeighbor

######## Setup #######

######## Tests ########
def test_boundary_enum():
    A = Boundary.NAN_INC
    B = Boundary.NAN_EXC
    C = Boundary.EXTRAPOLATE    
    D = Boundary.FLAT
    E = Boundary.ZERO_INC
    F = Boundary.ZERO_EXC
    G = Boundary.ONE_INC
    H = Boundary.ONE_EXC
    alph = [A, B, C, D, E, F, G, H]
    for i in range(len(alph)):
        for j in range(i):
            assert alph[i] is not alph[j]
        assert alph[i] is alph[i]
    # Make sure this fails
    failed = False
    try:
        I = Boundary.banana
    except AttributeError:
        failed = True
        pass
    assert failed

def test_bounds_obj():
    # Initialize
    limits_1d = np.arange(2)
    bounds = Bounds(limits_1d)
    # Default value
    assert bounds[0] is Boundary.NAN_INC
    assert bounds[1] is Boundary.NAN_INC
    # Set only one
    bounds[1] = Boundary.FLAT
    assert bounds[0] is Boundary.NAN_INC
    assert bounds[1] is Boundary.FLAT
    # Fail to set something stupid
    failed = False
    try:
        bounds[0] = "telephone"
    except TypeError:
        failed = True
    assert failed

    # 2D test
    limits_2d = np.arange(4).reshape((2,2,))
    bounds = Bounds(limits_2d)
    assert bounds[1,1] is Boundary.NAN_INC
    bounds[1,1] = Boundary.FLAT
    assert bounds[1,1] is Boundary.FLAT

def test_CubicSpline():
    myfunc = lambda x: 4. + np.sin(x / (2*np.pi))
    # Initialize some data
    x_train = np.linspace(0.,1.,51)
    y_train =  myfunc(x_train)
    # Initialize
    CS = CubicSpline(x_train, y_train)
    ## Check ndim
    assert CS.ndim == 1
    failed = False
    # Check readonly
    try:
        CS.ndim = 2
    except:
        failed = True
    assert failed
    # Check caching
    CS._ndim = 2
    assert CS.ndim == 1
    # Check unstacked
    assert len(CS.x_train_unstacked) == 1
    # This is a loop over one thing
    for x in CS.x_train_unstacked:
        assert all(x == x_train)
    # Check y_train
    assert all(CS.y_train == y_train)
    # Check y_err
    assert CS.y_err is None
    # Check limits
    assert CS.limits.shape == (1,2)
    assert CS.limits[0,0] == 0.
    assert CS.limits[0,1] == 1.
    # Check bounds
    assert CS.bounds[0,0] is Boundary.NAN_INC
    assert CS.bounds[0,1] is Boundary.NAN_INC

    ## Evaluations ##
    x_eval = np.linspace(-1.,2.,151)
    mask_in = (x_eval > x_train[0]) & (x_eval < x_train[-1])
    mask_lt = x_eval <  x_train[0]
    mask_gt = x_eval >  x_train[-1]
    mask_el = x_eval == x_train[0]
    mask_eg = x_eval == x_train[-1]
    # Make sure all of these cases are explored
    masks = [mask_in, mask_lt, mask_gt, mask_el, mask_eg]
    for mask in masks:
        assert np.sum(mask) > 0
    # Make sure the interpolation works
    assert np.allclose(y_train, CS(x_train))
    
    ## Start checking cases
    # NAN_INC
    CS.bounds[0,0] = Boundary.NAN_INC
    CS.bounds[0,1] = Boundary.NAN_INC
    assert CS.bounds[0,0] is Boundary.NAN_INC
    assert CS.bounds[0,1] is Boundary.NAN_INC
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.allclose(y_eval[mask_el], myfunc(x_eval[mask_el]))
    assert np.allclose(y_eval[mask_eg], myfunc(x_eval[mask_eg]))
    assert np.all(np.isnan(y_eval[mask_lt]))
    assert np.all(np.isnan(y_eval[mask_gt]))
    # NAN_EXC
    CS.bounds[0,0] = Boundary.NAN_EXC
    CS.bounds[0,1] = Boundary.NAN_EXC
    assert CS.bounds[0,0] is Boundary.NAN_EXC
    assert CS.bounds[0,1] is Boundary.NAN_EXC
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.all(np.isnan(y_eval[mask_el]))
    assert np.all(np.isnan(y_eval[mask_eg]))
    assert np.all(np.isnan(y_eval[mask_lt]))
    assert np.all(np.isnan(y_eval[mask_gt]))
    # EXTRAPOLATE
    CS.bounds[0,0] = Boundary.EXTRAPOLATE
    CS.bounds[0,1] = Boundary.EXTRAPOLATE
    assert CS.bounds[0,0] is Boundary.EXTRAPOLATE
    assert CS.bounds[0,1] is Boundary.EXTRAPOLATE
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.allclose(y_eval[mask_el], myfunc(x_eval[mask_el]))
    assert np.allclose(y_eval[mask_eg], myfunc(x_eval[mask_eg]))
    assert not np.any(np.isnan(y_eval[mask_lt]))
    assert not np.any(np.isnan(y_eval[mask_gt]))
    # FLAT
    CS.bounds[0,0] = Boundary.FLAT
    CS.bounds[0,1] = Boundary.FLAT
    assert CS.bounds[0,0] is Boundary.FLAT
    assert CS.bounds[0,1] is Boundary.FLAT
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.allclose(y_eval[mask_el], myfunc(x_eval[mask_el]))
    assert np.allclose(y_eval[mask_eg], myfunc(x_eval[mask_eg]))
    assert np.allclose(y_eval[mask_lt], myfunc(x_eval[mask_el])[0])
    assert np.allclose(y_eval[mask_gt], myfunc(x_eval[mask_eg])[0])
    # ZERO_INC
    CS.bounds[0,0] = Boundary.ZERO_INC
    CS.bounds[0,1] = Boundary.ZERO_INC
    assert CS.bounds[0,0] is Boundary.ZERO_INC
    assert CS.bounds[0,1] is Boundary.ZERO_INC
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.allclose(y_eval[mask_el], myfunc(x_eval[mask_el]))
    assert np.allclose(y_eval[mask_eg], myfunc(x_eval[mask_eg]))
    assert np.allclose(y_eval[mask_lt], 0.)
    assert np.allclose(y_eval[mask_gt], 0.)
    # ZERO_EXC
    CS.bounds[0,0] = Boundary.ZERO_EXC
    CS.bounds[0,1] = Boundary.ZERO_EXC
    assert CS.bounds[0,0] is Boundary.ZERO_EXC
    assert CS.bounds[0,1] is Boundary.ZERO_EXC
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.allclose(y_eval[mask_el], 0.)
    assert np.allclose(y_eval[mask_eg], 0.)
    assert np.allclose(y_eval[mask_lt], 0.)
    assert np.allclose(y_eval[mask_gt], 0.)
    # ONE_INC
    CS.bounds[0,0] = Boundary.ONE_INC
    CS.bounds[0,1] = Boundary.ONE_INC
    assert CS.bounds[0,0] is Boundary.ONE_INC
    assert CS.bounds[0,1] is Boundary.ONE_INC
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.allclose(y_eval[mask_el], myfunc(x_eval[mask_el]))
    assert np.allclose(y_eval[mask_eg], myfunc(x_eval[mask_eg]))
    assert np.allclose(y_eval[mask_lt], 1.)
    assert np.allclose(y_eval[mask_gt], 1.)
    # ONE_EXC
    CS.bounds[0,0] = Boundary.ONE_EXC
    CS.bounds[0,1] = Boundary.ONE_EXC
    assert CS.bounds[0,0] is Boundary.ONE_EXC
    assert CS.bounds[0,1] is Boundary.ONE_EXC
    y_eval = CS(x_eval)
    assert np.allclose(y_eval[mask_in], myfunc(x_eval[mask_in]))
    assert np.allclose(y_eval[mask_el], 1.)
    assert np.allclose(y_eval[mask_eg], 1.)
    assert np.allclose(y_eval[mask_lt], 1.)
    assert np.allclose(y_eval[mask_gt], 1.)

def test_NearestNeighbor2D():
    # Training data
    x_train = np.asarray([
        [0., 0.], 
        [1., 0.],
        [0., 1.],
        [1., 1.]
    ])
    y_train = np.asarray([2., 3., 4., 5.,])
    # Initialize the thing
    NN = NearestNeighbor(2, x_train, y_train)
    # Define some training data
    x_eval = np.asarray([
        [0.1, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.9, 0.9],
        [-1., -1.], 
        [-1., 0.],
        [-1., 0.5],
        [-1., 1.],
        [-1., 2.],
        [0., -1.], 
        [0., 0.],
        [0., 0.5],
        [0., 1.],
        [0., 2.],
        [0.5, -1.], 
        [0.5, 0.],
        [0.5, 0.5],
        [0.5, 1.],
        [0.5, 2.],
        [1., -1.], 
        [1., 0.],
        [1., 0.5],
        [1., 1.],
        [1., 2.],
        [1.5, -1.], 
        [1.5, 0.],
        [1.5, 0.5],
        [1.5, 1.],
        [1.5, 2.],
    ])
    ## Start checking cases
    # NAN_INC
    NN.bounds[0,0] = Boundary.NAN_INC
    NN.bounds[0,1] = Boundary.NAN_INC
    NN.bounds[1,0] = Boundary.NAN_INC
    NN.bounds[1,1] = Boundary.NAN_INC
    assert NN.bounds[0,0] is Boundary.NAN_INC
    assert NN.bounds[0,1] is Boundary.NAN_INC
    assert NN.bounds[1,0] is Boundary.NAN_INC
    assert NN.bounds[1,1] is Boundary.NAN_INC
    y_eval = NN(x_eval)
    assert np.all(y_eval[:4] == y_train)
    assert all(np.isnan(y_eval[[4,5,6,7,8,9,13,14,18,19,23,24,25,26,27,28]]))
    assert not any(np.isnan(y_eval[[10,11,12,15,16,17,20,21,22]]))
    # NAN_EXC
    NN.bounds[0,0] = Boundary.NAN_EXC
    NN.bounds[0,1] = Boundary.NAN_EXC
    NN.bounds[1,0] = Boundary.NAN_EXC
    NN.bounds[1,1] = Boundary.NAN_EXC
    assert NN.bounds[0,0] is Boundary.NAN_EXC
    assert NN.bounds[0,1] is Boundary.NAN_EXC
    assert NN.bounds[1,0] is Boundary.NAN_EXC
    assert NN.bounds[1,1] is Boundary.NAN_EXC
    y_eval = NN(x_eval)
    assert np.all(y_eval[:4] == y_train)
    assert all(np.isnan(y_eval[[4,5,6,7,8,9,13,14,18,19,23,24,25,26,27,28]]))
    assert all(np.isnan(y_eval[[10,11,12,15,17,20,21,22]]))
    assert not any(np.isnan(y_eval[[16,]]))
    # EXTRAPOLATE
    NN.bounds[0,0] = Boundary.EXTRAPOLATE
    NN.bounds[0,1] = Boundary.EXTRAPOLATE
    NN.bounds[1,0] = Boundary.EXTRAPOLATE
    NN.bounds[1,1] = Boundary.EXTRAPOLATE
    assert NN.bounds[0,0] is Boundary.EXTRAPOLATE
    assert NN.bounds[0,1] is Boundary.EXTRAPOLATE
    assert NN.bounds[1,0] is Boundary.EXTRAPOLATE
    assert NN.bounds[1,1] is Boundary.EXTRAPOLATE
    y_eval = NN(x_eval)
    assert not any(np.isnan(y_eval))
    assert all(y_eval > 1.)
    # FLAT
    NN.bounds[0,0] = Boundary.FLAT
    NN.bounds[0,1] = Boundary.FLAT
    NN.bounds[1,0] = Boundary.FLAT
    NN.bounds[1,1] = Boundary.FLAT
    assert NN.bounds[0,0] is Boundary.FLAT
    assert NN.bounds[0,1] is Boundary.FLAT
    assert NN.bounds[1,0] is Boundary.FLAT
    assert NN.bounds[1,1] is Boundary.FLAT
    y_eval = NN(x_eval)
    assert not any(np.isnan(y_eval))
    assert all(y_eval > 1.)
    assert np.all(y_eval[:4] == y_train)
    assert all(y_eval[[4,5]] == 2.)
    assert all(y_eval[[7,8]] == 4.)
    assert all(y_eval[[9,10]] == 2.)
    assert all(y_eval[[12,13]] == 4.)
    assert all(y_eval[[19,20]] == 3.)
    assert all(y_eval[[22,23]] == 5.)
    assert all(y_eval[[24,25]] == 3.)
    assert all(y_eval[[27,28]] == 5.)
    # ONE_INC
    NN.bounds[0,0] = Boundary.ONE_INC
    NN.bounds[0,1] = Boundary.ONE_INC
    NN.bounds[1,0] = Boundary.ONE_INC
    NN.bounds[1,1] = Boundary.ONE_INC
    assert NN.bounds[0,0] is Boundary.ONE_INC
    assert NN.bounds[0,1] is Boundary.ONE_INC
    assert NN.bounds[1,0] is Boundary.ONE_INC
    assert NN.bounds[1,1] is Boundary.ONE_INC
    y_eval = NN(x_eval)
    assert np.all(y_eval[:4] == y_train)
    assert not any(np.isnan(y_eval))
    assert all(y_eval[[4,5,6,7,8,9,13,14,18,19,23,24,25,26,27,28]] == 1.)
    assert all(y_eval[[10,11,12,15,17,20,21,22]] > 1.)
    assert y_eval[16] > 1.
    # ONE_EXC
    NN.bounds[0,0] = Boundary.ONE_EXC
    NN.bounds[0,1] = Boundary.ONE_EXC
    NN.bounds[1,0] = Boundary.ONE_EXC
    NN.bounds[1,1] = Boundary.ONE_EXC
    assert NN.bounds[0,0] is Boundary.ONE_EXC
    assert NN.bounds[0,1] is Boundary.ONE_EXC
    assert NN.bounds[1,0] is Boundary.ONE_EXC
    assert NN.bounds[1,1] is Boundary.ONE_EXC
    y_eval = NN(x_eval)
    assert np.all(y_eval[:4] == y_train)
    assert not any(np.isnan(y_eval))
    assert all(y_eval[[4,5,6,7,8,9,13,14,18,19,23,24,25,26,27,28]] == 1.)
    assert all(y_eval[[10,11,12,15,17,20,21,22]] == 1.)
    assert y_eval[16] > 1.
    # ZERO_INC
    NN.bounds[0,0] = Boundary.ZERO_INC
    NN.bounds[0,1] = Boundary.ZERO_INC
    NN.bounds[1,0] = Boundary.ZERO_INC
    NN.bounds[1,1] = Boundary.ZERO_INC
    assert NN.bounds[0,0] is Boundary.ZERO_INC
    assert NN.bounds[0,1] is Boundary.ZERO_INC
    assert NN.bounds[1,0] is Boundary.ZERO_INC
    assert NN.bounds[1,1] is Boundary.ZERO_INC
    y_eval = NN(x_eval)
    assert np.all(y_eval[:4] == y_train)
    assert not any(np.isnan(y_eval))
    assert all(y_eval[[4,5,6,7,8,9,13,14,18,19,23,24,25,26,27,28]] == 0.)
    assert all(y_eval[[10,11,12,15,17,20,21,22]] > 1.)
    assert y_eval[16] > 1.
    # ZERO_EXC
    NN.bounds[0,0] = Boundary.ZERO_EXC
    NN.bounds[0,1] = Boundary.ZERO_EXC
    NN.bounds[1,0] = Boundary.ZERO_EXC
    NN.bounds[1,1] = Boundary.ZERO_EXC
    assert NN.bounds[0,0] is Boundary.ZERO_EXC
    assert NN.bounds[0,1] is Boundary.ZERO_EXC
    assert NN.bounds[1,0] is Boundary.ZERO_EXC
    assert NN.bounds[1,1] is Boundary.ZERO_EXC
    y_eval = NN(x_eval)
    assert np.all(y_eval[:4] == y_train)
    assert not any(np.isnan(y_eval))
    assert all(y_eval[[4,5,6,7,8,9,13,14,18,19,23,24,25,26,27,28]] == 0.)
    assert all(y_eval[[10,11,12,15,17,20,21,22]] == 0.)
    assert y_eval[16] > 1.

######## Main ########
def main():
    test_boundary_enum()
    test_bounds_obj()
    test_CubicSpline()
    test_NearestNeighbor2D()
    return

######## Execution ########
if __name__ == "__main__":
    main()
