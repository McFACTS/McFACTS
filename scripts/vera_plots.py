#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os.path import abspath, isfile, isdir, split, join
import sys

## Local imports ##
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
from mcfacts.vis import data
from mcfacts.outputs.mergerfile import MERGER_FIELD_NAMES

######## Arg ########
def arg():
    """Argument parser function for vera_plots.py

    Returns
    -------
    opts : dict
        Dictionary of options for code evaluation
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname-mergers",
        default="runs/output_mergers_population.dat",
        type=str, help="output_mergers file")
    parser.add_argument("--cdf-fields",
        nargs="+",default=[],help="Fields for cdf plots")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--fname-nal", default=None,
        help="Load Vera's Gaussians")
    opts = parser.parse_args()
    assert isfile(opts.fname_mergers)
    opts.wkdir = split(abspath(opts.fname_mergers))[0]
    assert isdir(opts.wkdir)
    return opts

######## Load data ########
def load_mergers_txt(fname, verbose=False):
    """Load output_mergers_population.dat
    
    Parameters
    ----------
    fname : str
        The path to the output_mergers_population.dat file
    verbose : bool
        Verbose output flag
    """
    #mergers = np.loadtxt(fname, skiprows=2)
    mergers = np.loadtxt(fname)
    # Initialize dictionary
    merger_dict = {}
    # Loop the column names
    for i, item in enumerate(MERGER_FIELD_NAMES):
        # Find the correct index
        merger_dict[item] = mergers[:,i]
    ## Select finite values
    mask = np.isfinite(merger_dict['chi_eff'])
    print("Removing %d nans out of %d sample mergers"%(np.sum(~mask), mask.size), file=sys.stderr)
    for item in merger_dict:
        merger_dict[item] = merger_dict[item][mask]

    ## Print things
    # Loop the merger dict
    if verbose:
        for item in merger_dict:
            print(item, merger_dict[item].dtype, merger_dict[item].shape, merger_dict[item][1])
            #print(item, merger_dict[item].dtype, merger_dict[item].shape)
            #print(merger_dict[item])
    # Return the merger dict
    return merger_dict

######## Functions ########
def simple_cdf(x):
    """Construct a simple cdf from some samples
    
    Parameters
    ----------
    x : np.ndarray
        The array of sample values used to construct a cdf
    
    Returns
    -------
    _x : np.ndarray
        The x array, but sorted
    cdf : np.ndarray
        The cdf value at the corresponding value of _x
    """
    _x = np.sort(x)
    cdf = np.arange(_x.size)
    cdf = cdf / np.max(cdf)
    return _x, cdf

def nal_cdf(fname_nal,n=1000):
    """Use gwalk.multivariate_normal object to construct cdfs for a GW population

    Parameters
    ----------
    fname_nal : str
        Path to the [GWTC-X].nal.hdf5 file from Vera's external work
    n : int
        The number of samples to draw

    Returns
    -------
    nal_dict : dict
        A dictionary holding sample and cdf data from the Normal Approximate Likelihood
        method
    """
    from gwalk import MultivariateNormal
    from xdata import Database
    from basil_core.astro.coordinates import m1_m2_of_mc_eta, M_of_mc_eta
    assert isfile(fname_nal)
    db = Database(fname_nal)
    events = db.list_items()
    _group = "aligned3d:PublicationSamples:select"
    mc =        []
    eta =       []
    chi_eff =   []
    M =         []
    eta_cdf = None,
    for item in events:
        group = join(item,_group)
        p_astro = db.attr_value(item, "p_astro")
        if p_astro > 0.5:
            assert db.exists(group)
            _MV = MultivariateNormal.load(fname_nal, group)
            out = _MV.sample_normal(n)
            _mc, _eta, _chi_eff = out[:,0], out[:,1], out[:,2]
            _M = M_of_mc_eta(_mc, _eta)
            mc =        np.append(mc, _mc).flatten()
            eta =       np.append(eta, _eta).flatten()
            chi_eff =   np.append(chi_eff, _chi_eff).flatten()
            M =         np.append(M, _M).flatten()
    chi_eff[chi_eff > 1.] = 1.
    chi_eff[chi_eff < -1.] = -1.
    mc, mc_cdf = simple_cdf(mc)
    eta, eta_cdf = simple_cdf(eta)
    chi_eff, chi_eff_cdf = simple_cdf(chi_eff)
    M, M_cdf = simple_cdf(M)
    nal_dict = {
                "mc"            : mc,
                "mc_cdf"        : mc_cdf,
                "eta"           : eta,
                "eta_cdf"       : eta_cdf,
                "chi_eff"       : chi_eff,
                "chi_eff_cdf"   : chi_eff_cdf,
                "M"             : M,
                "M_cdf"         : M_cdf,
               }
    return nal_dict
    

######## Plots ########
def plot_cdf(merger_dict, label, fname):
    """The simplest plot I could imagine to plot the cdf for a field

    Parameters
    ----------
    merger_dict : dict
        The dictionary with all of the sample values
    label : str
        The key to the dictionary for the field we want to use
    fname : str
        The path to where we want to save the plot
    """
    x, y = simple_cdf(merger_dict[label])
    from matplotlib import pyplot as plt
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(label)
    ax.set_ylabel("CDF")
    plt.savefig(fname)
    plt.close()


def plot_nal_cdf(merger_dict, label, fname, nal_dict):
    """A simple plot for NAL cdfs

    Parameters
    ----------
    merger_dict : dict
        The dictionary with all of the sample values
    label : str
        The key to the dictionary for the field we want to use
    fname : str
        The path to where we want to save the plot
    nal_dict : dict
        The dictionary we are using for NAL data
    """
    x, y = simple_cdf(merger_dict[label])
    from matplotlib import pyplot as plt
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    fig.suptitle(label)
    ax.plot(x, y, label="mcfacts")
    ax.plot(nal_dict[label], nal_dict["%s_cdf"%(label)], label="GWTC-2")
    ax.set_xlabel(label)
    ax.set_ylabel("CDF")
    fig.legend()
    plt.savefig(fname)
    plt.close()


######## Main ########
def main():
    opts = arg()
    merger_dict = load_mergers_txt(opts.fname_mergers, verbose=opts.verbose)

    #### Cdf plots ####
    for _item in opts.cdf_fields:
        assert _item in merger_dict, "%s not found in %s"%(_item, str(opts.cdf))
        fname_item = join(opts.wkdir, "mergers_cdf_%s.png"%(_item))
        plot_cdf(merger_dict, _item, fname_item)

    #### NAL plots ####
    if not opts.fname_nal is None:
        nal_dict = nal_cdf(opts.fname_nal)
        _item = "chi_eff"
        fname_item = join(opts.wkdir, "mergers_nal_cdf_%s.png"%(_item))
        plot_nal_cdf(merger_dict, _item, fname_item, nal_dict)
        _item = "M"
        fname_item = join(opts.wkdir, "mergers_nal_cdf_%s.png"%(_item))
        plot_nal_cdf(merger_dict, _item, fname_item, nal_dict)
    return

######## Execution ########
if __name__ == "__main__":
    main()
