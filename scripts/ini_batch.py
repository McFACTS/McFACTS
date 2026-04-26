#!/usr/bin/env python3
"""Adapted from Vera -- Loop through inifiles in directory and make runs

Example call signature:
python scripts/ini_batch.py \
    --inifile-directory recipes/paper_4 \
    --force \
    --seed 3456789108 \
    --mcfacts-args "--galaxy_num 100"

python scripts/em_plots.py \
    --runs-directory recipes/
"""
######## Imports ########
#### Standard Library ####
import os
import argparse
import warnings
#### Third Party ####
import numpy as np
#### Homemade ####
#### Local ####

######## Argparse ########
def arg():
    """Define arguments for script"""
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--inifile-directory", required=True,
        type=str, help="Directory with the infiles you want to run")
    parser.add_argument("--runtime-directory", default='./',
        type=str, help="Where to launch runs from")
    parser.add_argument("--mcfacts-sim", default="./scripts/mcfacts_sim.py",
        type=str, help="If your runtime directory is not the root directory of the mcfacts repository, you may need to specify the location of mcfacts_sim.py")
    parser.add_argument("--em-plots", default="./scripts/em_plots.py", type=str)
    #parser.add_argument("--plots-directory", default="./")
    parser.add_argument("--seed", default=None, type=int,
        help="By default use a different seed for each run. If given, specify.")
    parser.add_argument("--mcfacts-args", default=None, type=str,
        help="Argument string for mcfacts_sim.py")
    parser.add_argument("--force", "-f", action='store_true',
        help="Overwrite existing directories")
    # Parse arguments
    opts = parser.parse_args()
    
    ## Check options for sense ##
    if not os.path.isdir(opts.inifile_directory):
        raise ValueError(f"Invalid inifile directory: {opts.inifile_directory}")
    if not os.path.isdir(opts.runtime_directory):
        raise ValueError(f"Invalid runtime directory: {opts.runtime_directory}")
    if not os.path.isfile(opts.mcfacts_sim):
        raise ValueError(f"Cannot find script: {opts.mcfacts_sim}")
    if not os.path.isfile(opts.em_plots):
        raise ValueError(f"Cannot find script: {opts.em_plots}")
    
    # Test seed
    rs = np.random.RandomState(seed=opts.seed)
    return opts

######## Main ########
def main():
    # Get arguments
    opts = arg()
    # List files
    files = os.listdir(opts.inifile_directory)
    # Loop files
    for basename in files:
        # Get absolute path
        filename = os.path.join(opts.inifile_directory, basename)
        # Check extension
        if not filename.endswith(".ini"):
            continue
        # We have identified an inifile. Get the tag.
        tag = basename.split(".")[0]
        # Identify working directory
        wkdir = os.path.join(opts.runtime_directory, f"runs_{tag}")
        # Identify log file
        stdout_redirect = os.path.join(wkdir, f"{tag}.out")
        # Check if directory exists
        if os.path.isdir(wkdir):
            # Warn the user
            warnings.warn(f"Path exists: {wkdir}")
            # Make sure this directory looks like  a run directory
            if os.path.isfile(stdout_redirect) and opts.force:
                # Execute order 66
                cmd = f"rm -rf {wkdir}"
                print(cmd)
                os.system(cmd)
            else:
                raise IOError(f"Working directory for {tag} already exists! ({wkdir})")
            
        # Create directory
        os.mkdir(wkdir)
        # Touch output file
        with open(stdout_redirect, 'w') as F:
            pass
        # Copy inifile into working directory
        cmd = f"cp {filename} {wkdir}"
        print(cmd)
        os.system(cmd)
        # Point at local inifile
        filename = os.path.join(wkdir, basename)

        ## Setup run command ##
        # Initialize command
        mcfacts_command = f"python {opts.mcfacts_sim} -w {wkdir} --fname-ini {filename} "
        plots_command = f"python {opts.em_plots} --plots-directory {wkdir} --fname-mergers {wkdir}/output_mergers_population.dat --plots-directory {wkdir}" 
        # Check seed
        if opts.seed is not None:
            mcfacts_command = mcfacts_command + f"--seed {opts.seed} "
        # Pass other arguments
        if opts.mcfacts_args is not None:
            mcfacts_command = mcfacts_command + opts.mcfacts_args
        # Add redirection
        mcfacts_command = mcfacts_command + f" | tee {stdout_redirect}"
        #plots_command = plots_command + f" | tee {stdout_redirect}"
        # Execute command
        print(mcfacts_command)
        os.system(mcfacts_command)
        print(plots_command)
        os.system(plots_command)
        

######## Execution ########
if __name__ == "__main__":
    main()
