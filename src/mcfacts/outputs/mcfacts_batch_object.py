#!/usr/bin/env python
'''Handle mcfacts outputs as inputs

Mcfacts outputs are not all that big, so we're going to store them in RAM
'''
######## Imports ########
import numpy as np
from os.path import isfile, isdir, split, join, abspath
import os
from astropy import units as u
#### Mcfacts ####
from mcfacts.outputs import mergerfile
from mcfacts.inputs.ReadInputs import INPUT_TYPES

######## Objects ########
class McfactsBatch(object):
    '''Batch of mcfacts runs with some initialization parameters'''

    def __init__(
                 self,
                 ini_dict,
                 data_arr,
                 label=None,
                ):
        '''Initialize a mcfacts batch'''
        # Store initialization settings
        assert isinstance(ini_dict, dict)
        # Give units to some parameters
        ini_dict['mass_smbh'] = ini_dict['mass_smbh'] * u.solMass
        ini_dict['M_nsc'] = ini_dict['M_nsc'] * u.solMass
        # Update self
        self.__dict__.update(ini_dict)
        # Initialize systems dict
        self.systems = {}
        # Assign nsystems
        self.nsystems = min(data_arr.size, data_arr.shape[0])
        # Check data_arr
        assert isinstance(data_arr, np.ndarray)
        assert len(data_arr.shape) == 2
        # Initialize _key_index
        _key_index = 0

        ## Extract fields ##
        # Check for zero systems
        if self.nsystems == 0:
            # Loop the mergerfile names
            for key in mergerfile.names_rec:
                self.systems[key] = np.asarray([])
            return

        # Check for iteration field
        if len(mergerfile.names_rec) != data_arr.shape[1]:
            # There should only be one unlisted field
            assert data_arr.shape[1] == len(mergerfile.names_rec) + 1
            # Get iteration
            self.systems['iter'] = data_arr[:,0]
            # Update key index
            _key_index += 1
        # Loop the mergerfile names
        for key in mergerfile.names_rec:
            # Assign variables
            self.systems[key] = data_arr[:,_key_index]
            # Increment key index
            _key_index += 1

    #### Properties ####
    @property
    def n_it(self):
        '''Return the number of iterations for this batch'''
        return self.n_iterations

    @property
    def avg_systems(self):
        '''Return the average number of systems for an iteration within this batch'''
        return self.nsystems // self.n_it

    @property
    def iter(self):
        '''Return the iteration number for each sample'''
        return self.systems['iter'].astype(int)

    @property
    def CM(self):
        '''Return CM for each sample'''
        # TODO more info and units
        return self.systems['CM']

    @property
    def total_mass_source(self):
        '''Return the total mass of each sample'''
        return self.systems['M'] * u.solMass

    @property
    def M(self):
        '''Alias for total_mass_source'''
        return self.total_mass_source

    @property
    def chi_eff(self):
        '''Effective spin'''
        return self.systems['chi_eff']

    @property
    def a_tot(self):
        '''a_tot'''
        # TODO more info and units
        return self.systems['a_tot']

    @property
    def spin_angle(self):
        '''spin_angle'''
        return self.systems['spin_angle']

    @property
    def m1(self):
        '''m1 from mcfacts'''
        return self.systems['m1'] * u.solMass

    @property
    def m2(self):
        '''m2 from mcfacts'''
        return self.systems['m2'] * u.solMass

    @property
    def a1(self):
        '''The spin component of m1'''
        return self.systems['a1']
    
    @property
    def a2(self):
        '''The spin component of m2'''
        return self.systems['a2']

    @property
    def theta1(self):
        '''theta1'''
        return self.systems['theta1']
    
    @property
    def theta2(self):
        '''theta2'''
        return self.systems['theta2']

    @property
    def gen1(self):
        '''The generation of first component'''
        return self.systems['gen1'].astype(int)

    @property
    def gen2(self):
        '''The generation of second component; not to be confused with gentoo'''
        return self.systems['gen2'].astype(int)

    @property
    def t_merge(self):
        '''The time'''
        return self.systems['t_merge']

    @property
    def chi_p(self):
        '''The precessing spin component'''
        return self.systems['chi_p']

    @staticmethod
    def from_runs(run_directory, **kwargs):
        '''Generate a mcfacts_batch object from runs'''
        # Check that we are looking at a valid directory
        assert isdir(run_directory)
        # Get files in directory
        dir_files = os.listdir(run_directory)
        dir_files.sort()
        # Identify output*.dat
        fname_mergers = None
        for _file in dir_files:
            if (_file.startswith('output_mergers_population') and _file.endswith('.dat')):
                fname_mergers = abspath(join(run_directory, _file))
                assert isfile(fname_mergers)
        # Check that we found output*.dat
        assert not (fname_mergers is None)
        # Find Vera's stand-in log file
        fname_log = None
        for _file in dir_files:
            if (_file.startswith('out') and _file.endswith('.log')):
                fname_log = abspath(join(run_directory, _file))
                assert isfile(fname_log)
        # Check that we found out*.log
        assert not (fname_log is None)
        # Initialize parameter dict
        ini_dict = {}
        with open(fname_log, 'r') as F:
            for line in F:
                line = line.split("=")
                if len(line) == 2:
                    key = line[0].rstrip(' ')
                    value = line[1].rstrip(' \n').lstrip(' ')
                    if key in INPUT_TYPES:
                        value = INPUT_TYPES[key](value)
                    ini_dict[key] = value
        # Load data
        data_arr = np.loadtxt(fname_mergers)
        # make at least 2d
        data_arr = np.atleast_2d(data_arr)
        return McfactsBatch(ini_dict, data_arr, **kwargs)
                
######## Args for testing ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory', required=True, type=str,
        help="Run directory")
    opts = parser.parse_args()
    return opts

######## Main for testing ########
def main():
    opts = arg()
    MB = McfactsBatch.from_runs(opts.run_directory)
    print(MB.systems.keys())
    for item in MB.systems:
        print(item)
        assert hasattr(MB, item)

######## Execution ########
if __name__ == "__main__":
    main()
