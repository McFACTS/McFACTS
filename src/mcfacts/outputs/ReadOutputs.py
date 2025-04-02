"""Define output handling functions for McFACTS

Logfile
-------
    "bin_num_max"                   : int
        Maximum allowable number of binaries at a given time. Sets array size.
    "fname_ini"                     : str
        Name of the ini file used for this run.
    "fname_output_mergers"          : str
        Name of the output file for mergers.
    "fname_output"                  : str
        Name of the output file for the run.
    "fname_snapshots_bh"            : str
        Name of the output file recording BH info at each snapshot.
    "saves_snapshots"               : int
        1: save snapshots, 0: don't save snapshots
    "verbose"                       : int
        1: print extra info, 0: don't print extra info
    "work_directory"                : str
        The working directory used for the run.
    "seed"                          : int
        Random seed used for the run.
    "fname_log"                     : str
        Name of the log file storing details of the run.
    "runtime_directory"             : str
        The runtime directory used for the run.

"""

# Third party
from astropy import constants as ct
# Local imports 
from mcfacts.inputs.ReadInputs import INPUT_TYPES

# Dictionary of types
OUTPUT_TYPES = {
    "bin_num_max"                   : int,
    "fname_ini"                     : str,
    "fname_output_mergers"          : str,
    "fname_output"                  : str,
    "fname_snapshots_bh"            : str,
    "verbose"                       : int,
    "work_directory"                : str,
    "seed"                          : int,
    "fname_log"                     : str,
    "runtime_directory"             : str,
}
# Ensure none of the data types are bool to avoid issues casting ascii to boolean
if bool in OUTPUT_TYPES.values():
    raise ValueError("[ReadOutputs.py] Boolean data types are not allowed in"
                     "the OUTPUT_TYPES dictionary. Please use int instead.")

# Add the INPUT_TYPES to the OUTPUT_TYPES dictionary
for key, value in INPUT_TYPES.items():
    OUTPUT_TYPES[key] = value



def ReadLog(fname_log, verbose=0):
    """Output log parser

    Parameters
    ----------
    fname_log : str
        Path and name to the log file from a McFACTS run
    verbose : int, optional
        Print extra info, by default 0
    """

    # Read in the file
    with open(fname_log, 'r') as f:
        lines = f.readlines()

    # Initialize the dictionary
    log_dict = {}
    extra_values = {}

    # Loop through the lines
    for line in lines:
        # Split the line
        key, value = line.split('=')
        # Strip the values of any whitespace
        key = key.strip()
        value = value.strip()
        # Convert the values
        if key in OUTPUT_TYPES:
            log_dict[key] = OUTPUT_TYPES[key](value)
        # Store unexpected lines to report below
        if not (key in OUTPUT_TYPES):
            log_dict[key] = OUTPUT_TYPES[key](value)
            extra_values[key] = value
        
    # Print the dictionary
    if verbose:
        for key, value in log_dict.items():
            print(f"{key}: {value}")

    # Warning if extra values are found
    if len(extra_values) > 0:
        print(f"~~~~~~~~~~~~~~~~~~~~~~\n",
               f"[ReadOutputs.py] Warning!: The log file you're using contains {len(extra_values)} additional"
               "entries not found in OUTPUT_TYPES. They have been added to the log"
               "dictionary as a STRING type. Please verify their types before you"
               "use them in your analysis or add them to the OUTPUT_TYPES dictionary.")
        for key, value in extra_values.items():
            print(f"  {key}: {value}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~")
        

    return log_dict
    