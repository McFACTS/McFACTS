======================================
Adding New Parameters to McFACTS
======================================

Inputs
******

If you want to add a new input parameter to McFACTS, it *must* be added
to the following locations to be properly recognized by the code.

1. ``IOdocumentation.txt``
2. ``src/mcfacts/inputs/ReadInputs.py``
3. ``src/mcfacts/inputs/data/model_choice.ini``


The IOdocumentation.txt File
--------------------------------

This file is intended as a reference for users to understand the affects of setting a parameter to a given value.
It should follow this format:

.. code-block::

    <name> : <type>
        <description>

where `<description>` explains how the code uses the assigned value or lists all settings if the parameter is a flag.
Include the default value where relevant.

.. Warning::

    DO NOT use boolean types. Use integers with 0/1 behavior.

Here are some examples of entries in `IOdocumentation.txt`_

.. code-block::

    timestep_num : int
        Disk lifetime is set by choosing your number of timesteps:
        lifetime = timestep * number_of_timesteps
        choosing default settings gives 1Myr lifetime
        default : 100

    flag_thermal_feedback : int
        Switch to incorporate a feedback model from accreting embedded BH. Changes migration rates.
        feedback = 1 makes all BH subject to gas torque modification due to feedback
        feedback = 0 means no BH gas torque modifications (Type I migration only).
        Feedback model right now is Hanka et al. 2022
        default : 1

    disk_star_initial_mass_cutoff : float
        Maximum star mass
        default: 300 M_sun

The ReadInputs.py File
--------------------------

**Two** entries are required in `ReadInputs.py`_.

The first goes in the list of parameters in the dostring at the beginning of the file. The entry should
contain the variable name in double quotes ``("")``, the object type, and a brief description or possible values
such as:

.. code-block:: python

    """Define input handling functions for mcfacts_sim

    Inifile
    -------
        "disk_model_name"               : str
            'sirko_goodman' or 'thompson_etal'
        .
        .
        .
    """

and so on.

The second is in the ``INPUT_TYPES`` dictionary.

.. code-block:: python

    # Dictionary of types
    INPUT_TYPES = {
        "disk_model_name"               : str,
        "flag_use_pagn"                 : int,
        "flag_add_stars"                : int,
        ...
        }

The data/model_choice.ini File
----------------------------------

.. Warning::
    This file should **NEVER** be edited unless adding a new input parameter or changing the default behavior.

This is the initialization file (``*.ini``) that controls the default behavior of McFACTS under the hood.
Set the value of your new parameter the same way you would in the ini file you use at runtime.

.. code-block::

    # timestep in years (float)
    timestep_duration_yr = 1.e4

Outputs
*******

McFACTS writes information to a few different files with the following extensions ``*.dat`` and ``*.log``

- ``*.dat`` files contain values calculated during runtime
- ``*.log`` files contains a list of values assigned to each parameter and other information helpful for troubleshooting

Data files
----------

If you want to add a new value to an existing ``*.dat`` file:

#. Ensure the relevant Class object contains your new value
#. Add the column name to the appropriate list in ``src/mcfacts/outputs/columns.py``. This puts the column name
    in the header for that column in the ``*.dat`` file
#. Add an entry for the value to the section in `IOdocumentation.txt`_ describing the file
#. Use the value as appropriate in analysis scripts



Log files
---------

The ``mcfacts.log`` file automatically records all values passed as command-line options to ``mcfacts_sim.py`` and all
parameters set by default and or by the user's ``*.ini`` file. You can choose a differen name for the log file by
passing the ``--fname-log`` option to ``mcfacts_sim.py``

.. code-block:: bash

    $ python mcfacts_sim.py --fname-log myname.log

The log file will include any any new input parameters you created if you followed the instructions in
`The ReadInputs.py File`_ above.

..
    Reference shortcuts
.. _`IOdocumentation.txt`: https://github.com/McFACTS/McFACTS/blob/main/IOdocumentation.txt
.. _`ReadInputs.py`: https://github.com/McFACTS/McFACTS/blob/main/src/mcfacts/inputs/ReadInputs.py
