======================================
Adding New Parameters to McFACTS
======================================

Inputs
******

If you want to add a new input parameter to McFACTS, it *must* be added
to the following locations to be properly recognized by the code.

1. `IOdocumentation.txt`_
2. `ReadInputs.py`_
3. Default `model_choice.ini`_


.. _sect-iodoc:

The `IOdocumentation.txt`_ File
------------------------------------

This file is intended as a reference for users to understand the affects of setting a parameter to a given value.
It should follow this format:

.. code-block:: text

    <name> : <type>
        <description>

where :code:`<description>` explains how the code uses the assigned value or lists all settings if the parameter is a flag.
Include the default value where relevant.

.. Warning::

    DO NOT use boolean types. Use integers with 0/1 behavior.

Here are some examples of entries in `IOdocumentation.txt`_

.. code-block:: text

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

The `ReadInputs.py`_ File
------------------------------

**Two** entries are required in :py:obj:`mcfacts.inputs.ReadInputs`.

The first entry goes in the list of parameters in the dostring at the beginning of the file. The entry should
contain the variable name in double quotes ``("")``, the object type, and a brief description or possible values
as follows:

.. literalinclude:: ../../src/mcfacts/inputs/ReadInputs.py
    :caption: Source code snippet of ReadInputs.py
    :lines: 1-10

The second entry goes in the :py:obj:`mcfacts.inputs.ReadInputs.INPUT_TYPES` dictionary.

.. code-block:: python

    # Dictionary of types
    INPUT_TYPES = {
        "disk_model_name"               : str,
        "flag_use_pagn"                 : int,
        "flag_add_stars"                : int,
        ...
        }

The Default `model_choice.ini`_ File
-----------------------------------------

.. Warning::
    This file should **NEVER** be edited unless adding a new input parameter or changing the default behavior.

This is the initialization file (``*.ini``) that controls the default behavior of McFACTS under the hood.
Set the value of your new parameter the same way you would in the ini file you use at runtime, e.g.:

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

If you want to add a new column to a ``*.dat`` file:

#. Ensure the relevant Class object contains your new value
#. Add the column name to the appropriate list in :py:obj:`mcfacts.outputs.columns`
#. Add an entry for that value to the corresponding file description in `IOdocumentation.txt`_
#. Use the value as appropriate in analysis scripts

Log files
---------

The ``mcfacts.log`` file automatically records all values passed as command-line options to ``mcfacts_sim.py`` and all
parameters set by default and or by the user's ``*.ini`` file. You can choose a different name for the log file by
passing the ``--fname-log`` option to ``mcfacts_sim.py``

.. code-block:: bash

    $ python mcfacts_sim.py --fname-log lincoln.log

The log file will record the runtime value of any new input parameters you created if you followed the instructions
in the `Inputs`_ section above.

..
    Reference shortcuts
.. _`IOdocumentation.txt`: https://github.com/McFACTS/McFACTS/blob/main/IOdocumentation.txt
.. _`ReadInputs.py`: https://github.com/McFACTS/McFACTS/blob/main/src/mcfacts/inputs/ReadInputs.py
.. _`model_choice.ini`: https://github.com/McFACTS/McFACTS/blob/main/src/mcfacts/inputs/data/model_choice.ini