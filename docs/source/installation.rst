Installation
============

To clone and run this application, you'll need `Git <https://git-scm.com>`_.

The latest development version is available directly from our `GitHub Repo <https://github.com/mcfacts/mcfacts>`_. To start, clone the repository:

.. code-block:: bash

   $ git clone https://github.com/McFACTS/McFACTS
   $ cd McFACTS


User Setup
^^^^^^^^^^

Navigate to the :code:`McFACTS/` directory and run

.. code-block:: bash

   pip install .


Developer Setup
^^^^^^^^^^^^^^^

Navigate to the :code:`McFACTS/` directory and run

.. code-block:: bash

   python -m pip install --editable .

Done! Below are some extra commands that you might find helpful:

Running McFACTS
^^^^^^^^^^^^^^^

Try a default McFACTS run and make some visualizations:

.. code-block:: bash

   # Using the Makefile
   $ make plots

   # Invoking the Python Script
   $ python scripts/mcfacts_sim.py --fname-ini ./recipes/model_choice.ini --seed 3456789108


Our default inputs are located at `./recipes/model_choice.ini`_. Edit this file or create your own
:code:`my_model.ini` file with different inputs.

To use a different ini file, replace the file path after the :code:`--fname-ini` argument:

.. code-block:: bash

   $ python mcfacts_sim.py --fname-ini /path/to/my_model.ini

Output Files
^^^^^^^^^^^^

Output files will appear in :code:`runs/`. For each timestep, there will be an :code:`output_bh_single_ts.dat` and :code:`output_bh_binary_ts.dat` where `ts` is the timestep number (0-N)---these files track every single/binary in the simulation at that timestep.

The entire run will have a single :code:`output_mergers.dat` file, which gives the details of every merger throughout the run. If you are trying to get distributions of merger properties, you probably only need `output_mergers.dat`, but if you are trying to track the history of individual mergers or want to know the state of the nuclear star cluster after an AGN of some duration, you will want the larger output suite.

.. _`./recipes/model_choice.ini`: https://github.com/McFACTS/McFACTS/blob/main/recipes/model_choice.ini