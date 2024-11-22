Installation
============

To clone and run this application, you'll need `Git <https://git-scm.com>`_ and `Conda <https://docs.conda.io/en/latest/>`_.

The latest development version is available directly from our `GitHub Repo <https://github.com/bmckernan/test_mcfacts>`_. To start, clone the repository:

.. code-block:: bash

   $ git clone https://github.com/bmckernan/test_mcfacts
   $ cd test_mcfacts

Automated Setup
---------------

Contained in the `Makefile` are a few make commands to get everything setup and running.

.. note::

   If you are running on Windows or want to set things up manually, skip to the Manual Setup section below.

.. code-block:: bash

   # Create the conda environment and install required packages
   $ make setup

   # Activate the conda environment that was created for us
   $ conda activate mcfacts-dev

   # Run mcfacts_sim.py with default initial values, then run population_plots.py
   $ make plots

Done! Below are some extra commands that you might find helpful:

.. code-block:: bash

   # Delete your runs directory and other output files
   $ make clean

   # Re install the mcfacts package for changes to take effect
   $ make install

Manual Setup
------------

We recommend that you create a Conda environment for working with McFACTS. You can do this by running:

.. code-block:: bash

   # Create the conda environment with some default packages installed
   $ conda create --name mcfacts-dev "python>=3.10.4" pip -c conda-forge -c defaults

   # Activate the environment we just created
   $ conda activate mcfacts-dev

   # Install the mcfacts package
   $ python -m pip install --editable .

   # Now all that we have left to do is run McFACTS!
   $ python mcfacts_sim.py --galaxy_num 10 --fname-ini ./recipes/model_choice_old.ini --fname-log out.log --seed 3456789012

Our default inputs are located at `./recipes/model_choice_old.ini`. Edit this file or create your own `model_choice.ini` file with different inputs.

To use a different ini file, replace the file path after the `--fname-ini` argument:

.. code-block:: bash

   $ python mcfacts_sim.py --fname-ini /path/to/your/file.ini

Output Files
------------

Output files will appear in `runs`. For each timestep, there will be an `output_bh_single_ts.dat` and `output_bh_binary_ts.dat` where `ts` is the timestep number (0-N)---these files track every single/binary in the simulation at that timestep.

The entire run will have a single `output_mergers.dat` file, which gives the details of every merger throughout the run. If you are trying to get distributions of merger properties, you probably only need `output_mergers.dat`, but if you are trying to track the history of individual mergers or want to know the state of the nuclear star cluster after an AGN of some duration, you will want the larger output suite.
