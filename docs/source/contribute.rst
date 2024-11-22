Contribute
==========

Want to contribute? Great! We've got a lot of stuff for you to work on.

Documentation
-------------

You can find documentation for our code and modules at our `Read the Docs <https://mcfacts.readthedocs.io>`_.

Input and outputs are documented in `IOdocumentation.txt`.

Want to build or browse the docs locally? Run the following:

.. code-block:: bash

   # Switch to the mcfacts-dev environment and install required packages to build the docs
   $ conda activate mcfacts-dev
   $ conda install sphinx sphinx_rtd_theme sphinx-autoapi sphinx-automodapi

   # Switch to the docs directory
   $ cd docs

   # Clean up any previous generated docs
   $ make clean

   # Generate the html version of the docs in ./docs/build/html
   $ make html