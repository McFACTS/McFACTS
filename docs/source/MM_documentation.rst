Documentation
============

You can find documentation for our code and modules at our `Read the Docs <https://mcfacts.readthedocs.io>`_. For those planning to contribute to McFACTS documentation, there are a couple extra packages you'll need to install.

Installing Sphinx
*************

To install via pip, run:

.. code-block:: bash

   $ pip install sphinx sphinx_rtd_theme sphinx-autoapi sphinx-automodapi

To install these packages into a conda environment, activate the conda environment you're using for McFACTS, and run:

.. code-block:: bash

   $ conda install sphinx sphinx_rtd_theme sphinx-autoapi sphinx-automodapi

To generate the documentation pages, run the following:

.. code-block:: bash

   # Switch to the docs directory
   $ cd docs

   # Clean up any previously generated docs
   $ make clean

   # Generate the html version of the docs in ./docs/build/html
   $ make html

.. note::

   If you create a new documentation file, it must be added to the list in `docs/source/index.rst`_.

Inputs and outputs are documented in `IOdocumentation.txt`_.

..
   List of link shortcuts

.. _`Google Form`: https://docs.google.com/forms/d/e/1FAIpQLSeupzj8ledPslYc0bHbnJHKB7_LKlr8SY3SfbEVyL5AfeFlVg/viewform
.. _`README`: https://github.com/McFACTS/McFACTS/blob/main/README.md
.. _`IOdocumentation.txt`: https://github.com/McFACTS/McFACTS/blob/main/IOdocumentation.txt
.. _`Style Guide`: https://github.com/McFACTS/McFACTS/blob/main/docs/source/style.rst
.. _`Commit Guide`: https://github.com/McFACTS/McFACTS/blob/main/docs/source/gitcommitmsg.rst
.. _`Pull Request Template`: https://github.com/McFACTS/McFACTS/blob/main/.github/PULL_REQUEST_TEMPLATE.md
.. _`Issue Tracker`: https://github.com/McFACTS/McFACTS/issues
.. _`semantic commit messages`: https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716
.. _`MANIFEST.in`: https://github.com/McFACTS/McFACTS/blob/main/MANIFEST.in
.. _`setup.py`: https://github.com/McFACTS/McFACTS/blob/main/setup.py
.. _`docs/source/index.rst`: https://github.com/McFACTS/McFACTS/blob/main/docs/source/index.rst

