****************************
How to Contribute to McFACTS 
****************************

This guide will describe the steps and requirements to contribute `your` awesome additions to McFACTS.

Stay in Touch
-------------

First, opt-in to everything McFACTS by using this `Google Form <https://docs.google.com/forms/d/e/1FAIpQLSeupzj8ledPslYc0bHbnJHKB7_LKlr8SY3SfbEVyL5AfeFlVg/viewform>`_ to join our mailing list.

Report Bugs or Feature Requests
-------------------------------

Use our `GitHub Issue tracker <https://github.com/McFACTS/McFACTS/issues>`_ to report any bugs you encounter or think of any **cool** features you think would benefit McFACTS.

Contributing Code
-----------------

We're happy to consider pull requests to improve our documentation and code base.

Documentation
*************

Take a look around.
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

McFACTS Code
************

Follow the process for installing the code in our `README <https://github.com/McFACTS/McFACTS/blob/main/README.md>`_.

See the `McFACTS style guide <https://github.com/McFACTS/McFACTS/blob/main/docs/source/style.rst>`_ for writing code that conforms with our documentation generator.

Commit Messages
***************

For full details see our `commit guide <https://github.com/McFACTS/McFACTS/blob/main/docs/source/gitcommitmsg.rst>`_.

Here is a quick reference for creating `semantic commit messages <https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716>`_ that enable us to easily:

#. automatically generate our changelog
#. review the code's history

All commits should follow this general format:

.. code-block::

   <type>[optional scope]: <description>

   [optional body]

   [optional footer(s)]

``<description>``: is present tense and gramatically structured to complete the sentence, "If applied, this commit will _____."

Generating Pull Requests
************************

Pull requests should comply with these requirements:

#. Direct pull requests at the ``mcfacts/main-dev`` branch.
#. Include all information outlined in the `Pull Request Template <https://github.com/McFACTS/McFACTS/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_ (automatically populates the description field when initiating a pull request).
#. Categorize your pull request using one (or more!) option from this `list <https://github.com/McFACTS/McFACTS/labels>`_ of labels.

Extending McFACTS with other languages
**************************************

Python can be slow... ::

      ____      ___
    /_*_*_*\  |  o | 
   |_*_*_*_*|/ ___\| 
   |_________/     
   |_|_| |_|_|

You may know ways to speed up McFACTS by writing interfaces to compiled languages to handle computationally intensive tasks.
This sounds awesome! But, please ask the dev team first so you don't waste your time on something we may not implement.

To ensure McFACTS remains useable, stable, and maintainable for future users and our core dev team, we `require` the following conditions be met.

#. Any code written in a language which extends Python (C, Fortran, Rust, etc...) must have a working unit test which checks for accuracy.
#. As a result, there must be a pure Python version of the function that exists somewhere to do the same math, that we can check our results against.
#. The pure Python version of such a function should be maintained so that when new physics is brought into the extension, not only the extension is modified, but the Python used to test against the extension as well.
#. Any pull request introducing or modifying an extension to Python in another language must pass the ``test-build`` test.

   .. code-block:: bash

      make test-build

#. Any pull request introducing or modifying an extension to Python in another language must be reviewed by somebody who understands the language.
