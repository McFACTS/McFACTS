****************************
How to Contribute to McFACTS 
****************************

This guide will describe the steps and requirements to contribute `your` awesome additions to McFACTS.

Stay in Touch
-------------

First, opt-in to everything McFACTS by using this `Google Form`_ to join our mailing list.

Report Bugs or Feature Requests
-------------------------------

Use our GitHub `Issue Tracker`_ to report any bugs you encounter or think
of any **cool** features you think would benefit McFACTS.

Contributing Code
-----------------

We're happy to consider pull requests to improve our documentation and code base.

Documentation
*************

Take a look around.
You can find documentation for our code and modules at our `Read the Docs <https://mcfacts.readthedocs.io>`_.

Input and outputs are documented in `IOdocumentation.txt`_.

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

.. note::

   If you create a new documentation file, it must be added to the list in `docs/source/index.rst`_.

McFACTS Code
************

Follow the process for installing the code in our `README`_.

See the `Style Guide`_ for writing code that conforms with our documentation generator.

Commit Messages
***************

For full details see our `Commit Guide`_.

Here is a quick reference for creating `semantic commit messages`_ that enable us to easily:

#. automatically generate our changelog
#. review the code's history

All commits should follow this general format:

.. code-block::

   <type>[optional scope]: <description>

   [optional body]

   [optional footer(s)]

``<description>``: is present tense and gramatically structured to complete the sentence,
"If applied, this commit will _____."

Generating Pull Requests
************************

Pull requests should comply with these requirements:

#. Direct all new pull requests at the ``mcfacts/main-dev`` branch.
#. Include all information outlined in the `Pull Request Template`_ (automatically populates the description field when
   initiating a pull request).
#. Categorize your pull request using one (or more!) option from this
   `list <https://github.com/McFACTS/McFACTS/labels>`_ of labels.

Pull Requests to `main`
***********************

Generating a pull request to `main` should be done when it is time to make a new stable branch of McFACTS.

Reasons to create a pull request to main:

#. To create a version of McFACTS with a tag and a release in order to cite the code.
#. To preserve a feature as stable.
#. To support downstream users of McFACTS.
#. For housekeeping, when a stable version hasn't been preserved for a while.

Steps to setup a pull request to `main`:

#. Create a staging branch with the updated version number; E.g. `staging-0.4.1`.
#. Update the `VERSION` in the Makefile, run `make version`, and commit the updated files to the staging branch.
#. On Github, navigate to `<https://github.com/McFACTS/McFACTS/pulls>`_ and search: `is:pr is:merged base:main <https://github.com/McFACTS/McFACTS/pulls?q=is%3Apr+is%3Amerged+base%3Amain>_` to find the most recent previous PR to `main`. Look at the changelog to identify the two most recent PRs mentioned in that changelog.
#. Next, navigate to `<https://github.com/McFACTS/McFACTS/pulls>`_ and search: `is:pr is:merged base:main-dev <https://github.com/McFACTS/McFACTS/pulls?q=is%3Apr+is%3Amerged+base%3Amain-dev>_`. This will want a list all of the PRs which have been merged into main-dev. You want to find any PRs which are more new than either of the two most recent PRs mentioned in the changelog. You can create a link to each PR by right clicking from the search screen and selecting 'copy link'.
#. Create the draft pull request and follow the format:

.. code::block::

   ### McFACTS Version `X.Y.Z`

    ### Summary
    **Added**
    - Things added in the set of PRs

    **Changed**
    - Things changed in the set of PRs

    **Fixed**
    - Things fixed in the set of PRs

    ### Changelog
    ### Individual PRs (summaries) in the order they were merged into main-dev

    [PR NUMBER: PR Title](https://github.com/McFACTS/McFACTS/pull/[PR NUMBER])
    - Brief description of PR; can be copied from the actual PR

    [PR NUMBER: PR Title](https://github.com/McFACTS/McFACTS/pull/[PR NUMBER])
    - Brief description of PR; can be copied from the actual PR

    [THIS PR NUMBER: THIS PR Title](https://github.com/McFACTS/McFACTS/pull/[THIS PR NUMBER])
    - Updated Version to X.Y.Z
    - Description of everything going to main which is not in another PR.

#. Copy everything below Changelog and add it to the top of CHANGELOG.md, including a header with a link to this PR. Commit that file to the staging branch


Extending McFACTS with other languages
**************************************

Python can be slow... ::

      ____      ___
    /_*_*_*\  |  o | 
   |_*_*_*_*|/ ___\| 
   |_________/     
   |_|_| |_|_|

You may know ways to speed up McFACTS by writing interfaces to compiled languages to handle
computationally intensive tasks. This sounds awesome! But, please ask the dev team first so you don't waste your time
on something we may not implement.

To ensure McFACTS remains useable, stable, and maintainable for future users and our core dev team, we `require` the
following conditions be met.

#. Any code written in a language which extends Python (C, Fortran, Rust, etc...) must have a working unit test
   which checks for accuracy.
#. As a result, there must be a pure Python version of the function that exists somewhere to do the same math, that
   we can check our results against.
#. The pure Python version of such a function should be maintained so that when new physics is brought into the
   extension, not only the extension is modified, but the Python used to test against the extension as well.
#. Any pull request introducing or modifying an extension to Python in another language must pass the
   ``test-build`` test.

   .. code-block:: bash

      make test-build

#. Any pull request introducing or modifying an extension to Python in another language must be reviewed by somebody
   who understands the language.

Additionally, several filetypes (E.g. :code:`.dat`) are excluded in the :code:`.gitignore` file, and therefore may be
missed with a :code:`git status` or :code:`git add --all command`. They scan still be committed to the repository if
they are added manually with 

   .. code-block:: bash
   
      git add -f [PATH/TO/FILE.dat]

Data Files
***********

When adding a data file to the source of McFACTS, you must take these steps:

#. Add the data file to McFACTS requires putting it in the relevant :code:`src/mcfacts` subdirectory.
#. Add the data file to `MANIFEST.in`_.
#. Add the data file to `setup.py`_.
#. Test that McFACTS runs and the file is accessible by running :code:`make test-build`

Keep in mind that data files distributed alongside McFACTS should be less than a Megabyte.
GitHub and Pypi offer free services.
Large data files slow down these services and create bulk which is hard to reduce in the history of a repository.

..
   List of link shortcuts

.. _`Google Form`: https://docs.google.com/forms/d/e/1FAIpQLSeupzj8ledPslYc0bHbnJHKB7_LKlr8SY3SfbEVyL5AfeFlVg/viewform
.. _`README`: https://github.com/McFACTS/McFACTS/blob/main/README.md
.. _`IOdocumentation.txt`: https://github.com/McFACTS/McFACTS/blob/main/IOdocumentation.txt
.. _`Style Guide`: https://github.com/McFACTS/McFACTS/blob/main/docs/source/style.rst
.. _`Commit Guide`: https://github.com/McFACTS/McFACTS/blob/main/docs/source/commitmsg.rst
.. _`Pull Request Template`: https://github.com/McFACTS/McFACTS/blob/main/.github/PULL_REQUEST_TEMPLATE.md
.. _`Issue Tracker`: https://github.com/McFACTS/McFACTS/issues
.. _`semantic commit messages`: https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716
.. _`MANIFEST.in`: https://github.com/McFACTS/McFACTS/blob/main/MANIFEST.in
.. _`setup.py`: https://github.com/McFACTS/McFACTS/blob/main/setup.py
.. _`docs/source/index.rst`: https://github.com/McFACTS/McFACTS/blob/main/docs/source/index.rst
