===============
Commit Messages
===============

See the `Contributing Guide`_ for more
information on making changes to McFACTS.

.. _`Contributing Guide`: https://github.com/McFACTS/McFACTS/blob/main/docs/source/contribute.rst


`Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_ is a formatting convention that enables
us to easily

#. review the code's history
#. automatically generate our changelog

Please follow these guidelines when committing changes as you work on McFACTS.

Formatting Template
*******************

.. code-block:: none

   <type>[optional scope]: <description>

   [optional body]

   [optional footer(s)]

See `Examples`_ below for...examples.

Message Subject
^^^^^^^^^^^^^^^

The first line cannot be longer than 70 characters. The second line is always blank, and other lines should be no
longer than 80 characters.
The ``type`` and ``scope`` must be lower case.

**<type> must be one of:**

* ``feat`` – a new feature is introduced with the changes
* ``fix`` – a bug fix has occurred
* ``chore`` – changes that do not relate to a fix or feature and don't modify src or test files (for example updating dependencies)
* ``refactor`` – refactored code that neither fixes a bug nor adds a feature
* ``docs`` – updates to documentation such as a the README or other markdown files
* ``style`` – changes that do not affect the meaning of the code, likely related to code formatting such as white-space, missing semi-colons, and so on.
* ``test`` – including new or correcting previous tests
* ``perf`` – performance improvements
* ``ci`` – continuous integration related
* ``build`` – changes that affect the build system or external dependencies
* ``revert`` – reverts a previous commit

**[scope] examples:**

* ``phys`` - physics
* ``io`` - input/output
* ``vis`` - visualization
* ``setup``
* ``ext`` - external module interfaces
* etc.

**<description> should be:**

#. present tense
#. gramatically structured to complete the sentence, "If applied, this commit will _____."


The ``[optional body]`` can be used to include additional details when necessary, and \
``[optional footer(s)]`` should be used when the change addresses a reported issue, ``Closes #123``).

See References and Resources for examples.

Examples
***********************

Commit message meeting the minimum requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: none

   style: correct whitespace usage in solve_disk function

Commit message with the optional scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   fix(setup): change sign for binary formation under Snoopy criterion

Commit message with an optional scope and body
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   feat(phys): treat disk capture with Luffy and Zaraki methods

   1. follow Luffy et al. 1999 under X conditions
   2. follow Zaraki et al. 2005 under Y, and Z conditions

Badness 1000
^^^^^^^^^^^^

.. code-block:: none

   updated things again

This one does not follow the format, is not descriptive, and leaves me as a reviewer/bug hunter with a few questions:

1. Did you update the spherical cow approximation for calculating the coriolis force?
2. Was this a simple change to the sign of the integer flag that sets chirality?
3. What did you do last time since you said "again"?

References and Resources:
*************************

#. `https://www.conventionalcommits.org <https://www.conventionalcommits.org/en/v1.0.0/>`_
#. `Semantic Commit Messages <https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716>`_
#. `<https://sparkbox.com/foundry/semantic_commit_messages>`_
#. `<https://karma-runner.github.io/1.0/dev/git-commit-msg.html>`_

