import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'McFacts'
copyright = '2024, McFacts Contributors'
author = 'McFacts Contributors'
release = '0.0.0'

sys.path.insert(0, os.path.abspath('../../src'))

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add napoleon to the extensions list
extensions = [
    'sphinx.ext.napoleon',
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'matplotlib': ('https://matplotlib.org/stable', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy', None),
                       'astropy': ('https://docs.astropy.org/en/stable', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None)}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for auto summary

autoclass_content = 'both'
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_docstring_signature = True
autosummary_mock_imports = ['pagn']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {"display_version": True, "logo_only" : True}
html_last_updated_fmt = "%Y %b %d at %H:%M:%S UTC"
html_show_sourcelink = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
