import os
import sys

# Insert the path to your source code directory (src/) into sys.path.
# This allows Sphinx to find and import your 'bayinx' package.
sys.path.insert(0, os.path.abspath('../../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Bayinx'
copyright = '2025, Todd Pocuca'
author = 'Todd Pocuca'
release = '0.5.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# **EDITED**: Added essential extensions for API documentation and themes
extensions = [
    'sphinx.ext.autodoc',      # Imports and processes docstrings from Python modules
    'sphinx.ext.napoleon',     # Supports Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # Adds links to the source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# **EDITED**: Switched to the 'sphinx_rtd_theme'
html_theme = 'alabaster'
html_static_path = ['_static']
