from __future__ import annotations

# Configuration file for the Sphinx documentation builder.
#
# This file configures the Sphinx documentation generator to build the documentation for the mink project.
# For a comprehensive list of configuration options, refer to the official Sphinx documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project Information -----------------------------------------------------
# This section contains metadata about the project, which is used in various parts of the generated documentation.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project: str = "mink"
copyright: str = "2024, Kevin Zakka"
author: str = "Kevin Zakka"

# -- General Configuration ---------------------------------------------------
# This section configures general settings for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# List of Sphinx extensions to enable
extensions: list[str] = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings
    "sphinx.ext.coverage",  # Check for documentation coverage
    "sphinx-mathjax-offline",  # Use MathJax for rendering math equations offline
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx_favicon",  # Add favicons to the documentation
]

# Paths to templates that the theme can use
templates_path: list[str] = ["_templates"]

# List of patterns to ignore when looking for source files
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# Mapping of source file suffixes to file types
source_suffix: dict[str, str] = {".rst": "restructuredtext"}

# Style for syntax highlighting in the documentation
pygments_style: str = "sphinx"

# Configuration for the Napoleon extension
napoleon_numpy_docstring: bool = False  # Do not use NumPy style docstrings
napoleon_use_rtype: bool = False  # Do not include the return type in the docstring

# -- Options for HTML Output -------------------------------------------------
# This section configures the HTML output options for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme to use for HTML and HTML Help pages
html_theme: str = "sphinx_rtd_theme"

# Name of the HTML help builder output file
htmlhelp_basename: str = "minkdoc"