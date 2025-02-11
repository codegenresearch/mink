# Configuration file for the Sphinx documentation builder.
#
# This file configures the Sphinx documentation generator to build the documentation for the mink project.
# For a comprehensive list of configuration options, refer to the official Sphinx documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project Information -----------------------------------------------------
# This section contains metadata about the project, which is used in various parts of the generated documentation.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"

# -- General Configuration ---------------------------------------------------
# This section configures general settings for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# List of Sphinx extensions to enable
extensions = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings
    "sphinx.ext.coverage",  # Check for documentation coverage
    "sphinx-mathjax-offline",  # Use MathJax for rendering math equations offline
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx_favicon",  # Add favicons to the documentation
]

# Paths to templates that the theme can use
templates_path = ["_templates"]

# List of patterns to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mapping of source file suffixes to their corresponding parsers
source_suffix = {".rst": "restructuredtext"}

# Style for syntax highlighting in the documentation
pygments_style = "sphinx"

# Configuration for the Napoleon extension
napoleon_numpy_docstring = False  # Do not use NumPy style docstrings
napoleon_use_rtype = False  # Do not include the return type in the docstring

# Configuration for the autodoc extension
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Type aliases for autodoc
autodoc_type_aliases = {}

# -- Options for HTML Output -------------------------------------------------
# This section configures the HTML output options for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme to use for HTML and HTML Help pages
html_theme = "sphinx_rtd_theme"

# Name of the HTML help builder output file
htmlhelp_basename = "minkdoc"