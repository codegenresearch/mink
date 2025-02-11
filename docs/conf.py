# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import toml

# Read project version from pyproject.toml
with open("../pyproject.toml", "r") as file:
    pyproject = toml.load(file)
project_version = pyproject["tool"]["poetry"]["version"]

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"
version = project_version
release = project_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

# Include both type hints in the signature and description
autodoc_typehints = "both"
# Separate class signature and docstring
autodoc_class_signature = "separated"
# Default options for autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": False,
    "exclude-members": "__init__, __post_init__, __new__",
}

# Paths to templates
templates_path = ["_templates"]
# Patterns to exclude from source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {".rst": "restructuredtext"}

# Pygments syntax highlighting style
pygments_style = "sphinx"

# Type aliases for autodoc
autodoc_type_aliases = {
    "npt.ArrayLike": "ArrayLike",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme for HTML output
html_theme = "sphinx_rtd_theme"
# Base name for HTML help builder output
htmlhelp_basename = "minkdoc"