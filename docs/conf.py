# Configuration file for the Sphinx documentation builder.
#
# For a comprehensive list of built-in configuration values, refer to the official documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project Information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"

# -- General Configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext"}

pygments_style = "sphinx"

napoleon_numpy_docstring = False
napoleon_use_rtype = False

# Autodoc configuration
autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "inherited-members": True,
    "exclude-members": "__weakref__",
}

# Autodoc type aliases
autodoc_type_aliases = {
    "MyType": "int",
}

# -- Options for HTML Output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

htmlhelp_basename = "minkdoc"