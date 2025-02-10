# Configuration file for the Sphinx documentation builder.

# Project information
project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": False,
    "exclude-members": "__init__, __post_init__, __new__",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext"}

pygments_style = "sphinx"

autodoc_type_aliases = {
    "npt.ArrayLike": "ArrayLike",
}

# Options for HTML output
html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "minkdoc"