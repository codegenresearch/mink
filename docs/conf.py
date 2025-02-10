# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import toml

# Read project version from pyproject.toml
pyproject_path = Path(__file__).absolute().parent.parent / "pyproject.toml"
pyproject = toml.load(pyproject_path)
version = pyproject["tool"]["poetry"]["version"]

# Prepend 'v' to the version if it is not purely alphabetical
if not version.isalpha():
    version = f"v{version}"

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "minkdoc"