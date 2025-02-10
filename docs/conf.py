# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import toml

# Read project version from pyproject.toml
pyproject_path: Path = Path(__file__).parent.parent / "pyproject.toml"
pyproject = toml.load(pyproject_path)
project_version: str = pyproject["tool"]["poetry"]["version"]

# Ensure version string is prefixed with 'v' if it's not purely alphabetical
if not project_version.isalpha():
    version: str = f"v{project_version}"
else:
    version: str = project_version

project: str = "mink"
copyright: str = "2024, Kevin Zakka"
author: str = "Kevin Zakka"
release: str = version

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