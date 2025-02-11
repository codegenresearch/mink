# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import toml

# Read project version from pyproject.toml
pyproject_path: Path = Path("../pyproject.toml")
pyproject = toml.load(pyproject_path)
project_version: str = pyproject["tool"]["poetry"]["version"]

project: str = "mink"
copyright: str = "2024, Kevin Zakka"
author: str = "Kevin Zakka"
version: str = f"v{project_version}" if not project_version.startswith("v") else project_version
release: str = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

# Include both type hints in the signature and description
autodoc_typehints: str = "both"
# Separate class signature and docstring
autodoc_class_signature: str = "separated"
# Default options for autodoc
autodoc_default_options: dict[str, bool | str | list[str]] = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": False,
    "exclude-members": "__init__, __post_init__, __new__",
}

# Paths to templates
templates_path: list[str] = ["_templates"]
# Patterns to exclude from source files
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix: dict[str, str] = {".rst": "restructuredtext"}

# Pygments syntax highlighting style
pygments_style: str = "sphinx"

# Type aliases for autodoc
autodoc_type_aliases: dict[str, str] = {
    "npt.ArrayLike": "ArrayLike",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme for HTML output
html_theme: str = "sphinx_rtd_theme"
# Base name for HTML help builder output
htmlhelp_basename: str = "minkdoc"