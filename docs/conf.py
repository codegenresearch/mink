# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import toml
from pathlib import Path

# Read project version from pyproject.toml
pyproject_path: Path = Path(__file__).absolute().parent.parent / "pyproject.toml"
pyproject = toml.load(pyproject_path)
version: str = pyproject["tool"]["poetry"]["version"]
if not version[0].isnumeric():
    version = f"v{version}"
release: str = version

project: str = "mink"
copyright: str = "2024, Kevin Zakka"
author: str = "Kevin Zakka"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

autodoc_typehints: str = "both"
autodoc_class_signature: str = "separated"
autodoc_default_options: dict[str, bool | str | list[str]] = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": False,
    "exclude-members": "__init__, __post_init__, __new__",
}

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix: dict[str, str] = {".rst": "restructuredtext"}
pygments_style: str = "sphinx"

autodoc_type_aliases: dict[str, str] = {
    "npt.ArrayLike": "ArrayLike",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme: str = "sphinx_rtd_theme"
htmlhelp_basename: str = "minkdoc"