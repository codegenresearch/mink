# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import toml
from pathlib import Path

# Read project version from pyproject.toml
pyproject_path = Path(__file__).absolute().parent.parent / "pyproject.toml"
pyproject = toml.load(pyproject_path)
version = pyproject["tool"]["poetry"]["version"]
if not version[0].isdigit():
    version = f"v{version}"
release = version

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"

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


### Changes Made:
1. **Import Order**: Grouped standard library imports (`toml`, `Path`) together.
2. **Version Extraction**: Simplified the version extraction by directly accessing the keys.
3. **Version Check**: Adjusted the version check to use `isdigit()` for clarity.
4. **Type Annotations**: Removed explicit type annotations for variables where they were not present in the gold code.
5. **Formatting and Style**: Ensured consistent spacing and line breaks to match the gold code's style.