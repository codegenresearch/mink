# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

import toml
from pathlib import Path

# Read project version from pyproject.toml
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
pyproject = toml.load(pyproject_path)
version = pyproject["tool"]["poetry"]["version"]

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"
release = version

# -- General configuration ---------------------------------------------------

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

html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "minkdoc"


### Adjustments Made:
1. **Import Order**: Moved `from pathlib import Path` to be after `import toml` to follow the standard library import order.
2. **Version Extraction**: Simplified the version extraction to directly access the nested dictionary.
3. **Variable Initialization**: Ensured the `version` variable is initialized and formatted consistently.
4. **Whitespace and Formatting**: Ensured consistent spacing and formatting around comments and sections.
5. **Comment Consistency**: Ensured comments are formatted consistently with the gold code, including spacing and line breaks.