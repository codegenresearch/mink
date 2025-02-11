# Configuration file for the Sphinx documentation builder.
#
# For a comprehensive list of built-in configuration values, refer to the official documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project Information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project_name = "mink"
project_copyright = "2024, Kevin Zakka"
project_author = "Kevin Zakka"

# -- General Configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

template_paths = ["_templates"]
excluded_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix_mapping = {".rst": "restructuredtext"}

pygments_style_name = "sphinx"

napoleon_use_numpy_docstring = False
napoleon_include_return_type = False

# -- Options for HTML Output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme_name = "sphinx_rtd_theme"

html_help_basename = "mink_documentation"