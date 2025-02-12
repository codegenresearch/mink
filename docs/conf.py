# Configuration file for the Sphinx documentation builder.
#
# For a comprehensive list of built-in configuration values, refer to the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project Information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project_name = "mink"
project_copyright = "2024, Kevin Zakka"
project_author = "Kevin Zakka"

# -- General Configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# List of Sphinx extensions to enable
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

# Template file directories
templates_directories = ["_templates"]

# Directories to exclude from the build process
directories_to_exclude = ["_build", "Thumbs.db", ".DS_Store"]

# Mapping of file extensions to their corresponding source parsers
source_file_suffixes = {".rst": "restructuredtext"}

# Pygments syntax highlighting style
pygments_syntax_style = "sphinx"

# Disable NumPy style docstrings
numpy_style_docstrings_enabled = False

# Disable return type annotations in docstrings
return_type_annotations_enabled = False

# -- Options for HTML Output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# HTML theme to use for the documentation
html_documentation_theme = "sphinx_rtd_theme"

# Base name for the HTML help builder output
html_help_documentation_base_name = "minkdoc"