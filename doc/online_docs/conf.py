# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import packages_distributions, version as distribution_version

# The import package this documentation describes. The distribution that ships
# it is resolved below rather than hardcoded, so packaging renames do not stale
# the docs.
import_package = 'PharmaPy'

# Resolve the installed distribution *before* the source tree goes on sys.path
# below. A checkout can carry build metadata from an earlier editable install --
# in particular a stale `PharmaPy.egg-info` predating the rename to
# `pharmapy-sim` -- and importlib.metadata treats any `*.egg-info` on sys.path as
# an installed distribution. Resolving first keeps that stale metadata out of the
# lookup; doing it after the insert reports both names and fails the build.
#
# `packages_distributions` was added in Python 3.10, so building the docs
# requires 3.10 or newer even though `[project].requires-python` allows 3.9 for
# the package itself. CI, Read the Docs, and the pixi `docs` environment pin 3.11.
distribution_names = sorted(set(packages_distributions().get(import_package, [])))
if len(distribution_names) != 1:
    raise RuntimeError(
        f"Expected exactly one installed distribution providing "
        f"{import_package!r}, found {distribution_names!r}. Build the "
        f"documentation from an environment where PharmaPy is installed, for "
        f"example `pixi run docs` or `pip install -e \".[docs]\"`. If several "
        f"names are listed, delete stale `*.egg-info` directories from the "
        f"repository root and reinstall."
    )

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
# sys.path.insert(0, os.path.abspath('sphinxext'))


# Setup of mock libraries to override the build fails for C based libraries

#import mock

autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "pandas", "autograd", "assimulo", "cyipopt"]
 
#MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate', 'assimulo']
#for mod_name in MOCK_MODULES:
# sys.modules[mod_name] = mock.Mock()


# -- Project information -----------------------------------------------------

project = import_package
copyright = '2023, Purdue University and the PharmaPy contributors'
author = 'The original PharmaPy developers and PharmaPy-org contributors'

# The full version, including alpha/beta/rc tags, read from the distribution
# resolved during path setup above.
release = distribution_version(distribution_names[0])

version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.doctest', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'nbsphinx', 'sphinxcontrib.bibtex' 
]

bibtex_bibfiles = ['references.bib']

# Render the example notebooks from their committed outputs instead of executing
# them. nbsphinx defaults to 'auto', which executes any notebook that has no
# outputs; under the warning-as-error docs build (`-W` in CI,
# `fail_on_warning: true` on Read the Docs) an execution error in a notebook
# would then fail the build of an unrelated change. Keeping this 'never' makes
# the build depend only on what is committed. `ipykernel` stays in the docs
# dependencies so contributors can regenerate those committed notebook outputs
# in the locked documentation environment.
nbsphinx_execute = 'never'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'images/PharmaPy_logo.jpeg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['images']
html_context = {
    "footer_logos": {
        "row1": [
            {
                "alt": "Purdue University Logo",
                # "src": "Purdue_footer_logo.png",
                "src": "purdue_logo.png",
                "href": "https://www.purdue.edu/",
            },
            {
                "alt": "U.S. Food and Drug Administration Logo",
                "src": "fda_logo.png",
                "href": "https://www.fda.gov/",
            },
        ],
    }
}
