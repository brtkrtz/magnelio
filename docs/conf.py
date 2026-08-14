# Sphinx configuration for the Magnelio scientific documentation.
#
# Build:  sphinx-build -b html docs docs/_build/html
# Needs:  sphinx, myst-parser, sphinxcontrib-bibtex, pydata-sphinx-theme,
#         sphinx-gallery (mirrors the pyproject [docs] extra)

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Magnelio"
author = "Bernd Breitkreutz"
copyright = "2026, Bernd Breitkreutz"

try:
    from magnelio._version import __version__ as release
except Exception:
    release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    # Tutorial sources are runnable scripts in the public examples tree;
    # the HTML pages and .ipynb downloads are generated from them.
    "examples_dirs": ["../examples/tutorials"],
    "gallery_dirs": ["tutorials"],
    # Only scripts named plot_* are executed at build time; other
    # scripts are rendered and downloadable without execution.
    "filename_pattern": r"/plot_",
    "download_all_examples": False,
    # The tutorials are a numbered curriculum: order by file name, not
    # by the default code-line count.
    "within_subsection_order": "FileNameSortKey",
    # Strip sphinx_gallery_* config comments (thumbnail selection etc.)
    # from the rendered code blocks.
    "remove_config_comments": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_member_order = "groupwise"
autodoc_typehints = "description"

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = []
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "Magnelio Documentation"
