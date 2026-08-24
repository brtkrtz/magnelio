# Sphinx configuration for the Magnelio scientific documentation.
#
# Build:  sphinx-build -b html docs docs/_build/html
# Needs:  sphinx, myst-parser, sphinxcontrib-bibtex, pydata-sphinx-theme,
#         sphinx-gallery (mirrors the pyproject [docs] extra)

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath("../src"))


def _format_warning(message, category, filename, lineno, line=None):
    """Render a warning as category and text, without its origin.

    Sphinx-gallery writes every warning an executed tutorial raises
    into the rendered page, through ``warnings.formatwarning``.  The
    default form leads with the *absolute* path of the file that
    raised and echoes its source line, so a published page ends up
    carrying the build machine's directory layout and a fragment of
    library internals.  Neither belongs there: these warnings are
    about the reader's model, not about our source, and the message
    is written to stand on its own.

    This is global for the build, so a Python-level warning from an
    extension loses its location in the build log too.  Sphinx's own
    warnings — the ones a docs build is actually diagnosed by — go
    through its logger and are unaffected.
    """
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _format_warning

project = "Magnelio"
author = "Bernd Breitkreutz"
copyright = "2026, Bernd Breitkreutz"

# The published site carries two channels, each a full build in its own
# directory (see .github/workflows/docs.yml): "stable" from the newest
# release tag, "dev" from main.  The build learns which one it is from
# the environment; anything unset — a local build, a fork's CI — is dev.
docs_channel = os.environ.get("MAGNELIO_DOCS_CHANNEL", "dev")
docs_base_url = "https://brtkrtz.github.io/magnelio"

try:
    from magnelio._version import __version__ as release
except Exception:
    release = "0.0.0"

# main keeps the version of the last release until the next bump, so the
# dev build would otherwise present itself as that release.  The local
# version segment says what it is, and the theme reads the word "dev"
# out of this string to word its warning banner.
if docs_channel != "stable":
    release = f"{release}+dev"

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
    # Gallery sources are runnable scripts in the public examples tree;
    # the HTML pages and .ipynb downloads are generated from them.
    # Tutorials are the ordered curriculum; how-to guides are
    # unordered task recipes (unnumbered file names, alphabetical).
    "examples_dirs": ["../examples/tutorials", "../examples/howto"],
    "gallery_dirs": ["tutorials", "howto"],
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
html_baseurl = f"{docs_base_url}/{docs_channel}/"
html_theme_options = {
    "switcher": {
        # Absolute, and served from the site *root* rather than from a
        # channel's own _static: a released build is frozen the day it
        # is published, so a switcher shipped inside it would forever
        # list the channels that existed back then.  Reading one shared
        # file keeps every published page's menu current.
        "json_url": f"{docs_base_url}/switcher.json",
        "version_match": docs_channel,
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The banner is meant for the dev channel, to tell the reader it
    # documents unreleased code.  It cannot decide that by itself: the
    # theme compares this build's ``release`` against the ``version``
    # field of the switcher entry marked preferred, and only when *both*
    # parse as release numbers.  Ours are channel names, so the
    # comparison never runs and the banner would appear on every page of
    # every channel — including the released one, where its "switch to
    # stable" button links to the page it is already on.  Decide here.
    "show_version_warning_banner": docs_channel != "stable",
}
