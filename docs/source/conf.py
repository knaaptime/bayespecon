# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

from packaging.version import Version

# Ensure pyproj uses the active environment's proj.db during notebook execution.
try:
    from pathlib import Path

    from pyproj import datadir as pyproj_datadir

    candidates = [
        Path(sys.prefix) / "share" / "proj",
        Path(pyproj_datadir.get_data_dir()),
    ]
    for candidate in candidates:
        proj_db = candidate / "proj.db"
        if proj_db.exists():
            proj_data_dir = str(candidate)
            os.environ["PROJ_LIB"] = proj_data_dir
            os.environ["PROJ_DATA"] = proj_data_dir
            break
except Exception:
    pass

import bayespecon

# Diagnostic: surface the underlying ImportError when sphinx autosummary fails
# to import bayespecon submodules on CI. Remove once the docs build is green.
import traceback as _tb

for _modname in (
    "bayespecon.models.flow",
    "bayespecon.models.flow_panel",
    "bayespecon.diagnostics.lmtests",
):
    try:
        __import__(_modname)
        print(f"[conf.py diagnostic] imported {_modname}")
    except Exception:  # noqa: BLE001
        print(f"[conf.py diagnostic] FAILED to import {_modname}:")
        _tb.print_exc()

project = "bayespecon"
copyright = "2018-, pysal developers"  # noqa: A001 - shadowing a Python builtin
author = "pysal developers"

version = Version(bayespecon.__version__).public  # remove commit hash
release = version

language = "en"
html_title = project

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_immaterial",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

bibtex_bibfiles = ["_static/references.bib"]
bibtex_reference_style = "author_year"

master_doc = "index"

templates_path = [
    "_templates",
]
exclude_patterns = []

intersphinx_mapping = {
    "geopandas": ("https://geopandas.org/en/latest", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "pymc": ("https://docs.pymc.io/en/stable", None),
    "libpysal": ("https://pysal.org/libpysal/stable/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
}
suppress_warnings = ["ref.ref"]

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/pysal_logo.svg"
html_favicon = "_static/pysal_favicon.ico"
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-code",
    },
    "site_url": "https://pysal.org/bayespecon",
    "repo_url": "https://github.com/pysal/bayespecon/",
    "edit_uri": "blob/main/docs",
    "repo_name": "pysal/bayespecon",
    "features": [
        # "navigation.expand",
        # "navigation.tabs",
        # "navigation.tabs.sticky",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.footer",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
        # "content.tabs.link",
        "content.code.copy",
        "content.action.edit",
        # "content.action.view",
        # "content.tooltips",
        # "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "black",
            "accent": "red",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "red",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
    "version_dropdown": True,
    "version_json": "https://pysal.org/bayespecon/versions.json",
}
nb_execution_mode = "force"
nb_execution_timeout = -1
nb_kernel_rgx_aliases = {".*": "python3"}
nb_merge_streams = True
nb_execution_raise_on_error = True
nb_execution_show_tb = True
autodoc_typehints = "none"


def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(bayespecon.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "bayespecon/%s#L%d-L%d" % find_source()  # noqa: UP031
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "main" if "dev" in release else ("v" + release)
    return f"https://github.com/pysal/bayespecon/blob/{tag}/{filename}"
