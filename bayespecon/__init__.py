"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes and Bayesian specification tests.

Submodules and attributes are loaded lazily following SPEC 1
(https://scientific-python.org/specs/spec-0001/) so that ``import bayespecon``
is cheap and does not eagerly import ``pymc``/``pytensor``/``arviz``. The
public API surface is declared in the sibling ``__init__.pyi`` stub for
static type checkers and IDE autocomplete.

Examples
--------
Import a model class directly from the package namespace::

        from bayespecon import SAR
"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespecon")
