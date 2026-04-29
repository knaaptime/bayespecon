"""Model class exports for bayespecon.

This subpackage groups cross-sectional and panel spatial model classes under a
single import surface. Submodules and classes are loaded lazily — see SPEC 1
(https://scientific-python.org/specs/spec-0001/) — so importing this package
does not eagerly import ``pymc``/``pytensor``.
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
