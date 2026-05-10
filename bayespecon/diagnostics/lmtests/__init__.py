"""Bayesian LM test functions

- :mod:`bayespecon.diagnostics.lmtests.core` — shared helpers
  (:class:`BayesianLMTestResult`, ``_compute_residuals``, ``_lm_scalar``, etc.)
- :mod:`bayespecon.diagnostics.lmtests.cross_sectional` — cross-sectional tests
  (OLS-null, SAR-null, SLX-null, SDM-null, SDEM-null)
- :mod:`bayespecon.diagnostics.lmtests.panel` — panel tests
  (pooled/FE/RE variants)
- :mod:`bayespecon.diagnostics.lmtests.flow` — flow-of-flows tests
  (destination/origin/network/intra)

All public test functions are re-exported from this package root.

Symbols are loaded lazily via ``lazy_loader`` (SPEC 1).
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
