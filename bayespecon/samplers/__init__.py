"""Custom Gibbs samplers for spatial econometric models.

Subpackages
-----------
gaussian
    Gaussian (SAR/SEM/SDM/SDEM) Gibbs samplers.
panel
    Panel (FE/RE) Gibbs samplers.
negbin
    Negative Binomial (P\u00f3lya-Gamma) Gibbs samplers.
_utils
    Internal utilities (not public API).

Import from the subpackages directly::

    from bayespecon.samplers.gaussian import GaussianSARGibbs
    from bayespecon.samplers.panel import REGibbsEstimation
    from bayespecon.samplers.negbin import run_chain

This module intentionally re-exports nothing: sampler families register
their Gibbs entries on *their own* import, and every model module imports
the family it needs \u2014 keeping ``bayespecon.models`` imports fast.
"""
