"""Panel (FE/RE) Gibbs samplers.

This subpackage provides Gibbs samplers for panel spatial models with
random effects, along with Kronecker-product utilities for panel
log-likelihood computation.

Public API
----------
REGibbsEstimation
    Base class for random-effects Gibbs sampler configuration and execution.
    (Formerly ``REGBibbsEstimation`` — typo corrected.)
GaussianSARREGibbs
    Gibbs sampler for SAR panel models with random effects.
GaussianSEMREGibbs
    Gibbs sampler for SEM panel models with random effects.
REGibbsPriors
    Dataclass holding prior hyperparameters for the RE Gibbs sampler.
"""

from ._re_estimation import REGibbsEstimation, GaussianSARREGibbs, GaussianSEMREGibbs
from ._re_core import REGibbsPriors

__all__ = [
    "REGibbsEstimation",
    "GaussianSARREGibbs",
    "GaussianSEMREGibbs",
    "REGibbsPriors",
]