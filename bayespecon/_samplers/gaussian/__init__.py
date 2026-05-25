"""Gaussian (SAR/SEM/SDM/SDEM) Gibbs samplers.

This subpackage provides the 3-block Gaussian Gibbs sampler for spatial
autoregressive models, along with the JAX-accelerated backend and
chain-running orchestration.

Public API
----------
GibbsEstimation
    Base class for Gaussian Gibbs sampler configuration and execution.
GaussianSARGibbs
    Gibbs sampler for SAR and SDM models (lag dependence).
GaussianSEMGibbs
    Gibbs sampler for SEM and SDEM models (error dependence).
GaussianGibbsPriors
    Dataclass holding prior hyperparameters for the Gaussian Gibbs sampler.
"""

from ._estimation import GibbsEstimation, GaussianSARGibbs, GaussianSEMGibbs
from ._core import GaussianGibbsPriors

__all__ = [
    "GibbsEstimation",
    "GaussianSARGibbs",
    "GaussianSEMGibbs",
    "GaussianGibbsPriors",
]