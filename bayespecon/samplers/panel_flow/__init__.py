"""Gaussian panel flow Gibbs sampler with Kronecker eigenbasis FFBS.

Provides a Gibbs sampler for Gaussian spatial interaction panel models
with AR(1) temporal dynamics and separable Kronecker origin-destination
spatial structure. The key algorithmic innovation is the eigenbasis
decomposition that reduces the :math:`n^2`-dimensional Kalman filter to
:math:`n^2` independent scalar filters, giving :math:`O(n^2 T)` cost
per FFBS draw.

Public API
----------
run_gaussian_panel_flow_chain
    Top-level chain runner with setup, warmup, and sampling.
PanelGaussianState
    Mutable state for one Gibbs sweep.
PanelGaussianCache
    Immutable precomputed constants.
PanelGaussianPriors
    Prior hyperparameters.
PanelGaussianTrace
    Posterior draws storage.
"""

from ._chain import run_gaussian_panel_flow_chain
from ._state import (
    PanelGaussianCache,
    PanelGaussianPriors,
    PanelGaussianState,
    PanelGaussianTrace,
)

# JAX-native path (conditional import)
try:
    from ._chain_jax import run_gaussian_panel_flow_chain_jax
    from ._state_jax import JAXPanelGaussianState
except ImportError:
    pass

__all__ = [
    "run_gaussian_panel_flow_chain",
    "PanelGaussianState",
    "PanelGaussianCache",
    "PanelGaussianPriors",
    "PanelGaussianTrace",
]

# Add JAX exports if available
try:
    from ._chain_jax import run_gaussian_panel_flow_chain_jax  # noqa: F811
    from ._state_jax import JAXPanelGaussianState  # noqa: F811

    __all__ += [
        "run_gaussian_panel_flow_chain_jax",
        "JAXPanelGaussianState",
    ]
except ImportError:
    pass
