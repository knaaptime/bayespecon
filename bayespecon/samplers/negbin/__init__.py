"""Negative Binomial (Pólya-Gamma) Gibbs samplers.

This subpackage provides Pólya-Gamma augmented Gibbs samplers for
spatial count models (NegBin), with both NumPy and JAX backends.

Public API
----------
GibbsState
    State container for the cross-sectional PG Gibbs sampler.
GibbsPriors
    Prior hyperparameters for the cross-sectional PG Gibbs sampler.
GibbsCache
    Precomputed cache for the cross-sectional PG Gibbs sampler.
run_chain
    Run a single cross-sectional PG Gibbs chain.
JAXGibbsState
    JAX-compatible state container (equinox Module).
"""

from ._core import GibbsCache, GibbsPriors, GibbsState, JAXGibbsState, run_chain

__all__ = [
    "GibbsState",
    "GibbsPriors",
    "GibbsCache",
    "run_chain",
    "JAXGibbsState",
    "FlowGibbsState",
    "FlowGibbsPriors",
    "FlowGibbsCache",
    "FlowGibbsSliceState",
    "FlowGibbsCacheNS",
    "FlowGibbsSliceStateNS",
    "run_flow_chain_separable",
    "run_flow_chain_nonseparable",
]
