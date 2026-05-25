"""Negative Binomial (Pólya-Gamma) Gibbs samplers.

This subpackage provides Pólya-Gamma augmented Gibbs samplers for
spatial count models (NegBin, Poisson), including cross-sectional
and flow variants, with both NumPy and JAX backends.

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
FlowGibbsState
    State container for the flow PG Gibbs sampler.
FlowGibbsPriors
    Prior hyperparameters for the flow PG Gibbs sampler.
FlowGibbsCache
    Precomputed cache for the flow PG Gibbs sampler.
run_flow_chain_separable
    Run a single flow PG Gibbs chain (separable model).
run_flow_chain_nonseparable
    Run a single flow PG Gibbs chain (non-separable model).
"""

from ._core import GibbsCache, GibbsPriors, GibbsState, JAXGibbsState, run_chain
from ._flow import (
    FlowGibbsCache,
    FlowGibbsCacheNS,
    FlowGibbsPriors,
    FlowGibbsSliceState,
    FlowGibbsSliceStateNS,
    FlowGibbsState,
    run_flow_chain_nonseparable,
    run_flow_chain_separable,
)

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
