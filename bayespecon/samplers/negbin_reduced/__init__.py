"""Pólya–Gamma Gibbs sampler for the reduced-form SAR Negative Binomial.

This sampler targets the canonical spatial-econometric SAR-NB model

.. math::

    y_i \\sim \\mathrm{NegBin}(\\mu_i, \\alpha), \\qquad
    \\mu = \\exp\\{(I - \\rho W)^{-1} X\\beta\\}

— that is, spatial dependence enters only through the *mean propagator*
:math:`(I - \\rho W)^{-1}` and overdispersion is handled entirely by the
NB dispersion parameter :math:`\\alpha`.  There is **no** latent spatial
random effect (no :math:`\\sigma`).  Contrast with
:mod:`bayespecon.samplers.negbin`, which samples the structural-form
model :math:`\\eta = \\rho W \\eta + X\\beta + \\nu,\\; \\nu \\sim N(0,
\\sigma^2 I)`.

The reduced form is the form most commonly written down in the spatial
econometrics literature (LeSage & Pace 2009), and is the preferred
default when you do not have substantive reason to model an unobserved
spatially-smoothed latent field.
"""

from ._core import (
    ReducedGibbsCache,
    ReducedGibbsPriors,
    ReducedGibbsState,
    ReducedKrylovBasis,
    run_chain,
)
from ._flow import (
    FlowReducedGibbsCache,
    FlowReducedGibbsPriors,
    FlowReducedGibbsState,
    run_chain_separable,
    run_chain_unrestricted,
)
from ._jax import run_chains_jax_reduced

__all__ = [
    "ReducedGibbsCache",
    "ReducedGibbsPriors",
    "ReducedGibbsState",
    "run_chain",
    "FlowReducedGibbsCache",
    "FlowReducedGibbsPriors",
    "FlowReducedGibbsState",
    "run_chain_separable",
    "run_chain_unrestricted",
]
