"""Custom Gibbs samplers for non-Gaussian spatial models.

This package provides block-Gibbs sampling primitives for models where
NUTS performs poorly due to posterior geometry (banana-shaped correlations,
non-conjugate likelihoods, etc.). The primary entry point is
:func:`pg_gibbs`, which implements the Pólya–Gamma Gibbs sampler for
structural-form SAR Negative Binomial.

Internal primitives (prefixed with ``_``) are designed for reuse by
future Gibbs samplers (SpatialProbitGibbs, TobitGibbs, etc.).
"""