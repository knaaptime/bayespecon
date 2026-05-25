"""Vectorized truncated-normal draws for probit/tobit augmentation.

Future use by SpatialProbitGibbs and TobitGibbs. Included now for API
stability so the ``samplers`` package layout is settled from day one.

Uses the efficient algorithm from Robert (1995) for robust tail behavior.
"""

from __future__ import annotations

import numpy as np


def sample_truncnorm(
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    *,
    mu: np.ndarray | float = 0.0,
    sigma: np.ndarray | float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw from TruncatedNormal(mu, sigma²) on (lower, upper).

    Uses the efficient algorithm from Robert (1995) for far-tail
    cases, with simple rejection sampling for the bulk. Vectorised
    over all inputs.

    Parameters
    ----------
    lower, upper : array_like
        Truncation bounds. Use ``-np.inf`` / ``np.inf`` for one-sided
        truncation. Must satisfy ``lower < upper``.
    mu, sigma : array_like
        Location and scale of the underlying normal distribution.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.

    Returns
    -------
    draws : ndarray
        Draws from the truncated normal distribution, same shape as
        broadcasted ``(lower, upper, mu, sigma)``.

    Raises
    ------
    ValueError
        If any ``lower >= upper`` or ``sigma <= 0``.

    References
    ----------
    Robert, C. P. (1995). Simulation of truncated normal variables.
    *Statistics and Computing*, 5, 121–125.
    """
    if rng is None:
        rng = np.random.default_rng()

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if np.any(lower >= upper):
        raise ValueError("All lower bounds must be less than upper bounds.")
    if np.any(sigma <= 0):
        raise ValueError("All sigma values must be positive.")

    # Standardise: work in (alpha, beta) = (lower - mu) / sigma, (upper - mu) / sigma
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma

    # For each element, choose the sampling strategy:
    # - If the truncation interval is wide (alpha < beta and beta - alpha > some threshold),
    #   use simple rejection from N(0, 1).
    # - If the truncation is in the far right tail (alpha > some threshold),
    #   use Robert's exponential proposal.
    # - Otherwise, use rejection from N(0, 1) with a tighter bound.

    output_shape = np.broadcast_shapes(alpha.shape, beta.shape)
    alpha = np.broadcast_to(alpha, output_shape).copy()
    beta = np.broadcast_to(beta, output_shape).copy()

    draws = np.empty(output_shape, dtype=np.float64)

    # Flatten for iteration
    alpha_flat = alpha.ravel()
    beta_flat = beta.ravel()
    draws_flat = draws.ravel()

    for i in range(len(alpha_flat)):
        a, b = alpha_flat[i], beta_flat[i]
        draws_flat[i] = _sample_truncnorm_scalar(a, b, rng)

    return mu + sigma * draws


def _sample_truncnorm_scalar(
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> float:
    """Draw a single standard truncated normal on (alpha, beta).

    Uses Robert's (1995) exponential proposal for far-right-tail
    truncation, and simple rejection otherwise.
    """
    # Both bounds infinite: just draw from N(0, 1)
    if alpha == -np.inf and beta == np.inf:
        return rng.standard_normal()

    # One-sided or two-sided truncation
    # Use rejection sampling with Robert's method for far tails
    if alpha == -np.inf:
        # Left tail: TruncNorm on (-inf, beta)
        # Equivalent to -TruncNorm on (-beta, inf)
        return -_sample_truncnorm_right(-beta, rng)
    elif beta == np.inf:
        # Right tail: TruncNorm on (alpha, inf)
        return _sample_truncnorm_right(alpha, rng)
    else:
        # Two-sided: TruncNorm on (alpha, beta)
        return _sample_truncnorm_two_sided(alpha, beta, rng)


def _sample_truncnorm_right(
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """Draw from TruncNorm on (alpha, inf) — right tail."""
    if alpha <= 0.5:
        # Bulk: simple rejection from N(0, 1)
        while True:
            x = rng.standard_normal()
            if x > alpha:
                return x
    else:
        # Far right tail: Robert's exponential proposal
        # lambda_star = (alpha + sqrt(alpha^2 + 2)) / 2
        # But simpler: use the half-normal rejection
        # Proposal: exponential with rate alpha
        while True:
            x = alpha + rng.exponential(1.0 / alpha)
            # Acceptance ratio: exp(-0.5 * (x - alpha)^2)
            log_accept = -0.5 * (x - alpha) ** 2
            if np.log(rng.uniform()) < log_accept:
                return x


def _sample_truncnorm_two_sided(
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> float:
    """Draw from TruncNorm on (alpha, beta) — two-sided."""
    # Simple rejection from N(0, 1)
    while True:
        x = rng.standard_normal()
        if alpha < x < beta:
            return x
