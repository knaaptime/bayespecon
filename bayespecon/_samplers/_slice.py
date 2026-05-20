"""Univariate slice sampler with stepping-out and shrinkage.

Implements Neal (2003) slice sampling for 1-D distributions with bounded
or unbounded support. Returns both the new sample and the log-density
value at that sample, allowing callers to cache expensive evaluations
(e.g., log-determinant in the ρ update).

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def slice_sample_1d(
    log_density: Callable[[float], float],
    x0: float,
    lower: float,
    upper: float,
    *,
    w: float = 1.0,
    max_steps_out: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Draw one sample from a univariate distribution via slice sampling.

    Uses Neal's stepping-out procedure to find the slice interval, then
    shrinks to find the new point. The log-density may be called multiple
    times per draw (typically 5–15 calls).

    Parameters
    ----------
    log_density : callable
        Log-density function (up to a normalising constant). Must accept
        a single float and return a float. May be called multiple times.
    x0 : float
        Current state (starting point).
    lower, upper : float
        Support bounds for the variable. Use ``-np.inf`` / ``np.inf``
        for unbounded support.
    w : float, default=1.0
        Initial step-out width for interval expansion.
    max_steps_out : int, default=50
        Maximum number of stepping-out iterations to prevent infinite
        loops on very flat densities.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.

    Returns
    -------
    x_new : float
        New sample point.
    log_density_new : float
        Log-density evaluated at ``x_new``. Callers can cache this to
        avoid redundant evaluations when the log-density is expensive
        (e.g., log-determinant in the ρ update).

    Raises
    ------
    ValueError
        If ``x0`` is not in ``[lower, upper]`` or if ``lower >= upper``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if lower >= upper:
        raise ValueError(f"lower ({lower}) must be less than upper ({upper}).")
    if x0 < lower or x0 > upper:
        raise ValueError(f"x0 ({x0}) must be in [lower, upper] = [{lower}, {upper}].")

    # Evaluate log-density at current point
    log_y0 = log_density(x0)

    # Draw vertical level: log(u) where u ~ Uniform(0, f(x0))
    log_u = log_y0 + np.log(rng.uniform())

    # --- Stepping out ---
    # Initialise interval [L, R] around x0
    u_rand = rng.uniform()
    L = x0 - u_rand * w
    R = L + w

    # Clamp to support bounds
    L = max(L, lower)
    R = min(R, upper)

    # Step out left
    steps_out_left = 0
    while L > lower and log_density(L) > log_u and steps_out_left < max_steps_out:
        L -= w
        L = max(L, lower)
        steps_out_left += 1

    # Step out right
    steps_out_right = 0
    while R < upper and log_density(R) > log_u and steps_out_right < max_steps_out:
        R += w
        R = min(R, upper)
        steps_out_right += 1

    # --- Shrinkage ---
    # Sample from [L, R] and shrink until we find a point above log_u
    while True:
        x_new = L + rng.uniform() * (R - L)
        log_density_new = log_density(x_new)

        if log_density_new > log_u:
            # Accept: point is above the slice
            return x_new, log_density_new

        # Shrink the interval
        if x_new < x0:
            L = x_new
        else:
            R = x_new

        # Safety: if interval has collapsed, return x0
        if R - L < 1e-15:
            return x0, log_y0