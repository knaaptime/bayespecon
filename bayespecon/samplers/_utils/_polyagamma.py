"""Vectorized Pólya–Gamma draw wrapper.

Thin wrapper around the ``polyagamma`` package that centralises the call
site and provides a consistent interface. Dispatches to the library's
hybrid sampler (Devroye for integer h, alternate for non-integer h) when
all h values are integer, and forces ``method="alternate"`` otherwise.

If ``polyagamma`` is unavailable, an ``ImportError`` is raised with a
helpful message.
"""

from __future__ import annotations

import numpy as np


def sample_polyagamma(
    h: np.ndarray,
    z: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw vectorized Pólya–Gamma samples.

    For each element i, draws ω_i ~ PG(h_i, z_i) where PG is the
    Pólya–Gamma distribution (Polson, Scott & Windle, 2013).

    Parameters
    ----------
    h : ndarray of shape (n,)
        Shape parameters. For NB augmentation: h_i = y_i + alpha.
        Must be positive.
    z : ndarray of shape (n,)
        Tilting parameters. For NB augmentation: z_i = eta_i.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.

    Returns
    -------
    omega : ndarray of shape (n,)
        PG(h, z) draws. All elements are positive.

    Raises
    ------
    ImportError
        If the ``polyagamma`` package is not installed.
    ValueError
        If ``h`` and ``z`` have different shapes or if any h <= 0.
    """
    try:
        from polyagamma import random_polyagamma as _pg_draw
    except ImportError:
        raise ImportError(
            "The 'polyagamma' package is required for Pólya–Gamma sampling. "
            "Install it with: pip install polyagamma"
        )

    if rng is None:
        rng = np.random.default_rng()

    h = np.asarray(h, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if h.shape != z.shape:
        raise ValueError(
            f"h and z must have the same shape, got {h.shape} and {z.shape}."
        )
    if np.any(h <= 0):
        raise ValueError("All h values must be positive.")

    # Dispatch on h: the hybrid method (method=None, the default) selects the
    # faster Devroye algorithm for integer h (e.g. logit with h=1), but Devroye
    # requires integer-valued h.  For the NB-PG augmentation, h_i = y_i + alpha
    # is typically non-integer, so we must use "alternate" in that case.
    # When all h values are integer, we let the library pick the best method.
    _all_integer = np.all(np.equal(np.mod(h, 1.0), 0.0))
    method = None if _all_integer else "alternate"
    omega = _pg_draw(h=h, z=z, method=method, random_state=rng)
    return np.asarray(omega, dtype=np.float64)
