"""Vectorized Pólya–Gamma draw wrapper.

Thin wrapper around the ``polyagamma`` package that centralises the call
site and provides a consistent interface. If ``polyagamma`` is unavailable,
a ``ImportError`` is raised with a helpful message.

The wrapper exists so that future fallback implementations (e.g., a numba
Devroye sampler) can be swapped in without touching call sites.
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

    # random_polyagamma broadcasts h and z, and accepts a Generator via random_state.
    # Use method="alternate" because the default hybrid may select the "devroye"
    # method for small h, which requires integer-valued h.  In the NB-PG augmentation
    # h_i = y_i + alpha is typically non-integer, so we must avoid devroye.
    omega = _pg_draw(h=h, z=z, method="alternate", random_state=rng)
    return np.asarray(omega, dtype=np.float64)
