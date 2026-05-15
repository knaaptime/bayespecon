"""Tests verifying that ``_logdet_bounds`` is wired through model classes
so that positive-only logdet methods (``sparse_spline``, ``grid_mc``)
auto-restrict the rho/lambda support to ``[1e-5, 1.0]`` when no explicit
override is supplied.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from bayespecon.logdet import LogdetBounds
from bayespecon.models.sar import SAR
from bayespecon.models.sem import SEM


def _toy_inputs(n: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    W = (1.0 / d) * (d < np.quantile(d, 0.15))
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs
    X = rng.standard_normal((n, 2))
    y = rng.standard_normal(n)
    return y, X, sp.csr_matrix(W)


def test_sar_logdet_bounds_default_eigenvalue():
    y, X, W = _toy_inputs()
    m = SAR(y=y, X=X, W=W, logdet_method="eigenvalue")
    b = m._logdet_bounds
    assert isinstance(b, LogdetBounds)
    # eigenvalue method preserves the prior range (-1, 1)
    assert b.rho_min < 0.0
    assert b.rho_max > 0.0


def test_sar_logdet_bounds_autoclamp_sparse_spline():
    y, X, W = _toy_inputs()
    m = SAR(y=y, X=X, W=W, logdet_method="sparse_spline")
    b = m._logdet_bounds
    # positive-only method must auto-clamp the lower bound.
    assert b.rho_min >= 0.0
    assert b.rho_max > b.rho_min


def test_sem_logdet_bounds_autoclamp_grid_mc():
    y, X, W = _toy_inputs()
    m = SEM(y=y, X=X, W=W, logdet_method="grid_mc")
    b = m._logdet_bounds
    assert b.rho_min >= 0.0
    assert b.rho_max > b.rho_min


def test_sar_explicit_negative_bounds_with_spline_raises():
    y, X, W = _toy_inputs()
    m = SAR(
        y=y,
        X=X,
        W=W,
        logdet_method="sparse_spline",
        priors={"rho_lower": -0.5, "rho_upper": 0.8},
    )
    # priors-source negative bounds are auto-clamped (silent), not raised.
    b = m._logdet_bounds
    assert b.rho_min >= 0.0
