"""Fast edge-case tests for logdet utilities."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import logdet


def _toy_w(n: int = 5) -> np.ndarray:
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    return W / np.where(rs == 0, 1, rs)


def test_stable_rho_grid_validation():
    with pytest.raises(ValueError, match="positive"):
        logdet._stable_rho_grid(-0.5, 0.5, 0.0)
    with pytest.raises(ValueError, match="greater"):
        logdet._stable_rho_grid(0.5, 0.5, 0.1)


def test_lndetmc_parameter_validation():
    W = _toy_w(4)
    with pytest.raises(ValueError, match="order"):
        logdet.mc(order=0, iter=5, W=W)
    with pytest.raises(ValueError, match="iter"):
        logdet.mc(order=5, iter=0, W=W)
    with pytest.raises(ValueError, match="nonnegative rho"):
        logdet.mc(order=5, iter=5, W=W, rmin=-0.1, rmax=0.5)


def test_lndetint_validation():
    W = _toy_w(4)
    with pytest.raises(ValueError, match="nonnegative rho"):
        logdet.spline(W, rmin=-0.1, rmax=0.5, n_grid=50)
    with pytest.raises(ValueError, match="at least 20"):
        logdet.spline(W, rmin=0.0, rmax=0.5, n_grid=10)


def test_make_logdet_fn_unknown_method_raises():
    W = _toy_w(4)
    with pytest.raises(ValueError, match="Unknown method"):
        logdet.make_logdet_fn(W, method="not-a-method")


def test_chebyshev_parameter_validation():
    W = _toy_w(4)
    with pytest.raises(ValueError, match="order must be positive"):
        logdet.chebyshev(W, order=0)
    with pytest.raises(ValueError, match="rmax must be greater than rmin"):
        logdet.chebyshev(W, order=10, rmin=0.5, rmax=0.5)
    with pytest.raises(ValueError, match="rmax must be greater than rmin"):
        logdet.chebyshev(W, order=10, rmin=0.5, rmax=0.3)


def test_make_logdet_fn_1d_eigs_falls_back_to_eigenvalue_for_grid_like_methods():
    W = _toy_w(4)
    eigs = np.linalg.eigvals(W).real
    fn = logdet.make_logdet_fn(eigs, method="dense_grid")
    assert callable(fn)


def test_make_logdet_fn_int_mc_reject_negative_rho_min_for_matrix_input():
    W = _toy_w(4)
    with pytest.raises(ValueError, match="nonnegative rho"):
        logdet.make_logdet_fn(W, method="spline", rho_min=-0.5, rho_max=0.5)
    with pytest.raises(ValueError, match="nonnegative rho"):
        logdet.make_logdet_fn(W, method="mc", rho_min=-0.5, rho_max=0.5)
