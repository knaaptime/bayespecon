"""Tests for Stochastic Lanczos Quadrature (SLQ) logdet."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet import (
    SLQPrecompute,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
    slq_logdet_eval,
    slq_logdet_eval_vec,
    slq_logdet_precompute,
)


def _toy_w(n: int = 50, seed: int = 0) -> sp.csr_matrix:
    """Small row-standardised sparse W for testing."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    W = (1.0 / d) * (d < np.quantile(d, 0.2))
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs
    return sp.csr_matrix(W)


def _exact_logdet(rho: float, W: sp.csr_matrix) -> float:
    eigs = np.linalg.eigvals(W.toarray())
    return float(np.sum(np.log(np.abs(1.0 - rho * eigs))))


class TestSLQPrecompute:
    def test_returns_precompute_object(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        assert isinstance(pre, SLQPrecompute)
        assert pre.n_probes == 5
        assert pre.lanczos_deg == 10
        assert pre.n == 30

    def test_nodes_and_weights_shapes(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        assert pre.nodes.shape == (5, 10)
        assert pre.weights.shape == (5, 10)


class TestSLQEval:
    def test_eval_returns_float(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=10, lanczos_deg=20)
        val = slq_logdet_eval(pre, 0.3)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_eval_at_zero_is_zero(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        assert abs(slq_logdet_eval(pre, 0.0)) < 1e-10

    def test_eval_vec_matches_scalar(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=15)
        rhos = np.array([0.1, 0.3, 0.5])
        vec_result = slq_logdet_eval_vec(pre, rhos)
        scalar_results = np.array([slq_logdet_eval(pre, r) for r in rhos])
        np.testing.assert_allclose(vec_result, scalar_results, rtol=1e-10)

    def test_eval_vec_shape(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        rhos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = slq_logdet_eval_vec(pre, rhos)
        assert result.shape == (5,)


class TestSLQFactoryIntegration:
    def test_make_logdet_numpy_fn_slq(self):
        W = _toy_w(30)
        fn = make_logdet_numpy_fn(W, eigs=None, method="slq")
        val = fn(0.3)
        assert np.isfinite(val)

    def test_make_logdet_numpy_vec_fn_slq(self):
        W = _toy_w(30)
        fn = make_logdet_numpy_vec_fn(W, eigs=None, method="slq")
        rhos = np.array([0.1, 0.3, 0.5])
        result = fn(rhos)
        assert result.shape == (3,)

    def test_make_logdet_fn_slq_pytensor(self):
        import pytensor

        W = _toy_w(30)
        fn = make_logdet_fn(W, method="slq")
        import pytensor.tensor as pt

        rho_t = pt.dscalar()
        expr = fn(rho_t)
        f = pytensor.function([rho_t], expr)
        val = f(0.3)
        assert np.isfinite(val)

    def test_auto_select_slq_for_large_n(self):
        from bayespecon._logdet import resolve_logdet_method

        # SLQ is opt-in, not auto-selected
        assert resolve_logdet_method(None, n=10000) == "cheb_stochastic"
        assert resolve_logdet_method("slq", n=10000) == "slq"
