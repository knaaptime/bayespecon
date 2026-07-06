"""Tests for Cholesky-Chebyshev log-determinant."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from bayespecon._logdet._chol_cheb import (
    CholChebPrecompute,
    _adaptive_order,
    _d_symmetrize,
    chol_cheb_logdet_eval,
    chol_cheb_logdet_eval_vec,
    chol_cheb_logdet_precompute,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_W():
    """Small rook contiguity W with known eigenvalues."""
    from libpysal import graph

    from bayespecon import dgp

    gdf = dgp.simulate_sar(n=20, create_gdf=True)
    W = graph.Graph.build_contiguity(gdf, rook=True).transform("r").sparse.toarray()
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_eigs(small_W):
    return np.linalg.eigvals(small_W.toarray())


# ---------------------------------------------------------------------------
# D-symmetrisation
# ---------------------------------------------------------------------------


class TestDSymmetrise:
    def test_symmetric(self, small_W):
        """W_sym should be symmetric."""
        W_sym = _d_symmetrize(small_W)
        dense = W_sym.toarray()
        assert np.allclose(dense, dense.T, atol=1e-12)

    def test_preserves_eigenvalues(self, small_W):
        """W_sym should have the same eigenvalues as W."""
        W_sym = _d_symmetrize(small_W)
        eigs_W = np.sort(np.linalg.eigvals(small_W.toarray()).real)
        eigs_sym = np.sort(np.linalg.eigvals(W_sym.toarray()).real)
        assert np.allclose(eigs_W, eigs_sym, atol=1e-10)

    def test_spectrum_in_unit_circle(self, small_W):
        """Row-standardised W should have eigenvalues in [-1, 1]."""
        eigs = np.linalg.eigvals(small_W.toarray()).real
        assert np.all(np.abs(eigs) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_returns_precompute(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=10)
        assert isinstance(pre, CholChebPrecompute)
        assert pre.order == 10
        assert pre.n == small_W.shape[0]
        assert pre.coeffs.shape == (10,)

    def test_custom_interval(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=8, rho_min=0.2, rho_max=0.7)
        assert pre.rho_min == 0.2
        assert pre.rho_max == 0.7

    def test_accepts_dense(self, small_W):
        """Should accept dense input as well as sparse."""
        pre_dense = chol_cheb_logdet_precompute(small_W.toarray(), order=8)
        pre_sparse = chol_cheb_logdet_precompute(small_W, order=8)
        assert np.allclose(pre_dense.coeffs, pre_sparse.coeffs, atol=1e-8)


# ---------------------------------------------------------------------------
# Adaptive order selection
# ---------------------------------------------------------------------------


class TestAdaptiveOrder:
    def test_standard_range(self):
        """Narrow interval [0.1, 0.8] should use order=15."""
        assert _adaptive_order(0.1, 0.8) == 15

    def test_wide_range(self):
        """Wider interval should use higher order."""
        assert _adaptive_order(-0.5, 0.95) == 30

    def test_full_theoretical_range(self):
        """[-0.95, 0.95] should use order=50."""
        assert _adaptive_order(-0.95, 0.95) == 50

    def test_extreme_range(self):
        """[-0.99, 0.99] should use order=100."""
        assert _adaptive_order(-0.99, 0.99) == 100

    def test_auto_order_used_when_none(self, small_W):
        """When order=None, the precompute should auto-select based on interval."""
        pre = chol_cheb_logdet_precompute(
            small_W, order=None, rho_min=-0.95, rho_max=0.95
        )
        assert pre.order == 50

        pre = chol_cheb_logdet_precompute(small_W, order=None, rho_min=0.1, rho_max=0.8)
        assert pre.order == 15

    def test_explicit_order_overrides_auto(self, small_W):
        """Explicit order should override auto-selection."""
        pre = chol_cheb_logdet_precompute(small_W, order=8, rho_min=-0.95, rho_max=0.95)
        assert pre.order == 8


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_exact_at_nodes(self, small_W, small_eigs):
        """At Chebyshev nodes, the polynomial should match the exact logdet."""
        order = 12
        pre = chol_cheb_logdet_precompute(
            small_W, order=order, rho_min=0.1, rho_max=0.8
        )

        # Recompute the nodes
        k = np.arange(1, order + 1)
        nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
        rho_nodes = 0.5 * (0.8 - 0.1) * nodes_cos + 0.5 * (0.8 + 0.1)

        for rho in rho_nodes:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, float(rho))
            assert abs(approx - exact) < 1e-8, f"rho={rho}: {approx} vs {exact}"

    def test_near_exact_between_nodes(self, small_W, small_eigs):
        """Between nodes, Chebyshev interpolation should be very accurate."""
        pre = chol_cheb_logdet_precompute(small_W, order=15, rho_min=0.1, rho_max=0.8)

        for rho in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, rho)
            # Chebyshev interpolation error should be tiny for smooth functions
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_zero_rho(self, small_W):
        """At ρ=0, log|I| = 0.  Chebyshev extrapolation to endpoint should be close."""
        pre = chol_cheb_logdet_precompute(small_W, order=15, rho_min=0.01, rho_max=0.8)
        val = chol_cheb_logdet_eval(pre, 0.0)
        assert abs(val) < 1e-3

    def test_monotone(self, small_W):
        """log|det(I - ρW)| should be monotonically decreasing for ρ > 0
        (as ρ→1, det→0 so logdet→-∞)."""
        pre = chol_cheb_logdet_precompute(small_W, order=15, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.15, 0.75, 20)
        vals = [chol_cheb_logdet_eval(pre, r) for r in rhos]
        diffs = np.diff(vals)
        assert np.all(diffs < 0), f"Not monotone decreasing: diffs={diffs}"

    def test_full_theoretical_range(self, small_W, small_eigs):
        """Adaptive order should give good accuracy across [-0.95, 0.95]."""
        pre = chol_cheb_logdet_precompute(
            small_W, order=None, rho_min=-0.95, rho_max=0.95
        )
        assert pre.order == 50
        for rho in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-4, f"rho={rho}: {approx} vs {exact}"

    def test_negative_rho(self, small_W, small_eigs):
        """Negative ρ should work — I - ρW is still SPD for |ρ| < 1."""
        pre = chol_cheb_logdet_precompute(small_W, order=20, rho_min=-0.8, rho_max=-0.1)
        for rho in [-0.7, -0.5, -0.3, -0.15]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"


# ---------------------------------------------------------------------------
# Vectorized evaluation
# ---------------------------------------------------------------------------


class TestVectorized:
    def test_vec_matches_scalar(self, small_W):
        """Vectorized eval should match scalar eval for each element."""
        pre = chol_cheb_logdet_precompute(small_W, order=12, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.15, 0.75, 50)
        vec_vals = chol_cheb_logdet_eval_vec(pre, rhos)
        scalar_vals = np.array([chol_cheb_logdet_eval(pre, r) for r in rhos])
        assert np.allclose(vec_vals, scalar_vals, atol=1e-12)

    def test_vec_shape(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=10)
        rhos = np.linspace(0.1, 0.8, 30)
        vals = chol_cheb_logdet_eval_vec(pre, rhos)
        assert vals.shape == (30,)

    def test_vec_empty(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=10)
        vals = chol_cheb_logdet_eval_vec(pre, np.array([]))
        assert vals.shape == (0,)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactory:
    def test_scalar_factory(self, small_W, small_eigs):
        """make_logdet_numpy_fn with cheb_cholesky should match exact logdet."""
        fn = make_logdet_numpy_fn(
            small_W, small_eigs, method="cheb_cholesky", rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.3, 0.5, 0.7]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = fn(rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_vec_factory(self, small_W, small_eigs):
        """make_logdet_numpy_vec_fn with cheb_cholesky should match exact logdet."""
        fn = make_logdet_numpy_vec_fn(
            small_W, small_eigs, method="cheb_cholesky", rho_min=0.1, rho_max=0.8
        )
        rhos = np.linspace(0.15, 0.75, 40)
        vals = fn(rhos)
        exact = np.array([np.sum(np.log(np.abs(1.0 - r * small_eigs))) for r in rhos])
        assert np.allclose(vals, exact, atol=1e-6)

    def test_factory_T(self, small_W, small_eigs):
        """T multiplier should scale the output."""
        fn = make_logdet_numpy_fn(
            small_W, small_eigs, method="cheb_cholesky", rho_min=0.1, rho_max=0.8, T=3
        )
        rho = 0.5
        exact = 3.0 * np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        assert abs(fn(rho) - exact) < 1e-6

    def test_auto_select_midrange(self):
        """Auto-select should pick cheb_cholesky for n in (500, 20000]."""
        from bayespecon._logdet import resolve_logdet_method

        assert resolve_logdet_method(None, n=501) == "cheb_cholesky"
        assert resolve_logdet_method(None, n=1000) == "cheb_cholesky"
        assert resolve_logdet_method(None, n=10000) == "cheb_cholesky"
        assert resolve_logdet_method(None, n=20000) == "cheb_cholesky"
        assert resolve_logdet_method(None, n=50000) == "cheb_stochastic"
