"""Tests for AAA rational approximation log-determinant."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from bayespecon._logdet._aaa import (
    AAAPrecompute,
    _aaa_algorithm,
    _lu_logdet,
    aaa_logdet_eval,
    aaa_logdet_eval_vec,
    aaa_logdet_precompute,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_symmetric_W():
    """Small symmetric rook contiguity W (for ground-truth comparison)."""
    from libpysal import graph

    from bayespecon import dgp

    gdf = dgp.simulate_sar(n=20, create_gdf=True)
    W = graph.Graph.build_contiguity(gdf, rook=True).transform("r").sparse.toarray()
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_nonsymmetric_W():
    """Small non-symmetric W (KNN-like directed graph)."""
    np.random.seed(42)
    n = 20
    # Build a directed KNN graph: each node connects to 3 random neighbors
    rows, cols, vals = [], [], []
    for i in range(n):
        # Pick 3 random neighbors (not self)
        neighbors = np.random.choice(
            [j for j in range(n) if j != i], size=3, replace=False
        )
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            vals.append(1.0)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Row-standardize
    degrees = np.array(A.getnnz(axis=1), dtype=np.float64)
    D_inv = sp.diags(1.0 / degrees)
    W = D_inv @ A
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_eigs(small_symmetric_W):
    return np.linalg.eigvals(small_symmetric_W.toarray())


# ---------------------------------------------------------------------------
# LU logdet
# ---------------------------------------------------------------------------


class TestLULogdet:
    def test_matches_eigenvalue(self, small_symmetric_W, small_eigs):
        """LU logdet should match eigenvalue computation."""
        rho = 0.5
        A = sp.eye(small_symmetric_W.shape[0], format="csc") - rho * small_symmetric_W
        lu_ld = _lu_logdet(A)
        eig_ld = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        assert abs(lu_ld - eig_ld) < 1e-8

    def test_nonsymmetric(self, small_nonsymmetric_W):
        """LU logdet should work for non-symmetric W."""
        rho = 0.3
        n = small_nonsymmetric_W.shape[0]
        A = sp.eye(n, format="csc") - rho * small_nonsymmetric_W
        lu_ld = _lu_logdet(A)
        # Compare with numpy dense
        dense_ld = np.linalg.slogdet(A.toarray())[1]
        assert abs(lu_ld - dense_ld) < 1e-6


# ---------------------------------------------------------------------------
# AAA algorithm
# ---------------------------------------------------------------------------


class TestAAAAlgorithm:
    def test_polynomial_function(self):
        """AAA should exactly reproduce a polynomial with enough points."""
        z = np.linspace(-1, 1, 100)
        f = z**2  # simple polynomial
        sp_z, sp_f, w = _aaa_algorithm(z, f, tol=1e-12)
        # Should need very few support points for a polynomial
        assert len(sp_z) <= 10
        # Evaluate at a test point
        rho = 0.5
        diff = rho - sp_z
        n_val = np.sum(w * sp_f / diff)
        d_val = np.sum(w / diff)
        result = n_val / d_val
        assert abs(result - 0.25) < 1e-10

    def test_rational_function(self):
        """AAA should exactly reproduce a rational function."""
        z = np.linspace(0.1, 0.9, 100)
        f = 1.0 / (1.0 - z)  # has a pole at z=1
        sp_z, sp_f, w = _aaa_algorithm(z, f, tol=1e-12)
        # Should converge fast for a simple rational
        assert len(sp_z) <= 10
        # Evaluate
        rho = 0.5
        diff = rho - sp_z
        n_val = np.sum(w * sp_f / diff)
        d_val = np.sum(w / diff)
        result = n_val / d_val
        assert abs(result - 2.0) < 1e-10

    def test_log_function(self):
        """AAA should approximate log(1 - rho*x) well."""
        z = np.linspace(0.1, 0.8, 200)
        x = 0.7  # fixed eigenvalue
        f = np.log(np.abs(1.0 - z * x))
        sp_z, sp_f, w = _aaa_algorithm(z, f, tol=1e-10)
        # Should converge in a few points
        assert len(sp_z) <= 20
        # Check accuracy at several points
        for rho in [0.15, 0.3, 0.45, 0.6, 0.75]:
            diff = rho - sp_z
            n_val = np.sum(w * sp_f / diff)
            d_val = np.sum(w / diff)
            result = n_val / d_val
            exact = np.log(np.abs(1.0 - rho * x))
            assert abs(result - exact) < 1e-8, f"rho={rho}: {result} vs {exact}"


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_returns_precompute(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        assert isinstance(pre, AAAPrecompute)
        assert pre.n == small_symmetric_W.shape[0]
        assert len(pre.support_points) >= 2
        assert len(pre.support_points) == len(pre.support_values)
        assert len(pre.support_points) == len(pre.weights)

    def test_nonsymmetric_W(self, small_nonsymmetric_W):
        """Should work with non-symmetric W."""
        pre = aaa_logdet_precompute(small_nonsymmetric_W, rho_min=0.1, rho_max=0.8)
        assert isinstance(pre, AAAPrecompute)
        assert pre.n == small_nonsymmetric_W.shape[0]

    def test_custom_interval(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.2, rho_max=0.7)
        assert pre.rho_min == 0.2
        assert pre.rho_max == 0.7


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_accuracy_symmetric(self, small_symmetric_W, small_eigs):
        """AAA should match exact logdet for symmetric W."""
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        for rho in [0.15, 0.3, 0.45, 0.6, 0.75]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = aaa_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_accuracy_nonsymmetric(self, small_nonsymmetric_W):
        """AAA should match exact logdet for non-symmetric W."""
        pre = aaa_logdet_precompute(small_nonsymmetric_W, rho_min=0.1, rho_max=0.8)
        n = small_nonsymmetric_W.shape[0]
        for rho in [0.15, 0.3, 0.45, 0.6, 0.75]:
            A = sp.eye(n, format="csc") - rho * small_nonsymmetric_W
            exact = np.linalg.slogdet(A.toarray())[1]
            approx = aaa_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_exact_at_support_points(self, small_symmetric_W, small_eigs):
        """At support points, the approximant should be exact."""
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        for sp_rho, sp_val in zip(pre.support_points, pre.support_values):
            val = aaa_logdet_eval(pre, float(sp_rho))
            assert abs(val - sp_val) < 1e-10


# ---------------------------------------------------------------------------
# Vectorized evaluation
# ---------------------------------------------------------------------------


class TestVectorized:
    def test_vec_matches_scalar(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.15, 0.75, 50)
        vec_vals = aaa_logdet_eval_vec(pre, rhos)
        scalar_vals = np.array([aaa_logdet_eval(pre, r) for r in rhos])
        assert np.allclose(vec_vals, scalar_vals, atol=1e-10)

    def test_vec_shape(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.1, 0.8, 30)
        vals = aaa_logdet_eval_vec(pre, rhos)
        assert vals.shape == (30,)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactory:
    def test_scalar_factory(self, small_symmetric_W, small_eigs):
        fn = make_logdet_numpy_fn(
            small_symmetric_W, small_eigs, method="aaa", rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.3, 0.5, 0.7]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = fn(rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_vec_factory(self, small_symmetric_W, small_eigs):
        fn = make_logdet_numpy_vec_fn(
            small_symmetric_W, small_eigs, method="aaa", rho_min=0.1, rho_max=0.8
        )
        rhos = np.linspace(0.15, 0.75, 40)
        vals = fn(rhos)
        exact = np.array([np.sum(np.log(np.abs(1.0 - r * small_eigs))) for r in rhos])
        assert np.allclose(vals, exact, atol=1e-6)

    def test_factory_nonsymmetric(self, small_nonsymmetric_W):
        """Factory should work with non-symmetric W via AAA."""
        n = small_nonsymmetric_W.shape[0]
        eigs = np.linalg.eigvals(small_nonsymmetric_W.toarray())
        fn = make_logdet_numpy_fn(
            small_nonsymmetric_W, eigs, method="aaa", rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.5]:
            exact = np.sum(np.log(np.abs(1.0 - rho * eigs)))
            approx = fn(rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_factory_T(self, small_symmetric_W, small_eigs):
        fn = make_logdet_numpy_fn(
            small_symmetric_W, small_eigs, method="aaa", rho_min=0.1, rho_max=0.8, T=3
        )
        rho = 0.5
        exact = 3.0 * np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        assert abs(fn(rho) - exact) < 1e-6
