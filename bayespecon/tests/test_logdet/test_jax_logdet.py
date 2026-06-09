"""Tests for JAX-native logdet evaluation functions.

Tests ``jax_logdet_chebyshev``, ``jax_logdet_trace_poly``, and
``make_logdet_jax_fn`` against exact eigenvalue-based log-determinants.

All tests are skipped when JAX is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

# Skip entire module if JAX is not available
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX not installed",
)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bayespecon._logdet import (
    compute_flow_traces,
    jax_logdet_chebyshev,
    jax_logdet_trace_poly,
    make_logdet_jax_fn,
)
from bayespecon._logdet._grids import chebyshev

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_w() -> np.ndarray:
    """Small row-stochastic matrix with spectral radius <= 1."""
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def _exact_logdet(rho: float, W: np.ndarray) -> float:
    """Exact log|I - rho*W| via numpy slogdet."""
    n = W.shape[0]
    return float(np.linalg.slogdet(np.eye(n) - rho * W)[1])


def _exact_logdet_eigenvalue(rho: float, eigs: np.ndarray) -> float:
    """Exact log|I - rho*W| from eigenvalues."""
    return float(np.sum(np.log(np.abs(1.0 - rho * eigs))))


# ---------------------------------------------------------------------------
# jax_logdet_chebyshev
# ---------------------------------------------------------------------------


class TestJaxLogdetChebyshev:
    """Tests for jax_logdet_chebyshev()."""

    def test_scalar_matches_exact(self):
        """Chebyshev JAX evaluation matches exact for small matrix."""
        W = _toy_w()
        out = chebyshev(W, order=20, rmin=-0.5, rmax=0.5)
        coeffs = out["coeffs"]
        rmin, rmax = out["rmin"], out["rmax"]

        for rho in [0.0, 0.1, 0.3, -0.3]:
            approx = float(jax_logdet_chebyshev(jnp.float64(rho), coeffs, rmin, rmax))
            exact = _exact_logdet(rho, W)
            assert abs(approx - exact) < 0.05, (
                f"rho={rho}: cheb={approx:.6f}, exact={exact:.6f}"
            )

    def test_vectorized_matches_scalar(self):
        """Vectorized evaluation matches element-wise scalar evaluation."""
        W = _toy_w()
        out = chebyshev(W, order=20, rmin=-0.5, rmax=0.5)
        coeffs = out["coeffs"]
        rmin, rmax = out["rmin"], out["rmax"]

        rho_arr = jnp.array([-0.3, -0.1, 0.0, 0.1, 0.3])
        vec_result = jax_logdet_chebyshev(rho_arr, coeffs, rmin, rmax)

        for i, rho in enumerate(rho_arr):
            scalar_result = jax_logdet_chebyshev(rho, coeffs, rmin, rmax)
            np.testing.assert_allclose(vec_result[i], scalar_result, rtol=1e-12)

    def test_jit_compilable(self):
        """Function compiles inside jax.jit without errors."""
        W = _toy_w()
        out = chebyshev(W, order=20, rmin=-0.5, rmax=0.5)
        coeffs = out["coeffs"]
        rmin, rmax = out["rmin"], out["rmax"]

        @jax.jit
        def compiled(rho):
            return jax_logdet_chebyshev(rho, coeffs, rmin, rmax)

        result = float(compiled(jnp.float64(0.3)))
        exact = _exact_logdet(0.3, W)
        assert abs(result - exact) < 0.05

    def test_autodiff_works(self):
        """JAX grad produces finite gradients."""
        W = _toy_w()
        out = chebyshev(W, order=20, rmin=-0.5, rmax=0.5)
        coeffs = out["coeffs"]
        rmin, rmax = out["rmin"], out["rmax"]

        grad_fn = jax.grad(lambda rho: jax_logdet_chebyshev(rho, coeffs, rmin, rmax))
        g = grad_fn(jnp.float64(0.3))
        assert jnp.isfinite(g)

    def test_single_coefficient(self):
        """m=1 returns c_0 (constant)."""
        coeffs = np.array([5.0])
        result = jax_logdet_chebyshev(jnp.float64(0.3), coeffs, rmin=-1.0, rmax=1.0)
        np.testing.assert_allclose(float(result), 5.0, rtol=1e-12)

    def test_empty_coefficients(self):
        """m=0 returns zero."""
        coeffs = np.array([])
        result = jax_logdet_chebyshev(jnp.float64(0.3), coeffs, rmin=-1.0, rmax=1.0)
        np.testing.assert_allclose(float(result), 0.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# jax_logdet_trace_poly
# ---------------------------------------------------------------------------


class TestJaxLogdetTracePoly:
    """Tests for jax_logdet_trace_poly()."""

    @pytest.fixture()
    def trace_data(self):
        """Compute trace estimates for a small test matrix."""
        rng = np.random.default_rng(42)
        n = 10
        W_dense = rng.random((n, n))
        W_dense /= W_dense.sum(axis=1, keepdims=True)
        W_sp = sp.csr_matrix(W_dense)
        traces = compute_flow_traces(W_sp, miter=30, riter=50, random_state=0)
        eigs = np.linalg.eigvals(W_dense).real
        return W_dense, traces, eigs

    def test_matches_eigenvalue_within_tolerance(self, trace_data):
        """Trace polynomial matches exact eigenvalue log-det within 5%."""
        W_dense, traces, eigs = trace_data

        for rho in [0.05, 0.2, 0.4, 0.6]:
            approx = float(jax_logdet_trace_poly(jnp.float64(rho), traces))
            exact = _exact_logdet_eigenvalue(rho, eigs)
            rel_err = abs(approx - exact) / (abs(exact) + 1e-12)
            assert rel_err < 0.05, (
                f"rho={rho}: trace_poly={approx:.6f}, exact={exact:.6f}, "
                f"rel_err={rel_err:.4f}"
            )

    def test_vectorized_matches_scalar(self, trace_data):
        """Vectorized evaluation matches element-wise scalar evaluation."""
        _, traces, _ = trace_data

        rho_arr = jnp.array([0.05, 0.2, 0.4, 0.6])
        vec_result = jax_logdet_trace_poly(rho_arr, traces)

        for i, rho in enumerate(rho_arr):
            scalar_result = jax_logdet_trace_poly(rho, traces)
            np.testing.assert_allclose(vec_result[i], scalar_result, rtol=1e-12)

    def test_jit_compilable(self, trace_data):
        """Function compiles inside jax.jit without errors."""
        _, traces, eigs = trace_data

        @jax.jit
        def compiled(rho):
            return jax_logdet_trace_poly(rho, traces)

        result = float(compiled(jnp.float64(0.3)))
        exact = _exact_logdet_eigenvalue(0.3, eigs)
        rel_err = abs(result - exact) / (abs(exact) + 1e-12)
        assert rel_err < 0.02

    def test_autodiff_works(self, trace_data):
        """JAX grad produces finite gradients."""
        _, traces, _ = trace_data

        grad_fn = jax.grad(lambda rho: jax_logdet_trace_poly(rho, traces))
        g = grad_fn(jnp.float64(0.3))
        assert jnp.isfinite(g)

    def test_empty_traces(self):
        """Empty trace array returns zero."""
        traces = np.array([])
        result = jax_logdet_trace_poly(jnp.float64(0.3), traces)
        np.testing.assert_allclose(float(result), 0.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# make_logdet_jax_fn
# ---------------------------------------------------------------------------


class TestMakeLogdetJaxFn:
    """Tests for make_logdet_jax_fn()."""

    def test_eigenvalue_method(self):
        """Eigenvalue method matches exact logdet."""
        W = _toy_w()

        fn = make_logdet_jax_fn(W, method="eigenvalue")

        for rho in [0.0, 0.1, 0.3, -0.3]:
            approx = float(fn(jnp.float64(rho)))
            exact = _exact_logdet(rho, W)
            np.testing.assert_allclose(approx, exact, rtol=1e-10)

    def test_chebyshev_method(self):
        """Chebyshev method matches exact within approximation tolerance."""
        W = _toy_w()

        fn = make_logdet_jax_fn(W, method="chebyshev", rho_min=-0.5, rho_max=0.5)

        for rho in [0.0, 0.1, 0.3, -0.3]:
            approx = float(fn(jnp.float64(rho)))
            exact = _exact_logdet(rho, W)
            assert abs(approx - exact) < 0.05, (
                f"rho={rho}: cheb={approx:.6f}, exact={exact:.6f}"
            )

    def test_trace_mc_method(self):
        """chebyshev + hutchinson matches exact within Chebyshev tolerance."""
        rng = np.random.default_rng(42)
        n = 10
        W_dense = rng.random((n, n))
        W_dense /= W_dense.sum(axis=1, keepdims=True)
        eigs = np.linalg.eigvals(W_dense).real

        fn = make_logdet_jax_fn(W_dense, method="chebyshev")

        for rho in [0.05, 0.2, 0.4]:
            approx = float(fn(jnp.float64(rho)))
            exact = _exact_logdet_eigenvalue(rho, eigs)
            rel_err = abs(approx - exact) / (abs(exact) + 1e-12)
            assert rel_err < 0.05, (
                f"rho={rho}: trace_mc={approx:.6f}, exact={exact:.6f}"
            )

    def test_eigenvalue_from_eigs_array(self):
        """Passing 1-D eigenvalue array skips eigendecomposition."""
        W = _toy_w()
        eigs = np.linalg.eigvals(W).real

        fn = make_logdet_jax_fn(eigs, method="eigenvalue")

        for rho in [0.0, 0.1, 0.3, -0.3]:
            approx = float(fn(jnp.float64(rho)))
            exact = _exact_logdet(rho, W)
            np.testing.assert_allclose(approx, exact, rtol=1e-10)

    def test_panel_multiplier(self):
        """T > 1 multiplies the logdet by T."""
        W = _toy_w()

        fn_t1 = make_logdet_jax_fn(W, method="eigenvalue", T=1)
        fn_t3 = make_logdet_jax_fn(W, method="eigenvalue", T=3)

        rho = jnp.float64(0.3)
        np.testing.assert_allclose(
            float(fn_t3(rho)), 3.0 * float(fn_t1(rho)), rtol=1e-10
        )

    def test_jit_compilable(self):
        """Returned function works inside jax.jit."""
        W = _toy_w()
        fn = make_logdet_jax_fn(W, method="eigenvalue")

        @jax.jit
        def compiled(rho):
            return fn(rho)

        result = float(compiled(jnp.float64(0.3)))
        exact = _exact_logdet(0.3, W)
        np.testing.assert_allclose(result, exact, rtol=1e-10)

    def test_autodiff_works(self):
        """JAX grad works through the returned function."""
        W = _toy_w()
        fn = make_logdet_jax_fn(W, method="eigenvalue")

        grad_fn = jax.grad(fn)
        g = grad_fn(jnp.float64(0.3))
        assert jnp.isfinite(g)

    def test_chebyshev_autodiff(self):
        """JAX grad works through Chebyshev method."""
        W = _toy_w()
        fn = make_logdet_jax_fn(W, method="chebyshev", rho_min=-0.5, rho_max=0.5)

        grad_fn = jax.grad(fn)
        g = grad_fn(jnp.float64(0.3))
        assert jnp.isfinite(g)

    def test_unsupported_method_raises(self):
        """Grid/spline methods raise ValueError."""
        W = _toy_w()
        with pytest.raises(ValueError, match="does not have a JAX-native"):
            make_logdet_jax_fn(W, method="grid_dense")

    def test_sparse_input(self):
        """Sparse W input works for eigenvalue method."""
        W = _toy_w()
        W_sp = sp.csr_matrix(W)

        fn = make_logdet_jax_fn(W_sp, method="eigenvalue")

        for rho in [0.0, 0.1, 0.3]:
            approx = float(fn(jnp.float64(rho)))
            exact = _exact_logdet(rho, W)
            np.testing.assert_allclose(approx, exact, rtol=1e-10)

    def test_chebyshev_gradient_matches_eigenvalue(self):
        """Chebyshev gradient is close to eigenvalue gradient."""
        W = _toy_w()
        fn_eig = make_logdet_jax_fn(W, method="eigenvalue")
        fn_cheb = make_logdet_jax_fn(W, method="chebyshev", rho_min=-0.5, rho_max=0.5)

        rho = jnp.float64(0.3)
        grad_eig = jax.grad(fn_eig)(rho)
        grad_cheb = jax.grad(fn_cheb)(rho)

        # Chebyshev gradient should be close to exact eigenvalue gradient
        np.testing.assert_allclose(float(grad_cheb), float(grad_eig), rtol=0.05)
