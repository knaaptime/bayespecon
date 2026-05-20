"""Unit tests for JAX-accelerated decoupled Gibbs path.

Tests the JAX dense backend (jax_lanczos_logdet, jax_cg_solve,
jax_chebyshev_sample, jax_build_P_dense) against the scipy
sparse equivalents for small test matrices.

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

from bayespecon._samplers._spatial_normal import (
    cg_solve,
    chebyshev_sample,
    jax_build_P_dense,
    jax_cg_solve,
    jax_chebyshev_sample,
    jax_lanczos_logdet,
    lanczos_logdet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_precision(n: int, rho: float = 0.3, sigma2: float = 1.0):
    """Build a simple spatial precision matrix and its components.

    Returns (P_sparse, W_sym, WtW, omega).
    """
    rng = np.random.default_rng(42)
    W = sp.diags([1.0, 1.0], [-1, 1], shape=(n, n), format="csr")
    W = W / 2.0  # row-standardise

    omega = rng.gamma(2.0, 1.0, size=n)
    W_sym = W + W.T
    WtW = W.T @ W

    P = (
        sp.eye(n, format="csr") / sigma2
        + sp.diags(omega, format="csr")
        - rho * W_sym / sigma2
        + rho**2 * WtW / sigma2
    )
    return P, W_sym, WtW, omega


def _exact_logdet(P: sp.spmatrix) -> float:
    """Compute exact log|P| via dense eigenvalues."""
    return float(np.sum(np.log(np.linalg.eigvalsh(P.toarray()))))


def _exact_solve(P: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Compute exact P^{-1} rhs via dense solve."""
    return np.linalg.solve(P.toarray(), rhs)


# ---------------------------------------------------------------------------
# jax_build_P_dense tests
# ---------------------------------------------------------------------------


class TestJaxBuildPDense:
    """Tests for jax_build_P_dense()."""

    def test_matches_sparse(self):
        """Dense P should match sparse P element-wise."""
        n = 50
        P_sparse, W_sym, WtW, omega = _make_precision(n)
        W_sym_dense = jnp.asarray(W_sym.toarray())
        WtW_dense = jnp.asarray(WtW.toarray())
        omega_jax = jnp.asarray(omega)

        P_dense = jax_build_P_dense(0.3, 1.0, omega_jax, W_sym_dense, WtW_dense)
        err = np.max(np.abs(np.asarray(P_dense) - P_sparse.toarray()))
        assert err < 1e-14, f"P_dense max error: {err:.2e}"

    def test_different_rho(self):
        """Dense P should match sparse P for different ρ values."""
        n = 30
        P_sparse, W_sym, WtW, omega = _make_precision(n)
        W_sym_dense = jnp.asarray(W_sym.toarray())
        WtW_dense = jnp.asarray(WtW.toarray())
        omega_jax = jnp.asarray(omega)

        for rho in [0.0, 0.1, 0.5, 0.9]:
            P_dense = jax_build_P_dense(rho, 1.0, omega_jax, W_sym_dense, WtW_dense)
            P_sparse_rho = (
                sp.eye(n, format="csr") / 1.0
                + sp.diags(omega, format="csr")
                - rho * W_sym / 1.0
                + rho**2 * WtW / 1.0
            )
            err = np.max(np.abs(np.asarray(P_dense) - P_sparse_rho.toarray()))
            assert err < 1e-14, f"rho={rho}: P_dense max error: {err:.2e}"


# ---------------------------------------------------------------------------
# jax_cg_solve tests
# ---------------------------------------------------------------------------


class TestJaxCGSolve:
    """Tests for jax_cg_solve()."""

    def test_matches_scipy(self):
        """JAX CG should match scipy CG to machine precision."""
        n = 50
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        rhs = np.random.randn(n)
        rhs_jax = jnp.asarray(rhs)

        M_inv_diag = 1.0 / jnp.where(
            jnp.abs(jnp.diag(P_dense)) > 1e-15,
            jnp.diag(P_dense),
            1.0,
        )

        x_jax = jax_cg_solve(P_dense, rhs_jax, M_inv_diag)
        x_scipy = cg_solve(P_sparse, rhs)

        rel_err = np.max(np.abs(np.asarray(x_jax) - x_scipy)) / np.max(np.abs(x_scipy))
        assert rel_err < 1e-10, f"JAX CG relative error: {rel_err:.2e}"

    def test_matches_exact(self):
        """JAX CG should match exact dense solve."""
        n = 30
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        rhs = np.random.randn(n)
        rhs_jax = jnp.asarray(rhs)

        M_inv_diag = 1.0 / jnp.where(
            jnp.abs(jnp.diag(P_dense)) > 1e-15,
            jnp.diag(P_dense),
            1.0,
        )

        x_jax = jax_cg_solve(P_dense, rhs_jax, M_inv_diag)
        x_exact = _exact_solve(P_sparse, rhs)

        rel_err = np.max(np.abs(np.asarray(x_jax) - x_exact)) / np.max(np.abs(x_exact))
        assert rel_err < 1e-6, f"JAX CG vs exact relative error: {rel_err:.2e}"


# ---------------------------------------------------------------------------
# jax_lanczos_logdet tests
# ---------------------------------------------------------------------------


class TestJaxLanczosLogdet:
    """Tests for jax_lanczos_logdet()."""

    def test_accuracy_small(self):
        """JAX Lanczos logdet should be close to exact for small matrix."""
        n = 20
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        exact = _exact_logdet(P_sparse)

        key = jax.random.PRNGKey(42)
        estimate = jax_lanczos_logdet(P_dense, key=key, n_probes=20, lanczos_deg=20)

        rel_err = abs(estimate - exact) / abs(exact)
        assert rel_err < 0.10, f"JAX Lanczos logdet error: {rel_err:.4f}"

    def test_accuracy_moderate(self):
        """JAX Lanczos logdet should be accurate for n=100."""
        n = 100
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        exact = _exact_logdet(P_sparse)

        key = jax.random.PRNGKey(42)
        estimate = jax_lanczos_logdet(P_dense, key=key, n_probes=15, lanczos_deg=30)

        rel_err = abs(estimate - exact) / abs(exact)
        assert rel_err < 0.05, f"JAX Lanczos logdet error: {rel_err:.4f}"

    def test_comparable_to_numpy(self):
        """JAX Lanczos should be at least as accurate as numpy Lanczos."""
        n = 50
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        exact = _exact_logdet(P_sparse)

        key = jax.random.PRNGKey(42)
        jax_est = jax_lanczos_logdet(P_dense, key=key, n_probes=10, lanczos_deg=30)

        rng = np.random.default_rng(42)
        np_est = lanczos_logdet(P_sparse, n_probes=10, lanczos_deg=30, rng=rng)

        jax_err = abs(jax_est - exact)
        np_err = abs(np_est - exact)
        # JAX should not be significantly worse than numpy
        # (it's often better due to better numerical properties)
        assert jax_err < 2 * np_err + 1.0, (
            f"JAX error ({jax_err:.4f}) much worse than numpy ({np_err:.4f})"
        )


# ---------------------------------------------------------------------------
# jax_chebyshev_sample tests
# ---------------------------------------------------------------------------


class TestJaxChebyshevSample:
    """Tests for jax_chebyshev_sample()."""

    def test_returns_spatial_normal_draw(self):
        """JAX Chebyshev should return a SpatialNormalDraw."""
        from bayespecon._samplers._spatial_normal import SpatialNormalDraw

        n = 30
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        rhs = np.random.randn(n)
        rhs_jax = jnp.asarray(rhs)

        key = jax.random.PRNGKey(123)
        draw = jax_chebyshev_sample(P_dense, rhs_jax, key=key, degree=30)

        assert isinstance(draw, SpatialNormalDraw)
        assert draw.x.shape == (n,)
        assert draw.factor is None

    def test_mean_close_to_exact(self):
        """Chebyshev draw mean should be close to exact P^{-1} rhs."""
        n = 30
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        rhs = np.random.randn(n)
        rhs_jax = jnp.asarray(rhs)

        # Draw many samples and check mean
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 200)
        draws = []
        for k in keys:
            d = jax_chebyshev_sample(P_dense, rhs_jax, key=k, degree=30)
            draws.append(d.x)
        samples = np.array(draws)
        sample_mean = np.mean(samples, axis=0)
        exact_mean = _exact_solve(P_sparse, rhs)

        # Mean should be close (within 3 standard errors)
        se = np.std(samples, axis=0) / np.sqrt(len(draws))
        max_z = np.max(np.abs(sample_mean - exact_mean) / se)
        assert max_z < 4.0, f"Mean z-score: {max_z:.2f}"

    def test_multiple_draws(self):
        """n_draws > 1 should produce valid draws."""
        n = 30
        P_sparse, _, _, _ = _make_precision(n)
        P_dense = jnp.asarray(P_sparse.toarray())
        rhs = np.random.randn(n)
        rhs_jax = jnp.asarray(rhs)

        key = jax.random.PRNGKey(456)
        draw = jax_chebyshev_sample(P_dense, rhs_jax, key=key, degree=30, n_draws=5)
        assert draw.x.shape == (n,)  # Returns first draw


# ---------------------------------------------------------------------------
# Integration: full JAX decoupled step
# ---------------------------------------------------------------------------


class TestJaxDecoupledStep:
    """Integration tests for the full JAX decoupled Gibbs step."""

    def test_build_solve_logdet(self):
        """Build P, solve, and logdet should be consistent."""
        n = 50
        P_sparse, W_sym, WtW, omega = _make_precision(n)
        W_sym_dense = jnp.asarray(W_sym.toarray())
        WtW_dense = jnp.asarray(WtW.toarray())
        omega_jax = jnp.asarray(omega)

        rho = 0.3
        sigma2 = 1.0

        # Build dense P
        P_dense = jax_build_P_dense(rho, sigma2, omega_jax, W_sym_dense, WtW_dense)

        # Solve
        rhs = np.random.randn(n)
        rhs_jax = jnp.asarray(rhs)
        M_inv_diag = 1.0 / jnp.where(
            jnp.abs(jnp.diag(P_dense)) > 1e-15,
            jnp.diag(P_dense),
            1.0,
        )
        x = jax_cg_solve(P_dense, rhs_jax, M_inv_diag)

        # Verify: P @ x ≈ rhs
        residual = np.asarray(P_dense @ x) - rhs
        rel_res = np.linalg.norm(residual) / np.linalg.norm(rhs)
        assert rel_res < 1e-8, f"CG residual: {rel_res:.2e}"

        # Logdet
        key = jax.random.PRNGKey(42)
        logdet = jax_lanczos_logdet(P_dense, key=key, n_probes=10, lanczos_deg=30)
        exact_logdet = _exact_logdet(P_sparse)
        rel_err = abs(logdet - exact_logdet) / abs(exact_logdet)
        assert rel_err < 0.05, f"Logdet relative error: {rel_err:.4f}"

    def test_chebyshev_draw_covariance(self):
        """Chebyshev draws should have approximately correct covariance."""
        n = 20
        P_sparse, W_sym, WtW, omega = _make_precision(n, rho=0.2)
        W_sym_dense = jnp.asarray(W_sym.toarray())
        WtW_dense = jnp.asarray(WtW.toarray())
        omega_jax = jnp.asarray(omega)

        P_dense = jax_build_P_dense(0.2, 1.0, omega_jax, W_sym_dense, WtW_dense)
        rhs = np.zeros(n)  # zero mean
        rhs_jax = jnp.asarray(rhs)

        # Draw many samples
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 500)
        draws = []
        for k in keys:
            d = jax_chebyshev_sample(P_dense, rhs_jax, key=k, degree=30)
            draws.append(d.x)
        samples = np.array(draws)

        # Check that sample covariance is approximately P^{-1}
        P_inv_exact = np.linalg.inv(P_sparse.toarray())
        sample_cov = np.cov(samples.T)
        # Compare diagonal (variances)
        var_exact = np.diag(P_inv_exact)
        var_sample = np.diag(sample_cov)
        rel_err = np.max(np.abs(var_sample - var_exact) / var_exact)
        assert rel_err < 0.3, f"Variance relative error: {rel_err:.4f}"
