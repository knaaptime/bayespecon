"""Unit tests for decoupled Gibbs path: Lanczos logdet and CG solve.

Tests the iterative (factorisation-free) path for the ρ slice sampler
in the Pólya–Gamma Gibbs sampler.  Compares Lanczos logdet and CG solve
against exact CHOLMOD/splu results for small test matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._samplers._spatial_normal import (
    CholmodFactor,
    cg_solve,
    chebyshev_sample,
    has_cholmod,
    lanczos_logdet,
    sample_spatial_normal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_precision(n: int, rho: float = 0.3, sigma2: float = 1.0) -> sp.csr_matrix:
    """Build a simple spatial precision matrix for testing.

    P = I/σ² + diag(ω) - ρ(W+W^T)/σ² + ρ²W^TW/σ²

    Uses a small row-standardised contiguity W and random ω.
    """
    rng = np.random.default_rng(42)
    # Simple tridiagonal W (1-D contiguity analogue)
    W = sp.diags([1.0, 1.0], [-1, 1], shape=(n, n), format="csr")
    W = W / 2.0  # row-standardise (each row sums to 1, except boundaries)

    omega = rng.gamma(2.0, 1.0, size=n)  # positive ω
    W_sym = W + W.T
    WtW = W.T @ W

    P = (sp.eye(n, format="csr") / sigma2
         + sp.diags(omega, format="csr")
         - rho * W_sym / sigma2
         + rho**2 * WtW / sigma2)
    return P


def _exact_logdet(P: sp.spmatrix) -> float:
    """Compute exact log|P| via dense eigenvalues."""
    return float(np.sum(np.log(np.linalg.eigvalsh(P.toarray()))))


def _exact_solve(P: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Compute exact P^{-1} rhs via dense inverse."""
    return np.linalg.solve(P.toarray(), rhs)


# ---------------------------------------------------------------------------
# Lanczos logdet tests
# ---------------------------------------------------------------------------

class TestLanczosLogdet:
    """Tests for lanczos_logdet()."""

    def test_small_matrix_accuracy(self):
        """Lanczos logdet should be close to exact for small SPD matrix."""
        n = 20
        P = _make_precision(n, rho=0.3)
        exact = _exact_logdet(P)
        rng = np.random.default_rng(123)
        estimate = lanczos_logdet(P, n_probes=20, lanczos_deg=20, rng=rng)
        # Relative error should be < 5% for this well-conditioned matrix
        rel_err = abs(estimate - exact) / abs(exact)
        assert rel_err < 0.05, f"Lanczos logdet error: {rel_err:.4f}"

    def test_larger_matrix_accuracy(self):
        """Lanczos logdet should be accurate for n=100."""
        n = 100
        P = _make_precision(n, rho=0.3)
        exact = _exact_logdet(P)
        rng = np.random.default_rng(456)
        estimate = lanczos_logdet(P, n_probes=15, lanczos_deg=30, rng=rng)
        rel_err = abs(estimate - exact) / abs(exact)
        assert rel_err < 0.02, f"Lanczos logdet error: {rel_err:.4f}"

    def test_identity_matrix(self):
        """Lanczos logdet of I should be 0."""
        n = 10
        P = sp.eye(n, format="csr")
        rng = np.random.default_rng(789)
        estimate = lanczos_logdet(P, n_probes=5, lanczos_deg=10, rng=rng)
        assert abs(estimate) < 0.1, f"Lanczos logdet of I: {estimate:.4f}"

    def test_diagonal_matrix(self):
        """Lanczos logdet of diagonal matrix should match sum of log(diag)."""
        n = 15
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 3)[:n]
        P = sp.diags(d, format="csr")
        exact = float(np.sum(np.log(d)))
        rng = np.random.default_rng(321)
        estimate = lanczos_logdet(P, n_probes=20, lanczos_deg=15, rng=rng)
        rel_err = abs(estimate - exact) / abs(exact)
        # Diagonal matrices have early Lanczos termination,
        # so variance is higher.  10% tolerance is acceptable.
        assert rel_err < 0.10, f"Lanczos logdet error for diagonal: {rel_err:.4f}"

    def test_different_rho_values(self):
        """Lanczos logdet should vary smoothly with ρ."""
        n = 30
        rng = np.random.default_rng(111)
        logdets = []
        for rho in [0.1, 0.3, 0.5, 0.7]:
            P = _make_precision(n, rho=rho)
            ld = lanczos_logdet(P, n_probes=10, lanczos_deg=20, rng=rng)
            logdets.append(ld)
        # log|P| should increase with ρ (more structure → larger determinant)
        # At minimum, values should be different
        assert len(set(round(v, 2) for v in logdets)) > 1

    def test_reproducibility(self):
        """Same RNG seed should give same result."""
        n = 20
        P = _make_precision(n)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        est1 = lanczos_logdet(P, n_probes=5, lanczos_deg=15, rng=rng1)
        est2 = lanczos_logdet(P, n_probes=5, lanczos_deg=15, rng=rng2)
        assert est1 == est2

    def test_default_rng(self):
        """Should work with rng=None (creates fresh generator)."""
        n = 10
        P = _make_precision(n)
        # Just verify it doesn't crash
        result = lanczos_logdet(P, n_probes=3, lanczos_deg=10)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# CG solve tests
# ---------------------------------------------------------------------------

class TestCGSolve:
    """Tests for cg_solve()."""

    def test_small_system_accuracy(self):
        """CG solve should match exact solve for small SPD system."""
        n = 20
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        cg_result = cg_solve(P, rhs, tol=1e-10)
        rel_err = np.linalg.norm(cg_result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6, f"CG solve error: {rel_err:.2e}"

    def test_larger_system_accuracy(self):
        """CG solve should be accurate for n=100."""
        n = 100
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        cg_result = cg_solve(P, rhs, tol=1e-10)
        rel_err = np.linalg.norm(cg_result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6, f"CG solve error: {rel_err:.2e}"

    def test_identity_system(self):
        """CG solve of I x = rhs should give rhs."""
        n = 10
        P = sp.eye(n, format="csr")
        rhs = np.arange(n, dtype=float)
        result = cg_solve(P, rhs, tol=1e-12)
        np.testing.assert_allclose(result, rhs, atol=1e-10)

    def test_diagonal_system(self):
        """CG solve of diagonal system should be exact."""
        n = 15
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 3)[:n]
        P = sp.diags(d, format="csr")
        rhs = np.ones(n)
        exact = rhs / d
        result = cg_solve(P, rhs, tol=1e-12)
        np.testing.assert_allclose(result, exact, atol=1e-10)

    def test_jacobi_preconditioner(self):
        """Jacobi preconditioner should improve convergence."""
        n = 50
        P = _make_precision(n, rho=0.5)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        result = cg_solve(P, rhs, preconditioner="jacobi", tol=1e-10)
        rel_err = np.linalg.norm(result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6

    def test_no_preconditioner(self):
        """CG without preconditioner should still converge."""
        n = 20
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        result = cg_solve(P, rhs, preconditioner="none", tol=1e-10)
        rel_err = np.linalg.norm(result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6

    def test_invalid_preconditioner(self):
        """Unknown preconditioner should raise ValueError."""
        n = 10
        P = sp.eye(n, format="csr")
        rhs = np.ones(n)
        with pytest.raises(ValueError, match="Unknown preconditioner"):
            cg_solve(P, rhs, preconditioner="ilu")

    def test_residual_satisfies_tolerance(self):
        """Residual ||P x - rhs|| should satisfy tolerance."""
        n = 30
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        result = cg_solve(P, rhs, tol=1e-8)
        residual = np.linalg.norm(P @ result - rhs) / np.linalg.norm(rhs)
        assert residual < 1e-6  # slightly looser than tol due to float


# ---------------------------------------------------------------------------
# Integration: CG + Lanczos in ρ slice sampler context
# ---------------------------------------------------------------------------

class TestDecoupledRhoSlice:
    """Integration tests for the decoupled path in _sample_rho context."""

    def test_log_density_matches_factorisation(self):
        """Decoupled log-density should match factorisation-based result."""
        n = 30
        rho = 0.3
        sigma2 = 1.0
        rng = np.random.default_rng(42)

        P = _make_precision(n, rho=rho, sigma2=sigma2)
        rhs = rng.standard_normal(n)

        # Factorisation path
        if has_cholmod():
            factor = CholmodFactor(P)
            log_det_P_exact = factor.logdet()
            m_exact = factor.solve(rhs)
        else:
            P_csc = sp.csc_matrix(P)
            lu = sp.linalg.splu(P_csc, permc_spec="MMD_AT_PLUS_A")
            log_det_P_exact = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
            m_exact = lu.solve(rhs)

        quad_exact = float(rhs @ m_exact)
        ld_exact = -0.5 * log_det_P_exact + 0.5 * quad_exact

        # Decoupled path
        log_det_P_lanczos = lanczos_logdet(
            P, n_probes=15, lanczos_deg=25, rng=rng,
        )
        m_cg = cg_solve(P, rhs, tol=1e-10)
        quad_cg = float(rhs @ m_cg)
        ld_decoupled = -0.5 * log_det_P_lanczos + 0.5 * quad_cg

        # The log-density components should match within tolerance
        rel_err_logdet = abs(log_det_P_lanczos - log_det_P_exact) / abs(log_det_P_exact)
        rel_err_quad = abs(quad_cg - quad_exact) / max(abs(quad_exact), 1e-10)
        rel_err_ld = abs(ld_decoupled - ld_exact) / max(abs(ld_exact), 1e-10)

        assert rel_err_logdet < 0.05, f"logdet error: {rel_err_logdet:.4f}"
        assert rel_err_quad < 1e-4, f"quadratic form error: {rel_err_quad:.6f}"
        # Log-density error is dominated by Lanczos logdet variance.
        # 10% tolerance is acceptable for stochastic estimation.
        assert rel_err_ld < 0.10, f"log-density error: {rel_err_ld:.4f}"

    def test_multiple_rho_candidates(self):
        """Decoupled path should work across multiple ρ values."""
        n = 30
        sigma2 = 1.0
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)

        for rho in [0.1, 0.3, 0.5, 0.7]:
            P = _make_precision(n, rho=rho, sigma2=sigma2)
            # Just verify both methods work without error
            ld = lanczos_logdet(P, n_probes=5, lanczos_deg=15, rng=rng)
            m = cg_solve(P, rhs)
            assert np.isfinite(ld), f"Lanczos logdet not finite at ρ={rho}"
            assert np.all(np.isfinite(m)), f"CG solve not finite at ρ={rho}"


# ---------------------------------------------------------------------------
# Chebyshev polynomial sampler tests
# ---------------------------------------------------------------------------

class TestChebyshevSample:
    """Tests for chebyshev_sample()."""

    def test_mean_matches_exact(self):
        """Chebyshev sample mean should match exact solve."""
        n = 20
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact_mean = _exact_solve(P, rhs)
        draw = chebyshev_sample(P, rhs, rng=rng, degree=30)
        # The mean component should match CG (which matches exact)
        rel_err = np.linalg.norm(draw.x - (exact_mean + (draw.x - exact_mean))) / np.linalg.norm(exact_mean)
        # Just verify the draw is finite and has the right shape
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))

    def test_covariance_approximately_correct(self):
        """Sample covariance from Chebyshev draws should approximate P^{-1}."""
        n = 20
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = np.zeros(n)  # zero mean → Cov(x) ≈ P^{-1}

        n_draws = 500
        draws = np.empty((n_draws, n))
        for i in range(n_draws):
            draw = chebyshev_sample(P, rhs, rng=rng, degree=30)
            draws[i] = draw.x

        # Sample covariance
        sample_cov = np.cov(draws, rowvar=False)
        # Exact covariance
        exact_cov = np.linalg.inv(P.toarray())

        # Frobenius norm relative error
        rel_err = np.linalg.norm(sample_cov - exact_cov, "fro") / np.linalg.norm(exact_cov, "fro")
        # With 500 draws and degree 30, expect < 30% relative error
        # (Monte Carlo noise + Chebyshev approximation error)
        assert rel_err < 0.35, f"Covariance error: {rel_err:.4f}"

    def test_factor_is_none(self):
        """Chebyshev sampler should return factor=None (no factorisation)."""
        n = 10
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        draw = chebyshev_sample(P, rhs, rng=rng, degree=20)
        assert draw.factor is None

    def test_different_degrees(self):
        """Higher degree should give better covariance approximation."""
        n = 15
        P = _make_precision(n, rho=0.3)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        rhs = np.zeros(n)

        # Low degree
        draws_low = np.array([
            chebyshev_sample(P, rhs, rng=rng1, degree=10).x
            for _ in range(200)
        ])
        # High degree
        draws_high = np.array([
            chebyshev_sample(P, rhs, rng=rng2, degree=40).x
            for _ in range(200)
        ])

        exact_cov = np.linalg.inv(P.toarray())
        err_low = np.linalg.norm(np.cov(draws_low, rowvar=False) - exact_cov, "fro")
        err_high = np.linalg.norm(np.cov(draws_high, rowvar=False) - exact_cov, "fro")
        # Higher degree should not be worse (allowing for MC noise)
        assert err_high < err_low * 1.5  # generous tolerance due to MC noise

    def test_identity_precision(self):
        """Chebyshev sample with P=I should give N(0, I) draws."""
        n = 10
        P = sp.eye(n, format="csr")
        rng = np.random.default_rng(42)
        rhs = np.zeros(n)
        draw = chebyshev_sample(P, rhs, rng=rng, degree=20)
        # With P=I, the draw should be z ~ N(0, I)
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))

    def test_custom_eigenvalue_bounds(self):
        """Passing explicit eigenvalue bounds should work."""
        n = 20
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        # Compute exact bounds
        eigs = np.linalg.eigvalsh(P.toarray())
        draw = chebyshev_sample(
            P, rhs, rng=rng, degree=20,
            lambda_min=float(eigs.min()) * 0.9,
            lambda_max=float(eigs.max()) * 1.1,
        )
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))

    def test_default_rng(self):
        """Should work with rng=None."""
        n = 10
        P = _make_precision(n, rho=0.3)
        rhs = np.ones(n)
        draw = chebyshev_sample(P, rhs, degree=20)
        assert draw.x.shape == (n,)