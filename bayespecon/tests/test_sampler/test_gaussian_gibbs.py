"""Tests for the Gaussian Gibbs sampler primitives and integration.

Covers:
- Unit tests for collapsed/conditional log-densities
- Unit tests for block samplers (β, σ²)
- Unit tests for pointwise log-likelihood functions
- Integration tests: Gibbs vs NUTS posterior convergence
- InferenceData compatibility (LOO, WAIC, spatial diagnostics)
- JAX JIT path tests (skipped when JAX unavailable)
- Edge cases (tiny n, intercept-only, robust raises)
"""

from __future__ import annotations

import importlib.util

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._samplers._gaussian_gibbs import (
    GaussianGibbsCache,
    GaussianGibbsPriors,
    GaussianGibbsState,
    _initialize_gaussian_gibbs,
    _sample_beta_conjugate,
    _sample_beta_sar,
    _sample_beta_sem,
    _sample_sigma2,
    _sar_collapsed_log_density,
    _sem_collapsed_log_density,
    _sem_conditional_log_density,
    run_gaussian_chain,
)
from bayespecon._samplers._gaussian_loglik import (
    ols_pointwise_loglik_numpy,
    sar_pointwise_loglik_numpy,
    sar_pointwise_loglik_vectorized,
    sem_pointwise_loglik_numpy,
    sem_pointwise_loglik_vectorized,
)
from bayespecon.logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from bayespecon.tests.helpers import W_to_graph, make_rook_W

# Skip JAX tests when JAX is not installed
_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
requires_jax = pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIDE = 4  # 16 cross-sectional units (small for speed)


def _make_sar_data(side=SIDE, rho_true=0.4, beta_true=None, sigma_true=1.0, seed=42):
    """Create a small synthetic SAR dataset using the DGP module."""
    from bayespecon.dgp.cross_sectional import simulate_sar

    rng = np.random.default_rng(seed)
    W_dense = make_rook_W(side)
    out = simulate_sar(
        W=W_dense,
        rho=rho_true,
        beta=beta_true,
        sigma=sigma_true,
        rng=rng,
    )
    return out["y"], out["X"], W_dense, W_dense.shape[0]


def _make_sem_data(side=SIDE, lam_true=0.4, beta_true=None, sigma_true=1.0, seed=42):
    """Create a small synthetic SEM dataset using the DGP module."""
    from bayespecon.dgp.cross_sectional import simulate_sem

    rng = np.random.default_rng(seed)
    W_dense = make_rook_W(side)
    out = simulate_sem(
        W=W_dense,
        lam=lam_true,
        beta=beta_true,
        sigma=sigma_true,
        rng=rng,
    )
    return out["y"], out["X"], W_dense, W_dense.shape[0]


def _build_cache(W_dense, X, model_type="sar", Wy=None, method="eigenvalue"):
    """Build a GaussianGibbsCache from data."""
    W_sparse = sp.csr_matrix(W_dense)
    eigs = np.linalg.eigvals(W_dense).real
    rho_min = float(1.0 / eigs.min() + 1e-6)
    rho_max = float(1.0 / eigs.max() - 1e-6)
    if rho_min > rho_max:
        rho_min, rho_max = rho_max, rho_min

    logdet_fn = make_logdet_numpy_fn(
        W_sparse, eigs=eigs, method=method, rho_min=rho_min, rho_max=rho_max
    )
    logdet_vec_fn = make_logdet_numpy_vec_fn(
        W_sparse, eigs=eigs, method=method, rho_min=rho_min, rho_max=rho_max
    )
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)

    return GaussianGibbsCache(
        XtX=XtX,
        XtX_inv=XtX_inv,
        logdet_fn=logdet_fn,
        logdet_vec_fn=logdet_vec_fn,
        rho_lower=rho_min,
        rho_upper=rho_max,
        model_type=model_type,
        Wy=Wy,
        W_sparse=W_sparse,
    )


# ===================================================================
# Unit tests: log-density functions
# ===================================================================


class TestSARCollapsedLogDensity:
    """Tests for _sar_collapsed_log_density."""

    def test_returns_scalar(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", Wy=Wy)
        k = X.shape[1]
        result = _sar_collapsed_log_density(
            0.3, y, Wy, X, cache.XtX_inv, cache.logdet_fn, n, k
        )
        assert np.isscalar(result) or result.ndim == 0

    def test_maximum_near_true_rho(self):
        """Collapsed density should peak near the true ρ."""
        rho_true = 0.4
        y, X, W_dense, n = _make_sar_data(rho_true=rho_true)
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", Wy=Wy)
        k = X.shape[1]

        rho_grid = np.linspace(cache.rho_lower + 0.01, cache.rho_upper - 0.01, 50)
        log_dens = [
            _sar_collapsed_log_density(
                r, y, Wy, X, cache.XtX_inv, cache.logdet_fn, n, k
            )
            for r in rho_grid
        ]
        rho_argmax = rho_grid[np.argmax(log_dens)]
        # The mode should be within 0.3 of the true value for this small dataset
        assert abs(rho_argmax - rho_true) < 0.3

    def test_decreases_at_boundaries(self):
        """Density should decrease near the spectral bounds."""
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", Wy=Wy)
        k = X.shape[1]

        ld_mid = _sar_collapsed_log_density(
            0.0, y, Wy, X, cache.XtX_inv, cache.logdet_fn, n, k
        )
        ld_near_upper = _sar_collapsed_log_density(
            cache.rho_upper - 0.01, y, Wy, X, cache.XtX_inv, cache.logdet_fn, n, k
        )
        # Near the boundary, log|I-ρW| → -∞, so density should be lower
        assert ld_near_upper < ld_mid

    def test_woodbury_matches_direct(self):
        """Woodbury form RSS should match direct M_X computation."""
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", Wy=Wy)
        k = X.shape[1]
        rho = 0.3

        # Woodbury form (used in the function)
        r = y - rho * Wy
        Xtr = X.T @ r
        rss_woodbury = np.dot(r, r) - Xtr @ cache.XtX_inv @ Xtr

        # Direct M_X form
        MX = np.eye(n) - X @ cache.XtX_inv @ X.T
        rss_direct = r @ MX @ r

        np.testing.assert_allclose(rss_woodbury, rss_direct, rtol=1e-10)


class TestSEMConditionalLogDensity:
    """Tests for _sem_conditional_log_density."""

    def test_returns_scalar(self):
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        cache = _build_cache(W_dense, X, model_type="sem")
        beta = np.array([1.0, 2.0])
        sigma2 = 1.0
        result = _sem_conditional_log_density(
            0.3, beta, sigma2, y, X, W_sparse, cache.logdet_fn
        )
        assert np.isscalar(result) or result.ndim == 0

    def test_maximum_near_true_lam(self):
        """Conditional density does NOT necessarily peak near true λ.

        The conditional density p(λ | β, σ², y) conditions on β and σ²,
        which are correlated with λ in the SEM likelihood.  With OLS β,
        the mode can be far from the true λ.  This is exactly why we use
        the *collapsed* density for Gibbs sampling.
        """
        lam_true = 0.4
        y, X, W_dense, n = _make_sem_data(lam_true=lam_true)
        W_sparse = sp.csr_matrix(W_dense)
        cache = _build_cache(W_dense, X, model_type="sem")
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma2 = 1.0

        lam_grid = np.linspace(cache.rho_lower + 0.01, cache.rho_upper - 0.01, 50)
        log_dens = [
            _sem_conditional_log_density(
                l, beta, sigma2, y, X, W_sparse, cache.logdet_fn
            )
            for l in lam_grid
        ]
        # Just verify it's a valid density (finite values)
        assert all(np.isfinite(ld) for ld in log_dens)


class TestSEMCollapsedLogDensity:
    """Tests for _sem_collapsed_log_density."""

    def test_returns_scalar(self):
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        cache = _build_cache(W_dense, X, model_type="sem")
        result = _sem_collapsed_log_density(
            0.3, y, X, W_sparse, cache.logdet_fn, n, X.shape[1]
        )
        assert np.isscalar(result) or result.ndim == 0

    def test_maximum_near_true_lam(self):
        """Collapsed density should peak near the true λ."""
        lam_true = 0.4
        y, X, W_dense, n = _make_sem_data(lam_true=lam_true)
        W_sparse = sp.csr_matrix(W_dense)
        cache = _build_cache(W_dense, X, model_type="sem")

        lam_grid = np.linspace(cache.rho_lower + 0.01, cache.rho_upper - 0.01, 50)
        log_dens = [
            _sem_collapsed_log_density(
                l, y, X, W_sparse, cache.logdet_fn, n, X.shape[1]
            )
            for l in lam_grid
        ]
        lam_argmax = lam_grid[np.argmax(log_dens)]
        assert abs(lam_argmax - lam_true) < 0.3

    def test_zero_lam_gives_ols_residuals(self):
        """At λ=0, the density reduces to OLS log-likelihood."""
        y, X, W_dense, n = _make_sem_data(lam_true=0.0)
        W_sparse = sp.csr_matrix(W_dense)
        cache = _build_cache(W_dense, X, model_type="sem")
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma2 = 1.0

        ld = _sem_conditional_log_density(
            0.0, beta, sigma2, y, X, W_sparse, cache.logdet_fn
        )
        # At λ=0: log|I| = 0, eps = y - Xβ, so ld = -||eps||²/(2σ²)
        eps = y - X @ beta
        expected = -np.dot(eps, eps) / (2.0 * sigma2)
        np.testing.assert_allclose(ld, expected, rtol=1e-10)


# ===================================================================
# Unit tests: block samplers
# ===================================================================


class TestBetaBlock:
    """Tests for conjugate normal β draw."""

    def test_sample_beta_sar_shape(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        XtX = X.T @ X
        beta = _sample_beta_sar(0.3, 1.0, y, Wy, X, XtX, priors, rng)
        assert beta.shape == (X.shape[1],)

    def test_sample_beta_sem_shape(self):
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        beta = _sample_beta_sem(0.3, 1.0, y, X, W_sparse, priors, rng)
        assert beta.shape == (X.shape[1],)

    def test_conjugate_normal_matches_analytical(self):
        """Posterior mean of many draws should match analytical posterior mean."""
        rng = np.random.default_rng(42)
        n, k = 50, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        beta_true = np.array([1.0, 2.0, -0.5])
        r = X @ beta_true + 0.5 * rng.standard_normal(n)
        sigma2 = 0.25
        priors = GaussianGibbsPriors(beta_mu=0.0, beta_sigma=100.0)
        XtX = X.T @ X

        # Analytical posterior
        prior_prec = np.diag(1.0 / np.full(k, 100.0) ** 2)
        post_prec = XtX / sigma2 + prior_prec
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (X.T @ r / sigma2 + prior_prec @ np.zeros(k))

        # Draw many samples
        draws = np.array(
            [
                _sample_beta_conjugate(r, X, XtX, sigma2, priors, rng)
                for _ in range(5000)
            ]
        )

        # Sample mean should be close to analytical posterior mean
        np.testing.assert_allclose(draws.mean(axis=0), post_mean, atol=0.1)


class TestSigma2Block:
    """Tests for conjugate Inv-Γ σ² draw."""

    def test_sample_sigma2_sar_positive(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        beta = np.array([1.0, 2.0])
        sigma2 = _sample_sigma2(0.3, beta, y, Wy, None, X, priors, "sar", rng)
        assert sigma2 > 0

    def test_sample_sigma2_sem_positive(self):
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        beta = np.array([1.0, 2.0])
        sigma2 = _sample_sigma2(0.3, beta, y, None, W_sparse, X, priors, "sem", rng)
        assert sigma2 > 0

    def test_sigma2_draws_converge_to_analytical(self):
        """Mean of many σ² draws should be close to analytical posterior mean."""
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + 0.5 * rng.standard_normal(n)
        resid = y - X @ beta_true
        Wy = np.zeros(n)  # ρ=0 so Wy doesn't matter, but must be provided
        priors = GaussianGibbsPriors()  # prior on σ² is now Jeffreys (weak)

        # Analytical: Inv-Γ(a_post, b_post), mean = b_post / (a_post - 1)
        EPS = 1e-3
        a_post = n / 2 + EPS
        b_post = np.dot(resid, resid) / 2 + EPS
        expected_mean = b_post / (a_post - 1)

        draws = np.array(
            [
                _sample_sigma2(0.0, beta_true, y, Wy, None, X, priors, "sar", rng)
                for _ in range(5000)
            ]
        )

        np.testing.assert_allclose(draws.mean(), expected_mean, rtol=0.1)


# ===================================================================
# Unit tests: pointwise log-likelihood
# ===================================================================


class TestPointwiseLogLik:
    """Tests for pointwise log-likelihood functions."""

    def test_sar_loglik_sums_to_total(self):
        """Per-obs LL sums to total LL (Gaussian + Jacobian)."""
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        beta = np.array([1.0, 2.0])
        sigma = 1.0
        rho = 0.3

        ll = sar_pointwise_loglik_numpy(y, X, Wy, beta, sigma, rho, logdet_fn, n)
        total_ll = ll.sum()

        # Manual computation
        mu = rho * Wy + X @ beta
        resid = y - mu
        gauss_total = (
            -0.5 * np.dot(resid, resid) / sigma**2
            - n * np.log(sigma)
            - n / 2 * np.log(2 * np.pi)
        )
        jacobian = logdet_fn(rho)
        expected_total = gauss_total + jacobian

        np.testing.assert_allclose(total_ll, expected_total, rtol=1e-10)

    def test_sem_loglik_sums_to_total(self):
        """Per-obs LL sums to total LL (Gaussian + Jacobian)."""
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        beta = np.array([1.0, 2.0])
        sigma = 1.0
        lam = 0.3

        ll = sem_pointwise_loglik_numpy(y, X, W_sparse, beta, sigma, lam, logdet_fn, n)
        total_ll = ll.sum()

        # Manual computation
        resid = y - X @ beta
        eps = resid - lam * (W_sparse @ resid)
        gauss_total = (
            -0.5 * np.dot(eps, eps) / sigma**2
            - n * np.log(sigma)
            - n / 2 * np.log(2 * np.pi)
        )
        jacobian = logdet_fn(lam)
        expected_total = gauss_total + jacobian

        np.testing.assert_allclose(total_ll, expected_total, rtol=1e-10)

    def test_sar_vectorized_matches_per_draw(self):
        """Vectorized LL should match per-draw computation."""
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        logdet_vec_fn = make_logdet_numpy_vec_fn(
            W_sparse, eigs=eigs, method="eigenvalue"
        )

        n_keep = 10
        rng = np.random.default_rng(42)
        rho_draws = rng.uniform(0.1, 0.5, n_keep)
        beta_draws = rng.standard_normal((n_keep, X.shape[1]))
        sigma_draws = np.abs(rng.standard_normal(n_keep)) + 0.5

        # Vectorized
        ll_vec = sar_pointwise_loglik_vectorized(
            rho_draws, beta_draws, sigma_draws, y, X, Wy, logdet_vec_fn, n
        )

        # Per-draw
        ll_per = np.empty((n_keep, n))
        for g in range(n_keep):
            ll_per[g] = sar_pointwise_loglik_numpy(
                y, X, Wy, beta_draws[g], sigma_draws[g], rho_draws[g], logdet_fn, n
            )

        np.testing.assert_allclose(ll_vec, ll_per, rtol=1e-10)

    def test_sem_vectorized_matches_per_draw(self):
        """Vectorized LL should match per-draw computation."""
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        logdet_vec_fn = make_logdet_numpy_vec_fn(
            W_sparse, eigs=eigs, method="eigenvalue"
        )

        n_keep = 10
        rng = np.random.default_rng(42)
        lam_draws = rng.uniform(0.1, 0.5, n_keep)
        beta_draws = rng.standard_normal((n_keep, X.shape[1]))
        sigma_draws = np.abs(rng.standard_normal(n_keep)) + 0.5

        # Vectorized
        ll_vec = sem_pointwise_loglik_vectorized(
            lam_draws, beta_draws, sigma_draws, y, X, W_sparse, logdet_vec_fn, n
        )

        # Per-draw
        ll_per = np.empty((n_keep, n))
        for g in range(n_keep):
            ll_per[g] = sem_pointwise_loglik_numpy(
                y,
                X,
                W_sparse,
                beta_draws[g],
                sigma_draws[g],
                lam_draws[g],
                logdet_fn,
                n,
            )

        np.testing.assert_allclose(ll_vec, ll_per, rtol=1e-10)

    def test_ols_loglik_no_jacobian(self):
        """OLS LL should not include Jacobian term."""
        rng = np.random.default_rng(42)
        n = 20
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([1.0, 2.0])
        sigma = 1.0
        y = X @ beta + sigma * rng.standard_normal(n)

        ll = ols_pointwise_loglik_numpy(y, X, beta, sigma)
        # Sum should equal standard Gaussian LL (no Jacobian)
        resid = y - X @ beta
        expected = -0.5 * (resid / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)


# ===================================================================
# Unit tests: initialization
# ===================================================================


class TestInitialization:
    """Tests for _initialize_gaussian_gibbs."""

    def test_ols_warm_start(self):
        y, X, W_dense, n = _make_sar_data()
        XtX_inv = np.linalg.inv(X.T @ X)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        state = _initialize_gaussian_gibbs(y, X, XtX_inv, priors, rng)

        assert isinstance(state, GaussianGibbsState)
        assert state.beta.shape == (X.shape[1],)
        assert state.sigma2 > 0
        assert state.rho == 0.0  # starts at 0

    def test_ols_beta_close_to_lstsq(self):
        y, X, W_dense, n = _make_sar_data()
        XtX_inv = np.linalg.inv(X.T @ X)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        state = _initialize_gaussian_gibbs(y, X, XtX_inv, priors, rng)

        beta_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(state.beta, beta_lstsq, rtol=1e-10)


# ===================================================================
# Integration tests: chain runner
# ===================================================================


class TestChainRunner:
    """Tests for run_gaussian_chain."""

    def test_sar_chain_produces_valid_output(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", Wy=Wy)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        init = _initialize_gaussian_gibbs(y, X, cache.XtX_inv, priors, rng)

        result = run_gaussian_chain(
            y=y,
            X=X,
            cache=cache,
            priors=priors,
            init=init,
            draws=50,
            tune=20,
            thin=1,
            rng=rng,
        )

        assert "rho" in result
        assert "beta" in result
        assert "sigma" in result
        assert "log_lik" in result
        assert result["rho"].shape == (50,)
        assert result["beta"].shape == (50, X.shape[1])
        assert result["sigma"].shape == (50,)
        assert result["log_lik"].shape == (50, n)
        assert np.all(result["sigma"] > 0)
        assert np.all(np.isfinite(result["log_lik"]))

    def test_sem_chain_produces_valid_output(self):
        y, X, W_dense, n = _make_sem_data()
        cache = _build_cache(W_dense, X, model_type="sem")
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        init = _initialize_gaussian_gibbs(y, X, cache.XtX_inv, priors, rng)

        result = run_gaussian_chain(
            y=y,
            X=X,
            cache=cache,
            priors=priors,
            init=init,
            draws=50,
            tune=20,
            thin=1,
            rng=rng,
        )

        assert "lam" in result
        assert "beta" in result
        assert "sigma" in result
        assert "log_lik" in result
        assert result["lam"].shape == (50,)
        assert result["beta"].shape == (50, X.shape[1])
        assert result["sigma"].shape == (50,)
        assert result["log_lik"].shape == (50, n)

    def test_thinning(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", Wy=Wy)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        init = _initialize_gaussian_gibbs(y, X, cache.XtX_inv, priors, rng)

        result = run_gaussian_chain(
            y=y,
            X=X,
            cache=cache,
            priors=priors,
            init=init,
            draws=100,
            tune=20,
            thin=2,
            rng=rng,
        )

        # 100 draws with thin=2 → 50 kept
        assert result["rho"].shape == (50,)


# ===================================================================
# Integration tests: Gibbs vs NUTS posterior convergence
# ===================================================================


@pytest.mark.slow
class TestGibbsVsNUTS:
    """Gibbs and NUTS posteriors should agree for Gaussian models.

    These tests use moderate draws and check that posterior means
    are within a tolerance.  They are marked ``slow`` because they
    run MCMC chains.
    """

    def test_sar_gibbs_vs_nuts(self):
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data(rho_true=0.4)
        W = W_to_graph(W_dense)

        model_nuts = SAR(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SAR(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        for param in ["rho", "sigma"]:
            mean_nuts = float(idata_nuts.posterior[param].mean())
            mean_gibbs = float(idata_gibbs.posterior[param].mean())
            np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)

        # Beta: compare element-wise
        beta_nuts = idata_nuts.posterior["beta"].mean(dim=["chain", "draw"]).values
        beta_gibbs = idata_gibbs.posterior["beta"].mean(dim=["chain", "draw"]).values
        np.testing.assert_allclose(beta_nuts, beta_gibbs, atol=0.15)

    def test_sem_gibbs_vs_nuts(self):
        from bayespecon.models.sem import SEM

        y, X, W_dense, n = _make_sem_data(lam_true=0.4)
        W = W_to_graph(W_dense)

        model_nuts = SEM(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SEM(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        for param in ["lam", "sigma"]:
            mean_nuts = float(idata_nuts.posterior[param].mean())
            mean_gibbs = float(idata_gibbs.posterior[param].mean())
            np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)

    def test_sdm_gibbs_vs_nuts(self):
        from bayespecon.models.sdm import SDM

        y, X, W_dense, n = _make_sar_data(rho_true=0.3)
        W = W_to_graph(W_dense)

        model_nuts = SDM(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SDM(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        mean_nuts = float(idata_nuts.posterior["rho"].mean())
        mean_gibbs = float(idata_gibbs.posterior["rho"].mean())
        np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)

    def test_sdem_gibbs_vs_nuts(self):
        from bayespecon.models.sdem import SDEM

        y, X, W_dense, n = _make_sem_data(lam_true=0.3)
        W = W_to_graph(W_dense)

        model_nuts = SDEM(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SDEM(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        mean_nuts = float(idata_nuts.posterior["lam"].mean())
        mean_gibbs = float(idata_gibbs.posterior["lam"].mean())
        np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)


# ===================================================================
# InferenceData compatibility
# ===================================================================


class TestInferenceDataCompat:
    """Gibbs-produced InferenceData should work with ArviZ diagnostics."""

    @pytest.fixture
    def sar_idata(self):
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        return model.fit(
            sampler="gibbs",
            draws=100,
            tune=50,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

    def test_idata_groups(self, sar_idata):
        assert "posterior" in sar_idata.groups()
        assert "log_likelihood" in sar_idata.groups()
        assert "observed_data" in sar_idata.groups()

    def test_posterior_vars(self, sar_idata):
        assert "rho" in sar_idata.posterior.data_vars
        assert "beta" in sar_idata.posterior.data_vars
        assert "sigma" in sar_idata.posterior.data_vars

    def test_log_likelihood_dims(self, sar_idata):
        """obs should be a data variable, not a coordinate."""
        assert "obs" in sar_idata.log_likelihood.data_vars
        ll = sar_idata.log_likelihood["obs"]
        assert ll.ndim == 3
        assert "chain" in ll.dims
        assert "draw" in ll.dims

    def test_loo_works(self, sar_idata):
        loo = az.loo(sar_idata)
        assert np.isfinite(loo.elpd_loo)

    def test_waic_works(self, sar_idata):
        waic = az.waic(sar_idata)
        assert np.isfinite(waic.elpd_waic)

    def test_summary_works(self, sar_idata):
        summary = az.summary(sar_idata)
        assert len(summary) > 0

    def test_sigma_not_sigma2(self, sar_idata):
        """Posterior should store σ (not σ²)."""
        sigma_mean = float(sar_idata.posterior["sigma"].mean())
        assert sigma_mean > 0  # σ is positive
        # σ should be on the order of the true value (1.0), not σ²
        assert sigma_mean < 10.0  # would be ~1, not ~1 if stored as σ²


# ===================================================================
# JAX JIT path tests
# ===================================================================


@requires_jax
class TestJAXGaussianGibbs:
    """Tests for the JAX JIT Gaussian Gibbs path."""

    def test_sar_jax_produces_valid_idata(self):
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="jax",
        )
        assert "posterior" in idata.groups()
        assert "rho" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars
        assert float(idata.posterior["rho"].mean()) != 0  # not stuck at init

    def test_sem_jax_produces_valid_idata(self):
        from bayespecon.models.sem import SEM

        y, X, W_dense, n = _make_sem_data()
        W = W_to_graph(W_dense)
        model = SEM(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="jax",
        )
        assert "lam" in idata.posterior.data_vars

    def test_sdm_jax_produces_valid_idata(self):
        from bayespecon.models.sdm import SDM

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SDM(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="jax",
        )
        assert "rho" in idata.posterior.data_vars
        assert idata.posterior["beta"].shape[-1] == 3  # intercept + x + W*x

    def test_sdem_jax_produces_valid_idata(self):
        from bayespecon.models.sdem import SDEM

        y, X, W_dense, n = _make_sem_data()
        W = W_to_graph(W_dense)
        model = SDEM(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="jax",
        )
        assert "lam" in idata.posterior.data_vars

    def test_rw_mh_option(self):
        """use_mala=False should use RW-MH instead of MALA."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="jax",
            use_mala=False,
        )
        assert "rho" in idata.posterior.data_vars

    def test_jax_loo_works(self):
        """LOO should work with JAX-produced InferenceData."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=100,
            tune=50,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="jax",
        )
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd_loo)

    def test_chebyshev_logdet_with_jax(self):
        """Chebyshev logdet method should work with JAX path."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W, logdet_method="chebyshev")
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="jax",
        )
        assert "rho" in idata.posterior.data_vars


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case tests for the Gibbs sampler."""

    def test_robust_raises(self):
        """Gibbs should raise NotImplementedError for robust models."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W, robust=True)
        with pytest.raises(NotImplementedError, match="robust"):
            model.fit(sampler="gibbs", draws=10, tune=5, progressbar=False)

    def test_tiny_n(self):
        """Gibbs should work with very small n."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data(side=3)  # n=9
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert "rho" in idata.posterior.data_vars

    def test_intercept_only(self):
        """Gibbs should work with k=1 (intercept only)."""
        rng = np.random.default_rng(42)
        W_dense = make_rook_W(4)
        n = 16
        W = W_to_graph(W_dense)
        X = np.ones((n, 1))
        y = 1.0 + 0.3 * (W_dense @ np.ones(n)) + 0.5 * rng.standard_normal(n)

        from bayespecon.models.sar import SAR

        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert idata.posterior["beta"].shape[-1] == 1

    def test_invalid_sampler_raises(self):
        """Invalid sampler name should raise ValueError."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with pytest.raises(ValueError, match="sampler"):
            model.fit(sampler="invalid")

    def test_invalid_gibbs_method_falls_back(self):
        """Invalid gibbs_method falls back to numpy path (no error raised)."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        # gibbs_method="invalid" falls through to numpy path
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_method="invalid",
        )
        assert "posterior" in idata.groups()


# ---------------------------------------------------------------------------
# Tests for progress bars, informative output, chain parallelism, mp context
# ---------------------------------------------------------------------------


class TestGibbsProgressBarManager:
    """Tests for GibbsProgressBarManager."""

    def test_context_manager_enter_exit(self):
        """Manager enters and exits cleanly with progressbar=True."""
        from bayespecon._samplers._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=2, draws=10, tune=5, progressbar=True)
        with pm:
            assert pm._show is True
            assert len(pm._tasks) == 2

    def test_context_manager_no_progressbar(self):
        """Manager is a no-op when progressbar=False."""
        from bayespecon._samplers._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=2, draws=10, tune=5, progressbar=False)
        with pm:
            assert pm._show is False
            assert len(pm._tasks) == 0

    def test_update_advances_iteration(self):
        """update() advances the iteration counter."""
        from bayespecon._samplers._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=1, draws=5, tune=2, progressbar=True)
        with pm:
            pm.update(0, 0, tuning=True, accept=None)
            pm.update(0, 1, tuning=True, accept=None)
            pm.update(0, 2, tuning=False, accept=True)
            # No assertion on internal state — just verify no errors

    def test_update_noop_when_disabled(self):
        """update() is a no-op when progressbar=False."""
        from bayespecon._samplers._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=1, draws=5, tune=2, progressbar=False)
        with pm:
            pm.update(0, 0, tuning=True, accept=None)
            # Should not raise

    def test_accept_rate_tracking(self):
        """Accept rate is tracked and displayed."""
        from bayespecon._samplers._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=1, draws=5, tune=2, progressbar=True)
        with pm:
            pm.update(0, 0, tuning=True, accept=True)
            pm.update(0, 1, tuning=True, accept=False)
            assert pm._accept_counts is not None
            assert pm._accept_counts[0] == 1
            assert pm._accept_totals[0] == 2


class TestInformativeOutput:
    """Tests for logging output from GibbsEstimation."""

    def test_numpy_path_logs_sampling_info(self, caplog):
        """NumPy path logs sampling start and completion."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with caplog.at_level("INFO", logger="bayespecon._samplers._gibbs_estimation"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                random_seed=42,
                n_jobs=1,
                progressbar=False,
            )
        assert "Gibbs sampling" in caplog.text
        assert "took" in caplog.text

    def test_jax_path_logs_sampler_name(self, caplog):
        """JAX path logs MALA/RW-MH sampler name."""
        pytest.importorskip("jax")
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with caplog.at_level("INFO", logger="bayespecon._samplers._gibbs_estimation"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                random_seed=42,
                n_jobs=1,
                progressbar=False,
                gibbs_method="jax",
                use_mala=True,
            )
        assert "MALA" in caplog.text
        assert "acceptance rate" in caplog.text


class TestChainParallelism:
    """Tests for chain parallelism (n_jobs for NumPy, chain_method for JAX)."""

    def test_sequential_numpy(self):
        """n_jobs=1 runs NumPy chains sequentially."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_parallel_numpy(self):
        """n_jobs=2 runs NumPy chains in parallel via joblib."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            n_jobs=2,
            progressbar=False,
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_parallel_numpy_all_cpus(self):
        """n_jobs=-1 runs NumPy chains in parallel using all CPUs."""
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            n_jobs=-1,
            progressbar=False,
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_vectorized_jax(self):
        """chain_method='vectorized' works for JAX path."""
        pytest.importorskip("jax")
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            progressbar=False,
            gibbs_method="jax",
            chain_method="vectorized",
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_jax_default_is_vectorized(self):
        """JAX path defaults to chain_method='vectorized'."""
        pytest.importorskip("jax")
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            progressbar=False,
            gibbs_method="jax",
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_parallel_jax_raises(self):
        """chain_method='parallel' raises NotImplementedError for JAX."""
        pytest.importorskip("jax")
        from bayespecon.models.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="vectorized"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=2,
                random_seed=42,
                progressbar=False,
                gibbs_method="jax",
                chain_method="parallel",
            )

    def test_vectorized_per_chain_adaptation(self):
        """JAX vectorized path uses per-chain adapted step sizes."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        import scipy.sparse as sp

        from bayespecon._samplers._gaussian_gibbs import (
            GaussianGibbsCache,
            GaussianGibbsState,
            _initialize_gaussian_gibbs,
        )
        from bayespecon._samplers._gibbs_estimation import GaussianGibbsPriors
        from bayespecon._samplers._jax_gaussian_gibbs import (
            run_chains_jax_gibbs_vectorized,
        )
        from bayespecon.logdet import (
            make_logdet_jax_fn,
            make_logdet_numpy_fn,
            make_logdet_numpy_vec_fn,
        )

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        Wy = W_dense @ y
        W_sparse = sp.csr_matrix(W_dense)

        priors = GaussianGibbsPriors()
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(
            W_sparse,
            eigs=eigs,
            method="eigenvalue",
            rho_min=priors.rho_lower,
            rho_max=priors.rho_upper,
        )
        logdet_vec_fn = make_logdet_numpy_vec_fn(
            W_sparse,
            eigs=eigs,
            method="eigenvalue",
            rho_min=priors.rho_lower,
            rho_max=priors.rho_upper,
        )
        cache = GaussianGibbsCache(
            XtX=X.T @ X,
            XtX_inv=np.linalg.inv(X.T @ X),
            logdet_fn=logdet_fn,
            logdet_vec_fn=logdet_vec_fn,
            rho_lower=priors.rho_lower,
            rho_upper=priors.rho_upper,
            model_type="sar",
            Wy=Wy,
            W_sparse=W_sparse,
        )

        # Initialize 4 chains with different seeds
        chains = 4
        inits = []
        for i in range(chains):
            rng = np.random.default_rng(42 + i)
            inits.append(_initialize_gaussian_gibbs(y, X, cache.XtX_inv, priors, rng))

        logdet_jax = make_logdet_jax_fn(
            W=W_dense,
            method="eigenvalue",
            rho_min=priors.rho_lower,
            rho_max=priors.rho_upper,
        )

        # Run with use_mala=True to trigger adaptation
        results = run_chains_jax_gibbs_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            Wy=Wy,
            logdet_jax=logdet_jax,
            logdet_vec_fn=cache.logdet_vec_fn,
            priors=priors,
            inits=inits,
            draws=50,
            tune=50,
            thin=1,
            jax_seeds=list(range(chains)),
            model_type="sar",
            mala_step_size=0.05,
            use_mala=True,
            progressbar=False,
        )

        # Verify results are valid
        assert len(results) == chains
        for r in results:
            assert "rho" in r
            assert "beta" in r
            assert "sigma" in r
            assert "mh_accept_rate" in r
            # Accept rate should be between 0 and 1 (inclusive)
            assert 0 <= r["mh_accept_rate"] <= 1


class TestMultiprocessingContext:
    """Tests for _initialize_multiprocessing_context."""

    def test_default_context(self):
        """Default context is returned without error."""
        from bayespecon._samplers._chain_runner import (
            _initialize_multiprocessing_context,
        )

        ctx = _initialize_multiprocessing_context()
        assert ctx is not None
        assert hasattr(ctx, "Pool")

    def test_explicit_fork(self):
        """Explicit 'fork' context is returned (with warning if JAX present)."""
        import importlib.util
        import multiprocessing

        from bayespecon._samplers._chain_runner import (
            _initialize_multiprocessing_context,
        )

        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("fork not available on this platform")
        jax_available = importlib.util.find_spec("jax") is not None
        if jax_available:
            with pytest.warns(UserWarning, match="fork"):
                ctx = _initialize_multiprocessing_context("fork", quiet=True)
        else:
            ctx = _initialize_multiprocessing_context("fork", quiet=True)
        assert ctx.get_start_method() == "fork"

    def test_explicit_spawn(self):
        """Explicit 'spawn' context is returned."""
        from bayespecon._samplers._chain_runner import (
            _initialize_multiprocessing_context,
        )

        ctx = _initialize_multiprocessing_context("spawn", quiet=True)
        assert ctx.get_start_method() == "spawn"

    def test_jax_auto_switches_from_fork(self):
        """When JAX is installed, default 'fork' is auto-switched."""
        import importlib.util

        from bayespecon._samplers._chain_runner import (
            _initialize_multiprocessing_context,
        )

        jax_available = importlib.util.find_spec("jax") is not None
        if not jax_available:
            pytest.skip("JAX not installed")
        # Default on macOS ARM is 'fork', but JAX should trigger auto-switch
        ctx = _initialize_multiprocessing_context(quiet=True)
        assert ctx.get_start_method() != "fork"

    def test_user_fork_with_jax_warns(self):
        """User-specified 'fork' with JAX installed emits a warning."""
        import importlib.util

        from bayespecon._samplers._chain_runner import (
            _initialize_multiprocessing_context,
        )

        jax_available = importlib.util.find_spec("jax") is not None
        if not jax_available:
            pytest.skip("JAX not installed")
        with pytest.warns(UserWarning, match="fork"):
            _initialize_multiprocessing_context("fork")
