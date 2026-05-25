"""Unit tests for the full-JIT JAX Gibbs sampler.

Tests the ``_make_gibbs_step_with_data()`` and ``run_chain_jax()``
functions in ``bayespecon._samplers.negbin._jax``.

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

from bayespecon._samplers.negbin._jax import (
    _make_gibbs_step_with_data,
    _nb_loglik_pointwise_jax,
    _sample_alpha_python,
    run_chain_jax,
)
from bayespecon._samplers.negbin._core import GibbsPriors, GibbsState, JAXGibbsState
from bayespecon._logdet import make_logdet_jax_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_data(n=20, k=2, rho_true=0.3, seed=42):
    """Create a small synthetic SAR-NB dataset for testing.

    Returns (y, X, W, W_sym, WtW, W_eigs, true_params).
    """
    rng = np.random.default_rng(seed)

    # Row-standardised W (tridiagonal)
    W = sp.diags([1.0, 1.0], [-1, 1], shape=(n, n), format="csr")
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W = sp.diags(1.0 / row_sums) @ W

    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    if k > 2:
        X = np.column_stack([X, rng.standard_normal((n, k - 2))])

    beta_true = np.zeros(k)
    beta_true[0] = 1.0
    if k > 1:
        beta_true[1] = 0.5

    sigma2_true = 1.0
    alpha_true = 2.0

    A_rho = sp.eye(n) - rho_true * W
    eta_true = sp.linalg.spsolve(
        A_rho, X @ beta_true + rng.standard_normal(n) * np.sqrt(sigma2_true)
    )
    mu = np.exp(eta_true)
    y = rng.negative_binomial(alpha_true, alpha_true / (mu + alpha_true)).astype(
        np.float64
    )

    W_sym = W + W.T
    WtW = W.T @ W
    W_eigs = np.linalg.eigvals(W.toarray()).real

    true_params = {
        "beta": beta_true,
        "rho": rho_true,
        "sigma2": sigma2_true,
        "alpha": alpha_true,
        "eta": eta_true,
    }
    return y, X, W, W_sym, WtW, W_eigs, true_params


def _make_jax_components(W, W_sym, WtW, W_eigs, X):
    """Convert numpy/scipy arrays to JAX arrays for the Gibbs step."""
    from bayespecon._logdet import make_logdet_jax_fn

    W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
    WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
    logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")
    W_dense_jax = jnp.asarray(W.toarray(), dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)
    return W_sym_dense, WtW_dense, logdet_jax, W_dense_jax, X_jax, XtX_jax


# ---------------------------------------------------------------------------
# Test _make_gibbs_step_with_data
# ---------------------------------------------------------------------------


class TestMakeGibbsStepWithData:
    """Tests for _make_gibbs_step_with_data()."""

    def test_returns_callable(self):
        """Should return a JIT-compiled function."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense, WtW_dense, logdet_jax, W_dense_jax, X_jax, XtX_jax = (
            _make_jax_components(W, W_sym, WtW, W_eigs, X)
        )
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        gibbs_step = _make_gibbs_step_with_data(
            y_jax=y_jax,
            X_jax=X_jax,
            W_dense_jax=W_dense_jax,
            n=n,
            k=k,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            XtX_jax=XtX_jax,
            priors=priors,
            pg_n_terms=20,
            mh_proposal_sd=0.05,
            n_probes=10,
            lanczos_deg=30,
        )
        assert callable(gibbs_step)

    def test_step_output_shapes(self):
        """Gibbs step should return JAXGibbsState with correct shapes."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense, WtW_dense, logdet_jax, W_dense_jax, X_jax, XtX_jax = (
            _make_jax_components(W, W_sym, WtW, W_eigs, X)
        )
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        gibbs_step = _make_gibbs_step_with_data(
            y_jax=y_jax,
            X_jax=X_jax,
            W_dense_jax=W_dense_jax,
            n=n,
            k=k,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            XtX_jax=XtX_jax,
            priors=priors,
            pg_n_terms=20,
            mh_proposal_sd=0.05,
            n_probes=10,
            lanczos_deg=30,
        )

        state = JAXGibbsState(
            eta=jnp.asarray(params["eta"], dtype=jnp.float64),
            beta=jnp.asarray(params["beta"], dtype=jnp.float64),
            sigma2=jnp.float64(params["sigma2"]),
            rho=jnp.float64(params["rho"]),
            omega=jnp.ones(n, dtype=jnp.float64),
            alpha=jnp.float64(params["alpha"]),
        )

        key = jax.random.PRNGKey(42)
        new_state, accept = gibbs_step(state, key)

        assert new_state.eta.shape == (n,)
        assert new_state.beta.shape == (k,)
        assert new_state.sigma2.shape == ()
        assert new_state.rho.shape == ()
        assert new_state.omega.shape == (n,)
        assert new_state.alpha.shape == ()
        assert isinstance(bool(accept), bool)

    def test_step_produces_finite_values(self):
        """Gibbs step should produce finite values."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense, WtW_dense, logdet_jax, W_dense_jax, X_jax, XtX_jax = (
            _make_jax_components(W, W_sym, WtW, W_eigs, X)
        )
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        gibbs_step = _make_gibbs_step_with_data(
            y_jax=y_jax,
            X_jax=X_jax,
            W_dense_jax=W_dense_jax,
            n=n,
            k=k,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            XtX_jax=XtX_jax,
            priors=priors,
            pg_n_terms=20,
            mh_proposal_sd=0.05,
            n_probes=10,
            lanczos_deg=30,
        )

        state = JAXGibbsState(
            eta=jnp.asarray(params["eta"], dtype=jnp.float64),
            beta=jnp.asarray(params["beta"], dtype=jnp.float64),
            sigma2=jnp.float64(params["sigma2"]),
            rho=jnp.float64(params["rho"]),
            omega=jnp.ones(n, dtype=jnp.float64),
            alpha=jnp.float64(params["alpha"]),
        )

        key = jax.random.PRNGKey(42)
        new_state, accept = gibbs_step(state, key)

        assert jnp.all(jnp.isfinite(new_state.eta))
        assert jnp.all(jnp.isfinite(new_state.beta))
        assert jnp.isfinite(new_state.sigma2)
        assert jnp.isfinite(new_state.rho)
        assert jnp.all(jnp.isfinite(new_state.omega))
        assert jnp.isfinite(new_state.alpha)
        assert float(new_state.sigma2) > 0
        assert float(new_state.alpha) > 0
        assert -0.999 <= float(new_state.rho) <= 0.999

    def test_multiple_steps_dont_diverge(self):
        """Running 20 Gibbs steps should not diverge."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense, WtW_dense, logdet_jax, W_dense_jax, X_jax, XtX_jax = (
            _make_jax_components(W, W_sym, WtW, W_eigs, X)
        )
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        gibbs_step = _make_gibbs_step_with_data(
            y_jax=y_jax,
            X_jax=X_jax,
            W_dense_jax=W_dense_jax,
            n=n,
            k=k,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            XtX_jax=XtX_jax,
            priors=priors,
            pg_n_terms=20,
            mh_proposal_sd=0.05,
            n_probes=10,
            lanczos_deg=30,
        )

        state = JAXGibbsState(
            eta=jnp.asarray(params["eta"], dtype=jnp.float64),
            beta=jnp.asarray(params["beta"], dtype=jnp.float64),
            sigma2=jnp.float64(params["sigma2"]),
            rho=jnp.float64(params["rho"]),
            omega=jnp.ones(n, dtype=jnp.float64),
            alpha=jnp.float64(params["alpha"]),
        )

        key = jax.random.PRNGKey(42)
        alpha_rng = np.random.default_rng(42)
        accept_count = 0

        for i in range(20):
            key, step_key = jax.random.split(key)
            state, accept = gibbs_step(state, step_key)
            accept_count += int(accept)
            alpha_new = _sample_alpha_python(state, y, priors.alpha_sigma, alpha_rng)
            state = JAXGibbsState(
                eta=state.eta,
                beta=state.beta,
                sigma2=state.sigma2,
                rho=state.rho,
                omega=state.omega,
                alpha=jnp.float64(alpha_new),
            )

        # Check that values are still finite and reasonable
        assert jnp.all(jnp.isfinite(state.eta))
        assert jnp.all(jnp.isfinite(state.beta))
        assert float(state.sigma2) > 0
        assert float(state.alpha) > 0
        assert -0.999 <= float(state.rho) <= 0.999
        # MH acceptance rate should be reasonable (>30%)
        assert accept_count / 20 > 0.3, (
            f"MH acceptance rate too low: {accept_count / 20}"
        )


# ---------------------------------------------------------------------------
# Test run_chain_jax
# ---------------------------------------------------------------------------


class TestRunChainJax:
    """Tests for run_chain_jax()."""

    def test_output_shapes(self):
        """run_chain_jax should return dict with correct shapes."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
        logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")

        init = GibbsState(
            eta=params["eta"].copy(),
            beta=params["beta"].copy(),
            sigma2=params["sigma2"],
            rho=params["rho"],
            alpha=params["alpha"],
            omega=np.ones(n),
        )

        draws, tune = 50, 50
        result = run_chain_jax(
            y=y,
            X=X,
            W_sparse=W,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            init=init,
            draws=draws,
            tune=tune,
            thin=1,
            return_eta=False,
            rng=np.random.default_rng(42),
        )

        assert result["rho"].shape == (draws,)
        assert result["beta"].shape == (draws, k)
        assert result["sigma"].shape == (draws,)
        assert result["alpha"].shape == (draws,)
        assert result["log_lik"].shape == (draws, n)
        assert result["eta_norm"].shape == (draws,)
        assert "mh_accept_rate" in result
        assert 0 < result["mh_accept_rate"] <= 1

    def test_output_with_eta(self):
        """run_chain_jax with return_eta=True should include eta array."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
        logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")

        init = GibbsState(
            eta=params["eta"].copy(),
            beta=params["beta"].copy(),
            sigma2=params["sigma2"],
            rho=params["rho"],
            alpha=params["alpha"],
            omega=np.ones(n),
        )

        draws, tune = 30, 30
        result = run_chain_jax(
            y=y,
            X=X,
            W_sparse=W,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            init=init,
            draws=draws,
            tune=tune,
            thin=1,
            return_eta=True,
            rng=np.random.default_rng(42),
        )

        assert "eta" in result
        assert result["eta"].shape == (draws, n)

    def test_thinning(self):
        """run_chain_jax with thin>1 should return correct number of draws."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
        logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")

        init = GibbsState(
            eta=params["eta"].copy(),
            beta=params["beta"].copy(),
            sigma2=params["sigma2"],
            rho=params["rho"],
            alpha=params["alpha"],
            omega=np.ones(n),
        )

        draws, tune, thin = 100, 50, 2
        result = run_chain_jax(
            y=y,
            X=X,
            W_sparse=W,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            init=init,
            draws=draws,
            tune=tune,
            thin=thin,
            return_eta=False,
            rng=np.random.default_rng(42),
        )

        n_keep = draws // thin
        assert result["rho"].shape == (n_keep,)
        assert result["beta"].shape == (n_keep, k)

    def test_values_are_finite(self):
        """All output values should be finite (except log_lik during warmup)."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        n, k = X.shape
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
        logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")

        init = GibbsState(
            eta=params["eta"].copy(),
            beta=params["beta"].copy(),
            sigma2=params["sigma2"],
            rho=params["rho"],
            alpha=params["alpha"],
            omega=np.ones(n),
        )

        result = run_chain_jax(
            y=y,
            X=X,
            W_sparse=W,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            init=init,
            draws=50,
            tune=100,
            thin=1,
            return_eta=False,
            rng=np.random.default_rng(42),
        )

        assert np.all(np.isfinite(result["rho"]))
        assert np.all(np.isfinite(result["beta"]))
        assert np.all(np.isfinite(result["sigma"]))
        assert np.all(np.isfinite(result["alpha"]))
        assert np.all(np.isfinite(result["eta_norm"]))
        # log_lik may have some non-finite values during early warmup,
        # but post-warmup draws should be mostly finite
        finite_frac = np.mean(np.isfinite(result["log_lik"]))
        assert finite_frac > 0.9, f"Only {finite_frac:.1%} of log_lik values are finite"

    def test_rho_in_bounds(self):
        """All rho draws should be within prior bounds."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
        logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")

        init = GibbsState(
            eta=params["eta"].copy(),
            beta=params["beta"].copy(),
            sigma2=params["sigma2"],
            rho=params["rho"],
            alpha=params["alpha"],
            omega=np.ones(y.shape[0]),
        )

        result = run_chain_jax(
            y=y,
            X=X,
            W_sparse=W,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            init=init,
            draws=50,
            tune=50,
            thin=1,
            return_eta=False,
            rng=np.random.default_rng(42),
        )

        assert np.all(result["rho"] >= -0.999)
        assert np.all(result["rho"] <= 0.999)

    def test_sigma_positive(self):
        """All sigma draws should be positive."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
        logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")

        init = GibbsState(
            eta=params["eta"].copy(),
            beta=params["beta"].copy(),
            sigma2=params["sigma2"],
            rho=params["rho"],
            alpha=params["alpha"],
            omega=np.ones(y.shape[0]),
        )

        result = run_chain_jax(
            y=y,
            X=X,
            W_sparse=W,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            init=init,
            draws=50,
            tune=50,
            thin=1,
            return_eta=False,
            rng=np.random.default_rng(42),
        )

        assert np.all(result["sigma"] > 0)

    def test_alpha_positive(self):
        """All alpha draws should be positive."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        priors = GibbsPriors(rho_lower=-0.999, rho_upper=0.999)

        W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
        logdet_jax = make_logdet_jax_fn(W_eigs, method="eigenvalue")

        init = GibbsState(
            eta=params["eta"].copy(),
            beta=params["beta"].copy(),
            sigma2=params["sigma2"],
            rho=params["rho"],
            alpha=params["alpha"],
            omega=np.ones(y.shape[0]),
        )

        result = run_chain_jax(
            y=y,
            X=X,
            W_sparse=W,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            init=init,
            draws=50,
            tune=50,
            thin=1,
            return_eta=False,
            rng=np.random.default_rng(42),
        )

        assert np.all(result["alpha"] > 0)


# ---------------------------------------------------------------------------
# Test _sample_alpha_python
# ---------------------------------------------------------------------------


class TestSampleAlphaPython:
    """Tests for _sample_alpha_python()."""

    def test_returns_positive(self):
        """Alpha should always be positive."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        state = JAXGibbsState(
            eta=jnp.asarray(params["eta"], dtype=jnp.float64),
            beta=jnp.zeros(2, dtype=jnp.float64),
            sigma2=jnp.float64(1.0),
            rho=jnp.float64(0.0),
            alpha=jnp.float64(params["alpha"]),
            omega=jnp.ones(20, dtype=jnp.float64),
        )
        alpha_new = _sample_alpha_python(
            state, y, alpha_sigma=10.0, rng=np.random.default_rng(42)
        )
        assert alpha_new > 0

    def test_reasonable_value(self):
        """Alpha should be in a reasonable range."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        state = JAXGibbsState(
            eta=jnp.asarray(params["eta"], dtype=jnp.float64),
            beta=jnp.zeros(2, dtype=jnp.float64),
            sigma2=jnp.float64(1.0),
            rho=jnp.float64(0.0),
            alpha=jnp.float64(params["alpha"]),
            omega=jnp.ones(20, dtype=jnp.float64),
        )
        alpha_new = _sample_alpha_python(
            state, y, alpha_sigma=10.0, rng=np.random.default_rng(42)
        )
        assert 0.01 < alpha_new < 100  # Very wide range


# ---------------------------------------------------------------------------
# Test _nb_loglik_pointwise_jax
# ---------------------------------------------------------------------------


class TestNBLoglikPointwiseJax:
    """Tests for _nb_loglik_pointwise_jax()."""

    def test_output_shape(self):
        """Log-likelihood should have shape (n,)."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        ll = _nb_loglik_pointwise_jax(y, params["eta"], params["alpha"])
        assert ll.shape == (20,)

    def test_finite_values(self):
        """Log-likelihood values should be finite for reasonable inputs."""
        y, X, W, W_sym, WtW, W_eigs, params = _make_test_data(n=20, k=2)
        ll = _nb_loglik_pointwise_jax(y, params["eta"], params["alpha"])
        assert np.all(np.isfinite(ll))
