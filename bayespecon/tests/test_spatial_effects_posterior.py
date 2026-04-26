"""Tests for Bayesian spatial effects (impact measures) inference.

Verifies that posterior-draw-level computation of direct, indirect, and total
effects matches the existing posterior-mean computation, and that the
DataFrame returned by spatial_effects() is correctly constructed.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest

from bayespecon import SAR, SDM, SLX, SDEM, SEM
from bayespecon.diagnostics.spatial_effects import (
    _build_effects_dataframe,
    _compute_bayesian_pvalue,
    _compute_ci,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_W(n: int = 5) -> np.ndarray:
    """Create a simple row-standardised rook weights matrix."""
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1
        if i < n - 1:
            W[i, i + 1] = 1
    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


def _W_to_graph(W: np.ndarray):
    """Convert a numpy weights matrix to a libpysal Graph."""
    from libpysal.graph import Graph
    from scipy.sparse import csr_matrix
    return Graph.from_sparse(csr_matrix(W))


def _set_posterior_means(model, beta: np.ndarray, rho: float | None = None) -> None:
    """Inject posterior means into a model for testing without MCMC."""
    posterior: dict[str, np.ndarray] = {
        "beta": np.array([[beta]], dtype=float),
    }
    if rho is not None:
        posterior["rho"] = np.array([[rho]], dtype=float)
    model._idata = az.from_dict(posterior=posterior)


def _set_posterior_draws(model, beta_draws: np.ndarray, rho_draws: np.ndarray | None = None) -> None:
    """Inject posterior draws into a model for testing.

    Parameters
    ----------
    beta_draws : np.ndarray, shape (chains, draws, k) or (draws, k)
    rho_draws : np.ndarray, shape (chains, draws) or (draws,), optional
    """
    if beta_draws.ndim == 2:
        # Add chain dimension
        beta_draws = beta_draws[np.newaxis, :, :]
    posterior: dict[str, np.ndarray] = {
        "beta": beta_draws,
    }
    if rho_draws is not None:
        if rho_draws.ndim == 1:
            rho_draws = rho_draws[np.newaxis, :]
        posterior["rho"] = rho_draws
    model._idata = az.from_dict(posterior=posterior)


# ------------------------------------------------------------------
# Tests for helper functions
# ------------------------------------------------------------------

class TestComputeBayesianPvalue:
    def test_all_positive(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pval = _compute_bayesian_pvalue(samples)
        assert pval.shape == (1,)
        # All positive => P(effect < 0) = 0 => p = 2 * 0 = 0
        assert pval[0] == pytest.approx(0.0)

    def test_all_negative(self):
        samples = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        pval = _compute_bayesian_pvalue(samples)
        # All negative => P(effect > 0) = 0 => p = 0
        assert pval[0] == pytest.approx(0.0)

    def test_symmetric_around_zero(self):
        samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        pval = _compute_bayesian_pvalue(samples)
        # P(>0) = 2/5, P(<0) = 2/5 => p = 2 * 2/5 = 4/5
        assert pval[0] == pytest.approx(0.8)

    def test_multidimensional(self):
        samples = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, 0.5]])
        pval = _compute_bayesian_pvalue(samples)
        assert pval.shape == (2,)
        assert pval[0] == pytest.approx(0.0)  # all positive
        # Second column: P(>0) = 1/3, P(<0) = 2/3 => p = 2 * 1/3
        assert pval[1] == pytest.approx(2.0 / 3.0)


class TestComputeCI:
    def test_1d(self):
        samples = np.arange(100, dtype=float)
        ci = _compute_ci(samples)
        assert len(ci) == 1
        # 2.5th percentile of [0..99] is 2.475, 97.5th is 96.525
        assert ci[0][0] == pytest.approx(2.475, abs=0.01)
        assert ci[0][1] == pytest.approx(96.525, abs=0.01)

    def test_2d(self):
        samples = np.column_stack([np.arange(100, dtype=float), np.arange(100, 200, dtype=float)])
        ci = _compute_ci(samples)
        assert len(ci) == 2


# ------------------------------------------------------------------
# Tests for SAR spatial effects posterior
# ------------------------------------------------------------------

class TestSARSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        """Posterior mean of draw-level effects should match existing point estimate."""
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SAR(y=y, X=X, W=_W_to_graph(W))

        # Set up posterior draws
        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        # Get posterior effects
        direct_samples, indirect_samples, total_samples = model._compute_spatial_effects_posterior()

        # Verify shapes (intercept excluded from effects)
        assert direct_samples.shape == (G, 1)
        assert indirect_samples.shape == (G, 1)
        assert total_samples.shape == (G, 1)

        # Verify that indirect = total - direct
        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)

        # Verify that posterior mean is close to existing computation
        # (within Monte Carlo error for 100 draws)
        rho_mean = float(np.mean(rho_draws))
        beta_mean = np.mean(beta_draws, axis=0)
        _set_posterior_means(model, beta=beta_mean, rho=rho_mean)
        existing = model._compute_spatial_effects()

        np.testing.assert_allclose(
            np.mean(direct_samples, axis=0),
            existing["direct"],
            atol=0.15,  # Allow MC error
        )

    def test_spatial_effects_result(self):
        """Test that spatial_effects() returns a DataFrame."""
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SAR(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        result = model.spatial_effects()
        assert isinstance(result, pd.DataFrame)
        assert result.attrs["model_type"] == "SAR"
        assert result.attrs["n_draws"] == G
        # SAR excludes intercept from effects
        assert len(result.index) == 1
        # Check expected columns
        expected_cols = {
            "direct", "direct_ci_lower", "direct_ci_upper", "direct_pvalue",
            "indirect", "indirect_ci_lower", "indirect_ci_upper", "indirect_pvalue",
            "total", "total_ci_lower", "total_ci_upper", "total_pvalue",
        }
        assert set(result.columns) == expected_cols
        # P-values should be in [0, 1]
        assert np.all(result["direct_pvalue"].values >= 0)
        assert np.all(result["direct_pvalue"].values <= 1)
        assert np.all(result["indirect_pvalue"].values >= 0)
        assert np.all(result["indirect_pvalue"].values <= 1)

    def test_return_posterior_samples(self):
        """Test that spatial_effects(return_posterior_samples=True) returns DataFrame and dict."""
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SAR(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        df, samples = model.spatial_effects(return_posterior_samples=True)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(samples, dict)
        assert "direct" in samples
        assert "indirect" in samples
        assert "total" in samples
        assert samples["direct"].shape == (G, 1)
        assert samples["indirect"].shape == (G, 1)
        assert samples["total"].shape == (G, 1)
        # Verify posterior means match DataFrame values
        np.testing.assert_allclose(df["direct"].values, np.mean(samples["direct"], axis=0))


# ------------------------------------------------------------------
# Tests for SDM spatial effects posterior
# ------------------------------------------------------------------

class TestSDMSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SDM(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        # SDM beta = [intercept, x1, W*x1]
        beta_draws = np.random.randn(G, 3)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        direct_samples, indirect_samples, total_samples = model._compute_spatial_effects_posterior()

        # SDM has k_wx = 1 (only x1 is lagged)
        assert direct_samples.shape == (G, 1)
        assert indirect_samples.shape == (G, 1)
        assert total_samples.shape == (G, 1)

        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)

        # Compare with existing
        rho_mean = float(np.mean(rho_draws))
        beta_mean = np.mean(beta_draws, axis=0)
        _set_posterior_means(model, beta=beta_mean, rho=rho_mean)
        existing = model._compute_spatial_effects()

        np.testing.assert_allclose(
            np.mean(direct_samples, axis=0),
            existing["direct"],
            atol=0.15,
        )


# ------------------------------------------------------------------
# Tests for SLX spatial effects posterior
# ------------------------------------------------------------------

class TestSLXSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SLX(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 3)  # [intercept, x1, W*x1]
        _set_posterior_draws(model, beta_draws)

        direct_samples, indirect_samples, total_samples = model._compute_spatial_effects_posterior()

        assert direct_samples.shape == (G, 1)
        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)

        # Compare with existing
        beta_mean = np.mean(beta_draws, axis=0)
        _set_posterior_means(model, beta=beta_mean)
        existing = model._compute_spatial_effects()

        np.testing.assert_allclose(
            np.mean(direct_samples, axis=0),
            existing["direct"],
            atol=0.15,
        )


# ------------------------------------------------------------------
# Tests for SDEM spatial effects posterior
# ------------------------------------------------------------------

class TestSDEMSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SDEM(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 3)  # [intercept, x1, W*x1]
        _set_posterior_draws(model, beta_draws)

        direct_samples, indirect_samples, total_samples = model._compute_spatial_effects_posterior()

        assert direct_samples.shape == (G, 1)
        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)

        beta_mean = np.mean(beta_draws, axis=0)
        _set_posterior_means(model, beta=beta_mean)
        existing = model._compute_spatial_effects()

        np.testing.assert_allclose(
            np.mean(direct_samples, axis=0),
            existing["direct"],
            atol=0.15,
        )


# ------------------------------------------------------------------
# Tests for SEM spatial effects posterior
# ------------------------------------------------------------------

class TestSEMSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SEM(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        _set_posterior_draws(model, beta_draws)

        direct_samples, indirect_samples, total_samples = model._compute_spatial_effects_posterior()

        # SEM excludes intercept from effects
        assert direct_samples.shape == (G, 1)
        # SEM: indirect = 0, total = direct
        np.testing.assert_allclose(indirect_samples, 0.0)
        np.testing.assert_allclose(total_samples, direct_samples)

        beta_mean = np.mean(beta_draws, axis=0)
        _set_posterior_means(model, beta=beta_mean)
        existing = model._compute_spatial_effects()

        np.testing.assert_allclose(
            np.mean(direct_samples, axis=0),
            existing["direct"],
            atol=0.15,
        )


# ------------------------------------------------------------------
# Tests for _build_effects_dataframe
# ------------------------------------------------------------------

class TestBuildEffectsDataFrame:
    def test_construction(self):
        G = 50
        k = 3
        rng = np.random.default_rng(42)
        direct_samples = rng.normal(size=(G, k))
        indirect_samples = rng.normal(size=(G, k))
        total_samples = direct_samples + indirect_samples
        feature_names = ["x1", "x2", "x3"]

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type="SAR",
        )

        assert isinstance(df, pd.DataFrame)
        assert df.attrs["model_type"] == "SAR"
        assert df.attrs["n_draws"] == G
        assert list(df.index) == feature_names
        assert df.index.name == "variable"
        # Check all expected columns
        expected_cols = {
            "direct", "direct_ci_lower", "direct_ci_upper", "direct_pvalue",
            "indirect", "indirect_ci_lower", "indirect_ci_upper", "indirect_pvalue",
            "total", "total_ci_lower", "total_ci_upper", "total_pvalue",
        }
        assert set(df.columns) == expected_cols
        # Check posterior means match
        np.testing.assert_allclose(df["direct"].values, np.mean(direct_samples, axis=0))
        np.testing.assert_allclose(df["indirect"].values, np.mean(indirect_samples, axis=0))
        np.testing.assert_allclose(df["total"].values, np.mean(total_samples, axis=0))


