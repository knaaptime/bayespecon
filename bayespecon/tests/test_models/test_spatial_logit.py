"""Fast build/method tests for SARSpatialLogit and SEMSpatialLogit."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon import dgp
from bayespecon.models.priors import SEMSpatialLogitPriors, SpatialLogitPriors
from bayespecon.models.sem_spatial_logit import SEMSpatialLogit
from bayespecon.models.spatial_logit import SARSpatialLogit
from bayespecon.tests.helpers import W_to_graph, make_line_W


def _binary_data(seed: int = 101):
    """Create a small binary dataset for testing."""
    rng = np.random.default_rng(seed)
    n = 20
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.6 * x1
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    """SARSpatialLogit construction and validation."""

    def test_construct_matrix_mode(self):
        """Model can be constructed in matrix mode."""
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        assert model._y.shape == (20,)
        assert model._X.shape == (20, 2)

    def test_rejects_non_binary_y(self):
        """Model should reject y with values outside {0, 1}."""
        _, X, W = _binary_data()
        y_bad = np.array([0.0, 0.5, 1.0, 2.0] * 5)
        with pytest.raises(ValueError, match="binary"):
            SARSpatialLogit(y=y_bad, X=X, W=W)

    def test_rejects_robust(self):
        """robust=True should raise NotImplementedError."""
        y, X, W = _binary_data()
        with pytest.raises(NotImplementedError, match="robust"):
            SARSpatialLogit(y=y, X=X, W=W, robust=True)

    def test_priors_cls(self):
        """Model should use SpatialLogitPriors."""
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        assert model._priors_cls is SpatialLogitPriors

    def test_spatial_params(self):
        """Model should declare rho as spatial parameter."""
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        assert model._spatial_params == ("rho",)
        assert model._jacobian_param == "rho"
        assert model._model_type == "sar_logit"


# ---------------------------------------------------------------------------
# _build_pymc_model test
# ---------------------------------------------------------------------------


class TestBuildPyMCModel:
    """_build_pymc_model should raise NotImplementedError."""

    def test_raises(self):
        """Gibbs-only model should not build a PyMC model."""
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="Gibbs"):
            model._build_pymc_model()


# ---------------------------------------------------------------------------
# fit() kwarg rejection test
# ---------------------------------------------------------------------------


class TestFitKwargRejection:
    """fit() should reject NUTS-specific kwargs."""

    def test_rejects_nuts_sampler(self):
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="nuts_sampler"):
            model.fit(nuts_sampler="numpyro")

    def test_rejects_target_accept(self):
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="target_accept"):
            model.fit(target_accept=0.95)


# ---------------------------------------------------------------------------
# fitted_probabilities test
# ---------------------------------------------------------------------------


class TestFittedProbabilities:
    """fitted_probabilities should return probabilities in [0, 1]."""

    def test_output_range(self):
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        n = X.shape[0]
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "rho": np.array([0.15, 0.16]),
            }
        )
        probs = model.fitted_probabilities()
        assert probs.shape == (n,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.all(np.isfinite(probs))


# ---------------------------------------------------------------------------
# spatial_effects test
# ---------------------------------------------------------------------------


class TestSpatialEffects:
    """spatial_effects should compute direct/indirect/total impacts."""

    def test_effects_keys(self):
        y, X, W = _binary_data()
        model = SARSpatialLogit(y=y, X=X, W=W)
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "rho": np.array([0.15, 0.16]),
            }
        )
        effects = model._compute_spatial_effects()
        assert "direct" in effects
        assert "indirect" in effects
        assert "total" in effects
        assert "feature_names" in effects


# ---------------------------------------------------------------------------
# DGP integration test
# ---------------------------------------------------------------------------


class TestDGPIntegration:
    """simulate_sar_logit should produce valid data."""

    def test_dgp_output(self):
        out = dgp.simulate_sar_logit(n=5, rho=0.3, seed=42)
        assert "y" in out
        assert "X" in out
        assert "W_sparse" in out
        assert "params_true" in out
        n_obs = out["y"].shape[0]
        assert n_obs == 25  # 5x5 grid
        assert set(np.unique(out["y"])) <= {0.0, 1.0}
        assert out["params_true"]["rho"] == 0.3

    def test_model_from_dgp(self):
        """Model should accept data from simulate_sar_logit."""
        out = dgp.simulate_sar_logit(n=5, rho=0.3, seed=42)
        model = SARSpatialLogit(y=out["y"], X=out["X"], W=out["W_graph"])
        assert model._y.shape == (25,)


# ---------------------------------------------------------------------------
# SEMSpatialLogit tests
# ---------------------------------------------------------------------------


class TestSEMConstruction:
    """SEMSpatialLogit construction and validation."""

    def test_construct_matrix_mode(self):
        """Model can be constructed in matrix mode."""
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        assert model._y.shape == (20,)
        assert model._X.shape == (20, 2)

    def test_rejects_non_binary_y(self):
        """Model should reject y with values outside {0, 1}."""
        _, X, W = _binary_data()
        y_bad = np.array([0.0, 0.5, 1.0, 2.0] * 5)
        with pytest.raises(ValueError, match="binary"):
            SEMSpatialLogit(y=y_bad, X=X, W=W)

    def test_rejects_robust(self):
        """robust=True should raise NotImplementedError."""
        y, X, W = _binary_data()
        with pytest.raises(NotImplementedError, match="robust"):
            SEMSpatialLogit(y=y, X=X, W=W, robust=True)

    def test_priors_cls(self):
        """Model should use SEMSpatialLogitPriors."""
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        assert model._priors_cls is SEMSpatialLogitPriors

    def test_spatial_params(self):
        """Model should declare lam as spatial parameter."""
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        assert model._spatial_params == ("lam",)
        assert model._jacobian_param == "lam"
        assert model._model_type == "sem_logit"


class TestSEMBuildPyMCModel:
    """_build_pymc_model should raise NotImplementedError."""

    def test_raises(self):
        """Gibbs-only model should not build a PyMC model."""
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="Gibbs"):
            model._build_pymc_model()


class TestSEMFitKwargRejection:
    """fit() should reject NUTS-specific kwargs."""

    def test_rejects_nuts_sampler(self):
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="nuts_sampler"):
            model.fit(nuts_sampler="numpyro")

    def test_rejects_target_accept(self):
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="target_accept"):
            model.fit(target_accept=0.95)


class TestSEMFittedProbabilities:
    """fitted_probabilities should return probabilities in [0, 1]."""

    def test_output_range(self):
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        n = X.shape[0]
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "lam": np.array([0.15, 0.16]),
            }
        )
        probs = model.fitted_probabilities()
        assert probs.shape == (n,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.all(np.isfinite(probs))


class TestSEMSpatialEffects:
    """spatial_effects should compute direct/indirect/total impacts."""

    def test_effects_keys(self):
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "lam": np.array([0.15, 0.16]),
            }
        )
        effects = model._compute_spatial_effects()
        assert "direct" in effects
        assert "indirect" in effects
        assert "total" in effects
        assert "feature_names" in effects

    def test_sem_indirect_zero(self):
        """For SEM, indirect effects should be zero."""
        y, X, W = _binary_data()
        model = SEMSpatialLogit(y=y, X=X, W=W)
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "lam": np.array([0.15, 0.16]),
            }
        )
        effects = model._compute_spatial_effects()
        assert np.allclose(effects["indirect"], 0.0)


class TestSEMDGPIntegration:
    """simulate_sem_logit should produce valid data."""

    def test_dgp_output(self):
        out = dgp.simulate_sem_logit(n=5, lam=0.3, seed=42)
        assert "y" in out
        assert "X" in out
        assert "W_sparse" in out
        assert "params_true" in out
        n_obs = out["y"].shape[0]
        assert n_obs == 25  # 5x5 grid
        assert set(np.unique(out["y"])) <= {0.0, 1.0}
        assert out["params_true"]["lam"] == 0.3

    def test_model_from_dgp(self):
        """Model should accept data from simulate_sem_logit."""
        out = dgp.simulate_sem_logit(n=5, lam=0.3, seed=42)
        model = SEMSpatialLogit(y=out["y"], X=out["X"], W=out["W_graph"])
        assert model._y.shape == (25,)


class TestSEMFitIntegration:
    """SEMSpatialLogit.fit() should produce valid InferenceData."""

    def test_fit_small(self):
        """Fit on small DGP data should return valid InferenceData."""
        out = dgp.simulate_sem_logit(n=5, lam=0.3, seed=42)
        model = SEMSpatialLogit(y=out["y"], X=out["X"], W=out["W_graph"])
        idata = model.fit(draws=50, tune=50, chains=2, random_seed=42)
        assert "lam" in idata.posterior
        assert "beta" in idata.posterior
        assert idata.posterior["lam"].shape[0] == 2  # chains
        assert idata.posterior["beta"].shape[-1] == 2  # k coefficients
