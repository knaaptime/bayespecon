"""Fast build/method tests for SARNegativeBinomial."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytest

from bayespecon import SARNegativeBinomial, dgp

from .helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _count_data(seed: int = 101):
    rng = np.random.default_rng(seed)
    n = 10
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.6 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def test_sar_negbin_build_pymc_model():
    y, X, W = _count_data()
    model = SARNegativeBinomial(y=y, X=X, W=W)
    pymc_model = model._build_pymc_model()

    assert isinstance(pymc_model, pm.Model)
    assert "rho" in pymc_model.named_vars
    assert "alpha" in pymc_model.named_vars


def test_sar_negbin_rejects_noninteger_or_negative_y():
    _, X, W = _count_data(seed=102)

    y_bad = np.array([0.0, 1.2, 2.0, 1.0])
    X_bad = np.column_stack([np.ones(4), np.arange(4)])
    with pytest.raises(ValueError, match="integer-valued"):
        SARNegativeBinomial(y=y_bad, X=X_bad, W=W_to_graph(make_line_W(4)))

    y_neg = np.array([0.0, 1.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        SARNegativeBinomial(y=y_neg, X=X_bad, W=W_to_graph(make_line_W(4)))


def test_sar_negbin_fitted_values_and_effects_with_mock_posterior():
    y, X, W = _count_data(seed=103)
    model = SARNegativeBinomial(y=y, X=X, W=W)

    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    fitted = model.fitted_values()
    effects = model.spatial_effects()
    count_effects, samples = model.spatial_effects(
        scale="count", return_posterior_samples=True
    )

    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(fitted > 0)
    assert "direct" in effects.columns
    assert np.all(np.isfinite(effects["direct"].values))
    assert "direct" in count_effects.columns
    assert np.all(np.isfinite(count_effects["direct"].values))
    assert count_effects.attrs["scale"] == "count"
    assert samples["direct"].shape[1] == 1
    assert not np.allclose(effects["direct"].values, count_effects["direct"].values)


def test_sar_negbin_spatial_effects_rejects_unknown_scale():
    y, X, W = _count_data(seed=104)
    model = SARNegativeBinomial(y=y, X=X, W=W)
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    with pytest.raises(ValueError, match="scale must be either 'logmean' or 'count'"):
        model.spatial_effects(scale="response")


def test_simulate_sar_negbin_output_contract():
    W = W_to_graph(make_line_W(8))
    out = dgp.simulate_sar_negbin(W=W, rho=0.25, alpha=1.5, seed=42)

    assert {"y", "X", "mu", "W_dense", "W_graph", "params_true"}.issubset(out)
    y = out["y"]
    assert y.ndim == 1
    assert np.all(y >= 0)
    assert np.allclose(y, np.round(y))
    assert out["params_true"]["alpha"] == 1.5
