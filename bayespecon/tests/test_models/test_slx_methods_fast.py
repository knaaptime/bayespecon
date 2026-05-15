"""Fast method/build tests for SLX model."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytest

from bayespecon import SLX
from bayespecon.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _cs_data(seed: int = 110):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.4 + 0.8 * x1 + rng.normal(scale=0.3, size=n)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def _cs_data_no_intercept(seed: int = 111):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2])
    y = 0.8 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def test_slx_build_pymc_model_and_beta_names():
    y, X, W = _cs_data()
    model = SLX(y=y, X=X, W=W)

    pymc_model = model._build_pymc_model()
    assert isinstance(pymc_model, pm.Model)

    names = model._beta_names()
    assert len(names) > X.shape[1]
    assert any(name.startswith("W*") for name in names)


def test_slx_fitted_values_and_effects_with_mock_posterior():
    y, X, W = _cs_data_no_intercept(seed=111)
    model = SLX(y=y, X=X, W=W)

    k = model._X.shape[1]
    beta = np.linspace(0.3, 0.3 + 0.1 * (2 * k - 1), 2 * k)
    model._idata = _idata({"beta": np.stack([beta, beta + 1e-3])})

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert set(effects.columns) == {
        "direct",
        "direct_ci_lower",
        "direct_ci_upper",
        "direct_pvalue",
        "indirect",
        "indirect_ci_lower",
        "indirect_ci_upper",
        "indirect_pvalue",
        "total",
        "total_ci_lower",
        "total_ci_upper",
        "total_pvalue",
    }
    assert np.all(np.isfinite(effects["direct"].values))


def test_slx_fitted_values_raises_on_unexpected_beta_dimension():
    y, X, W = _cs_data(seed=112)
    model = SLX(y=y, X=X, W=W)

    bad_beta = np.array([0.3, 0.9])  # expected length is 3
    model._idata = _idata({"beta": np.stack([bad_beta, bad_beta + 1e-3])})

    with pytest.raises(ValueError, match="Unexpected beta dimension"):
        model.fitted_values()
