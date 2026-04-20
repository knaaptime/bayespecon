"""Tests for retaining the built PyMC model after fitting."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm

from bayespecon import SAR, SARPanelFE, SpatialProbit
from .helpers  import W_to_graph, make_line_W


def _fake_sample(*args, **kwargs):
    """Return a minimal InferenceData object for fast unit tests."""
    return az.from_dict(posterior={"dummy": np.array([[0.0]])})


def test_sar_exposes_pymc_model_after_fit(monkeypatch):
    monkeypatch.setattr(pm, "sample", _fake_sample)

    W = W_to_graph(make_line_W(8))
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(8), rng.standard_normal(8)])
    y = rng.standard_normal(8)

    model = SAR(y=y, X=X, W=W)
    assert model.pymc_model is None

    model.fit(draws=1, tune=1, chains=1)

    assert isinstance(model.pymc_model, pm.Model)


def test_panel_exposes_pymc_model_after_fit(monkeypatch):
    monkeypatch.setattr(pm, "sample", _fake_sample)

    N, T = 5, 3
    W = W_to_graph(make_line_W(N))
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(N * T), rng.standard_normal(N * T)])
    y = rng.standard_normal(N * T)

    model = SARPanelFE(y=y, X=X, W=W, N=N, T=T)
    assert model.pymc_model is None

    model.fit(draws=1, tune=1, chains=1)

    assert isinstance(model.pymc_model, pm.Model)


def test_spatial_probit_exposes_pymc_model_after_fit(monkeypatch):
    monkeypatch.setattr(pm, "sample", _fake_sample)

    W = W_to_graph(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    X = np.array(
        [
            [1.0, -0.2],
            [1.0, 0.1],
            [1.0, 0.5],
            [1.0, -0.4],
        ],
        dtype=float,
    )
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    region_ids = np.array([0, 0, 1, 1], dtype=int)

    model = SpatialProbit(y=y, X=X, W=W, region_ids=region_ids)
    assert model.pymc_model is None

    model.fit(draws=1, tune=1, chains=1)

    assert isinstance(model.pymc_model, pm.Model)
