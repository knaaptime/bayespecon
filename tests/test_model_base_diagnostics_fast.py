"""Fast tests for base diagnostic bundles without full MCMC runs."""

from __future__ import annotations

import arviz as az
import numpy as np

from bayespecon import OLS, OLSPanelFE
from tests.helpers import W_to_graph, make_line_W


def _idata_beta(beta: np.ndarray):
    # shape: chain=1, draw=2, coef=k
    draws = np.stack([beta, beta + 1e-3], axis=0)[None, :, :]
    return az.from_dict(posterior={"beta": draws})


def test_spatial_model_diagnostics_bundle_runs_with_mock_posterior():
    rng = np.random.default_rng(20)
    n = 10
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.4 + 1.1 * x1 + rng.normal(scale=0.4, size=n)
    W = W_to_graph(make_line_W(n))

    model = OLS(y=y, X=X, W=W)
    model._idata = _idata_beta(np.array([0.3, 1.0]))

    out = model.diagnostics(arch_lags=[1, 2], autocorr_lags=[1, 2])
    assert set(out.keys()) == {"regression", "heteroskedasticity", "autocorrelation", "outliers"}
    assert "bpagan" in out["heteroskedasticity"]
    assert "arch" in out["heteroskedasticity"]


def test_panel_model_diagnostics_bundle_runs_with_mock_posterior():
    rng = np.random.default_rng(21)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 0.8 * x1 + rng.normal(scale=0.3, size=n)
    W = W_to_graph(make_line_W(N))

    model = OLSPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata_beta(np.array([0.2, 0.9]))

    out = model.diagnostics(arch_lags=[1, 2], autocorr_lags=[1, 2])
    assert set(out.keys()) == {"regression", "heteroskedasticity", "autocorrelation", "outliers", "panel"}
    assert set(out["panel"].keys()) == {"structure", "pesaran_cd"}
