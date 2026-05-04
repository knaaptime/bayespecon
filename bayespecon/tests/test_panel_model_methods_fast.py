"""Fast method-level tests for static panel FE model classes."""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pymc as pm
import pytest
from libpysal.graph import Graph

from bayespecon import (
    OLSPanelFE,
    SARPanelFE,
    SDEMPanelFE,
    SDMPanelFE,
    SEMPanelFE,
    SLXPanelFE,
)
from bayespecon.models.panel_base import SpatialPanelModel

from .helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _panel_data(seed: int = 60):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 0.8 * x1 + rng.normal(scale=0.25, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def _graph_from_dense_without_transform(W_dense: np.ndarray) -> Graph:
    rows, cols = np.nonzero(W_dense)
    weights = W_dense[rows, cols]
    return Graph.from_arrays(rows, cols, weights.astype(float))


def test_panel_fe_build_pymc_models():
    y, X, W, N, T = _panel_data()

    models = [
        OLSPanelFE(y=y, X=X, W=W, N=N, T=T, model=1),
        SARPanelFE(y=y, X=X, W=W, N=N, T=T, model=1),
        SEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1),
        SDMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1),
        SDEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1),
        SLXPanelFE(y=y, X=X, W=W, N=N, T=T, model=1),
    ]

    for model in models:
        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)


def test_panel_fe_fitted_values_and_effects_with_mock_posteriors():
    y, X, W, N, T = _panel_data(seed=61)

    beta_2 = np.array([0.2, 0.9])
    beta_3 = np.array([0.2, 0.9, 0.15])  # k=2 + kw=1

    ols = OLSPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    ols._idata = _idata({"beta": np.stack([beta_2, beta_2 + 1e-3])})

    sar = SARPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sar._idata = _idata(
        {
            "beta": np.stack([beta_2, beta_2 + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    sem = SEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sem._idata = _idata(
        {
            "beta": np.stack([beta_2, beta_2 + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    sdm = SDMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sdm._idata = _idata(
        {
            "beta": np.stack([beta_3, beta_3 + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    sdem = SDEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sdem._idata = _idata(
        {
            "beta": np.stack([beta_3, beta_3 + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    slx = SLXPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    slx._idata = _idata(
        {
            "beta": np.stack([beta_3, beta_3 + 1e-3]),
        }
    )

    for model in [ols, sar, sem, sdm, sdem, slx]:
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


def test_panel_fe_spatial_effects_accept_numeric_row_standardized_graph_without_transform():
    y, X, _, N, T = _panel_data(seed=65)
    beta_3 = np.array([0.2, 0.9, 0.15])
    W = _graph_from_dense_without_transform(make_line_W(N))

    assert getattr(W, "transformation", None) == "O"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SDMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)

    model._idata = _idata(
        {
            "beta": np.stack([beta_3, beta_3 + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    assert model._is_row_std is True
    assert not any("row-standardised" in str(w.message) for w in caught)

    effects = model.spatial_effects()
    assert np.all(np.isfinite(effects[["direct", "indirect", "total"]].to_numpy()))


def test_panel_fe_warns_on_non_row_standardized_graph_without_transform():
    y, X, _, N, T = _panel_data(seed=66)
    W = _graph_from_dense_without_transform(
        np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )

    assert getattr(W, "transformation", None) == "O"

    with pytest.warns(UserWarning, match="row-standardised"):
        model = OLSPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)

    assert model._is_row_std is False


def test_sem_panel_fe_fit_adds_log_likelihood_when_missing(monkeypatch):
    y, X, W, N, T = _panel_data(seed=62)
    model = SEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)

    posterior = {
        "lam": np.array([[0.1, 0.11]]),
        "beta": np.array([[[0.2, 0.9], [0.21, 0.91]]]),
        "sigma": np.array([[1.0, 1.1]]),
    }
    fake_idata = az.from_dict(posterior=posterior)

    def _fake_super_fit(self, **kwargs):
        return fake_idata

    monkeypatch.setattr(SpatialPanelModel, "fit", _fake_super_fit)
    out = model.fit(draws=2, tune=1, chains=1, progressbar=False)

    assert "log_likelihood" in out.groups()
    assert "obs" in out.log_likelihood


def test_sem_panel_fe_fit_returns_early_when_log_likelihood_exists(monkeypatch):
    y, X, W, N, T = _panel_data(seed=63)
    model = SEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)

    n = y.shape[0]
    fake_idata = az.from_dict(
        posterior={
            "lam": np.array([[0.1, 0.11]]),
            "beta": np.array([[[0.2, 0.9], [0.21, 0.91]]]),
            "sigma": np.array([[1.0, 1.1]]),
        },
        log_likelihood={"obs": np.zeros((1, 2, n), dtype=float)},
    )

    def _fake_super_fit(self, **kwargs):
        return fake_idata

    monkeypatch.setattr(SpatialPanelModel, "fit", _fake_super_fit)
    out = model.fit(draws=2, tune=1, chains=1, progressbar=False)

    assert out is fake_idata


def test_sdem_panel_fe_fit_adds_log_likelihood_when_missing(monkeypatch):
    y, X, W, N, T = _panel_data(seed=64)
    model = SDEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)

    posterior = {
        "lam": np.array([[0.1, 0.11]]),
        "beta": np.array([[[0.2, 0.9, 0.15], [0.21, 0.91, 0.16]]]),
        "sigma": np.array([[1.0, 1.1]]),
    }
    fake_idata = az.from_dict(posterior=posterior)

    def _fake_super_fit(self, **kwargs):
        return fake_idata

    monkeypatch.setattr(SpatialPanelModel, "fit", _fake_super_fit)
    out = model.fit(draws=2, tune=1, chains=1, progressbar=False)

    assert "log_likelihood" in out.groups()
    assert "obs" in out.log_likelihood


def test_sar_panel_fe_fit_applies_jacobian_when_loglik_group_present(monkeypatch):
    """Jacobian correction should run when idata already has log_likelihood.

    This should hold even when the caller does not explicitly pass
    idata_kwargs={"log_likelihood": True}.
    """
    y, X, W, N, T = _panel_data(seed=67)
    model = SARPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)

    n = y.shape[0]
    fake_idata = az.from_dict(
        posterior={
            "rho": np.array([[0.2, 0.21]]),
            "beta": np.array([[[0.2, 0.9], [0.21, 0.91]]]),
            "sigma": np.array([[1.0, 1.1]]),
        },
        log_likelihood={"obs": np.zeros((1, 2, n), dtype=float)},
    )

    called = {"ok": False}

    def _fake_super_fit(self, **kwargs):
        return fake_idata

    def _fake_attach(idata, spatial_param, T):
        called["ok"] = True
        assert spatial_param == "rho"
        assert T == model._T

    monkeypatch.setattr(SpatialPanelModel, "fit", _fake_super_fit)
    monkeypatch.setattr(
        model, "_attach_jacobian_corrected_log_likelihood", _fake_attach
    )

    out = model.fit(draws=2, tune=1, chains=1, progressbar=False)

    assert out is fake_idata
    assert called["ok"]
