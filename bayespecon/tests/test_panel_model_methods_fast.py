"""Fast method-level tests for static panel FE model classes."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm

from bayespecon import OLSPanelFE, SARPanelFE, SEMPanelFE, SDMPanelFE, SDEMPanelFE, SLXPanelFE
from bayespecon.models.panel_base import SpatialPanelModel
from .helpers  import W_to_graph, make_line_W


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
    sar._idata = _idata({
        "beta": np.stack([beta_2, beta_2 + 1e-3]),
        "rho": np.array([0.2, 0.201]),
    })

    sem = SEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sem._idata = _idata({
        "beta": np.stack([beta_2, beta_2 + 1e-3]),
        "lam": np.array([0.1, 0.101]),
    })

    sdm = SDMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sdm._idata = _idata({
        "beta": np.stack([beta_3, beta_3 + 1e-3]),
        "rho": np.array([0.2, 0.201]),
    })

    sdem = SDEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sdem._idata = _idata({
        "beta": np.stack([beta_3, beta_3 + 1e-3]),
        "lam": np.array([0.1, 0.101]),
    })

    slx = SLXPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    slx._idata = _idata({
        "beta": np.stack([beta_3, beta_3 + 1e-3]),
    })

    for model in [ols, sar, sem, sdm, sdem, slx]:
        fitted = model.fitted_values()
        effects = model.spatial_effects()
        assert fitted.shape == y.shape
        assert np.all(np.isfinite(fitted))
        assert set(effects.keys()) == {"direct", "indirect", "total", "feature_names"}
        assert np.all(np.isfinite(effects["direct"]))


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
