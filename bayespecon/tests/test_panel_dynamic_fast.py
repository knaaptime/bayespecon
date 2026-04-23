"""Fast tests for dynamic panel model methods without full MCMC."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from bayespecon import DLMPanelFE, SDMRPanelFE, SDMUPanelFE, SARPanelDEDynamic, SEMPanelDEDynamic, SDEMPanelDEDynamic, SLXPanelDEDynamic
from .helpers  import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _panel_data(seed: int = 50):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 0.9 * x1 + rng.normal(scale=0.25, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def test_dynamic_panel_models_fitted_values_and_effects_with_mock_posteriors():
    y, X, W, N, T = _panel_data()

    # k=2, kw=1 => beta length is 3
    beta = np.array([0.25, 0.85, 0.10])

    dlm = DLMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    dlm._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "phi": np.array([0.4, 0.401]),
    })

    sdmr = SDMRPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sdmr._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "phi": np.array([0.4, 0.401]),
        "rho": np.array([0.2, 0.201]),
    })

    sdmu = SDMUPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sdmu._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "phi": np.array([0.4, 0.401]),
        "rho": np.array([0.2, 0.201]),
        "theta": np.array([-0.1, -0.099]),
    })

    for model in [dlm, sdmr, sdmu]:
        fitted = model.fitted_values()
        effects = model.spatial_effects()

        assert fitted.shape[0] == N * (T - 1)
        assert np.all(np.isfinite(fitted))
        assert set(effects.keys()) == {"direct", "indirect", "total", "feature_names"}
        assert np.all(np.isfinite(effects["direct"]))


def test_dynamic_dlm_no_wx_branch_uses_feature_names():
    y, _, W, N, T = _panel_data(seed=51)
    X = np.ones((N * T, 1), dtype=float)  # only intercept => no WX columns

    model = DLMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata({
        "beta": np.array([[0.3], [0.301]]),
        "phi": np.array([0.5, 0.501]),
    })

    effects = model.spatial_effects()
    assert effects["feature_names"] == ["x0"]
    assert np.allclose(effects["indirect"], 0.0)


def test_dynamic_models_require_at_least_two_periods():
    rng = np.random.default_rng(52)
    N, T = 4, 1
    y = rng.normal(size=N * T)
    X = np.column_stack([np.ones(N * T), rng.normal(size=N * T)])
    W = W_to_graph(make_line_W(N))

    model = DLMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata({
        "beta": np.array([[0.2, 0.8], [0.201, 0.801]]),
        "phi": np.array([0.3, 0.301]),
    })

    with pytest.raises(ValueError, match="T >= 2"):
        model.fitted_values()


def test_dynamic_sar_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85])

    model = SARPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "phi": np.array([0.4, 0.401]),
        "rho": np.array([0.2, 0.201]),
    })

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
    assert np.all(np.isfinite(fitted))
    assert set(effects.keys()) == {"direct", "indirect", "total", "feature_names"}
    assert np.all(np.isfinite(effects["direct"]))
    # SAR has indirect effects via rho
    assert not np.allclose(effects["indirect"], 0.0)


def test_dynamic_sem_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85])

    model = SEMPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "phi": np.array([0.4, 0.401]),
        "lam": np.array([0.2, 0.201]),
    })

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
    assert np.all(np.isfinite(fitted))
    # SEM has no indirect effects
    assert np.allclose(effects["indirect"], 0.0)


def test_dynamic_sdem_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85, 0.10])  # k=2, kw=1

    model = SDEMPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "phi": np.array([0.4, 0.401]),
        "lam": np.array([0.2, 0.201]),
    })

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
    assert np.all(np.isfinite(fitted))
    assert set(effects.keys()) == {"direct", "indirect", "total", "feature_names"}


def test_dynamic_slx_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85, 0.10])  # k=2, kw=1

    model = SLXPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "phi": np.array([0.4, 0.401]),
    })

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
    assert np.all(np.isfinite(fitted))
    assert set(effects.keys()) == {"direct", "indirect", "total", "feature_names"}


def test_dynamic_sar_panel_builds_pymc_model():
    y, X, W, N, T = _panel_data()
    model = SARPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    pymc_model = model._build_pymc_model()
    assert "rho" in [v.name for v in pymc_model.free_RVs]
    assert "phi" in [v.name for v in pymc_model.free_RVs]


def test_dynamic_sem_panel_builds_pymc_model():
    y, X, W, N, T = _panel_data()
    model = SEMPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    pymc_model = model._build_pymc_model()
    assert "lam" in [v.name for v in pymc_model.free_RVs]
    assert "phi" in [v.name for v in pymc_model.free_RVs]


def test_dynamic_sdem_panel_builds_pymc_model():
    y, X, W, N, T = _panel_data()
    model = SDEMPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    pymc_model = model._build_pymc_model()
    assert "lam" in [v.name for v in pymc_model.free_RVs]
    assert "phi" in [v.name for v in pymc_model.free_RVs]


def test_dynamic_slx_panel_builds_pymc_model():
    y, X, W, N, T = _panel_data()
    model = SLXPanelDEDynamic(y=y, X=X, W=W, N=N, T=T, model=1)
    pymc_model = model._build_pymc_model()
    assert "phi" in [v.name for v in pymc_model.free_RVs]
    # SLX has no spatial lag parameter
    assert "rho" not in [v.name for v in pymc_model.free_RVs]
    assert "lam" not in [v.name for v in pymc_model.free_RVs]
