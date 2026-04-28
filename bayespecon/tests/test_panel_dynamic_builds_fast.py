"""Fast build and branch tests for dynamic panel model classes."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm

from bayespecon import OLSPanelDynamic, SDMRPanelDynamic, SDMUPanelDynamic

from .helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _panel_data(seed: int = 100):
    rng = np.random.default_rng(seed)
    N, T = 4, 4
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.3 + 0.7 * x1 + rng.normal(scale=0.2, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def test_dynamic_panel_build_pymc_models_and_prepare_dynamic_design_cache():
    y, X, W, N, T = _panel_data()

    models = [
        OLSPanelDynamic(y=y, X=X, W=W, N=N, T=T, model=0),
        SDMRPanelDynamic(y=y, X=X, W=W, N=N, T=T, model=0),
        SDMUPanelDynamic(y=y, X=X, W=W, N=N, T=T, model=0),
    ]

    for model in models:
        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)

        # Exercise the cached branch in _prepare_dynamic_design.
        z0 = model._Z_dyn
        model._prepare_dynamic_design()
        assert model._Z_dyn is z0


def test_dynamic_sdm_models_no_wx_branch_effects_and_names():
    y, _, W, N, T = _panel_data(seed=101)

    # Use X with 2 columns (intercept + x1) so WX has 1 column
    X = np.column_stack([np.ones(N * T), np.linspace(-1, 1, N * T)])

    sdmr = SDMRPanelDynamic(y=y, X=X, W=W, N=N, T=T, model=0)
    sdmr._idata = _idata(
        {
            "beta": np.array([[0.2, 0.5, 0.1], [0.201, 0.501, 0.101]]),
            "phi": np.array([0.4, 0.401]),
            "rho": np.array([0.1, 0.101]),
        }
    )

    sdmu = SDMUPanelDynamic(y=y, X=X, W=W, N=N, T=T, model=0)
    sdmu._idata = _idata(
        {
            "beta": np.array([[0.2, 0.5, 0.1], [0.201, 0.501, 0.101]]),
            "phi": np.array([0.4, 0.401]),
            "rho": np.array([0.1, 0.101]),
            "theta": np.array([0.0, 0.001]),
        }
    )

    for model in [sdmr, sdmu]:
        eff = model.spatial_effects()
        # SDM models with WX terms report effects for lagged covariates only
        assert len(eff.index) >= 1
        assert np.all(np.isfinite(eff["direct"].values))
        assert np.all(np.isfinite(eff["indirect"].values))
        assert np.all(np.isfinite(eff["total"].values))


def test_dynamic_beta_names_include_wx_labels_when_present():
    y, X, W, N, T = _panel_data(seed=102)

    model = OLSPanelDynamic(y=y, X=X, W=W, N=N, T=T, model=0)
    names = model._beta_names()

    assert any(name.startswith("W*") for name in names)
    assert len(names) > X.shape[1]
