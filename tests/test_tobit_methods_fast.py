"""Fast tests for cross-sectional Tobit model fitted/effects methods."""

from __future__ import annotations

import arviz as az
import numpy as np

from bayespecon import SARTobit, SEMTobit, SDMTobit
from tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _cs_data(seed: int = 40):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 1.0 * x1 + rng.normal(scale=0.3, size=n)
    y[[1, 4]] = 0.0  # censored observations
    W = W_to_graph(make_line_W(n))
    return y, X, W


def test_sar_sem_tobit_fitted_values_and_effects_run_with_mock_posteriors():
    y, X, W = _cs_data()

    sar = SARTobit(y=y, X=X, W=W)
    sem = SEMTobit(y=y, X=X, W=W)

    beta = np.array([0.3, 0.9])
    yc_sar = np.vstack([
        np.linspace(0.05, 0.15, sar._censored_idx.size),
        np.linspace(0.06, 0.16, sar._censored_idx.size),
    ])
    yc_sem = np.vstack([
        np.linspace(0.05, 0.15, sem._censored_idx.size),
        np.linspace(0.06, 0.16, sem._censored_idx.size),
    ])

    sar._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "rho": np.array([0.2, 0.201]),
        "y_cens_gap": yc_sar,
    })
    sem._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "lam": np.array([0.1, 0.101]),
        "y_cens_gap": yc_sem,
    })

    for m in [sar, sem]:
        fitted = m.fitted_values()
        effects = m.spatial_effects()
        assert fitted.shape == y.shape
        assert np.all(np.isfinite(fitted))
        assert np.all(np.isfinite(effects["direct"]))

    sem_eff = sem.spatial_effects()
    assert np.allclose(sem_eff["indirect"], 0.0)


def test_sdm_tobit_fitted_values_and_effects_run_with_mock_posterior():
    y, X, W = _cs_data(seed=41)
    sdm = SDMTobit(y=y, X=X, W=W)

    # k=2, kw=1 when intercept is excluded from WX terms
    beta = np.array([0.25, 0.8, 0.15])
    yc = np.vstack([
        np.linspace(0.05, 0.15, sdm._censored_idx.size),
        np.linspace(0.06, 0.16, sdm._censored_idx.size),
    ])

    sdm._idata = _idata({
        "beta": np.stack([beta, beta + 1e-3]),
        "rho": np.array([0.18, 0.181]),
        "y_cens_gap": yc,
    })

    fitted = sdm.fitted_values()
    effects = sdm.spatial_effects()

    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(np.isfinite(effects["direct"]))
    assert len(effects["feature_names"]) == 1
