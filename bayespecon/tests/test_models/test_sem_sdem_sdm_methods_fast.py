"""Fast method/build tests for SEM, SDEM, and SDM models."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm

from bayespecon import SDEM, SDM, SEM
from bayespecon.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _cs_data(seed: int = 90):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.5 + 0.9 * x1 + rng.normal(scale=0.3, size=n)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def test_sem_sdem_sdm_build_pymc_models():
    y, X, W = _cs_data()

    sem = SEM(y=y, X=X, W=W)
    sdem = SDEM(y=y, X=X, W=W)
    sdm = SDM(y=y, X=X, W=W)

    for model in [sem, sdem, sdm]:
        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)


def test_sem_sdem_sdm_fitted_values_and_effects_with_mock_posteriors():
    y, X, W = _cs_data(seed=91)

    sem = SEM(y=y, X=X, W=W)
    sem._idata = _idata(
        {
            "beta": np.stack([np.array([0.3, 0.8]), np.array([0.301, 0.801])]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    # k=2, kw=1 -> beta length 3 for SDEM/SDM
    beta3 = np.array([0.3, 0.8, 0.15])

    sdem = SDEM(y=y, X=X, W=W)
    sdem._idata = _idata(
        {
            "beta": np.stack([beta3, beta3 + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    sdm = SDM(y=y, X=X, W=W)
    sdm._idata = _idata(
        {
            "beta": np.stack([beta3, beta3 + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    for model in [sem, sdem, sdm]:
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

    sem_eff = sem.spatial_effects()
    assert np.allclose(sem_eff["indirect"].values, 0.0)


def test_sdem_and_sdm_beta_names_include_spatially_lagged_labels():
    y, X, W = _cs_data(seed=92)

    sdem = SDEM(y=y, X=X, W=W)
    sdm = SDM(y=y, X=X, W=W)

    for model in [sdem, sdm]:
        names = model._beta_names()
        assert len(names) > X.shape[1]
        assert any(name.startswith("W*") for name in names)
