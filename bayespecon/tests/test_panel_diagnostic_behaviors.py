"""Fast behavior tests for panel diagnostic wiring and guard paths."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest

from bayespecon import OLSPanelFE, OLSPanelRE, SARPanelFE
from .helpers  import W_to_graph, make_line_W


def _idata_with_beta(beta: np.ndarray, rho: float | None = None):
    """Create minimal posterior container for panel model methods."""
    beta_draws = np.stack([beta, beta + 1e-3], axis=0)[None, :, :]
    posterior = {"beta": beta_draws}
    if rho is not None:
        posterior["rho"] = np.array([[rho, rho + 1e-3]])
    return az.from_dict(posterior=posterior)


def _make_panel_xy(N: int, T: int):
    rng = np.random.default_rng(123)
    n = N * T
    x1 = rng.normal(size=n)
    X = pd.DataFrame({"Intercept": np.ones(n), "x1": x1})
    y = 0.5 + 1.2 * x1 + rng.normal(scale=0.3, size=n)
    return y, X


def test_panel_diagnostics_with_re_model_rejected_for_non_hausman_class():
    """Only classes implementing hausman_test should accept re_model arg."""
    N, T = 4, 3
    W = W_to_graph(make_line_W(N))
    y, X = _make_panel_xy(N, T)

    model = SARPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    model._idata = _idata_with_beta(np.array([0.2, 1.1]), rho=0.3)

    with pytest.raises(TypeError, match="does not implement hausman_test"):
        model.panel_diagnostics(re_model=object())


def test_ols_panel_fe_hausman_rejects_wrong_model_type():
    """OLSPanelFE.hausman_test should enforce OLSPanelRE comparator type."""
    N, T = 4, 3
    W = W_to_graph(make_line_W(N))
    y, X = _make_panel_xy(N, T)

    fe_model = OLSPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    fe_model._idata = _idata_with_beta(np.array([0.1, 1.0]))

    with pytest.raises(TypeError, match="instance of OLSPanelRE"):
        fe_model.hausman_test(object())


def test_ols_panel_fe_hausman_requires_shared_coefficients():
    """Hausman should fail fast when FE and RE coefficient names do not overlap."""
    N, T = 4, 3
    W = W_to_graph(make_line_W(N))
    rng = np.random.default_rng(321)
    n = N * T

    y = rng.normal(size=n)
    X_fe = pd.DataFrame({"fe1": rng.normal(size=n), "fe2": rng.normal(size=n)})
    X_re = pd.DataFrame({"re1": rng.normal(size=n), "re2": rng.normal(size=n)})

    fe_model = OLSPanelFE(y=y, X=X_fe, W=W, N=N, T=T, model=1)
    re_model = OLSPanelRE(y=y, X=X_re, W=W, N=N, T=T)
    fe_model._idata = _idata_with_beta(np.array([0.2, -0.1]))
    re_model._idata = _idata_with_beta(np.array([0.3, -0.2]))

    with pytest.raises(ValueError, match="No shared coefficients"):
        fe_model.hausman_test(re_model)


def test_ols_panel_fe_hausman_drops_demeaned_intercept():
    """Intercept should be excluded when FE transform makes it unidentified."""
    N, T = 4, 3
    W = W_to_graph(make_line_W(N))
    y, X = _make_panel_xy(N, T)

    fe_model = OLSPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    re_model = OLSPanelRE(y=y, X=X, W=W, N=N, T=T)

    fe_model._idata = _idata_with_beta(np.array([0.4, 1.0]))
    re_model._idata = _idata_with_beta(np.array([0.5, 0.9]))

    out = fe_model.hausman_test(re_model)
    assert out.name == "hausman_fe_re"
    assert out.extra["coefficients"] == ["x1"]
    assert out.extra["n_coef"] == 1
