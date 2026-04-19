"""Fast tests for model-level spatial specification diagnostics."""

from __future__ import annotations

import numpy as np

from bayespecon import (
    OLS,
    SAR,
    SDM,
    SDEM,
    SEM,
    SLX,
    OLSPanelFE,
    SARPanelFE,
    SEMPanelFE,
)
from tests.helpers import W_to_graph, make_line_W


def _cross_section_data(seed: int = 10):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.3 + 1.2 * x1 + rng.normal(scale=0.3, size=n)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def _panel_data(seed: int = 11):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 0.9 * x1 + rng.normal(scale=0.25, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def _assert_prob_dict(d: dict):
    for v in d.values():
        if hasattr(v, "pvalue"):
            p = float(v.pvalue)
            assert np.isfinite(p)
            assert 0.0 <= p <= 1.0


def test_cross_sectional_model_spatial_specification_tests_run():
    y, X, W = _cross_section_data()

    for cls in [OLS, SLX, SAR, SDM, SEM, SDEM]:
        model = cls(y=y, X=X, W=W)
        out = model.spatial_specification_tests()
        assert isinstance(out, dict)
        assert "moran" in out
        _assert_prob_dict(out)


def test_panel_fe_model_spatial_specification_tests_run():
    y, X, W, N, T = _panel_data()

    ols_fe = OLSPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sar_fe = SARPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)
    sem_fe = SEMPanelFE(y=y, X=X, W=W, N=N, T=T, model=1)

    out_ols = ols_fe.spatial_specification_tests()
    out_sar = sar_fe.spatial_specification_tests()
    out_sem = sem_fe.spatial_specification_tests()

    assert set(out_ols.keys()) == {"lm_error", "lm_sar", "lm_joint"}
    assert set(out_sar.keys()) == {"lm_error_conditional", "lr_sar"}
    assert set(out_sem.keys()) == {"lm_sar_conditional", "lr_error"}

    _assert_prob_dict(out_ols)
    _assert_prob_dict(out_sar)
    _assert_prob_dict(out_sem)
