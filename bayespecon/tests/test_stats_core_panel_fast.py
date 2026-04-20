"""Fast unit tests for stats.core and stats.panel helpers."""

from __future__ import annotations

import numpy as np

from bayespecon import stats


def _toy_cross_section(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    W = W / np.where(rs == 0, 1, rs)
    y = 0.4 + 1.1 * x1 + rng.normal(scale=0.4, size=n)
    return y, X, W


def _toy_panel(seed: int = 1):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    W = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < N - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    W = W / np.where(rs == 0, 1, rs)
    y = 0.2 + 0.9 * x1 + rng.normal(scale=0.3, size=n)
    return y, X, W, N, T


def test_stats_core_tests_return_valid_shapes_and_probabilities():
    y, X, W = _toy_cross_section()

    out_err = stats.lmerror(y, X, W)
    out_lag = stats.lmlag(y, X, W)
    out_rho = stats.lmrho(y, X, W)
    out_rhor = stats.lmrhorob(y, X, W)
    out_mor = stats.moran(y, X, W)
    out_wal = stats.walds(y, X, W)
    out_lr = stats.lratios(y, X, W)
    out_lmsar = stats.lmsar(y, X, W, W)

    for out in [out_err, out_lag, out_rho, out_rhor, out_mor, out_wal, out_lr, out_lmsar]:
        assert "prob" in out
        assert np.isfinite(out["prob"])
        assert 0.0 <= out["prob"] <= 1.0

    assert np.isfinite(out_rho["lmrho"])
    assert np.isfinite(out_rhor["lmrhorob"])
    assert np.isfinite(out_mor["morani"])


def test_panel_core_estimators_and_lm_lr_tests_produce_valid_output():
    y, X, W, N, _ = _toy_panel()

    sem = stats.sem_panel_FE_LY(y, X, W, N)
    sar = stats.sar_panel_FE_LY(y, X, W, N)
    sarar = stats.sarar_panel_FE_LY(y, X, W, W, N)

    assert sem["meth"] == "sem_panel_FE_LY"
    assert sar["meth"] == "sar_panel_FE_LY"
    assert sarar["meth"] == "sarar_panel_FE_LY"
    assert sem["ytrans"].shape[0] == y.shape[0] - N
    assert sar["xtrans"].shape[0] == y.shape[0] - N

    out_e = stats.lm_f_err(y, X, W, N)
    out_s = stats.lm_f_sar(y, X, W, N)
    out_j = stats.lm_f_joint(y, X, W, W, N)
    out_ec = stats.lm_f_err_c(y, X, W, W, N)
    out_sc = stats.lm_f_sar_c(y, X, W, W, N)
    out_lre = stats.lr_f_err(y, X, W, N)
    out_lrs = stats.lr_f_sar(y, X, W, N)

    for out in [out_e, out_s, out_j, out_ec, out_sc, out_lre, out_lrs]:
        assert "prob" in out
        assert np.isfinite(out["prob"])
        assert 0.0 <= out["prob"] <= 1.0


def test_panel_objective_functions_return_finite_values():
    y, X, W, N, T = _toy_panel()
    rho_grid = np.linspace(-0.5, 0.5, 7)
    det = np.column_stack([rho_grid, np.zeros_like(rho_grid)])

    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    v1 = stats.f_sarpanel(0.1, det, epe0=1.0, eped=2.0, epe0d=0.5, N=N, T=T)
    v2 = stats.f2_sarpanel(np.r_[np.zeros(X.shape[1]), 0.1, 1.0], y, X, W, det, T)
    v3 = stats.f_sempanel(0.1, e, W, det, T)
    v4 = stats.f2_sempanel(np.r_[np.zeros(X.shape[1]), 0.1, 1.0], y, X, W, det, T)
    v5 = stats.f_sarar_panel(np.array([0.1, 0.1]), y, X, W, W, det, det, T)
    v6 = stats.f2_sarar_panel(np.r_[np.zeros(X.shape[1]), 0.1, 0.1, 1.0], y, X, W, W, det, det, T)

    for v in [v1, v2, v3, v4, v5, v6]:
        assert np.isfinite(v)
