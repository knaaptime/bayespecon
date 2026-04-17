"""Parameter recovery tests for panel spatial Tobit models."""

from __future__ import annotations

import numpy as np
import pytest

from bayespreg import SARPanelTobit, SEMPanelTobit
from tests.helpers import PANEL_N, PANEL_T, SAMPLE_KWARGS

pytestmark = pytest.mark.slow

RHO_TRUE = 0.35
LAM_TRUE = 0.35
BETA_TRUE = np.array([1.0, 1.4])
SIGMA_TRUE = 0.8
CENSOR_TRUE = 0.0

ABS_TOL_SPATIAL = 0.35
ABS_TOL_BETA = 0.60


def _make_panel_sar_tobit_data(rng: np.random.Generator, W: np.ndarray, N: int, T: int):
    A_inv = np.linalg.inv(np.eye(N) - RHO_TRUE * W)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = np.column_stack([np.ones(N), rng.standard_normal(N)])
        eps = SIGMA_TRUE * rng.standard_normal(N)
        y_lat_t = A_inv @ (Xt @ BETA_TRUE + eps)
        y_list.append(np.maximum(CENSOR_TRUE, y_lat_t))
        X_list.append(Xt)
    return np.concatenate(y_list), np.vstack(X_list)


def _make_panel_sem_tobit_data(rng: np.random.Generator, W: np.ndarray, N: int, T: int):
    A_inv = np.linalg.inv(np.eye(N) - LAM_TRUE * W)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = np.column_stack([np.ones(N), rng.standard_normal(N)])
        u_t = A_inv @ (SIGMA_TRUE * rng.standard_normal(N))
        y_lat_t = Xt @ BETA_TRUE + u_t
        y_list.append(np.maximum(CENSOR_TRUE, y_lat_t))
        X_list.append(Xt)
    return np.concatenate(y_list), np.vstack(X_list)


@pytest.mark.slow
def test_sar_panel_tobit_recovers_rho_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X = _make_panel_sar_tobit_data(rng, W_panel_dense, PANEL_N, PANEL_T)
    model = SARPanelTobit(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, censoring=CENSOR_TRUE)
    idata = model.fit(**SAMPLE_KWARGS)

    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for bhat, btrue in zip(beta_hat, BETA_TRUE):
        assert abs(bhat - btrue) < ABS_TOL_BETA


@pytest.mark.slow
def test_sem_panel_tobit_recovers_lam_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X = _make_panel_sem_tobit_data(rng, W_panel_dense, PANEL_N, PANEL_T)
    model = SEMPanelTobit(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, censoring=CENSOR_TRUE)
    idata = model.fit(**SAMPLE_KWARGS)

    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for bhat, btrue in zip(beta_hat, BETA_TRUE):
        assert abs(bhat - btrue) < ABS_TOL_BETA
