"""Parameter recovery tests for spatial Tobit models.

These tests validate that posterior means from the spatial Tobit classes
recover core data-generating parameters within practical tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespreg import SARTobit, SDMTobit, SEMTobit
from tests.helpers import SAMPLE_KWARGS

pytestmark = pytest.mark.slow

RHO_TRUE = 0.4
LAM_TRUE = 0.4
BETA_TRUE = np.array([1.0, 1.5])
BETA2_TRUE = np.array([0.7])
SIGMA_TRUE = 0.8
CENSOR_TRUE = 0.0

ABS_TOL_SPATIAL = 0.30
ABS_TOL_BETA = 0.55
ABS_TOL_WX = 0.70


def _make_sar_tobit_data(rng: np.random.Generator, W: np.ndarray):
    n = W.shape[0]
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    eps = SIGMA_TRUE * rng.standard_normal(n)
    y_lat = np.linalg.solve(np.eye(n) - RHO_TRUE * W, X @ BETA_TRUE + eps)
    y_obs = np.maximum(CENSOR_TRUE, y_lat)
    return y_obs, X


def _make_sem_tobit_data(rng: np.random.Generator, W: np.ndarray):
    n = W.shape[0]
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    u = np.linalg.solve(np.eye(n) - LAM_TRUE * W, SIGMA_TRUE * rng.standard_normal(n))
    y_lat = X @ BETA_TRUE + u
    y_obs = np.maximum(CENSOR_TRUE, y_lat)
    return y_obs, X


def _make_sdm_tobit_data(rng: np.random.Generator, W: np.ndarray):
    n = W.shape[0]
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Wx = W @ X[:, 1:]
    eps = SIGMA_TRUE * rng.standard_normal(n)
    y_lat = np.linalg.solve(np.eye(n) - RHO_TRUE * W, X @ BETA_TRUE + Wx @ BETA2_TRUE + eps)
    y_obs = np.maximum(CENSOR_TRUE, y_lat)
    return y_obs, X


@pytest.mark.slow
def test_sar_tobit_recovers_rho_and_beta(rng, W_dense, W_graph):
    y, X = _make_sar_tobit_data(rng, W_dense)
    model = SARTobit(y=y, X=X, W=W_graph, censoring=CENSOR_TRUE)
    idata = model.fit(**SAMPLE_KWARGS)

    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for bhat, btrue in zip(beta_hat, BETA_TRUE):
        assert abs(bhat - btrue) < ABS_TOL_BETA


@pytest.mark.slow
def test_sem_tobit_recovers_lam_and_beta(rng, W_dense, W_graph):
    y, X = _make_sem_tobit_data(rng, W_dense)
    model = SEMTobit(y=y, X=X, W=W_graph, censoring=CENSOR_TRUE)
    idata = model.fit(**SAMPLE_KWARGS)

    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for bhat, btrue in zip(beta_hat, BETA_TRUE):
        assert abs(bhat - btrue) < ABS_TOL_BETA


@pytest.mark.slow
def test_sdm_tobit_recovers_rho_and_beta(rng, W_dense, W_graph):
    y, X = _make_sdm_tobit_data(rng, W_dense)
    model = SDMTobit(y=y, X=X, W=W_graph, censoring=CENSOR_TRUE)
    idata = model.fit(**SAMPLE_KWARGS)

    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    combined_true = np.concatenate([BETA_TRUE, BETA2_TRUE])
    for j, (bhat, btrue) in enumerate(zip(beta_hat, combined_true)):
        tol = ABS_TOL_WX if j >= len(BETA_TRUE) else ABS_TOL_BETA
        assert abs(bhat - btrue) < tol
