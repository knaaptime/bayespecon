"""Parameter recovery tests for cross-sectional spatial models.

Each test generates synthetic data from known parameters, fits the model
once, and asserts that **all** posterior means are within tolerance of the
true values.  One fit per model — per-parameter splits were independent
refits that duplicated MCMC cost without adding coverage.

Run with::

    pytest tests/test_crosssectional.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import SAR, SDEM, SDM, SEM, SLX
from bayespecon.tests.helpers import (
    SAMPLE_KWARGS,
    make_sar_data,
    make_sdem_data,
    make_sdm_data,
    make_sem_data,
    make_slx_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# True parameters used across all cross-sectional tests
RHO_TRUE = 0.5
LAM_TRUE = 0.5
BETA_TRUE = np.array([1.0, 2.0])
BETA2_TRUE = np.array([0.8])  # spatially-lagged X coefficient (SLX/SDM/SDEM)
SIGMA_TRUE = 0.8

# Recovery tolerance: posterior mean must be within this absolute distance
# of the true value.
ABS_TOL_SPATIAL = 0.25  # for rho / lambda
ABS_TOL_BETA = 0.50  # for regression coefficients (short-chain MCMC variability)
ABS_TOL_WX = 0.65  # for lagged-X coefficients (harder to recover at N=36)


def _assert_beta(idata, true_beta, label, tol_wx=ABS_TOL_WX):
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    n_x = len(BETA_TRUE)
    for j, (bhat, btrue) in enumerate(zip(beta_hat, true_beta)):
        tol = tol_wx if j >= n_x else ABS_TOL_BETA
        assert abs(bhat - btrue) < tol, (
            f"{label} beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


def _assert_scalar(idata, name, true, tol, label):
    hat = float(idata.posterior[name].mean())
    assert abs(hat - true) < tol, (
        f"{label} {name}: expected ≈{true}, got {hat:.3f}"
    )


def test_sar_recovers_rho_and_beta(rng, W_dense, W_graph):
    y, X = make_sar_data(rng, W_dense, rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    idata = SAR(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SAR")
    _assert_beta(idata, BETA_TRUE, "SAR")


def test_sem_recovers_lam_and_beta(rng, W_dense, W_graph):
    y, X = make_sem_data(rng, W_dense, lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    idata = SEM(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SEM")
    _assert_beta(idata, BETA_TRUE, "SEM")


def test_slx_recovers_beta(rng, W_dense, W_graph):
    y, X = make_slx_data(
        rng, W_dense, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SLX(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, "SLX")


def test_sdm_recovers_rho_and_beta(rng, W_dense, W_graph):
    y, X = make_sdm_data(
        rng, W_dense, rho=RHO_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SDM(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SDM")
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, "SDM")


def test_sdem_recovers_lam_and_beta(rng, W_dense, W_graph):
    y, X = make_sdem_data(
        rng, W_dense, lam=LAM_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SDEM(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SDEM")
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, "SDEM")
