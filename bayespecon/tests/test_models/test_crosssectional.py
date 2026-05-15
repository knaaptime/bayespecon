"""Parameter recovery tests for cross-sectional spatial models.

Each test generates synthetic data from known parameters, fits the model
with a moderate number of MCMC draws, and asserts the posterior mean is
within a reasonable tolerance of the true value.

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

# Recovery tolerance: posterior mean must be within this fraction of the range
# or within this absolute distance of the true value.
ABS_TOL_SPATIAL = 0.25  # for rho / lambda
ABS_TOL_BETA = 0.50  # for regression coefficients (short-chain MCMC variability)
ABS_TOL_WX = 0.65  # for lagged-X coefficients (harder to recover at N=36)


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------


def test_sar_recovers_rho(rng, W_dense, W_graph):
    """SAR posterior mean of rho should be close to the true rho."""
    y, X = make_sar_data(rng, W_dense, rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SAR(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SAR rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


def test_sar_recovers_beta(rng, W_dense, W_graph):
    """SAR posterior means of beta should be close to the true betas."""
    y, X = make_sar_data(rng, W_dense, rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SAR(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SAR beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SEM
# ---------------------------------------------------------------------------


def test_sem_recovers_lam(rng, W_dense, W_graph):
    """SEM posterior mean of lambda should be close to the true lambda."""
    y, X = make_sem_data(rng, W_dense, lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SEM(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SEM lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


def test_sem_recovers_beta(rng, W_dense, W_graph):
    """SEM posterior means of beta should be close to the true betas."""
    y, X = make_sem_data(rng, W_dense, lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SEM(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SEM beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SLX
# ---------------------------------------------------------------------------


def test_slx_recovers_beta(rng, W_dense, W_graph):
    """SLX posterior means of beta (X + WX) should be close to true values."""
    y, X = make_slx_data(
        rng, W_dense, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    model = SLX(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    combined_true = np.concatenate([BETA_TRUE, BETA2_TRUE])
    for j, (bhat, btrue) in enumerate(zip(beta_hat, combined_true)):
        tol = ABS_TOL_WX if j >= len(BETA_TRUE) else ABS_TOL_BETA
        assert abs(bhat - btrue) < tol, (
            f"SLX beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDM
# ---------------------------------------------------------------------------


def test_sdm_recovers_rho(rng, W_dense, W_graph):
    """SDM posterior mean of rho should be close to the true rho."""
    y, X = make_sdm_data(
        rng, W_dense, rho=RHO_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    model = SDM(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SDM rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


def test_sdm_recovers_beta(rng, W_dense, W_graph):
    """SDM posterior means of beta (X + WX) should be close to true values."""
    y, X = make_sdm_data(
        rng, W_dense, rho=RHO_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    model = SDM(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    combined_true = np.concatenate([BETA_TRUE, BETA2_TRUE])
    for j, (bhat, btrue) in enumerate(zip(beta_hat, combined_true)):
        tol = ABS_TOL_WX if j >= len(BETA_TRUE) else ABS_TOL_BETA
        assert abs(bhat - btrue) < tol, (
            f"SDM beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDEM
# ---------------------------------------------------------------------------


def test_sdem_recovers_lam(rng, W_dense, W_graph):
    """SDEM posterior mean of lambda should be close to the true lambda."""
    y, X = make_sdem_data(
        rng, W_dense, lam=LAM_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    model = SDEM(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SDEM lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


def test_sdem_recovers_beta(rng, W_dense, W_graph):
    """SDEM posterior means of beta should be close to true values."""
    y, X = make_sdem_data(
        rng, W_dense, lam=LAM_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    model = SDEM(y=y, X=X, W=W_graph)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    combined_true = np.concatenate([BETA_TRUE, BETA2_TRUE])
    for j, (bhat, btrue) in enumerate(zip(beta_hat, combined_true)):
        tol = ABS_TOL_WX if j >= len(BETA_TRUE) else ABS_TOL_BETA
        assert abs(bhat - btrue) < tol, (
            f"SDEM beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )
