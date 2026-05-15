"""Parameter recovery tests for spatial panel fixed-effects models.

Each test generates balanced panel data from known parameters, fits the model,
and asserts the posterior mean is within tolerance of the true value.

Run with::

    pytest tests/test_panel_fe.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import (
    OLSPanelFE,
    OLSPanelRE,
    SARPanelFE,
    SDEMPanelFE,
    SDMPanelFE,
    SEMPanelFE,
    SLXPanelFE,
)

from bayespecon.tests.helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_ols_data,
    make_panel_sar_data,
    make_panel_sdem_fe_data,
    make_panel_sdm_fe_data,
    make_panel_sem_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# True parameters
RHO_TRUE = 0.4
LAM_TRUE = 0.4
BETA_TRUE = np.array([1.0, 2.0])
BETA2_TRUE = np.array([0.8])  # WX coefficient for SDM/SDEM/SLX
SIGMA_TRUE = 0.8

ABS_TOL_SIGMA = 0.35
ABS_TOL_SPATIAL = 0.25
ABS_TOL_BETA = 0.35  # panel FE beta slightly harder to recover at small N*T
ABS_TOL_WX = 0.65  # WX coefficients are harder to recover


# ---------------------------------------------------------------------------
# OLS Panel FE (unit fixed effects)
# ---------------------------------------------------------------------------


def test_ols_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """OLSPanelFE (unit FE) posterior means of beta should match truth."""
    y, X, df = make_panel_ols_data(
        rng, W_panel_dense, PANEL_N, PANEL_T, beta=BETA_TRUE, sigma=SIGMA_TRUE
    )
    model = OLSPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    # Under unit-FE demeaning the intercept is wiped; check the slope
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"OLSPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


def test_ols_panel_fe_recovers_sigma(rng, W_panel_dense, W_panel_graph):
    """OLSPanelFE posterior mean of sigma should be close to the true value.

    With model=1 (unit FE), the intercept is absorbed by demeaning and
    sigma reflects the residual variance.  We set sigma_alpha=0 so the
    DGP has no unit effects and the model sigma matches the DGP sigma.
    """
    y, X, df = make_panel_ols_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=0.0,
    )
    model = OLSPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    sigma_hat = float(idata.posterior["sigma"].mean())
    assert abs(sigma_hat - SIGMA_TRUE) < ABS_TOL_SIGMA, (
        f"OLSPanelFE sigma: expected ≈{SIGMA_TRUE}, got {sigma_hat:.3f}"
    )


# ---------------------------------------------------------------------------
# SAR Panel FE
# ---------------------------------------------------------------------------


def test_sar_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SARPanelFE posterior mean of rho should be close to the true rho."""
    y, X, _ = make_panel_sar_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SARPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SARPanelFE rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


def test_sar_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SARPanelFE posterior means of beta (slope) should match truth."""
    y, X, _ = make_panel_sar_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SARPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"SARPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# SEM Panel FE
# ---------------------------------------------------------------------------


def test_sem_panel_fe_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SEMPanelFE posterior mean of lambda should be close to the true value."""
    y, X, _ = make_panel_sem_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SEMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SEMPanelFE lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


def test_sem_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SEMPanelFE posterior means of beta (slope) should match truth."""
    y, X, _ = make_panel_sem_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SEMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"SEMPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# SDM Panel FE  (uses SDM-type data with WX terms)
# ---------------------------------------------------------------------------


def test_sdm_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SDMPanelFE posterior mean of rho should be close to the true rho."""
    y, X, _ = make_panel_sdm_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SDMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SDMPanelFE rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


def test_sdm_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDMPanelFE posterior means of beta (slope) should match truth."""
    y, X, _ = make_panel_sdm_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SDMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    # beta covers [X, WX]; check the X slope (index 1)
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"SDMPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# SDEM Panel FE  (uses SDEM-type data with WX terms)
# ---------------------------------------------------------------------------


def test_sdem_panel_fe_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelFE posterior mean of lambda should be close to the true value."""
    y, X, _ = make_panel_sdem_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SDEMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SDEMPanelFE lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


def test_sdem_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelFE posterior means of beta (slope) should match truth."""
    y, X, _ = make_panel_sdem_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SDEMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    # beta covers [X, WX]; check the X slope (index 1)
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"SDEMPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# SLX Panel FE  (uses SDM-type data with rho=0 to generate WX signal)
# ---------------------------------------------------------------------------


def test_slx_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SLXPanelFE posterior means of beta (slope) should match truth."""
    y, X, _ = make_panel_sdm_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=0.0,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SLXPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    # beta covers [X, WX]; check the X slope (index 1)
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"SLXPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )
