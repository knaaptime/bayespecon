"""Parameter recovery tests for spatial panel fixed-effects models.

Each test generates balanced panel data from known parameters, fits the model,
and asserts the posterior mean is within tolerance of the true value.

Run with::

    pytest tests/test_panel_fe.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import OLSPanelFE, OLSPanelRE, SARPanelFE, SDEMPanelFE, SDMPanelFE, SEMPanelFE
from .helpers  import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_ols_data,
    make_panel_sar_data,
    make_panel_sem_data,
)

pytestmark = pytest.mark.slow

# True parameters
RHO_TRUE = 0.4
LAM_TRUE = 0.4
BETA_TRUE = np.array([1.0, 2.0])
SIGMA_TRUE = 0.8

ABS_TOL_SPATIAL = 0.25
ABS_TOL_BETA = 0.35   # panel FE beta slightly harder to recover at small N*T


# ---------------------------------------------------------------------------
# OLS Panel FE (unit fixed effects)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ols_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """OLSPanelFE (unit FE) posterior means of beta should match truth."""
    y, X, df = make_panel_ols_data(rng, W_panel_dense, PANEL_N, PANEL_T,
                                    beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = OLSPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    # Under unit-FE demeaning the intercept is wiped; check the slope
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"OLSPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# SAR Panel FE
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sar_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SARPanelFE posterior mean of rho should be close to the true rho."""
    y, X, _ = make_panel_sar_data(rng, W_panel_dense, PANEL_N, PANEL_T,
                                   rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SARPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SARPanelFE rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


@pytest.mark.slow
def test_sar_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SARPanelFE posterior means of beta (slope) should match truth."""
    y, X, _ = make_panel_sar_data(rng, W_panel_dense, PANEL_N, PANEL_T,
                                   rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SARPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"SARPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# SEM Panel FE
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sem_panel_fe_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SEMPanelFE posterior mean of lambda should be close to the true value."""
    y, X, _ = make_panel_sem_data(rng, W_panel_dense, PANEL_N, PANEL_T,
                                   lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SEMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SEMPanelFE lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


@pytest.mark.slow
def test_sem_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SEMPanelFE posterior means of beta (slope) should match truth."""
    y, X, _ = make_panel_sem_data(rng, W_panel_dense, PANEL_N, PANEL_T,
                                   lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SEMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"SEMPanelFE beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# SDM Panel FE  (uses SAR-type data; WX term adds extra signal)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sdm_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SDMPanelFE posterior mean of rho should be close to the true rho."""
    y, X, _ = make_panel_sar_data(rng, W_panel_dense, PANEL_N, PANEL_T,
                                   rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SDMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SDMPanelFE rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


# ---------------------------------------------------------------------------
# SDEM Panel FE  (uses SEM-type data)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sdem_panel_fe_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelFE posterior mean of lambda should be close to the true value."""
    y, X, _ = make_panel_sem_data(rng, W_panel_dense, PANEL_N, PANEL_T,
                                   lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    model = SDEMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=1)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SDEMPanelFE lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )
