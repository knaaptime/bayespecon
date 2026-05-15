"""Parameter recovery tests for dynamic panel model classes.

Each test generates balanced panel data from known parameters, fits the
model, and asserts the posterior mean is within tolerance of the true value.

**Design notes**

Dynamic panel models with ``model=1`` (unit FE demeaning) suffer from
Nickell bias: the demeaned lagged dependent variable is correlated with
the demeaned error, biasing φ toward zero.  To obtain clean parameter
recovery, we use ``model=0`` (pooled) and generate data **without** unit
effects (``sigma_alpha=0``), so the DGP matches the model specification
exactly.

Run with::

    pytest tests/test_panel_dynamic_recovery.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import (
    OLSPanelDynamic,
    SARPanelDynamic,
    SDEMPanelDynamic,
    SDMRPanelDynamic,
    SDMUPanelDynamic,
    SEMPanelDynamic,
    SLXPanelDynamic,
)
from bayespecon.tests.helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_dlm_data,
    make_panel_sar_dynamic_data,
    make_panel_sdem_dynamic_data,
    make_panel_sdmr_data,
    make_panel_sdmu_data,
    make_panel_sem_dynamic_data,
    make_panel_slx_dynamic_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# True parameters
PHI_TRUE = 0.4
RHO_TRUE = 0.3
LAM_TRUE = 0.3
THETA_TRUE = -0.1
BETA_TRUE = np.array([1.0, 2.0])
SIGMA_TRUE = 1.0
# sigma_alpha=0 so DGP matches the pooled (model=0) specification exactly
SIGMA_ALPHA_TRUE = 0.0

# Tolerances — dynamic panels with pooled specification
ABS_TOL_PHI = 0.25
ABS_TOL_SPATIAL = 0.35
ABS_TOL_BETA = 0.50
ABS_TOL_THETA = 0.40


# ---------------------------------------------------------------------------
# DLM Panel FE  (non-spatial dynamic)
# ---------------------------------------------------------------------------


def test_dlm_panel_fe_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """OLSPanelDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_dlm_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = OLSPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"OLSPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_dlm_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """OLSPanelDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_dlm_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = OLSPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"OLSPanelDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDMR Panel FE  (restricted SDM dynamic)
# ---------------------------------------------------------------------------


def test_sdmr_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SDMRPanelDynamic posterior mean of rho should be close to the true value."""
    y, X, _ = make_panel_sdmr_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMRPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SDMRPanelDynamic rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


def test_sdmr_panel_fe_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SDMRPanelDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sdmr_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMRPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SDMRPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_sdmr_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDMRPanelDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sdmr_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMRPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SDMRPanelDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDMU Panel FE  (unrestricted SDM dynamic)
# ---------------------------------------------------------------------------


def test_sdmu_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelDynamic posterior mean of rho should be close to the true value."""
    y, X, _ = make_panel_sdmu_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        theta=THETA_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SDMUPanelDynamic rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


def test_sdmu_panel_fe_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sdmu_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        theta=THETA_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SDMUPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_sdmu_panel_fe_recovers_theta(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelDynamic posterior mean of theta should be close to the true value."""
    y, X, _ = make_panel_sdmu_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        theta=THETA_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    theta_hat = float(idata.posterior["theta"].mean())
    assert abs(theta_hat - THETA_TRUE) < ABS_TOL_THETA, (
        f"SDMUPanelDynamic theta: expected ≈{THETA_TRUE}, got {theta_hat:.3f}"
    )


def test_sdmu_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sdmu_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        theta=THETA_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SDMUPanelDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SAR Panel DE Dynamic  (SAR with lagged DV, no WX)
# ---------------------------------------------------------------------------


def test_sar_panel_de_dynamic_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SARPanelDynamic posterior mean of rho should be close to the true value."""
    y, X, _ = make_panel_sar_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SARPanelDynamic rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


def test_sar_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SARPanelDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sar_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SARPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_sar_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SARPanelDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sar_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SARPanelDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SEM Panel DE Dynamic  (SEM with lagged DV)
# ---------------------------------------------------------------------------


def test_sem_panel_de_dynamic_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SEMPanelDynamic posterior mean of lam should be close to the true value."""
    y, X, _ = make_panel_sem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SEMPanelDynamic lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


def test_sem_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SEMPanelDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SEMPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_sem_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SEMPanelDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SEMPanelDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDEM Panel DE Dynamic  (SDEM with lagged DV)
# ---------------------------------------------------------------------------


def test_sdem_panel_de_dynamic_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelDynamic posterior mean of lam should be close to the true value."""
    y, X, _ = make_panel_sdem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDEMPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SDEMPanelDynamic lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


def test_sdem_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sdem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDEMPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SDEMPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_sdem_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sdem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDEMPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SDEMPanelDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SLX Panel DE Dynamic  (SLX with lagged DV)
# ---------------------------------------------------------------------------


def test_slx_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SLXPanelDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_slx_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SLXPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SLXPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_slx_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SLXPanelDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_slx_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SLXPanelDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SLXPanelDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )
