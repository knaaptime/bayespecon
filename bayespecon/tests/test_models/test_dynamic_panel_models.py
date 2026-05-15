"""Parameter recovery tests for dynamic panel model implementations.

Run with::

    pytest tests/test_dynamic_panel_models.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import OLSPanelDynamic, SDMRPanelDynamic, SDMUPanelDynamic
from bayespecon.tests.helpers import (
    SAMPLE_KWARGS,
    W_to_graph,
    make_line_W,
    make_panel_dlm_data,
    make_panel_sdmr_data,
    make_panel_sdmu_data,
)

pytestmark = pytest.mark.slow

PHI_TRUE = 0.4
RHO_TRUE = 0.3
THETA_TRUE = -0.1
BETA_TRUE = np.array([1.0, 2.0])
SIGMA_TRUE = 0.8
N_DYNAMIC = 8
T_DYNAMIC = 12

ABS_TOL_DYNAMIC = 0.35
ABS_TOL_BETA = 0.50


def test_dlm_panel_fe_recovers_phi(rng):
    W = make_line_W(N_DYNAMIC)
    W_graph = W_to_graph(W)
    y, X, _ = make_panel_dlm_data(
        rng,
        W,
        N_DYNAMIC,
        T_DYNAMIC,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=0.0,
    )
    model = OLSPanelDynamic(y=y, X=X, W=W_graph, N=N_DYNAMIC, T=T_DYNAMIC, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())

    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_DYNAMIC, (
        f"OLSPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_dlm_panel_fe_recovers_beta(rng):
    W = make_line_W(N_DYNAMIC)
    W_graph = W_to_graph(W)
    y, X, _ = make_panel_dlm_data(
        rng,
        W,
        N_DYNAMIC,
        T_DYNAMIC,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=0.0,
    )
    model = OLSPanelDynamic(y=y, X=X, W=W_graph, N=N_DYNAMIC, T=T_DYNAMIC, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values

    assert abs(beta_hat[1] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"OLSPanelDynamic beta[1]: expected ≈{BETA_TRUE[1]}, got {beta_hat[1]:.3f}"
    )


def test_sdmr_panel_fe_recovers_rho_phi(rng):
    W = make_line_W(N_DYNAMIC)
    W_graph = W_to_graph(W)
    y, X, _ = make_panel_sdmr_data(
        rng,
        W,
        N_DYNAMIC,
        T_DYNAMIC,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=0.0,
    )
    model = SDMRPanelDynamic(y=y, X=X, W=W_graph, N=N_DYNAMIC, T=T_DYNAMIC, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    phi_hat = float(idata.posterior["phi"].mean())

    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_DYNAMIC, (
        f"SDMRPanelDynamic rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_DYNAMIC, (
        f"SDMRPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


def test_sdmu_panel_fe_recovers_rho_phi_theta(rng):
    W = make_line_W(N_DYNAMIC)
    W_graph = W_to_graph(W)
    y, X, _ = make_panel_sdmu_data(
        rng,
        W,
        N_DYNAMIC,
        T_DYNAMIC,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        theta=THETA_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=0.0,
    )
    model = SDMUPanelDynamic(y=y, X=X, W=W_graph, N=N_DYNAMIC, T=T_DYNAMIC, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    phi_hat = float(idata.posterior["phi"].mean())
    theta_hat = float(idata.posterior["theta"].mean())

    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_DYNAMIC, (
        f"SDMUPanelDynamic rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_DYNAMIC, (
        f"SDMUPanelDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )
    assert abs(theta_hat - THETA_TRUE) < ABS_TOL_DYNAMIC, (
        f"SDMUPanelDynamic theta: expected ≈{THETA_TRUE}, got {theta_hat:.3f}"
    )
