"""Parameter recovery tests for spatial panel random-effects models.

Each test generates balanced panel data (including unit random effects),
fits the RE model, and checks that the posterior mean of key parameters
is within tolerance of the true value.

Run with::

    pytest tests/test_panel_re.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import OLSPanelRE, SARPanelRE, SEMPanelRE
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
SIGMA_ALPHA_TRUE = 0.5

ABS_TOL_SPATIAL = 0.25
ABS_TOL_BETA = 0.35
ABS_TOL_SIGMA_ALPHA = 0.30


# ---------------------------------------------------------------------------
# OLS Panel RE
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ols_panel_re_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """OLSPanelRE posterior means of beta should match truth."""
    y, X, _ = make_panel_ols_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        beta=BETA_TRUE, sigma=SIGMA_TRUE, sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = OLSPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"OLSPanelRE beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


@pytest.mark.slow
def test_ols_panel_re_recovers_sigma_alpha(rng, W_panel_dense, W_panel_graph):
    """OLSPanelRE posterior mean of sigma_alpha should be close to the true value."""
    y, X, _ = make_panel_ols_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        beta=BETA_TRUE, sigma=SIGMA_TRUE, sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = OLSPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    sa_hat = float(idata.posterior["sigma_alpha"].mean())
    assert abs(sa_hat - SIGMA_ALPHA_TRUE) < ABS_TOL_SIGMA_ALPHA, (
        f"OLSPanelRE sigma_alpha: expected ≈{SIGMA_ALPHA_TRUE}, got {sa_hat:.3f}"
    )


# ---------------------------------------------------------------------------
# SAR Panel RE
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sar_panel_re_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SARPanelRE posterior mean of rho should be close to the true rho."""
    y, X, _ = make_panel_sar_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SARPanelRE rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


@pytest.mark.slow
def test_sar_panel_re_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SARPanelRE posterior means of beta should match truth."""
    y, X, _ = make_panel_sar_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SARPanelRE beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


@pytest.mark.slow
def test_sar_panel_re_recovers_sigma_alpha(rng, W_panel_dense, W_panel_graph):
    """SARPanelRE posterior mean of sigma_alpha should be close to the true value."""
    y, X, _ = make_panel_sar_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    sa_hat = float(idata.posterior["sigma_alpha"].mean())
    assert abs(sa_hat - SIGMA_ALPHA_TRUE) < ABS_TOL_SIGMA_ALPHA, (
        f"SARPanelRE sigma_alpha: expected ≈{SIGMA_ALPHA_TRUE}, got {sa_hat:.3f}"
    )


# ---------------------------------------------------------------------------
# SEM Panel RE
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sem_panel_re_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SEMPanelRE posterior mean of lambda should be close to the true value."""
    y, X, _ = make_panel_sem_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SEMPanelRE lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


@pytest.mark.slow
def test_sem_panel_re_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SEMPanelRE posterior means of beta should match truth."""
    y, X, _ = make_panel_sem_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SEMPanelRE beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


@pytest.mark.slow
def test_sem_panel_re_recovers_sigma_alpha(rng, W_panel_dense, W_panel_graph):
    """SEMPanelRE posterior mean of sigma_alpha should be close to the true value."""
    y, X, _ = make_panel_sem_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    sa_hat = float(idata.posterior["sigma_alpha"].mean())
    assert abs(sa_hat - SIGMA_ALPHA_TRUE) < ABS_TOL_SIGMA_ALPHA, (
        f"SEMPanelRE sigma_alpha: expected ≈{SIGMA_ALPHA_TRUE}, got {sa_hat:.3f}"
    )
