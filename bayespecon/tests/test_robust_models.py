"""Tests for robust (Student-t) error distribution across model classes.

Each test verifies that:
1. ``robust=True`` builds and samples without error
2. ``nu`` (degrees of freedom) appears in the posterior
3. ``robust=False`` (default) still works (backward compatibility)
4. SpatialProbit raises ``NotImplementedError`` when ``robust=True``

Run with::

    pytest tests/test_robust_models.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import (
    OLS,
    SAR,
    SDM,
    SDEM,
    SEM,
    SLX,
    SARPanelFE,
    SEMPanelFE,
    SDMPanelFE,
    SDEMPanelFE,
    SLXPanelFE,
    OLSPanelFE,
    SARPanelRE,
    SEMPanelRE,
    OLSPanelRE,
    SARTobit,
    SEMTobit,
    SDMTobit,
    SARPanelTobit,
    SEMPanelTobit,
    SpatialProbit,
)
from bayespecon.dgp import simulate_sar, simulate_sem

from .helpers import (
    SAMPLE_KWARGS,
    PANEL_N,
    PANEL_T,
    make_rook_W,
    make_line_W,
    W_to_graph,
)

pytestmark = pytest.mark.slow

# Minimal sampling for build/smoke tests — just need posterior to exist
QUICK_KWARGS = dict(tune=100, draws=100, chains=2, random_seed=42, progressbar=False)

SIDE = 6  # 36 cross-sectional units


# ---------------------------------------------------------------------------
# Cross-sectional model tests
# ---------------------------------------------------------------------------

class TestRobustCrossSectional:
    """Robust (Student-t) error distribution for cross-sectional models."""

    @pytest.fixture
    def sar_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(W=W, rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng)
        return out["y"], out["X"], W

    @pytest.fixture
    def sem_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sem(W=W, lam=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng)
        return out["y"], out["X"], W

    def test_ols_robust_builds_and_samples(self, sar_data):
        y, X, W = sar_data
        W_graph = W_to_graph(W)
        model = OLS(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior, "nu should be in posterior when robust=True"
        assert "beta" in idata.posterior

    def test_ols_default_still_works(self, sar_data):
        y, X, W = sar_data
        W_graph = W_to_graph(W)
        model = OLS(y=y, X=X, W=W_graph)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior, "nu should NOT be in posterior when robust=False"
        assert "beta" in idata.posterior

    def test_sar_robust_builds_and_samples(self, sar_data):
        y, X, W = sar_data
        W_graph = W_to_graph(W)
        model = SAR(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior
        assert "rho" in idata.posterior

    def test_slx_robust_builds_and_samples(self, sar_data):
        y, X, W = sar_data
        W_graph = W_to_graph(W)
        model = SLX(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior
        assert "beta" in idata.posterior

    def test_sem_robust_builds_and_samples(self, sem_data):
        y, X, W = sem_data
        W_graph = W_to_graph(W)
        model = SEM(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior
        assert "lam" in idata.posterior

    def test_sdm_robust_builds_and_samples(self, sar_data):
        y, X, W = sar_data
        W_graph = W_to_graph(W)
        model = SDM(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior
        assert "rho" in idata.posterior

    def test_sdem_robust_builds_and_samples(self, sem_data):
        y, X, W = sem_data
        W_graph = W_to_graph(W)
        model = SDEM(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior
        assert "lam" in idata.posterior


# ---------------------------------------------------------------------------
# Panel FE model tests
# ---------------------------------------------------------------------------

class TestRobustPanelFE:
    """Robust (Student-t) error distribution for panel FE models."""

    @pytest.fixture
    def panel_data(self, rng):
        from bayespecon.dgp import simulate_panel_sar_fe
        W = make_line_W(PANEL_N)
        W_graph = W_to_graph(W)
        out = simulate_panel_sar_fe(
            N=PANEL_N, T=PANEL_T, rho=0.3, beta=np.array([1.0, 2.0]),
            sigma=0.5, rng=rng, W=W,
        )
        return out["y"], out["X"], W_graph

    def test_ols_panel_fe_robust(self, panel_data):
        y, X, W_graph = panel_data
        model = OLSPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, model=1, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior

    def test_sar_panel_fe_robust(self, panel_data):
        y, X, W_graph = panel_data
        model = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, model=1, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior

    def test_sem_panel_fe_robust(self, panel_data):
        y, X, W_graph = panel_data
        model = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, model=1, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior


# ---------------------------------------------------------------------------
# Panel RE model tests
# ---------------------------------------------------------------------------

class TestRobustPanelRE:
    """Robust (Student-t) error distribution for panel RE models."""

    @pytest.fixture
    def panel_re_data(self, rng):
        from bayespecon.dgp import simulate_panel_sar_re
        W = make_line_W(PANEL_N)
        W_graph = W_to_graph(W)
        out = simulate_panel_sar_re(
            N=PANEL_N, T=PANEL_T, rho=0.3, beta=np.array([1.0, 2.0]),
            sigma=0.5, rng=rng, W=W,
        )
        return out["y"], out["X"], W_graph

    def test_ols_panel_re_robust(self, panel_re_data):
        y, X, W_graph = panel_re_data
        model = OLSPanelRE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior

    def test_sar_panel_re_robust(self, panel_re_data):
        y, X, W_graph = panel_re_data
        model = SARPanelRE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior

    def test_sem_panel_re_robust(self, panel_re_data):
        y, X, W_graph = panel_re_data
        model = SEMPanelRE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior


# ---------------------------------------------------------------------------
# Tobit model tests
# ---------------------------------------------------------------------------

class TestRobustTobit:
    """Robust (Student-t) error distribution for Tobit models."""

    @pytest.fixture
    def sar_tobit_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(W=W, rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng)
        # Apply censoring at 0
        y = np.maximum(out["y"], 0.0)
        return y, out["X"], W

    @pytest.fixture
    def sem_tobit_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sem(W=W, lam=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng)
        y = np.maximum(out["y"], 0.0)
        return y, out["X"], W

    def test_sar_tobit_robust(self, sar_tobit_data):
        y, X, W = sar_tobit_data
        W_graph = W_to_graph(W)
        model = SARTobit(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior

    def test_sem_tobit_robust(self, sem_tobit_data):
        y, X, W = sem_tobit_data
        W_graph = W_to_graph(W)
        model = SEMTobit(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior

    def test_sdm_tobit_robust(self, sar_tobit_data):
        y, X, W = sar_tobit_data
        W_graph = W_to_graph(W)
        model = SDMTobit(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior


# ---------------------------------------------------------------------------
# SpatialProbit — should raise NotImplementedError
# ---------------------------------------------------------------------------

class TestRobustSpatialProbit:
    """SpatialProbit should raise NotImplementedError when robust=True."""

    def test_spatial_probit_raises(self, rng):
        W = make_line_W(6)
        W_graph = W_to_graph(W)
        m = 6  # number of regions
        region_ids = np.repeat(np.arange(m), 3)  # 3 obs per region
        n = m * 3
        X = rng.standard_normal((n, 2))
        y = rng.binomial(1, 0.5, size=n).astype(float)
        model = SpatialProbit(y=y, X=X, W=W_graph, region_ids=region_ids, robust=True)
        with pytest.raises(NotImplementedError, match="Robust.*not supported.*SpatialProbit"):
            model._build_pymc_model()


# ---------------------------------------------------------------------------
# Nu prior parameter test
# ---------------------------------------------------------------------------

class TestNuPriorParameters:
    """Test that nu_lam parameter is passed through correctly."""

    def test_custom_nu_lam(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(W=W, rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng)
        W_graph = W_to_graph(W)
        model = OLS(y=out["y"], X=out["X"], W=W_graph, robust=True, priors={"nu_lam": 1/10})
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" in idata.posterior
        # With nu_lam=1/10, mean of Exponential is 10, so nu should be > 2
        nu_mean = float(idata.posterior["nu"].mean())
        assert nu_mean > 2.0, f"nu mean should be > 2, got {nu_mean:.2f}"