"""Tests for panel flow models.

This file targets the first implementation slice:
- FlowPanelModel behavior through SARFlowPanel
- SARFlowPanel construction and demeaning
- SARFlowSeparablePanel model build
- Parameter recovery tests for all 4 panel flow variants
"""

from __future__ import annotations

import numpy as np
import pytest
from libpysal.graph import Graph

from bayespecon.graph import flow_design_matrix
from bayespecon.models.flow_panel import (
    PoissonSARFlowPanel,
    PoissonSARFlowSeparablePanel,
    SARFlowPanel,
    SARFlowSeparablePanel,
)
from bayespecon.tests.helpers import SAMPLE_KWARGS


def _make_ring_graph(n: int) -> Graph:
    """Ring-contiguity Graph (row-standardized) for n units."""
    focal = np.concatenate([np.arange(n), np.arange(n)])
    neighbor = np.concatenate([np.roll(np.arange(n), 1), np.roll(np.arange(n), -1)])
    weight = np.ones(len(focal), dtype=float)
    G = Graph.from_arrays(focal, neighbor, weight)
    return G.transform("r")


def _make_panel_flow_stack(n: int, T: int, k: int, seed: int = 0):
    """Create synthetic panel-flow stacks (time-first)."""
    rng = np.random.default_rng(seed)
    y_list = []
    X_list = []
    col_names = None

    for _ in range(T):
        X_reg = rng.standard_normal((n, k))
        design = flow_design_matrix(X_reg)
        if col_names is None:
            col_names = design.feature_names
        X_list.append(design.combined)
        y_list.append(rng.standard_normal(n * n))

    y = np.concatenate(y_list)
    X = np.vstack(X_list)
    return y, X, col_names


def _make_panel_count_vector(n: int, T: int, seed: int = 0):
    """Create synthetic non-negative integer panel-flow counts."""
    rng = np.random.default_rng(seed)
    lam = np.exp(rng.standard_normal(n * n * T) * 0.2)
    return rng.poisson(lam).astype(np.int64)


class TestFlowPanelModelConstruction:
    def test_sar_flow_panel_builds(self):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        y, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=10)

        model = SARFlowPanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
        )

        assert model._n == n
        assert model._T == T
        assert model._N_flow == n * n
        assert model._y.shape == (n * n * T,)
        assert model._X.shape[0] == n * n * T
        assert model._Wd_y.shape == (n * n * T,)

    def test_pair_fixed_effect_demeaning_zero_mean_by_pair(self):
        n = 4
        T = 4
        G = _make_ring_graph(n)
        y, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=11)

        model = SARFlowPanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=1,
            miter=5,
            titer=50,
            trace_seed=0,
        )

        y2 = model._y.reshape(T, n * n)
        np.testing.assert_allclose(y2.mean(axis=0), 0.0, atol=1e-10)

    def test_invalid_y_length_raises(self):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        y, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=12)

        with pytest.raises(ValueError, match=r"n\^2\*T"):
            SARFlowPanel(
                y=y[:-1],
                G=G,
                X=X,
                T=T,
                col_names=col_names,
                miter=5,
                titer=50,
                trace_seed=0,
            )


class TestSeparablePanelModelBuild:
    def test_sar_flow_separable_panel_builds_pymc_model(self):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        y, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=13)

        model = SARFlowSeparablePanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            trace_seed=0,
        )

        pm_model = model._build_pymc_model()
        assert pm_model is not None
        assert "rho_w" in pm_model.named_vars


class TestPoissonPanelModelBuild:
    def test_poisson_panel_builds_pymc_model(self):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=21)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=22)

        model = PoissonSARFlowPanel(
            y=y_counts,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
        )

        pm_model = model._build_pymc_model()
        assert pm_model is not None
        assert "lambda" in pm_model.named_vars

    def test_poisson_panel_requires_pooled_model(self):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=1, seed=23)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=24)

        with pytest.raises(ValueError, match="model=0 only"):
            PoissonSARFlowPanel(
                y=y_counts,
                G=G,
                X=X,
                T=T,
                col_names=col_names,
                model=1,
                miter=5,
                titer=50,
                trace_seed=0,
            )

    def test_poisson_panel_rejects_non_integer_observations(self):
        n = 4
        T = 2
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=1, seed=25)
        y_bad = _make_panel_count_vector(n=n, T=T, seed=26).astype(float) + 0.5

        with pytest.raises(ValueError, match="integer-valued"):
            PoissonSARFlowPanel(
                y=y_bad,
                G=G,
                X=X,
                T=T,
                col_names=col_names,
                model=0,
                miter=5,
                titer=50,
                trace_seed=0,
            )

    def test_poisson_separable_panel_builds_pymc_model(self):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=27)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=28)

        model = PoissonSARFlowSeparablePanel(
            y=y_counts,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            trace_seed=0,
        )

        pm_model = model._build_pymc_model()
        assert pm_model is not None
        assert "rho_w" in pm_model.named_vars

    @pytest.mark.parametrize("logdet_method", ["eigenvalue", "chebyshev", "mc_poly"])
    def test_poisson_separable_panel_supports_logdet_methods(self, logdet_method):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=31)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=32)

        model = PoissonSARFlowSeparablePanel(
            y=y_counts,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            logdet_method=logdet_method,
            trace_seed=0,
        )

        pm_model = model._build_pymc_model()
        lp = pm_model.point_logps(pm_model.initial_point())
        assert all(np.isfinite(v) for v in lp.values())

    def test_poisson_separable_panel_rejects_unknown_logdet_method(self):
        n = 4
        T = 3
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=33)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=34)

        with pytest.raises(ValueError, match="logdet_method"):
            PoissonSARFlowSeparablePanel(
                y=y_counts,
                G=G,
                X=X,
                T=T,
                col_names=col_names,
                model=0,
                logdet_method="spline",
                trace_seed=0,
            )

    def test_poisson_panel_logp_compiles_multiperiod(self):
        """Verify that logp evaluation succeeds for T>1 (catches n²T vs n² shape errors)."""
        n = 4
        T = 3
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=50)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=51)

        model = PoissonSARFlowPanel(
            y=y_counts,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        lp = pm_model.point_logps(pm_model.initial_point())
        assert all(np.isfinite(v) for v in lp.values())

    def test_poisson_separable_panel_logp_compiles_multiperiod(self):
        """Verify that logp evaluation succeeds for T>1 (separable variant)."""
        n = 4
        T = 3
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=2, seed=52)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=53)

        model = PoissonSARFlowSeparablePanel(
            y=y_counts,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        lp = pm_model.point_logps(pm_model.initial_point())
        assert all(np.isfinite(v) for v in lp.values())

    def test_poisson_panel_fit_approx_returns_posterior(self):
        n = 4
        T = 2
        G = _make_ring_graph(n)
        _, X, col_names = _make_panel_flow_stack(n=n, T=T, k=1, seed=54)
        y_counts = _make_panel_count_vector(n=n, T=T, seed=55)

        model = PoissonSARFlowPanel(
            y=y_counts,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
        )

        idata = model.fit_approx(
            draws=10,
            n=20,
            method="advi",
            progressbar=False,
            random_seed=0,
        )

        assert "beta" in idata.posterior.data_vars
        assert "lambda" not in idata.posterior.data_vars


@pytest.mark.slow
def test_sar_flow_panel_fit_smoke():
    """Minimal posterior smoke test for the unrestricted panel flow model."""
    n = 4
    T = 2
    G = _make_ring_graph(n)
    y, X, col_names = _make_panel_flow_stack(n=n, T=T, k=1, seed=15)

    model = SARFlowPanel(
        y=y,
        G=G,
        X=X,
        T=T,
        col_names=col_names,
        model=0,
        miter=5,
        titer=30,
        trace_seed=0,
        restrict_positive=True,
    )
    idata = model.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
    assert "rho_d" in idata.posterior
    assert "rho_o" in idata.posterior
    assert "rho_w" in idata.posterior
    assert "beta" in idata.posterior
    assert "sigma" in idata.posterior


# ---------------------------------------------------------------------------
# Parameter recovery tests (slow — deselected by default)
# ---------------------------------------------------------------------------

# Panel flow dimensions
PANEL_FLOW_N = 6  # 36 O-D pairs per period
PANEL_FLOW_T = 5  # 5 periods → 180 total obs

# True parameter values
PF_RHO_D_TRUE = 0.25
PF_RHO_O_TRUE = 0.25
PF_RHO_W_TRUE = 0.10
PF_BETA_D_TRUE = np.array([1.0, -0.5])
PF_BETA_O_TRUE = np.array([0.5, 0.3])
PF_SIGMA_TRUE = 1.0
PF_SIGMA_ALPHA_TRUE = 0.5

# Separable-model true values — asymmetric so rho_d ≠ rho_o (breaks swap symmetry)
# rho_w = -0.4*0.3 = -0.12 provides clear identification signal
PF_RHO_D_SEP_TRUE = 0.40
PF_RHO_O_SEP_TRUE = 0.30

# Poisson panel true values
PP_RHO_D_TRUE = 0.3
PP_RHO_O_TRUE = 0.2
PP_RHO_W_TRUE = 0.1
PP_N = 6  # 36 O-D pairs per period — matches PANEL_FLOW_N; Poisson identifies well
PANEL_FLOW_T_POISSON = 3  # fewer periods keeps cost down (each step is more expensive)

# Separable Poisson panel — asymmetric so rho_d ≠ rho_o (breaks swap symmetry)
# rho_w = -0.4*0.3 = -0.12 provides clear identification signal
PP_RHO_D_SEP_TRUE = 0.40
PP_RHO_O_SEP_TRUE = 0.30
PANEL_FLOW_T_POISSON_SEP = 4  # one extra period for the separable variant
# Larger grid for separable panel: bilinear rho_w term needs more data
PP_N_SEP = 10  # 100 O-D pairs * T=4 = 400 total — more data to identify bilinear rho_w

# Poisson models are more expensive per step (no conjugacy); use fewer samples
POISSON_SAMPLE_KWARGS: dict = dict(
    tune=400, draws=600, chains=2, random_seed=42, progressbar=False
)
# Separable Poisson: bilinear rho_w term makes the posterior harder to tune;
# 1500 tune steps gives NUTS enough budget to adapt away from (0, 0).
POISSON_SEP_SAMPLE_KWARGS: dict = dict(
    tune=1800, draws=1500, chains=2, random_seed=42, progressbar=False
)

# Tolerances
ABS_TOL_RHO = 0.25
ABS_TOL_RHO_POI = 0.30
ABS_TOL_BETA = 0.40
ABS_TOL_BETA_POI = 0.45
ABS_TOL_SIGMA = 0.35


@pytest.mark.slow
class TestSARFlowPanelRecovery:
    """SARFlowPanel posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_panel(self):
        from bayespecon.dgp.flows import generate_panel_flow_data

        G = _make_ring_graph(PANEL_FLOW_N)
        out = generate_panel_flow_data(
            n=PANEL_FLOW_N,
            T=PANEL_FLOW_T,
            G=G,
            rho_d=PF_RHO_D_TRUE,
            rho_o=PF_RHO_O_TRUE,
            rho_w=PF_RHO_W_TRUE,
            beta_d=PF_BETA_D_TRUE,
            beta_o=PF_BETA_O_TRUE,
            sigma=PF_SIGMA_TRUE,
            sigma_alpha=PF_SIGMA_ALPHA_TRUE,
            seed=42,
        )
        model = SARFlowPanel(
            y=out["y"],
            G=G,
            X=out["X"],
            T=PANEL_FLOW_T,
            col_names=out["col_names"],
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
            restrict_positive=True,
        )
        idata = model.fit(**SAMPLE_KWARGS)
        return idata

    def test_sar_flow_panel_recovers_rho_d(self, fitted_panel):
        rho_hat = float(fitted_panel.posterior["rho_d"].mean())
        assert abs(rho_hat - PF_RHO_D_TRUE) < ABS_TOL_RHO, (
            f"SARFlowPanel rho_d: expected ≈{PF_RHO_D_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_panel_recovers_rho_o(self, fitted_panel):
        rho_hat = float(fitted_panel.posterior["rho_o"].mean())
        assert abs(rho_hat - PF_RHO_O_TRUE) < ABS_TOL_RHO, (
            f"SARFlowPanel rho_o: expected ≈{PF_RHO_O_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_panel_recovers_rho_w(self, fitted_panel):
        rho_hat = float(fitted_panel.posterior["rho_w"].mean())
        assert abs(rho_hat - PF_RHO_W_TRUE) < ABS_TOL_RHO, (
            f"SARFlowPanel rho_w: expected ≈{PF_RHO_W_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_panel_recovers_sigma(self, fitted_panel):
        sigma_hat = float(fitted_panel.posterior["sigma"].mean())
        assert abs(sigma_hat - PF_SIGMA_TRUE) < ABS_TOL_SIGMA, (
            f"SARFlowPanel sigma: expected ≈{PF_SIGMA_TRUE}, got {sigma_hat:.3f}"
        )


@pytest.mark.slow
class TestSARFlowSeparablePanelRecovery:
    """SARFlowSeparablePanel posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_separable_panel(self):
        from bayespecon.dgp.flows import generate_panel_flow_data_separable

        G = _make_ring_graph(PANEL_FLOW_N)
        out = generate_panel_flow_data_separable(
            n=PANEL_FLOW_N,
            T=PANEL_FLOW_T,
            G=G,
            rho_d=PF_RHO_D_SEP_TRUE,
            rho_o=PF_RHO_O_SEP_TRUE,
            beta_d=PF_BETA_D_TRUE,
            beta_o=PF_BETA_O_TRUE,
            sigma=PF_SIGMA_TRUE,
            sigma_alpha=PF_SIGMA_ALPHA_TRUE,
            seed=42,
        )
        model = SARFlowSeparablePanel(
            y=out["y"],
            G=G,
            X=out["X"],
            T=PANEL_FLOW_T,
            col_names=out["col_names"],
            model=0,
            trace_seed=0,
        )
        idata = model.fit(**SAMPLE_KWARGS)
        return idata

    def test_separable_panel_recovers_rho_d(self, fitted_separable_panel):
        rho_hat = float(fitted_separable_panel.posterior["rho_d"].mean())
        assert abs(rho_hat - PF_RHO_D_SEP_TRUE) < ABS_TOL_RHO, (
            f"SARFlowSeparablePanel rho_d: expected ≈{PF_RHO_D_SEP_TRUE}, got {rho_hat:.3f}"
        )

    def test_separable_panel_recovers_rho_o(self, fitted_separable_panel):
        rho_hat = float(fitted_separable_panel.posterior["rho_o"].mean())
        assert abs(rho_hat - PF_RHO_O_SEP_TRUE) < ABS_TOL_RHO, (
            f"SARFlowSeparablePanel rho_o: expected ≈{PF_RHO_O_SEP_TRUE}, got {rho_hat:.3f}"
        )


@pytest.mark.slow
class TestPoissonFlowPanelRecovery:
    """PoissonSARFlowPanel posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_poisson_panel(self):
        from bayespecon.dgp.flows import generate_panel_poisson_flow_data

        G = _make_ring_graph(PP_N)
        out = generate_panel_poisson_flow_data(
            n=PP_N,
            T=PANEL_FLOW_T_POISSON,
            G=G,
            rho_d=PP_RHO_D_TRUE,
            rho_o=PP_RHO_O_TRUE,
            rho_w=PP_RHO_W_TRUE,
            seed=42,
        )
        model = PoissonSARFlowPanel(
            y=out["y"],
            G=G,
            X=out["X"],
            T=PANEL_FLOW_T_POISSON,
            col_names=out["col_names"],
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        idata = model.fit(**POISSON_SAMPLE_KWARGS)
        return idata

    def test_poisson_panel_recovers_rho_d(self, fitted_poisson_panel):
        rho_hat = float(fitted_poisson_panel.posterior["rho_d"].mean())
        assert abs(rho_hat - PP_RHO_D_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlowPanel rho_d: expected ≈{PP_RHO_D_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_panel_recovers_rho_o(self, fitted_poisson_panel):
        rho_hat = float(fitted_poisson_panel.posterior["rho_o"].mean())
        assert abs(rho_hat - PP_RHO_O_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlowPanel rho_o: expected ≈{PP_RHO_O_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_panel_recovers_rho_w(self, fitted_poisson_panel):
        rho_hat = float(fitted_poisson_panel.posterior["rho_w"].mean())
        assert abs(rho_hat - PP_RHO_W_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlowPanel rho_w: expected ≈{PP_RHO_W_TRUE}, got {rho_hat:.3f}"
        )


@pytest.mark.slow
class TestPoissonFlowSeparablePanelRecovery:
    """PoissonSARFlowSeparablePanel posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_poisson_separable_panel(self):
        from bayespecon.dgp.flows import generate_panel_poisson_flow_data_separable

        G = _make_ring_graph(PP_N_SEP)
        out = generate_panel_poisson_flow_data_separable(
            n=PP_N_SEP,
            T=PANEL_FLOW_T_POISSON_SEP,
            G=G,
            rho_d=PP_RHO_D_SEP_TRUE,
            rho_o=PP_RHO_O_SEP_TRUE,
            seed=42,
        )
        model = PoissonSARFlowSeparablePanel(
            y=out["y"],
            G=G,
            X=out["X"],
            T=PANEL_FLOW_T_POISSON_SEP,
            col_names=out["col_names"],
            model=0,
            trace_seed=0,
        )
        idata = model.fit(**POISSON_SEP_SAMPLE_KWARGS)
        return idata

    def test_poisson_separable_panel_recovers_rho_d(
        self, fitted_poisson_separable_panel
    ):
        rho_hat = float(fitted_poisson_separable_panel.posterior["rho_d"].mean())
        assert abs(rho_hat - PP_RHO_D_SEP_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlowSeparablePanel rho_d: expected ≈{PP_RHO_D_SEP_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_separable_panel_recovers_rho_o(
        self, fitted_poisson_separable_panel
    ):
        rho_hat = float(fitted_poisson_separable_panel.posterior["rho_o"].mean())
        assert abs(rho_hat - PP_RHO_O_SEP_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlowSeparablePanel rho_o: expected ≈{PP_RHO_O_SEP_TRUE}, got {rho_hat:.3f}"
        )
