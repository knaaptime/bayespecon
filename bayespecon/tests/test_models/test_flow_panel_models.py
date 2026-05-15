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

from bayespecon.models.flow_panel import (
    OLSFlowPanel,
    PoissonFlowPanel,
    PoissonSARFlowPanel,
    PoissonSARFlowSeparablePanel,
    SARFlowPanel,
    SARFlowSeparablePanel,
)
from bayespecon.tests.helpers import SAMPLE_KWARGS


def _flow_test_graph(n: int):
    """Return a small Graph for tests via bayespecon's DGP machinery."""
    from bayespecon.dgp.flows import generate_flow_data

    return generate_flow_data(n=n, seed=0)["G"]


def _panel_flow_stack(n: int, T: int, k: int, seed: int = 0):
    """Build a noise-only panel-flow stack via bayespecon's panel DGP."""
    from bayespecon.dgp.flows import generate_panel_flow_data

    out = generate_panel_flow_data(
        n=n,
        T=T,
        k=k,
        rho_d=0.0,
        rho_o=0.0,
        rho_w=0.0,
        beta_d=np.zeros(k),
        beta_o=np.zeros(k),
        sigma=1.0,
        sigma_alpha=0.0,
        seed=seed,
    )
    return out["y"], out["X"], out["col_names"]


def _panel_count_vector(n: int, T: int, seed: int = 0):
    """Build noise-only Poisson panel-flow counts via bayespecon's panel DGP."""
    from bayespecon.dgp.flows import generate_panel_poisson_flow_data

    out = generate_panel_poisson_flow_data(
        n=n,
        T=T,
        k=1,
        rho_d=0.0,
        rho_o=0.0,
        rho_w=0.0,
        beta_d=np.zeros(1),
        beta_o=np.zeros(1),
        seed=seed,
    )
    return out["y"]


class TestFlowPanelModelConstruction:
    def test_sar_flow_panel_builds(self):
        n = 4
        T = 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=10)

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
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=11)

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
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=12)

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
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=13)

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
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=21)
        y_counts = _panel_count_vector(n=n, T=T, seed=22)

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
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=1, seed=23)
        y_counts = _panel_count_vector(n=n, T=T, seed=24)

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
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=1, seed=25)
        y_bad = _panel_count_vector(n=n, T=T, seed=26).astype(float) + 0.5

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
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=27)
        y_counts = _panel_count_vector(n=n, T=T, seed=28)

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

    @pytest.mark.parametrize("logdet_method", ["eigenvalue", "chebyshev", "trace_mc"])
    def test_poisson_separable_panel_supports_logdet_methods(self, logdet_method):
        n = 4
        T = 3
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=31)
        y_counts = _panel_count_vector(n=n, T=T, seed=32)

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
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=33)
        y_counts = _panel_count_vector(n=n, T=T, seed=34)

        with pytest.raises(ValueError, match="logdet_method"):
            PoissonSARFlowSeparablePanel(
                y=y_counts,
                G=G,
                X=X,
                T=T,
                col_names=col_names,
                model=0,
                logdet_method="sparse_spline",
                trace_seed=0,
            )

    def test_poisson_panel_logp_compiles_multiperiod(self):
        """Verify that logp evaluation succeeds for T>1 (catches n²T vs n² shape errors)."""
        n = 4
        T = 3
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=50)
        y_counts = _panel_count_vector(n=n, T=T, seed=51)

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
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=52)
        y_counts = _panel_count_vector(n=n, T=T, seed=53)

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
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=1, seed=54)
        y_counts = _panel_count_vector(n=n, T=T, seed=55)

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

    @pytest.mark.parametrize(
        ("model_cls", "method", "extra_kwargs"),
        [
            (SARFlowPanel, "advi", {"miter": 5, "titer": 50}),
            (SARFlowSeparablePanel, "fullrank_advi", {}),
        ],
    )
    def test_gaussian_panel_fit_approx_returns_expected_posterior_vars(
        self, model_cls, method, extra_kwargs
    ):
        n = 4
        T = 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=1, seed=56)

        model = model_cls(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            trace_seed=0,
            **extra_kwargs,
        )

        idata = model.fit_approx(
            draws=10,
            n=20,
            method=method,
            progressbar=False,
            random_seed=0,
        )

        assert model.approximation is not None
        assert {"beta", "rho_d", "rho_o", "rho_w", "sigma"} <= set(
            idata.posterior.data_vars
        )

    def test_poisson_separable_panel_fit_approx_returns_expected_posterior_vars(self):
        n = 4
        T = 2
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=1, seed=57)
        y_counts = _panel_count_vector(n=n, T=T, seed=58)

        model = PoissonSARFlowSeparablePanel(
            y=y_counts,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            trace_seed=0,
        )

        idata = model.fit_approx(
            draws=10,
            n=20,
            method="fullrank_advi",
            progressbar=False,
            random_seed=0,
        )

        assert model.approximation is not None
        assert {"beta", "rho_d", "rho_o", "rho_w"} <= set(idata.posterior.data_vars)
        assert "lambda" not in idata.posterior.data_vars


@pytest.mark.slow
def test_sar_flow_panel_fit_smoke():
    """Minimal posterior smoke test for the unrestricted panel flow model."""
    n = 4
    T = 2
    G = _flow_test_graph(n)
    y, X, col_names = _panel_flow_stack(n=n, T=T, k=1, seed=15)

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

        G = _flow_test_graph(PANEL_FLOW_N)
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
        # Default panel-flow DGP is lognormal; SARFlowPanel has Gaussian
        # likelihood, so fit on np.log(y) (== eta).
        model = SARFlowPanel(
            y=np.log(out["y"]),
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

    def test_sar_flow_panel_recovers_beta(self, fitted_panel):
        # DGP design layout is [alpha, intra, beta_d (k cols), beta_o (k cols),
        # ..., gamma_dist]; recover the beta_d / beta_o blocks only.
        k = len(PF_BETA_D_TRUE)
        beta_hat = fitted_panel.posterior["beta"].mean(dim=("chain", "draw")).values
        beta_d_hat = beta_hat[2 : 2 + k]
        beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
        assert np.allclose(beta_d_hat, PF_BETA_D_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowPanel beta_d: expected ≈{PF_BETA_D_TRUE}, got {beta_d_hat}"
        )
        assert np.allclose(beta_o_hat, PF_BETA_O_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowPanel beta_o: expected ≈{PF_BETA_O_TRUE}, got {beta_o_hat}"
        )


@pytest.mark.slow
class TestSARFlowSeparablePanelRecovery:
    """SARFlowSeparablePanel posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_separable_panel(self):
        from bayespecon.dgp.flows import generate_panel_flow_data_separable

        G = _flow_test_graph(PANEL_FLOW_N)
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
        # Default panel-flow DGP is lognormal; fit on np.log(y).
        model = SARFlowSeparablePanel(
            y=np.log(out["y"]),
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

    def test_separable_panel_recovers_beta(self, fitted_separable_panel):
        # Same design layout as SARFlowPanel: [alpha, intra, beta_d (k),
        # beta_o (k), ..., gamma_dist].
        k = len(PF_BETA_D_TRUE)
        beta_hat = (
            fitted_separable_panel.posterior["beta"].mean(dim=("chain", "draw")).values
        )
        beta_d_hat = beta_hat[2 : 2 + k]
        beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
        assert np.allclose(beta_d_hat, PF_BETA_D_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowSeparablePanel beta_d: expected ≈{PF_BETA_D_TRUE}, "
            f"got {beta_d_hat}"
        )
        assert np.allclose(beta_o_hat, PF_BETA_O_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowSeparablePanel beta_o: expected ≈{PF_BETA_O_TRUE}, "
            f"got {beta_o_hat}"
        )


@pytest.mark.slow
class TestPoissonFlowPanelRecovery:
    """PoissonSARFlowPanel posterior means should be close to true DGP values."""

    # ADVI-based fit recovers all three rho's well within the shared
    # NUTS tolerance, so use a tighter class-local bound here.
    ABS_TOL_RHO_POI_ADVI = 0.27

    @pytest.fixture(scope="class")
    def fitted_poisson_panel(self):
        from bayespecon.dgp.flows import generate_panel_poisson_flow_data

        G = _flow_test_graph(PP_N)
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
        # NUTS is prohibitively expensive for the 3-parameter SAR Poisson flow
        # panel. Use mean-field ADVI: fullrank ADVI is numerically unstable for
        # this model (the Cholesky factor of the joint covariance over the
        # Dirichlet stick-breaking unconstrained reals can drive rho proposals
        # outside the stationary region, making the SAR matrix singular).
        # Posterior means are robust to the mean-field factorisation here
        # because the rho's are well-identified individually.
        idata = model.fit_approx(
            method="advi",
            n=20000,
            draws=2000,
            random_seed=42,
            progressbar=False,
        )
        return idata

    def test_poisson_panel_recovers_rho_d(self, fitted_poisson_panel):
        rho_hat = float(fitted_poisson_panel.posterior["rho_d"].mean())
        assert abs(rho_hat - PP_RHO_D_TRUE) < self.ABS_TOL_RHO_POI_ADVI, (
            f"PoissonSARFlowPanel rho_d: expected ≈{PP_RHO_D_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_panel_recovers_rho_o(self, fitted_poisson_panel):
        rho_hat = float(fitted_poisson_panel.posterior["rho_o"].mean())
        assert abs(rho_hat - PP_RHO_O_TRUE) < self.ABS_TOL_RHO_POI_ADVI, (
            f"PoissonSARFlowPanel rho_o: expected ≈{PP_RHO_O_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_panel_recovers_rho_w(self, fitted_poisson_panel):
        rho_hat = float(fitted_poisson_panel.posterior["rho_w"].mean())
        assert abs(rho_hat - PP_RHO_W_TRUE) < self.ABS_TOL_RHO_POI_ADVI, (
            f"PoissonSARFlowPanel rho_w: expected ≈{PP_RHO_W_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_panel_recovers_beta(self, fitted_poisson_panel):
        # Poisson SAR DGP uses default beta_d = beta_o = ones(k) with k=2;
        # design layout [alpha, intra, beta_d, beta_o, ..., gamma_dist].
        k = 2
        beta_true = np.ones(k)
        beta_hat = (
            fitted_poisson_panel.posterior["beta"].mean(dim=("chain", "draw")).values
        )
        beta_d_hat = beta_hat[2 : 2 + k]
        beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
        assert np.allclose(beta_d_hat, beta_true, atol=ABS_TOL_BETA_POI), (
            f"PoissonSARFlowPanel beta_d: expected ≈{beta_true}, got {beta_d_hat}"
        )
        assert np.allclose(beta_o_hat, beta_true, atol=ABS_TOL_BETA_POI), (
            f"PoissonSARFlowPanel beta_o: expected ≈{beta_true}, got {beta_o_hat}"
        )


@pytest.mark.slow
class TestPoissonFlowSeparablePanelRecovery:
    """PoissonSARFlowSeparablePanel posterior means should be close to true DGP values."""

    # Larger panel than the shared PP_N_SEP/PANEL_FLOW_T_POISSON_SEP defaults:
    # the bilinear rho_w = -rho_d * rho_o constraint is weakly identified at
    # the smaller smoke-test sizes used elsewhere in this module.
    PP_N_SEP_REC = 20
    PANEL_T_SEP_REC = 5

    @pytest.fixture(scope="class")
    def fitted_poisson_separable_panel(self):
        from bayespecon.dgp.flows import generate_panel_poisson_flow_data_separable

        G = _flow_test_graph(self.PP_N_SEP_REC)
        out = generate_panel_poisson_flow_data_separable(
            n=self.PP_N_SEP_REC,
            T=self.PANEL_T_SEP_REC,
            G=G,
            rho_d=PP_RHO_D_SEP_TRUE,
            rho_o=PP_RHO_O_SEP_TRUE,
            seed=42,
        )
        model = PoissonSARFlowSeparablePanel(
            y=out["y"],
            G=G,
            X=out["X"],
            T=self.PANEL_T_SEP_REC,
            col_names=out["col_names"],
            model=0,
            trace_seed=0,
        )
        # Same rationale as TestPoissonFlowPanelRecovery: NUTS on this Poisson
        # SAR likelihood is prohibitively slow (~25+ min) and convergence is
        # unreliable (max-treedepth, low ESS). Mean-field ADVI recovers
        # rho_d, rho_o and beta within tolerance on the larger panel above.
        idata = model.fit_approx(
            method="advi",
            n=40000,
            draws=2000,
            random_seed=42,
            progressbar=False,
        )
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

    def test_poisson_separable_panel_recovers_beta(
        self, fitted_poisson_separable_panel
    ):
        # Default Poisson Separable DGP: beta_d = beta_o = ones(k) with k=2.
        k = 2
        beta_true = np.ones(k)
        beta_hat = (
            fitted_poisson_separable_panel.posterior["beta"]
            .mean(dim=("chain", "draw"))
            .values
        )
        beta_d_hat = beta_hat[2 : 2 + k]
        beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
        assert np.allclose(beta_d_hat, beta_true, atol=ABS_TOL_BETA_POI), (
            f"PoissonSARFlowSeparablePanel beta_d: expected ≈{beta_true}, "
            f"got {beta_d_hat}"
        )
        assert np.allclose(beta_o_hat, beta_true, atol=ABS_TOL_BETA_POI), (
            f"PoissonSARFlowSeparablePanel beta_o: expected ≈{beta_true}, "
            f"got {beta_o_hat}"
        )


def _ols_panel_flow_data(n: int, T: int, beta_d, beta_o, sigma, seed=0):
    """Stack T panel-flow periods with rho = 0 via bayespecon's panel DGP."""
    from bayespecon.dgp.flows import generate_panel_flow_data

    out = generate_panel_flow_data(
        n=n,
        T=T,
        k=len(beta_d),
        rho_d=0.0,
        rho_o=0.0,
        rho_w=0.0,
        beta_d=beta_d,
        beta_o=beta_o,
        sigma=sigma,
        sigma_alpha=0.0,
        seed=seed,
    )
    return out["G"], out["y"], out["X"], out["col_names"]


class TestOLSFlowPanelEffects:
    """Tests for the new non-spatial OLSFlowPanel gravity baseline."""

    def test_olsflow_panel_builds_and_skips_logdet_precompute(self):
        n = 4
        T = 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=20)

        model = OLSFlowPanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
        )
        # Trace / separable precomputation must be skipped.
        assert model._traces is None
        assert model._separable_logdet_fn is None
        assert model._n == n
        assert model._T == T

    def test_olsflow_panel_posterior_predictive_shape(self):
        n = 4
        T = 2
        G, y, X, col_names = _ols_panel_flow_data(
            n=n,
            T=T,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=1,
        )
        model = OLSFlowPanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        y_rep = model.posterior_predictive(n_draws=5, random_seed=3)
        assert y_rep.shape == (5, n * n * T)
        assert np.all(np.isfinite(y_rep))

    @pytest.mark.parametrize("fe_mode", [1, 2])
    def test_olsflow_panel_fe_modes_smoke(self, fe_mode):
        """Pair FE (1) and time FE (2) wire through `_demean_panel` correctly."""
        n = 4
        T = 3
        G, y, X, col_names = _ols_panel_flow_data(
            n=n,
            T=T,
            beta_d=[0.5],
            beta_o=[0.3],
            sigma=0.2,
            seed=fe_mode,
        )
        model = OLSFlowPanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=fe_mode,
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        post = model._idata.posterior
        assert "beta" in post.data_vars
        assert "sigma" in post.data_vars
        assert "rho_d" not in post.data_vars
        assert "rho_o" not in post.data_vars


class TestPoissonGravityFlowPanel:
    """Tests for the new aspatial Poisson gravity panel baseline (PoissonFlowPanel)."""

    def _make_panel_count_data(self, n=4, T=2, seed=0):
        from bayespecon.dgp.flows import generate_panel_poisson_flow_data

        G = _flow_test_graph(n)
        data = generate_panel_poisson_flow_data(
            n=n,
            T=T,
            G=G,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[0.3],
            beta_o=[0.2],
            k=1,
            seed=seed,
        )
        return G, data

    def test_constructs_and_skips_logdet_precompute(self):
        G, data = self._make_panel_count_data(n=4, T=2)
        model = PoissonFlowPanel(
            y=data["y"],
            G=G,
            X=data["X"],
            T=2,
            col_names=data["col_names"],
            model=0,
        )
        assert model._traces is None
        assert model._separable_logdet_fn is None
        assert model._y_int_vec.dtype == np.int64

    def test_rejects_fe_modes(self):
        G, data = self._make_panel_count_data(n=4, T=2)
        with pytest.raises(ValueError, match="model=0 only"):
            PoissonFlowPanel(
                y=data["y"],
                G=G,
                X=data["X"],
                T=2,
                col_names=data["col_names"],
                model=1,
            )

    def test_rejects_negative_and_non_integer(self):
        G, data = self._make_panel_count_data(n=4, T=2)
        y_bad = data["y"].astype(np.float64) + 0.5
        with pytest.raises(ValueError, match="integer-valued"):
            PoissonFlowPanel(
                y=y_bad,
                G=G,
                X=data["X"],
                T=2,
                col_names=data["col_names"],
                model=0,
            )
        y_neg = data["y"].copy()
        y_neg[0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            PoissonFlowPanel(
                y=y_neg,
                G=G,
                X=data["X"],
                T=2,
                col_names=data["col_names"],
                model=0,
            )

    def test_pymc_model_has_no_spatial_or_sigma(self):
        G, data = self._make_panel_count_data(n=4, T=2)
        model = PoissonFlowPanel(
            y=data["y"],
            G=G,
            X=data["X"],
            T=2,
            col_names=data["col_names"],
            model=0,
        )
        pm_model = model._build_pymc_model()
        names = {v.name for v in pm_model.unobserved_RVs}
        assert "beta" in names
        assert "sigma" not in names
        assert not any(n.startswith("rho_") for n in names)

    def test_posterior_predictive_shape_and_integer(self):
        G, data = self._make_panel_count_data(n=4, T=2, seed=1)
        model = PoissonFlowPanel(
            y=data["y"],
            G=G,
            X=data["X"],
            T=2,
            col_names=data["col_names"],
            model=0,
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        y_rep = model.posterior_predictive(n_draws=5, random_seed=3)
        assert y_rep.shape == (5, 4 * 4 * 2)
        assert np.all(y_rep >= 0)
        assert np.allclose(y_rep, np.round(y_rep))

    def test_spatial_effects_network_zero(self):
        G, data = self._make_panel_count_data(n=5, T=2, seed=2)
        model = PoissonFlowPanel(
            y=data["y"],
            G=G,
            X=data["X"],
            T=2,
            col_names=data["col_names"],
            model=0,
        )
        model.fit(draws=40, tune=40, chains=1, progressbar=False, random_seed=0)
        df = model.spatial_effects(mode="combined")
        assert np.isclose(df.xs("network", level="effect")["mean"].iloc[0], 0.0)


@pytest.mark.slow
class TestPoissonGravityFlowPanelRecovery:
    """PoissonFlowPanel β posterior should recover DGP coefficients under rho=0."""

    def test_recovers_beta(self):
        from bayespecon.dgp.flows import generate_panel_poisson_flow_data

        n, T = 8, 3
        G = _flow_test_graph(n)
        beta_d = np.array([0.5, -0.3])
        beta_o = np.array([0.4, 0.2])
        data = generate_panel_poisson_flow_data(
            n=n,
            T=T,
            G=G,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=beta_d,
            beta_o=beta_o,
            k=2,
            seed=42,
        )
        model = PoissonFlowPanel(
            y=data["y"],
            G=G,
            X=data["X"],
            T=T,
            col_names=data["col_names"],
            model=0,
        )
        idata = model.fit(
            tune=400,
            draws=600,
            chains=2,
            random_seed=42,
            progressbar=False,
        )
        beta_hat = idata.posterior["beta"].mean(dim=("chain", "draw")).values
        bd_hat = beta_hat[2 : 2 + len(beta_d)]
        bo_hat = beta_hat[2 + len(beta_d) : 2 + 2 * len(beta_d)]
        assert np.allclose(bd_hat, beta_d, atol=0.40), (
            f"βd: expected {beta_d}, got {bd_hat}"
        )
        assert np.allclose(bo_hat, beta_o, atol=0.40), (
            f"βo: expected {beta_o}, got {bo_hat}"
        )


# ---------------------------------------------------------------------------
# Pointwise log-likelihood (with Jacobian correction)
# ---------------------------------------------------------------------------


class TestFlowPanelLogLikelihood:
    """Verify ``log_likelihood`` group is attached for panel flow fits and
    usable for ``az.loo`` / ``az.compare``."""

    def _check_loo(self, idata, n_obs):
        import arviz as az

        assert hasattr(idata, "log_likelihood")
        assert "obs" in idata.log_likelihood.data_vars
        ll = idata.log_likelihood["obs"].values
        assert ll.shape[2] == n_obs
        assert np.isfinite(ll).all()
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd_loo)

    def test_sar_flow_panel_loglik(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        m = SARFlowPanel(
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
        idata = m.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        self._check_loo(idata, n_obs=n * n * T)

    def test_sar_flow_separable_panel_loglik(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        m = SARFlowSeparablePanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            trace_seed=0,
        )
        idata = m.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        self._check_loo(idata, n_obs=n * n * T)

    def test_ols_flow_panel_loglik(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        m = OLSFlowPanel(y=y, G=G, X=X, T=T, col_names=col_names, model=0)
        idata = m.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        self._check_loo(idata, n_obs=n * n * T)

    def test_poisson_flow_panel_loglik(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        _, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        y_int = _panel_count_vector(n=n, T=T, seed=1)
        m = PoissonFlowPanel(
            y=y_int,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
        )
        idata = m.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        self._check_loo(idata, n_obs=n * n * T)

    def test_compare_flow_panel_models(self):
        import arviz as az

        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        kw = dict(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        m_sar = SARFlowPanel(
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
        m_sep = SARFlowSeparablePanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            model=0,
            trace_seed=0,
        )
        m_ols = OLSFlowPanel(y=y, G=G, X=X, T=T, col_names=col_names, model=0)
        idata_sar = m_sar.fit(**kw)
        idata_sep = m_sep.fit(**kw)
        idata_ols = m_ols.fit(**kw)
        comp = az.compare({"sar": idata_sar, "sep": idata_sep, "ols": idata_ols})
        assert "rank" in comp.columns
        assert len(comp) == 3
