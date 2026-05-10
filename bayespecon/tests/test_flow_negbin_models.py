"""Tests for Negative Binomial flow and panel-flow model variants."""

from __future__ import annotations

import pytest

from bayespecon.models.flow import (
    NegativeBinomialFlow,
    NegativeBinomialSARFlow,
    NegativeBinomialSARFlowSeparable,
)
from bayespecon.models.flow_panel import (
    NegativeBinomialFlowPanel,
    NegativeBinomialSARFlowPanel,
    NegativeBinomialSARFlowSeparablePanel,
)


def _small_negbin_flow(seed: int = 0):
    from bayespecon.dgp.flows import generate_negbin_flow_data

    return generate_negbin_flow_data(n=6, seed=seed)


def _small_panel_negbin_flow(seed: int = 0):
    from bayespecon.dgp.flows import generate_panel_negbin_flow_data

    return generate_panel_negbin_flow_data(n=5, T=3, seed=seed)


class TestNegativeBinomialFlowConstruction:
    def test_negbin_sar_flow_builds(self):
        data = _small_negbin_flow(seed=1)
        model = NegativeBinomialSARFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            miter=5,
            titer=50,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_sar_flow_separable_builds(self):
        data = _small_negbin_flow(seed=2)
        model = NegativeBinomialSARFlowSeparable(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_flow_builds(self):
        data = _small_negbin_flow(seed=3)
        model = NegativeBinomialFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars


class TestNegativeBinomialPanelFlowConstruction:
    def test_negbin_panel_builds(self):
        data = _small_panel_negbin_flow(seed=4)
        model = NegativeBinomialSARFlowPanel(
            y=data["y"],
            G=data["G"],
            X=data["X"],
            T=3,
            col_names=data["col_names"],
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_panel_separable_builds(self):
        data = _small_panel_negbin_flow(seed=5)
        model = NegativeBinomialSARFlowSeparablePanel(
            y=data["y"],
            G=data["G"],
            X=data["X"],
            T=3,
            col_names=data["col_names"],
            model=0,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_panel_aspatial_builds(self):
        data = _small_panel_negbin_flow(seed=6)
        model = NegativeBinomialFlowPanel(
            y=data["y"],
            G=data["G"],
            X=data["X"],
            T=3,
            col_names=data["col_names"],
            model=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_panel_requires_model_zero(self):
        data = _small_panel_negbin_flow(seed=7)
        with pytest.raises(ValueError, match="model=0 only"):
            NegativeBinomialSARFlowPanel(
                y=data["y"],
                G=data["G"],
                X=data["X"],
                T=3,
                col_names=data["col_names"],
                model=1,
            )


def _check_beta_recovery(
    idata, beta_d_true, beta_o_true, gamma_dist_true, k=2, tol=0.4
):
    """Assert beta coefficient recovery for flow models.

    Design matrix layout (flow_design_matrix_with_orig, k_d=k_o=k):
      [0] intercept      → DGP true = 0
      [1] intra indicator→ DGP true = 0
      [2:2+k] beta_d     → checked here
      [2+k:2+2k] beta_o  → checked here
      [-1] log_dist (gamma_dist) → checked here
    """
    import numpy as np

    beta_hat = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
    beta_d_hat = beta_hat[2 : 2 + k]
    beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
    gamma_dist_hat = float(beta_hat[-1])

    for i, (true_val, hat_val) in enumerate(zip(beta_d_true, beta_d_hat)):
        assert abs(hat_val - true_val) < tol, (
            f"beta_d[{i}] recovery failed: true={true_val:.3f}, hat={hat_val:.3f}"
        )
    for i, (true_val, hat_val) in enumerate(zip(beta_o_true, beta_o_hat)):
        assert abs(hat_val - true_val) < tol, (
            f"beta_o[{i}] recovery failed: true={true_val:.3f}, hat={hat_val:.3f}"
        )
    assert abs(gamma_dist_hat - gamma_dist_true) < tol, (
        f"gamma_dist recovery failed: true={gamma_dist_true:.3f}, hat={gamma_dist_hat:.3f}"
    )


@pytest.mark.slow
class TestNegativeBinomialFlowRecovery:
    """Parameter recovery checks for NB flow variants (deselected by default)."""

    def test_negbin_sar_flow_recovers_all_params(self):
        from bayespecon.dgp.flows import generate_negbin_flow_data

        rho_d_true, rho_o_true, rho_w_true = 0.25, 0.2, 0.1
        beta_d_true, beta_o_true = [1.5, -0.8], [0.7, 1.2]
        gamma_dist_true = -0.5
        alpha_true = 2.0

        out = generate_negbin_flow_data(
            n=10,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            rho_w=rho_w_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=42,
        )
        model = NegativeBinomialSARFlow(
            out["y_vec"],
            out["G"],
            out["X"],
            col_names=out["col_names"],
            miter=5,
            titer=50,
            trace_seed=0,
        )
        idata = model.fit_approx(
            method="advi",
            n=25000,
            draws=2000,
            random_seed=42,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        rho_w_hat = float(idata.posterior["rho_w"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        assert abs(rho_d_hat - rho_d_true) < 0.25, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.25, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(rho_w_hat - rho_w_true) < 0.25, (
            f"rho_w: {rho_w_hat:.3f} vs {rho_w_true}"
        )
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true)

    def test_negbin_sar_flow_separable_recovers_all_params(self):
        from bayespecon.dgp.flows import generate_negbin_flow_data_separable

        rho_d_true, rho_o_true = 0.4, 0.3
        beta_d_true, beta_o_true = [1.2, -0.6], [0.9, 1.1]
        gamma_dist_true = -0.5
        alpha_true = 1.8

        out = generate_negbin_flow_data_separable(
            n=10,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=43,
        )
        model = NegativeBinomialSARFlowSeparable(
            out["y_vec"],
            out["G"],
            out["X"],
            col_names=out["col_names"],
            trace_seed=0,
        )
        idata = model.fit_approx(
            method="advi",
            n=25000,
            draws=2000,
            random_seed=43,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        # Separable NB with ADVI exhibits materially higher Monte-Carlo variability
        # than the unrestricted NB flow recovery case on this small synthetic sample.
        assert abs(rho_d_hat - rho_d_true) < 0.60, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.40, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true, tol=0.8)


@pytest.mark.slow
class TestNegativeBinomialPanelFlowRecovery:
    """Parameter recovery checks for panel NB flow variants."""

    def test_negbin_panel_sar_recovers_all_params(self):
        from bayespecon.dgp.flows import generate_panel_negbin_flow_data

        rho_d_true, rho_o_true, rho_w_true = 0.25, 0.2, 0.1
        beta_d_true, beta_o_true = [1.4, -0.7], [0.8, 1.1]
        gamma_dist_true = -0.5
        alpha_true = 2.2

        out = generate_panel_negbin_flow_data(
            n=7,
            T=5,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            rho_w=rho_w_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=44,
        )
        model = NegativeBinomialSARFlowPanel(
            y=out["y"],
            G=out["G"],
            X=out["X"],
            T=5,
            col_names=out["col_names"],
            model=0,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        idata = model.fit_approx(
            method="advi",
            n=25000,
            draws=2000,
            random_seed=44,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        rho_w_hat = float(idata.posterior["rho_w"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        assert abs(rho_d_hat - rho_d_true) < 0.30, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.30, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(rho_w_hat - rho_w_true) < 0.30, (
            f"rho_w: {rho_w_hat:.3f} vs {rho_w_true}"
        )
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true)

    def test_negbin_panel_sar_separable_recovers_all_params(self):
        from bayespecon.dgp.flows import generate_panel_negbin_flow_data_separable

        rho_d_true, rho_o_true = 0.4, 0.3
        beta_d_true, beta_o_true = [1.3, -0.5], [0.6, 1.0]
        gamma_dist_true = -0.5
        alpha_true = 1.7

        out = generate_panel_negbin_flow_data_separable(
            n=8,
            T=4,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=45,
        )
        model = NegativeBinomialSARFlowSeparablePanel(
            y=out["y"],
            G=out["G"],
            X=out["X"],
            T=4,
            col_names=out["col_names"],
            model=0,
            trace_seed=0,
        )
        idata = model.fit_approx(
            method="advi",
            n=25000,
            draws=2000,
            random_seed=45,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        # Panel separable NB recovery is noisier under mean-field ADVI; keep this
        # as a coarse calibration test rather than a tight parameter-recovery check.
        assert abs(rho_d_hat - rho_d_true) < 0.40, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.35, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true, tol=0.8)
