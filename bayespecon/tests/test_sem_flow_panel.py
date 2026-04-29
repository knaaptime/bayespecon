"""Tests for SEMFlowPanel / SEMFlowSeparablePanel models."""

from __future__ import annotations

import numpy as np
import pytest


class TestGeneratePanelSemFlowData:
    def test_output_shapes(self):
        from bayespecon.dgp.flows import generate_panel_sem_flow_data

        n, T = 5, 3
        out = generate_panel_sem_flow_data(
            n=n,
            T=T,
            lam_d=0.2,
            lam_o=0.15,
            lam_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            sigma_alpha=0.3,
            seed=0,
            distribution="normal",
        )
        N = n * n
        assert out["y"].shape == (N * T,)
        assert out["X"].shape[0] == N * T
        assert out["distribution"] == "normal"
        assert "params_true" in out

    def test_separable_substitutes_rho_w(self):
        from bayespecon.dgp.flows import (
            generate_panel_sem_flow_data,
            generate_panel_sem_flow_data_separable,
        )

        kwargs = dict(
            n=5,
            T=3,
            lam_d=0.3,
            lam_o=0.2,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            sigma_alpha=0.0,
            seed=1,
            distribution="normal",
        )
        sep = generate_panel_sem_flow_data_separable(**kwargs)
        full = generate_panel_sem_flow_data(lam_w=-0.3 * 0.2, **kwargs)
        np.testing.assert_allclose(sep["y"], full["y"])


class TestSemFlowPanelConstruction:
    def setup_method(self):
        from bayespecon.dgp.flows import generate_panel_sem_flow_data

        self.n, self.T = 5, 3
        self.data = generate_panel_sem_flow_data(
            n=self.n,
            T=self.T,
            lam_d=0.0,
            lam_o=0.0,
            lam_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            sigma_alpha=0.0,
            seed=2,
            distribution="normal",
        )

    def test_sem_flow_panel_builds(self):
        from bayespecon.models.flow_panel import SEMFlowPanel

        model = SEMFlowPanel(
            self.data["y"],
            self.data["G"],
            self.data["X"],
            T=self.T,
            col_names=self.data["col_names"],
            miter=5,
            trace_seed=0,
        )
        assert model._n == self.n
        assert model._T == self.T

    def test_sem_flow_separable_panel_builds(self):
        from bayespecon.models.flow_panel import SEMFlowSeparablePanel

        model = SEMFlowSeparablePanel(
            self.data["y"],
            self.data["G"],
            self.data["X"],
            T=self.T,
            col_names=self.data["col_names"],
            trace_seed=0,
        )
        assert model._n == self.n

    def test_pymc_model_builds(self):
        from bayespecon.models.flow_panel import SEMFlowPanel

        model = SEMFlowPanel(
            self.data["y"],
            self.data["G"],
            self.data["X"],
            T=self.T,
            col_names=self.data["col_names"],
            miter=5,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        assert pm_model is not None


class TestSemFlowPanelRecovery:
    def test_rho_recovery(self):
        """Panel SEM should recover lam_d, lam_o reasonably (lam_w is noisier)."""
        from bayespecon.dgp.flows import generate_panel_sem_flow_data
        from bayespecon.models.flow_panel import SEMFlowPanel

        rho_d_true, rho_o_true, rho_w_true = 0.25, 0.20, 0.10
        data = generate_panel_sem_flow_data(
            n=15,
            T=4,
            lam_d=rho_d_true,
            lam_o=rho_o_true,
            lam_w=rho_w_true,
            beta_d=[1.0, -0.5],
            beta_o=[0.8, 0.3],
            sigma=0.6,
            sigma_alpha=0.3,
            seed=11,
            distribution="normal",
        )
        model = SEMFlowPanel(
            data["y"],
            data["G"],
            data["X"],
            T=4,
            col_names=data["col_names"],
            miter=15,
            trace_riter=30,
            trace_seed=0,
        )
        idata = model.fit(
            draws=300,
            tune=300,
            chains=2,
            target_accept=0.9,
            random_seed=7,
            progressbar=False,
        )
        post = idata.posterior
        for name, true in [("lam_d", rho_d_true), ("lam_o", rho_o_true)]:
            samples = post[name].values.ravel()
            mean, sd = samples.mean(), samples.std()
            assert abs(mean - true) < 4 * sd, (
                f"{name}: mean={mean:.3f}, true={true}, sd={sd:.3f}"
            )

    def test_separable_constraint_panel(self):
        from bayespecon.dgp.flows import generate_panel_sem_flow_data_separable
        from bayespecon.models.flow_panel import SEMFlowSeparablePanel

        data = generate_panel_sem_flow_data_separable(
            n=8,
            T=3,
            lam_d=0.2,
            lam_o=0.15,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            sigma_alpha=0.0,
            seed=3,
            distribution="normal",
        )
        model = SEMFlowSeparablePanel(
            data["y"],
            data["G"],
            data["X"],
            T=3,
            col_names=data["col_names"],
            miter=10,
            trace_seed=0,
        )
        idata = model.fit(
            draws=200,
            tune=200,
            chains=1,
            target_accept=0.9,
            random_seed=0,
            progressbar=False,
        )
        post = idata.posterior
        rd = post["lam_d"].values.ravel()
        ro = post["lam_o"].values.ravel()
        rw = post["lam_w"].values.ravel()
        np.testing.assert_allclose(rw, -rd * ro, atol=1e-10)
