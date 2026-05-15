"""Tests for SEMFlow / SEMFlowSeparable cross-sectional flow models.

Smoke tests use small n=5 grids and minimal draws to keep CI runtime tight.
A single recovery test uses n=8 and moderate draws to confirm posteriors
concentrate near the true values.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


class TestGenerateSemFlowData:
    def test_output_keys_and_shapes(self):
        from bayespecon.dgp.flows import generate_sem_flow_data

        n = 5
        out = generate_sem_flow_data(
            n=n,
            lam_d=0.2,
            lam_o=0.1,
            lam_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=0,
            distribution="normal",
        )
        N = n * n
        assert out["y_vec"].shape == (N,)
        assert out["y_mat"].shape == (n, n)
        assert out["eta_vec"].shape == (N,)
        assert out["X"].shape[0] == N
        assert out["distribution"] == "normal"
        assert out["lam_d"] == 0.2
        assert out["lam_w"] == 0.05

    def test_separable_substitutes_rho_w(self):
        from bayespecon.dgp.flows import (
            generate_sem_flow_data,
            generate_sem_flow_data_separable,
        )

        kwargs = dict(
            n=5,
            lam_d=0.3,
            lam_o=0.2,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=1,
            distribution="normal",
        )
        sep = generate_sem_flow_data_separable(**kwargs)
        full = generate_sem_flow_data(lam_w=-0.3 * 0.2, **kwargs)
        np.testing.assert_allclose(sep["y_vec"], full["y_vec"])
        assert sep["lam_w"] == pytest.approx(-0.3 * 0.2)

    def test_unstable_rho_warns(self):
        from bayespecon.dgp.flows import generate_sem_flow_data

        with pytest.warns(UserWarning, match="stability"):
            generate_sem_flow_data(
                n=5,
                lam_d=0.6,
                lam_o=0.5,
                lam_w=0.0,
                beta_d=[1.0],
                beta_o=[0.5],
                sigma=0.5,
                seed=0,
                distribution="normal",
            )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSemFlowConstruction:
    def setup_method(self):
        from bayespecon.dgp.flows import generate_sem_flow_data

        self.n = 5
        self.N = self.n * self.n
        out = generate_sem_flow_data(
            n=self.n,
            lam_d=0.0,
            lam_o=0.0,
            lam_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=10,
            distribution="normal",
        )
        self.G = out["G"]
        self.X = out["X"]
        self.col_names = out["col_names"]
        self.y_vec = out["y_vec"]

    def test_sem_flow_builds(self):
        from bayespecon.models.flow import SEMFlow

        model = SEMFlow(
            self.y_vec,
            self.G,
            self.X,
            col_names=self.col_names,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        assert model._n == self.n
        assert model._N == self.N
        # Lagged X precomputed
        assert model._Wd_X.shape == self.X.shape
        assert model._Wo_X.shape == self.X.shape
        assert model._Ww_X.shape == self.X.shape

    def test_sem_flow_separable_builds(self):
        from bayespecon.models.flow import SEMFlowSeparable

        model = SEMFlowSeparable(
            self.y_vec,
            self.G,
            self.X,
            col_names=self.col_names,
            trace_seed=0,
        )
        assert model._n == self.n

    def test_pymc_model_builds(self):
        from bayespecon.models.flow import SEMFlow

        model = SEMFlow(
            self.y_vec,
            self.G,
            self.X,
            col_names=self.col_names,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        assert pm_model is not None

    def test_pymc_model_separable_builds(self):
        from bayespecon.models.flow import SEMFlowSeparable

        model = SEMFlowSeparable(
            self.y_vec,
            self.G,
            self.X,
            col_names=self.col_names,
            trace_seed=0,
        )
        pm_model = model._build_pymc_model()
        assert pm_model is not None


# ---------------------------------------------------------------------------
# Recovery (slow but small enough for default suite)
# ---------------------------------------------------------------------------


class TestSemFlowRecovery:
    def test_parameter_recovery_normal(self):
        from bayespecon.dgp.flows import generate_sem_flow_data
        from bayespecon.models.flow import SEMFlow

        rho_d_true, rho_o_true, rho_w_true = 0.25, 0.20, 0.10
        beta_d_true = [1.0, -0.5]
        beta_o_true = [0.8, 0.3]
        sigma_true = 0.6

        data = generate_sem_flow_data(
            n=20,
            lam_d=rho_d_true,
            lam_o=rho_o_true,
            lam_w=rho_w_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            sigma=sigma_true,
            gamma_dist=-0.4,
            seed=11,
            distribution="normal",
        )

        model = SEMFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            miter=20,
            trace_riter=30,
            trace_seed=0,
        )
        idata = model.fit(
            draws=400,
            tune=400,
            chains=2,
            target_accept=0.9,
            random_seed=7,
            progressbar=False,
        )
        post = idata.posterior

        # rho parameters within ~3*sd of truth (Monte Carlo + finite n tolerance)
        for name, true in [
            ("lam_d", rho_d_true),
            ("lam_o", rho_o_true),
            ("lam_w", rho_w_true),
        ]:
            samples = post[name].values.ravel()
            mean = samples.mean()
            sd = samples.std()
            assert abs(mean - true) < 4 * sd, (
                f"{name}: mean={mean:.3f}, true={true}, sd={sd:.3f}"
            )

        # sigma close to truth
        sigma_mean = post["sigma"].values.ravel().mean()
        assert abs(sigma_mean - sigma_true) < 0.1

        # beta coefficients (positions 2,3 = beta_d, 4,5 = beta_o in design)
        beta_post = post["beta"].values.reshape(-1, post["beta"].shape[-1]).mean(axis=0)
        np.testing.assert_allclose(beta_post[2:4], beta_d_true, atol=0.15)
        np.testing.assert_allclose(beta_post[4:6], beta_o_true, atol=0.15)

    def test_separable_constraint_holds(self):
        """SEMFlowSeparable should impose lam_w = -lam_d * lam_o exactly."""
        from bayespecon.dgp.flows import generate_sem_flow_data_separable
        from bayespecon.models.flow import SEMFlowSeparable

        data = generate_sem_flow_data_separable(
            n=10,
            lam_d=0.2,
            lam_o=0.15,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=3,
            distribution="normal",
        )
        model = SEMFlowSeparable(
            data["y_vec"],
            data["G"],
            data["X"],
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
