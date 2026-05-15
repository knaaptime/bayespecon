"""Tests for bayespecon.models.flow and supporting logdet flow functions.

Smoke tests use small n=5 grids and minimal draws (100, chains=1) to keep
CI runtime reasonable.  Recovery tests use n=8.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon.tests.helpers import SAMPLE_KWARGS

# ---------------------------------------------------------------------------
# logdet flow helper functions
# ---------------------------------------------------------------------------


class TestBarryPaceTraces:
    def test_shape(self):
        from bayespecon.logdet import _barry_pace_traces

        n = 10
        W = sp.random(n, n, density=0.3, format="csr", random_state=0, dtype=np.float64)
        # Row-normalise
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        rng = np.random.default_rng(0)
        out = _barry_pace_traces(W, order=5, iter=10, rng=rng)
        assert out.shape == (5, 10)

    def test_exact_tr_W_override(self):
        """tr(W) = 0 for zero-diagonal W must be set in row 0."""
        from bayespecon.logdet import _barry_pace_traces

        n = 8
        W = sp.random(n, n, density=0.4, format="csr", random_state=1, dtype=np.float64)
        W.setdiag(0)
        W.eliminate_zeros()
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        rng = np.random.default_rng(2)
        out = _barry_pace_traces(W, order=4, iter=20, rng=rng)
        # All rows 0 should be the exact tr(W) = 0
        np.testing.assert_allclose(out[0, :], 0.0, atol=1e-12)


class TestComputeFlowTraces:
    def test_shape(self):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.logdet import compute_flow_traces

        n = 6
        G = generate_flow_data(n=n, seed=0)["G"]
        W = G.sparse.tocsr().astype(np.float64)
        traces = compute_flow_traces(W, miter=5, riter=20, random_state=0)
        assert traces.shape == (5,)

    def test_tr_W2_close_to_exact(self):
        """tr(W^2) estimate should be within 20% of exact for ring graph."""
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.logdet import compute_flow_traces

        n = 10
        G = generate_flow_data(n=n, seed=0)["G"]
        W = G.sparse.tocsr().astype(np.float64)
        exact_tr_W2 = float(W.multiply(W.T).sum())
        traces = compute_flow_traces(W, miter=5, riter=200, random_state=42)
        # traces[1] = tr(W^2) (exact override from _barry_pace_traces)
        np.testing.assert_allclose(traces[1], exact_tr_W2, atol=1e-10)


class TestFlowLogdetPolyCoeffs:
    def test_output_tuple_length(self):
        from bayespecon.logdet import _flow_logdet_poly_coeffs

        n = 5
        miter = 3
        traces = np.zeros(miter)
        result = _flow_logdet_poly_coeffs(traces, n, miter)
        assert len(result) == 8

    def test_array_shapes_consistent(self):
        from bayespecon.logdet import _flow_logdet_poly_coeffs

        n = 5
        miter = 4
        traces = np.array([0.0, 2.0, 0.0, 0.1])
        poly_a, poly_b, poly_c, poly_coeffs, m_a, m_b, m_c, m_coeffs = (
            _flow_logdet_poly_coeffs(traces, n, miter)
        )
        # All poly arrays same length
        assert len(poly_a) == len(poly_b) == len(poly_c) == len(poly_coeffs)
        # miter arrays same length
        assert len(m_a) == len(m_b) == len(m_c) == len(m_coeffs)
        # miter triples: (miter+1)(miter+2)//2 combinations
        expected_miter = (miter + 1) * (miter + 2) // 2
        assert len(m_a) == expected_miter

    def test_k1_terms_zero_for_zero_diagonal_W(self):
        """For zero-diagonal W (tr(W)=0), all k=1 poly_coeffs should be 0."""
        from bayespecon.logdet import _flow_logdet_poly_coeffs

        n = 6
        miter = 5
        # traces[0] = tr(W) = 0 for zero-diagonal W
        traces = np.array([0.0, 3.0, 0.0, 0.5, 0.1])
        poly_a, poly_b, poly_c, poly_coeffs, *_ = _flow_logdet_poly_coeffs(
            traces, n, miter
        )
        # k=1 triples: a+b+c=1 → (1,0,0), (0,1,0), (0,0,1)
        k1_mask = (poly_a + poly_b + poly_c) == 1
        np.testing.assert_allclose(poly_coeffs[k1_mask], 0.0, atol=1e-10)

    def test_raises_wrong_traces_length(self):
        from bayespecon.logdet import _flow_logdet_poly_coeffs

        with pytest.raises(ValueError, match=r"len\(traces\)"):
            _flow_logdet_poly_coeffs(np.zeros(4), n=5, miter=3)


class TestFlowLogdetSeparableIdentity:
    """Verify that flow_logdet_pytensor matches the exact log-det at a test point."""

    def test_zero_rho_gives_zero_logdet(self):
        """log|I| = 0, so at rho=0 the log-det should be ~0."""
        import pytensor.tensor as pt

        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.logdet import (
            _flow_logdet_poly_coeffs,
            compute_flow_traces,
            flow_logdet_pytensor,
        )

        n = 5
        G = generate_flow_data(n=n, seed=0)["G"]
        W = G.sparse.tocsr().astype(np.float64)
        miter = 10
        traces = compute_flow_traces(W, miter=miter, riter=50, random_state=7)
        result = _flow_logdet_poly_coeffs(traces, n, miter)
        poly_a, poly_b, poly_c, poly_coeffs = result[:4]
        miter_a, miter_b, miter_c, miter_coeffs = result[4:]

        rho_d_pt = pt.scalar("rho_d")
        rho_o_pt = pt.scalar("rho_o")
        rho_w_pt = pt.scalar("rho_w")

        expr = flow_logdet_pytensor(
            rho_d_pt,
            rho_o_pt,
            rho_w_pt,
            poly_a,
            poly_b,
            poly_c,
            poly_coeffs,
            miter_a,
            miter_b,
            miter_c,
            miter_coeffs,
            miter=miter,
            titer=800,
        )
        import pytensor

        fn = pytensor.function([rho_d_pt, rho_o_pt, rho_w_pt], expr)
        val = float(fn(0.0, 0.0, 0.0))
        np.testing.assert_allclose(val, 0.0, atol=1e-6)

    def test_separable_identity(self):
        """For rho_w = -rho_d*rho_o the flow log-det should equal
        n*logdet_eigenvalue(rho_d, eigs) + n*logdet_eigenvalue(rho_o, eigs)."""
        import pytensor
        import pytensor.tensor as pt

        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.logdet import (
            _flow_logdet_poly_coeffs,
            compute_flow_traces,
            flow_logdet_pytensor,
            logdet_eigenvalue,
        )

        n = 5
        G = generate_flow_data(n=n, seed=0)["G"]
        W = G.sparse.tocsr().astype(np.float64)
        eigs = np.linalg.eigvals(W.toarray()).real

        miter = 15
        traces = compute_flow_traces(W, miter=miter, riter=100, random_state=7)
        result = _flow_logdet_poly_coeffs(traces, n, miter)
        poly_a, poly_b, poly_c, poly_coeffs = result[:4]
        miter_a, miter_b, miter_c, miter_coeffs = result[4:]

        rd, ro = 0.3, 0.2
        rw = -rd * ro

        # Flow log-det via trace series
        rho_d_pt = pt.scalar("rho_d")
        rho_o_pt = pt.scalar("rho_o")
        rho_w_pt = pt.scalar("rho_w")
        expr = flow_logdet_pytensor(
            rho_d_pt,
            rho_o_pt,
            rho_w_pt,
            poly_a,
            poly_b,
            poly_c,
            poly_coeffs,
            miter_a,
            miter_b,
            miter_c,
            miter_coeffs,
            miter=miter,
            titer=800,
        )
        fn_flow = pytensor.function([rho_d_pt, rho_o_pt, rho_w_pt], expr)
        flow_val = float(fn_flow(rd, ro, rw))

        # Exact separable log-det via eigenvalues
        rho_scalar = pt.scalar("rho")
        fn_eig = pytensor.function([rho_scalar], logdet_eigenvalue(rho_scalar, eigs))
        separable_val = n * float(fn_eig(rd)) + n * float(fn_eig(ro))

        # Should match within Monte Carlo trace estimation error (~1% tolerance)
        np.testing.assert_allclose(flow_val, separable_val, rtol=0.05)


# ---------------------------------------------------------------------------
# FlowModel construction
# ---------------------------------------------------------------------------


class TestFlowModelConstruction:
    def setup_method(self):
        from bayespecon.dgp.flows import generate_flow_data

        self.n = 5
        self.N = self.n * self.n
        out = generate_flow_data(
            n=self.n,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=10,
        )
        self.G = out["G"]
        self.X_regional = out["X_regional"]
        self.design = out["design"]
        self.X = out["X"]
        self.col_names = out["col_names"]
        rng = np.random.default_rng(10)
        self.y_vec = rng.standard_normal(self.N)
        self.y_mat = self.y_vec.reshape(self.n, self.n)

    def test_sar_flow_builds_from_vec(self):
        from bayespecon.models.flow import SARFlow

        model = SARFlow(
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

    def test_sar_flow_builds_from_mat(self):
        from bayespecon.models.flow import SARFlow

        model = SARFlow(
            self.y_mat,
            self.G,
            self.X,
            col_names=self.col_names,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        np.testing.assert_allclose(model._y_vec, self.y_vec, atol=1e-12)

    def test_sar_flow_separable_builds(self):
        from bayespecon.models.flow import SARFlowSeparable

        model = SARFlowSeparable(
            self.y_vec, self.G, self.X, col_names=self.col_names, trace_seed=0
        )
        assert model._n == self.n

    def test_wrong_y_length_raises(self):
        from bayespecon.models.flow import SARFlow

        with pytest.raises(ValueError, match="N="):
            SARFlow(np.zeros(self.N + 1), self.G, self.X, miter=5, trace_seed=0)

    def test_asymmetric_beta_shapes_work_in_dgp(self):
        """generate_flow_data should support beta_d and beta_o of different lengths."""
        from bayespecon.dgp.flows import generate_flow_data

        data = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[1.0, 2.0],
            sigma=1.0,
            seed=0,
        )
        # Design matrix should have k_d=1 dest cols + k_o=2 orig cols + k_d=1 intra cols
        assert data["design"].k_d == 1
        assert data["design"].k_o == 2
        assert data["beta_d"].shape == (1,)
        assert data["beta_o"].shape == (2,)

    def test_pymc_model_builds_without_error(self):
        from bayespecon.models.flow import SARFlow

        model = SARFlow(
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

    def test_pymc_model_separable_builds_without_error(self):
        from bayespecon.models.flow import SARFlowSeparable

        model = SARFlowSeparable(
            self.y_vec, self.G, self.X, col_names=self.col_names, trace_seed=0
        )
        pm_model = model._build_pymc_model()
        assert pm_model is not None


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


class TestGenerateFlowData:
    def setup_method(self):
        self.n = 5

    def test_output_keys(self):
        from bayespecon.dgp.flows import generate_flow_data

        out = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=0,
        )
        expected = {
            "y_vec",
            "y_mat",
            "X",
            "X_regional",
            "col_names",
            "design",
            "W",
            "G",
            "rho_d",
            "rho_o",
            "rho_w",
            "sigma",
            "beta_d",
            "beta_o",
        }
        assert expected.issubset(set(out.keys()))

    def test_y_shapes(self):
        from bayespecon.dgp.flows import generate_flow_data

        out = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=1,
        )
        assert out["y_vec"].shape == (self.n * self.n,)
        assert out["y_mat"].shape == (self.n, self.n)

    def test_y_vec_mat_consistency(self):
        from bayespecon.dgp.flows import generate_flow_data

        out = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=2,
        )
        np.testing.assert_allclose(out["y_vec"], out["y_mat"].ravel(), atol=1e-12)

    def test_reproducibility(self):
        from bayespecon.dgp.flows import generate_flow_data

        out1 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=99,
        )
        out2 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=99,
        )
        np.testing.assert_allclose(out1["y_vec"], out2["y_vec"], atol=1e-12)

    def test_different_seeds_differ(self):
        from bayespecon.dgp.flows import generate_flow_data

        out1 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=1,
        )
        out2 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=2,
        )
        assert not np.allclose(out1["y_vec"], out2["y_vec"])


class TestFlowDistributionDefault:
    """``distribution`` argument: default lognormal, opt-in normal."""

    n = 5
    kwargs = dict(
        rho_d=0.2, rho_o=0.2, rho_w=0.1, beta_d=[1.0], beta_o=[0.5], sigma=1.0, seed=11
    )

    def test_default_is_lognormal_positive(self):
        from bayespecon.dgp.flows import generate_flow_data

        out = generate_flow_data(n=self.n, **self.kwargs)
        assert out["distribution"] == "lognormal"
        assert (out["y_vec"] > 0).all()
        np.testing.assert_allclose(np.log(out["y_vec"]), out["eta_vec"], atol=1e-10)
        np.testing.assert_allclose(out["eta_vec"], out["eta_mat"].ravel(), atol=1e-12)

    def test_normal_optin_matches_eta(self):
        from bayespecon.dgp.flows import generate_flow_data

        out = generate_flow_data(n=self.n, distribution="normal", **self.kwargs)
        assert out["distribution"] == "normal"
        np.testing.assert_array_equal(out["y_vec"], out["eta_vec"])

    def test_invalid_distribution_raises(self):
        from bayespecon.dgp.flows import generate_flow_data

        with pytest.raises(ValueError, match="distribution"):
            generate_flow_data(n=self.n, distribution="poisson", **self.kwargs)

    def test_panel_default_is_lognormal_positive(self):
        from bayespecon.dgp.flows import generate_panel_flow_data

        out = generate_panel_flow_data(
            n=4,
            T=2,
            rho_d=0.2,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            sigma_alpha=0.0,
            seed=3,
        )
        assert out["distribution"] == "lognormal"
        assert out["params_true"]["distribution"] == "lognormal"
        assert (out["y"] > 0).all()
        np.testing.assert_allclose(np.log(out["y"]), out["eta"], atol=1e-10)


# ---------------------------------------------------------------------------
# Smoke fit tests (minimal draws — just check posterior shapes)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSARFlowFitSmoke:
    """Minimal sampling smoke tests.  Marked slow; skipped in fast CI runs."""

    def setup_method(self):
        from bayespecon.dgp.flows import generate_flow_data

        self.n = 5
        out = generate_flow_data(
            n=self.n,
            rho_d=0.25,
            rho_o=0.25,
            rho_w=0.10,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=42,
        )
        self.G = out["G"]
        self.y = out["y_vec"]
        self.X = out["X"]  # (N, p) full O-D design matrix
        self.col_names = out["col_names"]

    def test_sar_flow_fit_posterior_keys(self):
        from bayespecon.models.flow import SARFlow

        model = SARFlow(
            self.y,
            self.G,
            self.X,
            col_names=self.col_names,
            miter=5,
            titer=50,
            trace_seed=0,
            restrict_positive=True,
        )
        idata = model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        posterior = idata.posterior
        assert "rho_d" in posterior
        assert "rho_o" in posterior
        assert "rho_w" in posterior
        assert "beta" in posterior
        assert "sigma" in posterior

    def test_sar_flow_separable_fit_posterior_keys(self):
        from bayespecon.models.flow import SARFlowSeparable

        model = SARFlowSeparable(
            self.y, self.G, self.X, col_names=self.col_names, trace_seed=0
        )
        idata = model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        posterior = idata.posterior
        assert "rho_d" in posterior
        assert "rho_o" in posterior
        assert "rho_w" in posterior

    def test_sar_flow_rho_d_o_w_in_unit_interval(self):
        """With Dirichlet prior, rho_d, rho_o, rho_w should stay in [0, 1]."""
        from bayespecon.models.flow import SARFlow

        model = SARFlow(
            self.y,
            self.G,
            self.X,
            col_names=self.col_names,
            miter=5,
            titer=50,
            trace_seed=0,
            restrict_positive=True,
        )
        idata = model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        rho_d = idata.posterior["rho_d"].values.ravel()
        rho_o = idata.posterior["rho_o"].values.ravel()
        rho_w = idata.posterior["rho_w"].values.ravel()
        assert (rho_d >= 0).all() and (rho_d <= 1).all()
        assert (rho_o >= 0).all() and (rho_o <= 1).all()
        assert (rho_w >= 0).all() and (rho_w <= 1).all()

    def test_sar_flow_inference_data_property(self):
        from bayespecon.models.flow import SARFlow

        model = SARFlow(
            self.y,
            self.G,
            self.X,
            col_names=self.col_names,
            miter=5,
            titer=50,
            trace_seed=0,
        )
        assert model.inference_data is None  # before fit
        model.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        assert model.inference_data is not None


# ---------------------------------------------------------------------------
# Effects accounting identity (small, fast)
# ---------------------------------------------------------------------------


class TestEffectsAccountingIdentity:
    """total = origin + destination + intra + network (per draw, per variable)."""

    def test_effects_sum_to_total(self):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlowSeparable

        n = 4
        out = generate_flow_data(
            n=n,
            rho_d=0.3,
            rho_o=0.2,
            rho_w=-0.06,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=7,
        )
        G = out["G"]
        model = SARFlowSeparable(
            out["y_vec"], G, out["X"], col_names=out["col_names"], trace_seed=0
        )
        # Use minimal posterior (mock-style: just run prior predictive)
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        effects = model._compute_spatial_effects_posterior(draws=5)

        total = effects["total"]
        check = (
            effects["origin"]
            + effects["destination"]
            + effects["intra"]
            + effects["network"]
        )
        np.testing.assert_allclose(total, check, atol=1e-8)


# ---------------------------------------------------------------------------
# SparseFlowSolveOp unit tests
# ---------------------------------------------------------------------------


class TestSparseFlowSolveOp:
    """Forward pass matches scipy.sparse.linalg.spsolve; gradients are finite."""

    @pytest.fixture(autouse=True)
    def _build_op(self):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.ops import SparseFlowSolveOp

        n = 4
        G = generate_flow_data(n=n, seed=0)["G"]
        W = G.sparse.astype(np.float64).tocsr()
        # Row-standardise
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        W = sp.diags(1.0 / row_sums) @ W

        I_n = sp.eye(n, format="csr")
        I_N = sp.eye(n * n, format="csr")
        Wd = sp.kron(I_n, W, format="csr")
        Wo = sp.kron(W, I_n, format="csr")
        Ww = sp.kron(W, W, format="csr")

        self.n = n
        self.Wd = Wd
        self.Wo = Wo
        self.Ww = Ww
        self.I_N = I_N
        self.op = SparseFlowSolveOp(Wd, Wo, Ww)

    def test_forward_matches_scipy(self):
        """Op.perform output equals spsolve(A, b)."""
        import pytensor
        import pytensor.tensor as pt

        rho_d = pt.dscalar("rho_d")
        rho_o = pt.dscalar("rho_o")
        rho_w = pt.dscalar("rho_w")
        b = pt.dvector("b")
        eta = self.op(rho_d, rho_o, rho_w, b)
        fn = pytensor.function([rho_d, rho_o, rho_w, b], eta)

        rd, ro, rw = 0.2, 0.15, 0.05
        b_val = np.arange(1, self.n**2 + 1, dtype=np.float64)
        A = self.I_N - rd * self.Wd - ro * self.Wo - rw * self.Ww
        expected = sp.linalg.spsolve(A, b_val)

        result = fn(rd, ro, rw, b_val)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gradient_rho_d_finite(self):
        """Gradient w.r.t. rho_d is finite and non-zero."""
        import pytensor
        import pytensor.tensor as pt

        rho_d = pt.dscalar("rho_d")
        rho_o = pt.dscalar("rho_o")
        rho_w = pt.dscalar("rho_w")
        b = pt.dvector("b")
        eta = self.op(rho_d, rho_o, rho_w, b)
        loss = eta.sum()
        g_rd = pt.grad(loss, rho_d)
        fn = pytensor.function([rho_d, rho_o, rho_w, b], g_rd)

        b_val = np.ones(self.n**2, dtype=np.float64)
        result = fn(0.2, 0.15, 0.05, b_val)
        assert np.isfinite(result)
        assert result != 0.0

    def test_gradient_b_matches_adjoint(self):
        """Gradient w.r.t. b equals (A^T)^{-1} ones."""
        import pytensor
        import pytensor.tensor as pt

        rho_d = pt.dscalar("rho_d")
        rho_o = pt.dscalar("rho_o")
        rho_w = pt.dscalar("rho_w")
        b = pt.dvector("b")
        eta = self.op(rho_d, rho_o, rho_w, b)
        loss = eta.sum()
        g_b = pt.grad(loss, b)
        fn = pytensor.function([rho_d, rho_o, rho_w, b], g_b)

        rd, ro, rw = 0.2, 0.15, 0.05
        b_val = np.ones(self.n**2, dtype=np.float64)
        # grad_b = (A^T)^{-1} g,  g = ones (because sum)
        A = self.I_N - rd * self.Wd - ro * self.Wo - rw * self.Ww
        expected = sp.linalg.spsolve(A.T.tocsr(), b_val)
        result = fn(rd, ro, rw, b_val)
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_distinct_op_instances_have_different_ids(self):
        """Two separate SparseFlowSolveOp instances get distinct _op_id values."""
        from bayespecon.ops import SparseFlowSolveOp

        op2 = SparseFlowSolveOp(self.Wd, self.Wo, self.Ww)
        assert self.op._op_id != op2._op_id


# ---------------------------------------------------------------------------
# PoissonSARFlow construction tests
# ---------------------------------------------------------------------------


class TestPoissonFlowConstruction:
    """PoissonSARFlow validates input and builds a PyMC model."""

    @pytest.fixture
    def small_data(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data

        return generate_poisson_flow_data(n=9, seed=0)

    def test_accepts_integer_array(self, small_data):
        from bayespecon.models.flow import PoissonSARFlow

        G = small_data["G"]
        y = small_data["y_vec"]
        model = PoissonSARFlow(
            y, G, small_data["X"], col_names=small_data["col_names"], trace_seed=0
        )
        assert model._y_int_vec.dtype == np.int64

    def test_rounds_close_floats(self, small_data):
        from bayespecon.models.flow import PoissonSARFlow

        G = small_data["G"]
        y_float = small_data["y_vec"].astype(np.float64) + 1e-14
        model = PoissonSARFlow(
            y_float, G, small_data["X"], col_names=small_data["col_names"], trace_seed=0
        )
        assert model._y_int_vec.dtype == np.int64

    def test_raises_on_non_integer_floats(self, small_data):
        from bayespecon.models.flow import PoissonSARFlow

        G = small_data["G"]
        y_bad = small_data["y_vec"].astype(np.float64) + 0.5
        with pytest.raises(ValueError, match="integer-valued"):
            PoissonSARFlow(
                y_bad,
                G,
                small_data["X"],
                col_names=small_data["col_names"],
                trace_seed=0,
            )

    def test_raises_on_negative_values(self, small_data):
        from bayespecon.models.flow import PoissonSARFlow

        G = small_data["G"]
        y_neg = small_data["y_vec"].copy()
        y_neg[0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            PoissonSARFlow(
                y_neg,
                G,
                small_data["X"],
                col_names=small_data["col_names"],
                trace_seed=0,
            )

    def test_model_builds_without_error(self, small_data):
        from bayespecon.models.flow import PoissonSARFlow

        G = small_data["G"]
        model = PoissonSARFlow(
            small_data["y_vec"],
            G,
            small_data["X"],
            col_names=small_data["col_names"],
            restrict_positive=False,
            trace_seed=0,
        )
        built = model._build_pymc_model()
        assert built is not None

    def test_model_coords_contain_coefficient(self, small_data):
        from bayespecon.models.flow import PoissonSARFlow

        G = small_data["G"]
        model = PoissonSARFlow(
            small_data["y_vec"],
            G,
            small_data["X"],
            col_names=small_data["col_names"],
            trace_seed=0,
        )
        built = model._build_pymc_model()
        assert "coefficient" in built.coords

    def test_separable_construction(self, small_data):
        from bayespecon.models.flow import PoissonSARFlowSeparable

        G = small_data["G"]
        model = PoissonSARFlowSeparable(
            small_data["y_vec"],
            G,
            small_data["X"],
            col_names=small_data["col_names"],
            trace_seed=0,
        )
        built = model._build_pymc_model()
        assert built is not None

    @pytest.mark.parametrize("logdet_method", ["eigenvalue", "chebyshev", "trace_mc"])
    def test_separable_construction_supports_logdet_methods(
        self, small_data, logdet_method
    ):
        from bayespecon.models.flow import PoissonSARFlowSeparable

        G = small_data["G"]
        model = PoissonSARFlowSeparable(
            small_data["y_vec"],
            G,
            small_data["X"],
            col_names=small_data["col_names"],
            logdet_method=logdet_method,
            trace_seed=0,
        )
        built = model._build_pymc_model()
        lp = built.point_logps(built.initial_point())
        assert all(np.isfinite(v) for v in lp.values())

    def test_separable_construction_rejects_unknown_logdet_method(self, small_data):
        from bayespecon.models.flow import PoissonSARFlowSeparable

        G = small_data["G"]
        with pytest.raises(ValueError, match="logdet_method"):
            PoissonSARFlowSeparable(
                small_data["y_vec"],
                G,
                small_data["X"],
                col_names=small_data["col_names"],
                logdet_method="sparse_spline",
                trace_seed=0,
            )


# ---------------------------------------------------------------------------
# generate_poisson_flow_data unit tests
# ---------------------------------------------------------------------------


class TestGeneratePoissonFlowData:
    """DGP function returns correct types, shapes, and numerical properties."""

    @pytest.fixture(scope="class")
    def data(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data

        return generate_poisson_flow_data(n=9, k=2, seed=7)

    def test_keys_present(self, data):
        expected = {
            "y_vec",
            "y_mat",
            "eta_vec",
            "lambda_vec",
            "Xd",
            "Xo",
            "X",
            "col_names",
            "design",
            "W",
            "G",
            "rho_d",
            "rho_o",
            "rho_w",
            "beta_d",
            "beta_o",
        }
        assert expected <= set(data.keys())

    def test_y_vec_dtype_int64(self, data):
        assert data["y_vec"].dtype == np.int64

    def test_y_mat_dtype_int64(self, data):
        assert data["y_mat"].dtype == np.int64

    def test_y_vec_shape(self, data):
        n = data["W"].shape[0]
        assert data["y_vec"].shape == (n * n,)

    def test_y_mat_shape(self, data):
        n = data["W"].shape[0]
        assert data["y_mat"].shape == (n, n)

    def test_y_vec_non_negative(self, data):
        assert np.all(data["y_vec"] >= 0)

    def test_lambda_equals_exp_eta(self, data):
        np.testing.assert_allclose(
            data["lambda_vec"], np.exp(data["eta_vec"]), rtol=1e-12
        )

    def test_y_mat_ravels_to_y_vec(self, data):
        np.testing.assert_array_equal(data["y_mat"].ravel(), data["y_vec"])

    def test_reproducible(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data

        d1 = generate_poisson_flow_data(n=9, seed=42)
        d2 = generate_poisson_flow_data(n=9, seed=42)
        np.testing.assert_array_equal(d1["y_vec"], d2["y_vec"])

    def test_different_seeds_differ(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data

        d1 = generate_poisson_flow_data(n=9, seed=1)
        d2 = generate_poisson_flow_data(n=9, seed=2)
        assert not np.array_equal(d1["y_vec"], d2["y_vec"])

    def test_accepts_err_hetero_gdf_args(self):
        """API parity: accepts err_hetero and gdf without error."""
        from bayespecon.dgp.flows import generate_poisson_flow_data

        data = generate_poisson_flow_data(n=4, err_hetero=True)
        assert "y_vec" in data

    def test_x_shape(self, data):
        n = data["W"].shape[0]
        k = 2
        # X is now the full O-D design matrix (N, p)
        N = n * n
        assert data["X"].shape[0] == N
        # Xd and Xo are still (n, k) regional attributes
        assert data["Xd"].shape == (n, k)
        assert data["Xo"].shape == (n, k)

    def test_rho_values_preserved(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data

        d = generate_poisson_flow_data(n=9, rho_d=0.25, rho_o=0.10, rho_w=0.05)
        assert d["rho_d"] == 0.25
        assert d["rho_o"] == 0.10
        assert d["rho_w"] == 0.05


# ---------------------------------------------------------------------------
# PoissonSARFlow smoke fit tests (marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPoissonFlowFitSmoke:
    """Minimal-draw fits produce expected posterior keys and shapes."""

    @pytest.fixture(scope="class")
    def fitted_model(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonSARFlow

        data = generate_poisson_flow_data(n=9, seed=0)
        G = data["G"]
        model = PoissonSARFlow(
            data["y_vec"], G, data["X"], col_names=data["col_names"], trace_seed=0
        )
        model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        return model

    def test_posterior_has_beta(self, fitted_model):
        idata = fitted_model.inference_data
        assert "beta" in idata.posterior.data_vars

    def test_posterior_has_rho_d(self, fitted_model):
        assert "rho_d" in fitted_model.inference_data.posterior.data_vars

    def test_no_sigma_in_posterior(self, fitted_model):
        """Poisson model has no sigma parameter."""
        assert "sigma" not in fitted_model.inference_data.posterior.data_vars

    def test_beta_shape(self, fitted_model):
        beta = fitted_model.inference_data.posterior["beta"]
        # (chains, draws, k_coefficients)
        assert beta.ndim == 3

    def test_lambda_not_stored_by_default(self, fitted_model):
        assert "lambda" not in fitted_model.inference_data.posterior.data_vars

    @pytest.fixture(scope="class")
    def fitted_separable(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonSARFlowSeparable

        data = generate_poisson_flow_data(n=9, seed=1)
        G = data["G"]
        model = PoissonSARFlowSeparable(
            data["y_vec"], G, data["X"], col_names=data["col_names"], trace_seed=0
        )
        model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        return model

    def test_separable_rho_w_deterministic(self, fitted_separable):
        """rho_w should equal -rho_d * rho_o for all draws."""
        post = fitted_separable.inference_data.posterior
        rho_w = post["rho_w"].values.ravel()
        rho_d = post["rho_d"].values.ravel()
        rho_o = post["rho_o"].values.ravel()
        np.testing.assert_allclose(rho_w, -rho_d * rho_o, atol=1e-10)


class TestPoissonFlowApproximation:
    def test_fit_approx_advi_returns_posterior_without_lambda(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonSARFlow

        data = generate_poisson_flow_data(n=4, seed=3)
        model = PoissonSARFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
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

    def test_fit_approx_fullrank_keeps_small_deterministics(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonSARFlowSeparable

        data = generate_poisson_flow_data(n=4, seed=4)
        model = PoissonSARFlowSeparable(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )

        idata = model.fit_approx(
            draws=10,
            n=20,
            method="fullrank_advi",
            progressbar=False,
            random_seed=0,
        )

        assert "rho_w" in idata.posterior.data_vars
        assert "lambda" not in idata.posterior.data_vars

    def test_fit_approx_rejects_unknown_method(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonSARFlow

        data = generate_poisson_flow_data(n=4, seed=5)
        model = PoissonSARFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )

        with pytest.raises(ValueError, match="fit_approx method"):
            model.fit_approx(method="laplace", progressbar=False)


# ---------------------------------------------------------------------------
# Parameter recovery tests (slow — deselected by default)
# ---------------------------------------------------------------------------

# Flow-specific dimensions and true parameter values
FLOW_N = 8  # 64 O-D pairs — enough for reliable identification
RHO_D_TRUE = 0.25
RHO_O_TRUE = 0.25
RHO_W_TRUE = 0.10
BETA_D_TRUE = np.array([1.0, -0.5])
BETA_O_TRUE = np.array([0.5, 0.3])
SIGMA_TRUE = 1.0

# Separable-model true values — asymmetric so rho_d ≠ rho_o (breaks swap
# symmetry); rho_w = -0.4*0.3 = -0.12 provides clear identification signal
RHO_D_SEP_TRUE = 0.40
RHO_O_SEP_TRUE = 0.30
# Larger grid for separable: the bilinear rho_w=-rho_d*rho_o term creates a
# harder posterior surface — more obs improves identification reliably
FLOW_N_SEP = 12  # 144 O-D pairs (vs 64 for unrestricted model)

# Poisson-specific true values
POI_RHO_D_TRUE = 0.3
POI_RHO_O_TRUE = 0.2
POI_RHO_W_TRUE = 0.1

# Separable Poisson — asymmetric for the same reason
POI_RHO_D_SEP_TRUE = 0.40
POI_RHO_O_SEP_TRUE = 0.30

# Poisson models are more expensive per step (no conjugacy); use fewer samples
POISSON_SAMPLE_KWARGS: dict = dict(
    tune=400, draws=600, chains=2, random_seed=42, progressbar=False
)
# Separable Poisson: bilinear rho_w term makes the posterior harder to tune;
# extra steps ensure the mass matrix adapts before drawing
POISSON_SEP_SAMPLE_KWARGS: dict = dict(
    tune=400, draws=400, chains=2, random_seed=42, progressbar=False
)

# Tolerances — wider than standard panel models because 3 spatial
# parameters are harder to identify simultaneously
ABS_TOL_RHO = 0.20
ABS_TOL_RHO_SEP = 0.25  # separable model: slightly wider
ABS_TOL_RHO_POI = 0.25  # Poisson: harder
ABS_TOL_BETA = 0.35
ABS_TOL_BETA_SEP = 0.40
ABS_TOL_BETA_POI = 0.40
ABS_TOL_SIGMA = 0.35


# ---------------------------------------------------------------------------
# Smoke tests: SARFlowSeparable with alternative logdet methods
# ---------------------------------------------------------------------------


class TestSARFlowSeparableLogdetMethods:
    """SARFlowSeparable fits without error for each logdet method."""

    @pytest.fixture(params=["eigenvalue", "chebyshev", "trace_mc"], scope="class")
    def fitted_model(self, request):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlowSeparable

        out = generate_flow_data(
            n=5,
            rho_d=0.3,
            rho_o=0.2,
            rho_w=-0.06,
            beta_d=np.array([0.5, -0.3]),
            beta_o=np.array([0.4, 0.2]),
            sigma=1.0,
            seed=0,
        )
        G = out["G"]
        model = SARFlowSeparable(
            out["y_vec"],
            G,
            out["X"],
            col_names=out["col_names"],
            logdet_method=request.param,
            miter=15,
            trace_seed=0,
        )
        idata = model.fit(**{**SAMPLE_KWARGS, "draws": 50, "tune": 50, "chains": 1})
        return idata, request.param

    def test_posterior_has_rho_d(self, fitted_model):
        idata, _ = fitted_model
        assert "rho_d" in idata.posterior

    def test_posterior_has_rho_o(self, fitted_model):
        idata, _ = fitted_model
        assert "rho_o" in idata.posterior

    def test_rho_w_is_deterministic(self, fitted_model):
        idata, _ = fitted_model
        assert "rho_w" in idata.posterior

    def test_beta_shape(self, fitted_model):
        idata, _ = fitted_model
        assert idata.posterior["beta"].shape[-1] > 0

    def test_invalid_method_raises(self):
        import pytest

        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlowSeparable

        out = generate_flow_data(
            n=4,
            rho_d=0.2,
            rho_o=0.1,
            rho_w=-0.02,
            beta_d=np.array([1.0]),
            beta_o=np.array([0.5]),
            sigma=1.0,
            seed=1,
        )
        G = out["G"]
        with pytest.raises(ValueError, match="logdet_method"):
            SARFlowSeparable(
                out["y_vec"],
                G,
                out["X"],
                col_names=out["col_names"],
                logdet_method="traces",
            )


@pytest.mark.slow
class TestSARFlowRecovery:
    """SARFlow posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_sar_flow(self):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlow

        out = generate_flow_data(
            n=FLOW_N,
            rho_d=RHO_D_TRUE,
            rho_o=RHO_O_TRUE,
            rho_w=RHO_W_TRUE,
            beta_d=BETA_D_TRUE,
            beta_o=BETA_O_TRUE,
            sigma=SIGMA_TRUE,
            seed=42,
        )
        G = out["G"]
        # Default DGP is lognormal; SARFlow has Gaussian likelihood, so fit on
        # the latent scale.  np.log(y_vec) == eta_vec by construction.
        model = SARFlow(
            np.log(out["y_vec"]),
            G,
            out["X"],
            col_names=out["col_names"],
            miter=5,
            titer=50,
            trace_seed=0,
            restrict_positive=True,
        )
        idata = model.fit(**SAMPLE_KWARGS)
        return idata

    def test_sar_flow_recovers_rho_d(self, fitted_sar_flow):
        rho_hat = float(fitted_sar_flow.posterior["rho_d"].mean())
        assert abs(rho_hat - RHO_D_TRUE) < ABS_TOL_RHO, (
            f"SARFlow rho_d: expected ≈{RHO_D_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_recovers_rho_o(self, fitted_sar_flow):
        rho_hat = float(fitted_sar_flow.posterior["rho_o"].mean())
        assert abs(rho_hat - RHO_O_TRUE) < ABS_TOL_RHO, (
            f"SARFlow rho_o: expected ≈{RHO_O_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_recovers_rho_w(self, fitted_sar_flow):
        rho_hat = float(fitted_sar_flow.posterior["rho_w"].mean())
        assert abs(rho_hat - RHO_W_TRUE) < ABS_TOL_RHO, (
            f"SARFlow rho_w: expected ≈{RHO_W_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_recovers_sigma(self, fitted_sar_flow):
        sigma_hat = float(fitted_sar_flow.posterior["sigma"].mean())
        assert abs(sigma_hat - SIGMA_TRUE) < ABS_TOL_SIGMA, (
            f"SARFlow sigma: expected ≈{SIGMA_TRUE}, got {sigma_hat:.3f}"
        )


@pytest.mark.slow
class TestSARFlowSeparableRecovery:
    """SARFlowSeparable posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_separable(self):
        from bayespecon.dgp.flows import generate_flow_data_separable
        from bayespecon.models.flow import SARFlowSeparable

        out = generate_flow_data_separable(
            n=FLOW_N_SEP,
            rho_d=RHO_D_SEP_TRUE,
            rho_o=RHO_O_SEP_TRUE,
            beta_d=BETA_D_TRUE,
            beta_o=BETA_O_TRUE,
            sigma=SIGMA_TRUE,
            seed=42,
        )
        G = out["G"]
        # Default DGP is lognormal; fit on the latent (log) scale.
        model = SARFlowSeparable(
            np.log(out["y_vec"]),
            G,
            out["X"],
            col_names=out["col_names"],
            trace_seed=0,
        )
        idata = model.fit(**SAMPLE_KWARGS)
        return idata

    def test_separable_recovers_rho_d(self, fitted_separable):
        rho_hat = float(fitted_separable.posterior["rho_d"].mean())
        assert abs(rho_hat - RHO_D_SEP_TRUE) < ABS_TOL_RHO_SEP, (
            f"SARFlowSeparable rho_d: expected ≈{RHO_D_SEP_TRUE}, got {rho_hat:.3f}"
        )

    def test_separable_recovers_rho_o(self, fitted_separable):
        rho_hat = float(fitted_separable.posterior["rho_o"].mean())
        assert abs(rho_hat - RHO_O_SEP_TRUE) < ABS_TOL_RHO_SEP, (
            f"SARFlowSeparable rho_o: expected ≈{RHO_O_SEP_TRUE}, got {rho_hat:.3f}"
        )


@pytest.mark.slow
class TestPoissonFlowRecovery:
    """PoissonSARFlow posterior means should be close to true DGP values."""

    # Larger N than the smoke-test default: NUTS on the 3-parameter Poisson
    # SAR flow likelihood is prohibitively slow, and at n=8 the rho's are
    # weakly identified. n=12 (144 OD pairs) gives ADVI enough signal.
    N_REC = 12

    @pytest.fixture(scope="class")
    def fitted_poisson(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonSARFlow

        data = generate_poisson_flow_data(
            n=self.N_REC,
            rho_d=POI_RHO_D_TRUE,
            rho_o=POI_RHO_O_TRUE,
            rho_w=POI_RHO_W_TRUE,
            seed=42,
        )
        G = data["G"]
        model = PoissonSARFlow(
            data["y_vec"],
            G,
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )
        # Mean-field ADVI: NUTS on this likelihood is prohibitively slow and
        # full-rank ADVI is numerically unstable (the Cholesky factor over the
        # rho parameters can drive proposals outside the SAR stationary
        # region, making the flow operator singular). See the same treatment
        # in TestPoissonFlowPanelRecovery / TestPoissonFlowSeparablePanelRecovery.
        idata = model.fit_approx(
            method="advi",
            n=20000,
            draws=2000,
            random_seed=42,
            progressbar=False,
        )
        return idata

    def test_poisson_flow_recovers_rho_d(self, fitted_poisson):
        rho_hat = float(fitted_poisson.posterior["rho_d"].mean())
        assert abs(rho_hat - POI_RHO_D_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlow rho_d: expected ≈{POI_RHO_D_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_flow_recovers_rho_o(self, fitted_poisson):
        rho_hat = float(fitted_poisson.posterior["rho_o"].mean())
        assert abs(rho_hat - POI_RHO_O_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlow rho_o: expected ≈{POI_RHO_O_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_flow_recovers_rho_w(self, fitted_poisson):
        rho_hat = float(fitted_poisson.posterior["rho_w"].mean())
        assert abs(rho_hat - POI_RHO_W_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlow rho_w: expected ≈{POI_RHO_W_TRUE}, got {rho_hat:.3f}"
        )


@pytest.mark.slow
class TestPoissonFlowSeparableRecovery:
    """PoissonSARFlowSeparable posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_poisson_separable(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data_separable
        from bayespecon.models.flow import PoissonSARFlowSeparable

        data = generate_poisson_flow_data_separable(
            n=12,
            rho_d=POI_RHO_D_SEP_TRUE,
            rho_o=POI_RHO_O_SEP_TRUE,
            seed=42,
        )
        G = data["G"]
        model = PoissonSARFlowSeparable(
            data["y_vec"],
            G,
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )
        idata = model.fit(**POISSON_SEP_SAMPLE_KWARGS)
        return idata

    def test_poisson_separable_recovers_rho_d(self, fitted_poisson_separable):
        rho_hat = float(fitted_poisson_separable.posterior["rho_d"].mean())
        assert abs(rho_hat - POI_RHO_D_SEP_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlowSeparable rho_d: expected ≈{POI_RHO_D_SEP_TRUE}, got {rho_hat:.3f}"
        )

    def test_poisson_separable_recovers_rho_o(self, fitted_poisson_separable):
        rho_hat = float(fitted_poisson_separable.posterior["rho_o"].mean())
        assert abs(rho_hat - POI_RHO_O_SEP_TRUE) < ABS_TOL_RHO_POI, (
            f"PoissonSARFlowSeparable rho_o: expected ≈{POI_RHO_O_SEP_TRUE}, got {rho_hat:.3f}"
        )


# ---------------------------------------------------------------------------
# kron_solve_vec / kron_solve_matrix utilities
# ---------------------------------------------------------------------------


class TestKronSolveUtilities:
    """Verify kron_solve_vec and kron_solve_matrix match scipy reference."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from bayespecon.ops import kron_solve_matrix, kron_solve_vec

        self.kron_solve_vec = kron_solve_vec
        self.kron_solve_matrix = kron_solve_matrix

        rng = np.random.default_rng(42)
        self.n = 6
        N = self.n**2
        W_dense = rng.random((self.n, self.n))
        W_dense /= W_dense.sum(axis=1, keepdims=True)
        self.W = sp.csr_matrix(W_dense)
        I_n = sp.eye(self.n, format="csr", dtype=np.float64)
        rd, ro = 0.3, 0.4
        self.Ld = (I_n - rd * self.W).tocsr()
        self.Lo = (I_n - ro * self.W).tocsr()
        # Full N×N system matrix for reference
        Wd = sp.kron(sp.eye(self.n), self.W, format="csr")
        Wo = sp.kron(self.W, sp.eye(self.n), format="csr")
        Ww = sp.kron(self.W, self.W, format="csr")
        self.A = sp.eye(N, format="csr") - rd * Wd - ro * Wo + rd * ro * Ww
        self.b = rng.random(N)
        self.B = rng.random((N, 4))

    def test_kron_solve_vec_matches_scipy(self):
        eta_ref = sp.linalg.spsolve(self.A, self.b)
        eta_kron = self.kron_solve_vec(self.Lo, self.Ld, self.b, self.n)
        np.testing.assert_allclose(eta_kron, eta_ref, atol=1e-10)

    def test_kron_solve_matrix_matches_scipy(self):
        H_ref = sp.linalg.spsolve(self.A, self.B)
        H_kron = self.kron_solve_matrix(self.Lo, self.Ld, self.B, self.n)
        np.testing.assert_allclose(H_kron, H_ref, atol=1e-10)

    def test_kron_solve_matrix_single_column_matches_vec(self):
        b = self.b
        h_vec = self.kron_solve_vec(self.Lo, self.Ld, b, self.n)
        h_mat = self.kron_solve_matrix(self.Lo, self.Ld, b[:, np.newaxis], self.n)
        np.testing.assert_allclose(h_mat[:, 0], h_vec, atol=1e-12)


# ---------------------------------------------------------------------------
# Public spatial_effects + posterior_predictive (Phase 3.3)
# ---------------------------------------------------------------------------


class TestFlowSpatialEffectsAndPPC:
    @pytest.fixture(scope="class")
    def fitted(self):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlow

        n = 5
        data = generate_flow_data(
            n=n,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=0.1,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.2],
            sigma=1.0,
            seed=11,
        )
        G = data["G"]
        model = SARFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )
        model.fit(draws=40, tune=40, chains=1, progressbar=False, random_seed=0)
        return model

    def test_spatial_effects_returns_dataframe(self, fitted):
        df = fitted.spatial_effects()
        assert df.index.names == ["predictor", "side", "effect"]
        assert {"mean", "ci_lower", "ci_upper", "bayes_pvalue"} <= set(df.columns)
        # Five effect labels per predictor.
        effects = set(df.index.get_level_values("effect"))
        assert effects == {"origin", "destination", "intra", "network", "total"}

    def test_spatial_effects_return_samples(self, fitted):
        df, samples = fitted.spatial_effects(return_posterior_samples=True)
        for key in ["origin", "destination", "intra", "network", "total"]:
            arr = samples[key]
            assert arr.ndim == 2
            assert arr.shape[1] == fitted._k

    def test_posterior_predictive_shape(self, fitted):
        y_rep = fitted.posterior_predictive(n_draws=10, random_seed=3)
        assert y_rep.shape == (10, fitted._N)
        assert np.all(np.isfinite(y_rep))

    def test_posterior_predictive_reproducible(self, fitted):
        a = fitted.posterior_predictive(n_draws=5, random_seed=7)
        b = fitted.posterior_predictive(n_draws=5, random_seed=7)
        np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# LeSage-decomposition correctness tests for the new spatial_effects math
# ---------------------------------------------------------------------------


class TestFlowEffectsLeSageDecomposition:
    """Verify spatial_effects matches the LeSage (2008) calc_effects.m reference."""

    def test_helper_zero_rho_closed_form(self):
        """With A = I, effects collapse to closed-form expressions."""
        from bayespecon.models.flow import (
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 4, 2
        dmask, omask, imask = _build_flow_effect_masks(n)
        beta_d = np.array([2.0, -1.0])
        beta_o = np.array([0.5, 3.0])
        res = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
        )
        for p in range(k):
            bd, bo = beta_d[p], beta_o[p]
            assert np.isclose(res["total"][p], bd + bo)
            assert np.isclose(res["intra"][p], (bd + bo) / n)
            assert np.isclose(res["origin"][p], (n - 1) / n * bo)
            assert np.isclose(res["destination"][p], (n - 1) / n * bd)
            assert np.isclose(res["network"][p], 0.0)

    def test_helper_total_identity(self):
        """total == origin + destination + intra + network for arbitrary A."""
        import scipy.sparse as sp_local

        from bayespecon.models.flow import (
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 4, 2
        N = n * n
        dmask, omask, imask = _build_flow_effect_masks(n)
        # Random invertible A.
        rng = np.random.default_rng(0)
        M = rng.standard_normal((N, N))
        A = np.eye(N) + 0.05 * M
        lu = sp_local.linalg.splu(sp_local.csc_matrix(A))

        def solve(rhs):
            return lu.solve(rhs)

        beta_d = np.array([2.0, -1.0])
        beta_o = np.array([0.5, 3.0])
        res = _compute_flow_effects_lesage(
            solve,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
        )
        for p in range(k):
            s = (
                res["origin"][p]
                + res["destination"][p]
                + res["intra"][p]
                + res["network"][p]
            )
            assert np.isclose(res["total"][p], s)

    def test_calc_effects_reference_match(self):
        """End-to-end sanity: spatial_effects matches a hand-rolled LeSage solve at posterior means."""
        import scipy.sparse as sp_local

        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlow, _build_flow_effect_masks

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=3,
        )
        G = data["G"]
        model = SARFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)

        post = model.spatial_effects(return_posterior_samples=True)
        df, samples = post
        # Reference computation: rebuild for one draw and compare.
        rho_d = float(model.inference_data.posterior["rho_d"].values.reshape(-1)[0])
        rho_o = float(model.inference_data.posterior["rho_o"].values.reshape(-1)[0])
        rho_w = float(model.inference_data.posterior["rho_w"].values.reshape(-1)[0])
        beta = model.inference_data.posterior["beta"].values.reshape(
            -1, len(model._feature_names)
        )[0]
        k = model._k
        beta_d = beta[2 : 2 + k]
        beta_o = beta[2 + k : 2 + 2 * k]
        # Posterior β_intra (DGP value is 0, but the fitted draw is generally nonzero).
        beta_intra = beta[2 + 2 * k : 2 + 3 * k]

        N = n * n
        dmask, omask, imask = _build_flow_effect_masks(n)
        Wd = sp_local.kron(sp_local.eye(n), model._W_sparse, format="csr")
        Wo = sp_local.kron(model._W_sparse, sp_local.eye(n), format="csr")
        Ww = sp_local.kron(model._W_sparse, model._W_sparse, format="csr")
        A = (sp_local.eye(N) - rho_d * Wd - rho_o * Wo - rho_w * Ww).tocsc()
        lu = sp_local.linalg.splu(A)

        for p in range(k):
            shock = np.zeros((N, n))
            shock[dmask] = beta_d[p]
            shock[omask] = beta_o[p]
            # X_intra = intra_indicator * X_dest, so shocking X_dest also shocks
            # the intra cell by β_intra (Thomas-Agnan & LeSage 2014, §83.5).
            shock[imask] = beta_d[p] + beta_o[p] + beta_intra[p]
            T_resp = lu.solve(shock)
            ref_total = T_resp.sum() / N
            ref_intra = T_resp[imask].sum() / N
            ref_origin = T_resp[omask].sum() / N
            ref_dest = T_resp[dmask].sum() / N
            ref_network = ref_total - ref_origin - ref_dest - ref_intra

            assert np.isclose(samples["total"][0, p], ref_total)
            assert np.isclose(samples["intra"][0, p], ref_intra)
            assert np.isclose(samples["origin"][0, p], ref_origin)
            assert np.isclose(samples["destination"][0, p], ref_dest)
            assert np.isclose(samples["network"][0, p], ref_network)

    def test_separable_matches_unrestricted(self):
        """SARFlowSeparable effects ≈ SARFlow effects when rho_w = -rho_d * rho_o."""
        import scipy.sparse as sp_local

        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import (
            SARFlow,
            SARFlowSeparable,
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=-0.2 * 0.15,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=4,
        )
        G = data["G"]
        # Compute reference effects directly with both code paths at the same parameter point.
        from bayespecon.ops import kron_solve_matrix

        rd, ro = 0.2, 0.15
        rw = -rd * ro
        dmask, omask, imask = _build_flow_effect_masks(n)
        N = n * n
        Wd = sp_local.kron(sp_local.eye(n), G.sparse.tocsr(), format="csr")
        Wo = sp_local.kron(G.sparse.tocsr(), sp_local.eye(n), format="csr")
        Ww = sp_local.kron(G.sparse.tocsr(), G.sparse.tocsr(), format="csr")
        A = (sp_local.eye(N) - rd * Wd - ro * Wo - rw * Ww).tocsc()
        lu = sp_local.linalg.splu(A)

        I_n = sp_local.eye(n, format="csr")
        Ld = (I_n - rd * G.sparse.tocsr()).tocsr()
        Lo = (I_n - ro * G.sparse.tocsr()).tocsr()

        beta_d = np.array([1.0])
        beta_o = np.array([0.5])

        res_general = _compute_flow_effects_lesage(
            lambda r: lu.solve(r),
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            1,
        )
        res_kron = _compute_flow_effects_lesage(
            lambda r: kron_solve_matrix(Lo, Ld, r, n),
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            1,
        )
        for key in res_general:
            np.testing.assert_allclose(res_general[key], res_kron[key], atol=1e-10)


class TestFlowPanelSpatialEffectsAndPPC:
    """Smoke + correctness tests for the new FlowPanelModel public API."""

    @pytest.fixture(scope="class")
    def fitted_panel(self):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.graph import flow_design_matrix
        from bayespecon.models.flow_panel import SARFlowPanel

        n = 4
        T = 3
        G = generate_flow_data(n=n, seed=0)["G"]
        rng = np.random.default_rng(5)
        y_list, X_list = [], []
        col_names = None
        for _ in range(T):
            Xr = rng.standard_normal((n, 1))
            d = flow_design_matrix(Xr)
            if col_names is None:
                col_names = d.feature_names
            X_list.append(d.combined)
            y_list.append(rng.standard_normal(n * n))
        y = np.concatenate(y_list)
        X = np.vstack(X_list)
        model = SARFlowPanel(
            y=y,
            G=G,
            X=X,
            T=T,
            col_names=col_names,
            trace_seed=0,
        )
        model.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        return model

    def test_panel_spatial_effects_dataframe(self, fitted_panel):
        df = fitted_panel.spatial_effects()
        assert df.index.names == ["predictor", "side", "effect"]
        effects = set(df.index.get_level_values("effect"))
        assert effects == {"origin", "destination", "intra", "network", "total"}

    def test_panel_total_identity(self, fitted_panel):
        df, samples = fitted_panel.spatial_effects(return_posterior_samples=True)
        s = (
            samples["origin"]
            + samples["destination"]
            + samples["intra"]
            + samples["network"]
        )
        np.testing.assert_allclose(samples["total"], s, atol=1e-10)

    def test_panel_posterior_predictive(self, fitted_panel):
        y_rep = fitted_panel.posterior_predictive(n_draws=5, random_seed=11)
        assert y_rep.shape == (5, fitted_panel._N_flow * fitted_panel._T)
        assert np.all(np.isfinite(y_rep))


# ---------------------------------------------------------------------------
# Phase 4: Thomas-Agnan & LeSage (2014) enhancements
# ---------------------------------------------------------------------------


class TestFlowEffectsAsymmetricAndIntra:
    """Tests for the new asymmetric Xo/Xd shocks and beta_intra contribution."""

    def test_helper_intra_shock_includes_beta_intra(self):
        """β_intra adds (β_intra)/n to the intra effect on the dest side under A=I."""
        from bayespecon.models.flow import (
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 4, 1
        dmask, omask, imask = _build_flow_effect_masks(n)
        beta_d = np.array([1.0])
        beta_o = np.array([0.5])
        beta_intra = np.array([2.0])

        res_with = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
            beta_intra=beta_intra,
        )
        res_without = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
            beta_intra=None,
        )
        # Intra increases by β_intra/n (single perturbed region averaged).
        assert np.isclose(
            res_with["intra"][0] - res_without["intra"][0],
            beta_intra[0] / n,
        )
        # Total increases by β_intra/n as well.
        assert np.isclose(
            res_with["total"][0] - res_without["total"][0],
            beta_intra[0] / n,
        )
        # Network unchanged (intra is part of the "owner" region's column).
        assert np.isclose(res_with["network"][0], res_without["network"][0])

    def test_helper_per_side_keys_decompose_combined(self):
        """combined_eff == dest_eff + orig_eff for every effect type."""
        from bayespecon.models.flow import (
            _EFFECT_KEYS,
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 5, 2
        dmask, omask, imask = _build_flow_effect_masks(n)
        rng = np.random.default_rng(7)
        beta_d = rng.standard_normal(k)
        beta_o = rng.standard_normal(k)
        beta_intra = rng.standard_normal(k)

        res = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
            beta_intra=beta_intra,
        )
        for eff in _EFFECT_KEYS:
            np.testing.assert_allclose(
                res[eff], res[f"dest_{eff}"] + res[f"orig_{eff}"], atol=1e-12
            )

    def test_asymmetric_design_separate_mode(self):
        """Asymmetric Xo!=Xd auto-detects to 'separate' mode and reports both sides."""
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.graph import flow_design_matrix_with_orig
        from bayespecon.models.flow import SARFlow

        n = 4
        rng = np.random.default_rng(0)
        Xd = rng.standard_normal((n, 1))
        Xo = rng.standard_normal((n, 1))  # different from Xd

        data = generate_flow_data(
            n=n,
            rho_d=0.1,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=1,
        )
        G = data["G"]
        # Override design with asymmetric Xo!=Xd
        design = flow_design_matrix_with_orig(Xd, Xo, col_names=["x"])
        model = SARFlow(
            data["y_vec"],
            G,
            design.combined,
            col_names=design.feature_names,
            trace_seed=0,
        )
        assert model._symmetric_xo_xd is False
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        df = model.spatial_effects()  # auto -> separate
        sides = set(df.index.get_level_values("side"))
        assert sides == {"dest", "orig"}

    def test_combined_mode_matches_dest_plus_orig(self):
        """mode='combined' equals the sum of per-side rows from mode='separate'."""
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlow

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.15,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=2,
        )
        model = SARFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        df_combined = model.spatial_effects(mode="combined")
        df_separate = model.spatial_effects(mode="separate")

        for eff in ("origin", "destination", "intra", "network", "total"):
            cmean = df_combined.xs(eff, level="effect")["mean"].iloc[0]
            d = df_separate.xs(("dest", eff), level=("side", "effect"))["mean"].iloc[0]
            o = df_separate.xs(("orig", eff), level=("side", "effect"))["mean"].iloc[0]
            assert np.isclose(cmean, d + o, atol=1e-10)


class TestOLSFlowEffects:
    """Tests for the new non-spatial OLSFlow gravity baseline."""

    def test_olsflow_fits_and_effects_table_83_1(self):
        """OLSFlow analytic effects: TE = βd + βo, NE = 0, IE = (βd+βo+β_intra)/n."""
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import OLSFlow

        n = 5
        # ρ = 0 in the DGP so the OLS model is correctly specified.
        data = generate_flow_data(
            n=n,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.1,
            seed=0,
        )
        # DGP defaults to lognormal; OLS fit on the latent (log) scale.
        model = OLSFlow(
            np.log(data["y_vec"]), data["G"], data["X"], col_names=data["col_names"]
        )
        model.fit(draws=40, tune=40, chains=1, progressbar=False, random_seed=0)
        df = model.spatial_effects(mode="combined")

        # NE = 0 exactly (closed form).
        assert np.isclose(df.xs("network", level="effect")["mean"].iloc[0], 0.0)
        # TE close to βd + βo (sampling-noise tolerance).
        te = df.xs("total", level="effect")["mean"].iloc[0]
        assert np.isclose(te, 1.0 + 0.5, atol=0.2)
        # IE = (TE + β_intra)/n with β_intra ≈ 0 in the DGP.
        ie = df.xs("intra", level="effect")["mean"].iloc[0]
        assert np.isclose(ie, te / n, atol=0.1)

    def test_olsflow_posterior_predictive_shape(self):
        """OLSFlow posterior_predictive returns (n_draws, N) finite values."""
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import OLSFlow

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=1,
        )
        model = OLSFlow(
            data["y_vec"], data["G"], data["X"], col_names=data["col_names"]
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        y_rep = model.posterior_predictive(n_draws=5, random_seed=3)
        assert y_rep.shape == (5, n * n)
        assert np.all(np.isfinite(y_rep))


class TestPoissonGravityFlow:
    """Tests for the new aspatial Poisson gravity flow baseline (PoissonFlow)."""

    @pytest.fixture
    def small_data(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data

        # rho = 0 so the aspatial PoissonFlow is correctly specified.
        return generate_poisson_flow_data(n=6, rho_d=0.0, rho_o=0.0, rho_w=0.0, seed=0)

    def test_constructs_and_builds_pymc_model(self, small_data):
        from bayespecon.models.flow import PoissonFlow

        model = PoissonFlow(
            small_data["y_vec"],
            small_data["G"],
            small_data["X"],
            col_names=small_data["col_names"],
        )
        assert model._y_int_vec.dtype == np.int64
        pm_model = model._build_pymc_model()
        names = {v.name for v in pm_model.unobserved_RVs}
        # No spatial-lag or sigma parameters should be sampled.
        assert "rho_d" not in names and "rho_o" not in names and "rho_w" not in names
        assert "sigma" not in names
        assert "beta" in names

    def test_rounds_close_floats_and_rejects_negatives(self, small_data):
        from bayespecon.models.flow import PoissonFlow

        y_float = small_data["y_vec"].astype(np.float64) + 1e-14
        m = PoissonFlow(
            y_float, small_data["G"], small_data["X"], col_names=small_data["col_names"]
        )
        assert m._y_int_vec.dtype == np.int64

        y_bad = small_data["y_vec"].astype(np.float64) + 0.5
        with pytest.raises(ValueError, match="integer-valued"):
            PoissonFlow(
                y_bad,
                small_data["G"],
                small_data["X"],
                col_names=small_data["col_names"],
            )

        y_neg = small_data["y_vec"].copy()
        y_neg[0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            PoissonFlow(
                y_neg,
                small_data["G"],
                small_data["X"],
                col_names=small_data["col_names"],
            )

    def test_posterior_predictive_shape_and_integer(self, small_data):
        from bayespecon.models.flow import PoissonFlow

        model = PoissonFlow(
            small_data["y_vec"],
            small_data["G"],
            small_data["X"],
            col_names=small_data["col_names"],
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        y_rep = model.posterior_predictive(n_draws=5, random_seed=3)
        assert y_rep.shape == (5, model._N)
        assert np.all(y_rep >= 0)
        assert np.allclose(y_rep, np.round(y_rep))

    def test_spatial_effects_match_olsflow_pattern(self, small_data):
        """NE = 0 (closed form under A=I_N), keys parity with OLSFlow."""
        from bayespecon.models.flow import PoissonFlow

        model = PoissonFlow(
            small_data["y_vec"],
            small_data["G"],
            small_data["X"],
            col_names=small_data["col_names"],
        )
        model.fit(draws=40, tune=40, chains=1, progressbar=False, random_seed=0)
        df = model.spatial_effects(mode="combined")
        assert np.isclose(df.xs("network", level="effect")["mean"].iloc[0], 0.0)


@pytest.mark.slow
class TestPoissonGravityFlowRecovery:
    """PoissonFlow posterior β should recover DGP coefficients under rho=0."""

    def test_recovers_beta(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonFlow

        beta_d = np.array([0.5, -0.3])
        beta_o = np.array([0.4, 0.2])
        data = generate_poisson_flow_data(
            n=10,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=beta_d,
            beta_o=beta_o,
            seed=42,
        )
        model = PoissonFlow(
            data["y_vec"],
            data["G"],
            data["X"],
            col_names=data["col_names"],
            trace_seed=0,
        )
        idata = model.fit(**POISSON_SAMPLE_KWARGS)
        beta_hat = idata.posterior["beta"].mean(dim=("chain", "draw")).values
        # Layout: [intercept, intra, βd₁, βd₂, βo₁, βo₂] (matches OLSFlow)
        # Compare destination/origin slope blocks within Poisson tolerance.
        bd_hat = beta_hat[2 : 2 + len(beta_d)]
        bo_hat = beta_hat[2 + len(beta_d) : 2 + 2 * len(beta_d)]
        assert np.allclose(bd_hat, beta_d, atol=ABS_TOL_BETA_POI), (
            f"βd: expected {beta_d}, got {bd_hat}"
        )
        assert np.allclose(bo_hat, beta_o, atol=ABS_TOL_BETA_POI), (
            f"βo: expected {beta_o}, got {bo_hat}"
        )


# ---------------------------------------------------------------------------
# Pointwise log-likelihood (with Jacobian correction)
# ---------------------------------------------------------------------------


class TestFlowLogLikelihood:
    """Verify ``log_likelihood`` group is attached and usable for model
    comparison via ``az.loo`` / ``az.waic`` / ``az.compare``."""

    def setup_method(self):
        from bayespecon.dgp.flows import (
            generate_flow_data,
            generate_poisson_flow_data,
        )

        self.n = 5
        self.gauss = generate_flow_data(
            n=self.n,
            rho_d=0.20,
            rho_o=0.20,
            rho_w=0.05,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=0,
        )
        self.G = self.gauss["G"]
        self.poi = generate_poisson_flow_data(n=self.n, k=2, seed=0)

    def _check_loo(self, idata):
        import arviz as az

        assert hasattr(idata, "log_likelihood")
        assert "obs" in idata.log_likelihood.data_vars
        ll = idata.log_likelihood["obs"].values
        assert ll.shape[2] == self.n * self.n  # N obs
        assert np.isfinite(ll).all()
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd_loo)

    def test_sar_flow_loglik(self):
        from bayespecon.models.flow import SARFlow

        m = SARFlow(
            self.gauss["y_vec"],
            self.G,
            self.gauss["X"],
            col_names=self.gauss["col_names"],
            miter=5,
            titer=50,
            trace_seed=0,
            restrict_positive=True,
        )
        idata = m.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        self._check_loo(idata)

    def test_sar_flow_separable_loglik(self):
        from bayespecon.models.flow import SARFlowSeparable

        m = SARFlowSeparable(
            self.gauss["y_vec"],
            self.G,
            self.gauss["X"],
            col_names=self.gauss["col_names"],
            trace_seed=0,
        )
        idata = m.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        self._check_loo(idata)

    def test_ols_flow_loglik(self):
        from bayespecon.models.flow import OLSFlow

        m = OLSFlow(
            self.gauss["y_vec"],
            self.G,
            self.gauss["X"],
            col_names=self.gauss["col_names"],
        )
        idata = m.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        self._check_loo(idata)

    def test_poisson_flow_loglik(self):
        from bayespecon.models.flow import PoissonFlow

        m = PoissonFlow(
            self.poi["y_vec"],
            self.poi["G"],
            self.poi["X"],
            col_names=self.poi["col_names"],
            trace_seed=0,
        )
        idata = m.fit(draws=200, tune=300, chains=1, progressbar=False, random_seed=0)
        # PoissonFlow uses self.poi's own G (not self.G); validate against m._N.
        import arviz as az

        assert hasattr(idata, "log_likelihood")
        ll = idata.log_likelihood["obs"].values
        assert ll.shape[2] == m._N
        assert np.isfinite(ll).all()
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd_loo)

    def test_compare_flow_models(self):
        """az.compare runs across SARFlow/SARFlowSeparable/OLSFlow."""
        import arviz as az

        from bayespecon.models.flow import OLSFlow, SARFlow, SARFlowSeparable

        m_sar = SARFlow(
            self.gauss["y_vec"],
            self.G,
            self.gauss["X"],
            col_names=self.gauss["col_names"],
            miter=5,
            titer=50,
            trace_seed=0,
            restrict_positive=True,
        )
        m_sep = SARFlowSeparable(
            self.gauss["y_vec"],
            self.G,
            self.gauss["X"],
            col_names=self.gauss["col_names"],
            trace_seed=0,
        )
        m_ols = OLSFlow(
            self.gauss["y_vec"],
            self.G,
            self.gauss["X"],
            col_names=self.gauss["col_names"],
        )
        kw = dict(draws=40, tune=40, chains=1, progressbar=False, random_seed=0)
        idata_sar = m_sar.fit(**kw)
        idata_sep = m_sep.fit(**kw)
        idata_ols = m_ols.fit(**kw)
        comp = az.compare({"sar": idata_sar, "sep": idata_sep, "ols": idata_ols})
        assert "rank" in comp.columns
        assert len(comp) == 3

    def test_sar_flow_jacobian_matches_pytensor(self):
        """SARFlow._compute_jacobian_log_det matches pytensor logdet at draws."""
        import pytensor
        import pytensor.tensor as pt

        from bayespecon.logdet import flow_logdet_pytensor
        from bayespecon.models.flow import SARFlow

        m = SARFlow(
            self.gauss["y_vec"],
            self.G,
            self.gauss["X"],
            col_names=self.gauss["col_names"],
            miter=5,
            titer=50,
            trace_seed=0,
            restrict_positive=True,
        )
        idata = m.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        post = idata.posterior
        np_jac = m._compute_jacobian_log_det(post)

        a, b, c = pt.scalar("a"), pt.scalar("b"), pt.scalar("c")
        expr = flow_logdet_pytensor(
            a,
            b,
            c,
            m._poly_a,
            m._poly_b,
            m._poly_c,
            m._poly_coeffs,
            m._miter_a,
            m._miter_b,
            m._miter_c,
            m._miter_coeffs,
            m.miter,
            m.titer,
        )
        fn = pytensor.function([a, b, c], expr)
        rd = post["rho_d"].values.reshape(-1)
        ro = post["rho_o"].values.reshape(-1)
        rw = post["rho_w"].values.reshape(-1)
        pt_jac = np.array([float(fn(x, y, z)) for x, y, z in zip(rd, ro, rw)])
        np.testing.assert_allclose(np_jac, pt_jac, atol=1e-10)
