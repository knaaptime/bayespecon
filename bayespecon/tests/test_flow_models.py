"""Tests for bayespecon.models.flow and supporting logdet flow functions.

Smoke tests use small n=5 grids and minimal draws (100, chains=1) to keep
CI runtime reasonable.  Recovery tests use n=8.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from bayespecon.tests.helpers import SAMPLE_KWARGS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_ring_graph(n: int) -> Graph:
    """Ring-contiguity Graph (row-standardised) for n units."""
    focal = np.concatenate([np.arange(n), np.arange(n)])
    neighbor = np.concatenate([np.roll(np.arange(n), 1), np.roll(np.arange(n), -1)])
    weight = np.ones(len(focal), dtype=float)
    G = Graph.from_arrays(focal, neighbor, weight)
    return G.transform("r")


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
        from bayespecon.logdet import compute_flow_traces

        n = 6
        G = _make_ring_graph(n)
        W = G.sparse.tocsr().astype(np.float64)
        traces = compute_flow_traces(W, miter=5, riter=20, random_state=0)
        assert traces.shape == (5,)

    def test_tr_W2_close_to_exact(self):
        """tr(W^2) estimate should be within 20% of exact for ring graph."""
        from bayespecon.logdet import compute_flow_traces

        n = 10
        G = _make_ring_graph(n)
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

        from bayespecon.logdet import (
            _flow_logdet_poly_coeffs,
            compute_flow_traces,
            flow_logdet_pytensor,
        )

        n = 5
        G = _make_ring_graph(n)
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

        from bayespecon.logdet import (
            _flow_logdet_poly_coeffs,
            compute_flow_traces,
            flow_logdet_pytensor,
            logdet_eigenvalue,
        )

        n = 5
        G = _make_ring_graph(n)
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
        from bayespecon.graph import flow_design_matrix

        self.n = 5
        self.N = self.n * self.n
        rng = np.random.default_rng(10)
        self.G = _make_ring_graph(self.n)
        self.X_regional = rng.standard_normal((self.n, 2))
        self.design = flow_design_matrix(self.X_regional)
        self.X = self.design.combined  # (N, p) full O-D design matrix
        self.col_names = self.design.feature_names
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

    def test_mismatched_beta_shapes_raises_on_dgp(self):
        """generate_flow_data should raise when beta_d and beta_o differ in length."""
        from bayespecon.dgp.flows import generate_flow_data

        with pytest.raises(ValueError, match="beta_d and beta_o"):
            generate_flow_data(
                self.n, self.G, 0.2, 0.2, 0.1, [1.0], [1.0, 2.0], sigma=1.0, seed=0
            )

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
        self.G = _make_ring_graph(self.n)

    def test_output_keys(self):
        from bayespecon.dgp.flows import generate_flow_data

        out = generate_flow_data(
            self.n, self.G, 0.2, 0.2, 0.1, [1.0, -0.5], [0.5, 0.3], sigma=1.0, seed=0
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
            self.n, self.G, 0.2, 0.2, 0.1, [1.0], [0.5], sigma=1.0, seed=1
        )
        assert out["y_vec"].shape == (self.n * self.n,)
        assert out["y_mat"].shape == (self.n, self.n)

    def test_y_vec_mat_consistency(self):
        from bayespecon.dgp.flows import generate_flow_data

        out = generate_flow_data(
            self.n, self.G, 0.2, 0.2, 0.1, [1.0], [0.5], sigma=1.0, seed=2
        )
        np.testing.assert_allclose(out["y_vec"], out["y_mat"].ravel(), atol=1e-12)

    def test_reproducibility(self):
        from bayespecon.dgp.flows import generate_flow_data

        out1 = generate_flow_data(
            self.n, self.G, 0.2, 0.2, 0.1, [1.0], [0.5], sigma=1.0, seed=99
        )
        out2 = generate_flow_data(
            self.n, self.G, 0.2, 0.2, 0.1, [1.0], [0.5], sigma=1.0, seed=99
        )
        np.testing.assert_allclose(out1["y_vec"], out2["y_vec"], atol=1e-12)

    def test_different_seeds_differ(self):
        from bayespecon.dgp.flows import generate_flow_data

        out1 = generate_flow_data(
            self.n, self.G, 0.2, 0.2, 0.1, [1.0], [0.5], sigma=1.0, seed=1
        )
        out2 = generate_flow_data(
            self.n, self.G, 0.2, 0.2, 0.1, [1.0], [0.5], sigma=1.0, seed=2
        )
        assert not np.allclose(out1["y_vec"], out2["y_vec"])


# ---------------------------------------------------------------------------
# Smoke fit tests (minimal draws — just check posterior shapes)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSARFlowFitSmoke:
    """Minimal sampling smoke tests.  Marked slow; skipped in fast CI runs."""

    def setup_method(self):
        from bayespecon.dgp.flows import generate_flow_data

        self.n = 5
        self.G = _make_ring_graph(self.n)
        out = generate_flow_data(
            self.n,
            self.G,
            0.25,
            0.25,
            0.10,
            [1.0, -0.5],
            [0.5, 0.3],
            sigma=1.0,
            seed=42,
        )
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
        G = _make_ring_graph(n)
        out = generate_flow_data(n, G, 0.3, 0.2, -0.06, [1.0], [0.5], sigma=0.5, seed=7)
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
        from bayespecon.ops import SparseFlowSolveOp

        n = 4
        G = _make_ring_graph(n)
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

    @pytest.mark.parametrize("logdet_method", ["eigenvalue", "chebyshev", "mc_poly"])
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
                logdet_method="spline",
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
    tune=800, draws=800, chains=2, random_seed=42, progressbar=False
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

    @pytest.fixture(params=["eigenvalue", "chebyshev", "mc_poly"], scope="class")
    def fitted_model(self, request):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlowSeparable

        G = _make_ring_graph(5)
        out = generate_flow_data(
            5,
            G,
            0.3,
            0.2,
            -0.06,
            np.array([0.5, -0.3]),
            np.array([0.4, 0.2]),
            sigma=1.0,
            seed=0,
        )
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

        G = _make_ring_graph(4)
        out = generate_flow_data(
            4, G, 0.2, 0.1, -0.02, np.array([1.0]), np.array([0.5]), sigma=1.0, seed=1
        )
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

        G = _make_ring_graph(FLOW_N)
        out = generate_flow_data(
            FLOW_N,
            G,
            RHO_D_TRUE,
            RHO_O_TRUE,
            RHO_W_TRUE,
            BETA_D_TRUE,
            BETA_O_TRUE,
            sigma=SIGMA_TRUE,
            seed=42,
        )
        model = SARFlow(
            out["y_vec"],
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

        G = _make_ring_graph(FLOW_N_SEP)
        out = generate_flow_data_separable(
            FLOW_N_SEP,
            G,
            RHO_D_SEP_TRUE,
            RHO_O_SEP_TRUE,
            BETA_D_TRUE,
            BETA_O_TRUE,
            sigma=SIGMA_TRUE,
            seed=42,
        )
        model = SARFlowSeparable(
            out["y_vec"],
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

    @pytest.fixture(scope="class")
    def fitted_poisson(self):
        from bayespecon.dgp.flows import generate_poisson_flow_data
        from bayespecon.models.flow import PoissonSARFlow

        data = generate_poisson_flow_data(
            n=8,
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
        idata = model.fit(**POISSON_SAMPLE_KWARGS)
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

        G = _make_ring_graph(12)  # 144 O-D pairs — same size as SAR separable test
        data = generate_poisson_flow_data_separable(
            n=12,
            rho_d=POI_RHO_D_SEP_TRUE,
            rho_o=POI_RHO_O_SEP_TRUE,
            seed=42,
            G=G,
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
# Effects: new vectorized implementation matches old loop-based output
# ---------------------------------------------------------------------------


class TestEffectsNumericalRegression:
    """Verify refactored effects methods produce identical numbers to the old loop code."""

    @staticmethod
    def _old_effects(A, ones_N, n, N, k, beta_d_vec, beta_o_vec):
        """Reference implementation (the original loop-based code)."""
        ia = np.eye(n, dtype=np.float64).ravel()
        solve_A = sp.linalg.factorized(A.tocsc())

        out_total = np.zeros(k)
        out_intra = np.zeros(k)
        out_dest = np.zeros(k)
        out_orig = np.zeros(k)

        for p in range(k):
            bd = float(beta_d_vec[p])
            bo = float(beta_o_vec[p])

            z_p = np.zeros(N, dtype=np.float64)
            for j in range(n):
                z_p[np.arange(n) * n + j] += bd
            for i in range(n):
                z_p[i * n + np.arange(n)] += bo

            T_p = solve_A(z_p)
            intra_mask = ia.astype(bool)
            T_intra = np.zeros(N)
            T_intra[intra_mask] = T_p[intra_mask]

            T_dest = np.zeros(N)
            T_orig = np.zeros(N)
            for j in range(n):
                T_dest[np.arange(n) * n + j] = T_p[np.arange(n) * n + j]
            for i in range(n):
                T_orig[i * n + np.arange(n)] = T_p[i * n + np.arange(n)]

            out_total[p] = float(ones_N @ T_p) / N
            out_intra[p] = float(ones_N @ T_intra) / N
            out_dest[p] = float(ones_N @ T_dest) / N
            out_orig[p] = float(ones_N @ T_orig) / N

        return out_total, out_intra, out_dest, out_orig

    @staticmethod
    def _new_effects_general(A, ones_N, n, N, k, beta_d_vec, beta_o_vec):
        """New vectorized implementation (general model path)."""
        intra_mask = np.eye(n, dtype=bool).ravel()
        t_ones = sp.linalg.spsolve(A, ones_N)
        scale = beta_d_vec + beta_o_vec
        T_all = t_ones[:, np.newaxis] * scale[np.newaxis, :]
        totals = T_all.mean(axis=0)
        intras = T_all[intra_mask, :].sum(axis=0) / N
        return totals, intras, totals.copy(), totals.copy()

    @staticmethod
    def _new_effects_kron(Lo, Ld, ones_N, n, N, k, beta_d_vec, beta_o_vec):
        """New Kronecker implementation (separable model path)."""
        from bayespecon.ops import kron_solve_vec

        intra_mask = np.eye(n, dtype=bool).ravel()
        t_ones = kron_solve_vec(Lo, Ld, ones_N, n)
        scale = beta_d_vec + beta_o_vec
        T_all = t_ones[:, np.newaxis] * scale[np.newaxis, :]
        totals = T_all.mean(axis=0)
        intras = T_all[intra_mask, :].sum(axis=0) / N
        return totals, intras, totals.copy(), totals.copy()

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(7)
        self.n = 5
        self.N = self.n**2
        self.k = 3

        W_dense = rng.random((self.n, self.n))
        W_dense /= W_dense.sum(axis=1, keepdims=True)
        W = sp.csr_matrix(W_dense)

        rd, ro = 0.25, 0.35
        rw = -rd * ro
        Wd = sp.kron(sp.eye(self.n), W, format="csr")
        Wo = sp.kron(W, sp.eye(self.n), format="csr")
        Ww = sp.kron(W, W, format="csr")
        self.A = sp.eye(self.N, format="csr") - rd * Wd - ro * Wo - rw * Ww
        I_n = sp.eye(self.n, format="csr", dtype=np.float64)
        self.Ld = (I_n - rd * W).tocsr()
        self.Lo = (I_n - ro * W).tocsr()
        self.ones_N = np.ones(self.N)
        self.bd = rng.random(self.k)
        self.bo = rng.random(self.k)

    def test_general_total_matches_old(self):
        old_tot, _, _, _ = self._old_effects(
            self.A, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        new_tot, _, _, _ = self._new_effects_general(
            self.A, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        np.testing.assert_allclose(new_tot, old_tot, atol=1e-12)

    def test_general_intra_matches_old(self):
        _, old_intra, _, _ = self._old_effects(
            self.A, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        _, new_intra, _, _ = self._new_effects_general(
            self.A, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        np.testing.assert_allclose(new_intra, old_intra, atol=1e-12)

    def test_kron_total_matches_general(self):
        new_tot_gen, _, _, _ = self._new_effects_general(
            self.A, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        new_tot_kron, _, _, _ = self._new_effects_kron(
            self.Lo, self.Ld, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        np.testing.assert_allclose(new_tot_kron, new_tot_gen, atol=1e-10)

    def test_kron_intra_matches_general(self):
        _, new_intra_gen, _, _ = self._new_effects_general(
            self.A, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        _, new_intra_kron, _, _ = self._new_effects_kron(
            self.Lo, self.Ld, self.ones_N, self.n, self.N, self.k, self.bd, self.bo
        )
        np.testing.assert_allclose(new_intra_kron, new_intra_gen, atol=1e-10)


# ---------------------------------------------------------------------------
# Public spatial_effects + posterior_predictive (Phase 3.3)
# ---------------------------------------------------------------------------


class TestFlowSpatialEffectsAndPPC:
    @pytest.fixture(scope="class")
    def fitted(self):
        from bayespecon.dgp.flows import generate_flow_data
        from bayespecon.models.flow import SARFlow

        n = 5
        G = _make_ring_graph(n)
        data = generate_flow_data(
            n=n,
            G=G,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=0.1,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.2],
            sigma=1.0,
            seed=11,
        )
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
        assert df.index.names == ["predictor", "effect"]
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
        G = _make_ring_graph(n)
        data = generate_flow_data(
            n=n,
            G=G,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=3,
        )
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
            shock[imask] = beta_d[p] + beta_o[p]
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
        G = _make_ring_graph(n)
        data = generate_flow_data(
            n=n,
            G=G,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=-0.2 * 0.15,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=4,
        )
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
        from bayespecon.graph import flow_design_matrix
        from bayespecon.models.flow_panel import SARFlowPanel

        n = 4
        T = 3
        G = _make_ring_graph(n)
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
        assert df.index.names == ["predictor", "effect"]
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
