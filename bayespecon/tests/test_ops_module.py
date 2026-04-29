"""Tests for bayespecon.ops — Kronecker-factored flow solve ops.

Covers:
- Numerical equivalence of KroneckerFlowSolveOp vs reference Kronecker solve.
- Numerical equivalence of KroneckerFlowSolveMatrixOp vs reference.
- VJP correctness via pytensor.gradient.verify_grad.
- logp compile smoke test at a moderate n to catch regressions.
"""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.sparse as sp
from pytensor.gradient import verify_grad

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ring_W(n: int) -> sp.csr_matrix:
    """Row-standardised ring-contiguity weight matrix (n × n)."""
    row = np.concatenate([np.arange(n), np.arange(n)])
    col = np.concatenate([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n])
    data = np.ones(2 * n, dtype=np.float64)
    W = sp.csr_matrix((data, (row, col)), shape=(n, n))
    # row-standardise
    d = np.array(W.sum(axis=1)).ravel()
    W = sp.diags(1.0 / d) @ W
    return W.tocsr()


def _kron_ref(rd, ro, W, b):
    """Reference Kronecker solve: eta = (Lo ⊗ Ld)^{-1} b via dense Kronecker.

    A = I - rho_d (I⊗W) - rho_o (W⊗I) + rho_d*rho_o (W⊗W)
      = (I - rho_o W) ⊗ (I - rho_d W) = Lo ⊗ Ld
    """
    n = W.shape[0]
    I = np.eye(n)
    Ld = I - rd * W.toarray()
    Lo = I - ro * W.toarray()
    A = np.kron(Lo, Ld)
    return np.linalg.solve(A, b)


def _kron_ref_matrix(rd, ro, W, B):
    """Reference Kronecker matrix solve column by column."""
    return np.column_stack([_kron_ref(rd, ro, W, B[:, t]) for t in range(B.shape[1])])


def _flow_weight_mats(
    W: sp.csr_matrix,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Build unrestricted flow weight matrices from an n x n spatial weight matrix."""
    n = W.shape[0]
    I = sp.eye(n, format="csr")
    Wd = sp.kron(I, W, format="csr")
    Wo = sp.kron(W, I, format="csr")
    Ww = sp.kron(W, W, format="csr")
    return Wd, Wo, Ww


def _unrestricted_ref(rd, ro, rw, W, b):
    """Reference unrestricted flow solve using a dense n^2 x n^2 system."""
    Wd, Wo, Ww = _flow_weight_mats(W)
    N = W.shape[0] ** 2
    A = (sp.eye(N, format="csr") - rd * Wd - ro * Wo - rw * Ww).toarray()
    return np.linalg.solve(A, b)


def _unrestricted_ref_matrix(rd, ro, rw, W, B):
    """Reference unrestricted matrix solve column by column."""
    return np.column_stack(
        [_unrestricted_ref(rd, ro, rw, W, B[:, t]) for t in range(B.shape[1])]
    )


# ---------------------------------------------------------------------------
# SparseFlowSolveOp — numerical correctness
# ---------------------------------------------------------------------------


class TestSparseFlowSolveOp:
    @pytest.mark.parametrize("n", [3, 4])
    def test_matches_reference_unrestricted_solve(self, n):
        from bayespecon.ops import SparseFlowSolveOp

        rng = np.random.default_rng(10)
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rd, ro, rw = 0.2, -0.1, 0.05
        b = rng.normal(size=n * n)

        solve_op = SparseFlowSolveOp(Wd, Wo, Ww)
        rd_t = pt.dscalar("rd")
        ro_t = pt.dscalar("ro")
        rw_t = pt.dscalar("rw")
        b_t = pt.dvector("b")
        eta_t = solve_op(rd_t, ro_t, rw_t, b_t)
        f = pytensor.function([rd_t, ro_t, rw_t, b_t], eta_t)

        got = f(rd, ro, rw, b)
        ref = _unrestricted_ref(rd, ro, rw, W, b)
        np.testing.assert_allclose(got, ref, atol=1e-10)


class TestSparseFlowSolveOpVJP:
    def test_vjp_rho_and_rhs(self):
        from bayespecon.ops import SparseFlowSolveOp

        n = 3
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rng = np.random.default_rng(11)
        b_val = rng.normal(size=n * n)
        rd_val = np.float64(0.2)
        ro_val = np.float64(-0.1)
        rw_val = np.float64(0.05)

        solve_op = SparseFlowSolveOp(Wd, Wo, Ww)

        def f_scalar(rd, ro, rw, b):
            return pt.sum(solve_op(rd, ro, rw, b))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, rw_val, b_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# SparseFlowSolveMatrixOp — numerical correctness
# ---------------------------------------------------------------------------


class TestSparseFlowSolveMatrixOp:
    @pytest.mark.parametrize("T", [1, 3])
    def test_matches_reference_unrestricted_matrix_solve(self, T):
        from bayespecon.ops import SparseFlowSolveMatrixOp

        n = 3
        rng = np.random.default_rng(12)
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rd, ro, rw = 0.15, 0.1, -0.05
        B = rng.normal(size=(n * n, T))

        solve_op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        rw_t = pt.dscalar()
        B_t = pt.dmatrix()
        H_t = solve_op(rd_t, ro_t, rw_t, B_t)
        f = pytensor.function([rd_t, ro_t, rw_t, B_t], H_t)

        got = f(rd, ro, rw, B)
        ref = _unrestricted_ref_matrix(rd, ro, rw, W, B)
        np.testing.assert_allclose(got, ref, atol=1e-10)


class TestSparseFlowSolveMatrixOpVJP:
    def test_vjp_rho_and_rhs(self):
        from bayespecon.ops import SparseFlowSolveMatrixOp

        n, T = 3, 2
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rng = np.random.default_rng(13)
        B_val = rng.normal(size=(n * n, T))
        rd_val = np.float64(0.15)
        ro_val = np.float64(0.1)
        rw_val = np.float64(-0.05)

        solve_op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)

        def f_scalar(rd, ro, rw, B):
            return pt.sum(solve_op(rd, ro, rw, B))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, rw_val, B_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# KroneckerFlowSolveOp — numerical correctness
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveOp:
    @pytest.mark.parametrize("n", [4, 6])
    def test_matches_reference_kron_solve(self, n):
        from bayespecon.ops import KroneckerFlowSolveOp

        rng = np.random.default_rng(0)
        W = _ring_W(n)
        rd, ro = 0.3, -0.2
        b = rng.normal(size=n * n)

        solve_op = KroneckerFlowSolveOp(W, n)
        rd_t = pt.dscalar("rd")
        ro_t = pt.dscalar("ro")
        b_t = pt.dvector("b")
        eta_t = solve_op(rd_t, ro_t, b_t)
        f = pytensor.function([rd_t, ro_t, b_t], eta_t)

        got = f(rd, ro, b)
        ref = _kron_ref(rd, ro, W, b)
        np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_output_shape(self):
        from bayespecon.ops import KroneckerFlowSolveOp

        n = 5
        W = _ring_W(n)
        rng = np.random.default_rng(1)
        b = rng.normal(size=n * n)

        solve_op = KroneckerFlowSolveOp(W, n)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        b_t = pt.dvector()
        eta_t = solve_op(rd_t, ro_t, b_t)
        f = pytensor.function([rd_t, ro_t, b_t], eta_t)
        assert f(0.1, -0.1, b).shape == (n * n,)


# ---------------------------------------------------------------------------
# KroneckerFlowSolveOp — VJP (verify_grad)
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveOpVJP:
    def test_vjp_rho_d_rho_o(self):
        """verify_grad checks all inputs via finite differences."""
        from bayespecon.ops import KroneckerFlowSolveOp

        n = 3
        W = _ring_W(n)
        rng = np.random.default_rng(2)
        b_val = rng.normal(size=n * n)
        rd_val = np.float64(0.25)
        ro_val = np.float64(-0.15)

        solve_op = KroneckerFlowSolveOp(W, n)

        def f_scalar(rd, ro, b):
            # Sum to get a scalar for verify_grad
            return pt.sum(solve_op(rd, ro, b))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, b_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# KroneckerFlowSolveMatrixOp — numerical correctness
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveMatrixOp:
    @pytest.mark.parametrize("T", [1, 3])
    def test_matches_reference(self, T):
        from bayespecon.ops import KroneckerFlowSolveMatrixOp

        n = 4
        rng = np.random.default_rng(3)
        W = _ring_W(n)
        rd, ro = 0.2, 0.1
        B = rng.normal(size=(n * n, T))

        solve_op = KroneckerFlowSolveMatrixOp(W, n)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        B_t = pt.dmatrix()
        H_t = solve_op(rd_t, ro_t, B_t)
        f = pytensor.function([rd_t, ro_t, B_t], H_t)

        got = f(rd, ro, B)
        ref = _kron_ref_matrix(rd, ro, W, B)
        np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_output_shape(self):
        from bayespecon.ops import KroneckerFlowSolveMatrixOp

        n, T = 4, 5
        W = _ring_W(n)
        rng = np.random.default_rng(4)
        B = rng.normal(size=(n * n, T))

        solve_op = KroneckerFlowSolveMatrixOp(W, n)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        B_t = pt.dmatrix()
        H_t = solve_op(rd_t, ro_t, B_t)
        f = pytensor.function([rd_t, ro_t, B_t], H_t)
        assert f(0.1, -0.1, B).shape == (n * n, T)


# ---------------------------------------------------------------------------
# KroneckerFlowSolveMatrixOp — VJP (verify_grad)
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveMatrixOpVJP:
    def test_vjp_rho_d_rho_o(self):
        from bayespecon.ops import KroneckerFlowSolveMatrixOp

        n, T = 3, 2
        W = _ring_W(n)
        rng = np.random.default_rng(5)
        B_val = rng.normal(size=(n * n, T))
        rd_val = np.float64(0.2)
        ro_val = np.float64(-0.1)

        solve_op = KroneckerFlowSolveMatrixOp(W, n)

        def f_scalar(rd, ro, B):
            return pt.sum(solve_op(rd, ro, B))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, B_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# Smoke test: logp compiles for PoissonSARFlowSeparable at moderate n
# ---------------------------------------------------------------------------


class TestSeparablePoissonLogpCompiles:
    @pytest.mark.slow
    def test_logp_compiles_n20(self):
        """Check that the Kronecker op compiles inside PyMC at n=20."""
        from bayespecon.graph import flow_weight_matrices
        from bayespecon.models.flow import PoissonSARFlowSeparable

        n = 20
        rng = np.random.default_rng(6)
        G = _ring_W(n)  # use W directly as a proxy; need proper Graph
        pytest.importorskip("libpysal")
        from libpysal.graph import Graph

        focal = np.concatenate([np.arange(n), np.arange(n)])
        neighbor = np.concatenate([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n])
        weights = np.ones(2 * n, dtype=float)
        graph = Graph.from_arrays(focal, neighbor, weights).transform("r")

        y = rng.poisson(5.0, size=(n, n)).astype(np.float64)
        X = rng.normal(size=(n * n, 2))

        model_obj = PoissonSARFlowSeparable(y, graph, X)
        pm_model = model_obj._build_pymc_model()

        # Just test that logp can be evaluated at the initial point
        import pymc as pm

        with pm_model:
            ip = pm_model.initial_point()
            lp = pm_model.compile_logp()(ip)
        assert np.isfinite(lp)
