"""JAX dispatch parity tests for the custom Ops in :mod:`bayespecon.ops`.

These tests are skipped when JAX is not installed.  They verify that each Op
(forward and VJP) produces numerically identical outputs under the default
PyTensor C backend and the JAX backend, and that the dispatched models can
be sampled with ``nuts_sampler="blackjax"`` without falling back to PyMC.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.requires_jax

pytest.importorskip("jax")

import pytensor
import pytensor.tensor as pt

from bayespecon.ops import (
    KroneckerFlowSolveMatrixOp,
    KroneckerFlowSolveOp,
    SparseFlowSolveMatrixOp,
    SparseFlowSolveOp,
)


def _line_W(n):
    W = sp.lil_matrix((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rows = np.asarray(W.sum(axis=1)).ravel()
    rows[rows == 0] = 1.0
    return sp.diags(1.0 / rows) @ W.tocsr()


@pytest.fixture
def small_W():
    return _line_W(5)


@pytest.fixture
def kron_matrices(small_W):
    n = small_W.shape[0]
    Wd = sp.kron(sp.eye(n), small_W).tocsr()
    Wo = sp.kron(small_W, sp.eye(n)).tocsr()
    Ww = sp.kron(small_W, small_W).tocsr()
    return Wd, Wo, Ww, n


def _compile_pair(inputs, outputs):
    f_c = pytensor.function(inputs, outputs)
    f_j = pytensor.function(inputs, outputs, mode="JAX")
    return f_c, f_j


def _assert_close(c_out, j_out, atol=1e-10):
    if not isinstance(c_out, (list, tuple)):
        c_out = [c_out]
        j_out = [j_out]
    for c, j in zip(c_out, j_out):
        np.testing.assert_allclose(np.asarray(c), np.asarray(j), atol=atol, rtol=1e-10)


def test_kronecker_solve_forward_parity(small_W):
    n = small_W.shape[0]
    op = KroneckerFlowSolveOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, b)
    f_c, f_j = _compile_pair([rho_d, rho_o, b], eta)
    rng = np.random.default_rng(0)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.3, 0.2, bv), f_j(0.3, 0.2, bv))


def test_kronecker_solve_vjp_parity(small_W):
    n = small_W.shape[0]
    op = KroneckerFlowSolveOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, b)]
    f_c, f_j = _compile_pair([rho_d, rho_o, b], grads)
    rng = np.random.default_rng(1)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.4, -0.1, bv), f_j(0.4, -0.1, bv))


def test_kronecker_matrix_forward_parity(small_W):
    n = small_W.shape[0]
    T = 3
    op = KroneckerFlowSolveMatrixOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, B)
    f_c, f_j = _compile_pair([rho_d, rho_o, B], H)
    rng = np.random.default_rng(2)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.25, 0.15, Bv), f_j(0.25, 0.15, Bv))


def test_kronecker_matrix_vjp_parity(small_W):
    n = small_W.shape[0]
    T = 2
    op = KroneckerFlowSolveMatrixOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, B)
    loss = pt.sum(H * H)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, B)]
    f_c, f_j = _compile_pair([rho_d, rho_o, B], grads)
    rng = np.random.default_rng(3)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.2, 0.3, Bv), f_j(0.2, 0.3, Bv))


def test_sparse_flow_forward_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    op = SparseFlowSolveOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, rho_w, b)
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, b], eta)
    rng = np.random.default_rng(4)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.2, 0.15, -0.03, bv), f_j(0.2, 0.15, -0.03, bv))


def test_sparse_flow_vjp_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    op = SparseFlowSolveOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, rho_w, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, rho_w, b)]
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, b], grads)
    rng = np.random.default_rng(5)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.2, 0.15, -0.03, bv), f_j(0.2, 0.15, -0.03, bv))


def test_sparse_flow_matrix_forward_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    T = 3
    op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, rho_w, B)
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, B], H)
    rng = np.random.default_rng(6)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.2, 0.15, -0.03, Bv), f_j(0.2, 0.15, -0.03, Bv))


def test_sparse_flow_matrix_vjp_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    T = 2
    op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, rho_w, B)
    loss = pt.sum(H * H)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, rho_w, B)]
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, B], grads)
    rng = np.random.default_rng(7)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.2, 0.15, -0.03, Bv), f_j(0.2, 0.15, -0.03, Bv))


def test_sampler_resolution_with_jax_present():
    """When JAX is importable, requires_c_backend should not force a downgrade."""
    from bayespecon.models._sampler import _jax_dispatches_available, enforce_c_backend

    assert _jax_dispatches_available() is True
    assert (
        enforce_c_backend("blackjax", requires_c_backend=True, model_name="ToyFlow")
        == "blackjax"
    )
