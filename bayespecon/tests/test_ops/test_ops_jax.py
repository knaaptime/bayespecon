"""JAX dispatch parity tests for the custom Ops in :mod:`bayespecon.ops`.

These tests are skipped when JAX is not installed.  They verify that each Op
(forward and VJP) produces numerically identical outputs under the default
PyTensor C backend and the JAX backend, and that the dispatched models can
be sampled with ``nuts_sampler="blackjax"`` without falling back to PyMC.
"""

from __future__ import annotations

import warnings

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
    SparseSARSolveOp,
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


def test_klujax_backend_selected_when_installed(monkeypatch):
    pytest.importorskip("klujax")
    from bayespecon._jax_dispatch import _select_jax_sparse_backend

    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_BACKEND", "klujax")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    _select_jax_sparse_backend.cache_clear()
    assert _select_jax_sparse_backend() == "klujax"


def test_sparse_sar_jax_klujax_path_runs(monkeypatch):
    pytest.importorskip("klujax")

    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_BACKEND", "klujax")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")

    W = _line_W(5)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)

    f_c = pytensor.function([rho, b], eta)
    f_j = pytensor.function([rho, b], eta, mode="JAX")

    rng = np.random.default_rng(9)
    b_val = rng.standard_normal(5)

    np.testing.assert_allclose(
        np.asarray(f_c(0.2, b_val)),
        np.asarray(f_j(0.2, b_val)),
        atol=1e-10,
        rtol=1e-10,
    )


def test_jax_auto_prefers_klujax_when_available(monkeypatch):
    from bayespecon._jax_dispatch import _select_jax_sparse_backend

    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_BACKEND", "auto")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "0")
    monkeypatch.setattr("bayespecon._jax_dispatch._klujax_available", lambda: True)
    monkeypatch.setattr("bayespecon._jax_dispatch._umfpack_available", lambda: True)

    _select_jax_sparse_backend.cache_clear()
    assert _select_jax_sparse_backend() == "klujax"


def test_jax_auto_falls_to_callback_when_only_umfpack_available(monkeypatch):
    from bayespecon._jax_dispatch import _select_jax_sparse_backend

    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_BACKEND", "auto")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "0")
    monkeypatch.setattr("bayespecon._jax_dispatch._klujax_available", lambda: False)
    monkeypatch.setattr("bayespecon._jax_dispatch._umfpack_available", lambda: True)

    _select_jax_sparse_backend.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = _select_jax_sparse_backend()
    assert backend == "callback"
    msgs = [str(w.message) for w in caught]
    assert any("callback+umfpack" in m for m in msgs)


def test_jax_auto_falls_to_callback_scipy_when_no_optional_backends(monkeypatch):
    from bayespecon._jax_dispatch import _select_jax_sparse_backend

    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_BACKEND", "auto")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "0")
    monkeypatch.setattr("bayespecon._jax_dispatch._klujax_available", lambda: False)
    monkeypatch.setattr("bayespecon._jax_dispatch._umfpack_available", lambda: False)

    _select_jax_sparse_backend.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = _select_jax_sparse_backend()
    assert backend == "callback"
    msgs = [str(w.message) for w in caught]
    assert any("callback+scipy" in m for m in msgs)
    assert any("scikit-umfpack" in m for m in msgs)


# ---------------------------------------------------------------------------
# Lineax SAR-solver path
# ---------------------------------------------------------------------------


def _reset_jax_dispatch_caches() -> None:
    """Clear the JAX-dispatch selector caches so env changes take effect.

    ``register_jax_dispatch`` is ``lru_cache``-wrapped and re-runs the
    ``jax_funcify.register`` decorators on re-entry, which replaces the
    previously registered dispatcher closures.
    """
    from bayespecon._jax_dispatch import (
        _select_jax_sar_lineax_neumann_k,
        _select_jax_sar_lineax_precond,
        _select_jax_sar_lineax_solver,
        _select_jax_sar_solver,
        _select_jax_sparse_backend,
        register_jax_dispatch,
    )

    _select_jax_sparse_backend.cache_clear()
    _select_jax_sar_solver.cache_clear()
    _select_jax_sar_lineax_solver.cache_clear()
    _select_jax_sar_lineax_precond.cache_clear()
    _select_jax_sar_lineax_neumann_k.cache_clear()
    register_jax_dispatch.cache_clear()


def _setup_jax_gmres_dispatch(monkeypatch):
    """Configure environment for JAX-native GMRES dispatch tests."""
    monkeypatch.setenv("BAYESPECON_JAX_SAR_SOLVER", "jax_gmres")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    from bayespecon._jax_dispatch import register_jax_dispatch

    register_jax_dispatch()


# ---------------------------------------------------------------------------
# JAX-native GMRES SAR-solver path
# ---------------------------------------------------------------------------


def test_jax_gmres_solver_env(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_solver

    monkeypatch.setenv("BAYESPECON_JAX_SAR_SOLVER", "jax_gmres")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    assert _select_jax_sar_solver() == "jax_gmres"


def test_sparse_sar_jax_gmres_forward_parity(monkeypatch, lineax_env_reset):
    """JAX GMRES forward solve must match C-backend reference."""
    _setup_jax_gmres_dispatch(monkeypatch)

    W = _line_W(8)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)

    f_c = pytensor.function([rho, b], eta)
    f_j = pytensor.function([rho, b], eta, mode="JAX")

    rng = np.random.default_rng(31)
    b_val = rng.standard_normal(8)

    np.testing.assert_allclose(
        np.asarray(f_c(0.3, b_val)),
        np.asarray(f_j(0.3, b_val)),
        atol=1e-7,
        rtol=1e-7,
    )


def test_sparse_sar_jax_gmres_grad_parity(monkeypatch, lineax_env_reset):
    """Reverse-mode gradient parity for JAX GMRES path."""
    _setup_jax_gmres_dispatch(monkeypatch)

    W = _line_W(8)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho, b)]

    f_c = pytensor.function([rho, b], grads)
    f_j = pytensor.function([rho, b], grads, mode="JAX")

    rng = np.random.default_rng(32)
    b_val = rng.standard_normal(8)

    c_out = f_c(0.25, b_val)
    j_out = f_j(0.25, b_val)
    for c, j in zip(c_out, j_out):
        np.testing.assert_allclose(np.asarray(c), np.asarray(j), atol=1e-7, rtol=1e-7)


def test_jax_auto_falls_to_chebyshev_when_no_lineax(monkeypatch):
    from bayespecon._jax_dispatch import _resolve_auto_sar_solver

    monkeypatch.setattr("bayespecon._jax_dispatch._klujax_available", lambda: False)
    monkeypatch.setattr("bayespecon._jax_dispatch._lineax_available", lambda: False)
    assert _resolve_auto_sar_solver(100) == "chebyshev"


def test_jax_gmres_high_rho_correctness(monkeypatch, lineax_env_reset):
    """JAX GMRES must match dense reference for moderate-to-high rho."""
    _setup_jax_gmres_dispatch(monkeypatch)

    n = 64
    W = _line_W(n)
    rng = np.random.default_rng(33)
    b_val = rng.standard_normal(n)
    rho_val = 0.85

    A_dense = np.eye(n) - rho_val * W.toarray()
    eta_ref = np.linalg.solve(A_dense, b_val)

    op = SparseSARSolveOp(W)
    rho_pt = pt.dscalar("rho")
    b_pt = pt.dvector("b")
    eta = op(rho_pt, b_pt)
    f_j = pytensor.function([rho_pt, b_pt], eta, mode="JAX")

    out = np.asarray(f_j(rho_val, b_val))
    np.testing.assert_allclose(out, eta_ref, atol=1e-6, rtol=1e-6)


@pytest.fixture
def lineax_env_reset():
    """Reset JAX-dispatch caches before and after each Lineax test."""
    _reset_jax_dispatch_caches()
    yield
    _reset_jax_dispatch_caches()


def test_jax_sar_solver_auto_preserves_existing_backend(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_solver

    monkeypatch.delenv("BAYESPECON_JAX_SAR_SOLVER", raising=False)
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_BACKEND", "auto")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "0")
    monkeypatch.setattr("bayespecon._jax_dispatch._klujax_available", lambda: False)
    monkeypatch.setattr("bayespecon._jax_dispatch._umfpack_available", lambda: True)

    _reset_jax_dispatch_caches()
    # _select_jax_sar_solver returns "auto" when no explicit solver is set;
    # concrete resolution happens in _resolve_auto_sar_solver at Op time.
    assert _select_jax_sar_solver() == "auto"


def test_jax_sar_solver_explicit_lineax(monkeypatch, lineax_env_reset):
    pytest.importorskip("lineax")
    from bayespecon._jax_dispatch import _select_jax_sar_solver

    monkeypatch.setenv("BAYESPECON_JAX_SAR_SOLVER", "lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")

    _reset_jax_dispatch_caches()
    assert _select_jax_sar_solver() == "lineax"


def test_jax_sar_solver_lineax_missing_strict_raises(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_solver

    monkeypatch.setenv("BAYESPECON_JAX_SAR_SOLVER", "lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    monkeypatch.setattr("bayespecon._jax_dispatch._lineax_available", lambda: False)

    _reset_jax_dispatch_caches()
    with pytest.raises(ImportError):
        _select_jax_sar_solver()


def test_jax_sar_lineax_subsolver_default(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_lineax_solver

    monkeypatch.delenv("BAYESPECON_JAX_SAR_LINEAX_SOLVER", raising=False)
    _reset_jax_dispatch_caches()
    assert _select_jax_sar_lineax_solver() == "bicgstab"


def test_jax_sar_lineax_subsolver_gmres(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_lineax_solver

    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_SOLVER", "gmres")
    _reset_jax_dispatch_caches()
    assert _select_jax_sar_lineax_solver() == "gmres"


def _setup_lineax_dispatch(monkeypatch, sub_solver="bicgstab"):
    pytest.importorskip("lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SAR_SOLVER", "lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_SOLVER", sub_solver)
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    from bayespecon._jax_dispatch import register_jax_dispatch

    register_jax_dispatch()


@pytest.mark.parametrize("sub_solver", ["bicgstab", "gmres"])
def test_sparse_sar_jax_lineax_forward_parity(
    monkeypatch, lineax_env_reset, sub_solver
):
    _setup_lineax_dispatch(monkeypatch, sub_solver=sub_solver)

    W = _line_W(8)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)

    f_c = pytensor.function([rho, b], eta)
    f_j = pytensor.function([rho, b], eta, mode="JAX")

    rng = np.random.default_rng(11)
    b_val = rng.standard_normal(8)

    np.testing.assert_allclose(
        np.asarray(f_c(0.3, b_val)),
        np.asarray(f_j(0.3, b_val)),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize("sub_solver", ["bicgstab", "gmres"])
def test_sparse_sar_jax_lineax_grad_parity(monkeypatch, lineax_env_reset, sub_solver):
    """Reverse-mode gradient parity — the key correctness gate."""
    _setup_lineax_dispatch(monkeypatch, sub_solver=sub_solver)

    W = _line_W(8)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho, b)]

    f_c = pytensor.function([rho, b], grads)
    f_j = pytensor.function([rho, b], grads, mode="JAX")

    rng = np.random.default_rng(12)
    b_val = rng.standard_normal(8)

    c_out = f_c(0.25, b_val)
    j_out = f_j(0.25, b_val)
    for c, j in zip(c_out, j_out):
        np.testing.assert_allclose(np.asarray(c), np.asarray(j), atol=1e-7, rtol=1e-7)


def test_sparse_sar_jax_lineax_convergence_failure(monkeypatch, lineax_env_reset):
    """Capping the solver must not raise — ``throw=False`` returns silently.

    With ``throw=False`` inside the Lineax solve, a non-converged or
    near-singular system produces an arithmetic result (often NaN/Inf,
    but never a raised exception). NUTS will reject leapfrog steps with
    non-finite log-prob, which is the desired behaviour at the boundary
    of the stationary region.
    """
    pytest.importorskip("lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SAR_SOLVER", "lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_SOLVER", "bicgstab")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()

    # Monkeypatch the BiCGStab constructor used inside the closure to force
    # max_steps=1 so any non-trivial RHS triggers non-convergence.
    import lineax as lx

    real_bicgstab = lx.BiCGStab

    def _capped(*args, **kwargs):
        kwargs["max_steps"] = 1
        return real_bicgstab(*args, **kwargs)

    monkeypatch.setattr(lx, "BiCGStab", _capped)

    from bayespecon._jax_dispatch import register_jax_dispatch

    register_jax_dispatch()

    # Larger system + rho near 1 → ill-conditioned, exercises the
    # ``throw=False`` "do not raise" contract.
    W = _line_W(64)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)
    f_j = pytensor.function([rho, b], eta, mode="JAX")

    rng = np.random.default_rng(13)
    b_val = rng.standard_normal(64)

    # Must not raise. Output may be NaN/Inf or merely inaccurate; both
    # are acceptable — the contract is only that the program continues.
    out = np.asarray(f_j(0.999, b_val))
    assert out.shape == (64,)


# ---------------------------------------------------------------------------
# Lineax SAR-solver path — Neumann-series preconditioner (Phase D)
# ---------------------------------------------------------------------------


def test_jax_sar_lineax_precond_default(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import (
        _select_jax_sar_lineax_neumann_k,
        _select_jax_sar_lineax_precond,
    )

    monkeypatch.delenv("BAYESPECON_JAX_SAR_LINEAX_PRECOND", raising=False)
    monkeypatch.delenv("BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K", raising=False)
    _reset_jax_dispatch_caches()
    assert _select_jax_sar_lineax_precond() == "neumann"
    assert _select_jax_sar_lineax_neumann_k() == 3


def test_jax_sar_lineax_precond_disabled(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_lineax_precond

    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_PRECOND", "none")
    _reset_jax_dispatch_caches()
    assert _select_jax_sar_lineax_precond() == "none"


def test_jax_sar_lineax_neumann_k_override(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_lineax_neumann_k

    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K", "5")
    _reset_jax_dispatch_caches()
    assert _select_jax_sar_lineax_neumann_k() == 5


def test_jax_sar_lineax_neumann_k_invalid_strict_raises(monkeypatch, lineax_env_reset):
    from bayespecon._jax_dispatch import _select_jax_sar_lineax_neumann_k

    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K", "not-an-int")
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    with pytest.raises(ValueError):
        _select_jax_sar_lineax_neumann_k()


def _setup_lineax_dispatch_precond(
    monkeypatch, *, precond: str, neumann_k: int = 3, sub_solver: str = "bicgstab"
) -> None:
    pytest.importorskip("lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SAR_SOLVER", "lineax")
    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_SOLVER", sub_solver)
    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_PRECOND", precond)
    monkeypatch.setenv("BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K", str(neumann_k))
    monkeypatch.setenv("BAYESPECON_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    from bayespecon._jax_dispatch import register_jax_dispatch

    register_jax_dispatch()


@pytest.mark.parametrize("precond", ["neumann", "none"])
def test_sparse_sar_jax_lineax_precond_forward_parity(
    monkeypatch, lineax_env_reset, precond
):
    """Preconditioned and unpreconditioned paths must agree with C reference."""
    _setup_lineax_dispatch_precond(monkeypatch, precond=precond, neumann_k=3)

    W = _line_W(12)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)

    f_c = pytensor.function([rho, b], eta)
    f_j = pytensor.function([rho, b], eta, mode="JAX")

    rng = np.random.default_rng(21)
    b_val = rng.standard_normal(12)
    np.testing.assert_allclose(
        np.asarray(f_c(0.55, b_val)),
        np.asarray(f_j(0.55, b_val)),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize("precond", ["neumann", "none"])
def test_sparse_sar_jax_lineax_precond_grad_parity(
    monkeypatch, lineax_env_reset, precond
):
    """Preconditioning must not alter the analytic VJP solution."""
    _setup_lineax_dispatch_precond(monkeypatch, precond=precond, neumann_k=3)

    W = _line_W(12)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho, b)]

    f_c = pytensor.function([rho, b], grads)
    f_j = pytensor.function([rho, b], grads, mode="JAX")

    rng = np.random.default_rng(22)
    b_val = rng.standard_normal(12)
    for c, j in zip(f_c(0.55, b_val), f_j(0.55, b_val)):
        np.testing.assert_allclose(np.asarray(c), np.asarray(j), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("rho_val", [0.55, 0.85, 0.97])
def test_sparse_sar_jax_lineax_precond_high_rho_correctness(
    monkeypatch, lineax_env_reset, rho_val
):
    """Neumann-preconditioned solve must match the dense reference for
    :math:`\\rho` ranging from moderate to near-singular.

    This is the primary correctness gate for Phase D: the empirical
    failure mode users hit ("lineax often fails near :math:`\\rho \\to 1`")
    is replaced by a path that stays correct across the full prior
    support. ``max_steps`` is left at the dispatch default (``10 * n``)
    so the preconditioner is responsible for *speed*; this test only
    asserts *correctness* on systems where the bare path is known to
    misbehave.
    """
    _setup_lineax_dispatch_precond(monkeypatch, precond="neumann", neumann_k=3)

    n = 64
    W = _line_W(n)
    rng = np.random.default_rng(int(rho_val * 1000))
    b_val = rng.standard_normal(n)

    A_dense = np.eye(n) - rho_val * W.toarray()
    eta_ref = np.linalg.solve(A_dense, b_val)

    op = SparseSARSolveOp(W)
    rho_pt = pt.dscalar("rho")
    b_pt = pt.dvector("b")
    eta = op(rho_pt, b_pt)
    f_j = pytensor.function([rho_pt, b_pt], eta, mode="JAX")

    out = np.asarray(f_j(rho_val, b_val))
    np.testing.assert_allclose(out, eta_ref, atol=1e-6, rtol=1e-6)
