"""JAX dispatch registrations for the custom Ops in :mod:`bayespecon.ops`.

This module enables JAX-backed NUTS samplers (``"blackjax"``, ``"numpyro"``)
for models that depend on :class:`~bayespecon.ops.SparseFlowSolveOp`,
:class:`~bayespecon.ops.SparseFlowSolveMatrixOp`,
:class:`~bayespecon.ops.KroneckerFlowSolveOp`, and
:class:`~bayespecon.ops.KroneckerFlowSolveMatrixOp`.

The Kronecker Ops are translated into pure-JAX dense LU solves
(:math:`n\\times n`, jittable, vmappable).  The general sparse Ops are wrapped
in :func:`jax.pure_callback` because JAX has no CPU sparse direct solver;
their gradients are routed through the corresponding ``_*VJPOp`` whose JAX
dispatch is also a callback.  PyTensor inserts the VJP node into the symbolic
graph (via ``L_op``) *before* JAX transpilation, so JAX never has to
differentiate across the callback.

Availability is probed via :func:`importlib.util.find_spec`; the registration
function is a no-op when JAX or PyTensor's JAX dispatch module is missing,
so importing this module is always safe.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache


@lru_cache(maxsize=1)
def _jax_available() -> bool:
    """Return ``True`` if JAX and PyTensor's JAX dispatch are importable."""
    return (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("pytensor.link.jax.dispatch") is not None
    )


@lru_cache(maxsize=1)
def register_jax_dispatch() -> bool:
    """Register JAX dispatches for all Ops in :mod:`bayespecon.ops`.

    Idempotent (cached). Returns ``True`` if registration ran, ``False`` if
    JAX is not available.
    """
    if not _jax_available():
        return False

    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsla
    import numpy as np
    from pytensor.link.jax.dispatch import jax_funcify

    from .ops import (
        KroneckerFlowSolveMatrixOp,
        KroneckerFlowSolveOp,
        SparseFlowSolveMatrixOp,
        SparseFlowSolveOp,
        _KroneckerFlowVJPMatrixOp,
        _KroneckerFlowVJPOp,
        _SparseFlowVJPMatrixOp,
        _SparseFlowVJPOp,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dense(W):
        """scipy CSR -> dense float64 numpy array (closed over once)."""
        return np.asarray(W.toarray(), dtype=np.float64)

    def _reshape_F(arr, shape):
        """Equivalent to ``arr.reshape(shape, order='F')`` for 2D targets.

        For a 1D ``arr`` of length ``n*n`` reshaped to ``(n, n)`` Fortran-order,
        this is ``arr.reshape((n, n)).T``.
        """
        # Only used here for 1D -> (n, n)
        n = shape[0]
        return arr.reshape((n, n)).T

    def _ravel_F_2d(M):
        """Equivalent to ``M.ravel(order='F')`` for a 2D array."""
        return M.T.reshape(-1)

    # ------------------------------------------------------------------
    # Kronecker forward — pure JAX
    # ------------------------------------------------------------------

    @jax_funcify.register(KroneckerFlowSolveOp)
    def _funcify_kron_solve(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def kron_solve(rho_d, rho_o, b):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            Hb = _reshape_F(b, (n, n))  # (n, n)
            Hp = jsla.solve(Ld, Hb)  # Ld H' = Hb
            Z = jsla.solve(Lo.T, Hp.T)  # Lo^T Z = Hp^T
            # perform: Z.T.ravel(order='F') == Z.ravel()
            return Z.reshape(-1)

        return kron_solve

    # ------------------------------------------------------------------
    # Kronecker VJP — pure JAX
    # ------------------------------------------------------------------

    @jax_funcify.register(_KroneckerFlowVJPOp)
    def _funcify_kron_vjp(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def kron_vjp(rho_d, rho_o, eta, g):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d

            H_eta = _reshape_F(eta, (n, n))  # (n, n)
            Hg = _reshape_F(g, (n, n))  # (n, n)

            # Adjoint: (Lo^T ⊗ Ld^T) v = g  =>  Ld^T H_v Lo = Hg
            P = jsla.solve(Ld.T, Hg)  # Ld^T P = Hg
            Q = jsla.solve(Lo.T, P.T)  # Lo^T Q = P^T  (Q = H_v^T)
            H_v = Q.T  # (n, n)

            W_H = W_d @ H_eta  # (n, n)
            Ld_H = Ld @ H_eta  # (n, n)
            grad_rd = jnp.sum(H_v * (W_H @ Lo.T))
            grad_ro = jnp.sum(H_v * (Ld_H @ W_d.T))
            grad_b = _ravel_F_2d(H_v)
            return grad_rd, grad_ro, grad_b

        return kron_vjp

    # ------------------------------------------------------------------
    # Kronecker matrix forward / VJP — vmap over the single-vector path
    # ------------------------------------------------------------------

    @jax_funcify.register(KroneckerFlowSolveMatrixOp)
    def _funcify_kron_solve_matrix(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def _solve_one(rho_d, rho_o, b):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            Hb = _reshape_F(b, (n, n))
            Hp = jsla.solve(Ld, Hb)
            Z = jsla.solve(Lo.T, Hp.T)
            return Z.reshape(-1)

        def kron_solve_mat(rho_d, rho_o, B):
            # vmap over the column (T) axis: B is (N, T) -> output (T, N) -> (N, T)
            solver = jax.vmap(_solve_one, in_axes=(None, None, 1), out_axes=1)
            return solver(rho_d, rho_o, B)

        return kron_solve_mat

    @jax_funcify.register(_KroneckerFlowVJPMatrixOp)
    def _funcify_kron_vjp_matrix(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def _vjp_one(rho_d, rho_o, eta_col, g_col):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            H_eta = _reshape_F(eta_col, (n, n))
            Hg = _reshape_F(g_col, (n, n))
            P = jsla.solve(Ld.T, Hg)
            Q = jsla.solve(Lo.T, P.T)
            H_v = Q.T
            W_H = W_d @ H_eta
            Ld_H = Ld @ H_eta
            grad_rd = jnp.sum(H_v * (W_H @ Lo.T))
            grad_ro = jnp.sum(H_v * (Ld_H @ W_d.T))
            grad_b = _ravel_F_2d(H_v)
            return grad_rd, grad_ro, grad_b

        def kron_vjp_mat(rho_d, rho_o, H_eta, G):
            # vmap over column axis; sum scalar grads, stack vector grad
            vjper = jax.vmap(_vjp_one, in_axes=(None, None, 1, 1), out_axes=(0, 0, 1))
            grad_rd_per_t, grad_ro_per_t, grad_B = vjper(rho_d, rho_o, H_eta, G)
            return jnp.sum(grad_rd_per_t), jnp.sum(grad_ro_per_t), grad_B

        return kron_vjp_mat

    # ------------------------------------------------------------------
    # Sparse Ops — wrap scipy splu via jax.pure_callback
    # ------------------------------------------------------------------
    #
    # JAX has no CPU sparse direct solver. We use a host callback that runs
    # the existing perform() logic. Two distinct gradient paths must work:
    #
    # 1. PyTensor's symbolic L_op path inserts the VJP node into the graph
    #    BEFORE JAX transpilation.  Each VJP node has its own callback
    #    dispatch, so JAX never differentiates across the callback here.
    # 2. PyMC's JAX samplers (blackjax, numpyro) compile only the forward
    #    log-density and then call ``jax.grad`` on it.  ``jax.grad`` traces
    #    through ``pure_callback`` and raises ``Pure callbacks do not
    #    support JVP``.  To make this path work we wrap the forward solve
    #    in ``jax.custom_vjp`` — the bwd rule calls the existing analytic
    #    adjoint via another ``pure_callback``.

    def _make_solve_with_custom_vjp(forward_op, vjp_op, *, matrix: bool):
        """Build a ``custom_vjp``-decorated solver that reuses the Op callbacks."""

        def _host_solve(rd, ro, rw, rhs):
            outputs = [[None]]
            forward_op.perform(
                None,
                [np.asarray(rd), np.asarray(ro), np.asarray(rw), np.asarray(rhs)],
                outputs,
            )
            return outputs[0][0]

        def _host_vjp(rd, ro, rw, sol, g):
            outputs = [[None], [None], [None], [None]]
            vjp_op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(sol),
                    np.asarray(g),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        @jax.custom_vjp
        def solve(rho_d, rho_o, rho_w, rhs):
            return jax.pure_callback(
                _host_solve,
                jax.ShapeDtypeStruct(rhs.shape, jnp.float64),
                rho_d,
                rho_o,
                rho_w,
                rhs,
                vmap_method="sequential",
            )

        def solve_fwd(rho_d, rho_o, rho_w, rhs):
            sol = solve(rho_d, rho_o, rho_w, rhs)
            return sol, (rho_d, rho_o, rho_w, sol)

        def solve_bwd(residuals, g):
            rho_d, rho_o, rho_w, sol = residuals
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(sol.shape, jnp.float64),
            )
            grad_rd, grad_ro, grad_rw, grad_rhs = jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                sol,
                g,
                vmap_method="sequential",
            )
            return grad_rd, grad_ro, grad_rw, grad_rhs

        solve.defvjp(solve_fwd, solve_bwd)
        return solve, _host_vjp

    @jax_funcify.register(SparseFlowSolveOp)
    def _funcify_sparse_solve(op, **kwargs):
        solve, _ = _make_solve_with_custom_vjp(op, op._vjp_op, matrix=False)

        def sparse_solve(rho_d, rho_o, rho_w, b):
            return solve(rho_d, rho_o, rho_w, b)

        return sparse_solve

    @jax_funcify.register(_SparseFlowVJPOp)
    def _funcify_sparse_vjp(op, **kwargs):
        # Used by PyTensor's symbolic L_op path. Pure callback is fine
        # here because PyTensor never differentiates through this node
        # (it IS the gradient).
        def _host_vjp(rd, ro, rw, eta, g):
            outputs = [[None], [None], [None], [None]]
            op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(eta),
                    np.asarray(g),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        def sparse_vjp(rho_d, rho_o, rho_w, eta, g):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(eta.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                eta,
                g,
                vmap_method="sequential",
            )

        return sparse_vjp

    @jax_funcify.register(SparseFlowSolveMatrixOp)
    def _funcify_sparse_solve_matrix(op, **kwargs):
        solve, _ = _make_solve_with_custom_vjp(op, op._vjp_op, matrix=True)

        def sparse_solve_mat(rho_d, rho_o, rho_w, B):
            return solve(rho_d, rho_o, rho_w, B)

        return sparse_solve_mat

    @jax_funcify.register(_SparseFlowVJPMatrixOp)
    def _funcify_sparse_vjp_matrix(op, **kwargs):
        def _host_vjp(rd, ro, rw, H, G):
            outputs = [[None], [None], [None], [None]]
            op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(H),
                    np.asarray(G),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        def sparse_vjp_mat(rho_d, rho_o, rho_w, H, G):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(H.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                H,
                G,
                vmap_method="sequential",
            )

        return sparse_vjp_mat

    return True
