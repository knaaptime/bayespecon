"""Numba dispatch registrations for selected custom Ops in :mod:`bayespecon.ops`.

This module currently targets the cross-sectional SAR sparse solve family:
:class:`~bayespecon.ops.SparseSARSolveOp` and
:class:`~bayespecon.ops._SparseSARVJPOp`.

When a dense ``W`` view is available (small ``n``), dispatch uses pure
Numba nopython kernels based on ``np.linalg.solve``.  Otherwise, dispatch
falls back to explicit ``numba.objmode`` wrappers around each Op's existing
``perform`` implementation; this avoids PyTensor's generic fallback warning
while preserving correctness on sparse SuperLU/UMFPACK paths.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache


@lru_cache(maxsize=1)
def _numba_available() -> bool:
    """Return ``True`` if Numba and PyTensor's Numba dispatch are importable."""
    return (
        importlib.util.find_spec("numba") is not None
        and importlib.util.find_spec("pytensor.link.numba.dispatch.basic") is not None
    )


@lru_cache(maxsize=1)
def register_numba_dispatch() -> bool:
    """Register Numba dispatches for SAR sparse solve Ops.

    Idempotent (cached). Returns ``True`` if registration ran, ``False`` if
    Numba dispatch is not available.
    """
    if not _numba_available():
        return False

    import numba
    import numpy as np
    from pytensor.link.numba.dispatch.basic import numba_funcify

    from .ops import SparseSARSolveOp, _SparseSARVJPOp

    @numba_funcify.register(SparseSARSolveOp)
    def _funcify_sparse_sar_solve(op, **kwargs):
        n = op._n
        use_dense = op._W_dense is not None

        if use_dense:
            W_dense = np.asarray(op._W_dense, dtype=np.float64)
            I = np.eye(n, dtype=np.float64)

            @numba.njit
            def sparse_sar_solve(rho, b):
                A = I - rho * W_dense
                return np.linalg.solve(A, b)

            return sparse_sar_solve

        ret_sig = numba.types.float64[:]

        def _py_solve(rho, b):
            outputs = [[None]]
            op.perform(None, [np.asarray(rho), np.asarray(b)], outputs)
            return outputs[0][0]

        @numba.njit
        def sparse_sar_solve(rho, b):
            with numba.objmode(ret=ret_sig):
                ret = _py_solve(rho, b)
            return ret

        return sparse_sar_solve

    @numba_funcify.register(_SparseSARVJPOp)
    def _funcify_sparse_sar_vjp(op, **kwargs):
        n = op._n
        use_dense = op._W_dense is not None

        if use_dense:
            W_dense = np.asarray(op._W_dense, dtype=np.float64)
            I = np.eye(n, dtype=np.float64)

            @numba.njit
            def sparse_sar_vjp(rho, eta, g):
                A_t = (I - rho * W_dense).T
                v = np.linalg.solve(A_t, g)
                grad_rho = np.dot(v, W_dense @ eta)
                return grad_rho, v

            return sparse_sar_vjp

        ret_sig = numba.types.Tuple((numba.types.float64, numba.types.float64[:]))

        def _py_vjp(rho, eta, g):
            outputs = [[None], [None]]
            op.perform(None, [np.asarray(rho), np.asarray(eta), np.asarray(g)], outputs)
            return outputs[0][0], outputs[1][0]

        @numba.njit
        def sparse_sar_vjp(rho, eta, g):
            with numba.objmode(ret=ret_sig):
                ret = _py_vjp(rho, eta, g)
            return ret

        return sparse_sar_vjp

    return True
