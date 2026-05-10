"""Custom pytensor Ops for differentiable sparse linear solves.

This module provides two families of differentiable pytensor
:class:`~pytensor.graph.op.Op` classes that wrap
:func:`scipy.sparse.linalg.spsolve` with analytically derived gradients via
the **adjoint method**.

**General (unrestricted) flow models** — :class:`SparseFlowSolveOp` and
:class:`SparseFlowSolveMatrixOp` — handle the full three-parameter system
matrix :math:`A(\\rho_d, \\rho_o, \\rho_w)`.  They require an
:math:`N \\times N` sparse factorisation at every gradient evaluation
(:math:`N = n^2`).

**Separable flow models** — :class:`KroneckerFlowSolveOp` and
:class:`KroneckerFlowSolveMatrixOp` — exploit the constraint
:math:`\\rho_w = -\\rho_d \\rho_o` so that the system matrix factors as

.. math::

    A(\\rho_d, \\rho_o) = L_o \\otimes L_d,
    \\qquad L_k = I_n - \\rho_k W

All solves reduce to two :math:`n \\times n` factorisations, giving an
asymptotic speedup of :math:`O(n^3)` vs :math:`O(n^6)` — roughly
:math:`10{,}000\\times` faster at :math:`n = 100` and making
:math:`n = 1{,}000` feasible where :math:`N \\times N` SuperLU is not.

Background
----------
The system matrix for a spatial flow model with three Kronecker weight matrices
:math:`W_d = I_n \\otimes W`, :math:`W_o = W \\otimes I_n`,
:math:`W_w = W \\otimes W` is:

.. math::

    A(\\rho_d, \\rho_o, \\rho_w)
    = I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w,
    \\qquad N = n^2

Solving :math:`A \\eta = b` (with :math:`b = X\\beta`) gives the spatially
filtered log-mean :math:`\\eta` for the Poisson observation model
:math:`y \\sim \\operatorname{Poisson}(\\exp(\\eta))`.

Separable Kronecker factorisation
----------------------------------
When :math:`\\rho_w = -\\rho_d \\rho_o` the system matrix simplifies:

.. math::

    A = I_N - \\rho_d (I_n \\otimes W) - \\rho_o (W \\otimes I_n)
        + \\rho_d \\rho_o (W \\otimes W)
    = (I_n - \\rho_o W) \\otimes (I_n - \\rho_d W)
    = L_o \\otimes L_d

Note that :math:`W_d = I_n \\otimes W` and :math:`W_o = W \\otimes I_n`, so the
first (left) Kronecker factor contains :math:`\\rho_o` and the second (right)
factor contains :math:`\\rho_d`.

Via the **vec-permutation identity**
:math:`(A \\otimes B)\\operatorname{vec}(X) = \\operatorname{vec}(B X A^\\top)`,
the solve :math:`(L_o \\otimes L_d)\\eta = b` becomes:

.. math::

    L_d H L_o^\\top = B,
    \\qquad B = \\operatorname{mat}(b) \\in \\mathbb{R}^{n \\times n}

which is solved in two steps:

1. :math:`H' = L_d^{-1} B` — one :math:`n \\times n` sparse factorisation
   with :math:`n` right-hand-side columns.
2. :math:`Z = L_o^{-\\top} H'^\\top` — one more.
3. :math:`\\eta = \\operatorname{vec}(Z^\\top)`.

Adjoint gradient
----------------
For a scalar loss :math:`L = L(\\eta)`, the chain rule gives:

.. math::

    \\frac{\\partial L}{\\partial \\rho_k}
    = g^\\top \\frac{\\partial \\eta}{\\partial \\rho_k}
    = g^\\top \\bigl(-A^{-1} \\tfrac{\\partial A}{\\partial \\rho_k} \\eta\\bigr)
    = -v^\\top \\tfrac{\\partial A}{\\partial \\rho_k} \\eta

where :math:`g = \\partial L / \\partial \\eta` is the upstream gradient and

.. math::

    v = (A^\\top)^{-1} g

is the **adjoint solution** — one additional sparse direct solve per gradient
evaluation.  The gradient w.r.t. :math:`b` is:

.. math::

    \\frac{\\partial L}{\\partial b}
    = \\frac{\\partial}{\\partial b}\\bigl(g^\\top A^{-1} b\\bigr)
    = (A^{-1})^\\top g = v

For the Kronecker case :math:`A = L_o \\otimes L_d`, the partial derivatives
of :math:`A` are:

.. math::

    \\frac{\\partial A}{\\partial \\rho_d} = L_o \\otimes (-W), \\qquad
    \\frac{\\partial A}{\\partial \\rho_o} = (-W) \\otimes L_d

so the sensitivities become (using the vec-permutation identity with
:math:`H_v = \\operatorname{mat}(v)`, :math:`H_\\eta = \\operatorname{mat}(\\eta)`):

.. math::

    \\frac{\\partial L}{\\partial \\rho_d}
    = v^\\top (L_o \\otimes W)\\eta
    = \\operatorname{tr}\\!\\left(H_v^\\top W H_\\eta L_o^\\top\\right)
    = \\sum_{ij} (H_v)_{ij}\\,(W H_\\eta L_o^\\top)_{ij}

.. math::

    \\frac{\\partial L}{\\partial \\rho_o}
    = v^\\top (W \\otimes L_d)\\eta
    = \\operatorname{tr}\\!\\left(H_v^\\top L_d H_\\eta W^\\top\\right)
    = \\sum_{ij} (H_v)_{ij}\\,(L_d H_\\eta W^\\top)_{ij}

Both sensitivities reduce to :math:`n \\times n` matrix products — no
:math:`N \\times N` work required.

Cost
----
General ops: **2** :math:`N \\times N` sparse solves + **3** SpMVs per gradient step.

Kronecker ops: **4** :math:`n \\times n` sparse solves
(2 forward + 2 adjoint) + **2** :math:`n \\times n` dense products per
gradient step.  Memory: :math:`O(n^2)` vs :math:`O(N^2) = O(n^4)`.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import os
import threading
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import pytensor.tensor as pt
import scipy.linalg as sla
import scipy.sparse as sp
from pytensor.graph.basic import Apply

# Module-level counter ensures every Op instance gets a unique id so that
# pytensor does not incorrectly merge two distinct Op instances during graph
# optimisation (relevant when multiple flow models exist in one Python session).
_op_id_counter = itertools.count()


# ---------------------------------------------------------------------------
# Callback instrumentation (Phase 2 benchmark metrics)
# ---------------------------------------------------------------------------

_callback_stats_lock = threading.Lock()
_callback_count = 0
_callback_seconds = 0.0
_callback_by_op: dict[str, dict[str, float]] = {}


def reset_callback_stats() -> None:
    """Reset in-process callback counters used by benchmark instrumentation."""
    global _callback_count, _callback_seconds, _callback_by_op
    with _callback_stats_lock:
        _callback_count = 0
        _callback_seconds = 0.0
        _callback_by_op = {}


def get_callback_stats() -> dict[str, object]:
    """Return callback counter snapshot.

    Returns
    -------
    dict
        Keys:
        - ``count`` : total number of instrumented Op ``perform`` calls.
        - ``seconds`` : total wall-clock time spent in those calls.
        - ``by_op`` : per-op breakdown with ``count`` and ``seconds``.
    """
    with _callback_stats_lock:
        by_op_copy = {
            k: {"count": int(v["count"]), "seconds": float(v["seconds"])}
            for k, v in _callback_by_op.items()
        }
        return {
            "count": int(_callback_count),
            "seconds": float(_callback_seconds),
            "by_op": by_op_copy,
        }


def _record_callback(op_name: str, elapsed_seconds: float) -> None:
    global _callback_count, _callback_seconds
    with _callback_stats_lock:
        _callback_count += 1
        _callback_seconds += float(elapsed_seconds)
        bucket = _callback_by_op.setdefault(op_name, {"count": 0, "seconds": 0.0})
        bucket["count"] += 1
        bucket["seconds"] += float(elapsed_seconds)


@contextmanager
def _measure_callback(op_name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _record_callback(op_name, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Dense-LAPACK fast path for the Kronecker family
# ---------------------------------------------------------------------------
#
# For n in the regime that fits in memory (n^2 weights matrix), calling
# ``scipy.linalg.lu_factor`` (LAPACK ``dgetrf``) on the dense ``L_k = I - rho_k W``
# is several times faster than ``scipy.sparse.linalg.splu``: SuperLU spends
# most of its time in symbolic factorisation overhead at these sizes, whereas
# ``dgetrf`` is a single BLAS-3 kernel.  The forward and adjoint passes share
# the same factorisation (``lu_solve(..., trans=0)`` vs ``trans=1``), so one
# factorisation per Kronecker leg covers both directions.
#
# The threshold below caps the dense path so very large problems still use
# SuperLU.  Tunable via the ``BAYESPECON_KRON_DENSE_MAX`` env var.


def _kron_dense_max() -> int:
    """Largest ``n`` for which the Kronecker Ops use dense LAPACK over SuperLU."""
    try:
        return int(os.environ.get("BAYESPECON_KRON_DENSE_MAX", "512"))
    except (TypeError, ValueError):
        return 512


@lru_cache(maxsize=1)
def _umfpack_available() -> bool:
    """Return ``True`` when optional ``scikits.umfpack`` is importable."""
    try:
        return importlib.util.find_spec("scikits.umfpack") is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def _warn_sparse_auto_scipy_fallback_once() -> None:
    """Emit a one-time advisory warning for auto fallback to scipy sparse solve."""
    warnings.warn(
        "BAYESPECON_SPARSE_BACKEND=auto selected scipy sparse solves because optional "
        "dependency 'scikits.umfpack' is not installed. Estimation is likely faster "
        "with the 'scikit-umfpack' package installed.",
        RuntimeWarning,
        stacklevel=3,
    )


@lru_cache(maxsize=1)
def _select_sparse_backend() -> str:
    """Resolve sparse solve backend from env vars with robust fallback.

    Environment
    -----------
    BAYESPECON_SPARSE_BACKEND : {"auto", "scipy", "umfpack"}
        Default ``auto``. ``auto`` prefers ``umfpack`` when available.
    BAYESPECON_SPARSE_STRICT : {"0", "1", "false", "true"}
        If truthy, missing requested optional backends raise ImportError.
    """
    requested = os.environ.get("BAYESPECON_SPARSE_BACKEND", "auto").strip().lower()
    strict = os.environ.get("BAYESPECON_SPARSE_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if requested in {"", "auto"}:
        if _umfpack_available():
            return "umfpack"
        _warn_sparse_auto_scipy_fallback_once()
        return "scipy"

    if requested in {"scipy", "superlu"}:
        return "scipy"

    if requested in {"umfpack", "scikits.umfpack"}:
        if _umfpack_available():
            return "umfpack"
        msg = (
            "BAYESPECON_SPARSE_BACKEND=umfpack requested, but optional dependency "
            "'scikits.umfpack' is not installed. Install 'scikit-umfpack' "
            "for this backend. Falling back to scipy backend."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "scipy"

    msg = (
        f"Unknown BAYESPECON_SPARSE_BACKEND='{requested}'. "
        "Valid values are: auto, scipy, umfpack. Falling back to auto."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "umfpack" if _umfpack_available() else "scipy"


@lru_cache(maxsize=1)
def _get_umfpack_spsolve():
    """Import and return UMFPACK's sparse direct solver."""
    umfpack_mod = importlib.import_module("scikits.umfpack")
    return umfpack_mod.spsolve


def _solve_sparse_vector(A: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A x = rhs`` for vector RHS using configured sparse backend."""
    backend = _select_sparse_backend()
    rhs64 = np.asarray(rhs, dtype=np.float64)
    if backend == "umfpack":
        umfpack_spsolve = _get_umfpack_spsolve()
        return np.asarray(umfpack_spsolve(A.tocsc(), rhs64), dtype=np.float64)
    lu = sp.linalg.splu(A.tocsc())
    return np.asarray(lu.solve(rhs64), dtype=np.float64)


def _solve_sparse_matrix(A: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A X = rhs`` for matrix RHS using configured sparse backend."""
    backend = _select_sparse_backend()
    rhs64 = np.asarray(rhs, dtype=np.float64)
    if backend == "umfpack":
        umfpack_spsolve = _get_umfpack_spsolve()
        cols = [
            np.asarray(umfpack_spsolve(A.tocsc(), rhs64[:, j]), dtype=np.float64)
            for j in range(rhs64.shape[1])
        ]
        return np.column_stack(cols)
    lu = sp.linalg.splu(A.tocsc())
    return np.asarray(lu.solve(rhs64), dtype=np.float64)


class _FactorizedCallableSolver:
    """Adapter exposing ``solve`` for callables returned by ``factorized``."""

    __slots__ = ("_solve_fn",)

    def __init__(self, solve_fn) -> None:
        self._solve_fn = solve_fn

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        if trans != "N":
            raise ValueError("factorized solver adapter supports trans='N' only")
        return np.asarray(self._solve_fn(rhs), dtype=np.float64)


def _make_cached_umfpack_solver(A: sp.spmatrix) -> _FactorizedCallableSolver | None:
    """Build reusable UMFPACK factorized solver when available.

    Returns
    -------
    _FactorizedCallableSolver | None
        Reusable solver for repeated solves with the same matrix, or ``None``
        when a reusable UMFPACK factorization path is unavailable.
    """
    try:
        # Prefer UMFPACK path when scipy exposes the selector.
        use_solver = getattr(sp.linalg, "use_solver", None)
        if callable(use_solver):
            use_solver(useUmfpack=True, assumeSortedIndices=True)
        solve_fn = sp.linalg.factorized(A.tocsc())
        return _FactorizedCallableSolver(solve_fn)
    except Exception:
        return None


class _DenseLU:
    """Lightweight wrapper exposing the same ``solve`` API as ``SuperLU``.

    Holds a LAPACK ``(lu, piv)`` pair from :func:`scipy.linalg.lu_factor`
    and dispatches via :func:`scipy.linalg.lu_solve`.  ``trans="T"`` maps to
    LAPACK ``trans=1`` (transpose, no conjugate, real matrices).
    """

    __slots__ = ("_lu", "_piv")

    def __init__(self, A_dense: np.ndarray) -> None:
        self._lu, self._piv = sla.lu_factor(
            A_dense, overwrite_a=False, check_finite=False
        )

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        t = 1 if trans == "T" else 0
        return sla.lu_solve((self._lu, self._piv), rhs, trans=t, check_finite=False)


def _factor_kron_factor(
    W_dense: np.ndarray,
    W_sparse: sp.csr_matrix,
    rho: float,
    n: int,
    I_dense: np.ndarray | None = None,
):
    """Return an LU factorisation of ``I - rho * W`` using dense LAPACK when small.

    Falls back to ``scipy.sparse.linalg.splu`` for ``n > BAYESPECON_KRON_DENSE_MAX``.
    The returned object exposes ``.solve(rhs, trans=...)`` regardless of path.
    """
    if n <= _kron_dense_max() and W_dense is not None:
        I_ref = I_dense if I_dense is not None else np.eye(n, dtype=np.float64)
        L = I_ref - float(rho) * W_dense
        return _DenseLU(L)
    L_sparse = (
        sp.eye(n, format="csr", dtype=np.float64) - float(rho) * W_sparse
    ).tocsc()
    return sp.linalg.splu(L_sparse)


class _SparseFlowVJPOp(pt.Op):
    r"""Vector-Jacobian product (VJP) for :class:`SparseFlowSolveOp`.

    Computes all four partial derivatives of the scalar loss :math:`L` with
    respect to the inputs :math:`(\\rho_d, \\rho_o, \\rho_w, b)` of the
    forward Op in a single ``perform`` call.

    Algorithm
    ---------
     1. Consume the forward solution :math:`\eta = A^{-1} b` from the
         parent Op output.
     2. **Adjoint solve** :math:`v = (A^\top)^{-1} g`.
    3. **Sensitivity scalars** for each :math:`k \\in \\{d, o, w\\}`:

       .. math::

           \\frac{\\partial L}{\\partial \\rho_k}
           = -v^\\top W_k \\eta

    4. **Gradient w.r.t. the RHS vector** :math:`b`:

       .. math::

           \\frac{\\partial L}{\\partial b} = v

    Parameters
    ----------
    Wd, Wo, Ww : scipy.sparse.csr_matrix
        Kronecker flow weight matrices, shared with the parent
        :class:`SparseFlowSolveOp` instance (not copied).
    """

    __props__ = ("_op_id",)

    def __init__(
        self,
        Wd: sp.csr_matrix,
        Wo: sp.csr_matrix,
        Ww: sp.csr_matrix,
    ) -> None:
        self._Wd = Wd
        self._Wo = Wo
        self._Ww = Ww
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_adjoint(
        self, rd: float, ro: float, rw: float, g: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho)^T v = g`` with lightweight LU cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        g64 = np.asarray(g, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(
                self._cached_solver.solve(g64, trans="T"), dtype=np.float64
            )

        A_t = (
            self._I - rhos[0] * self._Wd.T - rhos[1] * self._Wo.T - rhos[2] * self._Ww.T
        )
        return _solve_sparse_vector(A_t, g64)

    def make_node(self, rho_d, rho_o, rho_w, eta, g):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        eta = pt.as_tensor_variable(eta)
        g = pt.as_tensor_variable(g)
        return Apply(
            self,
            [rho_d, rho_o, rho_w, eta, g],
            [pt.dscalar(), pt.dscalar(), pt.dscalar(), pt.dvector()],
        )

    def perform(self, node, inputs, outputs):
        rd, ro, rw, eta, g = inputs
        eta = np.asarray(eta, dtype=np.float64)
        v = self._solve_adjoint(
            float(rd), float(ro), float(rw), np.asarray(g, dtype=np.float64)
        )
        outputs[0][0] = np.asarray(float(v @ (self._Wd @ eta)), dtype=np.float64)
        outputs[1][0] = np.asarray(float(v @ (self._Wo @ eta)), dtype=np.float64)
        outputs[2][0] = np.asarray(float(v @ (self._Ww @ eta)), dtype=np.float64)
        outputs[3][0] = np.asarray(v, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        # Three scalar outputs + one vector matching b/g shape
        eta_shape = input_shapes[3]
        return [(), (), (), eta_shape]

    def grad(self, inputs, output_grads):
        # Second-order gradients are not required for NUTS (first-order only).
        return [pt.zeros_like(inp) for inp in inputs]


class SparseFlowSolveOp(pt.Op):
    r"""Differentiable sparse solve :math:`\eta = A(\rho)^{-1} b`.

    Wraps :func:`scipy.sparse.linalg.spsolve` as a pytensor
    :class:`~pytensor.graph.op.Op` with analytically exact first-order
    gradients derived via the adjoint method.

    The system matrix is:

    .. math::

        A(\rho_d, \rho_o, \rho_w)
        = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w

    where :math:`W_d = I_n \otimes W`, :math:`W_o = W \otimes I_n`,
    :math:`W_w = W \otimes W` are the Kronecker-product flow weight matrices
    and :math:`N = n^2`.

    This Op is used by :class:`~bayespecon.models.flow.PoissonSARFlow` to embed
    the implicit spatial filter on the **log-mean** of a Poisson observation
    model:

    .. math::

        \eta &= A^{-1} X\beta \\
        \lambda_{ij} &= \exp(\eta_{ij}) \\
        y_{ij} &\sim \operatorname{Poisson}(\lambda_{ij})

    The Jacobian log-determinant :math:`\log|A(\rho)|` is added separately
    via :func:`~bayespecon.logdet.flow_logdet_pytensor` (identical to the
    Gaussian SAR flow model).

    Gradient derivation
    -------------------
    For a scalar loss :math:`L`, implicit differentiation of :math:`A\eta = b`
    gives :math:`dA\, \eta + A\, d\eta = db`, so:

    .. math::

        d\eta = A^{-1}(db - dA\, \eta)

    The VJPs are:

    .. math::

        \frac{\partial L}{\partial \rho_k}
        = g^\top \frac{\partial \eta}{\partial \rho_k}
        = -v^\top W_k \eta,
        \qquad
        \frac{\partial L}{\partial b} = v

    where :math:`v = (A^\top)^{-1} g` and
    :math:`g = \partial L / \partial \eta` is the upstream gradient.
    See :class:`_SparseFlowVJPOp` for the implementation.

    Per-gradient-evaluation cost: **2 sparse direct solves** (SuperLU) +
    3 sparse matrix-vector products.  For :math:`n \leq 100`
    (:math:`N \leq 10^4`) this is fast enough for NUTS sampling.

    Parameters
    ----------
    Wd : scipy.sparse.csr_matrix, shape (N, N)
        Destination weight matrix :math:`W_d = I_n \otimes W`.
    Wo : scipy.sparse.csr_matrix, shape (N, N)
        Origin weight matrix :math:`W_o = W \otimes I_n`.
    Ww : scipy.sparse.csr_matrix, shape (N, N)
        Network weight matrix :math:`W_w = W \otimes W`.

    Examples
    --------
    >>> from bayespecon.ops import SparseFlowSolveOp
    >>> from bayespecon.graph import flow_weight_matrices
    >>> import pytensor.tensor as pt, pytensor
    >>> wms = flow_weight_matrices(G)
    >>> op = SparseFlowSolveOp(wms["destination"], wms["origin"], wms["network"])
    >>> rho_d, rho_o, rho_w = pt.scalars("rho_d", "rho_o", "rho_w")
    >>> b = pt.vector("b")
    >>> eta = op(rho_d, rho_o, rho_w, b)
    >>> fn = pytensor.function([rho_d, rho_o, rho_w, b], eta)
    """

    __props__ = ("_op_id",)

    def __init__(
        self,
        Wd: sp.csr_matrix,
        Wo: sp.csr_matrix,
        Ww: sp.csr_matrix,
    ) -> None:
        self._Wd = Wd.tocsr().astype(np.float64)
        self._Wo = Wo.tocsr().astype(np.float64)
        self._Ww = Ww.tocsr().astype(np.float64)
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._vjp_op = _SparseFlowVJPOp(self._Wd, self._Wo, self._Ww)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_forward(
        self, rd: float, ro: float, rw: float, b: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho) eta = b`` with lightweight LU cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        b64 = np.asarray(b, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        A = self._I - rhos[0] * self._Wd - rhos[1] * self._Wo - rhos[2] * self._Ww
        return _solve_sparse_vector(A, b64)

    def make_node(self, rho_d, rho_o, rho_w, b):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        b = pt.as_tensor_variable(b)
        return Apply(self, [rho_d, rho_o, rho_w, b], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        """Compute :math:`\\eta = A(\\rho)^{-1} b` via a sparse direct solver.

        Uses a single SuperLU factorisation via :func:`scipy.sparse.linalg.splu`.
        """
        rd, ro, rw, b = inputs
        outputs[0][0] = self._solve_forward(
            float(rd), float(ro), float(rw), np.asarray(b, dtype=np.float64)
        )

    def infer_shape(self, fgraph, node, input_shapes):
        # Output has same shape as b
        return [input_shapes[3]]

    def L_op(self, inputs, outputs, output_grads):
        """Compute the VJP via the adjoint method.

        Delegates to :class:`_SparseFlowVJPOp`, which performs:

        1. Forward re-solve: :math:`\\eta = A^{-1} b`.
        2. Adjoint solve: :math:`v = (A^\\top)^{-1} g`.
        3. Sensitivity scalars:
           :math:`\\partial L / \\partial \\rho_k = -v^\\top W_k \\eta`.
        4. Gradient w.r.t. :math:`b`: :math:`v`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho_d, rho_o, rho_w, b]``.
        outputs : list of TensorVariable
            ``[eta]`` (symbolic forward output; not used directly here).
        output_grads : list of TensorVariable
            ``[g]`` where :math:`g = \\partial L / \\partial \\eta`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho_d, grad_rho_o, grad_rho_w, grad_b]``.
        """
        rd, ro, rw, b = inputs
        eta = outputs[0]
        g = output_grads[0]
        grad_rd, grad_ro, grad_rw, grad_b = self._vjp_op(rd, ro, rw, eta, g)
        return [grad_rd, grad_ro, grad_rw, grad_b]


class _SparseFlowVJPMatrixOp(pt.Op):
    """Vector-Jacobian product (VJP) for :class:`SparseFlowSolveMatrixOp`.

    Same adjoint-method derivation as :class:`_SparseFlowVJPOp`, extended to
    a matrix right-hand side :math:`B` of shape :math:`(N, T)`.  One
    LU factorisation covers all :math:`T` columns.

    Algorithm
    ---------
    1. **Forward re-solve** :math:`H = A^{-1} B`, result shape :math:`(N, T)`.
    2. **Adjoint solve** :math:`V = (A^\\top)^{-1} G`, same shape.
    3. **Sensitivity scalars** for each :math:`k \\in \\{d, o, w\\}`:

       .. math::

           \\frac{\\partial L}{\\partial \\rho_k}
           = -\\sum_t v_t^\\top W_k h_t
           = -\\operatorname{sum}(V \\odot (W_k H))

    4. **Gradient w.r.t. B**: :math:`V`.
    """

    __props__ = ("_op_id",)

    def __init__(self, Wd, Wo, Ww):
        self._Wd = Wd
        self._Wo = Wo
        self._Ww = Ww
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_adjoint_matrix(
        self, rd: float, ro: float, rw: float, G: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho)^T V = G`` for matrix RHS with cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        G64 = np.asarray(G, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(
                self._cached_solver.solve(G64, trans="T"), dtype=np.float64
            )

        A_t = (
            self._I - rhos[0] * self._Wd.T - rhos[1] * self._Wo.T - rhos[2] * self._Ww.T
        )
        return _solve_sparse_matrix(A_t, G64)

    def make_node(self, rho_d, rho_o, rho_w, H, G):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        H = pt.as_tensor_variable(H)
        G = pt.as_tensor_variable(G)
        return Apply(
            self,
            [rho_d, rho_o, rho_w, H, G],
            [pt.dscalar(), pt.dscalar(), pt.dscalar(), pt.dmatrix()],
        )

    def perform(self, node, inputs, outputs):
        rd, ro, rw, H, G = inputs
        H = np.asarray(H, dtype=np.float64)
        V = self._solve_adjoint_matrix(
            float(rd), float(ro), float(rw), np.asarray(G, dtype=np.float64)
        )
        outputs[0][0] = np.asarray(np.sum(V * (self._Wd @ H)), dtype=np.float64)
        outputs[1][0] = np.asarray(np.sum(V * (self._Wo @ H)), dtype=np.float64)
        outputs[2][0] = np.asarray(np.sum(V * (self._Ww @ H)), dtype=np.float64)
        outputs[3][0] = np.asarray(V, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        H_shape = input_shapes[3]
        return [(), (), (), H_shape]

    def grad(self, inputs, output_grads):
        return [pt.zeros_like(inp) for inp in inputs]


class SparseFlowSolveMatrixOp(pt.Op):
    r"""Differentiable sparse solve :math:`H = A(\rho)^{-1} B` for matrix RHS.

    Extends :class:`SparseFlowSolveOp` to a matrix right-hand side
    :math:`B \in \mathbb{R}^{N \times T}`, which arises in panel Poisson flow
    models where :math:`T` time periods share the same system matrix
    :math:`A(\rho_d, \rho_o, \rho_w)`.

    One LU factorisation of :math:`A` covers all :math:`T` columns, so the
    cost per gradient evaluation is **2 sparse direct solves** (vs. :math:`2T`
    for a per-period loop).

    Parameters
    ----------
    Wd, Wo, Ww : scipy.sparse.csr_matrix, shape (N, N)
        Kronecker flow weight matrices (shared with parent model; not copied).
    """

    __props__ = ("_op_id",)

    def __init__(self, Wd, Wo, Ww):
        self._Wd = Wd.tocsr().astype(np.float64)
        self._Wo = Wo.tocsr().astype(np.float64)
        self._Ww = Ww.tocsr().astype(np.float64)
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._vjp_op = _SparseFlowVJPMatrixOp(self._Wd, self._Wo, self._Ww)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_forward_matrix(
        self, rd: float, ro: float, rw: float, B: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho) H = B`` for matrix RHS with cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        B64 = np.asarray(B, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(B64), dtype=np.float64)

        A = self._I - rhos[0] * self._Wd - rhos[1] * self._Wo - rhos[2] * self._Ww
        return _solve_sparse_matrix(A, B64)

    def make_node(self, rho_d, rho_o, rho_w, B):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        B = pt.as_tensor_variable(B)
        return Apply(self, [rho_d, rho_o, rho_w, B], [pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        rd, ro, rw, B = inputs
        outputs[0][0] = self._solve_forward_matrix(
            float(rd), float(ro), float(rw), np.asarray(B, dtype=np.float64)
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[3]]

    def L_op(self, inputs, outputs, output_grads):
        rd, ro, rw, B = inputs
        H = outputs[0]
        G = output_grads[0]
        grad_rd, grad_ro, grad_rw, grad_B = self._vjp_op(rd, ro, rw, H, G)
        return [grad_rd, grad_ro, grad_rw, grad_B]


# ---------------------------------------------------------------------------
# Kronecker-factored ops for separable Poisson flow models
# (rho_w = -rho_d * rho_o  =>  A = L_d ⊗ L_o,  L_k = I_n - rho_k * W)
# ---------------------------------------------------------------------------


class _KroneckerFlowVJPOp(pt.Op):
    r"""Vector-Jacobian product for :class:`KroneckerFlowSolveOp`.

    Computes the partial derivatives of a scalar loss :math:`L` with respect
    to the inputs :math:`(\rho_d, \rho_o, b)` of the forward Op using the
    **Kronecker adjoint method** — all arithmetic stays in :math:`n \times n`
    space; no :math:`N \times N` (:math:`N = n^2`) matrix is formed.

    Algorithm
    ---------
    Let :math:`L_k = I_n - \rho_k W`,
    :math:`H_x = \operatorname{mat}(x) \in \mathbb{R}^{n \times n}` denote
    column-major reshaping (``order='F'``), and
    :math:`A = L_o \otimes L_d` the system matrix.

    **1. Forward re-solve** :math:`\eta = (L_o \otimes L_d)^{-1} b`.

    The vec-permutation identity
    :math:`(L_o \otimes L_d)\operatorname{vec}(H) = \operatorname{vec}(L_d H L_o^\top)`
    gives the equivalent dense system :math:`L_d H_\eta L_o^\top = H_b`, solved as:

    .. math::

        H' = L_d^{-1} H_b, \qquad
        Z = L_o^{-\top} H'^{\,\top}, \qquad
        \eta = \operatorname{vec}(Z^{\top})

    (Here :math:`Z = H_\eta^\top`, so :math:`Z^\top = H_\eta`.)

    **2. Adjoint solve** :math:`v = (L_o \otimes L_d)^{-\top} g`.

    Since :math:`(L_o \otimes L_d)^\top = L_o^\top \otimes L_d^\top`,
    the vec-identity gives the dense system :math:`L_d^\top H_v L_o = H_g`:

    .. math::

        P = L_d^{-\top} H_g, \qquad
        Q = L_o^{-\top} P^{\top}, \qquad
        v = \operatorname{vec}(Q^{\top})

    (Here :math:`Q = H_v^\top`, so :math:`Q^\top = H_v`.)

    **3. Sensitivity scalars** via the vec-permutation trace identity.

    For :math:`A = L_o \otimes L_d`:

    .. math::

        \frac{\partial A}{\partial \rho_d} = L_o \otimes (-W), \qquad
        \frac{\partial A}{\partial \rho_o} = (-W) \otimes L_d

    Using :math:`\partial L/\partial \rho_k = -v^\top (\partial A/\partial \rho_k)\eta`
    and :math:`(B \otimes C)\operatorname{vec}(H) = \operatorname{vec}(C H B^\top)`:

    .. math::

        \frac{\partial L}{\partial \rho_d}
        = v^\top (L_o \otimes W)\eta
        = \operatorname{tr}\!\left(H_v^\top W H_\eta L_o^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(W H_\eta L_o^\top)_{ij}

    .. math::

        \frac{\partial L}{\partial \rho_o}
        = v^\top (W \otimes L_d)\eta
        = \operatorname{tr}\!\left(H_v^\top L_d H_\eta W^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(L_d H_\eta W^\top)_{ij}

    **4.** :math:`\partial L / \partial b = v`.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardised spatial weight matrix on the *n* spatial units.
        The Kronecker flow matrices :math:`W_d = I_n \otimes W` and
        :math:`W_o = W \otimes I_n` are implicit.
    n : int
        Number of spatial units.  :math:`N = n^2` is the number of O-D pairs.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        # Cached dense view of W; ~n^2 * 8 bytes (trivial for n <= 500).
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, eta, g):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        eta = pt.as_tensor_variable(eta)
        g = pt.as_tensor_variable(g)
        return Apply(
            self,
            [rho_d, rho_o, eta, g],
            [pt.dscalar(), pt.dscalar(), pt.dvector()],
        )

    def perform(self, node, inputs, outputs):
        rd, ro, eta, g = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)
        Ld = self._I - float(rd) * self._W  # only used for L_d @ H below
        Lo = self._I - float(ro) * self._W  # only used for L_o @ ... below

        H_eta = np.asarray(eta, dtype=np.float64).reshape(n, n, order="F")

        # (Lo⊗Ld)^T = Lo^T⊗Ld^T,  (Lo^T⊗Ld^T) vec(Y) = vec(Ld^T Y Lo)
        # Adjoint solve: v = (Lo ⊗ Ld)^{-T} g
        Hg = g.reshape(n, n, order="F")
        P = np.asarray(
            lu_d.solve(np.asarray(Hg, dtype=np.float64), trans="T"), dtype=np.float64
        )
        Q = np.asarray(lu_o.solve(P.T, trans="T"), dtype=np.float64)
        H_v = Q.T  # (n, n)

        # Sensitivities for A = Lo ⊗ Ld:
        #   dA/drho_d = Lo ⊗ (-W),  dL/drho_d = v^T (Lo⊗W) eta
        #             = tr(H_v^T W H_eta Lo^T) = sum(H_v * (W H_eta) @ Lo^T)
        #   dA/drho_o = (-W) ⊗ Ld,  dL/drho_o = v^T (W⊗Ld) eta
        #             = tr(H_v^T Ld H_eta W^T) = sum(H_v * (Ld H_eta) @ W^T)
        W_H = self._W @ H_eta  # (n, n)
        Ld_H = Ld @ H_eta  # (n, n)
        # sum(H_v * (W_H @ Lo.T)) = sum(H_v * (Lo @ W_H.T).T)  [avoids Lo.toarray()]
        # sum(H_v * (Ld_H @ W.T)) = sum(H_v * (W @ Ld_H.T).T)  [avoids W.toarray()]
        outputs[0][0] = np.asarray(np.sum(H_v * (Lo @ W_H.T).T), dtype=np.float64)
        outputs[1][0] = np.asarray(np.sum(H_v * (self._W @ Ld_H.T).T), dtype=np.float64)
        outputs[2][0] = H_v.ravel(order="F").astype(np.float64)  # v = vec(H_v)

    def infer_shape(self, fgraph, node, input_shapes):
        eta_shape = input_shapes[2]
        return [(), (), eta_shape]

    def grad(self, inputs, output_grads):
        return [pt.zeros_like(inp) for inp in inputs]


class KroneckerFlowSolveOp(pt.Op):
    r"""Differentiable Kronecker-factored solve for separable Poisson flow models.

    Computes :math:`\eta = A(\rho_d, \rho_o)^{-1} b` where the system matrix
    exploits the separability constraint :math:`\rho_w = -\rho_d \rho_o`:

    .. math::

        A(\rho_d, \rho_o)
        = I_N - \rho_d (I_n \otimes W) - \rho_o (W \otimes I_n)
          + \rho_d \rho_o (W \otimes W)
        = (I_n - \rho_o W) \otimes (I_n - \rho_d W)
        = L_o \otimes L_d

    where :math:`N = n^2`, :math:`L_k = I_n - \rho_k W`, :math:`W_d = I_n \otimes W`,
    and :math:`W_o = W \otimes I_n`.  Note the order: the **left** Kronecker
    factor :math:`L_o` is associated with :math:`\rho_o` (origin effect) and the
    **right** factor :math:`L_d` with :math:`\rho_d` (destination effect).

    Algorithm
    ---------
    Via the vec-permutation identity
    :math:`(A \otimes B)\operatorname{vec}(X) = \operatorname{vec}(B X A^\top)`,
    the solve :math:`(L_o \otimes L_d)\eta = b` is equivalent to
    :math:`L_d H L_o^\top = B` where
    :math:`B = \operatorname{mat}(b) \in \mathbb{R}^{n \times n}` uses
    column-major (Fortran) ordering.  This is solved in two steps:

    1. :math:`H' = L_d^{-1} B` — sparse solve with :math:`n` RHS columns.
    2. :math:`Z = L_o^{-\top} H'^{\,\top}` — second sparse solve
       (:math:`Z = H_\eta^\top`).
    3. :math:`\eta = \operatorname{vec}(Z^\top)`.

    Complexity
    ----------
    Each gradient evaluation requires **4** :math:`n \times n` sparse
    factorisations (2 forward + 2 adjoint) plus **2** dense
    :math:`n \times n` matrix products — all :math:`O(n^3)`.

    Compare to :class:`SparseFlowSolveOp` which requires **2**
    :math:`N \times N` (:math:`N = n^2`) factorisations — :math:`O(n^6)`.

    Speedup at representative sizes:

    ======  ===============  ========================
    n       N = n²           Approx. speedup
    ======  ===============  ========================
    10      100              ×100
    50      2 500            ×15 000
    100     10 000           ×250 000
    ======  ===============  ========================

    Gradient derivation
    -------------------
    For a scalar loss :math:`L`, implicit differentiation of
    :math:`(L_o \otimes L_d)\eta = b` and the formula
    :math:`dL/d\rho_k = -v^\top (\partial A/\partial \rho_k) \eta` give:

    .. math::

        \frac{\partial A}{\partial \rho_d} = L_o \otimes (-W), \qquad
        \frac{\partial A}{\partial \rho_o} = (-W) \otimes L_d

    where :math:`v = A^{-\top} g` is the adjoint solution and
    :math:`g = \partial L / \partial \eta`.  Using the vec-permutation identity
    with :math:`H_x = \operatorname{mat}(x)` (column-major reshape):

    .. math::

        \frac{\partial L}{\partial \rho_d}
        = \operatorname{tr}\!\left(H_v^\top W H_\eta L_o^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(W H_\eta L_o^\top)_{ij}

    .. math::

        \frac{\partial L}{\partial \rho_o}
        = \operatorname{tr}\!\left(H_v^\top L_d H_\eta W^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(L_d H_\eta W^\top)_{ij}

    See :class:`_KroneckerFlowVJPOp` for the implementation.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardised spatial weight matrix on the *n* spatial units.
        Only this :math:`n \times n` matrix is stored; the
        :math:`N \times N` Kronecker matrices are never allocated.
    n : int
        Number of spatial units.

    Notes
    -----
    ``rho_w`` is **not** an input to this Op.  The caller declares
    ``rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)`` for trace
    reporting; the Op implicitly uses the factorised form.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._vjp_op = _KroneckerFlowVJPOp(self._W, n)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, b):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        b = pt.as_tensor_variable(b)
        return Apply(self, [rho_d, rho_o, b], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        r"""Compute :math:`\eta = (L_o \otimes L_d)^{-1} b`.

        Applies the two-step Kronecker solve:

        1. :math:`L_d H' = H_b` — sparse solve (``spsolve(Ld, Hb)``).
        2. :math:`L_o^\top Z = H'^\top` — second sparse solve
           (``spsolve(Lo.T, Hp.T)``), yielding :math:`Z = H_\eta^\top`.
        3. :math:`\eta = \operatorname{vec}(Z^\top)` — column-major flatten.
        """
        rd, ro, b = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)

        Hb = b.reshape(n, n, order="F")
        Hp = np.asarray(lu_d.solve(np.asarray(Hb, dtype=np.float64)), dtype=np.float64)
        Z = np.asarray(lu_o.solve(Hp.T, trans="T"), dtype=np.float64)
        outputs[0][0] = Z.T.ravel(order="F").astype(np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[2]]

    def L_op(self, inputs, outputs, output_grads):
        r"""Compute VJPs via the Kronecker adjoint method.

        Delegates to :class:`_KroneckerFlowVJPOp`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho_d, rho_o, b]``.
        outputs : list of TensorVariable
            ``[eta]`` (symbolic; not used directly).
        output_grads : list of TensorVariable
            ``[g]`` where :math:`g = \partial L / \partial \eta`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho_d, grad_rho_o, grad_b]``.
        """
        rd, ro, b = inputs
        eta = outputs[0]
        g = output_grads[0]
        grad_rd, grad_ro, grad_b = self._vjp_op(rd, ro, eta, g)
        return [grad_rd, grad_ro, grad_b]


class _KroneckerFlowVJPMatrixOp(pt.Op):
    r"""Vector-Jacobian product for :class:`KroneckerFlowSolveMatrixOp`.

    Extends :class:`_KroneckerFlowVJPOp` to a matrix right-hand side
    :math:`B \in \mathbb{R}^{N \times T}` (:math:`T` time periods).

    Two sparse factorisations — one for :math:`L_d` and one for :math:`L_o^\top`
    — cover all :math:`T` columns simultaneously via batched
    :math:`(n, nT)` right-hand sides, so cost is still **4** :math:`n \times n`
    factorisations regardless of :math:`T`.

    Algorithm
    ---------
    For :math:`A = L_o \otimes L_d` and each period :math:`t`, the system
    :math:`(L_o \otimes L_d) \eta_t = b_t` is equivalent to
    :math:`L_d H_{\eta,t} L_o^\top = H_{b,t}` (vec-permutation identity).
    The :math:`T` slices are solved in batch:

    1. Pack :math:`B` as :math:`(n, nT)` in column-major order so that
       columns :math:`[tn : (t+1)n]` hold :math:`H_{b,t}`.
       Solve :math:`L_d H' = R_{(n,nT)}` — one factorisation covering all :math:`T`.
    2. Permute slices to build the transposed batch
       :math:`[(H'_t)^\top]_t` as an :math:`(n, nT)` matrix.
       Solve :math:`L_o^\top Z = \mathrm{RHS2}_{(n,nT)}` — second factorisation.
    3. Permute and reshape to :math:`(N, T)`:  column :math:`t` is
       :math:`\operatorname{vec}(Z_t^\top) = \eta_t`.

    The adjoint pass uses the same layout with :math:`L_d^\top` and
    :math:`L_o^\top` instead.

    Sensitivity scalars (summed over all :math:`T` periods):

    .. math::

        \frac{\partial L}{\partial \rho_d}
        = \sum_{t} \operatorname{tr}\!\left(H_{v,t}^\top W H_{\eta,t} L_o^\top\right)
        = \sum_{ijt} (H_v)_{ijt}\,(W H_\eta L_o^\top)_{ijt}

    .. math::

        \frac{\partial L}{\partial \rho_o}
        = \sum_{t} \operatorname{tr}\!\left(H_{v,t}^\top L_d H_{\eta,t} W^\top\right)
        = \sum_{ijt} (H_v)_{ijt}\,(L_d H_\eta W^\top)_{ijt}

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardised spatial weight matrix.
    n : int
        Number of spatial units.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, H_eta, G):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        H_eta = pt.as_tensor_variable(H_eta)
        G = pt.as_tensor_variable(G)
        return Apply(
            self,
            [rho_d, rho_o, H_eta, G],
            [pt.dscalar(), pt.dscalar(), pt.dmatrix()],
        )

    def _kron_solve(self, lu_o, lu_d, rhs, *, transpose_d=False):
        """Two-step Kronecker solve used for both forward and adjoint passes.

        Both ``(Lo⊗Ld) η = B`` (forward) and ``(Lo^T⊗Ld^T) v = G`` (adjoint)
        use one LU factorisation per Kronecker factor. The second step always
        solves against ``Lo^T`` via ``trans='T'`` on the LU of ``Lo``.

        For each period t the system ``L_first H'_t = H_b_t`` is solved
        simultaneously for all T periods via a single (n, n*T) RHS.
        The second step ``Lo_T Z_t = H'_t^T`` is likewise batched.

        Parameters
        ----------
        lu_o : scipy.sparse.linalg.SuperLU
            LU factorisation of ``Lo``.
        lu_d : scipy.sparse.linalg.SuperLU
            LU factorisation of ``Ld``.
        rhs : ndarray, shape (N, T)
            Right-hand-side matrix.
        transpose_d : bool, default False
            If True, solve the first step against ``Ld^T``.

        Returns
        -------
        ndarray, shape (N, T)
            Solution columns ``η_t = H_eta_t.ravel('F')``.
        """
        n, T = self._n, rhs.shape[1]
        # Step 1: L_first H'_t = H_b_t  (batch: (n, n*T) solve)
        R = rhs.reshape(n, n * T, order="F")  # (n, n*T): col t*n+j = col j of H_b_t
        Hp = np.asarray(
            lu_d.solve(
                np.asarray(R, dtype=np.float64), trans="T" if transpose_d else "N"
            ),
            dtype=np.float64,
        )
        Hp3 = Hp.reshape(n, n, T, order="F")  # (n, n, T): Hp3[:,:,t] = H'_t
        # Step 2: Lo_T Z_t = H'_t^T  (batch: (n, n*T) solve)
        # Pack RHS so that col t*n+j = H'_t[j,:] (j-th row of H'_t = j-th col of H'_t^T)
        # Hp3.transpose(2,0,1) shape (T,n,n): result[t,j,:] = H'_t[j,:]
        # C-order reshape to (T*n, n): result_2d[t*n+j, i] = H'_t[j, i]
        # Transpose to (n, T*n): RHS2[:, t*n+j] = H'_t[j, :] ✓
        RHS2 = Hp3.transpose(2, 0, 1).reshape(T * n, n).T  # (n, n*T)
        Z_h = np.asarray(
            lu_o.solve(np.asarray(RHS2, dtype=np.float64), trans="T"), dtype=np.float64
        )
        Z3 = Z_h.reshape(n, n, T, order="F")  # (n, n, T): Z3[:,:,t] = Z_t
        # result[:, t] = Z_t^T.ravel('F') = H_eta_t.ravel('F')
        # Z3.transpose(1,0,2): result[j,i,t] = Z3[i,j,t] = Z_t[i,j]
        # F-order reshape (n,n,T) → (N,T): result[i+n*j, t] = Z3[j,i,t] = Z_t[j,i] = Z_t^T[i,j]
        return Z3.transpose(1, 0, 2).reshape(n * n, T, order="F")  # (N, T)

    def perform(self, node, inputs, outputs):
        rd, ro, H_eta, G = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)
        Ld = self._I - float(rd) * self._W  # used only for sparse matmul below
        Lo = self._I - float(ro) * self._W  # used only for sparse multiply below

        H_eta = np.asarray(H_eta, dtype=np.float64)
        H_v = self._kron_solve(
            lu_o, lu_d, np.asarray(G, dtype=np.float64), transpose_d=True
        )

        # Reshape to (n, n, T) for Kronecker trace sums over all T periods
        T = H_eta.shape[1]
        He = H_eta.reshape(n, n, T, order="F")
        Hv = H_v.reshape(n, n, T, order="F")

        # Sparse matmul: W @ He_t and Ld @ He_t for all T simultaneously
        He_2d = He.reshape(n, n * T)  # (n, n*T)
        W_He = (self._W @ He_2d).reshape(n, n, T)  # (n, n, T)
        Ld_He = (Ld @ He_2d).reshape(n, n, T)  # (n, n, T)

        # sum(Hv * WHe_LoT)  = sum_{jk} Lo[j,k]  * S_d[j,k]
        # sum(Hv * LdHe_WT)  = sum_{jk} W[j,k]   * S_o[j,k]
        # where S[j,k] = sum_{it} XHe[i,k,t] * Hv[i,j,t]
        S_d = np.einsum("ikt,ijt->jk", W_He, Hv)  # (n, n)
        S_o = np.einsum("ikt,ijt->jk", Ld_He, Hv)  # (n, n)

        outputs[0][0] = np.asarray(Lo.multiply(S_d).sum(), dtype=np.float64)
        outputs[1][0] = np.asarray(self._W.multiply(S_o).sum(), dtype=np.float64)
        outputs[2][0] = np.asarray(H_v, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        H_shape = input_shapes[2]
        return [(), (), H_shape]

    def grad(self, inputs, output_grads):
        return [pt.zeros_like(inp) for inp in inputs]


class KroneckerFlowSolveMatrixOp(pt.Op):
    r"""Kronecker-factored solve for separable panel Poisson flow models.

    Extends :class:`KroneckerFlowSolveOp` to a matrix right-hand side
    :math:`B \in \mathbb{R}^{N \times T}` that arises when :math:`T` time
    periods share the same system matrix :math:`A = L_o \otimes L_d`
    (see :class:`KroneckerFlowSolveOp` for the factorisation derivation).

    Two :math:`n \times n` factorisations cover all :math:`T` columns
    simultaneously:

    .. math::

        \eta_t = (L_o \otimes L_d)^{-1} b_t, \quad t = 1, \ldots, T

    where columns :math:`b_t` of :math:`B` are the per-period :math:`X_t\beta`
    vectors stacked as :math:`(N, T)`.

    The batched solve reshapes :math:`B` to :math:`(n, nT)` so that a single
    call to :func:`scipy.sparse.linalg.spsolve` covers all :math:`T` columns
    with one :math:`L_d` factorisation, followed by one :math:`L_o^\top`
    factorisation for the second step.

    Complexity
    ----------
    **4** :math:`n \times n` sparse factorisations per gradient step, regardless
    of :math:`T`.  Compare to :class:`SparseFlowSolveMatrixOp` which requires
    **2** :math:`N \times N` factorisations (:math:`N = n^2`).

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardised spatial weight matrix.  Never stored as
        :math:`N \times N` Kronecker matrices.
    n : int
        Number of spatial units.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._vjp_op = _KroneckerFlowVJPMatrixOp(self._W, n)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, B):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        B = pt.as_tensor_variable(B)
        return Apply(self, [rho_d, rho_o, B], [pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        r"""Compute :math:`H = (L_o \otimes L_d)^{-1} B` for all :math:`T` columns.

        Applies the batched two-step Kronecker solve to
        :math:`B \in \mathbb{R}^{N \times T}` using one :math:`L_d`
        factorisation and one :math:`L_o^\top` factorisation:

        1. :math:`L_d H' = R_{(n,nT)}` — batch all :math:`T` slices
           side-by-side.
        2. :math:`L_o^\top Z = \mathrm{RHS2}_{(n,nT)}` — second batch solve,
           with slices permuted to present transposed :math:`H'_t` columns.
        3. Permute and reshape :math:`Z` back to :math:`(N, T)`.
        """
        rd, ro, B = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)

        # Forward: (Lo⊗Ld) η = b  →  Ld H' Lo^T = H_b
        # Step 1: Ld H' = R;  Step 2: Lo^T Z = H'^T (batched over T)
        result = self._vjp_op._kron_solve(
            lu_o,
            lu_d,
            np.asarray(B, dtype=np.float64),
        )
        outputs[0][0] = np.asarray(result, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[2]]

    def L_op(self, inputs, outputs, output_grads):
        r"""Compute VJPs via the Kronecker adjoint method.

        Delegates to :class:`_KroneckerFlowVJPMatrixOp`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho_d, rho_o, B]``.
        outputs : list of TensorVariable
            ``[H]`` (symbolic; not used directly).
        output_grads : list of TensorVariable
            ``[G]`` where :math:`G = \partial L / \partial H \in \mathbb{R}^{N \times T}`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho_d, grad_rho_o, grad_B]``.
        """
        rd, ro, B = inputs
        H = outputs[0]
        G = output_grads[0]
        grad_rd, grad_ro, grad_B = self._vjp_op(rd, ro, H, G)
        return [grad_rd, grad_ro, grad_B]


# ---------------------------------------------------------------------------
# Cross-sectional SAR sparse solve Op (single rho parameter)
# ---------------------------------------------------------------------------


class _SparseSARVJPOp(pt.Op):
    r"""Vector-Jacobian product for :class:`SparseSARSolveOp`.

    Computes partial derivatives of a scalar loss :math:`L` with respect to
    the inputs :math:`(\rho, b)` of the forward Op using the adjoint method.

    Algorithm
    ---------
    1. **Adjoint solve** :math:`v = (I - \rho W^\top)^{-1} g`.
    2. **Sensitivity scalar** for :math:`\rho`:

       .. math::

           \frac{\partial L}{\partial \rho}
           = v^\top W \eta

    3. **Gradient w.r.t.** :math:`b`: :math:`v`.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardised spatial weight matrix.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = W.shape[0]
        self._I = sp.eye(self._n, format="csr", dtype=np.float64)
        self._W_dense = W.toarray() if self._n <= _kron_dense_max() else None
        self._cached_rho: float | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_adjoint(self, rho_val: float, g: np.ndarray) -> np.ndarray:
        """Solve ``(I - rho W^T) v = g`` with lightweight factor cache reuse."""
        n = self._n
        rho_f = float(rho_val)
        g64 = np.asarray(g, dtype=np.float64)

        if n <= _kron_dense_max() and self._W_dense is not None:
            if self._cached_backend != "dense" or self._cached_rho != rho_f:
                A_T_dense = np.eye(n, dtype=np.float64) - rho_f * self._W_dense.T
                self._cached_solver = _DenseLU(A_T_dense)
                self._cached_backend = "dense"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        backend = _select_sparse_backend()
        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rho != rho_f:
                A_T = self._I - rho_f * self._W.transpose()
                self._cached_solver = sp.linalg.splu(A_T.tocsc())
                self._cached_backend = "scipy"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        if (
            self._cached_backend == "umfpack"
            and self._cached_rho == rho_f
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        A_T = self._I - rho_f * self._W.transpose()
        cached_solver = _make_cached_umfpack_solver(A_T)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = "umfpack"
            self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        return _solve_sparse_vector(A_T, g64)

    def make_node(self, rho, eta, g):
        rho = pt.as_tensor_variable(rho)
        eta = pt.as_tensor_variable(eta)
        g = pt.as_tensor_variable(g)
        return Apply(self, [rho, eta, g], [pt.dscalar(), pt.dvector()])

    def perform(self, node, inputs, outputs):
        rho_val, eta, g = inputs
        # Adjoint solve: v = (I - rho * W^T)^{-1} g
        # For symmetric-like W (queen contiguity), W^T ≈ W, but we use W^T for correctness.
        v = self._solve_adjoint(float(rho_val), np.asarray(g, dtype=np.float64))

        eta = np.asarray(eta, dtype=np.float64)
        # dL/drho = v^T W eta
        W_eta = self._W @ eta
        outputs[0][0] = np.asarray(float(v @ W_eta), dtype=np.float64)
        # dL/db = v
        outputs[1][0] = np.asarray(v, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        eta_shape = input_shapes[1]
        return [(), eta_shape]

    def grad(self, inputs, output_grads):
        # Second-order gradients not required for NUTS.
        return [pt.zeros_like(inp) for inp in inputs]


class SparseSARSolveOp(pt.Op):
    r"""Differentiable sparse solve :math:`\eta = (I - \rho W)^{-1} b`.

    Wraps :func:`scipy.sparse.linalg.splu` as a pytensor
    :class:`~pytensor.graph.op.Op` with analytically exact first-order
    gradients derived via the adjoint method.

    The system matrix is:

    .. math::

        A(\rho) = I_n - \rho W

    where :math:`W` is a row-standardised spatial weight matrix.

    This Op is used by :class:`~bayespecon.models.sar_negbin.SARNegativeBinomial`
    to embed the SAR-in-mean reduced form on the **log-mean** of a
    Negative Binomial observation model:

    .. math::

        \eta &= (I - \rho W)^{-1} X\beta \\
        \mu_i &= \exp(\eta_i) \\
        y_i &\sim \operatorname{NegBin}(\mu_i, \alpha)

    The Jacobian log-determinant :math:`\log|I - \rho W|` is added separately
    via the model's ``_logdet_pytensor_fn``.

    Gradient derivation
    -------------------
    For a scalar loss :math:`L`, implicit differentiation of
    :math:`(I - \rho W)\eta = b` gives:

    .. math::

        \frac{\partial L}{\partial \rho}
        = g^\top \frac{\partial \eta}{\partial \rho}
        = g^\top (I - \rho W)^{-1} W \eta
        = v^\top W \eta

    .. math::

        \frac{\partial L}{\partial b}
        = g^\top (I - \rho W)^{-1}
        = v

    where :math:`v = (I - \rho W^\top)^{-1} g` is the **adjoint solution**
    and :math:`g = \partial L / \partial \eta` is the upstream gradient.

    Per-gradient-evaluation cost: **2 sparse direct solves** (one forward,
    one adjoint) + 1 sparse matrix-vector product.  For queen-contiguity
    :math:`W` with :math:`n \leq 10{,}000` this is fast enough for NUTS
    sampling.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardised spatial weight matrix.

    Examples
    --------
    >>> from bayespecon.ops import SparseSARSolveOp
    >>> import pytensor.tensor as pt, pytensor
    >>> op = SparseSARSolveOp(W_csr)
    >>> rho = pt.scalar("rho")
    >>> b = pt.vector("b")
    >>> eta = op(rho, b)
    >>> fn = pytensor.function([rho, b], eta)
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = W.shape[0]
        self._I = sp.eye(self._n, format="csr", dtype=np.float64)
        self._W_dense = W.toarray() if self._n <= _kron_dense_max() else None
        # Pre-allocate dense identity once to avoid repeated np.eye() calls
        # during NUTS sampling when rho changes frequently.
        self._I_dense = (
            np.eye(self._n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._cached_rho: float | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._vjp_op = _SparseSARVJPOp(self._W)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_forward(self, rho_val: float, b: np.ndarray) -> np.ndarray:
        """Solve ``(I - rho W) eta = b`` with lightweight factor cache reuse."""
        n = self._n
        rho_f = float(rho_val)
        b64 = np.asarray(b, dtype=np.float64)

        if n <= _kron_dense_max() and self._W_dense is not None:
            if self._cached_backend != "dense" or self._cached_rho != rho_f:
                A_dense = self._I_dense - rho_f * self._W_dense
                self._cached_solver = _DenseLU(A_dense)
                self._cached_backend = "dense"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        backend = _select_sparse_backend()
        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rho != rho_f:
                A = self._I - rho_f * self._W
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        if (
            self._cached_backend == "umfpack"
            and self._cached_rho == rho_f
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        A = self._I - rho_f * self._W
        cached_solver = _make_cached_umfpack_solver(A)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = "umfpack"
            self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        return _solve_sparse_vector(A, b64)

    def make_node(self, rho, b):
        rho = pt.as_tensor_variable(rho)
        b = pt.as_tensor_variable(b)
        return Apply(self, [rho, b], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        r"""Compute :math:`\eta = (I - \rho W)^{-1} b` via a sparse direct solver."""
        rho_val, b = inputs
        outputs[0][0] = self._solve_forward(
            float(rho_val), np.asarray(b, dtype=np.float64)
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[1]]

    def L_op(self, inputs, outputs, output_grads):
        r"""Compute VJPs via the adjoint method.

        Delegates to :class:`_SparseSARVJPOp`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho, b]``.
        outputs : list of TensorVariable
            ``[eta]`` (symbolic forward output; not used directly here).
        output_grads : list of TensorVariable
            ``[g]`` where :math:`g = \partial L / \partial \eta`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho, grad_b]``.
        """
        rho, b = inputs
        eta = outputs[0]
        g = output_grads[0]
        grad_rho, grad_b = self._vjp_op(rho, eta, g)
        return [grad_rho, grad_b]


def _instrument_op_perform_class(op_cls: type[pt.Op], name: str) -> None:
    """Wrap ``perform`` to record callback timing/counters for benchmarks."""
    original = getattr(op_cls, "perform", None)
    if original is None or getattr(original, "_callback_instrumented", False):
        return

    def _wrapped(self, node, inputs, outputs):
        with _measure_callback(name):
            return original(self, node, inputs, outputs)

    _wrapped._callback_instrumented = True  # type: ignore[attr-defined]
    setattr(op_cls, "perform", _wrapped)


for _cls_name in (
    "_SparseFlowVJPOp",
    "SparseFlowSolveOp",
    "_SparseFlowVJPMatrixOp",
    "SparseFlowSolveMatrixOp",
    "_KroneckerFlowVJPOp",
    "KroneckerFlowSolveOp",
    "_KroneckerFlowVJPMatrixOp",
    "KroneckerFlowSolveMatrixOp",
    "_SparseSARVJPOp",
    "SparseSARSolveOp",
):
    _cls = globals().get(_cls_name)
    if _cls is not None:
        _instrument_op_perform_class(_cls, _cls_name)


# ---------------------------------------------------------------------------
# Standalone Kronecker solve utilities (no PyMC dependency)
# ---------------------------------------------------------------------------


def kron_solve_vec(
    Lo: sp.csr_matrix,
    Ld: sp.csr_matrix,
    b: np.ndarray,
    n: int,
) -> np.ndarray:
    r"""Solve :math:`(L_o \otimes L_d)\,\eta = b` via two :math:`n \times n` sparse solves.

    Uses the vec-permutation identity
    :math:`(L_o \otimes L_d)\operatorname{vec}(H) = \operatorname{vec}(L_d H L_o^\top)`:

    1. :math:`H' = L_d^{-1} H_b`
    2. :math:`Z  = L_o^{-1} H'^{\,\top}` (i.e. solve :math:`L_o Z = H'^\top`)
    3. :math:`\eta = \operatorname{vec}(Z^\top)`

    Parameters
    ----------
    Lo, Ld : scipy.sparse.csr_matrix, shape (n, n)
        Factor matrices :math:`L_o = I_n - \rho_o W` and
        :math:`L_d = I_n - \rho_d W`.
    b : ndarray, shape (N,) where :math:`N = n^2`
    n : int
        Number of spatial units.

    Returns
    -------
    eta : ndarray, shape (N,)
    """
    Hb = b.reshape(n, n, order="F")
    Hp = sp.linalg.spsolve(Ld, Hb)
    Z = sp.linalg.spsolve(Lo, Hp.T)
    return np.asarray(Z, dtype=np.float64).T.ravel(order="F")


def kron_solve_matrix(
    Lo: sp.csr_matrix,
    Ld: sp.csr_matrix,
    B: np.ndarray,
    n: int,
) -> np.ndarray:
    r"""Solve :math:`(L_o \otimes L_d)\,H = B` for a matrix RHS via batched two-step solve.

    Applies the same Kronecker algorithm as :func:`kron_solve_vec` to all
    *k* columns of *B* simultaneously using a single :math:`L_d` factorisation
    and a single :math:`L_o^\top` factorisation (both of size :math:`n \times n`).

    Parameters
    ----------
    Lo, Ld : scipy.sparse.csr_matrix, shape (n, n)
    B : ndarray, shape (N, k) where :math:`N = n^2`
    n : int

    Returns
    -------
    H : ndarray, shape (N, k)
    """
    k = B.shape[1]
    R = B.reshape(n, n * k, order="F")
    Hp = sp.linalg.spsolve(Ld, R)
    Hp3 = Hp.reshape(n, n, k, order="F")
    RHS2 = Hp3.transpose(2, 0, 1).reshape(k * n, n).T
    Z_h = sp.linalg.spsolve(Lo, RHS2)
    Z3 = Z_h.reshape(n, n, k, order="F")
    return np.asarray(
        Z3.transpose(1, 0, 2).reshape(n * n, k, order="F"), dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Numba dispatch registration (no-op when Numba is not installed)
# ---------------------------------------------------------------------------
from ._numba_dispatch import register_numba_dispatch as _register_numba_dispatch

_register_numba_dispatch()


# ---------------------------------------------------------------------------
# JAX dispatch registration (no-op when JAX is not installed)
# ---------------------------------------------------------------------------
from ._jax_dispatch import register_jax_dispatch as _register_jax_dispatch

_register_jax_dispatch()
