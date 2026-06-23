"""Sparse backend selection, dense LU fast path, and Kronecker factor helpers."""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from functools import lru_cache

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

# For n in the regime that fits in memory (n^2 weights matrix), calling
# ``scipy.linalg.lu_factor`` (LAPACK ``dgetrf``) on the dense ``L_k = I - rho_k W``
# is several times faster than ``scipy.sparse.linalg.splu``: SuperLU spends
# most of its time in symbolic factorisation overhead at these sizes, whereas
# ``dgetrf`` is a single BLAS-3 kernel.  The forward and adjoint passes share
# the same factorisation (``lu_solve(..., trans=0)`` vs ``trans=1``), so one
# factorisation per Kronecker leg covers both directions.
#
# The threshold below caps the dense path so very large problems still use
# SuperLU.  Tunable via the ``BAYESPECON_KRON_DENSE_MAX`` env var
# (see :mod:`bayespecon._config`).


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
        # Use factorized() for a single LU + batched solve, rather than
        # solving column-by-column which factors A once per column.
        try:
            solve_fn = sp.linalg.factorized(A.tocsc())
            return np.asarray(solve_fn(rhs64), dtype=np.float64)
        except Exception:
            # Fallback: column-by-column if factorized() is unavailable
            # (e.g. UMFPACK not linked).  This is slower but correct.
            umfpack_spsolve = _get_umfpack_spsolve()
            cols = [
                np.asarray(umfpack_spsolve(A.tocsc(), rhs64[:, j]), dtype=np.float64)
                for j in range(rhs64.shape[1])
            ]
            return np.column_stack(cols)
    lu = sp.linalg.splu(A.tocsc())
    return np.asarray(lu.solve(rhs64), dtype=np.float64)


class _FactorizedCallableSolver:
    """Adapter exposing ``solve`` for callables returned by ``factorized``.

    Wraps UMFPACK's ``factorized()`` callable to handle both 1-D vectors
    and 2-D matrix right-hand sides, matching the API of
    :class:`scipy.sparse.linalg.SuperLU`.
    """

    __slots__ = ("_solve_fn",)

    def __init__(self, solve_fn) -> None:
        self._solve_fn = solve_fn

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        if trans != "N":
            raise ValueError("factorized solver adapter supports trans='N' only")
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.ndim == 1:
            return np.asarray(self._solve_fn(rhs), dtype=np.float64)
        # 2-D matrix RHS: solve column-by-column since UMFPACK's factorized()
        # callable does not accept 2-D arrays.
        cols = [
            np.asarray(self._solve_fn(rhs[:, j]), dtype=np.float64)
            for j in range(rhs.shape[1])
        ]
        return np.column_stack(cols)


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
