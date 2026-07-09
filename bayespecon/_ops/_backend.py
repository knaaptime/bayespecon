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
    """Return ``True`` when ``sksparse.umfpack`` (scikit-sparse) is importable."""
    try:
        return importlib.util.find_spec("sksparse.umfpack") is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def _klu_available() -> bool:
    """Return ``True`` when ``sksparse.klu`` (scikit-sparse) is importable."""
    try:
        return importlib.util.find_spec("sksparse.klu") is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def _warn_sparse_auto_scipy_fallback_once() -> None:
    """Emit a one-time advisory warning for auto fallback to scipy sparse solve."""
    warnings.warn(
        "BAYESPECON_SPARSE_BACKEND=auto selected scipy sparse solves because neither "
        "KLU nor UMFPACK (from 'scikit-sparse') is available. Estimation is typically "
        "faster with 'scikit-sparse' installed.",
        RuntimeWarning,
        stacklevel=3,
    )


@lru_cache(maxsize=1)
def _select_sparse_backend() -> str:
    """Resolve sparse solve backend from env vars with robust fallback.

    Environment
    -----------
    BAYESPECON_SPARSE_BACKEND : {"auto", "scipy", "klu", "umfpack"}
        Default ``auto``. ``auto`` prefers ``klu`` (fastest for the
        structured ``I - rho W`` systems), then falls back to scipy's
        SuperLU.  ``klu`` and ``umfpack`` are provided by ``scikit-sparse``.
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
        if _klu_available():
            return "klu"
        if _umfpack_available():
            return "umfpack"
        _warn_sparse_auto_scipy_fallback_once()
        return "scipy"

    if requested in {"scipy", "superlu"}:
        return "scipy"

    if requested == "klu":
        if _klu_available():
            return "klu"
        msg = (
            "BAYESPECON_SPARSE_BACKEND=klu requested, but 'sksparse.klu' is not "
            "available. Install 'scikit-sparse' for this backend. Falling back to "
            "scipy backend."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "scipy"

    if requested in {"umfpack", "sksparse.umfpack"}:
        if _umfpack_available():
            return "umfpack"
        msg = (
            "BAYESPECON_SPARSE_BACKEND=umfpack requested, but 'sksparse.umfpack' is "
            "not available. Install 'scikit-sparse' for this backend. Falling back to "
            "scipy backend."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "scipy"

    msg = (
        f"Unknown BAYESPECON_SPARSE_BACKEND='{requested}'. "
        "Valid values are: auto, scipy, klu, umfpack. Falling back to auto."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    if _klu_available():
        return "klu"
    return "umfpack" if _umfpack_available() else "scipy"


@lru_cache(maxsize=1)
def _get_klu_factor():
    """Import and return ``sksparse.klu.klu_factor``."""
    return importlib.import_module("sksparse.klu").klu_factor


@lru_cache(maxsize=1)
def _get_umf_factor():
    """Import and return ``sksparse.umfpack.umf_factor``."""
    return importlib.import_module("sksparse.umfpack").umf_factor


def _sparse_factor(A_csc, backend: str):
    """Factorise ``A_csc`` with the requested KLU/UMFPACK backend."""
    if backend == "klu":
        return _get_klu_factor()(A_csc)
    return _get_umf_factor()(A_csc)


def _solve_sparse_vector(A: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A x = rhs`` for vector RHS using configured sparse backend."""
    backend = _select_sparse_backend()
    rhs64 = np.asarray(rhs, dtype=np.float64)
    if backend in ("klu", "umfpack"):
        factor = _sparse_factor(A.tocsc(), backend)
        return np.asarray(factor.solve(rhs64), dtype=np.float64)
    lu = sp.linalg.splu(A.tocsc())
    return np.asarray(lu.solve(rhs64), dtype=np.float64)


def _solve_sparse_matrix(A: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A X = rhs`` for matrix RHS using configured sparse backend."""
    backend = _select_sparse_backend()
    rhs64 = np.asarray(rhs, dtype=np.float64)
    if backend in ("klu", "umfpack"):
        # KLU/UMFPACK factors accept a 2-D RHS directly (single
        # factorisation, batched solve).
        factor = _sparse_factor(A.tocsc(), backend)
        return np.asarray(factor.solve(rhs64), dtype=np.float64)
    lu = sp.linalg.splu(A.tocsc())
    return np.asarray(lu.solve(rhs64), dtype=np.float64)


class _SparseFactorSolver:
    """Adapter exposing a ``SuperLU``-like ``solve`` over a KLU/UMFPACK factor.

    ``sksparse`` KLU/UMFPACK factors solve ``A x = rhs`` for both 1-D and
    2-D right-hand sides but do not accept a ``trans`` argument.  Callers
    that need the adjoint build ``A^T`` explicitly and solve with
    ``trans="N"``.
    """

    __slots__ = ("_factor",)

    def __init__(self, factor) -> None:
        self._factor = factor

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        if trans != "N":
            raise ValueError("sparse factor solver supports trans='N' only")
        rhs = np.asarray(rhs, dtype=np.float64)
        return np.asarray(self._factor.solve(rhs), dtype=np.float64)


def _make_cached_sparse_solver(
    A: sp.spmatrix, backend: str | None = None
) -> _SparseFactorSolver | None:
    """Build a reusable KLU/UMFPACK factor solver for repeated solves.

    Parameters
    ----------
    A : scipy.sparse matrix
        Matrix to factorise.
    backend : {"klu", "umfpack", "scipy"} or None, optional
        Backend to use.  When ``None`` the configured backend is resolved.

    Returns
    -------
    _SparseFactorSolver | None
        Reusable solver, or ``None`` when the resolved backend is scipy
        (no reusable KLU/UMFPACK factor) or factorisation fails.
    """
    if backend is None:
        backend = _select_sparse_backend()
    if backend not in ("klu", "umfpack"):
        return None
    try:
        return _SparseFactorSolver(_sparse_factor(A.tocsc(), backend))
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
