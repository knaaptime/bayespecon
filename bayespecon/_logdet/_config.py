"""Log-determinant configuration: method enum, resolution, and bounds.

Five methods are supported:

* ``"eigenvalue"`` — exact O(n) per-call after one-time O(n³) eigendecomposition.
* ``"slq"`` — stochastic Lanczos quadrature; D-symmetrised batched Lanczos
  with Gauss quadrature trace estimation → Chebyshev coefficients.
* ``"chebyshev"`` — Barry-Pace Monte Carlo traces → Chebyshev approximation; O(m) per call.
* ``"cheb_stochastic"`` — stochastic Chebyshev expansion (Han et al. 2015);
  operator-valued Chebyshev polynomials with geometric convergence via
  Bernstein ellipse.  Same matvec cost as ``chebyshev`` but better accuracy at high |ρ|.
* ``"traces"`` — multinomial trace expansion for unrestricted 3-parameter
  flow models (the only option when the system matrix doesn't factor).

When ``logdet_method`` is ``None`` the method is auto-selected:
``"eigenvalue"`` for n ≤ ``BAYESPECON_LOGDET_EIGEN_MAX_N`` (default 500),
``"cheb_cholesky"`` for n ≤ ``BAYESPECON_LOGDET_CHEB_MAX_N`` (default 20000)
when ``W`` is symmetric (undirected graph), ``"aaa"`` when ``W`` is
non-symmetric (directed graph), otherwise ``"cheb_stochastic"``
(geometric convergence, same cost as Barry-Pace).
``"slq"`` and ``"chebyshev"`` are available as explicit opt-ins.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping

# ---------------------------------------------------------------------------
# Cache constants
# ---------------------------------------------------------------------------

_LOGDET_FN_CACHE_MAXSIZE = 64
_LOGDET_FN_CACHE: OrderedDict[tuple, Any] = OrderedDict()

# ---------------------------------------------------------------------------
# Enum and type alias
# ---------------------------------------------------------------------------


class LogDetMethod(str, Enum):
    """Canonical log-determinant computation methods."""

    EIGENVALUE = "eigenvalue"
    SLQ = "slq"
    CHEBYSHEV = "chebyshev"
    CHEB_STOCHASTIC = "cheb_stochastic"
    CHEB_CHOLESKY = "cheb_cholesky"
    AAA = "aaa"
    TRACES = "traces"


VALID_LOGDET_METHODS: frozenset[str] = frozenset(m.value for m in LogDetMethod)

LogDetMethodName = Literal[
    "eigenvalue",
    "slq",
    "chebyshev",
    "cheb_stochastic",
    "cheb_cholesky",
    "aaa",
    "traces",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogdetBounds:
    """Resolved logdet method and rho interval."""

    method: str
    rho_min: float
    rho_max: float
    source: str


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------


def resolve_logdet_method(
    method: str | None,
    *,
    n: int,
    W=None,
) -> str:
    """Validate ``method`` and auto-select when ``None``.

    Parameters
    ----------
    method
        One of the valid method names, or ``None`` for auto-selection.
    n
        Spatial dimension; used for auto-selection.
    W
        Optional spatial weights matrix.  When ``method`` is ``None``
        and ``n`` is in the medium range, the auto-selector checks
        whether ``W`` is symmetric (undirected graph) to choose between
        ``"cheb_cholesky"`` (symmetric) and ``"aaa"`` (non-symmetric).
        If ``W`` is not provided, defaults to ``"cheb_cholesky"``.

    Returns
    -------
    str
        Canonical method name.
    """
    if method is None:
        return _auto_logdet_method(int(n), W=W)
    if method not in VALID_LOGDET_METHODS:
        valid = ", ".join(sorted(VALID_LOGDET_METHODS))
        raise ValueError(f"Unknown logdet method: {method!r}. Valid options: {valid}.")
    return method


def _is_symmetric_W(W) -> bool:
    """Check whether ``W`` is symmetric (undirected graph).

    Uses ``libpysal.graph.Graph.asymmetry(intrinsic=False)`` when ``W`` is a
    Graph object, falling back to sparse/dense matrix comparison otherwise.
    """
    import numpy as np
    import scipy.sparse as sp

    if W is None:
        return True  # default: assume symmetric

    # libpysal Graph: use built-in asymmetry check
    if hasattr(W, "asymmetry"):
        try:
            asym = W.asymmetry(intrinsic=False)
            return asym.empty
        except Exception:
            pass

    if sp.issparse(W):
        # Sparse difference stays sparse — never densify (n=20k dense is ~3.2GB).
        diff = (W.tocsr() - W.T.tocsr()).tocoo()
        if diff.nnz == 0:
            return True
        return bool(np.all(np.abs(diff.data) <= 1e-10))
    else:
        W_arr = np.asarray(W)
        if W_arr.ndim != 2:
            return True  # 1-D eigenvalue array — not applicable
        return np.allclose(W_arr, W_arr.T, atol=1e-10)


def _auto_logdet_method(n: int, W=None) -> str:
    """Auto-select based on matrix dimension n and W symmetry.

    - ``eigenvalue`` for n ≤ eigen_cutoff (default 500): exact O(n³) eigendecomposition.
    - ``cheb_cholesky`` for n ≤ cheb_cutoff (default 20000) when W is symmetric:
      exact logdet via sparse Cholesky at Chebyshev nodes with symbolic reuse.
      Measured setup (2D rook, adaptive order): ~194ms at n=10k, ~1.0s at n=40k,
      ~2.2s at n=60k.  Accuracy: 3e-6 (n=10k) to 2e-5 (n=60k).  Eval: ~1.3μs/ρ
      via Clenshaw.
    - ``aaa`` for n ≤ cheb_cutoff when W is non-symmetric (directed graph):
      exact logdet via sparse LU (KLU with symbolic reuse) at adaptively-selected
      AAA support points.  Rational approximation converges exponentially near
      singularities.  Uses an adaptive coarse grid of 16–30 LU factorisations
      (16 for narrow intervals clear of ±1), selecting ~7 support points.
      Measured setup ~152ms at n=10k; eval ~5μs/ρ; error 1e-9 to 2e-8.
    - ``cheb_stochastic`` for n > cheb_cutoff: stochastic Chebyshev expansion.
      Lower setup cost (~53ms at n=10k) but carries stochastic error ~0.2-1.9
      with 50 probes, ~0.5-3.5 with 200.  Eval: ~55μs/ρ.  Use when factorisation
      fill-in makes exact setup too expensive.
    """
    eigen_cutoff_raw = os.getenv("BAYESPECON_LOGDET_EIGEN_MAX_N", "500")
    cheb_cutoff_raw = os.getenv("BAYESPECON_LOGDET_CHEB_MAX_N", "20000")
    try:
        eigen_cutoff = max(1, int(eigen_cutoff_raw))
    except ValueError:
        eigen_cutoff = 500
    try:
        cheb_cutoff = max(eigen_cutoff + 1, int(cheb_cutoff_raw))
    except ValueError:
        cheb_cutoff = 20000
    if n <= eigen_cutoff:
        return "eigenvalue"
    if n <= cheb_cutoff:
        # Check W symmetry: cheb_cholesky for symmetric (undirected graph),
        # aaa for non-symmetric (directed graph: KNN, travel time, migration).
        if _is_symmetric_W(W):
            return "cheb_cholesky"
        else:
            return "aaa"
    # Stochastic Chebyshev (Han et al. 2015): geometric convergence via
    # Bernstein ellipse, avoids O(n³) eigendecomposition.
    return "cheb_stochastic"


def resolve_logdet_bounds(
    method: str | None,
    *,
    n: int,
    priors: Mapping[str, Any] | None = None,
    rho_min: float | None = None,
    rho_max: float | None = None,
    W=None,
) -> LogdetBounds:
    """Resolve rho bounds from explicit overrides, priors, or defaults.

    For row-standardised W the stability interval is approximately (-1, 1).

    ``W`` (when supplied) participates in auto-selection so that the
    method recorded here agrees with every other resolution site —
    without it a directed graph would be auto-routed to the
    symmetric-only ``cheb_cholesky``.
    """
    resolved_method = resolve_logdet_method(method, n=int(n), W=W)
    source = "default"

    if rho_min is not None or rho_max is not None:
        if rho_min is None or rho_max is None:
            raise ValueError("Both rho_min and rho_max must be provided together.")
        lo = float(rho_min)
        hi = float(rho_max)
        source = "override"
    else:
        p = priors or {}
        lo_prior = None
        hi_prior = None
        for lk, hk in (("rho_lower", "rho_upper"), ("lam_lower", "lam_upper")):
            if lk in p and hk in p:
                lo_prior = float(p[lk])
                hi_prior = float(p[hk])
                break

        if lo_prior is not None and hi_prior is not None:
            lo = lo_prior
            hi = hi_prior
            source = "prior"
        else:
            lo = -1.0
            hi = 1.0

    if hi <= lo:
        raise ValueError(f"Invalid rho interval: rho_min={lo}, rho_max={hi}.")

    return LogdetBounds(
        method=resolved_method,
        rho_min=float(lo),
        rho_max=float(hi),
        source=source,
    )
