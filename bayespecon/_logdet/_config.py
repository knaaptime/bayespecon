"""Log-determinant configuration, type aliases, and resolution functions.

This module contains enums, dataclasses, constants, and resolution functions
used across all logdet submodules.  It has no internal (intra-package)
imports so it sits at the bottom of the dependency graph.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Largest n for which `_build_logdet_grid` uses a single eigendecomposition
# rather than a per-rho slogdet loop.  Above this threshold the O(n^3) eigvals
# cost dominates and the iterative slogdet path is more memory-friendly.
_LOGDET_GRID_EIG_MAX = 4000
_LOGDET_FN_CACHE_MAXSIZE = 64
_LOGDET_FN_CACHE: OrderedDict[tuple, Any] = OrderedDict()


# ---------------------------------------------------------------------------
# Enums and type aliases
# ---------------------------------------------------------------------------


class LogDetMethod(str, Enum):
    """Canonical names for the log-determinant approximation methods.

    Each member is a string so the enum members can be used interchangeably
    with their string values (e.g. ``LogDetMethod.EIGENVALUE == "eigenvalue"``).
    """

    EXACT = "exact"
    EIGENVALUE = "eigenvalue"
    GRID_DENSE = "grid_dense"
    GRID_SPARSE = "grid_sparse"
    SPARSE_SPLINE = "sparse_spline"
    GRID_MC = "grid_mc"
    GRID_ILU = "grid_ilu"
    CHEBYSHEV = "chebyshev"


VALID_LOGDET_METHODS: frozenset[str] = frozenset(m.value for m in LogDetMethod)

#: Valid values for the ``trace_estimator`` kwarg of the ``make_logdet_*``
#: builders.  Selects which stochastic estimator constructs the Chebyshev
#: coefficients when an eigendecomposition is unavailable (n above the auto
#: cutoff).  ``"hutchpp"`` is the default; see ``docs/source/user-guide/
#: logdet_profiling.ipynb`` for the cost/accuracy frontier.
VALID_TRACE_ESTIMATORS: frozenset[str] = frozenset({"hutchinson", "hutchpp", "xtrace"})
TraceEstimatorName = Literal["hutchinson", "hutchpp", "xtrace"]

#: Public type alias for user-facing ``logdet_method`` parameters.  Use in
#: constructor signatures to enable IDE autocomplete and static checking:
#:
#: .. code-block:: python
#:
#:     def __init__(self, ..., logdet_method: LogDetMethodName | None = None):
#:         ...
LogDetMethodName = Literal[
    "exact",
    "eigenvalue",
    "grid_dense",
    "grid_sparse",
    "sparse_spline",
    "grid_mc",
    "grid_ilu",
    "chebyshev",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogdetBounds:
    """Resolved logdet method and rho interval used for approximation."""

    method: str
    rho_min: float
    rho_max: float
    source: str


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------


def resolve_logdet_method(method: str | None, *, n: int) -> str:
    """Validate ``method`` and auto-select when ``None``.

    Parameters
    ----------
    method
        User-supplied method name, or ``None`` for auto-selection.
    n
        Spatial dimension; used by auto-selection.

    Returns
    -------
    str
        Canonical method name (always one of :data:`VALID_LOGDET_METHODS`).

    Raises
    ------
    ValueError
        If ``method`` is not a recognised method name.
    """
    if method is None:
        return _auto_logdet_method(int(n))
    if method not in VALID_LOGDET_METHODS:
        valid = ", ".join(sorted(VALID_LOGDET_METHODS))
        raise ValueError(f"Unknown logdet method: {method!r}. Valid options: {valid}.")
    return method


def _resolve_trace_estimator(trace_estimator: str) -> str:
    """Validate the ``trace_estimator`` kwarg."""
    if trace_estimator not in VALID_TRACE_ESTIMATORS:
        valid = ", ".join(sorted(VALID_TRACE_ESTIMATORS))
        raise ValueError(
            f"Unknown trace_estimator: {trace_estimator!r}. Valid options: {valid}."
        )
    return trace_estimator


def _default_trace_k(trace_estimator: str) -> int:
    """Default probe count per trace estimator (see logdet_profiling notebook)."""
    return {"hutchinson": 30, "hutchpp": 50, "xtrace": 25}[trace_estimator]


def _auto_logdet_method(n: int) -> str:
    """Choose the recommended logdet method based on matrix size.

    Parameters
    ----------
    n : int
        Number of spatial units.

    Returns
    -------
    str
        ``'eigenvalue'`` for n less than or equal to the configured cutoff,
        otherwise ``'chebyshev'``.

    Notes
    -----
    The cutoff is configurable via environment variable
    ``BAYESPECON_LOGDET_EIGEN_MAX_N`` (default: ``500``). Lowering the
    cutoff avoids expensive dense eigendecompositions for larger empirical
    datasets while keeping exact evaluation on small to medium test cases.

    For ``n`` above the cutoff this returns ``"chebyshev"``; whether the
    Chebyshev coefficients are built from the exact eigenvalues or from a
    stochastic trace estimator is decided downstream by :func:`chebyshev`
    (based on the ``BAYESPECON_LOGDET_TRACEAX_MIN_N`` threshold) and
    parameterised via the ``trace_estimator`` kwarg of the ``make_logdet_*``
    builders.
    """
    cutoff_raw = os.getenv("BAYESPECON_LOGDET_EIGEN_MAX_N", "500")
    try:
        cutoff = max(1, int(cutoff_raw))
    except ValueError:
        cutoff = 500
    if n <= cutoff:
        return "eigenvalue"
    return "chebyshev"


def resolve_logdet_bounds(
    method: str | None,
    *,
    n: int,
    priors: Mapping[str, Any] | None = None,
    rho_min: float | None = None,
    rho_max: float | None = None,
) -> LogdetBounds:
    """Resolve method-specific rho bounds from defaults, priors, or overrides.

    Resolution precedence is: explicit overrides, prior-derived bounds,
    method defaults.

    For row-standardised W the spectral stability interval is always
    approximately (-1, 1), so the default bounds are sufficient without
    computing eigenvalues.  Pass explicit ``rho_min``/``rho_max`` when
    using non-row-standardised W.
    """
    resolved_method = method if method is not None else _auto_logdet_method(int(n))
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
        elif resolved_method in {"sparse_spline", "grid_mc"}:
            lo = 1e-5
            hi = 1.0
        else:
            lo = -1.0
            hi = 1.0

    if resolved_method in {"sparse_spline", "grid_mc"} and lo < 0.0:
        if source == "override":
            raise ValueError(
                f"method='{resolved_method}' requires a nonnegative rho range; "
                "use rho_min >= 0, or choose a different method."
            )
        # Auto-restrict to the supported positive sub-interval. Methods that
        # only handle nonnegative rho (sparse_spline, grid_mc) silently
        # clamp the lower bound when the prior/default would otherwise be
        # negative; explicit overrides still raise above.
        lo = 1e-5
        if hi <= lo:
            hi = 1.0

    if hi <= lo:
        raise ValueError(
            f"Invalid rho interval after resolution for method='{resolved_method}': "
            f"rho_min={lo}, rho_max={hi}."
        )

    return LogdetBounds(
        method=resolved_method,
        rho_min=float(lo),
        rho_max=float(hi),
        source=source,
    )
