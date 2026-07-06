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
otherwise ``"cheb_stochastic"`` (geometric convergence, same cost as Barry-Pace).
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
    TRACES = "traces"


VALID_LOGDET_METHODS: frozenset[str] = frozenset(m.value for m in LogDetMethod)

LogDetMethodName = Literal[
    "eigenvalue", "slq", "chebyshev", "cheb_stochastic", "traces"
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


def resolve_logdet_method(method: str | None, *, n: int) -> str:
    """Validate ``method`` and auto-select when ``None``.

    Parameters
    ----------
    method
        One of ``"eigenvalue"``, ``"chebyshev"``, ``"traces"``, or ``None``
        for auto-selection.
    n
        Spatial dimension; used for auto-selection.

    Returns
    -------
    str
        Canonical method name.
    """
    if method is None:
        return _auto_logdet_method(int(n))
    if method not in VALID_LOGDET_METHODS:
        valid = ", ".join(sorted(VALID_LOGDET_METHODS))
        raise ValueError(f"Unknown logdet method: {method!r}. Valid options: {valid}.")
    return method


def _auto_logdet_method(n: int) -> str:
    """Auto-select: ``eigenvalue`` for small n, ``chebyshev`` for medium, ``cheb_stochastic`` for large n."""
    eigen_cutoff_raw = os.getenv("BAYESPECON_LOGDET_EIGEN_MAX_N", "500")
    cheb_cutoff_raw = os.getenv("BAYESPECON_LOGDET_CHEB_MAX_N", "2000")
    try:
        eigen_cutoff = max(1, int(eigen_cutoff_raw))
    except ValueError:
        eigen_cutoff = 500
    try:
        cheb_cutoff = max(eigen_cutoff + 1, int(cheb_cutoff_raw))
    except ValueError:
        cheb_cutoff = 2000
    if n <= eigen_cutoff:
        return "eigenvalue"
    if n <= cheb_cutoff:
        # Deterministic Chebyshev from exact eigenvalues — no stochastic noise.
        return "chebyshev"
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
) -> LogdetBounds:
    """Resolve rho bounds from explicit overrides, priors, or defaults.

    For row-standardised W the stability interval is approximately (-1, 1).
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
