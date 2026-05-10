"""Log-determinant utilities for spatial models.

Provides log|I - rho*W| as a pytensor expression or a pre-computed
grid interpolation for large n, used as pm.Potential in spatial likelihoods.

"""

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

# Largest n for which `_build_logdet_grid` uses a single eigendecomposition
# rather than a per-rho slogdet loop.  Above this threshold the O(n^3) eigvals
# cost dominates and the iterative slogdet path is more memory-friendly.
_LOGDET_GRID_EIG_MAX = 4000
_LOGDET_FN_CACHE_MAXSIZE = 64
_LOGDET_FN_CACHE: OrderedDict[tuple, Any] = OrderedDict()


@dataclass(frozen=True)
class LogdetBounds:
    """Resolved logdet method and rho interval used for approximation."""

    method: str
    rho_min: float
    rho_max: float
    source: str


def resolve_logdet_bounds(
    method: str | None,
    *,
    n: int,
    eigs: np.ndarray | None = None,
    priors: Mapping[str, Any] | None = None,
    rho_min: float | None = None,
    rho_max: float | None = None,
    eps: float = 1e-6,
) -> LogdetBounds:
    """Resolve method-specific rho bounds from defaults, priors, or overrides.

    Resolution precedence is: explicit overrides, prior-derived bounds,
    method defaults.
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
        elif resolved_method in {"spline", "mc"}:
            lo = 1e-5
            hi = 1.0
        else:
            lo = -1.0
            hi = 1.0

    if resolved_method in {"spline", "mc"} and lo < 0.0:
        raise ValueError(
            f"method='{resolved_method}' requires a nonnegative rho range; "
            "use priors/overrides with rho_min >= 0, or choose a different method."
        )

    if eigs is not None:
        eigs_r = np.asarray(eigs, dtype=np.float64).real
        pos = eigs_r[eigs_r > 0.0]
        neg = eigs_r[eigs_r < 0.0]
        lo_stable = -np.inf if neg.size == 0 else float(1.0 / np.max(neg))
        hi_stable = np.inf if pos.size == 0 else float(1.0 / np.min(pos))

        lo_clip = lo_stable + eps if np.isfinite(lo_stable) else lo
        hi_clip = hi_stable - eps if np.isfinite(hi_stable) else hi

        if source == "override" and (lo < lo_clip or hi > hi_clip):
            raise ValueError(
                "Explicit rho bounds fall outside the spectral stability interval "
                f"({lo_clip:.6g}, {hi_clip:.6g})."
            )

        lo = max(lo, lo_clip)
        hi = min(hi, hi_clip)

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


def _stable_rho_grid(
    rmin: float, rmax: float, grid: float, eps: float = 1e-6
) -> np.ndarray:
    """Build a rho grid that excludes exact endpoints to avoid singularity hits.

    Parameters
    ----------
    rmin, rmax : float
        Endpoints of the rho domain (typically the SAR / SEM stability
        boundary, e.g. ``-1`` and ``1`` for row-standardised W).
    grid : float
        Spacing between consecutive grid points.
    eps : float, default 1e-6
        Endpoint stand-off.  Larger values give a more robust spline
        away from the eigen-singularity ``1/eig_max`` at the cost of
        coverage near the boundary; values below ``1e-8`` may produce
        ill-conditioned ``log|I - rho W|`` evaluations on
        weakly-stationary W.
    """
    if grid <= 0:
        raise ValueError("grid must be positive.")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    lo = rmin + eps
    hi = rmax - eps
    if hi <= lo:
        raise ValueError("rho interval too narrow after endpoint stabilization.")
    return np.arange(lo, hi + 0.5 * grid, grid, dtype=np.float64)


def logdet_eigenvalue(rho, eigs: np.ndarray) -> pt.TensorVariable:
    """Eigenvalue-based log|I - rho*W|.

    Pre-compute ``eigs = np.linalg.eigvals(W).real`` once; each evaluation
    costs O(n) and is exactly differentiable by pytensor autodiff.

    Parameters
    ----------
    rho : pytensor scalar
        Spatial parameter (rho or lambda).
    eigs : np.ndarray
        Real parts of W's eigenvalues, shape (n,).

    Returns
    -------
    pytensor.tensor.TensorVariable
        Symbolic log-determinant.

    Notes
    -----
    Stability requires ``|rho| < 1 / max(|eigs|)``; for row-standardised W
    this is ``|rho| < 1``. When ``1 - rho * eig_i`` is numerically zero
    (rho exactly at the stability boundary) the unguarded ``log`` would
    return ``-inf``. The argument is therefore clamped at a small floor
    (``1e-300``) before taking ``log``; this keeps NUTS gradients finite
    and produces a very large negative penalty rather than a hard NaN.
    Callers should still constrain ``rho`` away from ``1 / eig_max`` via
    the prior bounds.
    """
    eigs_t = pt.as_tensor_variable(eigs.astype(np.float64))
    arg = pt.abs(1.0 - rho * eigs_t)
    safe = pt.maximum(arg, 1e-300)
    return pt.sum(pt.log(safe))


def logdet_exact(rho, W_dense: np.ndarray) -> pt.TensorVariable:
    """Exact log|I - rho*W| as a pytensor expression.

    Suitable for n < ~1000.

    Parameters
    ----------
    rho : pytensor scalar
        Spatial autoregressive parameter symbol.
    W_dense : np.ndarray
        Dense spatial weights matrix.

    Returns
    -------
    pytensor.tensor.TensorVariable
        Symbolic log-determinant expression.
    """
    n = W_dense.shape[0]
    I = np.eye(n)
    return pt.log(pt.nlinalg.det(I - rho * W_dense))


def _build_logdet_grid(
    W_dense: np.ndarray, rho_min: float, rho_max: float, n_grid: int = 200
):
    """Pre-compute log-determinant values on a rho grid.

    Parameters
    ----------
    W_dense : np.ndarray
        Dense spatial weights matrix.
    rho_min : float
        Lower bound for rho grid.
    rho_max : float
        Upper bound for rho grid.
    n_grid : int, default=200
        Number of equally-spaced grid points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pair ``(rho_grid, logdet_grid)``.
    """
    rho_grid = np.linspace(rho_min + 1e-6, rho_max - 1e-6, n_grid)
    n = W_dense.shape[0]

    # Eigenvalue path: log|I - rho W| = sum_j log|1 - rho * lambda_j|.
    # One O(n^3) eigendecomposition replaces n_grid slogdet calls.  For
    # complex-conjugate eigenvalue pairs the modulus of (1 - rho lambda) is
    # real-positive so the sum-of-logs is exact (matches slogdet to machine
    # precision).  Falls back to the slogdet loop for very large n where
    # eigvals becomes the dominant cost.
    if n <= _LOGDET_GRID_EIG_MAX:
        eigs = np.linalg.eigvals(W_dense)
        # Shape: (n_grid, n).  log of |1 - rho * lambda|, summed across j.
        vals = 1.0 - rho_grid[:, None] * eigs[None, :]
        logdet_grid = np.log(np.abs(vals)).sum(axis=1).astype(np.float64)
        return rho_grid, logdet_grid

    I = np.eye(n)
    logdet_grid = np.empty(n_grid, dtype=np.float64)
    for i, r in enumerate(rho_grid):
        _, logdet_grid[i] = np.linalg.slogdet(I - r * W_dense)
    return rho_grid, logdet_grid


def sparse_grid(W, lmin: float, lmax: float, grid: float = 0.01) -> dict:
    """Compute an exact sparse-LU log-determinant grid.

    Parameters
    ----------
    W : array-like
        Spatial weights matrix.
    lmin : float
        Lower bound of the rho grid.
    lmax : float
        Upper bound of the rho grid.
    grid : float, default=0.01
        Grid step size.

    Returns
    -------
    dict
        Dictionary with ``rho`` and ``lndet`` vectors.
    """
    W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]
    I = sp.eye(n, format="csc", dtype=np.float64)
    rho = _stable_rho_grid(lmin, lmax, grid)
    lndet = np.empty_like(rho)

    for i, r in enumerate(rho):
        A = I - r * W_sp
        lu = spla.splu(A)
        lndet[i] = np.sum(np.log(np.abs(lu.U.diagonal())))

    return {"rho": rho, "lndet": lndet}


def spline(W, rmin: float = 0.0, rmax: float = 1.0, n_grid: int = 100) -> dict:
    """Compute a spline-interpolated log-determinant grid.

    Parameters
    ----------
    W : array-like
        Spatial weights matrix.
    rmin : float, default=0.0
        Lower bound of the rho grid.
    rmax : float, default=1.0
        Upper bound of the rho grid.
    n_grid : int, default=100
        Number of grid points.

    Returns
    -------
    dict
        Dictionary with ``rho`` and ``lndet`` vectors.
    """
    if n_grid < 20:
        raise ValueError("n_grid must be at least 20 for stable spline interpolation.")
    if rmin < 0.0:
        raise ValueError("spline is defined for nonnegative rho ranges (rmin >= 0).")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")

    W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]
    I = sp.eye(n, format="csc", dtype=np.float64)

    rho = np.linspace(rmin, rmax, n_grid, endpoint=False, dtype=np.float64)
    # Follow the original control-point pattern from the legacy routine.
    ctrl = (
        np.array(
            [10, 20, 40, 50, 60, 70, 80, 85, 90, 95, 96, 97, 98, 99, 100], dtype=int
        )
        - 1
    )
    ctrl = np.unique(np.clip(ctrl, 0, n_grid - 1))

    rho_sub = rho[ctrl]
    det_sub = np.empty_like(rho_sub)
    for i, r in enumerate(rho_sub):
        A = I - r * W_sp
        lu = spla.splu(A)
        det_sub[i] = np.sum(np.log(np.abs(lu.U.diagonal())))

    x = np.concatenate(([rmin], rho_sub))
    y = np.concatenate(([0.0], det_sub))
    spline = CubicSpline(x, y, extrapolate=False)
    lndet = spline(rho)
    lndet[0] = 0.0

    return {"rho": rho, "lndet": lndet}


def mc(
    order: int,
    iter: int,
    W,
    rmin: float = 1e-5,
    rmax: float = 1.0,
    grid: float = 0.01,
    random_state: int | None = None,
) -> dict:
    """Compute Monte Carlo log-determinant approximation (:cite:t:`barry1999MonteCarlo`).

    Parameters
    ----------
    order : int
        Number of moments in the stochastic trace expansion.
    iter : int
        Number of Monte Carlo probes.
    W : array-like
        Spatial weights matrix.
    rmin : float, default=1e-5
        Lower bound of the rho grid.
    rmax : float, default=1.0
        Upper bound of the rho grid.
    grid : float, default=0.01
        Grid step size.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with ``rho``, ``lndet``, ``up95``, and ``lo95`` vectors.
    """
    if order <= 0:
        raise ValueError("order must be positive.")
    if iter <= 0:
        raise ValueError("iter must be positive.")
    if rmin < 0.0:
        raise ValueError("mc is defined for nonnegative rho ranges (rmin >= 0).")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")
    W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]

    rng = np.random.default_rng(random_state)

    # Obtain per-probe raw trace estimates tr(W^k), then scale by 1/(k+1)
    raw = _barry_pace_traces(W_sp, order, iter, rng)  # (order, iter)
    k_arr = np.arange(1, order + 1, dtype=np.float64)[:, None]  # (order, 1)
    mavmomi = raw / k_arr  # tr(W^k) / k — matches original mavmomi format
    avmomi = mavmomi.mean(axis=1)

    rho = _stable_rho_grid(rmin, rmax, grid)
    # Build polynomial terms alpha^1..alpha^order.
    powers = np.power(rho[:, None], np.arange(1, order + 1, dtype=np.float64)[None, :])
    alomat = -powers

    lndet = alomat @ avmomi
    srvs = (alomat @ mavmomi).T
    sderr = np.sqrt(np.maximum(0.0, srvs.var(axis=0, ddof=0) / iter))

    fbound = (n * np.power(rho, order + 1)) / (
        (order + 1) * (1.0 - rho + np.finfo(float).eps)
    )
    lo95 = lndet - 1.96 * sderr - fbound
    up95 = lndet + 1.96 * sderr

    return {"rho": rho, "lndet": lndet, "up95": up95, "lo95": lo95}


def ilu(
    W,
    lmin: float,
    lmax: float,
    grid: float = 0.01,
    drop_tol: float = 1e-3,
    fill_factor: float = 10.0,
) -> dict:
    """Compute an ILU-based approximate log-determinant grid.

    Parameters
    ----------
    W : array-like
        Spatial weights matrix.
    lmin : float
        Lower bound of the rho grid.
    lmax : float
        Upper bound of the rho grid.
    grid : float, default=0.01
        Grid step size.
    drop_tol : float, default=1e-3
        Drop tolerance for ILU factorisation.
    fill_factor : float, default=10.0
        Fill-in control for ILU factorisation.

    Returns
    -------
    dict
        Dictionary with ``rho`` and approximate ``lndet`` vectors.
    """
    W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]
    I = sp.eye(n, format="csc", dtype=np.float64)
    rho = _stable_rho_grid(lmin, lmax, grid)
    lndet = np.empty_like(rho)

    for i, r in enumerate(rho):
        A = (I - r * W_sp).tocsc()
        ilu = spla.spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
        lndet[i] = np.sum(np.log(np.abs(ilu.L.diagonal()))) + np.sum(
            np.log(np.abs(ilu.U.diagonal()))
        )

    return {"rho": rho, "lndet": lndet}


def chebyshev(
    W,
    order: int = 20,
    rmin: float = -1.0,
    rmax: float = 1.0,
    random_state: int | None = None,
    eigs: np.ndarray | None = None,
    n_mc_iter: int = 30,
) -> dict:
    """Compute Chebyshev approximation of log|I - rho*W| (:cite:p:`pace2004ChebyshevApproximation`).

    Uses Chebyshev polynomials of the first kind to approximate the
    log-determinant over ``[rmin, rmax]``.  The approximation is
    near-minimax: for a given polynomial degree it minimises the
    maximum absolute error on the interval.

    Two computation strategies are supported:

    * **Eigenvalue-based** (default when *n* ≤ 2000): evaluates
      the exact log-determinant at Chebyshev nodes via eigenvalues,
      then computes Chebyshev coefficients from those values.
    * **Monte-Carlo trace-based** (automatically used when *n* > 2000):
      replaces exact traces with Barry-Pace stochastic trace estimates
      (:cite:t:`barry1999MonteCarlo`), avoiding the O(n³) eigenvalue
      decomposition.

    Parameters
    ----------
    W : array-like
        Spatial weights matrix (dense or sparse).
    order : int, default=20
        Number of Chebyshev terms (polynomial degree).  Higher values
        give better accuracy; 15–30 is usually sufficient.
    rmin : float, default=-1.0
        Lower bound of the rho interval.
    rmax : float, default=1.0
        Upper bound of the rho interval.
    random_state : int, optional
        Seed for the Monte Carlo trace estimator (only used when
        *n* > 2000).
    eigs : np.ndarray, optional
        Pre-computed real eigenvalues of *W*.  When supplied, the
        eigenvalue decomposition step is skipped, avoiding redundant
        O(n³) work when the caller already has them cached.
    n_mc_iter : int, default=30
        Number of Hutchinson probes used by the Monte-Carlo trace
        estimator (only consulted when *n* > 2000 and ``eigs`` is
        ``None``).  Larger values reduce stochastic variance at
        linear cost.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``coeffs`` : Chebyshev coefficients ``c_0, c_1, …, c_{m-1}``.
        - ``rmin``, ``rmax`` : interval bounds (echoed back).
        - ``order`` : polynomial degree used.
        - ``method`` : ``'eigenvalue'`` or ``'mc'`` indicating how
          coefficients were computed.

    Notes
    -----
    The Chebyshev approximation is

    .. math::

        \\ln|I_n - \\rho W| \\approx
            \\sum_{j=0}^{m-1} c_j \\, T_j\\!\\left(
                \\frac{2\\rho - r_{\\max} - r_{\\min}}
                     {r_{\\max} - r_{\\min}}
            \\right)

    where :math:`T_j` are Chebyshev polynomials of the first kind and
    the coefficients :math:`c_j` are computed via the discrete cosine
    transform of the log-determinant evaluated at Chebyshev nodes.

    The error bound for the *m*-term approximation on
    :math:`[r_{\\min}, r_{\\max}]` is

    .. math::

        |\\text{error}| \\leq
            \\frac{n\\,|\\rho|^{m+1}}{(m+1)(1-|\\rho|)}

    for row-standardised :math:`W` with :math:`|\\rho| < 1`.

    References
    ----------
    Pace, R.K. & LeSage, J.P. (2004). Chebyshev approximation of
    log-determinants of spatial weight matrices. *Computational
    Statistics & Data Analysis*, 45(2), 179–196.
    :cite:p:`pace2004ChebyshevApproximation`

    Trefethen, L.N. (2013). *Approximation Theory and Approximation
    Practice*. SIAM. (Background on Chebyshev nodes, the Clenshaw
    recurrence used in :func:`logdet_chebyshev` for stable evaluation,
    and near-minimax convergence rates for analytic functions.)
    """
    if order <= 0:
        raise ValueError("order must be positive.")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")

    # Allow caller to skip dense materialisation when only eigs are needed.
    if eigs is not None:
        eigs_arr = np.asarray(eigs, dtype=np.float64).real
        n = int(eigs_arr.shape[0])
        W_sp = None
    else:
        # Handle sparse matrices/arrays: don't convert to dense via np.asarray
        if sp.issparse(W) or hasattr(W, "format"):  # sp.csr_array check
            W_sp = sp.csr_matrix(W)
        else:
            W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
        n = W_sp.shape[0]
        eigs_arr = None

    # Chebyshev nodes on [-1, 1], mapped to [rmin, rmax]
    k = np.arange(1, order + 1)
    # Nodes: cos((2k-1)π / (2m)) for k=1..m  (Chebyshev nodes of the first kind)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    # Map from [-1, 1] to [rmin, rmax]
    rho_nodes = 0.5 * (rmax - rmin) * nodes_cos + 0.5 * (rmax + rmin)

    # Decide computation strategy
    # Only fall back to MC for truly large matrices; for small matrices
    # eigenvalue decomposition is fast and exact.
    use_mc = (eigs_arr is None) and (n > 2000)

    if not use_mc:
        # Eigenvalue-based: exact log-determinant at each Chebyshev node
        if eigs_arr is None:
            eigs_arr = np.linalg.eigvals(W_sp.toarray()).real
        logdet_at_nodes = np.sum(
            np.log(np.abs(1.0 - rho_nodes[:, None] * eigs_arr[None, :])), axis=1
        )
        method_used = "eigenvalue"
    else:
        # Monte Carlo trace-based: use Barry-Pace stochastic trace
        # ln|I - ρW| = -Σ_{k=1}^{∞} (ρ^k / k) tr(W^k)
        # Approximate tr(W^k) via MC, then evaluate at nodes
        rng = np.random.default_rng(random_state)

        # Compute MC trace estimates for k=1..order via batched Hutchinson
        # probes: one CSR×dense product per order covers all probes.
        U = rng.standard_normal((n, n_mc_iter))
        utu = np.einsum("ij,ij->j", U, U)  # (n_mc_iter,)
        V = U.copy()
        td = np.zeros(order, dtype=np.float64)
        for i in range(order):
            V = W_sp @ V
            tr_k = n * np.einsum("ij,ij->j", U, V) / utu  # (n_mc_iter,)
            td[i] = tr_k.mean() / (i + 1)

        # Evaluate power series at each node
        logdet_at_nodes = np.zeros(order, dtype=np.float64)
        for idx, r in enumerate(rho_nodes):
            powers = np.power(r, np.arange(1, order + 1, dtype=np.float64))
            logdet_at_nodes[idx] = -powers @ td
        method_used = "mc"

    # Compute Chebyshev coefficients via DCT-I
    # c_j = (2 - δ_{j,0}) / m * Σ_{k=1}^{m} f(ρ_k*) cos(j(2k-1)π / (2m))
    coeffs = np.zeros(order, dtype=np.float64)
    for j in range(order):
        scale = 2.0 / order if j > 0 else 1.0 / order
        coeffs[j] = scale * np.sum(
            logdet_at_nodes * np.cos(j * (2 * k - 1) * np.pi / (2 * order))
        )

    return {
        "coeffs": coeffs,
        "rmin": rmin,
        "rmax": rmax,
        "order": order,
        "method": method_used,
    }


def logdet_chebyshev(
    rho,
    coeffs: np.ndarray,
    rmin: float = -1.0,
    rmax: float = 1.0,
) -> pt.TensorVariable:
    """Evaluate Chebyshev approximation of log|I - rho*W| symbolically.

    Uses Clenshaw's algorithm for numerically stable evaluation of the
    Chebyshev series at a PyTensor scalar ``rho``.

    Parameters
    ----------
    rho : pytensor scalar
        Spatial autoregressive parameter symbol.
    coeffs : np.ndarray
        Chebyshev coefficients from :func:`chebyshev`, shape ``(m,)``.
    rmin : float, default=-1.0
        Lower bound of the rho interval (must match what was used to
        compute *coeffs*).
    rmax : float, default=1.0
        Upper bound of the rho interval (must match what was used to
        compute *coeffs*).

    Returns
    -------
    pytensor.tensor.TensorVariable
        Symbolic Chebyshev approximation of the log-determinant.

    Notes
    -----
    The mapped variable is

    .. math::

        x = \\frac{2\\rho - r_{\\max} - r_{\\min}}{r_{\\max} - r_{\\min}}

    and the approximation is evaluated via Clenshaw's recurrence:

    .. math::

        b_{m+1} = 0, \\quad b_m = c_m

        b_k = 2x \\, b_{k+1} - b_{k+2} + c_k

        f(x) = x \\, b_1 - b_2 + c_0
    """
    m = len(coeffs)
    if m == 0:
        return pt.zeros_like(rho)

    # Map rho ∈ [rmin, rmax] → x ∈ [-1, 1]
    x = (2.0 * rho - rmax - rmin) / (rmax - rmin)

    # Clenshaw's algorithm for Σ_{j=0}^{m-1} c_j T_j(x)
    # Iterate from k = m-1 down to k = 1:
    #   b_{m} = c_{m-1},  b_{m+1} = 0
    #   b_k = 2x b_{k+1} - b_{k+2} + c_k
    # Then: f(x) = c_0 + x*b_1 - b_2
    c = pt.as_tensor_variable(coeffs.astype(np.float64))

    if m == 1:
        # Only c_0 * T_0(x) = c_0
        return c[0] * pt.ones_like(rho)

    # Start: b_{m} = c_{m-1}, b_{m+1} = 0
    b_next = pt.zeros_like(rho)  # b_{m+1} = 0
    b_curr = c[m - 1]  # b_m = c_{m-1}

    # Iterate from k = m-2 down to k = 1
    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + c[k]
        b_next = b_curr
        b_curr = b_new

    # f(x) = c_0 + x*b_1 - b_2
    # After the loop, b_curr = b_1, b_next = b_2
    return c[0] + x * b_curr - b_next


def logdet_mc_poly_pytensor(
    rho,
    traces: np.ndarray,
) -> pt.TensorVariable:
    r"""Evaluate Barry-Pace trace polynomial approximation of log|I - rho*W| symbolically.

    Computes the truncated power-series approximation

    .. math::

        \log|I_n - \rho W| \approx -\sum_{k=1}^{m} \frac{\rho^k}{k}\,\hat{\tau}_k

    where :math:`\hat{\tau}_k \approx \text{tr}(W^k)` are the Barry-Pace
    stochastic trace estimates from :func:`compute_flow_traces`, using
    Horner's method for numerically stable evaluation.

    Unlike :func:`mc` (which builds a lookup table over a fixed rho grid),
    this function returns a symbolic :mod:`pytensor` expression valid for any
    :math:`\rho \in [-1, 1]` and is therefore suitable for use inside a
    PyMC model as a ``pm.Potential``.

    Parameters
    ----------
    rho : pytensor scalar
        Spatial autoregressive parameter symbol.
    traces : np.ndarray, shape (m,)
        Trace estimates ``traces[k-1] ≈ tr(W^k)`` for k=1..m from
        :func:`compute_flow_traces`.

    Returns
    -------
    pytensor.tensor.TensorVariable
        Symbolic polynomial approximation of the log-determinant.

    Notes
    -----
    Horner evaluation of :math:`-\sum_{k=1}^m w_k \rho^k` where
    :math:`w_k = \hat{\tau}_k / k`:

    .. math::

        -\rho \bigl(w_1 + \rho(w_2 + \rho(\cdots + \rho\, w_m)\cdots)\bigr)
    """
    m = len(traces)
    if m == 0:
        return pt.zeros_like(rho)
    k_arr = np.arange(1, m + 1, dtype=np.float64)
    w = (traces / k_arr).astype(np.float64)  # w[k-1] = tr_k / k
    w_t = pt.as_tensor_variable(w)

    # Horner's method, high-to-low coefficients
    result = w_t[m - 1]
    for j in range(m - 2, -1, -1):
        result = result * rho + w_t[j]
    result = result * rho
    return -result


def logdet_interpolated(
    rho,
    W_dense: np.ndarray,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    n_grid: int = 200,
):
    """Cubic spline interpolation of log|I - rho*W|.

    Pre-computes values on a rho grid at construction time and evaluates
    a cubic-spline piecewise polynomial symbolically.

    Parameters
    ----------
    rho : pytensor scalar
        Spatial autoregressive parameter symbol.
    W_dense : np.ndarray
        Dense spatial weights matrix.
    rho_min : float, default=-1.0
        Lower bound for rho grid.
    rho_max : float, default=1.0
        Upper bound for rho grid.
    n_grid : int, default=200
        Number of grid points.

    Returns
    -------
    pytensor.tensor.TensorVariable
        Interpolated symbolic log-determinant value.
    """
    rho_grid, logdet_grid = _build_logdet_grid(W_dense, rho_min, rho_max, n_grid)
    spline = CubicSpline(rho_grid, logdet_grid)

    # Convert spline to a callable on pytensor scalars via interpolation
    # We use a piecewise polynomial evaluated via pytensor scan-free approach:
    # store breakpoints and coefficients, evaluate via pt ops.
    breakpoints = pt.as_tensor_variable(spline.x.astype(np.float64))
    coefficients = pt.as_tensor_variable(
        spline.c.astype(np.float64)
    )  # (4, n_intervals)

    # Find the interval index for rho
    idx = pt.sum(pt.lt(breakpoints, rho)) - 1
    idx = pt.clip(idx, 0, len(spline.x) - 2)

    dx = rho - breakpoints[idx]
    c = coefficients[:, idx]  # shape (4,)
    # Evaluate cubic: c[0]*dx^3 + c[1]*dx^2 + c[2]*dx + c[3]
    value = c[0] * dx**3 + c[1] * dx**2 + c[2] * dx + c[3]
    return value


def _barry_pace_traces(
    W_sparse: sp.csr_matrix,
    order: int,
    iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate tr(W^k) for k=1..order via Barry-Pace Monte Carlo probes.

    This is the core stochastic trace estimation loop from
    :cite:t:`barry1999MonteCarlo`, extracted from :func:`mc` so that it can
    also be used by the flow log-determinant code.

    Parameters
    ----------
    W_sparse :
        Sparse n×n spatial weights matrix.
    order :
        Maximum trace power to estimate.
    iter :
        Number of Monte Carlo probes (random vectors).
    rng :
        NumPy random generator instance.

    Returns
    -------
    np.ndarray, shape (order, iter)
        Per-probe trace estimates.  Entry ``[k, j]`` is the estimate of
        ``tr(W^{k+1})`` from probe *j*.  Rows 0 and 1 are overridden with
        exact values (``tr(W)`` and ``tr(W²)``).
    """
    n = W_sparse.shape[0]
    # Batched Hutchinson: draw all probes at once and let scipy's CSR×dense
    # kernel amortise row-pointer traversal across the (n, iter) RHS.
    U = rng.standard_normal((n, iter))
    utu = np.einsum("ij,ij->j", U, U)  # (iter,)
    V = U.copy()
    traces = np.empty((order, iter), dtype=np.float64)
    for i in range(order):
        V = W_sparse @ V  # (n, iter)
        traces[i] = n * np.einsum("ij,ij->j", U, V) / utu
    # Override with exact values for k=1, 2
    traces[0, :] = float(W_sparse.diagonal().sum())  # tr(W) = 0 for zero-diagonal W
    if order >= 2:
        traces[1, :] = float(
            W_sparse.multiply(W_sparse.T).sum()
        )  # tr(W^2) = sum(W .* W')
    return traces


def compute_flow_traces(
    W_sparse,
    miter: int = 30,
    riter: int = 50,
    random_state: int | None = None,
) -> np.ndarray:
    """Estimate tr(W^k) for k=1..miter via Barry-Pace stochastic traces.

    Thin public wrapper around :func:`_barry_pace_traces`, mirroring
    ``ftrace1.m`` from the LeSage spatial flows toolbox.  Used by
    :func:`_flow_logdet_poly_coeffs` to pre-compute trace products for the
    flow log-determinant.

    Parameters
    ----------
    W_sparse : array-like or scipy.sparse matrix
        Row-standardised n×n spatial weights matrix.
    miter : int, default=30
        Number of trace orders to estimate (``traces[k-1] ≈ tr(W^k)`` for
        k=1..miter).  Higher values improve the polynomial approximation;
        30–50 is usually sufficient with ``titer=800`` for the geometric tail.
    riter : int, default=50
        Number of Monte Carlo probe vectors for trace estimation.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (miter,)
        Trace estimates: ``traces[k-1] ≈ tr(W^k)`` for k=1..miter.
    """
    if sp.issparse(W_sparse):
        W_sp = W_sparse.tocsr().astype(np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W_sparse, dtype=np.float64))
    rng = np.random.default_rng(random_state)
    raw = _barry_pace_traces(W_sp, miter, riter, rng)  # (miter, riter)
    return raw.mean(axis=1)  # (miter,)


def _flow_logdet_poly_coeffs(
    traces: np.ndarray,
    n: int,
    miter: int,
) -> tuple:
    """Pre-compute polynomial coefficients for the flow log-determinant.

    Ports the multinomial trace identity from ``fodet1.m`` (LeSage 2005).
    For the flow SAR weight matrix
    :math:`W_F = \\rho_d W_d + \\rho_o W_o + \\rho_w W_w` the log-determinant
    expands as:

    .. math::

        \\log|I_N - W_F| = -\\sum_{k=1}^{\\infty}
            \\frac{1}{k} \\text{tr}(W_F^k)

    where by Kronecker properties:

    .. math::

        \\text{tr}(W_F^k) = \\sum_{a+b+c=k} \\binom{k}{a,b,c}
            \\rho_d^a \\rho_o^b \\rho_w^c \\cdot
            \\text{tr}(W^{a+c})\\,\\text{tr}(W^{b+c})

    This function enumerates all (a, b, c) triples for k=1..miter and
    returns flat numpy arrays ready for vectorised evaluation inside
    :func:`flow_logdet_pytensor`.

    Parameters
    ----------
    traces : np.ndarray, shape (miter,)
        Trace estimates from :func:`compute_flow_traces`:
        ``traces[k-1] ≈ tr(W^k)`` for k=1..miter.
    n : int
        Spatial unit count (not flow count N=n²).  Used for ``tr(I_n) = n``
        (the k=0 trace).
    miter : int
        Highest polynomial order for the exact series.  Must equal
        ``len(traces)``.

    Returns
    -------
    tuple of 8 np.ndarray
        ``(poly_a, poly_b, poly_c, poly_coeffs,
           miter_a, miter_b, miter_c, miter_coeffs)``

        ``poly_*`` arrays cover all triples with :math:`a+b+c \\in [1, miter]`.
        ``poly_coeffs[i] = -C(k;a,b,c) \\cdot tw[a+c] \\cdot tw[b+c] / k``
        where tw[0]=n and tw[p]=tr(W^p) for p≥1.

        ``miter_*`` arrays cover only triples with :math:`a+b+c = miter`.
        ``miter_coeffs[i] = C(miter;a,b,c) \\cdot tw[a+c] \\cdot tw[b+c]``
        (positive, without the 1/k division — used for the geometric tail
        inside :func:`flow_logdet_pytensor`).

    Examples
    --------
    For ``miter = 2`` the routine enumerates seven exact triples
    (k=1: ``{(1,0,0), (0,1,0), (0,0,1)}``; k=2: ``{(2,0,0), (0,2,0),
    (0,0,2), (1,1,0), (1,0,1), (0,1,1)}`` — six triples plus the three
    k=1 ones gives nine, of which the cross terms with ``b+c=0`` collapse
    by symmetry to seven distinct rows).  Each row contributes
    :math:`\\rho_d^a \\rho_o^b \\rho_w^c` weighted by ``poly_coeffs[i]``;
    ``flow_logdet_pytensor`` evaluates the dot product
    ``poly_coeffs @ (rho_d**a * rho_o**b * rho_w**c)`` symbolically.  The
    ``miter_*`` arrays alone power the geometric-series tail
    :math:`\\sum_{k=miter+1}^{\\infty}` whose closed form is
    :math:`\\text{tail} \\cdot \\rho^{miter+1} / (1 - \\rho)`
    (after clipping :math:`\\rho_d + \\rho_o + \\rho_w` away from 1).
    """
    from math import factorial

    if len(traces) != miter:
        raise ValueError(f"len(traces)={len(traces)} must equal miter={miter}.")

    # tw[0] = n = tr(I_n),  tw[k] = tr(W^k) for k=1..miter
    tw = np.empty(miter + 1, dtype=np.float64)
    tw[0] = float(n)
    tw[1:] = traces

    poly_rows: list[tuple] = []
    miter_rows: list[tuple] = []

    for k in range(1, miter + 1):
        for a in range(k + 1):
            for b in range(k - a + 1):
                c = k - a - b
                multi = factorial(k) // (factorial(a) * factorial(b) * factorial(c))
                # Trace product: tr(W^{a+c}) * tr(W^{b+c}); indices a+c, b+c ∈ [0, k]
                trace_prod = float(tw[a + c] * tw[b + c])
                coeff = -float(multi) * trace_prod / k
                poly_rows.append((float(a), float(b), float(c), coeff))
                if k == miter:
                    miter_rows.append(
                        (float(a), float(b), float(c), float(multi) * trace_prod)
                    )

    poly_arr = np.array(poly_rows, dtype=np.float64)
    miter_arr = np.array(miter_rows, dtype=np.float64)

    return (
        poly_arr[:, 0],  # poly_a
        poly_arr[:, 1],  # poly_b
        poly_arr[:, 2],  # poly_c
        poly_arr[:, 3],  # poly_coeffs
        miter_arr[:, 0],  # miter_a
        miter_arr[:, 1],  # miter_b
        miter_arr[:, 2],  # miter_c
        miter_arr[:, 3],  # miter_coeffs
    )


def flow_logdet_pytensor(
    rho_d,
    rho_o,
    rho_w,
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    poly_c: np.ndarray,
    poly_coeffs: np.ndarray,
    miter_a: np.ndarray,
    miter_b: np.ndarray,
    miter_c: np.ndarray,
    miter_coeffs: np.ndarray,
    miter: int,
    titer: int = 800,
) -> "pt.TensorVariable":
    """Differentiable PyTensor log-determinant for the flow SAR model.

    Evaluates

    .. math::

        \\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|

    as a fully differentiable PyTensor expression suitable for use as
    ``pm.Potential("jacobian", flow_logdet_pytensor(...))``.

    The computation has two parts:

    1. **Polynomial part** (orders 1 to *miter*): vectorised sum over
       precomputed ``(a, b, c, coeff)`` triples — no Python loop at
       evaluation time.

    2. **Geometric tail** (orders *miter+1* to *titer*): closed-form sum
       using the upper-bound approximation
       :math:`\\text{tr}(W_F^k) \\approx s^{k-m} \\cdot \\text{tr}(W_F^m)`
       where :math:`s = \\rho_d + \\rho_o + \\rho_w` is the spectral-radius
       bound for row-stochastic W, following ``fodet1.m`` lines 60–70.

    Parameters
    ----------
    rho_d, rho_o, rho_w :
        PyTensor scalar variables for the three spatial parameters.
    poly_a, poly_b, poly_c, poly_coeffs :
        Precomputed exponent arrays and coefficients for the polynomial part,
        from :func:`_flow_logdet_poly_coeffs`.
    miter_a, miter_b, miter_c, miter_coeffs :
        Exponents and trace-product weights for the highest-order polynomial
        terms (k = miter), used to compute ``tr(W_F^miter)`` symbolically
        for the geometric tail.
    miter : int
        Highest polynomial order included in the exact series.
    titer : int, default=800
        Highest order included in the geometric tail approximation.

    Returns
    -------
    pytensor.tensor.TensorVariable
        Scalar log-determinant expression.
    """
    # --- Polynomial part: k = 1 .. miter ---
    pa = pt.as_tensor_variable(poly_a)
    pb = pt.as_tensor_variable(poly_b)
    pc = pt.as_tensor_variable(poly_c)
    pcoeffs = pt.as_tensor_variable(poly_coeffs)

    poly_part = pt.sum(
        pcoeffs * pt.power(rho_d, pa) * pt.power(rho_o, pb) * pt.power(rho_w, pc)
    )

    # --- Geometric tail: k = miter+1 .. titer ---
    # tr(W_F^miter) as a PyTensor expression
    ma = pt.as_tensor_variable(miter_a)
    mb = pt.as_tensor_variable(miter_b)
    mc_ = pt.as_tensor_variable(miter_c)
    mcoeffs = pt.as_tensor_variable(miter_coeffs)

    trace_last = pt.sum(
        mcoeffs * pt.power(rho_d, ma) * pt.power(rho_o, mb) * pt.power(rho_w, mc_)
    )

    # scalarparm = rho_d + rho_o + rho_w  (spectral radius bound for row-stochastic W)
    # The geometric tail series ``sum_j s^j / (miter+j)`` only converges
    # for ``|s| < 1``. We clip ``s`` strictly inside the unit interval to
    # avoid ``s^titer`` overflowing for j up to ~``titer-miter``; the
    # bias introduced very close to the boundary is preferable to NaN/Inf
    # gradients during NUTS adaptation.
    s_raw = rho_d + rho_o + rho_w
    scalarparm = pt.clip(s_raw, -1.0 + 1e-6, 1.0 - 1e-6)

    # tail_sum = sum_{j=1}^{titer-miter} scalarparm^j / (miter + j)
    j_arr = np.arange(1, titer - miter + 1, dtype=np.float64)
    recip_arr = pt.as_tensor_variable((1.0 / (miter + j_arr)).astype(np.float64))
    tail_sum = pt.dot(pt.power(scalarparm, j_arr), recip_arr)

    tail_part = -trace_last * tail_sum

    return poly_part + tail_part


def flow_logdet_numpy(
    rho_d,
    rho_o,
    rho_w,
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    poly_c: np.ndarray,
    poly_coeffs: np.ndarray,
    miter_a: np.ndarray,
    miter_b: np.ndarray,
    miter_c: np.ndarray,
    miter_coeffs: np.ndarray,
    miter: int,
    titer: int = 800,
) -> np.ndarray:
    """Vectorised numpy port of :func:`flow_logdet_pytensor`.

    Evaluates :math:`\\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|`
    for arrays of posterior draws ``(rho_d, rho_o, rho_w)``.  Used by
    :class:`~bayespecon.models.flow.FlowModel`-derived models to attach a
    complete pointwise log-likelihood to the InferenceData after sampling
    (the ``pm.Potential("jacobian", ...)`` Jacobian term is not captured by
    PyMC's automatic log-likelihood accounting).

    Parameters
    ----------
    rho_d, rho_o, rho_w : array-like, shape (G,) or scalar
        Posterior draws for the three spatial parameters.
    poly_a, poly_b, poly_c, poly_coeffs, miter_a, miter_b, miter_c, miter_coeffs :
        Precomputed exponent arrays and coefficients from
        :func:`_flow_logdet_poly_coeffs`.
    miter : int
        Highest polynomial order included in the exact series.
    titer : int, default 800
        Highest order included in the geometric tail approximation.

    Returns
    -------
    np.ndarray, shape (G,)
        Log-determinant evaluated at each draw.
    """
    rho_d = np.atleast_1d(np.asarray(rho_d, dtype=np.float64))
    rho_o = np.atleast_1d(np.asarray(rho_o, dtype=np.float64))
    rho_w = np.atleast_1d(np.asarray(rho_w, dtype=np.float64))

    # Polynomial part: vectorised over draws.
    poly_part = (
        poly_coeffs[None, :]
        * np.power(rho_d[:, None], poly_a[None, :])
        * np.power(rho_o[:, None], poly_b[None, :])
        * np.power(rho_w[:, None], poly_c[None, :])
    ).sum(axis=1)

    # tr(W_F^miter) symbolically per draw.
    trace_last = (
        miter_coeffs[None, :]
        * np.power(rho_d[:, None], miter_a[None, :])
        * np.power(rho_o[:, None], miter_b[None, :])
        * np.power(rho_w[:, None], miter_c[None, :])
    ).sum(axis=1)

    # Geometric tail (clip to avoid overflow at the boundary).
    s_raw = rho_d + rho_o + rho_w
    scalarparm = np.clip(s_raw, -1.0 + 1e-6, 1.0 - 1e-6)

    j_arr = np.arange(1, titer - miter + 1, dtype=np.float64)
    recip = 1.0 / (miter + j_arr)
    tail_sum = (np.power(scalarparm[:, None], j_arr[None, :]) * recip[None, :]).sum(
        axis=1
    )

    return poly_part - trace_last * tail_sum


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
    """
    cutoff_raw = os.getenv("BAYESPECON_LOGDET_EIGEN_MAX_N", "500")
    try:
        cutoff = max(1, int(cutoff_raw))
    except ValueError:
        cutoff = 500
    return "eigenvalue" if n <= cutoff else "chebyshev"


_GRID_SPLINE_METHODS = ("dense_grid", "sparse_grid", "spline", "mc", "ilu")


def _build_grid_spline(
    W_dense: np.ndarray,
    method: str,
    rho_min: float,
    rho_max: float,
    *,
    clamp_nonnegative: bool,
) -> CubicSpline:
    """Build a CubicSpline approximation for grid/interpolation methods."""
    if method == "dense_grid":
        rho_grid, logdet_grid = _build_logdet_grid(W_dense, rho_min, rho_max)
        return CubicSpline(rho_grid, logdet_grid)
    if method == "sparse_grid":
        out = sparse_grid(W_dense, rho_min, rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    if method == "spline":
        rmin = max(rho_min, 0.0) if clamp_nonnegative else rho_min
        out = spline(W_dense, rmin, rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    if method == "mc":
        rmin = max(rho_min, 1e-5) if clamp_nonnegative else rho_min
        out = mc(order=50, iter=30, W=W_dense, rmin=rmin, rmax=rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    if method == "ilu":
        out = ilu(W_dense, rho_min, rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    raise ValueError(
        f"Grid spline helper does not support method={method!r}. "
        f"Choose one of {_GRID_SPLINE_METHODS}."
    )


def _make_pytensor_interp_fn(spline_obj: CubicSpline, T: int):
    """Create a pytensor scalar interpolation closure from a SciPy CubicSpline."""
    breakpoints_np = spline_obj.x.astype(np.float64)
    coefficients_np = spline_obj.c.astype(np.float64)
    step = float(breakpoints_np[1] - breakpoints_np[0])
    bp0 = float(breakpoints_np[0])
    n_intervals = len(breakpoints_np) - 2

    def _interp(rho):
        bp = pt.as_tensor_variable(breakpoints_np)
        c = pt.as_tensor_variable(coefficients_np)
        idx = pt.cast(pt.floor((rho - bp0) / step), "int64")
        idx = pt.clip(idx, 0, n_intervals)
        dx = rho - bp[idx]
        coefs = c[:, idx]
        val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
        return val if T == 1 else T * val

    return _interp


def make_logdet_numpy_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
):
    """Return a **pure-numpy** ``(rho: float) -> float`` logdet evaluator.

    Used for post-sampling log-likelihood Jacobian computation (outside any
    PyMC/PyTensor graph context).  Mirrors :func:`make_logdet_fn` but returns
    a plain Python callable instead of a PyTensor expression.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        Row-standardised n×n spatial weights matrix.
    eigs : np.ndarray, optional
        Optional pre-computed real eigenvalues of W.
    method : str or None
        Same as :func:`make_logdet_fn`.  ``None`` auto-selects via
        :func:`_auto_logdet_method`.
    rho_min : float, default -1.0
        Lower bound (used for chebyshev/spline precomputation).
    rho_max : float, default 1.0
        Upper bound.

    Returns
    -------
    callable
        Function ``(rho: float) -> float`` computing log|I - rho*W|.
    """
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    if method is None:
        method = _auto_logdet_method(n)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(
                np.asarray(W_sparse.toarray(), dtype=np.float64)
            ).real
        _eigs = eigs.real.astype(np.float64)
        return lambda r: float(np.sum(np.log(np.abs(1.0 - r * _eigs))))

    elif method == "chebyshev":
        # Pass precomputed eigs to skip the redundant toarray + eigvals
        # that chebyshev() would otherwise perform internally.
        out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
        coeffs = out["coeffs"]
        rmin_cb, rmax_cb = out["rmin"], out["rmax"]
        m = len(coeffs)

        def _cheb_numpy(r):
            r = float(r)
            x = (2.0 * r - rmax_cb - rmin_cb) / (rmax_cb - rmin_cb)
            if m == 0:
                return 0.0
            if m == 1:
                return float(coeffs[0])
            b_next = 0.0
            b_curr = float(coeffs[m - 1])
            for k in range(m - 2, 0, -1):
                b_new = 2.0 * x * b_curr - b_next + float(coeffs[k])
                b_next = b_curr
                b_curr = b_new
            return float(coeffs[0]) + x * b_curr - b_next

        return _cheb_numpy

    elif method == "mc_poly":
        if sp.issparse(W_sparse):
            W_sp = W_sparse.tocsr().astype(np.float64)
        else:
            W_sp = sp.csr_matrix(np.asarray(W_sparse, dtype=np.float64))
        traces = compute_flow_traces(W_sp, miter=30, riter=50)
        m = len(traces)
        k_arr = np.arange(1, m + 1, dtype=np.float64)
        w = (traces / k_arr).astype(np.float64)

        def _mc_poly_numpy(r):
            r = float(r)
            result = w[m - 1]
            for j in range(m - 2, -1, -1):
                result = result * r + w[j]
            return -result * r

        return _mc_poly_numpy

    elif method in _GRID_SPLINE_METHODS:
        # Grid/spline methods: precompute numpy spline, then evaluate directly.
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
        spl = _build_grid_spline(
            W_dense,
            method,
            rho_min,
            rho_max,
            clamp_nonnegative=True,
        )
        return lambda r: float(spl(float(r)))

    else:
        # Fallback: exact numpy slogdet (slow but always correct)
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
        n_mat = W_dense.shape[0]
        I = np.eye(n_mat)
        return lambda r: float(np.linalg.slogdet(I - r * W_dense)[1])


def make_logdet_numpy_vec_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
):
    """Return a **vectorized** numpy ``(rho_arr: np.ndarray) -> np.ndarray`` logdet evaluator.

    Companion to :func:`make_logdet_numpy_fn` for batch evaluation over an
    array of posterior draws without a Python loop. Each supported method uses
    an array-native implementation.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        Row-standardised n×n spatial weights matrix.
    eigs : np.ndarray, optional
        Optional pre-computed real eigenvalues of W.
    method : str or None
        Same as :func:`make_logdet_numpy_fn`.
    rho_min : float, default -1.0
    rho_max : float, default 1.0

    Returns
    -------
    callable
        Function ``(rho_arr: np.ndarray) -> np.ndarray`` of shape ``(G,)``.
    """
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    if method is None:
        method = _auto_logdet_method(n)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(
                np.asarray(W_sparse.toarray(), dtype=np.float64)
            ).real
        _eigs = eigs.real.astype(np.float64)

        def _vec_eigenvalue(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            return np.sum(
                np.log(np.abs(1.0 - rho_arr[:, None] * _eigs[None, :])), axis=1
            )

        return _vec_eigenvalue

    if method == "chebyshev":
        out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
        coeffs = out["coeffs"].astype(np.float64)
        rmin_cb, rmax_cb = float(out["rmin"]), float(out["rmax"])
        m = len(coeffs)

        def _vec_chebyshev(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            x = (2.0 * rho_arr - rmax_cb - rmin_cb) / (rmax_cb - rmin_cb)

            if m == 0:
                return np.zeros_like(rho_arr, dtype=np.float64)
            if m == 1:
                return np.full_like(rho_arr, coeffs[0], dtype=np.float64)

            # Vectorized Clenshaw recurrence over all rho draws.
            b_next = np.zeros_like(x, dtype=np.float64)
            b_curr = np.full_like(x, coeffs[m - 1], dtype=np.float64)
            for k in range(m - 2, 0, -1):
                b_new = 2.0 * x * b_curr - b_next + coeffs[k]
                b_next = b_curr
                b_curr = b_new
            return coeffs[0] + x * b_curr - b_next

        return _vec_chebyshev

    if method == "mc_poly":
        if sp.issparse(W_sparse):
            W_sp = W_sparse.tocsr().astype(np.float64)
        else:
            W_sp = sp.csr_matrix(np.asarray(W_sparse, dtype=np.float64))
        traces = compute_flow_traces(W_sp, miter=30, riter=50)
        m = len(traces)
        k_arr = np.arange(1, m + 1, dtype=np.float64)
        w = (traces / k_arr).astype(np.float64)

        def _vec_mc_poly(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            if m == 0:
                return np.zeros_like(rho_arr, dtype=np.float64)
            # Evaluate p(r) = w1 + w2 r + ... + wm r^(m-1) with numpy polynomial,
            # then return -r * p(r) for the Barry-Pace truncated series.
            poly_val = np.polynomial.polynomial.polyval(rho_arr, w)
            return -rho_arr * poly_val

        return _vec_mc_poly

    if method in _GRID_SPLINE_METHODS:
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
        spl = _build_grid_spline(
            W_dense,
            method,
            rho_min,
            rho_max,
            clamp_nonnegative=True,
        )

        def _vec_grid(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            return np.asarray(spl(rho_arr), dtype=np.float64)

        return _vec_grid

    # Fallback: exact batched slogdet using stacked (G, n, n) matrices.
    W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
    n_mat = W_dense.shape[0]
    I = np.eye(n_mat, dtype=np.float64)

    def _vec_exact(rho_arr: np.ndarray) -> np.ndarray:
        rho_arr = np.asarray(rho_arr, dtype=np.float64)
        mats = I[None, :, :] - rho_arr[:, None, None] * W_dense[None, :, :]
        return np.linalg.slogdet(mats)[1]

    return _vec_exact


def make_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a function (rho) -> pytensor log|I - rho*W| expression.

    Parameters
    ----------
    W : np.ndarray or scipy.sparse matrix
        Either a 2-D dense ``(n, n)`` spatial weights matrix **or** a 1-D
        array of pre-computed real eigenvalues.  Passing eigenvalues skips the
        O(n³) decomposition inside this function; the ``'dense_grid'`` and
        ``'exact'`` methods are not available in that case and fall back to
        ``'eigenvalue'``.
    method : str
        Auto-selected when ``None`` (``"eigenvalue"`` for ``n <= 2000`` else
        ``"chebyshev"``). Supported values:

        ``"eigenvalue"`` — pre-compute W's eigenvalues once (O(n³)); every
        subsequent evaluation costs O(n) and is exact.
        ``"exact"`` — exact O(n³) symbolic det via pytensor (slow for n > 500).
        ``"dense_grid"`` — dense eigenvalue grid + cubic-spline interpolation.
        ``"sparse_grid"`` — sparse-LU grid + cubic-spline interpolation
        for large sparse W.
        ``"spline"`` — sparse-LU + cubic-spline interpolation on
        ``[max(rho_min, 0), rho_max]``.
        ``"mc"`` — Monte Carlo trace approximation
        (:cite:p:`barry1999MonteCarlo`) + spline interpolation.
        ``"ilu"`` — ILU-based approximation
        + spline interpolation.
        ``"chebyshev"`` — Chebyshev polynomial approximation
        (:cite:p:`pace2004ChebyshevApproximation`); near-minimax
        polynomial evaluated via Clenshaw's algorithm.
        O(m) per evaluation after O(n³) or O(R·n·m) pre-computation.
        ``"mc_poly"`` — Hutchinson stochastic trace estimator with a
        truncated polynomial expansion (used by the flow models).
    rho_min : float, default=-1.0
        Lower bound for the grid method.
    rho_max : float, default=1.0
        Upper bound for the grid method.
    T : int, default 1
        Panel time-period count.  The returned log-determinant is multiplied
        by *T*, exploiting
        ``log|I_{NT} - ρ(I_T⊗W_N)| = T · log|I_N - ρW_N|``.
        Leave at 1 for cross-sectional models.

    Returns
    -------
    callable
        Function mapping symbolic ``rho`` to symbolic log-determinant.

    Notes
    -----
    Per-method valid ``rho`` domains:

    * ``"eigenvalue"`` and ``"exact"`` — accept the full numerical
      stability domain ``rho ∈ (1/min(eigs), 1/max(eigs))``; for
      row-standardised W this reduces to ``rho ∈ (-1, 1)``.
    * ``"dense_grid"`` and ``"sparse_grid"`` — accept any
      ``rho ∈ (rho_min + 1e-6, rho_max - 1e-6)``; outside this interval
      the cubic spline extrapolates with rapidly degrading accuracy and
      **must not** be trusted.  Set ``rho_min`` / ``rho_max`` to match
      your prior bounds.
        * ``"spline"`` — restricted to ``rho ≥ 0`` because this routine builds
            its grid on
      ``[max(rho_min, 0), rho_max]``; passing negative ``rho`` is
      silently extrapolated, which is rarely intended.
    * ``"mc"`` — restricted to ``rho ≥ 1e-5`` for the same reason
      (``mc`` builds the grid on ``[max(rho_min, 1e-5), rho_max]``);
      negative ρ is supported only via separate calls with reversed
      sign on a row-standardised W.
    * ``"ilu"`` — accepts any ``rho`` inside the supplied grid bounds,
      but the ILU factorisation degrades with ``|rho|`` approaching the
      spectral boundary and should be paired with a tight prior.
    * ``"chebyshev"`` — exact within ``[rmin, rmax] = [rho_min, rho_max]``
      up to the polynomial order (default 20); evaluation outside the
      Chebyshev interval diverges rapidly and should never be attempted.
    * ``"mc_poly"`` — accuracy controlled by the polynomial order and
      number of Hutchinson probes; intended for very large sparse W
      where eigendecomposition is infeasible.
    """
    T = int(T)

    if sp.issparse(W):
        W_sparse = W.tocsr().astype(np.float64)
        if method is None:
            method = _auto_logdet_method(W_sparse.shape[0])

        if method == "chebyshev":
            out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max)
            coeffs_np = out["coeffs"]
            rmin_cb = out["rmin"]
            rmax_cb = out["rmax"]

            def _chebyshev_sparse_interp(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _chebyshev_sparse_interp

        # Other methods expect dense/eigenvalue inputs below.
        W = np.asarray(W_sparse.toarray(), dtype=np.float64)
    else:
        W = np.asarray(W, dtype=np.float64)

    if W.ndim == 1:
        # 1-D eigenvalue array supplied — skip O(n³) decomposition.
        eigs = W
        if method is None:
            method = "eigenvalue"
        if method == "eigenvalue":
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        if method == "chebyshev":
            # Build Chebyshev approximation directly from cached eigenvalues
            # (no dense W needed).
            out = chebyshev(None, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
            coeffs_np = out["coeffs"]
            rmin_cb = out["rmin"]
            rmax_cb = out["rmax"]

            def _chebyshev_eig_interp(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _chebyshev_eig_interp
        # Other methods (dense_grid, exact, sparse_grid, spline, mc, ilu)
        # require the full matrix; fall back to eigenvalue with a note.
        if method in ("dense_grid", "exact", "sparse_grid", "spline", "mc", "ilu"):
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        raise ValueError(
            f"Unknown method: {method!r}. Choose one of: 'eigenvalue', 'chebyshev', 'exact', 'dense_grid', 'sparse_grid', 'spline', 'mc', 'ilu'."
        )

    # 2-D dense matrix path.
    W_dense = W
    if method is None:
        method = _auto_logdet_method(W_dense.shape[0])
    if method in ("spline", "mc") and rho_min < 0.0:
        raise ValueError(
            f"method='{method}' is defined for nonnegative rho ranges; "
            "use rho_min >= 0 or choose 'eigenvalue'/'exact'/'dense_grid'/'sparse_grid'/'ilu'/'chebyshev'."
        )
    if method == "eigenvalue":
        eigs = np.linalg.eigvals(W_dense).real
        if T == 1:
            return lambda rho: logdet_eigenvalue(rho, eigs)
        return lambda rho: T * logdet_eigenvalue(rho, eigs)
    elif method == "exact":
        if T == 1:
            return lambda rho: logdet_exact(rho, W_dense)
        return lambda rho: T * logdet_exact(rho, W_dense)
    elif method in _GRID_SPLINE_METHODS:
        spline_obj = _build_grid_spline(
            W_dense,
            method,
            rho_min,
            rho_max,
            clamp_nonnegative=False,
        )
        return _make_pytensor_interp_fn(spline_obj, T)
    elif method == "chebyshev":
        out = chebyshev(W_dense, order=20, rmin=rho_min, rmax=rho_max)
        coeffs_np = out["coeffs"]
        rmin_cb = out["rmin"]
        rmax_cb = out["rmax"]

        def _chebyshev_interp(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _chebyshev_interp
    elif method == "mc_poly":
        W_sp = sp.csr_matrix(W_dense.astype(np.float64))
        traces = compute_flow_traces(W_sp, miter=30, riter=50)

        def _mc_poly_eval(rho):
            val = logdet_mc_poly_pytensor(rho, traces)
            return val if T == 1 else T * val

        return _mc_poly_eval
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose one of: 'eigenvalue', 'exact', 'dense_grid', 'sparse_grid', 'spline', 'mc', 'ilu', 'chebyshev', 'mc_poly'."
        )


def _hash_array(arr: np.ndarray) -> str:
    """Return a stable hash for numeric numpy arrays."""
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr_c.dtype).encode("ascii"))
    h.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _logdet_w_signature(W) -> tuple:
    """Build a stable cache signature for dense/sparse/eigenvalue W inputs."""
    if sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray(W_csr.shape, dtype=np.int64).tobytes())
        h.update(np.asarray([W_csr.nnz], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(W_csr.indptr).view(np.uint8))
        h.update(np.ascontiguousarray(W_csr.indices).view(np.uint8))
        h.update(np.ascontiguousarray(W_csr.data).view(np.uint8))
        return ("sparse", W_csr.shape, int(W_csr.nnz), h.hexdigest())

    W_arr = np.asarray(W, dtype=np.float64)
    if W_arr.ndim == 1:
        return ("eigs", W_arr.shape, _hash_array(W_arr))
    if W_arr.ndim == 2:
        return ("dense", W_arr.shape, _hash_array(W_arr))
    raise ValueError(f"Unsupported W with ndim={W_arr.ndim}; expected 1D or 2D.")


def clear_logdet_fn_cache() -> None:
    """Clear the shared cache of precomputed PyTensor logdet callables."""
    _LOGDET_FN_CACHE.clear()


def get_cached_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a shared cached ``make_logdet_fn`` callable.

    Cache key includes a stable signature of ``W`` plus ``method``, bounds,
    and panel multiplier ``T``. This avoids repeatedly rebuilding equivalent
    logdet approximations across model instances.
    """
    T = int(T)
    if method is None:
        if sp.issparse(W):
            n = int(W.shape[0])
            resolved_method = _auto_logdet_method(n)
        else:
            W_arr = np.asarray(W)
            if W_arr.ndim == 1:
                resolved_method = "eigenvalue"
            else:
                resolved_method = _auto_logdet_method(int(W_arr.shape[0]))
    else:
        resolved_method = method

    key = (
        _logdet_w_signature(W),
        resolved_method,
        float(rho_min),
        float(rho_max),
        T,
    )
    fn = _LOGDET_FN_CACHE.get(key)
    if fn is not None:
        _LOGDET_FN_CACHE.move_to_end(key)
        return fn

    fn = make_logdet_fn(
        W,
        method=resolved_method,
        rho_min=rho_min,
        rho_max=rho_max,
        T=T,
    )
    _LOGDET_FN_CACHE[key] = fn
    if len(_LOGDET_FN_CACHE) > _LOGDET_FN_CACHE_MAXSIZE:
        _LOGDET_FN_CACHE.popitem(last=False)
    return fn


def make_flow_separable_logdet(
    W_sparse,
    n: int,
    method: str | None = None,
    miter: int = 30,
    riter: int = 50,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    cheb_order: int = 20,
    random_state: int | None = None,
):
    r"""Pre-compute logdet data for separable flow models and return a logdet callable.

    For the separable constraint :math:`\rho_w = -\rho_d \rho_o` the full
    system log-determinant factors exactly as

    .. math::

        \log|L_o \otimes L_d|
        = n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    This function pre-computes the required data once at model initialisation
    and returns a closure that evaluates the expression as a symbolic
    :mod:`pytensor` scalar, suitable for ``pm.Potential``.

    Parameters
    ----------
    W_sparse : array-like or scipy.sparse matrix
        Row-standardised :math:`n \times n` spatial weights matrix.
    n : int
        Number of spatial units.
    method : str, default ``"eigenvalue"``
        ``"eigenvalue"`` — exact O(n) per-step evaluation after O(n³)
        eigendecomposition.  Exact for any rho.
        ``"chebyshev"`` — near-minimax Chebyshev polynomial; O(m) per step
        after O(n³) or O(R·n·m) precomputation via :func:`chebyshev`.
        ``"mc_poly"`` — Barry-Pace trace polynomial evaluated via Horner's
        method; O(miter) per step after O(riter·n·miter) stochastic
        precomputation via :func:`compute_flow_traces`.  Valid for
        :math:`\rho \in [-1, 1]`, unlike the grid-based :func:`mc`.
    miter : int, default 30
        Trace orders to estimate (``"mc_poly"`` only).
    riter : int, default 50
        Monte Carlo probe count (``"mc_poly"`` only).
    rho_min : float, default -1.0
        Lower bound of the rho domain (``"chebyshev"`` only).
    rho_max : float, default 1.0
        Upper bound of the rho domain (``"chebyshev"`` only).
    cheb_order : int, default 20
        Chebyshev polynomial order (``"chebyshev"`` only).
    random_state : int, optional
        Seed for MC trace estimation (``"mc_poly"`` only).

    Returns
    -------
    callable
        Function ``fn(rho_d, rho_o) -> pt.TensorVariable`` evaluating
        :math:`n\,f(\rho_d) + n\,f(\rho_o)` where
        :math:`f(\rho) = \log|I_n - \rho W|`.
    """
    if sp.issparse(W_sparse):
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
        W_sp = W_sparse.tocsr().astype(np.float64)
    else:
        W_dense = np.asarray(W_sparse, dtype=np.float64)
        W_sp = sp.csr_matrix(W_dense)

    if method is None:
        method = _auto_logdet_method(n)

    if method == "eigenvalue":
        eigs = np.linalg.eigvals(W_dense).real.astype(np.float64)
        return lambda rho_d, rho_o: (
            n * logdet_eigenvalue(rho_d, eigs) + n * logdet_eigenvalue(rho_o, eigs)
        )
    elif method == "chebyshev":
        out = chebyshev(W_dense, order=cheb_order, rmin=rho_min, rmax=rho_max)
        coeffs = out["coeffs"]
        rmin_cb = out["rmin"]
        rmax_cb = out["rmax"]
        return lambda rho_d, rho_o: (
            n * logdet_chebyshev(rho_d, coeffs, rmin=rmin_cb, rmax=rmax_cb)
            + n * logdet_chebyshev(rho_o, coeffs, rmin=rmin_cb, rmax=rmax_cb)
        )
    elif method == "mc_poly":
        traces = compute_flow_traces(
            W_sp, miter=miter, riter=riter, random_state=random_state
        )
        return lambda rho_d, rho_o: (
            n * logdet_mc_poly_pytensor(rho_d, traces)
            + n * logdet_mc_poly_pytensor(rho_o, traces)
        )
    else:
        raise ValueError(
            f"make_flow_separable_logdet: method={method!r} not recognised. "
            "Choose one of: 'eigenvalue', 'chebyshev', 'mc_poly'."
        )


def make_flow_separable_logdet_numpy(
    W_sparse,
    n: int,
    method: str | None = None,
    miter: int = 30,
    riter: int = 50,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    cheb_order: int = 20,
    random_state: int | None = None,
):
    r"""Pre-compute numeric logdet data for separable flow models.

    Returns a vectorized numpy closure for post-fit Jacobian reconstruction:

    .. math::

        n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    Parameters are aligned with :func:`make_flow_separable_logdet` for API
    symmetry, though ``miter``/``riter``/``cheb_order``/``random_state`` are
    currently controlled by the underlying method defaults.
    """
    if sp.issparse(W_sparse):
        W_sp = W_sparse.tocsr().astype(np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W_sparse, dtype=np.float64))

    if method is None:
        method = _auto_logdet_method(n)
    if method not in {"eigenvalue", "chebyshev", "mc_poly"}:
        raise ValueError(
            f"make_flow_separable_logdet_numpy: method={method!r} not recognised. "
            "Choose one of: 'eigenvalue', 'chebyshev', 'mc_poly'."
        )

    eigs = None
    if method == "eigenvalue":
        eigs = np.linalg.eigvals(np.asarray(W_sp.toarray(), dtype=np.float64)).real

    logdet_vec = make_logdet_numpy_vec_fn(
        W_sp,
        eigs,
        method=method,
        rho_min=rho_min,
        rho_max=rho_max,
    )

    def _eval(rho_d, rho_o) -> np.ndarray:
        rd = np.asarray(rho_d, dtype=np.float64).reshape(-1)
        ro = np.asarray(rho_o, dtype=np.float64).reshape(-1)
        if rd.shape != ro.shape:
            raise ValueError(
                "rho_d and rho_o must have the same shape for separable logdet evaluation."
            )
        return n * (logdet_vec(rd) + logdet_vec(ro))

    return _eval
