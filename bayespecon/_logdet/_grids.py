"""Grid builders and Chebyshev coefficient computation for log-determinant approximation.

Contains functions that pre-compute log-determinant values on a rho grid or
compute Chebyshev polynomial coefficients.  These are NumPy/SciPy-only — no
PyTensor or JAX dependencies.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

from ._config import _LOGDET_GRID_EIG_MAX


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
    spline_obj = CubicSpline(x, y, extrapolate=False)
    lndet = spline_obj(rho)
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
        ilu_obj = spla.spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
        lndet[i] = np.sum(np.log(np.abs(ilu_obj.L.diagonal()))) + np.sum(
            np.log(np.abs(ilu_obj.U.diagonal()))
        )

    return {"rho": rho, "lndet": lndet}


def chebyshev(
    W,
    order: int = 20,
    rmin: float = -1.0,
    rmax: float = 1.0,
    random_state: int | None = None,
    eigs: np.ndarray | None = None,
    n_mc_iter: int = 100,
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
    n_mc_iter : int, default=100
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
        - ``method`` : ``'eigenvalue'`` or ``'grid_mc'`` indicating how
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
        # Keep complex eigenvalues — np.abs computes the complex modulus
        # correctly for non-symmetric W.
        eigs_arr = np.asarray(eigs, dtype=np.complex128)
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
            eigs_arr = np.linalg.eigvals(W_sp.toarray())
        logdet_at_nodes = np.sum(
            np.log(np.abs(1.0 - rho_nodes[:, None] * eigs_arr[None, :])), axis=1
        )
        method_used = "eigenvalue"
    else:
        # Monte Carlo trace-based via Barry-Pace Hutchinson probes
        # (:cite:t:`barry1999MonteCarlo`).
        # ln|I - ρW| = -Σ_{k=1}^{∞} (ρ^k / k) tr(W^k)
        # Approximate tr(W^k) via MC, then evaluate at nodes.
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
        method_used = "grid_mc"

        # Evaluate power series at each node
        logdet_at_nodes = np.zeros(order, dtype=np.float64)
        for idx, r in enumerate(rho_nodes):
            powers = np.power(r, np.arange(1, order + 1, dtype=np.float64))
            logdet_at_nodes[idx] = -powers @ td

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
