"""Log-determinant utilities for spatial models.

Provides log|I - rho*W| as a pytensor expression or a pre-computed
grid interpolation for large n, used as pm.Potential in spatial likelihoods.

"""

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline


def _stable_rho_grid(rmin: float, rmax: float, grid: float) -> np.ndarray:
    """Build a rho grid that excludes exact endpoints to avoid singularity hits."""
    if grid <= 0:
        raise ValueError("grid must be positive.")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")
    eps = 1e-6
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
    """
    eigs_t = pt.as_tensor_variable(eigs.astype(np.float64))
    return pt.sum(pt.log(pt.abs(1.0 - rho * eigs_t)))


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


def _build_logdet_grid(W_dense: np.ndarray, rho_min: float, rho_max: float, n_grid: int = 200):
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
    I = np.eye(W_dense.shape[0])
    # Vectorised batched slogdet — faster than a sequential Python loop.
    A = I[np.newaxis] - rho_grid[:, np.newaxis, np.newaxis] * W_dense[np.newaxis]
    _, logdet_grid = np.linalg.slogdet(A)
    return rho_grid, logdet_grid


def sparse_grid(W, lmin: float, lmax: float, grid: float = 0.01) -> dict:
    """Compute exact sparse-LU log-determinant grid (MATLAB ``lndetfull`` style).

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
    """Compute spline-interpolated log-determinant grid (MATLAB ``lndetint`` style).

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
        raise ValueError("lndetint is defined for nonnegative rho ranges (rmin >= 0).")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")

    W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]
    I = sp.eye(n, format="csc", dtype=np.float64)

    rho = np.linspace(rmin, rmax, n_grid, endpoint=False, dtype=np.float64)
    # Follow the original control-point pattern from the MATLAB routine.
    ctrl = np.array([10, 20, 40, 50, 60, 70, 80, 85, 90, 95, 96, 97, 98, 99, 100], dtype=int) - 1
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
        raise ValueError("lndetmc is defined for nonnegative rho ranges (rmin >= 0).")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")
    W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]

    td = np.array([0.0, float(W_sp.multiply(W_sp).sum() / 2.0)], dtype=np.float64)
    oexact = len(td)

    rng = np.random.default_rng(random_state)
    mavmomi = np.zeros((order, iter), dtype=np.float64)

    for j in range(iter):
        u = rng.standard_normal(n)
        v = u.copy()
        utu = float(u @ u)
        for i in range(order):
            v = W_sp @ v
            # Barry-Pace moment construction uses the 1/k scaling here.
            mavmomi[i, j] = n * float(u @ v) / ((i + 1) * utu)

    mavmomi[:oexact, :] = td[:, None]
    avmomi = mavmomi.mean(axis=1)

    rho = _stable_rho_grid(rmin, rmax, grid)
    # Build polynomial terms alpha^1..alpha^order.
    powers = np.power(rho[:, None], np.arange(1, order + 1, dtype=np.float64)[None, :])
    alomat = -powers

    lndet = alomat @ avmomi
    srvs = (alomat @ mavmomi).T
    sderr = np.sqrt(np.maximum(0.0, srvs.var(axis=0, ddof=0) / iter))

    fbound = (n * np.power(rho, order + 1)) / ((order + 1) * (1.0 - rho + np.finfo(float).eps))
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
    """Compute ILU-based approximate log-determinant grid (MATLAB ``lndetichol`` analog).

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
        lndet[i] = np.sum(np.log(np.abs(ilu.L.diagonal()))) + np.sum(np.log(np.abs(ilu.U.diagonal())))

    return {"rho": rho, "lndet": lndet}


def chebyshev(
    W,
    order: int = 20,
    rmin: float = -1.0,
    rmax: float = 1.0,
    random_state: int | None = None,
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
    """
    if order <= 0:
        raise ValueError("order must be positive.")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")

    W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]

    # Chebyshev nodes on [-1, 1], mapped to [rmin, rmax]
    k = np.arange(1, order + 1)
    # Nodes: cos((2k-1)π / (2m)) for k=1..m  (Chebyshev nodes of the first kind)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    # Map from [-1, 1] to [rmin, rmax]
    rho_nodes = 0.5 * (rmax - rmin) * nodes_cos + 0.5 * (rmax + rmin)

    # Decide computation strategy
    # Only fall back to MC for truly large matrices; for small matrices
    # eigenvalue decomposition is fast and exact.
    use_mc = n > 2000

    if not use_mc:
        # Eigenvalue-based: exact log-determinant at each Chebyshev node
        eigs = np.linalg.eigvals(W_sp.toarray()).real
        logdet_at_nodes = np.array(
            [np.sum(np.log(np.abs(1.0 - r * eigs))) for r in rho_nodes]
        )
        method_used = "eigenvalue"
    else:
        # Monte Carlo trace-based: use Barry-Pace stochastic trace
        # ln|I - ρW| = -Σ_{k=1}^{∞} (ρ^k / k) tr(W^k)
        # Approximate tr(W^k) via MC, then evaluate at nodes
        n_mc_iter = 30
        rng = np.random.default_rng(random_state)

        # Compute MC trace estimates for k=1..order
        td = np.zeros(order, dtype=np.float64)
        for j in range(n_mc_iter):
            u = rng.standard_normal(n)
            v = u.copy()
            utu = float(u @ u)
            for i in range(order):
                v = W_sp @ v
                td[i] += n * float(u @ v) / ((i + 1) * utu)
        td /= n_mc_iter

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
    b_curr = c[m - 1]              # b_m = c_{m-1}

    # Iterate from k = m-2 down to k = 1
    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + c[k]
        b_next = b_curr
        b_curr = b_new

    # f(x) = c_0 + x*b_1 - b_2
    # After the loop, b_curr = b_1, b_next = b_2
    return c[0] + x * b_curr - b_next


def logdet_interpolated(rho, W_dense: np.ndarray, rho_min: float = -1.0, rho_max: float = 1.0, n_grid: int = 200):
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
    coefficients = pt.as_tensor_variable(spline.c.astype(np.float64))  # (4, n_intervals)

    # Find the interval index for rho
    idx = pt.sum(pt.lt(breakpoints, rho)) - 1
    idx = pt.clip(idx, 0, len(spline.x) - 2)

    dx = rho - breakpoints[idx]
    c = coefficients[:, idx]  # shape (4,)
    # Evaluate cubic: c[0]*dx^3 + c[1]*dx^2 + c[2]*dx + c[3]
    value = c[0] * dx**3 + c[1] * dx**2 + c[2] * dx + c[3]
    return value


def make_logdet_fn(W, method: str = "eigenvalue", rho_min: float = -1.0, rho_max: float = 1.0, T: int = 1):
    # Map legacy method names to new ones for backward compatibility
    legacy_map = {
        "grid": "dense_grid",
        "full": "sparse_grid",
        "int": "spline",
        "mc": "mc",
        "ichol": "ilu",
    }
    if method in legacy_map:
        method = legacy_map[method]
    """Return a function (rho) -> pytensor log|I - rho*W| expression.

    Parameters
    ----------
    W : np.ndarray
        Either a 2-D dense ``(n, n)`` spatial weights matrix **or** a 1-D
        array of pre-computed real eigenvalues.  Passing eigenvalues skips the
        O(n³) decomposition inside this function; the ``'grid'`` and
        ``'exact'`` methods are not available in that case and fall back to
        ``'eigenvalue'``.
    method : str
        ``"eigenvalue"`` — pre-compute W's eigenvalues once (O(n³)); every
        subsequent evaluation costs O(n) and is exact (default).
        ``"exact"`` — exact O(n³) symbolic det via pytensor (slow for n > 500).
        ``"grid"``  — spline interpolation over pre-computed grid (approximate).
        ``"full"``  — exact sparse-LU grid (MATLAB ``lndetfull`` style),
        then spline interpolation.
        ``"int"``   — sparse-LU + cubic-spline interpolation (MATLAB
        ``lndetint`` style).
        ``"mc"``    — Monte Carlo trace approximation (:cite:p:`barry1999MonteCarlo`),
        then spline interpolation.
        ``"ichol"`` — ILU-based approximation (MATLAB ``lndetichol`` analog),
        then spline interpolation.
        ``"chebyshev"`` — Chebyshev polynomial approximation
        (:cite:p:`pace2004ChebyshevApproximation`); near-minimax
        polynomial approximation evaluated via Clenshaw's algorithm.
        O(m) per evaluation after O(n³) or O(R·n·m) pre-computation.
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
    """
    T = int(T)
    W = np.asarray(W, dtype=np.float64)

    if W.ndim == 1:
        # 1-D eigenvalue array supplied — skip O(n³) decomposition.
        eigs = W
        if method in ("dense_grid", "exact", "sparse_grid", "spline", "mc", "ilu", "chebyshev"):
            method = "eigenvalue"
        if method == "eigenvalue":
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        raise ValueError(f"Unknown method: {method!r}. Choose one of: 'eigenvalue', 'exact', 'dense_grid', 'sparse_grid', 'spline', 'mc', 'ilu'.")

    # 2-D dense matrix path.
    W_dense = W
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
    elif method == "dense_grid":
        rho_grid, logdet_grid = _build_logdet_grid(W_dense, rho_min, rho_max)
        spline_obj = CubicSpline(rho_grid, logdet_grid)
        breakpoints_np = spline_obj.x.astype(np.float64)
        coefficients_np = spline_obj.c.astype(np.float64)
        # Uniform grid step enables O(1) index lookup instead of O(n_grid) scan.
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
    elif method == "sparse_grid":
        out = sparse_grid(W_dense, rho_min, rho_max)
        spline_obj = CubicSpline(out["rho"], out["lndet"])

        def _sparse_grid_interp(rho):
            bp = pt.as_tensor_variable(spline_obj.x.astype(np.float64))
            c = pt.as_tensor_variable(spline_obj.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline_obj.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _sparse_grid_interp
    elif method == "spline":
        out = spline(W_dense, rho_min, rho_max)
        spline_obj = CubicSpline(out["rho"], out["lndet"])

        def _spline_interp(rho):
            bp = pt.as_tensor_variable(spline_obj.x.astype(np.float64))
            c = pt.as_tensor_variable(spline_obj.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline_obj.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _spline_interp
    elif method == "mc":
        out = mc(order=50, iter=30, W=W_dense, rmin=rho_min, rmax=rho_max)
        spline_obj = CubicSpline(out["rho"], out["lndet"])

        def _mc_interp(rho):
            bp = pt.as_tensor_variable(spline_obj.x.astype(np.float64))
            c = pt.as_tensor_variable(spline_obj.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline_obj.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _mc_interp
    elif method == "ilu":
        out = ilu(W_dense, rho_min, rho_max)
        spline_obj = CubicSpline(out["rho"], out["lndet"])

        def _ilu_interp(rho):
            bp = pt.as_tensor_variable(spline_obj.x.astype(np.float64))
            c = pt.as_tensor_variable(spline_obj.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline_obj.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _ilu_interp
    elif method == "chebyshev":
        out = chebyshev(W_dense, order=20, rmin=rho_min, rmax=rho_max)
        coeffs_np = out["coeffs"]
        rmin_cb = out["rmin"]
        rmax_cb = out["rmax"]

        def _chebyshev_interp(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _chebyshev_interp
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose one of: 'eigenvalue', 'exact', 'dense_grid', 'sparse_grid', 'spline', 'mc', 'ilu', 'chebyshev'."
        )
