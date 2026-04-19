"""Log-determinant utilities for spatial models.

Provides log|I - rho*W| as a pytensor expression or a pre-computed
grid interpolation for large n, used as pm.Potential in spatial likelihoods.

The fastest approach for any n is the eigenvalue method: pre-computing
W's eigenvalues once (O(n³)) and evaluating sum(log(1 - rho*eigs)) per
step (O(n)), which is both exact and differentiable.
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


def lndetfull(W, lmin: float, lmax: float, grid: float = 0.01) -> dict:
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


def lndetint(W, rmin: float = 0.0, rmax: float = 1.0, n_grid: int = 100) -> dict:
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


def lndetmc(
    order: int,
    iter: int,
    W,
    rmin: float = 1e-5,
    rmax: float = 1.0,
    grid: float = 0.01,
    random_state: int | None = None,
) -> dict:
    """Compute Monte Carlo log-determinant approximation (MATLAB ``lndetmc`` style).

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


def lndetichol(
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
        ``"mc"``    — Monte Carlo trace approximation (MATLAB ``lndetmc`` style),
        then spline interpolation.
        ``"ichol"`` — ILU-based approximation (MATLAB ``lndetichol`` analog),
        then spline interpolation.
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
        if method in ("grid", "exact", "full", "int", "mc", "ichol"):
            method = "eigenvalue"
        if method == "eigenvalue":
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        raise ValueError(f"Unknown method: {method!r}.")

    # 2-D dense matrix path.
    W_dense = W
    if method in ("int", "mc") and rho_min < 0.0:
        raise ValueError(
            f"method='{method}' is defined for nonnegative rho ranges; "
            "use rho_min >= 0 or choose 'eigenvalue'/'exact'/'full'/'ichol'."
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
    elif method == "grid":
        rho_grid, logdet_grid = _build_logdet_grid(W_dense, rho_min, rho_max)
        spline = CubicSpline(rho_grid, logdet_grid)
        breakpoints_np = spline.x.astype(np.float64)
        coefficients_np = spline.c.astype(np.float64)
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
    elif method == "full":
        out = lndetfull(W_dense, rho_min, rho_max)
        spline = CubicSpline(out["rho"], out["lndet"])

        def _full_interp(rho):
            val = pt.as_tensor_variable(np.float64(0.0))
            bp = pt.as_tensor_variable(spline.x.astype(np.float64))
            c = pt.as_tensor_variable(spline.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _full_interp
    elif method == "int":
        out = lndetint(W_dense, rho_min, rho_max)
        spline = CubicSpline(out["rho"], out["lndet"])

        def _int_interp(rho):
            bp = pt.as_tensor_variable(spline.x.astype(np.float64))
            c = pt.as_tensor_variable(spline.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _int_interp
    elif method == "mc":
        out = lndetmc(order=50, iter=30, W=W_dense, rmin=rho_min, rmax=rho_max)
        spline = CubicSpline(out["rho"], out["lndet"])

        def _mc_interp(rho):
            bp = pt.as_tensor_variable(spline.x.astype(np.float64))
            c = pt.as_tensor_variable(spline.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _mc_interp
    elif method == "ichol":
        out = lndetichol(W_dense, rho_min, rho_max)
        spline = CubicSpline(out["rho"], out["lndet"])

        def _ichol_interp(rho):
            bp = pt.as_tensor_variable(spline.x.astype(np.float64))
            c = pt.as_tensor_variable(spline.c.astype(np.float64))
            idx = pt.cast(pt.floor((rho - bp[0]) / (bp[1] - bp[0])), "int64")
            idx = pt.clip(idx, 0, len(spline.x) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            val = coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]
            return val if T == 1 else T * val

        return _ichol_interp
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Choose 'eigenvalue', 'exact', 'grid', 'full', 'int', 'mc', or 'ichol'."
        )
