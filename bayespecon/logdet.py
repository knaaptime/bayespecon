"""Log-determinant utilities for spatial models.

Provides log|I - rho*W| as a pytensor expression or a pre-computed
grid interpolation for large n, used as pm.Potential in spatial likelihoods.

The fastest approach for any n is the eigenvalue method: pre-computing
W's eigenvalues once (O(n³)) and evaluating sum(log(1 - rho*eigs)) per
step (O(n)), which is both exact and differentiable.
"""

import numpy as np
import pytensor.tensor as pt
from scipy.interpolate import CubicSpline


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


def make_logdet_fn(W, method: str = "auto", rho_min: float = -1.0, rho_max: float = 1.0, T: int = 1):
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
        ``"auto"``  — same as ``"eigenvalue"``.
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
        if method in ("grid", "exact"):
            method = "eigenvalue"
        if method in ("auto", "eigenvalue"):
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        raise ValueError(f"Unknown method: {method!r}.")

    # 2-D dense matrix path.
    W_dense = W
    if method == "auto":
        method = "eigenvalue"

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
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'eigenvalue', 'exact', 'grid', or 'auto'.")
