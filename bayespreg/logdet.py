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
    n = W_dense.shape[0]
    I = np.eye(n)
    rho_grid = np.linspace(rho_min + 1e-6, rho_max - 1e-6, n_grid)
    logdet_grid = np.array([np.linalg.slogdet(I - r * W_dense)[1] for r in rho_grid])
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


def make_logdet_fn(W_dense: np.ndarray, method: str = "auto", rho_min: float = -1.0, rho_max: float = 1.0):
    """Return a function (rho) -> pytensor log|I - rho*W| expression.

    Parameters
    ----------
    W_dense : np.ndarray
        Dense spatial weights matrix.
    method : str
        "eigenvalue" — pre-compute W's eigenvalues once (O(n³)); every
        subsequent evaluation costs O(n) and is exact (default).
        "exact" — exact O(n³) symbolic det via pytensor (slow for n > 500).
        "grid"  — spline interpolation over pre-computed grid (approximate).
        "auto"  — same as "eigenvalue".

    rho_min : float, default=-1.0
        Lower bound for the grid method.
    rho_max : float, default=1.0
        Upper bound for the grid method.

    Returns
    -------
    callable
        Function mapping symbolic ``rho`` to symbolic log-determinant.
    """
    n = W_dense.shape[0]
    if method == "auto":
        method = "eigenvalue"

    if method == "eigenvalue":
        eigs = np.linalg.eigvals(W_dense).real
        return lambda rho: logdet_eigenvalue(rho, eigs)
    elif method == "exact":
        return lambda rho: logdet_exact(rho, W_dense)
    elif method == "grid":
        # Pre-compute grid once
        rho_grid, logdet_grid = _build_logdet_grid(W_dense, rho_min, rho_max)
        spline = CubicSpline(rho_grid, logdet_grid)
        breakpoints_np = spline.x.astype(np.float64)
        coefficients_np = spline.c.astype(np.float64)

        def _interp(rho):
            bp = pt.as_tensor_variable(breakpoints_np)
            c = pt.as_tensor_variable(coefficients_np)
            idx = pt.sum(pt.lt(bp, rho)) - 1
            idx = pt.clip(idx, 0, len(breakpoints_np) - 2)
            dx = rho - bp[idx]
            coefs = c[:, idx]
            return coefs[0] * dx**3 + coefs[1] * dx**2 + coefs[2] * dx + coefs[3]

        return _interp
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'eigenvalue', 'exact', 'grid', or 'auto'.")
