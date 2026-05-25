"""PyTensor symbolic log-determinant evaluation functions.

These return ``pt.TensorVariable`` and are used inside PyMC models as
``pm.Potential``.
"""

import numpy as np
import pytensor.tensor as pt
from scipy.interpolate import CubicSpline

from ._grids import _build_logdet_grid


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
