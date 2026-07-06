"""PyTensor symbolic log-determinant evaluation.

These return ``pt.TensorVariable`` and are used inside PyMC models as
``pm.Potential``.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt


def logdet_eigenvalue(rho, eigs: np.ndarray) -> pt.TensorVariable:
    """Eigenvalue-based log|I - ρW|.

    Pre-compute ``eigs = np.linalg.eigvals(W)`` once; each evaluation
    costs O(n) and is exactly differentiable.

    For complex eigenvalues ``λᵢ = a + bi``::

        |1 - ρλᵢ|² = (1 - ρa)² + (ρb)²
        log|1 - ρλᵢ| = 0.5 * log((1 - ρa)² + (ρb)²)

    This avoids ``pt.abs`` on complex tensors (no gradient in PyTensor).
    """
    eigs_arr = np.asarray(eigs)
    eigs_real = eigs_arr.real.astype(np.float64)
    eigs_imag = eigs_arr.imag.astype(np.float64)

    eigs_real_t = pt.as_tensor_variable(eigs_real)
    eigs_imag_t = pt.as_tensor_variable(eigs_imag)

    re = 1.0 - rho * eigs_real_t
    im = rho * eigs_imag_t
    mod_sq = re**2 + im**2
    safe = pt.maximum(mod_sq, 1e-300)
    return 0.5 * pt.sum(pt.log(safe))


def logdet_chebyshev(
    rho,
    coeffs: np.ndarray,
    rmin: float = -1.0,
    rmax: float = 1.0,
) -> pt.TensorVariable:
    """Evaluate Chebyshev approximation of log|I - ρW| via Clenshaw's recurrence.

    Parameters
    ----------
    rho : pytensor scalar
        Spatial parameter symbol.
    coeffs : np.ndarray
        Chebyshev coefficients from :func:`chebyshev`, shape ``(m,)``.
    rmin, rmax : float
        Interval bounds (must match what was used to compute *coeffs*).
    """
    m = len(coeffs)
    if m == 0:
        return pt.zeros_like(rho)

    x = (2.0 * rho - rmax - rmin) / (rmax - rmin)
    c = pt.as_tensor_variable(coeffs.astype(np.float64))

    if m == 1:
        return c[0] * pt.ones_like(rho)

    b_next = pt.zeros_like(rho)
    b_curr = c[m - 1]

    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + c[k]
        b_next = b_curr
        b_curr = b_new

    return c[0] + x * b_curr - b_next
