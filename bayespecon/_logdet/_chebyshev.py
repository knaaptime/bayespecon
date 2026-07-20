"""Chebyshev polynomial approximation of log|I - ρW|.

Two computation strategies:

* **Eigenvalue-based** (n ≤ 2000 or eigenvalues supplied): exact evaluation
  at Chebyshev nodes via eigendecomposition, then DCT-I for coefficients.
* **Monte-Carlo trace-based** (n > 2000): Barry-Pace Hutchinson probes
  (:cite:t:`barry1999MonteCarlo`) estimate tr(W^k), avoiding O(n³).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Barry-Pace trace estimation (shared core)
# ---------------------------------------------------------------------------


def _barry_pace_traces(
    W_sparse: sp.csr_matrix,
    order: int,
    iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate tr(W^k) for k=1..order via Barry-Pace Monte Carlo probes.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Sparse n×n spatial weights matrix.
    order : int
        Maximum trace power to estimate.
    iter : int
        Number of Monte Carlo probes (random vectors).
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray, shape (order, iter)
        Per-probe trace estimates.  Entry ``[k, j]`` estimates ``tr(W^{k+1})``
        from probe *j*.  Rows 0 and 1 are overridden with exact values.
    """
    n = W_sparse.shape[0]
    U = rng.standard_normal((n, iter))
    utu = np.einsum("ij,ij->j", U, U)
    V = U.copy()
    traces = np.empty((order, iter), dtype=np.float64)
    for i in range(order):
        V = W_sparse @ V
        traces[i] = n * np.einsum("ij,ij->j", U, V) / utu
    traces[0, :] = float(W_sparse.diagonal().sum())
    if order >= 2:
        traces[1, :] = float(W_sparse.multiply(W_sparse.T).sum())
    return traces


# ---------------------------------------------------------------------------
# Chebyshev coefficient builder
# ---------------------------------------------------------------------------


def chebyshev_gauss_nodes(order: int, lo: float, hi: float):
    """Chebyshev-Gauss nodes ``cos((2k-1)\u03c0/(2\u00b7order))`` mapped to ``[lo, hi]``.

    Returns ``(mapped_nodes, nodes_cos)`` for ``k = 1..order``.  Pair with
    :func:`chebyshev_coeffs_dct1` — the single source for the node/DCT-I
    combination used by every Chebyshev-in-\u03c1 logdet surrogate.
    """
    k = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    return 0.5 * (hi - lo) * nodes_cos + 0.5 * (hi + lo), nodes_cos


def chebyshev_coeffs_dct1(values: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients from values at Chebyshev-Gauss nodes (DCT-I).

    ``values[k-1] = f(x_k)`` at the nodes from :func:`chebyshev_gauss_nodes`;
    returns ``coeffs`` such that ``f(x) \u2248 \u03a3_j coeffs[j] T_j(x)``.
    """
    values = np.asarray(values, dtype=np.float64)
    order = len(values)
    k = np.arange(1, order + 1)
    coeffs = np.zeros(order, dtype=np.float64)
    for j in range(order):
        scale = 2.0 / order if j > 0 else 1.0 / order
        coeffs[j] = scale * np.sum(
            values * np.cos(j * (2 * k - 1) * np.pi / (2 * order))
        )
    return coeffs


def chebyshev(
    W,
    order: int = 20,
    rmin: float = -1.0,
    rmax: float = 1.0,
    random_state: int | None = None,
    eigs: np.ndarray | None = None,
    n_mc_iter: int = 100,
) -> dict:
    """Compute Chebyshev approximation of log|I - ρW| (:cite:p:`pace2004ChebyshevApproximation`).

    Near-minimax polynomial approximation over ``[rmin, rmax]``.

    Parameters
    ----------
    W : array-like
        Spatial weights matrix (dense or sparse).
    order : int, default 20
        Polynomial degree.  15–30 is usually sufficient.
    rmin : float, default -1.0
        Lower bound of the rho interval.
    rmax : float, default 1.0
        Upper bound of the rho interval.
    random_state : int, optional
        Seed for MC trace estimation (only when n > 2000 and no eigs).
    eigs : np.ndarray, optional
        Pre-computed eigenvalues of W (skips O(n³) decomposition).
    n_mc_iter : int, default 100
        Number of Hutchinson probes for the MC path.

    Returns
    -------
    dict
        ``{"coeffs", "rmin", "rmax", "order", "method"}`` where ``method``
        is ``"eigenvalue"`` or ``"mc"``.
    """
    if order <= 0:
        raise ValueError("order must be positive.")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")

    if eigs is not None:
        eigs_arr = np.asarray(eigs, dtype=np.complex128)
        n = int(eigs_arr.shape[0])
        W_sp = None
    else:
        if sp.issparse(W) or hasattr(W, "format"):
            W_sp = sp.csr_matrix(W)
        else:
            W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
        n = W_sp.shape[0]
        eigs_arr = None

    # Chebyshev nodes on [-1, 1], mapped to [rmin, rmax]
    k = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    rho_nodes = 0.5 * (rmax - rmin) * nodes_cos + 0.5 * (rmax + rmin)

    use_mc = (eigs_arr is None) and (n > 2000)

    if not use_mc:
        if eigs_arr is None:
            eigs_arr = np.linalg.eigvals(W_sp.toarray())
        logdet_at_nodes = np.sum(
            np.log(np.abs(1.0 - rho_nodes[:, None] * eigs_arr[None, :])), axis=1
        )
        method_used = "eigenvalue"
    else:
        rng = np.random.default_rng(random_state)
        traces = _barry_pace_traces(W_sp, order, n_mc_iter, rng)
        td = traces.mean(axis=1) / np.arange(1, order + 1)
        logdet_at_nodes = np.zeros(order, dtype=np.float64)
        for idx, r in enumerate(rho_nodes):
            powers = np.power(r, np.arange(1, order + 1, dtype=np.float64))
            logdet_at_nodes[idx] = -powers @ td
        method_used = "mc"

    # Chebyshev coefficients via DCT-I
    coeffs = chebyshev_coeffs_dct1(logdet_at_nodes)

    result = {
        "coeffs": coeffs,
        "rmin": rmin,
        "rmax": rmax,
        "order": order,
        "method": method_used,
    }

    return result
