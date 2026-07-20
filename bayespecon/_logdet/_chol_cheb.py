"""Cholesky-Chebyshev log-determinant for SPD ``I - ρW``.

For row-standardised ``W`` with spectrum in ``[-1, 1]``, the D-symmetrised
matrix ``W_sym = D^{-1/2} A D^{-1/2}`` is symmetric with the same eigenvalues
as ``W``.  This makes ``I - ρW_sym`` **symmetric positive definite** (SPD)
for all ``|ρ| < 1``, enabling **sparse Cholesky** factorisation:

* **Exact**: ``log|det(I - ρW)| = 2 Σ log(diag(L))`` — no stochastic noise.
* **Fast**: CHOLMOD sparse Cholesky is ~2× faster than sparse LU for SPD.
* **Scalable**: ``O(nnz^{1.5})`` per factorisation, no ``O(n³)`` eigendecomposition.
* **Full range**: works for ``ρ ∈ (-1, 1)`` — the entire stable region.
  Adaptive order selection automatically increases the Chebyshev degree for
  wider intervals (15 for ``[0.1, 0.8]``, 50 for ``[-0.95, 0.95]``, 100 for
  ``[-0.99, 0.99]``).

The method evaluates ``log|det(I - ρW)|`` exactly at ``order`` Chebyshev nodes
in ``[ρ_min, ρ_max]``, then fits a Chebyshev polynomial in ``ρ`` for ``O(order)``
per-``ρ`` evaluation via Clenshaw recurrence.

**Symbolic reuse**: all ``I - ρW`` matrices share the same sparsity pattern,
so CHOLMOD's symbolic analysis (AMD ordering + elimination tree) is performed
only once and reused for all subsequent numeric factorisations via
``factor.factorize()``.  This saves ~64% of per-node cost.

**When to use**: ``n ∈ (500, 20000]``, any ``ρ ∈ (-1, 1)``.  For ``n ≤ 500``
use ``eigenvalue`` (exact eigendecomposition).  For ``n > 20000`` use
``cheb_stochastic`` (avoids ``O(nnz^{1.5})`` Cholesky fill-in).  For
non-symmetric ``W`` (directed graphs: KNN, travel time) use ``aaa`` (rational
approximation via sparse LU).

**Benchmark** (2D rook grid, adaptive order, ρ ∈ [0.1, 0.8], 2026-07):

========== ============= ============= =========== ==================
n          chol setup    chol eval     chol error  stoch(200)
========== ============= ============= =========== ==================
484        9ms           1.3μs         2e-7        2.5ms, 0.46 err
4,900      91ms          1.3μs         2e-6        25ms, 0.74 err
10,000     194ms         1.3μs         3e-6        53ms, 0.75 err
20,000     432ms         1.4μs         6e-6        87ms, 1.7 err
40,000     997ms         1.4μs         1e-5        207ms, 1.8 err
60,000     2.2s          1.4μs         2e-5        331ms, 3.5 err
========== ============= ============= =========== ==================

Cholesky-Chebyshev is the accuracy leader across this range: exact (2e-7 to
2e-5 vs 0.5-3.5 for stochastic) and ~40× faster eval (1.3μs vs ~55μs).  Its
setup grows with Cholesky fill-in (≈2× the stochastic cost by n≈40k), the
crossover past which ``cheb_stochastic`` trades accuracy for cheaper setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._chebyshev import chebyshev_coeffs_dct1
from ._clenshaw import clenshaw_scalar, clenshaw_vec


@dataclass(frozen=True)
class CholChebPrecompute:
    """Precomputed Chebyshev coefficients from Cholesky log-determinant.

    Attributes
    ----------
    coeffs : np.ndarray, shape (order,)
        Chebyshev coefficients of ``log|I - ρW|`` in ``ρ``.
    rho_min : float
        Lower bound of the ρ approximation interval.
    rho_max : float
        Upper bound of the ρ approximation interval.
    order : int
        Chebyshev polynomial degree.
    n : int
        Matrix dimension.
    """

    coeffs: np.ndarray
    rho_min: float
    rho_max: float
    order: int
    n: int


def _adaptive_order(rho_min: float, rho_max: float) -> int:
    """Choose Chebyshev order based on interval width.

    The logdet function ``f(ρ) = Σ log(1 - ρλᵢ)`` has singularities at
    ``ρ = 1/λᵢ`` for each eigenvalue ``λᵢ`` of ``W``.  For row-standardised
    contiguity matrices, ``λ_max = 1`` (Perron) and ``λ_min ≈ -1``, so the
    closest singularities are at ``ρ = ±1``.  Chebyshev convergence depends
    on the distance from the interval to the nearest singularity.

    Parameters
    ----------
    rho_min, rho_max : float
        The ρ approximation interval.

    Returns
    -------
    int
        Chebyshev polynomial order (number of nodes).
    """
    width = rho_max - rho_min
    # Distance from interval to nearest singularity at ±1
    dist = min(abs(1.0 - rho_max), abs(1.0 + rho_min))
    if dist <= 0.01:
        return 50  # interval nearly touches singularity
    if width <= 0.71:
        return 15  # [0.1, 0.8] — standard empirical range
    if width <= 1.5:
        return 30  # e.g. [-0.5, 0.95]
    return 50  # [-0.95, 0.95] or wider


def _d_symmetrize(W: sp.csr_matrix) -> sp.csc_matrix:
    """D-symmetrise row-standardised ``W``.

    For ``W = D⁻¹A`` (row-standardised, ``A`` symmetric adjacency),
    ``W_sym = D^{1/2} W D^{-1/2} = D^{-1/2} A D^{-1/2}`` is symmetric
    with the **same eigenvalues** as ``W``.

    The symmetrizing degrees are recovered from the *values* of ``W``
    via the edge ratios ``D[i]/D[j] = W[j,i]/W[i,j]`` (BFS propagation).
    The neighbor count (``getnnz``) equals the standardizing degree only
    for binary adjacency — using it for weighted graphs breaks the
    symmetry that CHOLMOD relies on (it reads a single triangle) and
    silently corrupts the log-determinant.

    This makes ``I - ρW_sym`` SPD for ``|ρ| < 1``, enabling sparse Cholesky.

    Raises
    ------
    ValueError
        If no symmetrizing diagonal exists (directed graph, or weights
        inconsistent with ``W = D⁻¹A`` for symmetric ``A``).  Pass
        ``logdet_method="aaa"`` for such matrices.
    """
    n = W.shape[0]
    W = sp.csr_matrix(W)

    # Fast path: W already symmetric — no scaling needed.
    diff = (W - W.T).tocoo()
    if diff.nnz == 0 or np.all(np.abs(diff.data) <= 1e-12):
        return sp.csc_matrix(W)

    from ._slq import _recover_symmetrizing_diagonal

    D = _recover_symmetrizing_diagonal(W)
    if D is None or not np.all(np.isfinite(D)) or np.any(D <= 0):
        raise ValueError(
            "cheb_cholesky requires a D-symmetrizable W (row-standardised "
            "undirected graph); no valid symmetrizing diagonal was found. "
            'Use logdet_method="aaa" for directed or non-symmetrizable W.'
        )

    D_sqrt = np.sqrt(D)
    D_inv_sqrt = 1.0 / D_sqrt
    # W_sym = D^{1/2} W D^{-1/2}  — sparse scaling, no densification
    # W_sym[i,j] = sqrt(d_i) * W[i,j] / sqrt(d_j)
    W_coo = W.tocoo()
    scaled_data = D_sqrt[W_coo.row] * W_coo.data * D_inv_sqrt[W_coo.col]
    W_sym = sp.csc_matrix((scaled_data, (W_coo.row, W_coo.col)), shape=(n, n))

    # Hard guard: CHOLMOD reads one triangle of its input, so a
    # non-symmetric W_sym would produce a silently wrong logdet.
    sym_diff = (W_sym - W_sym.T).tocoo()
    sym_err = float(np.abs(sym_diff.data).max()) if sym_diff.nnz else 0.0
    if sym_err > 1e-10:
        raise ValueError(
            f"D-symmetrization failed (max asymmetry {sym_err:.2e}); W is "
            "not of the form D^-1 A with symmetric A. Use "
            'logdet_method="aaa" for this weights matrix.'
        )
    return W_sym


def chol_cheb_logdet_precompute(
    W,
    order: int | None = None,
    rho_min: float = 0.1,
    rho_max: float = 0.8,
) -> CholChebPrecompute:
    """Precompute Chebyshev coefficients via sparse Cholesky log-determinant.

    Evaluates ``log|det(I - ρW)|`` exactly at ``order`` Chebyshev nodes
    in ``[ρ_min, ρ_max]`` via sparse Cholesky factorisation, then fits
    a Chebyshev polynomial in ``ρ``.

    **Symbolic reuse**: all ``I - ρW`` matrices share the same sparsity
    pattern, so CHOLMOD's symbolic analysis (AMD ordering + elimination
    tree) is performed only once and reused for all subsequent numeric
    factorisations.  This saves ~64% of per-node cost.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, row-standardised).
    order : int or None, default None
        Chebyshev polynomial degree.  If ``None``, auto-selected based on
        the interval width: 15 for ``[0.1, 0.8]``, 30 for wider ranges,
        50 for ``[-0.95, 0.95]`` or wider.
    rho_min : float, default 0.1
        Lower bound of the ρ approximation interval.  Clamped to -0.99
        to avoid the singularity at ρ = -1.
    rho_max : float, default 0.8
        Upper bound of the ρ approximation interval.  Clamped to 0.99
        to avoid the singularity at ρ = 1.

    Returns
    -------
    CholChebPrecompute
        Precomputed Chebyshev coefficients.
    """
    from sksparse.cholmod import cho_factor as cholmod_cho_factor

    if sp.issparse(W) or hasattr(W, "format"):
        W_sp = sp.csr_matrix(W, dtype=np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))

    n = W_sp.shape[0]

    # Clamp interval away from singularities at ±1 (the logdet function
    # has logarithmic singularities there; Chebyshev cannot converge at
    # the singularity itself).  The sampler never explores ρ = ±1 exactly.
    rho_min = max(rho_min, -0.99)
    rho_max = min(rho_max, 0.99)

    # Auto-select order based on interval width if not specified
    if order is None:
        order = _adaptive_order(rho_min, rho_max)

    # D-symmetrise for SPD Cholesky
    W_sym = _d_symmetrize(W_sp)

    # Chebyshev nodes (Clenshaw-Curtis): cos((2j-1)π/(2*order)), j=1..order
    k = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    rho_nodes = 0.5 * (rho_max - rho_min) * nodes_cos + 0.5 * (rho_max + rho_min)

    # Evaluate logdet at each node via sparse Cholesky.
    # Key optimisation: all I - ρW share the same sparsity pattern, so we
    # perform symbolic analysis (AMD ordering + elimination tree) only once
    # and reuse it for every subsequent numeric factorisation.  This saves
    # ~64% of per-node cost (measured on n=2000 grids).
    logdet_vals = np.zeros(order, dtype=np.float64)
    factor = None
    for i, rho in enumerate(rho_nodes):
        A = sp.eye(n, format="csc") - rho * W_sym
        if factor is None:
            # First node: symbolic analysis + numeric factorisation
            factor = cholmod_cho_factor(A)
        else:
            # Subsequent nodes: numeric factorisation only (reuse symbolic)
            factor.factorize(A)
        logdet_vals[i] = factor.logdet()

    # DCT-I → Chebyshev coefficients
    coeffs = chebyshev_coeffs_dct1(logdet_vals)

    return CholChebPrecompute(
        coeffs=coeffs,
        rho_min=rho_min,
        rho_max=rho_max,
        order=order,
        n=n,
    )


def chol_cheb_logdet_eval(pre: CholChebPrecompute, rho: float) -> float:
    """Evaluate ``log|I - ρW|`` from precomputed Chebyshev coefficients.

    Uses Clenshaw recurrence: ``O(order)`` per evaluation.

    Parameters
    ----------
    pre : CholChebPrecompute
        Precomputed coefficients from :func:`chol_cheb_logdet_precompute`.
    rho : float
        Spatial autoregressive parameter.
    """
    return clenshaw_scalar(pre.coeffs, rho, pre.rho_min, pre.rho_max)


def chol_cheb_logdet_eval_vec(
    pre: CholChebPrecompute, rho_arr: np.ndarray
) -> np.ndarray:
    """Vectorized evaluation over an array of ρ values."""
    return clenshaw_vec(pre.coeffs, rho_arr, pre.rho_min, pre.rho_max)
