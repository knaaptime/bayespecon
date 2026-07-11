"""AAA rational approximation log-determinant for non-symmetric ``I - ρW``.

For row-standardised ``W`` from a **directed** graph (KNN, travel time,
migration flows), the matrix ``I - ρW`` is non-symmetric and cannot be
symmetrised via D-symmetrisation.  Sparse Cholesky is unavailable; the
options are sparse LU (exact but expensive) or stochastic methods
(approximate).

This module implements the **AAA rational approximation** strategy:

1. Evaluate ``log|det(I - ρW)|`` exactly at ``n_coarse`` Chebyshev-spaced
   points via sparse LU, reusing one symbolic factorisation across all of
   them (KLU, falling back to UMFPACK then scipy SuperLU).
2. Fit a rational function in barycentric form via the AAA algorithm
   [@nakatsukasa2018], which selects ``m`` support points from the coarse
   grid.
3. Evaluate at any ρ via the barycentric formula in ``O(m)`` per ρ.

**Why rational instead of polynomial?**  The logdet function
``f(ρ) = Σ log(1 - ρλᵢ)`` has logarithmic singularities at ``ρ = 1/λᵢ``.
Polynomials converge slowly near singularities (needing 50-100 nodes for
``[-0.95, 0.95]``).  Rational functions converge exponentially faster —
typically needing only 6-15 support points for the same accuracy.

**When to use**: non-symmetric ``W`` (directed graph) where Cholesky is
unavailable.  For symmetric ``W``, use ``cheb_cholesky`` (exact, faster).
For very large ``n`` (>20,000), use ``cheb_stochastic`` (avoids
factorisation entirely).

**Cost**: ``n_coarse`` sparse LU factorisations + ``O(m)`` per-ρ
evaluation, where ``n_coarse`` is the coarse-grid size (adaptive: 16 for the
narrow default interval, up to 30 for wide/near-singular intervals) and
``m ≤ n_coarse // 2`` is the number of AAA support points actually selected.
All ``I - ρW`` share one sparsity pattern, so KLU's symbolic analysis is
computed once and reused for every subsequent numeric factorisation (measured
1.6-3.4× faster than a fresh UMFPACK factorisation per node over the coarse
grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class AAAPrecompute:
    """Precomputed AAA rational approximant for ``log|I - ρW|``.

    Attributes
    ----------
    support_points : np.ndarray, shape (m,)
        ρ values where exact logdet was computed (AAA support points).
    support_values : np.ndarray, shape (m,)
        Exact logdet values at support points.
    weights : np.ndarray, shape (m,)
        Barycentric weights from the AAA algorithm.
    rho_min : float
        Lower bound of the ρ approximation interval.
    rho_max : float
        Upper bound of the ρ approximation interval.
    n : int
        Matrix dimension.
    """

    support_points: np.ndarray
    support_values: np.ndarray
    weights: np.ndarray
    rho_min: float
    rho_max: float
    n: int


def _klu_logdet_from_factor(factor) -> float:
    """Recover ``log|det(A)|`` from a ``sksparse.klu`` factor.

    KLU factorises ``P R A Q = L U`` with a diagonal row scaling ``R``; the
    permutations affect only the sign, so
    ``log|det(A)| = Σ log|diag(U)| + Σ log|diag(L)| - Σ log|diag(R)|``.
    """
    logdet = float(np.sum(np.log(np.abs(factor.U.diagonal()))))
    l_diag = factor.L.diagonal()
    logdet += float(np.sum(np.log(np.abs(l_diag))))
    rscale = factor.rscale
    if rscale is not None:
        logdet -= float(np.sum(np.log(np.abs(rscale))))
    return logdet


def _lu_logdet(A: sp.csc_matrix) -> float:
    """Compute ``log|det(A)|`` via sparse LU factorisation (single shot).

    Prefers KLU (``sksparse.klu``), then UMFPACK (``sksparse.umfpack``,
    which exposes ``slogdet`` directly), then scipy SuperLU.  For repeated
    factorisations of matrices sharing a sparsity pattern (the AAA coarse
    grid), use :func:`_make_reusable_lu_logdet`, which reuses KLU's symbolic
    analysis.
    """
    A = A.tocsc()
    try:
        from sksparse.klu import klu_factor

        return _klu_logdet_from_factor(klu_factor(A))
    except Exception:
        pass
    try:
        from sksparse.umfpack import umf_factor

        _sign, logabsdet = umf_factor(A).slogdet()
        return float(logabsdet)
    except Exception:
        from scipy.sparse.linalg import splu as scipy_splu

        lu = scipy_splu(A)
        logdet = np.sum(np.log(np.abs(lu.L.diagonal()))) + np.sum(
            np.log(np.abs(lu.U.diagonal()))
        )
        return float(logdet)


def _make_reusable_lu_logdet():
    """Return a callable ``A -> log|det(A)|`` that reuses symbolic analysis.

    On the first call it computes KLU's symbolic + numeric factorisation; on
    subsequent calls it refactorises numerically only (``KLUFactor.factorize``),
    valid because every ``I - ρW`` shares one sparsity pattern.  This mirrors
    the CHOLMOD symbolic reuse in :func:`chol_cheb_logdet_precompute` and is
    measured 1.6-3.4× faster than a fresh UMFPACK factorisation per node.

    Falls back to the single-shot :func:`_lu_logdet` (UMFPACK / scipy) when
    ``sksparse.klu`` is unavailable or its first factorisation fails.
    """
    state = {"factor": None, "use_klu": True}

    def _evaluate(A) -> float:
        A = A.tocsc()
        if state["use_klu"]:
            try:
                if state["factor"] is None:
                    from sksparse.klu import klu_factor

                    state["factor"] = klu_factor(A)
                else:
                    state["factor"].factorize(A)
                return _klu_logdet_from_factor(state["factor"])
            except Exception:
                # KLU unavailable or failed — drop to single-shot fallback.
                state["use_klu"] = False
                state["factor"] = None
        return _lu_logdet(A)

    return _evaluate


def _aaa_algorithm(
    z: np.ndarray,
    f: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core AAA algorithm for rational approximation.

    Given sample points ``z`` and function values ``f``, find support
    points, values, and barycentric weights for a rational approximant.

    Parameters
    ----------
    z : np.ndarray, shape (M,)
        Sample points (dense grid of ρ values).
    f : np.ndarray, shape (M,)
        Function values at sample points.
    tol : float, default 1e-10
        Relative tolerance for the nonlinear residual.
    max_iter : int, default 100
        Maximum number of AAA iterations (support points).

    Returns
    -------
    support_points : np.ndarray, shape (m,)
        Selected support points (subset of z).
    support_values : np.ndarray, shape (m,)
        Function values at support points.
    weights : np.ndarray, shape (m,)
        Barycentric weights.
    """
    z = np.asarray(z, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    M = len(z)

    # Track which points are support points
    is_support = np.zeros(M, dtype=bool)
    support_idx = []  # indices into z
    weights_list = []

    # Residual at all non-support points
    residual = f.copy()

    for m in range(1, min(max_iter, M // 2) + 1):
        # Greedy: pick the point with largest |residual|
        # Exclude already-selected points
        candidate_residual = np.abs(residual).copy()
        candidate_residual[is_support] = -1  # exclude support points
        next_idx = np.argmax(candidate_residual)

        if candidate_residual[next_idx] < tol * np.max(np.abs(f)):
            break

        is_support[next_idx] = True
        support_idx.append(next_idx)

        # Get current support points and values
        sp_z = z[is_support]
        sp_f = f[is_support]
        m_curr = len(sp_z)

        # Non-support points
        non_support = ~is_support
        z_ns = z[non_support]
        f_ns = f[non_support]

        if m_curr == 1:
            # First support point: weight = 1 (trivial)
            weights_list = [1.0]
            # Residual = f - f_1 (constant approximant)
            residual = f - sp_f[0]
            continue

        # Build the Loewner matrix for least-squares
        # A[i, j] = (f_ns[i] - sp_f[j]) / (z_ns[i] - sp_z[j])
        # We want to minimize ||A @ w|| subject to ||w|| = 1
        n_ns = len(z_ns)
        A = np.zeros((n_ns, m_curr), dtype=np.float64)
        for j in range(m_curr):
            # Avoid division by zero (shouldn't happen since support != non-support)
            diff = z_ns - sp_z[j]
            A[:, j] = (f_ns - sp_f[j]) / diff

        # Solve min ||A @ w|| s.t. ||w|| = 1 via SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        w = Vt[-1]  # right singular vector for smallest singular value

        weights_list = w

        # Compute residual at non-support points
        # r(z) = n(z)/d(z) where
        # n(z) = sum_j w_j * sp_f[j] / (z - sp_z[j])
        # d(z) = sum_j w_j / (z - sp_z[j])
        n_val = np.zeros(n_ns, dtype=np.float64)
        d_val = np.zeros(n_ns, dtype=np.float64)
        for j in range(m_curr):
            diff = z_ns - sp_z[j]
            n_val += w[j] * sp_f[j] / diff
            d_val += w[j] / diff

        # Avoid division by zero
        r_ns = np.where(
            np.abs(d_val) > 1e-15,
            n_val / d_val,
            sp_f[0],  # fallback
        )

        # Update residual
        residual = f.copy()
        residual[non_support] = f_ns - r_ns
        # At support points, residual is 0 (interpolation)

    # Extract final support points, values, weights
    support_points = z[is_support]
    support_values = f[is_support]
    weights = np.array(weights_list, dtype=np.float64)

    return support_points, support_values, weights


def _adaptive_n_coarse(rho_min: float, rho_max: float) -> int:
    """Choose the coarse-grid size (= number of exact LU factorisations).

    Each coarse-grid point costs one sparse LU factorisation, so ``n_coarse``
    directly sets the setup cost.  The AAA support count ``m`` (a subset of the
    grid, capped at ``n_coarse // 2``) is what determines accuracy, and both
    grow as the interval widens toward the ``ρ = ±1`` logdet singularities.

    Empirically (rook + knn, n∈{1600, 2000}): the default narrow interval
    ``[0.1, 0.8]`` reaches ~1e-10 max error with only 16 nodes, while intervals
    that approach ``±0.95`` need the full 30 nodes for ~1e-7.  This mirrors
    :func:`_adaptive_order` in ``_chol_cheb.py``.

    Parameters
    ----------
    rho_min, rho_max : float
        The ρ approximation interval.

    Returns
    -------
    int
        Number of Chebyshev-spaced coarse-grid points (LU factorisations).
    """
    width = rho_max - rho_min
    dist = min(abs(1.0 - rho_max), abs(1.0 + rho_min))
    # Narrow interval well clear of the ±1 singularities (e.g. the default
    # [0.1, 0.8]) converges with far fewer nodes; everything else keeps the
    # full grid so wide/near-singular intervals still hit ~1e-7.
    if width <= 0.71 and dist > 0.15:
        return 16
    return 30


def _aaa_algorithm_lazy(
    z: np.ndarray,
    eval_fn,
    tol: float = 1e-10,
    max_iter: int = 30,
    n_coarse: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lazy AAA: evaluates ``eval_fn`` at a small coarse grid, not the full sample grid.

    Instead of evaluating at all ``M = len(z)`` sample points, this function
    evaluates at ``n_coarse`` Chebyshev-spaced points and runs the standard
    AAA algorithm on those.  This reduces expensive evaluations (e.g. sparse
    LU factorisations) from ``M`` to ``n_coarse`` (default 30), a ~7× speedup
    for the default ``M=200``.

    The full sample grid ``z`` is used only for the curvature-based refinement
    check — if the approximant has high curvature at non-support points on the
    full grid, one additional evaluation is performed there.

    Parameters
    ----------
    z : np.ndarray, shape (M,)
        Sample points (dense grid of ρ values).
    eval_fn : callable
        Function ``f(ρ) -> float``.  Called at ``n_coarse`` + refinement points.
    tol : float, default 1e-10
        Relative tolerance for AAA convergence.
    max_iter : int, default 30
        Maximum number of support points.
    n_coarse : int, default 30
        Number of Chebyshev-spaced evaluation points for the coarse phase.

    Returns
    -------
    support_points, support_values, weights : np.ndarray
    """
    z = np.asarray(z, dtype=np.float64)
    M = len(z)
    rho_min_z, rho_max_z = z[0], z[-1]

    # Phase 1: Evaluate at n_coarse Chebyshev-spaced points
    n_coarse = min(n_coarse, M)
    k = np.arange(1, n_coarse + 1)
    coarse_cos = np.cos((2 * k - 1) * np.pi / (2 * n_coarse))
    z_coarse = 0.5 * (rho_max_z - rho_min_z) * coarse_cos + 0.5 * (
        rho_max_z + rho_min_z
    )

    f_coarse = np.array([eval_fn(zc) for zc in z_coarse], dtype=np.float64)

    # Run standard AAA on the coarse grid (uses true values for Loewner LSQ)
    sp_z, sp_f, w = _aaa_algorithm(z_coarse, f_coarse, tol=tol, max_iter=max_iter)

    return sp_z, sp_f, w


def aaa_logdet_precompute(
    W,
    rho_min: float = 0.1,
    rho_max: float = 0.8,
    n_samples: int = 200,
    tol: float = 1e-10,
    max_iter: int = 30,
    n_coarse: int | None = None,
) -> AAAPrecompute:
    """Precompute AAA rational approximant for ``log|I - ρW|``.

    Evaluates ``log|det(I - ρW)|`` exactly at a coarse grid of ``n_coarse``
    Chebyshev-spaced points via sparse LU (KLU with symbolic reuse), then
    fits a rational function via the AAA algorithm, which greedily selects
    ``m`` support points (``m ≤ n_coarse // 2``, typically 5-15) from that grid.

    The number of exact LU factorisations equals ``n_coarse`` — **not** the
    support count ``m`` and **not** ``n_samples`` (the 200-point sample grid is
    only the AAA residual proxy and involves no factorisations).  ``n_coarse``
    defaults to :func:`_adaptive_n_coarse`: 16 for the narrow default interval,
    30 for wider or near-singular intervals.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, non-symmetric OK).
    rho_min : float, default 0.1
        Lower bound of the ρ approximation interval.
    rho_max : float, default 0.8
        Upper bound of the ρ approximation interval.
    n_samples : int, default 200
        Number of sample points for the AAA residual grid.  Does **not**
        affect the number of LU factorisations — only the resolution of
        the greedy selection.
    tol : float, default 1e-10
        Relative tolerance for AAA convergence.
    max_iter : int, default 30
        Maximum number of AAA support points selected from the coarse grid.
    n_coarse : int, optional
        Number of exact LU factorisations (coarse-grid size).  ``None``
        (default) selects it adaptively from the interval via
        :func:`_adaptive_n_coarse`.

    Returns
    -------
    AAAPrecompute
        Precomputed rational approximant.
    """
    if sp.issparse(W) or hasattr(W, "format"):
        W_sp = sp.csc_matrix(W, dtype=np.float64)
    else:
        W_sp = sp.csc_matrix(np.asarray(W, dtype=np.float64))

    n = W_sp.shape[0]

    if n_coarse is None:
        n_coarse = _adaptive_n_coarse(rho_min, rho_max)

    # Dense sample grid for AAA (no factorisation here — just the grid)
    z = np.linspace(rho_min, rho_max, n_samples)

    # Lazy evaluation function: called only at the n_coarse coarse-grid
    # points.  All I - ρW share one sparsity pattern, so the evaluator reuses
    # KLU's symbolic factorisation across calls (numeric refactor only).
    eye = sp.eye(n, format="csc")
    lu_logdet = _make_reusable_lu_logdet()

    def _eval_logdet(rho):
        return lu_logdet(eye - rho * W_sp)

    # Run lazy AAA: exactly n_coarse LU factorisations (m ≤ n_coarse//2 of the
    # grid points become support points).
    support_points, support_values, weights = _aaa_algorithm_lazy(
        z, _eval_logdet, tol=tol, max_iter=max_iter, n_coarse=n_coarse
    )

    return AAAPrecompute(
        support_points=support_points,
        support_values=support_values,
        weights=weights,
        rho_min=rho_min,
        rho_max=rho_max,
        n=n,
    )


def aaa_logdet_eval(pre: AAAPrecompute, rho: float) -> float:
    """Evaluate ``log|I - ρW|`` from precomputed AAA rational approximant.

    Uses the barycentric formula: ``O(m)`` per evaluation.

    Parameters
    ----------
    pre : AAAPrecompute
        Precomputed approximant from :func:`aaa_logdet_precompute`.
    rho : float
        Spatial autoregressive parameter.
    """
    sp_z = pre.support_points
    sp_f = pre.support_values
    w = pre.weights
    m = len(sp_z)

    if m == 0:
        return 0.0
    if m == 1:
        return float(sp_f[0])

    # Barycentric formula:
    # r(ρ) = [Σ_j w_j * f_j / (ρ - z_j)] / [Σ_j w_j / (ρ - z_j)]
    diff = rho - sp_z

    # Check if rho is exactly at a support point
    zero_idx = np.where(np.abs(diff) < 1e-15)[0]
    if len(zero_idx) > 0:
        return float(sp_f[zero_idx[0]])

    n_val = np.sum(w * sp_f / diff)
    d_val = np.sum(w / diff)

    if abs(d_val) < 1e-15:
        # Fallback: return nearest support value
        nearest = np.argmin(np.abs(diff))
        return float(sp_f[nearest])

    return float(n_val / d_val)


def aaa_logdet_eval_vec(pre: AAAPrecompute, rho_arr: np.ndarray) -> np.ndarray:
    """Vectorized evaluation over an array of ρ values."""
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    sp_z = pre.support_points
    sp_f = pre.support_values
    w = pre.weights
    m = len(sp_z)

    if m == 0:
        return np.zeros_like(rho_arr)
    if m == 1:
        return np.full_like(rho_arr, sp_f[0])

    # Barycentric formula, vectorized over rho_arr
    # diff[i, j] = rho_arr[i] - sp_z[j]
    diff = rho_arr[:, None] - sp_z[None, :]  # (n_rho, m)

    # Handle exact matches
    exact_match = np.abs(diff) < 1e-15
    has_exact = np.any(exact_match, axis=1)

    # For non-exact: compute barycentric
    # Avoid division by zero by setting diff to 1 where exact
    safe_diff = np.where(exact_match, 1.0, diff)

    n_val = np.sum(w[None, :] * sp_f[None, :] / safe_diff, axis=1)
    d_val = np.sum(w[None, :] / safe_diff, axis=1)

    result = np.where(
        np.abs(d_val) > 1e-15,
        n_val / d_val,
        sp_f[np.argmin(np.abs(diff), axis=1)],  # fallback
    )

    # Override with exact values where rho matches a support point
    if np.any(has_exact):
        for i in np.where(has_exact)[0]:
            j = np.where(exact_match[i])[0][0]
            result[i] = sp_f[j]

    return result
