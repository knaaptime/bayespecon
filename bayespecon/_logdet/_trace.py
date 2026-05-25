"""Stochastic trace estimation.

This module provides lower-variance stochastic trace estimation for the
log-determinant computation in spatial models.  The Hutch++ estimator
(Meyer et al. 2021) replaces the basic Barry-Pace Hutchinson probes
used by :func:`~bayespecon.logdet._barry_pace_traces`, reducing the
number of matrix-vector products needed for a given accuracy level.

All core implementations are pure NumPy + SciPy sparse; the legacy
``traceax`` JAX integration is gated behind :func:`traceax_available`
but is not exercised by the active code path.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Any

import numpy as np
import scipy.sparse as sp


@lru_cache(maxsize=1)
def traceax_available() -> bool:
    """Return ``True`` when ``traceax``, ``lineax``, and ``equinox`` are all importable."""
    return (
        importlib.util.find_spec("traceax") is not None
        and importlib.util.find_spec("lineax") is not None
        and importlib.util.find_spec("equinox") is not None
    )


def _require_traceax() -> None:
    """Raise ImportError with a helpful message if traceax is not installed."""
    if not traceax_available():
        raise ImportError(
            "traceax-based trace estimation requires the optional dependencies "
            "'traceax', 'lineax', and 'equinox'. "
            "Install them with: pip install bayespecon[backend]"
        )


def _import_traceax():
    """Import and return the traceax, lineax, and jax modules.

    Returns
    -------
    tuple
        ``(traceax, lineax, jax)`` modules.
    """
    _require_traceax()
    import jax  # noqa: F811
    import lineax as lx  # noqa: F811
    import traceax as tx  # noqa: F811

    return tx, lx, jax


# ---------------------------------------------------------------------------
# Pure-NumPy variance-reduced trace estimation
# ---------------------------------------------------------------------------


def _hutchpp_traces_numpy(
    W_sparse: sp.csr_matrix,
    order: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hutch++ trace estimates for tr(W^i), i=1..order.

    Uses scipy CSR×dense batched matmuls (same speed as
    :func:`~bayespecon.logdet._barry_pace_traces`) but adds the
    Hutch++ low-rank correction for variance reduction.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised n×n spatial weights matrix.
    order : int
        Maximum trace power to estimate.
    k : int
        Total number of matrix-vector products (probes).
        Clamped to ``n`` if larger (QR constraint).
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray, shape (order,)
        Mean trace estimates: ``traces[i] ≈ tr(W^{i+1})``.
    """
    n = W_sparse.shape[0]
    k = min(k, n)

    # Phase 1: Low-rank basis (m = k//3 probes)
    m = max(1, k // 3)
    Omega_lr = rng.standard_normal((n, m))
    Y = W_sparse @ Omega_lr  # (n, m) — single CSR×dense call
    Q, _ = np.linalg.qr(Y)  # (n, m) orthonormal basis

    # Phase 2: Residual probes (k - m probes)
    Omega_res = rng.standard_normal((n, k - m))
    # Project out Q component: G = Omega_res - Q(Q^T Omega_res)
    G = Omega_res - Q @ (Q.T @ Omega_res)

    # Phase 3: Compute traces for all powers
    # Fuse Q and G into single (n, k) matrix to halve CSR×dense calls
    M = np.hstack([Q, G])  # (n, k)
    traces = np.empty(order, dtype=np.float64)

    for i in range(order):
        M = W_sparse @ M  # single CSR×dense call
        Q_current = M[:, :m]
        G_current = M[:, m:]
        lr_trace = np.sum(Q * Q_current)  # tr(Q^T W^{i+1} Q)
        res_trace = np.sum(G * G_current) / (k - m)
        traces[i] = lr_trace + res_trace

    return traces


def _hutchinson_traces_numpy(
    W_sparse: sp.csr_matrix,
    order: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Baseline Hutchinson trace estimates for tr(W^i), i=1..order.

    Same formula as :func:`~bayespecon.logdet._barry_pace_traces` but
    returning mean estimates only (no per-probe array).

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised n×n spatial weights matrix.
    order : int
        Maximum trace power to estimate.
    k : int
        Number of Monte Carlo probes.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray, shape (order,)
        Mean trace estimates: ``traces[i] ≈ tr(W^{i+1})``.
    """
    n = W_sparse.shape[0]
    Omega = rng.standard_normal((n, k))
    V = Omega.copy()
    traces = np.empty(order, dtype=np.float64)
    for i in range(order):
        V = W_sparse @ V
        traces[i] = np.sum(Omega * V) / k
    return traces


def _xtrace_traces_numpy(
    W_sparse: sp.csr_matrix,
    order: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""XTrace estimates for :math:`\mathrm{tr}(W^j),\; j=1,\ldots,\mathrm{order}`.

    Implements the genuine leave-one-out XTrace estimator of
    :cite:t:`epperly2024XTrace` (Algorithm 2.1, SIAM J. Matrix Anal.
    Appl.), adapted to powers of a single sparse operator ``W``.

    For each moment ``j``, the operator ``A = W^j`` and the leave-one-out
    quantity ``A Q_{-i}`` is recovered without extra matvecs by the
    identity

    .. math::

        A\,Q_{-i} \;=\; (W^j) (Y_j^{(-i)} R_{-i}^{-1})
                 \;=\; Y_{2j}^{(-i)} R_{-i}^{-1},

    where :math:`Y_l = W^l \Omega` is the pre-computed power chain of the
    probe matrix.  This keeps the total matvec budget at
    :math:`2\,\mathrm{order}\,k` — exactly twice Hutch++ — while
    retaining the leave-one-out variance reduction of XTrace.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Spatial weights matrix (assumed to have real spectrum, e.g.
        row-standardised contiguity).
    order : int
        Maximum trace power to estimate.
    k : int
        Number of probe vectors.  Must be ≥ 2; falls back to Hutchinson
        if ``k < 2``.  Clamped to ``n``.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray, shape (order,)
        Mean trace estimates: ``traces[i] ≈ tr(W^{i+1})``.

    Notes
    -----
    The estimator assumes an effectively symmetric operator (real
    spectrum), which holds for row-standardised spatial weights matrices
    (similar to a symmetric matrix via :math:`D^{1/2} W D^{-1/2}`).  For
    strictly non-symmetric matrices the residual term uses
    :math:`\psi^\top A v` rather than :math:`\psi^\top A^\top v`; the
    bias is small when the spectrum is real.

    The per-moment cost is :math:`O(n m^2 + m^4)` rather than the naive
    :math:`O(n m^3)`.  Key trick: with ``Q_full R_full = qr(Y_j)``, the
    leave-one-out QR of ``Y_j[:, -i] = Q_full @ R_full[:, -i]`` reduces
    to a small QR of the :math:`(m, m-1)` matrix ``R_full[:, -i]``
    ``= U_i T_i``, giving ``Q_{-i} = Q_full @ U_i`` and ``R_{-i} = T_i``.
    The leave-one-out work over ``i = 1..m`` is fully vectorised via
    batched :func:`numpy.linalg.qr` / :func:`numpy.linalg.solve` on
    stacks of small :math:`(m, m-1)` matrices, so the Python-level
    inner loop is eliminated.
    """
    n = W_sparse.shape[0]
    m = min(k, n)
    if m < 2:
        return _hutchinson_traces_numpy(W_sparse, order, k, rng)

    Psi = rng.choice([-1.0, 1.0], size=(n, m)).astype(np.float64)

    # Build chain Y[l] = W^l @ Psi for l = 0, 1, ..., 2*order.
    Y = [Psi]
    current = Psi
    for _ in range(2 * order):
        current = W_sparse @ current
        Y.append(current)

    # sel_indices[i] = [0, 1, ..., i-1, i+1, ..., m-1].  Shape (m, m-1).
    eye_m = np.arange(m)
    sel_indices = np.stack([np.delete(eye_m, i) for i in range(m)])  # (m, m-1)

    # Scatter indices used to "un-delete" per-probe vectors back into length m
    # without materialising the big (n, m, m-1) gather of Y_2j[:, sel_i].
    scatter_rows = np.repeat(eye_m, m - 1)        # (m*(m-1),)
    scatter_cols = sel_indices.ravel()            # (m*(m-1),)

    traces = np.empty(order, dtype=np.float64)
    inv_m = 1.0 / m

    for j in range(1, order + 1):
        Y_j = Y[j]
        Y_2j = Y[2 * j]

        # One thin QR per moment: Y_j = Q_full R_full,  (n, m) and (m, m).
        Q_full, R_full = np.linalg.qr(Y_j, mode="reduced")

        # Per-moment caches: each is O(n m^2) once, then independent of i.
        C_full = Q_full.T @ Y_2j          # (m, m) = Q_full^T (W^j) Q_full · R_full
        P_full = Q_full.T @ Psi           # (m, m) = Q_full^T Psi

        # --- Batched leave-one-out small QR ------------------------------
        # R_stack[i] = R_full[:, sel_i]  with shape (m, m, m-1).
        R_stack = R_full[:, sel_indices].transpose(1, 0, 2)
        # U_stack: (m, m, m-1) orthonormal,  T_stack: (m, m-1, m-1) upper-tri.
        U_stack, T_stack = np.linalg.qr(R_stack, mode="reduced")

        # --- Low-rank trace contribution: tr(Q_{-i}^T A Q_{-i}) ----------
        # = tr(U_i^T C_full[:, -i] T_i^{-1})  = tr(T_i^{-1} (U_i^T C_full[:, -i])).
        C_stack = C_full[:, sel_indices].transpose(1, 0, 2)              # (m, m, m-1)
        M_stack = np.einsum("imp,imq->ipq", U_stack, C_stack)            # (m, m-1, m-1)
        Z_stack = np.linalg.solve(T_stack, M_stack)                      # T_i^{-1} M_i
        lr_per_i = np.einsum("ikk->i", Z_stack)                          # (m,)

        # --- Hutchinson residual against the LOO complement --------------
        # QTpsi[i]   = U_i^T P_full[:, i]                                 -> (m-1,)
        # u_full[i]  = U_i @ QTpsi[i]                                     -> (m,)
        # v[i]       = psi_i - Q_full @ u_full[i]
        QTpsi_stack = np.einsum("imp,mi->ip", U_stack, P_full)            # (m, m-1)
        u_full_stack = np.einsum("imp,ip->im", U_stack, QTpsi_stack)      # (m, m)
        v_stack = Psi.T - u_full_stack @ Q_full.T                         # (m, n)

        # A Q_{-i} @ QTpsi[i]  =  Y_2j[:, -i] @ (T_i^{-1} QTpsi[i]).
        # Pad t_solve back to length m (insert 0 at column i) and multiply once.
        t_solve_stack = np.linalg.solve(
            T_stack, QTpsi_stack[..., None]
        )[..., 0]                                                          # (m, m-1)
        T_pad = np.zeros((m, m), dtype=np.float64)
        T_pad[scatter_rows, scatter_cols] = t_solve_stack.ravel()
        Av_proj_stack = T_pad @ Y_2j.T                                     # (m, n)
        Av_stack = Y_j.T - Av_proj_stack                                   # (m, n)

        resid_per_i = np.einsum("in,in->i", v_stack, Av_stack)            # (m,)

        traces[j - 1] = float((lr_per_i + resid_per_i).sum()) * inv_m

    return traces


# ---------------------------------------------------------------------------
# Trace estimation: tr(W^k) via variance-reduced estimators
# ---------------------------------------------------------------------------


def traceax_traces(
    W_sparse: sp.csr_matrix,
    order: int,
    k: int,
    estimator: str = "hutchpp",
    seed: int = 0,
) -> np.ndarray:
    """Estimate tr(W^k) for k=1..order via variance-reduced stochastic trace estimation.

    Uses pure-NumPy batched matmuls with scipy CSR matrices, matching the
    speed of :func:`~bayespecon.logdet._barry_pace_traces` while achieving
    lower variance via the Hutch++ correction.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised n×n spatial weights matrix.
    order : int
        Maximum trace power to estimate.
    k : int
        Number of matrix-vector products (probes) per trace estimate.
        Clamped to ``n`` if larger (QR constraint).
    estimator : str, default ``"hutchpp"``
        Estimator to use: ``"hutchpp"``, ``"xtrace"``, or ``"hutchinson"``.
    seed : int, default 0
        NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (order,)
        Mean trace estimates: ``traces[i] ≈ tr(W^{i+1})``.
        Entries 0 and 1 are overridden with exact values
        (``tr(W)`` and ``tr(W²)``), matching the convention
        of :func:`~bayespecon.logdet._barry_pace_traces`.

    Notes
    -----
    The variance-reduced Hutch++ estimator provides lower variance than
    the basic Hutchinson (Girard) estimator by splitting probes into a
    low-rank approximation (k//3) and a residual trace, reducing
    variance for matrices with decaying eigenvalue spectra (typical of
    spatial weights matrices).

    The XTrace estimator (Epperly–Tropp–Webber 2024) refines Hutch++
    via a leave-one-out construction that reuses every probe for both
    the low-rank basis and the residual estimate, at roughly 2× the
    matvec cost of Hutch++.

    For spatial weights matrices with a few dominant eigenvalues (common
    for row-standardised contiguity weights), Hutch++ can achieve the
    same accuracy as Hutchinson with 2–5× fewer probes.

    Raises
    ------
    ValueError
        If ``estimator`` is not one of the supported values.
    """
    if estimator not in ("hutchpp", "xtrace", "hutchinson"):
        raise ValueError(
            f"estimator must be 'hutchpp', 'xtrace', or 'hutchinson'; got {estimator!r}"
        )

    rng = np.random.default_rng(seed)

    if estimator == "hutchinson":
        traces = _hutchinson_traces_numpy(W_sparse, order, k, rng)
    elif estimator == "xtrace":
        traces = _xtrace_traces_numpy(W_sparse, order, k, rng)
    else:  # hutchpp
        traces = _hutchpp_traces_numpy(W_sparse, order, k, rng)

    # Override with exact values for k=1, 2 (same as _barry_pace_traces)
    traces[0] = float(W_sparse.diagonal().sum())  # tr(W) = 0 for zero-diagonal W
    if order >= 2:
        traces[1] = float(W_sparse.multiply(W_sparse.T).sum())  # tr(W^2) = sum(W .* W')

    return traces


# ---------------------------------------------------------------------------
# Direct logdet estimation via traceax
# ---------------------------------------------------------------------------


def logdet_traceax(
    W_sparse: sp.csr_matrix,
    rho: float,
    *,
    k: int = 30,
    estimator: str = "hutchpp",
    seed: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Estimate log|I - rho*W| via stochastic trace estimation.

    Uses the power-series identity:

    .. math::

        \\log|I_n - \\rho W| = -\\sum_{j=1}^{m} \\frac{\\rho^j}{j} \\text{tr}(W^j)

    where :math:`\\text{tr}(W^j)` is estimated via variance-reduced
    stochastic trace estimation.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised n×n spatial weights matrix.
    rho : float
        Spatial autoregressive parameter.
    k : int, default 30
        Number of matrix-vector products per trace estimate.
    estimator : str, default ``"hutchpp"``
        Estimator to use: ``"hutchpp"``, ``"xtrace"``, or ``"hutchinson"``.
    seed : int, default 0
        NumPy random seed for reproducibility.

    Returns
    -------
    tuple[float, dict]
        ``(logdet_estimate, info)`` where ``info`` contains:

        - ``"method"`` : the estimator name used

    Raises
    ------
    ValueError
        If ``estimator`` is not one of the supported values.
    """
    if estimator not in ("hutchpp", "xtrace", "hutchinson"):
        raise ValueError(
            f"estimator must be 'hutchpp', 'xtrace', or 'hutchinson'; got {estimator!r}"
        )

    traces = traceax_traces(W_sparse, order=k, k=k, estimator=estimator, seed=seed)
    j_arr = np.arange(1, len(traces) + 1, dtype=np.float64)
    weights = traces / j_arr  # tr(W^j) / j
    powers = np.power(rho, j_arr)
    logdet_est = -float(powers @ weights)

    info: dict[str, Any] = {"method": estimator}
    return logdet_est, info


# ---------------------------------------------------------------------------
# Trace estimation for Chebyshev MC path (drop-in for _barry_pace_traces)
# ---------------------------------------------------------------------------


def traceax_traces_for_chebyshev(
    W_sparse: sp.csr_matrix,
    order: int,
    n_mc_iter: int,
    estimator: str = "hutchpp",
    seed: int = 0,
) -> np.ndarray:
    """Estimate tr(W^k) for the Chebyshev MC path via traceax.

    This is a convenience wrapper around :func:`traceax_traces` that
    returns the mean trace estimates in the format expected by
    :func:`~bayespecon.logdet.chebyshev`'s MC path: a 1-D array of
    shape ``(order,)`` where entry ``[i]`` is the mean estimate of
    ``tr(W^{i+1}) / (i+1)``.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised n×n spatial weights matrix.
    order : int
        Maximum trace power to estimate.
    n_mc_iter : int
        Number of matrix-vector products per trace estimate.
    estimator : str, default ``"hutchpp"``
        Estimator to use.
    seed : int, default 0
        NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (order,)
        Mean trace estimates: ``td[i] = tr(W^{i+1}) / (i+1)``.
    """
    traces = traceax_traces(W_sparse, order, n_mc_iter, estimator=estimator, seed=seed)
    # traces shape: (order,)
    # Divide by (i+1) to get tr(W^k) / k
    k_arr = np.arange(1, order + 1, dtype=np.float64)
    td = traces / k_arr
    return td
