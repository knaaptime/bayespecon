"""Stochastic Lanczos Quadrature (SLQ) for log|I - ρW|.

For row-standardised W from an **undirected graph** (rook, queen, distance-band),
W = D⁻¹A where A is symmetric and D = diag(degrees).  W is diagonally similar
to the symmetric matrix W_sym = D^{1/2} W D^{-1/2}, which has the **same
eigenvalues** but real — enabling valid Gauss quadrature via Lanczos.

For directed W (non-symmetric sparsity pattern), falls back to Arnoldi with
complex Ritz values.

Algorithm (D-symmetrised Lanczos)
---------------------------------
1. Recover D from W's sparsity pattern (O(nnz) BFS).
2. Form W_sym = D^{1/2} W D^{-1/2} as a LinearOperator (two O(n) scalings
   + one O(nnz) sparse matvec — never materialised).
3. For each probe z: run k steps of Lanczos on W_sym from the unit start
   q₁ = z/‖z‖, build tridiagonal T_k, eigendecompose → (θ_i, v_i).  Canonical
   SLQ (Ubaru–Chen–Saad) weights: w_i = n · v_{1,i}².
4. Evaluate: log|I - ρW| ≈ (n/n_probes) Σ_j Σ_i v_{1,j,i}² · log(1 - ρθ_{j,i})

Gauss quadrature from k Lanczos steps is exact for polynomials of degree ≤ 2k-1,
giving 3× more spectral information per Krylov step than the Barry-Pace Taylor
series (degree k from k trace moments).

The ``n``-scaling (rather than the sample ``‖z‖²``) removes the χ² radial
fluctuation and makes a constant integrand exact per probe.  Note, however,
that SLQ estimates the *full* log-integral stochastically and — unlike
``cheb_stochastic``, which subtracts the exact ``μ₀ = n`` and ``μ₁ = tr(W̃)``
moments as control variates — carries the full ``‖log(I-ρW_sym)‖_F`` Hutchinson
variance.  On flat spatial spectra this makes SLQ *less* accurate than
``cheb_stochastic`` at equal probe counts; the limitation is inherent to the
per-probe quadrature, not the weight normalization.  SLQ is opt-in, not the
auto-selected default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class SLQPrecompute:
    """Precomputed SLQ quadrature rules for log|I - ρW|.

    Attributes
    ----------
    nodes : np.ndarray, shape (n_probes, k)
        Gauss quadrature nodes (real for D-symmetrised, complex for Arnoldi).
    weights : np.ndarray, shape (n_probes, k)
        Quadrature weights, ``n``-scaled (``Σᵢ weights = n`` per probe): real
        ``n · v₁ᵢ²`` for Lanczos, complex ``n · γᵢ`` (bilinear) for Arnoldi.
    n : int
        Matrix dimension.
    method : str
        "lanczos" (D-symmetrised) or "arnoldi" (non-symmetric fallback).
    """

    nodes: np.ndarray
    weights: np.ndarray
    n: int
    method: str = "lanczos"

    @property
    def n_probes(self) -> int:
        return self.nodes.shape[0]

    @property
    def lanczos_deg(self) -> int:
        return self.nodes.shape[1]


# ---------------------------------------------------------------------------
# D-recovery: find diagonal D such that D^{1/2} W D^{-1/2} is symmetric
# ---------------------------------------------------------------------------


def _recover_symmetrizing_diagonal(W: sp.csr_matrix) -> np.ndarray | None:
    """Recover D such that D^{1/2} W D^{-1/2} is symmetric.

    For W = D⁻¹A (row-standardised, A symmetric), D[i]/D[j] = W[j,i]/W[i,j]
    for each edge (i,j).  We propagate via BFS from node 0.

    Returns
    -------
    np.ndarray or None
        D (up to scalar multiple), or None if W has asymmetric sparsity
        (directed graph — D-symmetrisation not applicable).
    """
    n = W.shape[0]

    # Check symmetric sparsity pattern without densifying: the boolean
    # patterns differ iff their sparse XOR has any stored entries.
    pattern = (W != 0).tocsr()
    if (pattern != pattern.T.tocsr()).nnz > 0:
        return None

    # BFS to propagate D[i]/D[j] = W[j,i] / W[i,j], seeded per connected
    # component so disconnected graphs get a consistent D on every block.
    D = np.empty(n, dtype=np.float64)
    D[:] = np.nan

    # Build adjacency list for BFS
    W_coo = W.tocoo()
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(W_coo.row, W_coo.col):
        if i != j:
            adj[i].append(j)

    W_csr = W.tocsr()
    for seed in range(n):
        if not np.isnan(D[seed]):
            continue
        D[seed] = 1.0
        queue = deque([seed])
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                if np.isnan(D[j]):
                    # D[i] / D[j] = W[j,i] / W[i,j]
                    wij = W_csr[i, j]
                    wji = W_csr[j, i]
                    if abs(wij) < 1e-300 or abs(wji) < 1e-300:
                        continue
                    D[j] = D[i] * wij / wji
                    queue.append(j)

    # Isolated/unreachable-by-value nodes (e.g. zero rows) default to 1
    if np.any(np.isnan(D)):
        D[np.isnan(D)] = 1.0

    return D


# ---------------------------------------------------------------------------
# Batched Lanczos (all probes simultaneously — one block matvec per step)
# ---------------------------------------------------------------------------


def _batched_lanczos(
    matvec_fn,  # callable: (n, n_probes) -> (n, n_probes)
    n: int,
    k: int,
    Z: np.ndarray,
    n_probes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run k steps of **batched** Lanczos on W_sym for all probes at once.

    Fully vectorized: stores Lanczos vectors as (n, k, n_probes) and uses
    einsum for batched reorthogonalization.  One batched sparse matvec per
    Lanczos step; all probe operations are vectorized.

    Parameters
    ----------
    matvec_fn : callable
        Function that computes W_sym @ Q for a (n, n_probes) block.
    n : int
        Matrix dimension.
    k : int
        Lanczos steps per probe.
    Z : np.ndarray, shape (n, n_probes)
        Probe vectors (columns).
    n_probes : int

    Returns
    -------
    (nodes, weights, z_norms_sq) : tuple
    """
    # Normalize each probe column (unit starts; canonical SLQ scales by n).
    z_norms = np.linalg.norm(Z, axis=0)  # (n_probes,)
    Q0 = Z / np.where(z_norms < 1e-15, 1.0, z_norms)  # (n, n_probes)

    # 3D storage: Q_all[:, step, probe] — (n, k, n_probes)
    Q_all = np.zeros((n, k, n_probes), dtype=np.float64)
    Q_all[:, 0, :] = Q0

    alphas = np.zeros((n_probes, k), dtype=np.float64)
    betas = np.zeros((n_probes, k - 1), dtype=np.float64)
    active = np.ones(n_probes, dtype=bool)  # which probes haven't broken down
    deg = np.full(n_probes, k, dtype=int)  # effective Lanczos degree per probe

    # First matvec (batched)
    R = matvec_fn(Q0)  # (n, n_probes) — ONE batched matvec

    # alpha_0 = Q0' R (per-probe dot products)
    alphas[:, 0] = np.sum(Q0 * R, axis=0)  # (n_probes,)
    R = R - alphas[:, 0] * Q0  # broadcast: (n, n_probes)

    # Lanczos steps 1..k-1
    for i in range(1, k):
        # Compute beta and new q (vectorized)
        beta = np.linalg.norm(R, axis=0)  # (n_probes,)
        betas[:, i - 1] = beta

        # A probe that breaks down here (β≈0) has an i-dimensional Krylov
        # subspace; record its effective degree before deactivating it.
        newly_dead = active & (beta < 1e-15)
        deg[newly_dead] = i
        active &= beta >= 1e-15
        if not active.any():
            break

        # Normalize R → q_new (vectorized, with safe division)
        safe_beta = np.where(beta < 1e-15, 1.0, beta)
        q_new = R / safe_beta  # (n, n_probes) — zeros for inactive
        Q_all[:, i, :] = q_new

        # Batched matvec
        R_new = matvec_fn(q_new)  # (n, n_probes) — ONE batched matvec

        # alpha_i = q_new' R_new (vectorized)
        alphas[:, i] = np.sum(q_new * R_new, axis=0)

        # Three-term recurrence: R = R_new - alpha * q_new - beta_prev * Q_prev
        R = R_new - alphas[:, i] * q_new - betas[:, i - 1] * Q_all[:, i - 1, :]

        # Full reorthogonalization (vectorized via einsum)
        # For each probe j, project R[:, j] against Q_all[:, :i+1, j]
        # Q_slice: (n, i+1, n_probes), R: (n, n_probes)
        # coeffs = einsum('nsj,nj->sj', Q_slice, R)  → (i+1, n_probes)
        Q_slice = Q_all[:, : i + 1, :]  # (n, i+1, n_probes)
        proj_coeffs = np.einsum("nsj,nj->sj", Q_slice, R)  # (i+1, n_probes)
        # R -= Q_slice @ proj_coeffs (per-probe)
        R = R - np.einsum("nsj,sj->nj", Q_slice, proj_coeffs)

    # Eigendecompose each probe's tridiagonal (vectorized loop — k×k is tiny).
    # Canonical SLQ weight is n·v₁ᵢ² (unit start scaled by n), not ‖z‖²·v₁ᵢ²:
    # the χ² fluctuation of ‖z‖² is removed and Σᵢ n·v₁ᵢ² = n is exact per
    # probe (constant integrand recovered exactly).  This matches the
    # normalization already used by ``slq_to_chebyshev_coeffs``.
    nodes = np.zeros((n_probes, k), dtype=np.float64)
    weights = np.zeros((n_probes, k), dtype=np.float64)
    z_norms_sq = z_norms**2

    for j in range(n_probes):
        m = deg[j]
        # Build tridiagonal
        T = np.diag(alphas[j, :m])
        if m > 1:
            T += np.diag(betas[j, : m - 1], 1) + np.diag(betas[j, : m - 1], -1)
        theta, eigvecs = np.linalg.eigh(T)
        nodes[j, :m] = theta
        weights[j, :m] = n * eigvecs[0, :] ** 2

    return nodes, weights, z_norms_sq


# ---------------------------------------------------------------------------
# Arnoldi iteration (non-symmetric fallback, complex Ritz values)
# ---------------------------------------------------------------------------


def _arnoldi_iteration(
    W_op: spla.LinearOperator | sp.csr_matrix,
    n: int,
    k: int,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run k steps of Arnoldi on a non-symmetric operator starting from z.

    Returns ``(theta, gamma)`` where ``theta`` are the complex Ritz values of
    the Hessenberg ``H`` and ``gamma`` are the **bilinear** quadrature weights
    for the unit start ``q₁ = e₁``::

        e₁ᵀ f(H) e₁ = Σᵢ γᵢ f(θᵢ),   γᵢ = (e₁ᵀ V)ᵢ (V⁻¹ e₁)ᵢ

    where ``H = V diag(θ) V⁻¹``.  Because ``H`` is non-normal its eigenvectors
    are not orthogonal, so the symmetric-case rule ``|V[0, i]|²`` is wrong —
    the left/right (biorthogonal) product ``V[0, i]·(V⁻¹e₁)ᵢ`` is required and
    is generally complex.
    """
    z_norm = np.linalg.norm(z)
    if z_norm == 0:
        return np.empty(0, dtype=np.complex128), np.empty(0, dtype=np.complex128)

    q = z / z_norm

    Q = np.zeros((n, k), dtype=np.float64)
    H = np.zeros((k, k), dtype=np.float64)
    Q[:, 0] = q

    m = k
    for i in range(k - 1):
        w = W_op @ Q[:, i]
        for j in range(i + 1):
            H[j, i] = float(Q[:, j] @ w)
            w = w - H[j, i] * Q[:, j]
        H[i + 1, i] = np.linalg.norm(w)
        if H[i + 1, i] < 1e-15:
            H = H[: i + 1, : i + 1]
            m = i + 1
            break
        Q[:, i + 1] = w / H[i + 1, i]

    theta, V = np.linalg.eig(H)
    e1 = np.zeros(m, dtype=np.complex128)
    e1[0] = 1.0
    # γ = V[0, :] ∘ (V⁻¹ e₁): the biorthogonal bilinear-form weights.
    gamma = V[0, :] * np.linalg.solve(V, e1)

    return theta, gamma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def slq_logdet_precompute(
    W,
    n_probes: int = 50,
    lanczos_deg: int = 30,
    rng: np.random.Generator | None = None,
) -> SLQPrecompute:
    """Precompute SLQ quadrature rules for log|I - ρW|.

    For undirected-graph W (symmetric sparsity), uses D-symmetrised Lanczos
    with real eigenvalues and valid Gauss quadrature.  For directed W,
    falls back to Arnoldi with complex Ritz values.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse).
    n_probes : int, default 50
    lanczos_deg : int, default 30
    rng : np.random.Generator, optional
        Probe-vector RNG.  Defaults to a *seeded* generator so the
        precomputed quadrature (and thus the logdet approximation) is
        reproducible run-to-run; pass your own Generator to randomize.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if sp.issparse(W) or hasattr(W, "format"):
        W_sp = sp.csr_matrix(W)
        n = W_sp.shape[0]
    else:
        W_arr = np.asarray(W, dtype=np.float64)
        n = W_arr.shape[0]
        W_sp = sp.csr_matrix(W_arr)

    # Try D-symmetrisation
    D = _recover_symmetrizing_diagonal(W_sp)

    if D is not None:
        # D-symmetrised batched Lanczos (real eigenvalues, valid Gauss quadrature)
        # Use the sparse matrix directly for batched matvec (LinearOperator doesn't
        # support matmat efficiently).
        sqrt_D = np.sqrt(D)
        inv_sqrt_D = 1.0 / sqrt_D
        W_sp_for_lanczos = W_sp  # original sparse W

        def batch_matvec(Q_block):
            """W_sym @ Q = sqrt_D * (W @ (inv_sqrt_D * Q)) — handles 2D."""
            return sqrt_D[:, None] * (
                W_sp_for_lanczos @ (inv_sqrt_D[:, None] * Q_block)
            )

        method = "lanczos"
        Z = rng.standard_normal((n, n_probes))
        all_nodes, all_weights, _ = _batched_lanczos(
            batch_matvec, n, lanczos_deg, Z, n_probes
        )
        return SLQPrecompute(nodes=all_nodes, weights=all_weights, n=n, method=method)
    else:
        # Arnoldi fallback (complex Ritz values) — per-probe loop.  Unit
        # starts scaled by n (matching the Lanczos convention); the bilinear
        # weights γ are complex, so ``weights`` is complex here.
        W_op = W_sp
        method = "arnoldi"
        all_nodes = np.zeros((n_probes, lanczos_deg), dtype=np.complex128)
        all_weights = np.zeros((n_probes, lanczos_deg), dtype=np.complex128)

        for j in range(n_probes):
            z = rng.standard_normal(n)
            theta, gamma = _arnoldi_iteration(W_op, n, lanczos_deg, z)
            m = len(theta)
            all_nodes[j, :m] = theta
            all_weights[j, :m] = n * gamma

        return SLQPrecompute(nodes=all_nodes, weights=all_weights, n=n, method=method)


def _slq_log_vals(vals: np.ndarray, method: str) -> np.ndarray:
    """``log(1 - ρθ)`` for the quadrature nodes.

    Lanczos nodes are real, so ``log|1 - ρθ|`` (the real logdet integrand) is
    used directly.  Arnoldi nodes and bilinear weights are complex; the
    complex logarithm is required because ``Re(Σ γᵢ log(1-ρθᵢ))`` keeps the
    cross term ``Im(γ)·Im(log)`` that a magnitude-only log would drop.
    """
    if method == "arnoldi":
        vals = np.where(np.abs(vals) < 1e-300, 1e-300, vals)
        return np.log(vals.astype(np.complex128))
    return np.log(np.maximum(np.abs(vals), 1e-300))


def slq_logdet_eval(pre: SLQPrecompute, rho: float) -> float:
    """Evaluate log|I - ρW| from precomputed SLQ quadrature rules."""
    log_vals = _slq_log_vals(1.0 - rho * pre.nodes, pre.method)
    return float(np.real(np.sum(pre.weights * log_vals)) / pre.n_probes)


def slq_logdet_eval_vec(pre: SLQPrecompute, rho_arr: np.ndarray) -> np.ndarray:
    """Vectorized SLQ logdet evaluation over an array of ρ values."""
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    vals = 1.0 - rho_arr[:, None, None] * pre.nodes[None, :, :]
    log_vals = _slq_log_vals(vals, pre.method)
    return (
        np.real(np.sum(pre.weights[None, :, :] * log_vals, axis=(1, 2))) / pre.n_probes
    )


# ---------------------------------------------------------------------------
# SLQ → Chebyshev coefficient conversion (fast O(m) evaluation per ρ)
# ---------------------------------------------------------------------------


def slq_to_chebyshev_coeffs(
    pre: SLQPrecompute,
    W: sp.csr_matrix | None = None,
    order: int = 20,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
) -> dict:
    """Convert SLQ quadrature rules into Chebyshev polynomial coefficients.

    Uses SLQ's Gauss quadrature to estimate ``tr(W^k)`` for ``k=1..order``,
    then feeds into the same Taylor series + DCT-I pipeline as Barry-Pace.
    Gauss quadrature from *m* Lanczos steps is exact for polynomials of degree
    ``≤ 2m-1``, so ``m=30`` Lanczos steps give exact traces up to ``tr(W^59)``
    — far beyond the 20 traces needed.

    The first two traces (``tr(W)``, ``tr(W²)``) are overridden with exact
    values when ``W`` is provided, matching Barry-Pace's variance reduction.

    Parameters
    ----------
    pre : SLQPrecompute
        Quadrature rules from :func:`slq_logdet_precompute`.
    W : scipy.sparse.csr_matrix, optional
        Spatial weights matrix for exact trace overrides.
    order : int, default 20
        Chebyshev polynomial degree.
    rho_min, rho_max : float
        Interval bounds for the Chebyshev approximation.

    Returns
    -------
    dict
        ``{"coeffs", "rmin", "rmax", "order", "method"}`` — same format as
        :func:`chebyshev`, compatible with :func:`logdet_chebyshev`.
    """
    n = pre.n

    # Recover the per-probe spectral weights (Lanczos: v₁ᵢ²; Arnoldi: the
    # complex bilinear γᵢ) from the stored quadrature weights, which are the
    # n-scaled canonical form (Σᵢ weights = n per probe).  Dividing by that
    # row sum yields the unit-mass weights independent of the n scaling, so
    # ``tr(Wᵏ) ≈ (n / n_probes) Σ_j Σ_i eᵢ θᵢᵏ``.
    weight_sum = np.sum(pre.weights, axis=1)  # = n per probe
    e1_sq = pre.weights / weight_sum[:, None]  # (n_probes, k)
    nodes = pre.nodes  # real (lanczos) or complex (arnoldi)

    # Estimate traces via Gauss quadrature: tr(W^k) = (n/n_probes) Σ_j Σ_i e1² θ^k
    # (real part: Arnoldi's complex quadrature estimates a real trace).
    traces = np.zeros(order, dtype=np.float64)
    for p in range(1, order + 1):
        traces[p - 1] = n * np.real(np.mean(np.sum(e1_sq * nodes**p, axis=1)))

    # Override first two traces with exact values (major variance reduction)
    if W is not None:
        W_csr = sp.csr_matrix(W) if not sp.issparse(W) else W
        traces[0] = float(W_csr.diagonal().sum())
        if order >= 2:
            traces[1] = float(W_csr.multiply(W_csr.T).sum())

    # Taylor series: log|I - ρW| = -Σ_k tr(W^k) ρ^k / k
    # Evaluate at Chebyshev nodes, then DCT-I → Chebyshev coefficients
    k_arr = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k_arr - 1) * np.pi / (2 * order))
    rho_nodes = 0.5 * (rho_max - rho_min) * nodes_cos + 0.5 * (rho_max + rho_min)

    td = traces / np.arange(1, order + 1, dtype=np.float64)
    logdet_at_nodes = np.zeros(order, dtype=np.float64)
    for idx, r in enumerate(rho_nodes):
        powers = np.power(r, np.arange(1, order + 1, dtype=np.float64))
        logdet_at_nodes[idx] = -powers @ td

    # DCT-I → Chebyshev coefficients
    coeffs = np.zeros(order, dtype=np.float64)
    for j in range(order):
        scale = 2.0 / order if j > 0 else 1.0 / order
        coeffs[j] = scale * np.sum(
            logdet_at_nodes * np.cos(j * (2 * k_arr - 1) * np.pi / (2 * order))
        )

    return {
        "coeffs": coeffs,
        "rmin": rho_min,
        "rmax": rho_max,
        "order": order,
        "method": "slq",
    }
