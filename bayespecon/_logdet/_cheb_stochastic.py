"""Stochastic Chebyshev expansion of log|I - ρW|.

Extends the Barry-Pace idea (:cite:t:`barry1999MonteCarlo`) by replacing the
Taylor series ``log(1-ρx) = -Σ ρ^k x^k / k`` with a **Chebyshev expansion** in
the operator ``W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)``, following
:cite:t:`han2015LargescaleLogdeterminant`.

The Taylor truncation error is ``O(|ρλ_max|^p / (p(1-|ρ|)))`` — algebraic decay
that degrades as ``ρ → 1``.  The Chebyshev truncation error is ``O(ν^{-p})``
(Bernstein ellipse geometric decay) — **uniform across the entire ρ interval**.

Same computational structure as Barry-Pace:

* **Precompute**: ``p`` batched sparse matvecs via three-term recurrence
  ``v_{j+1} = 2W̃v_j - v_{j-1}``, ``k`` probes batched.  No reorthogonalization.
* **Per-ρ eval**: ``O(p)`` Clenshaw-like evaluation:
  ``(c₀(ρ)/2)·n + Σ c_j(ρ)·μ_j``.

**Deflation** (optional): When ``n_deflate > 0`` *and* ``W`` is symmetrizable
(undirected graph), the top-``n_deflate`` **eigenpairs** (by magnitude) of the
D-symmetrized, rescaled operator ``W̃_sym = D^{1/2} W̃ D^{-1/2}`` are captured
exactly via ``eigsh`` (applied matrix-free, never materialised) and removed
from the residual; stochastic Chebyshev then runs only on the deflated
residual, whose Frobenius norm — and hence Hutchinson variance — is smaller.
Because Chebyshev traces are similarity-invariant, ``tr(T_j(W̃_sym)) =
tr(T_j(W̃))``, and an eigenpair split decomposes the trace exactly
(``tr(T_j(W̃)) = Σᵢ T_j(λᵢ) − r·T_j(0) + tr(T_j(W̃_res))``) — unlike the
non-invariant singular-value split it replaced.  Directed W has an asymmetric
sparsity pattern and is not symmetrizable, so deflation is skipped with a
warning.  Spatial spectra are often flat enough that deflation barely helps,
so it is **off by default** (``n_deflate=0``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._slq import _recover_symmetrizing_diagonal


@dataclass(frozen=True)
class ChebStochasticPrecompute:
    """Precomputed stochastic Chebyshev moments for log|I - ρW|.

    Attributes
    ----------
    moments : np.ndarray, shape (order + 1,)
        Stochastic estimates of ``tr(T_j(W̃))`` for ``j = 0, .., order``,
        where ``W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)``.
    lam_min : float
        Lower spectral bound used for rescaling.
    lam_max : float
        Upper spectral bound used for rescaling.
    order : int
        Chebyshev polynomial degree (number of moments minus one).
    n : int
        Matrix dimension.
    """

    moments: np.ndarray
    lam_min: float
    lam_max: float
    order: int
    n: int


# ---------------------------------------------------------------------------
# Spectral bounds estimation
# ---------------------------------------------------------------------------


def _estimate_spectral_bounds(
    W: sp.csr_matrix,
    n_iters: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate [λ_min, λ_max] of W.

    For row-standardised W, λ_max = 1 (Perron) and |λ| ≤ ‖W‖_∞ = 1
    (Gershgorin), so the conservative bracket [-1, 1] is always valid.
    We use power iteration to tighten λ_max, and Gershgorin for λ_min.

    Looseness in the bracket costs convergence rate (more Chebyshev terms
    needed) but never correctness — the Chebyshev expansion still converges
    on the larger interval, just slower.

    Parameters
    ----------
    W : sp.csr_matrix
        Spatial weights matrix.
    n_iters : int, default 10
        Power iteration steps for λ_max refinement.
    rng : np.random.Generator, optional
    """
    n = W.shape[0]

    # Power iteration for λ_max
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iters):
        v = W @ v
        norm = np.linalg.norm(v)
        if norm < 1e-300:
            break
        v /= norm
    lam_max = float(np.real(v @ (W @ v)))

    # For row-standardised W, λ_max = 1 (Perron).  Be slightly conservative.
    lam_max = max(lam_max, 1.0)

    # Gershgorin bound: |λ| ≤ ‖W‖_∞ = max row sum = 1 for row-standardised
    # Conservative: lam_min = -lam_max
    lam_min = -lam_max

    return lam_min, lam_max


# ---------------------------------------------------------------------------
# Stochastic Chebyshev moments via three-term recurrence
# ---------------------------------------------------------------------------


def _chebyshev_moments(
    matvec,
    n: int,
    order: int,
    n_probes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate ``tr(T_j(A))`` for ``j = 0, .., order`` via Hutchinson probes.

    ``matvec`` applies the operator ``A`` (spectrum in [-1, 1]) to an
    ``(n, n_probes)`` block; it may be a bare sparse matmul or a matrix-free
    closure (e.g. the deflated residual).  Uses the three-term recurrence::

        v_0 = ω,  v_1 = Aω,  v_{j+1} = 2Av_j - v_{j-1}

    with moment estimate ``μ̂_j = (n/k) Σ_l ω_l^T v_j^{(l)} / ‖ω_l‖²``.

    Cost: ``order`` batched matvecs — identical to Barry-Pace, no
    reorthogonalization.

    Parameters
    ----------
    matvec : callable
        ``(n, n_probes) -> (n, n_probes)`` application of the operator.
    n : int
        Matrix dimension.
    order : int
        Maximum Chebyshev degree.
    n_probes : int
        Number of Hutchinson probe vectors.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Moment estimates ``μ̂_0, ..., μ̂_order``.  ``μ̂_0 = n`` is exact; the
        caller may override ``μ̂_1 = tr(A)`` with its exact value.
    """
    U = rng.standard_normal((n, n_probes))
    utu = np.einsum("ij,ij->j", U, U)

    # μ_0 = tr(T_0(A)) = tr(I) = n (exact)
    moments = np.zeros(order + 1, dtype=np.float64)
    moments[0] = float(n)

    # Three-term recurrence: v_0 = U, v_1 = A @ U
    v_prev = U
    v_curr = matvec(U)  # (n, n_probes) — 1st batched matvec
    moments[1] = n * np.mean(np.einsum("ij,ij->j", U, v_curr) / utu)

    for j in range(1, order):
        # v_{j+1} = 2 A v_j - v_{j-1}
        v_next = 2.0 * matvec(v_curr) - v_prev  # batched matvec

        # μ̂_{j+1} = (n / k) * Σ ω^T v_{j+1} / ‖ω‖²
        moments[j + 1] = n * np.mean(np.einsum("ij,ij->j", U, v_next) / utu)

        v_prev = v_curr
        v_curr = v_next

    return moments


def _cheb_recurrence(x: np.ndarray, order: int) -> np.ndarray:
    """Chebyshev polynomials ``T_j(x)`` for ``j = 0, .., order``.

    Parameters
    ----------
    x : np.ndarray, shape (r,)
        Evaluation points (assumed in [-1, 1]).
    order : int

    Returns
    -------
    np.ndarray, shape (order + 1, r)
        ``T[j, i] = T_j(x_i)``.
    """
    T = np.empty((order + 1, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    if order >= 1:
        T[1] = x
    for j in range(1, order):
        T[j + 1] = 2.0 * x * T[j] - T[j - 1]
    return T


def _deflated_moments(
    W_sp: sp.csr_matrix,
    D: np.ndarray,
    lam_min: float,
    lam_max: float,
    r: int,
    order: int,
    n_probes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Eigen-deflated Chebyshev moments for a symmetrizable ``W``.

    Captures the top-``r`` eigenpairs (by magnitude) of the D-symmetrized,
    rescaled operator ``W̃_sym`` exactly and estimates the remaining moments
    on the deflated residual.  Both operators are applied matrix-free — the
    ``n × n`` low-rank correction is never materialised.

    Combination (deflated directions sit at eigenvalue 0 in the residual)::

        tr(T_j(W̃)) = Σᵢ T_j(λᵢ) − r·T_j(0) + tr(T_j(W̃_res))

    with ``T_j(0) = cos(jπ/2)``.  ``μ₀`` and ``μ₁`` are overridden with their
    exact values by the caller.
    """
    n = W_sp.shape[0]
    spread = lam_max - lam_min
    sqrt_D = np.sqrt(D)
    inv_sqrt_D = 1.0 / sqrt_D
    scale = 2.0 / spread
    shift = (lam_max + lam_min) / spread

    def wtilde_sym(B: np.ndarray) -> np.ndarray:
        """Apply ``W̃_sym = scale·(D^{1/2} W D^{-1/2}) − shift·I`` to a block."""
        two_d = B if B.ndim == 2 else B[:, None]
        WB = sqrt_D[:, None] * (W_sp @ (inv_sqrt_D[:, None] * two_d))
        out = scale * WB - shift * two_d
        return out if B.ndim == 2 else out[:, 0]

    # Top-r eigenpairs by magnitude of the rescaled symmetric operator
    # (which="LM" captures both ends of the [-1, 1] spectrum).
    op = spla.LinearOperator(
        (n, n),
        matvec=wtilde_sym,
        rmatvec=wtilde_sym,
        matmat=wtilde_sym,
        dtype=np.float64,
    )
    try:
        lam_eig, U_eig = spla.eigsh(op, k=r, which="LM")
    except spla.ArpackNoConvergence as exc:  # pragma: no cover - defensive
        lam_eig, U_eig = exc.eigenvalues, exc.eigenvectors
        r = lam_eig.shape[0]
        warnings.warn(
            "eigsh did not converge during deflation; proceeding with the "
            f"{r} eigenpairs that did converge.",
            stacklevel=2,
        )
    if r == 0:  # pragma: no cover - defensive
        U_eig = np.zeros((n, 0), dtype=np.float64)
        lam_eig = np.zeros(0, dtype=np.float64)

    def residual_matvec(B: np.ndarray) -> np.ndarray:
        """W̃_res @ B = W̃_sym @ B − U (λ ∘ (Uᵀ B))."""
        return wtilde_sym(B) - U_eig @ (lam_eig[:, None] * (U_eig.T @ B))

    res_moments = _chebyshev_moments(residual_matvec, n, order, n_probes, rng)

    # Exact contribution of the r captured eigenpairs: Σᵢ T_j(λᵢ).
    eig_moments = _cheb_recurrence(np.clip(lam_eig, -1.0, 1.0), order).sum(axis=1)

    # Deflated directions contribute T_j(0) = cos(jπ/2) = [1, 0, -1, 0, ...].
    Tj0 = np.zeros(order + 1, dtype=np.float64)
    Tj0[0::4] = 1.0
    Tj0[2::4] = -1.0

    return eig_moments - r * Tj0 + res_moments


# ---------------------------------------------------------------------------
# Chebyshev coefficients of log|a - b·x| on [-1, 1]
# ---------------------------------------------------------------------------


def _log_cheb_coeffs(
    rho: float,
    lam_min: float,
    lam_max: float,
    order: int,
) -> np.ndarray:
    """Compute Chebyshev coefficients ``c_j(ρ)`` of ``log|a - b·x|`` on [-1, 1].

    Here ``a = 1 - ρ(λ_max+λ_min)/2`` and ``b = ρ(λ_max-λ_min)/2``, so that::

        log|I - ρW| = log|a - b·W̃|

    where ``W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)`` has spectrum in [-1, 1].

    Coefficients are computed via Clenshaw-Curtis quadrature (DCT-I), which is
    O(p²) — cheap for ``p ≤ 50``.

    Parameters
    ----------
    rho : float
        Spatial autoregressive parameter.
    lam_min, lam_max : float
        Spectral bounds of W.
    order : int
        Chebyshev polynomial degree.

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Chebyshev coefficients ``c_0, c_1, ..., c_order``.
    """
    a = 1.0 - rho * (lam_max + lam_min) / 2.0
    b = rho * (lam_max - lam_min) / 2.0

    if abs(b) < 1e-300:
        # ρ ≈ 0: log|a| = log(1) = 0
        return np.zeros(order + 1, dtype=np.float64)

    # Clenshaw-Curtis nodes: x_j = cos(πj/order), j = 0, ..., order
    k = np.arange(order + 1)
    x_nodes = np.cos(np.pi * k / order)
    f_vals = np.log(np.abs(a - b * x_nodes))

    # DCT-I: c_j = (2/order) * Σ w_k * f_k * cos(j*π*k/order)
    # with w_0 = w_order = 0.5, w_k = 1 otherwise
    w = np.ones(order + 1, dtype=np.float64)
    w[0] = 0.5
    w[-1] = 0.5

    coeffs = np.zeros(order + 1, dtype=np.float64)
    for j in range(order + 1):
        coeffs[j] = (2.0 / order) * np.sum(w * f_vals * np.cos(j * np.pi * k / order))
    coeffs[0] /= 2.0  # c_0/2 convention: store c₀/2 so eval uses sum c_j μ_j directly

    return coeffs


# ---------------------------------------------------------------------------
# Public API: precompute + eval
# ---------------------------------------------------------------------------


def cheb_stochastic_logdet_precompute(
    W,
    order: int = 15,
    n_probes: int = 50,
    n_deflate: int = 0,
    lam_min: float | None = None,
    lam_max: float | None = None,
    rng: np.random.Generator | None = None,
) -> ChebStochasticPrecompute:
    """Precompute stochastic Chebyshev moments for ``log|I - ρW|``.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse).
    order : int, default 15
        Chebyshev polynomial degree.  Truncation converges geometrically
        (Bernstein ellipse), so 15 terms suffice for ~0.3% accuracy at ρ=0.9
        — far fewer than Barry-Pace's 20-30 Taylor terms.
    n_probes : int, default 50
        Number of Hutchinson probes for moment estimation.  Since
        ``‖T_j(W̃)‖₂ ≤ 1`` uniformly in *j*, variance is bounded and 50
        probes match Barry-Pace's 100-probe accuracy at half the cost.
    n_deflate : int, default 0
        Number of top eigenpairs (by magnitude) to deflate exactly.  When
        ``n_deflate > 0`` *and* ``W`` is symmetrizable (undirected graph),
        the top-``n_deflate`` eigenpairs of the D-symmetrized, rescaled
        operator are captured with no stochastic noise (matrix-free
        ``eigsh``) and stochastic Chebyshev is applied only to the deflated
        residual.  Directed W is skipped with a warning.  Auto-selected as
        ``min(5, n // 100)`` when set to ``-1``.  Off by default: on the flat
        spectra typical of spatial ``W`` the variance reduction is usually
        marginal.
    lam_min, lam_max : float, optional
        Spectral bounds of W.  If not provided, estimated via power iteration.
    rng : np.random.Generator, optional
        Probe-vector RNG.  Defaults to a *seeded* generator so the
        precomputed moments (and thus the logdet approximation) are
        reproducible run-to-run; pass your own Generator to randomize.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if sp.issparse(W) or hasattr(W, "format"):
        W_sp = sp.csr_matrix(W, dtype=np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))

    n = W_sp.shape[0]

    # Auto-select deflation count
    if n_deflate == -1:
        n_deflate = min(5, n // 100)

    # Spectral bounds
    if lam_min is None or lam_max is None:
        est_min, est_max = _estimate_spectral_bounds(W_sp, rng=rng)
        lam_min = est_min if lam_min is None else lam_min
        lam_max = est_max if lam_max is None else lam_max

    # Rescale W → W̃ with spectrum in [-1, 1]
    # W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)
    spread = lam_max - lam_min
    if spread < 1e-300:
        # Degenerate case: W is a scalar multiple of I (shouldn't happen for spatial W)
        W_tilde = sp.csr_matrix((n, n), dtype=np.float64)
    else:
        W_tilde = (2.0 / spread) * W_sp - ((lam_max + lam_min) / spread) * sp.eye(
            n, format="csr"
        )
        W_tilde = W_tilde.tocsr()

    # Exact μ₁ = tr(W̃) (variance reduction; overrides the stochastic estimate).
    mu1_exact = float(W_tilde.diagonal().sum())

    # Deflation (symmetrizable W only): capture the top-r eigenpairs exactly
    # and run stochastic Chebyshev on the lower-variance deflated residual.
    r = min(n_deflate, n - 2) if n_deflate > 0 else 0
    D = _recover_symmetrizing_diagonal(W_sp) if r >= 1 and spread >= 1e-300 else None
    if n_deflate > 0 and r >= 1 and D is None:
        warnings.warn(
            "Eigen-deflation requires a symmetrizable (undirected) W, but the "
            "sparsity pattern of W is asymmetric (directed graph). Falling back "
            "to plain stochastic Chebyshev; n_deflate is ignored.",
            stacklevel=2,
        )

    if D is not None:
        moments = _deflated_moments(W_sp, D, lam_min, lam_max, r, order, n_probes, rng)
    else:
        # No deflation: standard stochastic Chebyshev on W̃.
        moments = _chebyshev_moments(lambda B: W_tilde @ B, n, order, n_probes, rng)

    # Exact low-order overrides on the total: μ₀ = n, μ₁ = tr(W̃).
    moments[0] = float(n)
    moments[1] = mu1_exact

    return ChebStochasticPrecompute(
        moments=moments,
        lam_min=lam_min,
        lam_max=lam_max,
        order=order,
        n=n,
    )


def cheb_stochastic_logdet_eval(pre: ChebStochasticPrecompute, rho: float) -> float:
    """Evaluate ``log|I - ρW|`` from precomputed stochastic Chebyshev moments.

    Computes Chebyshev coefficients ``c_j(ρ)`` on-the-fly (O(p²) via
    Clenshaw-Curtis), then evaluates::

        log|I - ρW| ≈ (c₀/2)·n + Σ_{j=1}^p c_j·μ_j

    Parameters
    ----------
    pre : ChebStochasticPrecompute
        Precomputed moments from :func:`cheb_stochastic_logdet_precompute`.
    rho : float
        Spatial autoregressive parameter.
    """
    coeffs = _log_cheb_coeffs(rho, pre.lam_min, pre.lam_max, pre.order)
    # log|I - ρW| = Σ c_j · μ_j  (c₀ already includes /2 convention from _log_cheb_coeffs)
    val = 0.0
    for j in range(pre.order + 1):
        val += coeffs[j] * pre.moments[j]
    return float(val)


def cheb_stochastic_logdet_eval_vec(
    pre: ChebStochasticPrecompute, rho_arr: np.ndarray
) -> np.ndarray:
    """Vectorized evaluation over an array of ρ values.

    Precomputes Chebyshev coefficients for each ρ, then evaluates via
    matrix-vector product (coeffs @ moments).
    """
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    n_rho = len(rho_arr)

    # Build coefficient matrix: (n_rho, order+1)
    all_coeffs = np.zeros((n_rho, pre.order + 1), dtype=np.float64)
    for i in range(n_rho):
        all_coeffs[i] = _log_cheb_coeffs(
            rho_arr[i], pre.lam_min, pre.lam_max, pre.order
        )

    # logdet_i = Σ c_j · μ_j  (c₀ already includes /2 convention)
    return all_coeffs @ pre.moments
