r"""Pólya–Gamma Gibbs sampler for the *reduced-form* SAR-NB model.

Target posterior
----------------
.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \quad
    \mu_i = \exp(\eta_i), \quad
    \eta = (I - \rho W)^{-1} X \beta.

Parameters are :math:`(\beta, \rho, \alpha)`.  Unlike
:mod:`bayespecon.samplers.negbin`, the latent field :math:`\eta` is a
*deterministic* function of :math:`(\beta, \rho)` — there is no
:math:`\sigma`-noise term and no n-dimensional augmentation step.

Pólya–Gamma augmentation
------------------------
With auxiliary variables :math:`\omega_i \sim \mathrm{PG}(y_i + \alpha,
\psi_i)` where :math:`\psi_i = \eta_i - \log\alpha`, the NB
log-likelihood becomes quadratic in :math:`\psi` with working response
:math:`\kappa_i = (y_i - \alpha)/2`.  Writing
:math:`\tilde X = (I - \rho W)^{-1} X`, the conditional posterior of
:math:`\beta` is the conjugate Gaussian

.. math::

    \beta \mid \omega, \rho, \alpha, y \;\sim\; N(m_\beta, \Sigma_\beta), \\
    \Sigma_\beta^{-1} = \tilde X^\top \Omega \tilde X + V_0^{-1}, \\
    m_\beta = \Sigma_\beta \bigl(\tilde X^\top (\kappa + \omega \log\alpha)
                                  + V_0^{-1} \mu_0\bigr).

Sweep
-----
Four blocks per iteration:

1. **ω | β, ρ, α, y** — vectorised PG draw at :math:`\psi`.
2. **β | ω, ρ, α, y** — conjugate normal via the construction above.
   Requires building :math:`\tilde X = A_\rho^{-1} X` (one sparse LU
   factorisation of :math:`A_\rho = I - \rho W` plus k triangular
   solves), then a :math:`k \times k` Cholesky.
3. **ρ | ω, α, y** — 1-D adaptive slice sampler on the
   **β-marginalised** conditional density.  With working response
   :math:`s_i = (y_i - \alpha) / (2\omega_i) + \log\alpha` and
   :math:`U(\rho) = A_\rho^{-1} X`, integrating out :math:`\beta\sim
   N(b_0, V_0)` gives :math:`s\mid\rho,\omega,\alpha \sim
   N(U b_0,\,\Omega^{-1} + U V_0 U^\top)`.  Via the matrix-determinant
   lemma / Woodbury identity with :math:`M(\rho) = V_0^{-1} + U^\top
   \Omega U` and :math:`r = s - U b_0`,

   .. math::

       \log p(\rho \mid \cdot)
         = -\tfrac{1}{2} \log |M(\rho)|
           - \tfrac{1}{2}\bigl(r^\top \Omega r
               - (U^\top \Omega r)^\top M^{-1} (U^\top \Omega r)\bigr)
           + \log p_0(\rho),

   up to terms independent of :math:`\rho`.  Marginalising β inside
   the ρ update breaks the β–ρ posterior correlation that would
   otherwise dominate single-site ρ mixing.  No :math:`\log|A_\rho|`
   Jacobian appears (η is not being integrated out).
4. **α | y, η** — 1-D slice on :math:`\log\alpha` of the NB
   log-likelihood + Half-Student-t prior.  Reuses
   :func:`bayespecon.samplers.negbin._core._sample_alpha`.

Per-sweep cost is dominated by the :math:`\rho` slice: each candidate
evaluation requires one sparse LU of :math:`A_\rho` plus one solve for
the n-vector :math:`X\beta`.  For row-standardised sparse :math:`W` this
is :math:`O(\mathrm{nnz}^{1.5})` per candidate (typically 5–15
candidates per draw).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from .._utils._spatial_normal import CholmodFactor
from ..negbin._core import _nb_loglik_pointwise, _sample_alpha

# ---------------------------------------------------------------------------
# Sparse solve for A_ρ = I − ρW
# ---------------------------------------------------------------------------

# When CHOLMOD is available we solve (I − ρW) x = b via the normal
# equations  A^T A x = A^T b  where  A^T A = I − ρ(W+W^T) + ρ² W^T W
# is SPD.  This avoids UMFPACK (``splu``), which can deadlock on macOS
# when Apple Accelerate BLAS is called concurrently from multiple
# processes.  CHOLMOD is also faster because it reuses the symbolic
# analysis across all ρ values.


class _CholmodNormalEqSolver:
    """Solve ``(I − ρW) x = b`` via CHOLMOD on the normal equations.

    The normal-equation matrix

    .. math::

        A_\\rho^T A_\\rho = I - \\rho (W + W^T) + \\rho^2 W^T W

    is symmetric positive definite for any non-singular ``A_ρ``.
    CHOLMOD factorises it and we recover ``x = A_ρ⁻¹ b`` from
    ``A_ρ^T A_ρ x = A_ρ^T b``.

    The ``cholmod_factor`` holds the symbolic analysis (computed once
    from a pattern matrix) so that only the numeric factorisation is
    needed for each new ρ.

    Parameters
    ----------
    cholmod_factor : CholmodFactor
        Pre-built CHOLMOD factor with the sparsity pattern of
        ``A^T A`` (any valid ρ gives the same pattern).
    W_csc : csc_matrix
        Spatial weights in CSC format.
    W_sym : csc_matrix
        ``W + W^T`` in CSC format (precomputed).
    WtW : csc_matrix
        ``W^T @ W`` in CSC format (precomputed).
    n : int
        Dimension of the spatial weights matrix.
    """

    def __init__(
        self,
        cholmod_factor: CholmodFactor,
        W_csc: sp.csc_matrix,
        W_sym: sp.csc_matrix,
        WtW: sp.csc_matrix,
        n: int,
    ) -> None:
        self._cholmod = cholmod_factor
        self._W_csc = W_csc
        self._W_sym = W_sym
        self._WtW = WtW
        self._n = n
        self._rho: float | None = None

    def factorize(self, rho: float) -> None:
        """Build and factorise ``A^T A`` at the given ρ."""
        AtA = sp.eye(self._n, format="csc") - rho * self._W_sym + rho**2 * self._WtW
        self._cholmod.factorize(AtA)
        self._rho = rho

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``(I − ρW) x = rhs`` via the normal equations.

        Works for both vector and matrix right-hand sides.
        """
        # A^T b  where A = I − ρW
        Atb = rhs - self._W_csc.T @ (self._rho * rhs)
        # CHOLMOD solve: A^T A x = A^T b
        return self._cholmod.solve(Atb)


def _factor_A(rho: float, W_csc: sp.csc_matrix, n: int) -> spla.SuperLU:
    """Factorise :math:`A_\\rho = I - \\rho W` via sparse LU (``splu``).

    Returns a :class:`scipy.sparse.linalg.SuperLU` object whose
    :meth:`solve` method handles single and multiple right-hand sides.

    .. deprecated::
        Used only as a fallback when CHOLMOD is not available.
        The CHOLMOD normal-equations path (``_CholmodNormalEqSolver``)
        is preferred to avoid UMFPACK deadlocks on macOS.
    """
    A = (sp.eye(n, format="csc") - rho * W_csc).tocsc()
    return spla.splu(A)


def _make_cholmod_pattern(
    W_csc: sp.csc_matrix,
    n: int,
) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
    """Precompute the CHOLMOD pattern matrix and sparse building blocks.

    Returns
    -------
    W_sym : csc_matrix
        ``W + W^T`` in CSC format.
    WtW : csc_matrix
        ``W^T @ W`` in CSC format.
    pattern : csc_matrix
        Sparsity pattern for ``A^T A = I − ρ(W+W^T) + ρ² W^T W``
        that covers all valid ρ.  Built as
        ``I + 0.5*(W+W^T) + 0.25*W^T@W`` so that every possible
        fill-in position is present.
    """
    W_sym = (W_csc + W_csc.T).tocsc()
    WtW = (W_csc.T @ W_csc).tocsc()
    pattern = (sp.eye(n, format="csc") + 0.5 * W_sym + 0.25 * WtW).tocsc()
    return W_sym, WtW, pattern


def _make_solver(
    rho: float,
    W_csc: sp.csc_matrix,
    n: int,
    cholmod_solver: _CholmodNormalEqSolver | None = None,
) -> _CholmodNormalEqSolver | spla.SuperLU:
    """Return a solver for ``(I − ρW) x = b``.

    When ``cholmod_solver`` is provided (CHOLMOD available), factorises
    the normal-equation matrix ``A^T A`` and returns the solver.
    Otherwise falls back to ``splu`` (UMFPACK).

    Both return types expose a ``.solve(rhs)`` method.
    """
    if cholmod_solver is not None:
        cholmod_solver.factorize(rho)
        return cholmod_solver
    return _factor_A(rho, W_csc, n)


# ---------------------------------------------------------------------------
# Shift-invert Krylov basis for fast ρ-slice evaluation
# ---------------------------------------------------------------------------

# Default Krylov degree and maximum |Δρ| for polynomial approximation.
_KRYLOV_DEGREE_DEFAULT = 8
_KRYLOV_DMAX_DEFAULT = 0.15


class ReducedKrylovBasis(NamedTuple):
    """Precomputed shift-invert Krylov basis for fast ρ-slice evaluation.

    At a centre point :math:`\\rho_c`, we factorise
    :math:`A_c = I - \\rho_c W` once and build the basis

    .. math::

        V_0 = A_c^{-1} X, \\quad
        V_{j+1} = A_c^{-1} (W V_j), \\quad j = 0, \\dots, m-1.

    For any nearby :math:`\\rho = \\rho_c + \\Delta\\rho` the
    β-marginalised slice density only needs

    .. math::

        U(\\rho) \\approx \\sum_{j=0}^{m} (\\Delta\\rho)^j V_j,

    which is a cheap ``einsum`` instead of a fresh factorisation.
    The approximation error decays geometrically in :math:`m` as
    :math:`O((\\Delta\\rho \\|A_c^{-1} W\\|)^{m+1})`.

    Attributes
    ----------
    rho_basis : float
        Centre point :math:`\\rho_c` at which the system was factored.
    solver : _CholmodNormalEqSolver or spla.SuperLU
        The factored solver for :math:`A_c = I - \\rho_c W`.
    V_stack : ndarray, shape (m+1, n, k)
        Krylov basis vectors stacked along axis 0.
    degree : int
        Krylov degree :math:`m` (number of correction terms beyond
        :math:`V_0`).
    """

    rho_basis: float
    solver: _CholmodNormalEqSolver | spla.SuperLU
    V_stack: np.ndarray
    degree: int


def _build_krylov_basis(
    rho_c: float,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    degree: int = _KRYLOV_DEGREE_DEFAULT,
    cholmod_solver: _CholmodNormalEqSolver | None = None,
) -> ReducedKrylovBasis:
    """Build a shift-invert Krylov basis at :math:`\\rho_c`.

    When ``cholmod_solver`` is provided, uses CHOLMOD normal equations
    instead of ``splu`` (UMFPACK).  Cost: 1 factorisation +
    ``(degree + 1)`` solve calls + ``degree`` sparse matmuls ``W @ V_j``.
    """
    solver = _make_solver(rho_c, W_csc, n, cholmod_solver=cholmod_solver)
    V0 = solver.solve(X)  # (n, k)
    m = degree
    V_stack = np.empty((m + 1, n, X.shape[1]), dtype=np.float64)
    V_stack[0] = V0
    for j in range(m):
        Wv = W_csc @ V_stack[j]  # (n, k)
        V_stack[j + 1] = solver.solve(Wv)
    return ReducedKrylovBasis(rho_basis=rho_c, solver=solver, V_stack=V_stack, degree=m)


def _eval_U_from_basis(
    basis: ReducedKrylovBasis,
    drho: float,
) -> np.ndarray:
    """Evaluate :math:`U(\\rho_c + \\Delta\\rho) \\approx \\sum (\\Delta\\rho)^j V_j`.

    Uses Horner's method for numerical stability.
    """
    # Horner: V_0 + drho*(V_1 + drho*(V_2 + ... + drho*V_m))
    result = basis.V_stack[basis.degree].copy()
    for j in range(basis.degree - 1, -1, -1):
        result = basis.V_stack[j] + drho * result
    return result


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReducedGibbsState:
    """Mutable state for one chain of the reduced-form SAR-NB Gibbs sampler.

    Parameters
    ----------
    beta : ndarray, shape (k,)
        Regression coefficients.
    rho : float
        Spatial autoregressive parameter.
    alpha : float
        NB dispersion (NB2 parameterisation; ``Var(y) = mu + mu^2 / alpha``).
    omega : ndarray, shape (n,)
        Pólya–Gamma auxiliary variables.
    """

    beta: np.ndarray
    rho: float
    alpha: float
    omega: np.ndarray


@dataclass
class ReducedGibbsPriors:
    """Prior hyperparameters for the reduced-form SAR-NB sampler.

    Notes
    -----
    The ``sigma2_*`` fields present on the structural-form
    :class:`bayespecon.samplers.negbin.GibbsPriors` are intentionally
    absent — this sampler has no :math:`\\sigma` parameter.
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    alpha_sigma: float = 2.5  # Half-Student-t scale for α
    alpha_nu: float = 3.0  # Half-Student-t degrees of freedom for α
    rho_lower: float = -0.999
    rho_upper: float = 0.999


class ReducedGibbsCache(NamedTuple):
    """Constants reused across sweeps.

    Attributes
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised spatial weights (csr for fast matvec).
    W_csc : scipy.sparse.csc_matrix
        Same matrix in csc format (preferred for ``splu`` fallback).
    rho_lower, rho_upper : float
        Support bounds for the ρ slice sampler.
    rho_adaptive_width : bool
        Whether to tune the ρ slice-sampler width during warmup.
    rho_slice_width_state : SliceWidthState
        Mutable width state for the adaptive ρ slice sampler.
    krylov_degree : int
        Krylov basis degree :math:`m` for the shift-invert polynomial
        approximation of :math:`(I - \\rho W)^{-1} X`.  Default 8.
        Set to 0 to disable Krylov acceleration (use exact solve per
        candidate, as in the legacy path).
    krylov_dmax : float
        Maximum :math:`|\\Delta\\rho|` for which the Krylov basis is
        used.  When a slice candidate falls outside this radius around
        the basis centre, a fresh factorisation is performed for
        that single candidate.  Default 0.15.
    cholmod_pattern : csc_matrix or None
        When CHOLMOD is available, a sparse matrix with the sparsity
        pattern for ``A^T A = I − ρ(W+W^T) + ρ² W^T W``.  The
        ``CholmodFactor`` is created from this pattern **in the worker
        process**, avoiding CHOLMOD/BLAS calls in the parent that
        accumulate state and cause deadlocks after many ``fit()`` calls.
        ``None`` when CHOLMOD is not installed (falls back to ``splu``).
    W_sym : csc_matrix or None
        ``W + W^T`` in CSC format (precomputed for CHOLMOD path).
    WtW : csc_matrix or None
        ``W^T @ W`` in CSC format (precomputed for CHOLMOD path).
    """

    W_sparse: sp.csr_matrix
    W_csc: sp.csc_matrix
    rho_lower: float
    rho_upper: float
    rho_adaptive_width: bool = True
    rho_slice_width_state: Optional[SliceWidthState] = None
    krylov_degree: int = _KRYLOV_DEGREE_DEFAULT
    krylov_dmax: float = _KRYLOV_DMAX_DEFAULT
    cholmod_pattern: Optional[sp.csc_matrix] = None
    W_sym: Optional[sp.csc_matrix] = None
    WtW: Optional[sp.csc_matrix] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ``_factor_A`` is defined above (alongside the CHOLMOD solver).


def _compute_eta(
    rho: float,
    Xbeta: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    cholmod_solver: _CholmodNormalEqSolver | None = None,
) -> tuple[np.ndarray, _CholmodNormalEqSolver | spla.SuperLU]:
    """Return :math:`\\eta = (I - \\rho W)^{-1} X\\beta` and the solver.

    The solver is returned so callers can reuse it (e.g. to compute
    :math:`\\tilde X = A^{-1} X` without re-factorising).
    """
    solver = _make_solver(rho, W_csc, n, cholmod_solver=cholmod_solver)
    eta = solver.solve(Xbeta)
    return eta


# ---------------------------------------------------------------------------
# Gibbs block samplers
# ---------------------------------------------------------------------------


def _sample_omega(
    y: np.ndarray,
    alpha: float,
    psi: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 1: draw :math:`\\omega \\sim \\mathrm{PG}(y + \\alpha, \\psi)`.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Integer responses.
    alpha : float
        NB dispersion parameter.
    psi : ndarray, shape (n,)
        Current tilting :math:`\\psi = \\eta - \\log\\alpha`.

    Notes
    -----
    The shape parameter ``h = y + alpha`` is clamped to ``1e-3`` to avoid
    the ``polyagamma`` ``"alternate"`` method's rejection of values
    below :math:`\\sim 10^{-3}` (see notes in the structural-form
    sampler).
    """
    h = np.maximum(y + alpha, 1e-3)
    return sample_polyagamma(h, psi, rng=rng)


def _sample_beta(
    beta_current: np.ndarray,
    Xtilde: np.ndarray,
    omega: np.ndarray,
    y: np.ndarray,
    alpha: float,
    priors: ReducedGibbsPriors,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Block 2: conjugate Gaussian draw for :math:`\beta`.

    Given :math:`\tilde X = A_\rho^{-1} X` and PG weights :math:`\omega`,
    the posterior is

    .. math::

        \beta \mid \cdot \sim N(m_\beta, \Sigma_\beta), \\
        \Sigma_\beta^{-1} = \tilde X^\top \Omega \tilde X + V_0^{-1}, \\
        m_\beta = \Sigma_\beta \bigl(\tilde X^\top (\kappa + \omega \log\alpha)
                                      + V_0^{-1} \mu_0\bigr),

    where :math:`\kappa = (y - \alpha)/2`.

    Parameters
    ----------
    beta_current : ndarray, shape (k,)
        Current draw (used only as a fallback if the linear solve fails).
    Xtilde : ndarray, shape (n, k)
        :math:`A_\rho^{-1} X` at the current :math:`\rho`.
    omega : ndarray, shape (n,)
        PG weights.
    y : ndarray, shape (n,)
        Integer responses.
    alpha : float
        NB dispersion.
    priors : ReducedGibbsPriors
    rng : numpy.random.Generator

    Returns
    -------
    beta_new : ndarray, shape (k,)
    """
    k = Xtilde.shape[1]
    kappa = 0.5 * (y - alpha)
    log_alpha = np.log(alpha)

    # Prior precision and mean
    beta_mu = priors.beta_mu
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = np.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / (np.asarray(beta_sigma, dtype=np.float64) ** 2)
    if np.isscalar(beta_mu):
        mu0 = np.full(k, float(beta_mu))
    else:
        mu0 = np.asarray(beta_mu, dtype=np.float64)
    V0_inv_mu0 = V0_inv_diag * mu0

    # Σ_β^{-1} = X̃ᵀ Ω X̃ + V₀⁻¹
    # rhs = X̃ᵀ (κ + ω log α) + V₀⁻¹ μ₀
    Xt_omega = Xtilde * omega[:, None]  # (n, k)
    Sigma_beta_inv = Xt_omega.T @ Xtilde
    # Add prior precision on the diagonal
    Sigma_beta_inv.flat[:: k + 1] += V0_inv_diag

    rhs = Xtilde.T @ (kappa + omega * log_alpha) + V0_inv_mu0

    # Posterior draw via Cholesky:
    #   Σ_β⁻¹ = L Lᵀ
    #   m_β   = Σ_β rhs       (solve L Lᵀ m = rhs)
    #   sample = m_β + L⁻ᵀ z  with z ~ N(0, I), since Cov(L⁻ᵀz) = (L Lᵀ)⁻¹ = Σ_β.
    from scipy.linalg import solve_triangular

    try:
        L = np.linalg.cholesky(Sigma_beta_inv)
    except np.linalg.LinAlgError:
        # First attempt failed — add a small ridge and retry.
        Sigma_beta_inv.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(Sigma_beta_inv)
        except np.linalg.LinAlgError:
            # Posterior precision is numerically singular (can happen when
            # omega collapses to near-zero for extreme alpha).  Reuse the
            # previous beta draw rather than crashing the chain.
            return beta_current

    w = solve_triangular(L, rhs, lower=True)
    m_beta = solve_triangular(L.T, w, lower=False)
    z = rng.standard_normal(k)
    delta = solve_triangular(L.T, z, lower=False)
    return m_beta + delta


def _prior_precision_and_mean(
    priors: ReducedGibbsPriors, k: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(V0_inv_diag, mu0, log_det_V0)`` for the β prior.

    ``log_det_V0`` is currently unused by the ρ slice (constant in
    :math:`\\rho`) but returned for completeness.
    """
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = np.full(k, 1.0 / (float(beta_sigma) ** 2))
        log_det_V0 = 2.0 * k * np.log(float(beta_sigma))
    else:
        sigma = np.asarray(beta_sigma, dtype=np.float64)
        V0_inv_diag = 1.0 / (sigma**2)
        log_det_V0 = 2.0 * float(np.sum(np.log(sigma)))
    beta_mu = priors.beta_mu
    if np.isscalar(beta_mu):
        mu0 = np.full(k, float(beta_mu))
    else:
        mu0 = np.asarray(beta_mu, dtype=np.float64)
    return V0_inv_diag, mu0, log_det_V0


def _rho_log_density_marginal(
    rho: float,
    omega: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    alpha: float,
    V0_inv_diag: np.ndarray,
    mu0: np.ndarray,
    rho_lower: float,
    rho_upper: float,
    *,
    basis: Optional[ReducedKrylovBasis] = None,
    krylov_dmax: float = _KRYLOV_DMAX_DEFAULT,
    cholmod_solver: Optional[_CholmodNormalEqSolver] = None,
) -> float:
    r"""β-marginalised conditional log-density for the ρ slice.

    When a :class:`ReducedKrylovBasis` is provided and
    :math:`|\rho - \rho_c| \leq \texttt{krylov_dmax}`, the expensive
    factorisation is replaced by a cheap Horner evaluation of the
    shift-invert Krylov polynomial.  Otherwise a fresh solve is
    performed — via CHOLMOD normal equations when
    ``cholmod_solver`` is provided, or ``splu`` as fallback.
    """
    if rho <= rho_lower or rho >= rho_upper:
        return -np.inf

    # --- Compute U = (I - ρW)^{-1} X ---
    use_basis = (
        basis is not None
        and basis.degree > 0
        and abs(rho - basis.rho_basis) <= krylov_dmax
    )
    if use_basis:
        drho = rho - basis.rho_basis
        U = _eval_U_from_basis(basis, drho)
    else:
        try:
            solver = _make_solver(rho, W_csc, n, cholmod_solver=cholmod_solver)
            U = solver.solve(X)  # (n, k)
        except (RuntimeError, ValueError):
            return -np.inf

    # --- Working response and residual ---
    log_alpha = np.log(alpha)
    kappa = 0.5 * (y - alpha)
    s = kappa / omega + log_alpha
    r = s - U @ mu0

    # M = V0^{-1} + U^T Ω U  (k x k)
    Uw = U * omega[:, None]
    M = U.T @ Uw
    k = M.shape[0]
    M.flat[:: k + 1] += V0_inv_diag

    v = Uw.T @ r  # = U^T Ω r

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        M.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return -np.inf

    from scipy.linalg import solve_triangular

    w = solve_triangular(L, v, lower=True)
    quad_pen = float(w @ w)
    rOr = float(np.dot(r, omega * r))
    log_det_M = 2.0 * float(np.sum(np.log(np.diag(L))))

    result = -0.5 * log_det_M - 0.5 * (rOr - quad_pen)
    # Guard against nan from numerical overflow in the Krylov path
    # or near-singular matrices.  Returning -inf causes the slice
    # sampler to reject the candidate and shrink the interval.
    if not np.isfinite(result):
        return -np.inf
    return result


def _sample_rho(
    state: ReducedGibbsState,
    cache: ReducedGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    priors: ReducedGibbsPriors,
    *,
    rng: np.random.Generator,
    sweep_idx: int,
    tune: int,
    basis: Optional[ReducedKrylovBasis] = None,
    cholmod_solver: Optional[_CholmodNormalEqSolver] = None,
) -> tuple[float, float]:
    """Block 3: 1-D adaptive slice on :math:`\\rho` with β marginalised.

    When ``basis`` is provided (``krylov_degree > 0``), the slice density
    is evaluated via the shift-invert Krylov polynomial instead of a
    fresh factorisation per candidate.  The basis is built once per sweep
    at the current ρ and reused for all candidates within ``krylov_dmax``.
    """
    n, k = X.shape
    rho_lower = cache.rho_lower
    rho_upper = cache.rho_upper
    V0_inv_diag, mu0, _ = _prior_precision_and_mean(priors, k)

    def log_density(rho: float) -> float:
        return _rho_log_density_marginal(
            rho=rho,
            omega=state.omega,
            y=y,
            X=X,
            W_csc=cache.W_csc,
            n=n,
            alpha=state.alpha,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            rho_lower=rho_lower,
            rho_upper=rho_upper,
            basis=basis,
            krylov_dmax=cache.krylov_dmax,
            cholmod_solver=cholmod_solver,
        )

    if cache.rho_adaptive_width and cache.rho_slice_width_state is not None:
        width_state = cache.rho_slice_width_state
        log_dens_x0 = log_density(state.rho)
        rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
            log_density=log_density,
            x0=state.rho,
            lower=rho_lower,
            upper=rho_upper,
            width_state=width_state,
            rng=rng,
            log_density_x0=log_dens_x0,
        )
        if sweep_idx < tune:
            update_slice_width(width_state, steps_left, steps_right)
    else:
        rho_new, log_density_new = slice_sample_1d(
            log_density=log_density,
            x0=state.rho,
            lower=rho_lower,
            upper=rho_upper,
            w=0.2,
            rng=rng,
        )
    return rho_new, log_density_new


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: ReducedGibbsPriors,
    cache: ReducedGibbsCache,
    init: ReducedGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
) -> dict[str, np.ndarray]:
    """Run one chain of the reduced-form SAR-NB PG-Gibbs sampler.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Integer responses.
    X : ndarray, shape (n, k)
        Design matrix (intercept column expected if desired).
    W_sparse : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardised spatial weights matrix.
    priors : ReducedGibbsPriors
        Prior hyperparameters.
    cache : ReducedGibbsCache
        Precomputed constants for the sweep (sparsity formats, slice
        width state, …).
    init : ReducedGibbsState
        Initial state.
    draws, tune : int
        Post-warmup draws and warmup sweeps respectively.
    thin : int, default 1
        Keep every ``thin``-th post-warmup draw.
    rng : numpy.random.Generator, optional
        Per-chain random state.
    chain_id : int, default 0
        Index used by ``progress_manager``.
    progress_manager : object, optional
        ``run_chains``-style progress callback.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, ``alpha``,
        ``log_lik`` (each indexed by post-warmup draw).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64)

    state = ReducedGibbsState(
        beta=np.asarray(init.beta, dtype=np.float64).copy(),
        rho=float(init.rho),
        alpha=float(init.alpha),
        omega=np.asarray(init.omega, dtype=np.float64).copy(),
    )

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    from ..negbin._core import GibbsState as _StructuralState

    # Whether to use Krylov acceleration (degree > 0) or exact solve per
    # candidate (degree == 0, legacy path).
    use_krylov = cache.krylov_degree > 0
    krylov_degree = cache.krylov_degree

    # Build the CHOLMOD normal-equations solver once per chain.
    # When CHOLMOD is available, this replaces all ``splu`` (UMFPACK)
    # calls with CHOLMOD on the SPD matrix A^T A, avoiding Apple
    # Accelerate BLAS deadlocks on macOS under concurrent access.
    # The CholmodFactor is created from the pattern matrix **here in
    # the worker**, not in the parent process, to avoid accumulating
    # CHOLMOD/BLAS state across many fit() calls.
    cholmod_solver: _CholmodNormalEqSolver | None = None
    if (
        cache.cholmod_pattern is not None
        and cache.W_sym is not None
        and cache.WtW is not None
    ):
        cholmod_factor = CholmodFactor(cache.cholmod_pattern)
        cholmod_solver = _CholmodNormalEqSolver(
            cholmod_factor=cholmod_factor,
            W_csc=cache.W_csc,
            W_sym=cache.W_sym,
            WtW=cache.WtW,
            n=n,
        )

    for i in range(total_iters):
        # --- Build Krylov basis at current ρ (or factorise for legacy) ---
        if use_krylov:
            try:
                basis = _build_krylov_basis(
                    state.rho,
                    X,
                    cache.W_csc,
                    n,
                    degree=krylov_degree,
                    cholmod_solver=cholmod_solver,
                )
            except (RuntimeError, ValueError):
                # CHOLMOD factorisation failed (e.g. A^T A not SPD for
                # extreme ρ).  Fall back to ρ = 0 (identity transform).
                state.rho = 0.0
                basis = _build_krylov_basis(
                    0.0,
                    X,
                    cache.W_csc,
                    n,
                    degree=krylov_degree,
                    cholmod_solver=cholmod_solver,
                )
            # η for the ω block: η = U(ρ_c) @ β via the basis
            eta = basis.V_stack[0] @ state.beta
        else:
            try:
                solver = _make_solver(
                    state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
                )
            except (RuntimeError, ValueError):
                state.rho = 0.0
                solver = _make_solver(
                    0.0, cache.W_csc, n, cholmod_solver=cholmod_solver
                )
            eta = solver.solve(X @ state.beta)
            basis = None

        psi = eta - np.log(state.alpha)

        # --- Block 1: ω | β, ρ, α, y ---
        state.omega = _sample_omega(y, state.alpha, psi, rng=rng)

        # --- Block 2: ρ | ω, α, y (β marginalised) ---
        state.rho, _ = _sample_rho(
            state=state,
            cache=cache,
            y=y,
            X=X,
            priors=priors,
            rng=rng,
            sweep_idx=i,
            tune=tune,
            basis=basis,
            cholmod_solver=cholmod_solver,
        )

        # --- Block 3: β | ρ_new, ω, α, y ---
        # Exact solve at the accepted ρ_new for an unbiased β draw.
        # If the solver fails (extreme ρ), fall back to ρ = 0.
        try:
            solver_new = _make_solver(
                state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
            )
        except (RuntimeError, ValueError):
            state.rho = 0.0
            solver_new = _make_solver(
                0.0, cache.W_csc, n, cholmod_solver=cholmod_solver
            )
        Xtilde = solver_new.solve(X)
        state.beta = _sample_beta(
            beta_current=state.beta,
            Xtilde=Xtilde,
            omega=state.omega,
            y=y,
            alpha=state.alpha,
            priors=priors,
            rng=rng,
        )

        # --- Block 4: α | y, η_new ---
        eta = Xtilde @ state.beta
        alpha_state = _StructuralState(
            eta=eta,
            beta=state.beta,
            sigma2=1.0,  # unused by _sample_alpha
            rho=state.rho,
            alpha=state.alpha,
            omega=state.omega,
        )
        state.alpha = _sample_alpha(alpha_state, y, priors, rng=rng)

        # --- Store post-warmup draw ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = state.rho
                beta_samples[idx] = state.beta
                alpha_samples[idx] = state.alpha
                log_lik_samples[idx] = _nb_loglik_pointwise(y, eta, state.alpha)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune, accept=None)

    return {
        "rho": rho_samples,
        "beta": beta_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
    }
