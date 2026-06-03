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

import os
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from ..negbin._core import _sample_alpha, _nb_loglik_pointwise


# ---------------------------------------------------------------------------
# Dense / sparse LU dispatch
# ---------------------------------------------------------------------------

# Switch to dense LAPACK ``lu_factor``/``lu_solve`` only for very small ``n``.
# Empirically (rook-grid W, k=3) sparse SuperLU is already faster than dense
# LAPACK from n ≳ 400 because the SAR weights matrix is highly sparse and the
# fill-in is small.  We keep a low dense threshold as a guard for tiny
# problems where Python/SuperLU overhead dominates.  Override at import time
# via the environment variable ``BAYESPECON_NEGBIN_DENSE_MAX``.
_DEFAULT_DENSE_MAX = 300
try:
    DENSE_LU_MAX = int(os.environ.get("BAYESPECON_NEGBIN_DENSE_MAX", _DEFAULT_DENSE_MAX))
except ValueError:  # pragma: no cover
    DENSE_LU_MAX = _DEFAULT_DENSE_MAX


class _AFactor:
    """Opaque LU factorisation of :math:`A_\\rho = I - \\rho W`.

    Dispatches to dense LAPACK (:func:`scipy.linalg.lu_factor` /
    :func:`scipy.linalg.lu_solve`) for ``n <= DENSE_LU_MAX`` and to sparse
    ``splu`` otherwise.  Callers see a uniform :meth:`solve` interface.
    """

    __slots__ = ("_dense", "_lu", "_piv", "_splu")

    def __init__(self, rho: float, W_csc: sp.csc_matrix, n: int) -> None:
        if n <= DENSE_LU_MAX:
            # Build A densely and factor with LAPACK ``dgetrf``.
            A_dense = -rho * W_csc.toarray()
            # Add the identity in-place.
            A_dense.flat[:: n + 1] += 1.0
            self._lu, self._piv = la.lu_factor(A_dense, overwrite_a=True)
            self._dense = True
            self._splu = None
        else:
            A = (sp.eye(n, format="csc") - rho * W_csc).tocsc()
            self._splu = spla.splu(A)
            self._dense = False
            self._lu = None
            self._piv = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self._dense:
            return la.lu_solve((self._lu, self._piv), rhs, overwrite_b=False)
        return self._splu.solve(rhs)


def _factor_A(rho: float, W_csc: sp.csc_matrix, n: int) -> _AFactor:
    """Factorise :math:`A_\\rho = I - \\rho W` using dense or sparse LU."""
    return _AFactor(rho, W_csc, n)

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
        Same matrix in csc format (preferred for ``splu``).
    rho_lower, rho_upper : float
        Support bounds for the ρ slice sampler.
    rho_adaptive_width : bool
        Whether to tune the ρ slice-sampler width during warmup.
    rho_slice_width_state : SliceWidthState
        Mutable width state for the adaptive ρ slice sampler.
    """

    W_sparse: sp.csr_matrix
    W_csc: sp.csc_matrix
    rho_lower: float
    rho_upper: float
    rho_adaptive_width: bool = True
    rho_slice_width_state: Optional[SliceWidthState] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ``_factor_A`` is defined above (alongside the dense/sparse dispatch).


def _compute_eta(
    rho: float,
    Xbeta: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
) -> tuple[np.ndarray, spla.SuperLU]:
    """Return :math:`\\eta = (I - \\rho W)^{-1} X\\beta` and the LU factor.

    The factor is returned so callers can reuse it (e.g. to compute
    :math:`\\tilde X = A^{-1} X` without re-factorising).
    """
    lu = _factor_A(rho, W_csc, n)
    eta = lu.solve(Xbeta)
    return eta, lu


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
        Sigma_beta_inv.flat[:: k + 1] += 1e-10
        L = np.linalg.cholesky(Sigma_beta_inv)

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
    factor_cache: Optional[dict] = None,
) -> float:
    r"""β-marginalised conditional log-density for the ρ slice.

    See the original derivation in the module docstring.  When
    ``factor_cache`` is given, the most recently computed ``(rho, lu, U)``
    triple is stashed there so that the runner can reuse it for the
    subsequent β draw at the accepted ρ.
    """
    if rho <= rho_lower or rho >= rho_upper:
        return -np.inf
    try:
        lu = _factor_A(rho, W_csc, n)
        U = lu.solve(X)  # (n, k)
    except RuntimeError:
        return -np.inf

    if factor_cache is not None:
        factor_cache["rho"] = rho
        factor_cache["lu"] = lu
        factor_cache["U"] = U

    # Working response and residual
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

    return -0.5 * log_det_M - 0.5 * (rOr - quad_pen)


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
    factor_cache: Optional[dict] = None,
) -> tuple[float, float]:
    """Block 3: 1-D adaptive slice on :math:`\\rho` with β marginalised.

    If ``factor_cache`` is supplied, the LU factor and ``U = A_ρ⁻¹X``
    from the most recent density evaluation are stashed there for the
    subsequent β draw to reuse, saving one LU factorisation per sweep.
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
            factor_cache=factor_cache,
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

    for i in range(total_iters):
        # --- Compute η at current ρ for the ω block ---
        try:
            lu = _factor_A(state.rho, cache.W_csc, n)
        except RuntimeError:
            # Singular A_ρ — should be impossible inside stability
            # bounds, but degrade gracefully by perturbing ρ.
            state.rho = 0.0
            lu = _factor_A(0.0, cache.W_csc, n)
        eta = lu.solve(X @ state.beta)
        psi = eta - np.log(state.alpha)

        # --- Block 1: ω | β, ρ, α, y ---
        state.omega = _sample_omega(y, state.alpha, psi, rng=rng)

        # --- Block 2: ρ | ω, α, y (β marginalised) ---
        factor_cache: dict = {}
        state.rho, _ = _sample_rho(
            state=state,
            cache=cache,
            y=y,
            X=X,
            priors=priors,
            rng=rng,
            sweep_idx=i,
            tune=tune,
            factor_cache=factor_cache,
        )

        # --- Block 3: β | ρ_new, ω, α, y ---
        # Reuse the LU + U=A_ρ⁻¹X from the slice's last density evaluation
        # if it was at the accepted ρ.  Slice sampling typically ends at
        # the accepted point, so this hit-rate is near 1.
        if factor_cache.get("rho") == state.rho and factor_cache.get("U") is not None:
            lu = factor_cache["lu"]
            Xtilde = factor_cache["U"]
        else:
            lu = _factor_A(state.rho, cache.W_csc, n)
            Xtilde = lu.solve(X)
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
