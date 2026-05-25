"""Gaussian spatial Gibbs sampler for SAR, SEM, SDM, SDEM models.

Implements a 3-block Gibbs sampler that exploits conditional conjugacy
in Gaussian spatial regression models:

1. β | ρ, σ², y  — conjugate normal (direct draw)
2. σ² | β, ρ, y  — conjugate inverse-gamma (direct draw)
3. ρ/λ | β, σ², y — 1-D slice or MALA (non-conjugate, scalar)

Only the spatial parameter (ρ or λ) is non-conjugate, and it is a
scalar — a 1-D slice or MALA update is trivial.  No NUTS adaptation,
no gradient through the full model graph, no banana geometry.

**SAR/SDM**: Uses a *collapsed* ρ log-density that integrates out β
and σ², giving better mixing.  The collapsed density only requires
log|I - ρW| (already available) and RSS(ρ) (a simple quadratic form).

**SEM/SDEM**: Uses an *un-collapsed* λ log-density conditional on
current β and σ².  Simpler to implement; mixing penalty is small for
a scalar parameter.

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.

LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial
Econometrics*. CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
)

# ---------------------------------------------------------------------------
# State and configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GaussianGibbsState:
    """Mutable state for the Gaussian spatial Gibbs sampler.

    Parameters
    ----------
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma2 : float
        Residual variance σ².
    rho : float
        Spatial autoregressive parameter (ρ for SAR/SDM, λ for SEM/SDEM).
    """

    beta: np.ndarray
    sigma2: float
    rho: float


@dataclass
class GaussianGibbsPriors:
    """Prior hyperparameters for Gaussian spatial Gibbs.

    Parameters
    ----------
    beta_mu : float or ndarray
        Prior mean for β.  Scalar is broadcast to all coefficients.
    beta_sigma : float or ndarray
        Prior standard deviation for β.  Scalar is broadcast.
    sigma2_alpha : float
        Shape hyperparameter of the ``InverseGamma(sigma2_alpha,
        sigma2_beta)`` prior on σ².  Matches the NUTS path exactly so
        that posteriors — and therefore LOO/WAIC — agree between the two
        samplers.  Conjugate with the Gaussian likelihood, so the σ²
        block is an exact closed-form draw (LeSage 2009 convention).
    sigma2_beta : float
        Scale (rate) hyperparameter of the InverseGamma prior on σ².
        Models typically resolve this to ``Var(y)`` at construction so
        the prior mean is scale-aware.
    rho_lower : float
        Lower bound for ρ/λ (from spectral stability).
    rho_upper : float
        Upper bound for ρ/λ (from spectral stability).
    """

    beta_mu: float | np.ndarray = 0.0
    beta_sigma: float | np.ndarray = 1e6
    sigma2_alpha: float = 2.0
    sigma2_beta: float = 1.0
    rho_lower: float = -0.999
    rho_upper: float = 0.999
    # Accepted for backward compatibility with callers that still pass
    # ``sigma_sigma=...`` (e.g. panel models).  Ignored by the sampler;
    # use ``sigma2_alpha`` / ``sigma2_beta`` instead.
    sigma_sigma: float = 10.0


@dataclass
class GaussianGibbsCache:
    """Precomputed data that doesn't change across Gibbs sweeps.

    Parameters
    ----------
    XtX : ndarray of shape (k, k)
        X^T X matrix.
    XtX_cho : tuple of (ndarray, bool)
        Cholesky factor of X^T X from ``scipy.linalg.cho_factor``.
        Used for solving linear systems and quadratic forms involving
        (X^T X)^{-1} without forming the explicit inverse.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable for arrays of rho values.
    rho_lower : float
        Lower bound for ρ/λ.
    rho_upper : float
        Upper bound for ρ/λ.
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed for SAR/SDM).
    W_sparse : csr_matrix or None
        Sparse W matrix (for SEM/SDEM residual filtering).
    """

    XtX: np.ndarray
    XtX_cho: tuple  # Cholesky factor from cho_factor(XtX)
    logdet_fn: Callable[[float], float]
    logdet_vec_fn: Callable
    rho_lower: float
    rho_upper: float
    model_type: str = "sar"
    Wy: np.ndarray | None = None
    W_sparse: sp.csr_matrix | None = None


# ---------------------------------------------------------------------------
# Block samplers
# ---------------------------------------------------------------------------


def _sample_beta_sar(
    rho: float,
    sigma2: float,
    y: np.ndarray,
    Wy: np.ndarray,
    X: np.ndarray,
    XtX: np.ndarray,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample β from conjugate normal posterior (SAR/SDM).

    Residuals: r = y - ρ W y.  The model becomes r = X β + ε,
    which is standard conjugate normal.

    Parameters
    ----------
    rho : float
        Current spatial autoregressive parameter.
    sigma2 : float
        Current residual variance.
    y : ndarray of shape (n,)
        Response vector.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    X : ndarray of shape (n, k)
        Design matrix.
    XtX : ndarray of shape (k, k)
        X^T X (precomputed).
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta : ndarray of shape (k,)
        New draw from the conditional posterior.
    """
    r = y - rho * Wy
    return _sample_beta_conjugate(r, X, XtX, sigma2, priors, rng)


def _sample_beta_sem(
    lam: float,
    sigma2: float,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample β from conjugate normal posterior (SEM/SDEM).

    For SEM, conditional on λ and σ², the model is:
        y* = X* β + ε,  ε ~ N(0, σ² I)
    where y* = (I - λW)y and X* = (I - λW)X.

    The posterior is:
        β | λ, σ², y ~ N(β̂, Σ_β)
        Σ_β = (X*^T X* / σ² + Λ₀⁻¹)⁻¹
        β̂ = Σ_β (X*^T y* / σ² + Λ₀⁻¹ μ₀)

    Parameters
    ----------
    lam : float
        Current spatial error parameter λ.
    sigma2 : float
        Current residual variance.
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Sparse spatial weights matrix.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta : ndarray of shape (k,)
        New draw from the conditional posterior.
    """
    # Transform y and X by (I - λW)
    y_star = y - lam * (W_sparse @ y)
    X_star = X - lam * (W_sparse @ X)
    XtX_star = X_star.T @ X_star
    return _sample_beta_conjugate(y_star, X_star, XtX_star, sigma2, priors, rng)


def _sample_beta_conjugate(
    r: np.ndarray,
    X: np.ndarray,
    XtX: np.ndarray,
    sigma2: float,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample β from conjugate normal posterior.

    Model: r = X β + ε,  ε ~ N(0, σ² I)
    Prior: β ~ N(μ₀, Λ₀)

    Posterior: β | · ~ N(β̂, Σ_β)
    where Σ_β = (X^T X / σ² + Λ₀⁻¹)⁻¹
          β̂ = Σ_β (X^T r / σ² + Λ₀⁻¹ μ₀)

    Parameters
    ----------
    r : ndarray of shape (n,)
        Response (or residual) vector.
    X : ndarray of shape (n, k)
        Design matrix.
    XtX : ndarray of shape (k, k)
        X^T X (precomputed).
    sigma2 : float
        Current residual variance.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta : ndarray of shape (k,)
        New draw from the conditional posterior.
    """
    k = X.shape[1]
    beta_sigma_arr = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=float), (k,))
    beta_mu_arr = np.broadcast_to(np.asarray(priors.beta_mu, dtype=float), (k,))
    prior_prec = np.diag(1.0 / beta_sigma_arr**2)
    prior_mean = np.array(beta_mu_arr, dtype=float)

    post_prec = XtX / sigma2 + prior_prec
    Xtr = X.T @ r
    rhs = Xtr / sigma2 + prior_prec @ prior_mean

    # Cholesky factorisation: post_prec = L Lᵀ (SPD, lower-triangular L)
    # post_mean = post_prec⁻¹ @ rhs via two triangular solves
    # β = post_mean + L⁻ᵀ z,  z ~ N(0, I)  avoids forming inv(post_prec)
    # Cov(L⁻ᵀ z) = L⁻ᵀ L⁻¹ = (L Lᵀ)⁻¹ = post_prec⁻¹  ✓
    # NB: must request lower=True; the scipy default returns the *upper*
    # Cholesky U (A = UᵀU), in which case solve_triangular(U, z, trans='T')
    # yields U⁻ᵀ z whose covariance is U⁻ᵀ U⁻¹ ≠ A⁻¹ — a silent bug.
    L, lower = cho_factor(post_prec, lower=True)
    post_mean = cho_solve((L, lower), rhs)
    z = rng.standard_normal(k)
    beta = post_mean + solve_triangular(L, z, lower=lower, trans="T")
    return beta


def _sample_sigma2(
    rho: float,
    beta: np.ndarray,
    y: np.ndarray,
    Wy: np.ndarray | None,
    W_sparse: sp.csr_matrix | None,
    X: np.ndarray,
    priors: GaussianGibbsPriors,
    model_type: str,
    rng: np.random.Generator,
) -> float:
    """Sample σ² from its conjugate Inverse-Gamma full conditional.

    With prior ``σ² ~ InverseGamma(α, β)`` and Gaussian likelihood the
    full conditional is

    .. math::

        \\sigma^2 \\mid \\beta, \\rho, y
            \\sim
            \\mathrm{InverseGamma}\\!\\left(\\alpha + \\tfrac{n}{2},\\;
                \\beta + \\tfrac{1}{2} \\lVert \\varepsilon \\rVert^2\\right),

    where the residual depends on the model:

    - SAR/SDM:   ε = y - ρ W y - X β
    - SEM/SDEM:  ε = (I - λ W)(y - X β)

    This is the standard LeSage (2009) / Anselin Bayesian-spatial Gibbs
    update.  The same prior is placed on σ² in the NUTS path so the two
    samplers target identical posteriors.

    Parameters
    ----------
    rho : float
        Current spatial parameter (ρ for SAR/SDM, λ for SEM/SDEM).
    beta : ndarray of shape (k,)
        Current regression coefficients.
    y : ndarray of shape (n,)
        Response vector.
    Wy : ndarray of shape (n,) or None
        W @ y (for SAR/SDM).
    W_sparse : csr_matrix or None
        Sparse W (for SEM/SDEM).
    X : ndarray of shape (n, k)
        Design matrix.
    priors : GaussianGibbsPriors
        Prior hyperparameters.  Uses ``sigma2_alpha`` (shape) and
        ``sigma2_beta`` (scale/rate) for the InverseGamma prior.
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    sigma2 : float
        Draw from the full conditional.
    """
    n = len(y)

    if model_type in ("sar", "sdm"):
        resid = y - rho * Wy - X @ beta
        ss = np.dot(resid, resid)
    else:  # sem, sdem
        resid_raw = y - X @ beta
        eps = resid_raw - rho * (W_sparse @ resid_raw)
        ss = np.dot(eps, eps)

    a_post = priors.sigma2_alpha + n / 2.0
    b_post = priors.sigma2_beta + ss / 2.0
    return 1.0 / rng.gamma(a_post, 1.0 / b_post)


# ---------------------------------------------------------------------------
# Collapsed ρ log-density (SAR/SDM)
# ---------------------------------------------------------------------------


def _sar_collapsed_log_density(
    rho: float,
    y: np.ndarray,
    Wy: np.ndarray,
    X: np.ndarray,
    XtX_cho: tuple,
    logdet_fn: Callable[[float], float],
    n: int,
    k: int,
) -> float:
    """Collapsed log p(ρ | y) for SAR/SDM Gaussian model.

    Integrates out β and σ² analytically.  The collapsed density is:

        log p(ρ | y) = log|I - ρW| - (n-k)/2 · log RSS(ρ) + const

    where RSS(ρ) = (y - ρWy)^T M_X (y - ρWy) and
    M_X = I - X(X^T X)^{-1} X^T.

    Uses the Woodbury form to avoid O(n²) M_X multiplication:

        r^T M_X r = r^T r - (X^T r)^T (X^T X)^{-1} (X^T r)

    which is O(nk + k²) instead of O(n²).

    Parameters
    ----------
    rho : float
        Spatial autoregressive parameter.
    y : ndarray of shape (n,)
        Response vector.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    X : ndarray of shape (n, k)
        Design matrix.
    XtX_cho : tuple of (ndarray, bool)
        Cholesky factor of X^T X from ``scipy.linalg.cho_factor``.
    logdet_fn : callable
        log|I - rho*W| callable.
    n : int
        Number of observations.
    k : int
        Number of regressors (including intercept).

    Returns
    -------
    log_density : float
        Collapsed log-density of ρ (up to a constant).
    """
    r = y - rho * Wy
    Xtr = X.T @ r  # (k,) — X^T r
    rss = np.dot(r, r) - Xtr @ cho_solve(XtX_cho, Xtr)
    logdet = logdet_fn(rho)
    return logdet - 0.5 * (n - k) * np.log(rss)


# ---------------------------------------------------------------------------
# Collapsed λ log-density (SEM/SDEM)
# ---------------------------------------------------------------------------


def _sem_collapsed_log_density(
    lam: float,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    logdet_fn: Callable[[float], float],
    n: int,
    k: int,
) -> float:
    """Collapsed log p(λ | y) for SEM/SDEM Gaussian model.

    Integrates out β and σ² analytically.  The collapsed density is:

        log p(λ | y) = log|I - λW|
                       - (1/2) log|X*^T X*|
                       - (n-k)/2 · log RSS(λ) + const

    where y* = (I - λW)y, X* = (I - λW)X, and
    RSS(λ) = y*^T y* - y*^T X* (X*^T X*)^{-1} X*^T y*.

    The extra term -(1/2) log|X*^T X*| appears because X* depends on λ
    (unlike SAR where X is fixed).

    Parameters
    ----------
    lam : float
        Spatial error parameter.
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Sparse spatial weights matrix.
    logdet_fn : callable
        log|I - lam*W| callable.
    n : int
        Number of observations.
    k : int
        Number of regressors (including intercept).

    Returns
    -------
    log_density : float
        Collapsed log-density of λ (up to a constant).
    """
    # Transform y and X by (I - λW)
    y_star = y - lam * (W_sparse @ y)
    X_star = X - lam * (W_sparse @ X)

    # Compute RSS(λ) using Woodbury form
    XtX_star = X_star.T @ X_star
    Xty_star = X_star.T @ y_star
    yty_star = np.dot(y_star, y_star)

    # RSS = y*^T y* - y*^T X* (X*^T X*)^{-1} X*^T y*
    # Use Cholesky for the SPD happy path; fall back to pinv for
    # rank-deficient X*^T X* (e.g. near-collinear WX columns).
    try:
        L, lower = cho_factor(XtX_star)
        sol = cho_solve((L, lower), Xty_star)
        rss = yty_star - Xty_star @ sol
    except np.linalg.LinAlgError:
        # Cholesky failed — matrix is not SPD, use pseudo-inverse
        XtX_star_inv = np.linalg.pinv(XtX_star)
        rss = yty_star - Xty_star @ XtX_star_inv @ Xty_star
    rss = max(rss, 1e-300)  # Prevent log(0)

    logdet = logdet_fn(lam)
    logdet_XtX = np.linalg.slogdet(XtX_star)[1]

    return logdet - 0.5 * logdet_XtX - 0.5 * (n - k) * np.log(rss)


# ---------------------------------------------------------------------------
# Un-collapsed λ log-density (SEM/SDEM) — kept for backward compatibility
# ---------------------------------------------------------------------------


def _sem_conditional_log_density(
    lam: float,
    beta: np.ndarray,
    sigma2: float,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    logdet_fn: Callable[[float], float],
) -> float:
    """Conditional log p(λ | β, σ², y) for SEM/SDEM Gaussian model.

    Uses the un-collapsed approach: condition on current β and σ².

        log p(λ | β, σ², y) = log|I - λW| - 1/(2σ²) ||ε||² + const

    where ε = (I - λW)(y - Xβ).

    Parameters
    ----------
    lam : float
        Spatial error parameter.
    beta : ndarray of shape (k,)
        Current regression coefficients.
    sigma2 : float
        Current residual variance.
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Sparse spatial weights matrix.
    logdet_fn : callable
        log|I - lam*W| callable.

    Returns
    -------
    log_density : float
        Conditional log-density of λ (up to a constant).
    """
    resid = y - X @ beta
    eps = resid - lam * (W_sparse @ resid)
    ss = np.dot(eps, eps)
    logdet = logdet_fn(lam)
    return logdet - 0.5 * ss / sigma2


# ---------------------------------------------------------------------------
# ρ/λ slice sampler
# ---------------------------------------------------------------------------


def _sample_rho_sar(
    state: GaussianGibbsState,
    cache: GaussianGibbsCache,
    priors: GaussianGibbsPriors,
    y: np.ndarray,
    Wy: np.ndarray,
    X: np.ndarray,
    n: int,
    k: int,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
    log_density_current: float | None = None,
) -> tuple[float, float]:
    """Slice sample ρ from the collapsed SAR/SDM log-density.

    Parameters
    ----------
    state : GaussianGibbsState
        Current Gibbs state.
    cache : GaussianGibbsCache
        Precomputed data.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    y : ndarray of shape (n,)
        Response vector.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    X : ndarray of shape (n, k)
        Design matrix.
    n : int
        Number of observations.
    k : int
        Number of regressors.
    rng : numpy.random.Generator
        Random state.
    slice_state : SliceWidthState
        Adaptive slice width state.
    log_density_current : float or None
        Cached log-density at current ρ (avoids recomputation).

    Returns
    -------
    rho_new : float
        New ρ draw.
    log_density_new : float
        Log-density at the new ρ (for caching).
    """

    def log_density(rho_val):
        return _sar_collapsed_log_density(
            rho_val,
            y,
            Wy,
            X,
            cache.XtX_cho,
            cache.logdet_fn,
            n,
            k,
        )

    rho_new, log_density_new, _, _ = slice_sample_1d_adaptive(
        log_density,
        state.rho,
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
        log_density_x0=log_density_current,
    )
    return rho_new, log_density_new


def _sample_lam_sem_collapsed(
    state: GaussianGibbsState,
    cache: GaussianGibbsCache,
    priors: GaussianGibbsPriors,
    y: np.ndarray,
    X: np.ndarray,
    n: int,
    k: int,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
    log_density_current: float | None = None,
) -> tuple[float, float]:
    """Slice sample λ from the collapsed SEM/SDEM log-density.

    Uses the collapsed density that integrates out β and σ² analytically,
    giving much better mixing than the un-collapsed conditional approach.

    Parameters
    ----------
    state : GaussianGibbsState
        Current Gibbs state.
    cache : GaussianGibbsCache
        Precomputed data.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    n : int
        Number of observations.
    k : int
        Number of regressors.
    rng : numpy.random.Generator
        Random state.
    slice_state : SliceWidthState
        Adaptive slice width state.
    log_density_current : float or None
        Cached log-density at current λ.

    Returns
    -------
    lam_new : float
        New λ draw.
    log_density_new : float
        Log-density at the new λ.
    """

    def log_density(lam_val):
        return _sem_collapsed_log_density(
            lam_val,
            y,
            X,
            cache.W_sparse,
            cache.logdet_fn,
            n,
            k,
        )

    lam_new, log_density_new, _, _ = slice_sample_1d_adaptive(
        log_density,
        state.rho,  # state.rho holds λ for SEM/SDEM
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
        log_density_x0=log_density_current,
    )
    return lam_new, log_density_new


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _initialize_gaussian_gibbs(
    y: np.ndarray,
    X: np.ndarray,
    XtX_cho: tuple,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
) -> GaussianGibbsState:
    """Warm-start the Gibbs sampler from an OLS fit.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    XtX_cho : tuple of (ndarray, bool)
        Cholesky factor of X^T X from ``scipy.linalg.cho_factor``.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    GaussianGibbsState
        Initial state with OLS-based starting values.
    """
    beta_ols = cho_solve(XtX_cho, X.T @ y)
    resid = y - X @ beta_ols
    sigma2_ols = np.dot(resid, resid) / len(y)
    # Start ρ/λ at 0 (no spatial effect)
    rho_init = 0.0

    return GaussianGibbsState(
        beta=beta_ols.copy(),
        sigma2=max(sigma2_ols, 1e-6),
        rho=rho_init,
    )


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_gaussian_chain(
    y: np.ndarray,
    X: np.ndarray,
    cache: GaussianGibbsCache,
    priors: GaussianGibbsPriors,
    init: GaussianGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    progressbar: bool = True,
    chain_id: int = 0,
    progress_manager: object | None = None,
) -> dict[str, np.ndarray]:
    """Run one chain of the Gaussian spatial Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    cache : GaussianGibbsCache
        Precomputed data.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    init : GaussianGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int, default 1
        Keep every thin-th draw after warmup.
    rng : numpy.random.Generator, optional
        Random state.
    progressbar : bool, default True
        Show progress bar for this chain.
    chain_id : int, default 0
        Chain index (0-based) for progress bar display.
    progress_manager : object or None
        ``GibbsProgressBarManager`` instance. If provided,
        ``update()`` is called after each iteration.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho`` (or ``lam``), ``beta``,
        ``sigma``, and ``log_lik``.  Each array has shape
        ``(n_keep, ...)`` where n_keep = draws // thin.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws
    model_type = cache.model_type

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64)

    # Copy initial state
    state = GaussianGibbsState(
        beta=init.beta.copy(),
        sigma2=init.sigma2,
        rho=init.rho,
    )

    # Adaptive slice width for ρ/λ
    rho_range = cache.rho_upper - cache.rho_lower
    slice_state = SliceWidthState(w=rho_range * 0.1)

    # Cached log-density for ρ/λ
    log_density_rho = None

    # Precompute Wy for SAR/SDM
    Wy = cache.Wy

    for i in range(total_iters):
        # --- Block 1: β | ρ, σ², y ---
        if model_type in ("sar", "sdm"):
            state.beta = _sample_beta_sar(
                state.rho,
                state.sigma2,
                y,
                Wy,
                X,
                cache.XtX,
                priors,
                rng,
            )
        else:  # sem, sdem
            state.beta = _sample_beta_sem(
                state.rho,
                state.sigma2,
                y,
                X,
                cache.W_sparse,
                priors,
                rng,
            )

        # --- Block 2: σ² | β, ρ/λ, y (conjugate Inv-Γ draw) ---
        state.sigma2 = _sample_sigma2(
            state.rho,
            state.beta,
            y,
            Wy,
            cache.W_sparse,
            X,
            priors,
            model_type,
            rng,
        )

        # --- Block 3: ρ/λ | β, σ², y (slice sampling) ---
        if model_type in ("sar", "sdm"):
            state.rho, log_density_rho = _sample_rho_sar(
                state,
                cache,
                priors,
                y,
                Wy,
                X,
                n,
                k,
                rng,
                slice_state,
                log_density_rho,
            )
        else:  # sem, sdem
            state.rho, log_density_rho = _sample_lam_sem_collapsed(
                state,
                cache,
                priors,
                y,
                X,
                n,
                k,
                rng,
                slice_state,
                log_density_rho,
            )

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            j = (i - tune) // thin
            rho_samples[j] = state.rho
            beta_samples[j] = state.beta
            sigma_samples[j] = np.sqrt(state.sigma2)

            # Pointwise log-likelihood (including Jacobian/n)
            sigma = np.sqrt(state.sigma2)
            if model_type in ("sar", "sdm"):
                mu = state.rho * Wy + X @ state.beta
                resid = y - mu
                ll = (
                    -0.5 * (resid / sigma) ** 2
                    - np.log(sigma)
                    - 0.5 * np.log(2.0 * np.pi)
                )
                jacobian = cache.logdet_fn(state.rho)
                ll += jacobian / n
            else:  # sem, sdem
                resid_raw = y - X @ state.beta
                eps = resid_raw - state.rho * (cache.W_sparse @ resid_raw)
                ll = (
                    -0.5 * (eps / sigma) ** 2
                    - np.log(sigma)
                    - 0.5 * np.log(2.0 * np.pi)
                )
                jacobian = cache.logdet_fn(state.rho)
                ll += jacobian / n
            # Clamp for numerical stability
            ll = np.where(np.isfinite(ll), ll, -1e10)
            log_lik_samples[j] = ll

        # Update progress bar
        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune, accept=None)

    # Name the spatial parameter appropriately
    param_name = "rho" if model_type in ("sar", "sdm") else "lam"
    result = {
        param_name: rho_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "log_lik": log_lik_samples,
    }

    return result
