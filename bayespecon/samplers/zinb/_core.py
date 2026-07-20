"""Pólya–Gamma Gibbs sampler for the zero-inflated SAR-NB model.

Orchestrates the 8-block Gibbs sweep:

  Selection equation (structural-form SAR-logit):
    1. ω^sel | η^sel, d       (PG augmentation, h = 1)
    2. η^sel | ω^sel, λ, γ    (spatial-normal draw, σ² = 1)
    3. γ     | η^sel, λ       (conjugate normal)
    4. λ     | γ, ω^sel, d    (collapsed 1-D slice, η^sel integrated out)

  Zero allocation:
    5. z_i | y_i, η^cnt_i, η^sel_i, α   (Bernoulli draw for y_i = 0)

  Count equation (reduced-form SAR-NB):
    6. ω^cnt | η^cnt, α, y, z  (PG augmentation, z-masked)
    7. β     | ω^cnt, ρ, α, y, z  (conjugate normal via X̃ = A_ρ^{-1} X)
    8. ρ     | ω^cnt, α, y, z  (β-marginalised 1-D slice, z-masked)
    9. α     | y, η^cnt, z     (1-D slice on log(α))

The selection equation is the structural-form SAR-logit from
:mod:`bayespecon.samplers.logit._core`. The count equation is the
reduced-form SAR-NB from :mod:`bayespecon.samplers.negbin_reduced._core`.
The zero-allocation block (5) is the only genuinely new algorithmic
content.

References
----------
Polson, N.G., Scott, J.G. & Windle, J. (2013). Bayesian inference
for logistic models using Pólya-Gamma latent variables.
*JASA* 108(504), 1339–1349.

Lambert, D. (1992). Zero-inflated Poisson regression, with an
application to defects in manufacturing. *Technometrics* 34(1), 1–14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from ...models.priors import ZINBGibbsPriors
from .._utils._polyagamma import sample_polyagamma
from .._utils._spatial_normal import CholmodFactor
from ..logit._core import (
    LogitGibbsCache,
    LogitGibbsPriors,
    LogitGibbsState,
)
from ..logit._core import (
    _sample_beta as _sample_gamma_sel,
)
from ..logit._core import (
    _sample_eta as _sample_eta_sel,
)
from ..logit._core import (
    _sample_omega as _sample_omega_sel,
)
from ..logit._core import (
    _sample_rho as _sample_lam,
)
from ..negbin._core import GibbsPriors as _NBGibbsPriors
from ..negbin._core import GibbsState as _NBGibbsState
from ..negbin._core import _nb_loglik_pointwise, _sample_alpha
from ..negbin_reduced._core import (
    ReducedGibbsCache,
    ReducedGibbsPriors,
    ReducedGibbsState,
    _CholmodNormalEqSolver,
    _make_solver,
)
from ..negbin_reduced._core import (
    _sample_beta as _sample_beta_cnt,
)
from ..negbin_reduced._core import (
    _sample_rho as _sample_rho_cnt,
)

# ---------------------------------------------------------------------------
# expit helper (avoid scipy dependency)
# ---------------------------------------------------------------------------


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# Data classes for state, priors, and precomputed cache
# ---------------------------------------------------------------------------


@dataclass
class ZINBGibbsState:
    """Mutable state carried through one ZINB Gibbs sweep.

    Parameters
    ----------
    eta_sel : ndarray of shape (n,)
        Latent log-odds for the selection equation.
    gamma : ndarray of shape (p,)
        Selection equation coefficients.
    lam : float
        Selection SAR parameter.
    omega_sel : ndarray of shape (n,)
        PG auxiliary variables for the logit selection.
    beta : ndarray of shape (k,)
        Count equation coefficients.
    rho : float
        Count SAR parameter.
    alpha : float
        NB dispersion parameter.
    omega_cnt : ndarray of shape (n,)
        PG auxiliary variables for the NB count (zero where z=0).
    z : ndarray of shape (n,)
        Zero-allocation indicators. z_i = 1 means observation i
        contributes to the count equation.
    """

    eta_sel: np.ndarray
    gamma: np.ndarray
    lam: float
    omega_sel: np.ndarray
    beta: np.ndarray
    rho: float
    alpha: float
    omega_cnt: np.ndarray
    z: np.ndarray


class ZINBGibbsCache(NamedTuple):
    """Precomputed data that doesn't change across sweeps."""

    # Selection equation cache (mirrors LogitGibbsCache)
    sel_cache: LogitGibbsCache
    # Count equation cache (mirrors ReducedGibbsCache)
    cnt_cache: ReducedGibbsCache
    # Data
    y: np.ndarray  # (n,) observed counts
    d: np.ndarray  # (n,) binary activity indicators
    Z: np.ndarray  # (n, p) selection covariates
    X: np.ndarray  # (n, k) count covariates
    # Weights
    W_sel_sparse: sp.csr_matrix
    W_cnt_sparse: sp.csr_matrix
    same_W: bool


# ---------------------------------------------------------------------------
# Block 5: Zero allocation
# ---------------------------------------------------------------------------


def _sample_z(
    y: np.ndarray,
    eta_cnt: np.ndarray,
    eta_sel: np.ndarray,
    alpha: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw latent zero-allocation indicators z_i.

    For y_i > 0: z_i = 1 (only count process generates positives).
    For y_i = 0:
      π_i = expit(η^sel_i)     (corridor activation probability)
      p_nb_zero = (α/(exp(η^cnt_i)+α))^α  (NB probability of zero)
      P(z=1 | y=0) = π_i · p_nb_zero / (π_i · p_nb_zero + (1 - π_i))

    Parameters
    ----------
    y : ndarray of shape (n,)
        Observed counts.
    eta_cnt : ndarray of shape (n,)
        Count log-intensity.
    eta_sel : ndarray of shape (n,)
        Selection log-odds.
    alpha : float
        NB dispersion.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    z : ndarray of shape (n,)
        Binary indicators. z_i = 1 means observation i contributes
        to the count equation.
    """
    z = np.ones(len(y), dtype=np.int8)
    zero_mask = y == 0
    if not np.any(zero_mask):
        return z

    pi = _expit(eta_sel[zero_mask])

    # Numerically stable NB zero probability
    log_p_nb_zero = alpha * np.log(alpha / (np.exp(eta_cnt[zero_mask]) + alpha))
    p_nb_zero = np.exp(np.clip(log_p_nb_zero, -700, 0))

    # Posterior probability z=1 given y=0
    numerator = pi * p_nb_zero
    denominator = numerator + (1.0 - pi)
    # Avoid division by zero (when both numerator and denominator → 0)
    p_z1 = np.where(denominator > 0, numerator / denominator, 0.0)
    p_z1 = np.clip(p_z1, 0.0, 1.0)

    z[zero_mask] = rng.binomial(1, p_z1).astype(np.int8)
    return z


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_zinb_chain(
    y: np.ndarray,
    d: np.ndarray,
    Z: np.ndarray,
    X: np.ndarray,
    W_sel_sparse: sp.csr_matrix,
    W_cnt_sparse: sp.csr_matrix,
    priors: ZINBGibbsPriors,
    cache: ZINBGibbsCache,
    init: ZINBGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
) -> dict[str, np.ndarray]:
    """Run one chain of the ZINB SAR Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Observed counts (non-negative integers).
    d : ndarray of shape (n,)
        Binary activity indicators (0/1).
    Z : ndarray of shape (n, p)
        Selection covariate matrix.
    X : ndarray of shape (n, k)
        Count covariate matrix.
    W_sel_sparse : csr_matrix of shape (n, n)
        Spatial weights for the selection equation.
    W_cnt_sparse : csr_matrix of shape (n, n)
        Spatial weights for the count equation.
    priors : ZINBGibbsPriors
        Prior hyperparameters.
    cache : ZINBGibbsCache
        Precomputed constants.
    init : ZINBGibbsState
        Initial state.
    draws, tune : int
        Post-warmup draws and warmup sweeps.
    thin : int, default 1
        Keep every ``thin``-th post-warmup draw.
    rng : numpy.random.Generator, optional
        Per-chain random state.
    chain_id : int, default 0
        Index used by progress manager.
    progress_manager : object, optional
        Progress callback.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``lam``, ``gamma``, ``rho``,
        ``beta``, ``alpha``, ``log_lik``, ``pi_mean``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y)
    p = Z.shape[1]
    k = X.shape[1]
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    # Pre-allocate storage
    lam_samples = np.empty(n_keep, dtype=np.float64)
    gamma_samples = np.empty((n_keep, p), dtype=np.float64)
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64)
    pi_mean_samples = np.empty(n_keep, dtype=np.float64)

    # Build sub-priors for the logit and NB blocks
    sel_priors = LogitGibbsPriors(
        beta_mu=priors.gamma_mu,
        beta_sigma=priors.gamma_sigma,
        rho_lower=priors.lam_lower,
        rho_upper=priors.lam_upper,
    )
    cnt_priors = ReducedGibbsPriors(
        beta_mu=priors.beta_mu,
        beta_sigma=priors.beta_sigma,
        rho_lower=priors.rho_lower,
        rho_upper=priors.rho_upper,
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
    )

    # Copy initial state
    state = ZINBGibbsState(
        eta_sel=init.eta_sel.copy(),
        gamma=init.gamma.copy(),
        lam=float(init.lam),
        omega_sel=init.omega_sel.copy(),
        beta=init.beta.copy(),
        rho=float(init.rho),
        alpha=float(init.alpha),
        omega_cnt=init.omega_cnt.copy(),
        z=init.z.copy(),
    )

    # Build logit sub-state for calling logit blocks
    sel_state = LogitGibbsState(
        eta=state.eta_sel,
        beta=state.gamma,
        rho=state.lam,
        omega=state.omega_sel,
    )

    # Build reduced-form NB sub-state for calling NB blocks
    cnt_state = ReducedGibbsState(
        beta=state.beta,
        rho=state.rho,
        alpha=state.alpha,
        omega=state.omega_cnt,
    )

    sel_cache = cache.sel_cache
    cnt_cache = cache.cnt_cache

    # CHOLMOD solver for the count equation (built once per chain)
    cholmod_solver = None
    if (
        cnt_cache.cholmod_pattern is not None
        and cnt_cache.W_sym is not None
        and cnt_cache.WtW is not None
    ):
        cholmod_factor = CholmodFactor(cnt_cache.cholmod_pattern)
        cholmod_solver = _CholmodNormalEqSolver(
            cholmod_factor=cholmod_factor,
            W_csc=cnt_cache.W_csc,
            W_sym=cnt_cache.W_sym,
            WtW=cnt_cache.WtW,
            n=n,
        )

    # Detect intercept column in X for reparameterisation
    intercept_col = -1
    for _j in range(k):
        if np.all(X[:, _j] == 1.0):
            intercept_col = _j
            break

    for i in range(total_iters):
        # ── SELECTION EQUATION (4 blocks) ──────────────────────────────
        # CRITICAL: The selection equation models the *latent* corridor
        # activity z, NOT the observed d = 1(y > 0).  Using d conflates
        # structural zeros with NB sampling zeros, biasing γ and λ.

        # Block 1: ω^sel | η^sel, z   (z = latent allocation, not d)
        state.omega_sel = _sample_omega_sel(state.eta_sel, rng=rng)
        sel_state.omega = state.omega_sel

        # Block 2: η^sel | ω^sel, λ, γ, z
        state.eta_sel, _ = _sample_eta_sel(
            sel_state,
            state.z.astype(np.float64),
            Z,
            W_sel_sparse,
            rng=rng,
            cache=sel_cache,
        )
        sel_state.eta = state.eta_sel

        # Recompute A_λ η^sel for the γ draw (no matrix build)
        A_lam_eta_sel = state.eta_sel - state.lam * (W_sel_sparse @ state.eta_sel)

        # Block 3: γ | η^sel, λ
        ZtZ = sel_cache.XtX  # reused: Z^T Z stored as XtX in logit cache
        state.gamma = _sample_gamma_sel(
            sel_state,
            Z,
            ZtZ,
            sel_priors,
            A_lam_eta_sel,
            rng=rng,
        )
        sel_state.beta = state.gamma

        # Block 4: λ | γ, ω^sel, z
        state.lam, _ = _sample_lam(
            sel_state,
            sel_cache,
            sel_priors,
            state.z.astype(np.float64),
            Z,
            rng=rng,
            log_density_current=None,
            sweep_idx=i,
            tune=tune,
        )
        sel_state.rho = state.lam

        # ── ZERO ALLOCATION (1 block) ─────────────────────────────────

        # Compute η^cnt = (I - ρ W_cnt)^{-1} X β for the z draw
        try:
            solver = _make_solver(
                state.rho,
                cnt_cache.W_csc,
                n,
                cholmod_solver=cholmod_solver,
            )
            eta_cnt = solver.solve(X @ state.beta)
        except (RuntimeError, ValueError):
            eta_cnt = X @ state.beta  # fallback: no spatial structure

        # Block 5: z | y, η^cnt, η^sel, α
        state.z = _sample_z(y, eta_cnt, state.eta_sel, state.alpha, rng=rng)

        # ── COUNT EQUATION (4 blocks, z-masked) ───────────────────────
        # z=0 observations must contribute EXACTLY zero to all NB
        # parameter updates.  The previous masking strategy (omega=1e-10,
        # y=alpha) leaked nonzero contributions into the NB log-likelihood
        # and the rho/beta conditionals.
        #
        # Strategy: set omega=0 exactly for z=0 observations.  The NB
        # sampler divides by omega in `s = kappa/omega + log_alpha`, but
        # when omega=0 we also have kappa=0 (since y=0 and alpha>0 for
        # z=0 structural zeros), so the limit is 0.  We handle this by
        # clamping s=0 wherever omega=0.

        # Block 6: ω^cnt | η^cnt, α, y, z
        # Draw PG only for z=1 observations.
        # For z=0 observations, set omega to a tiny value (1e-300)
        # instead of exactly 0, to avoid 0/0 = nan in the NB sampler's
        # internal `s = kappa/omega + log_alpha` computation.  With
        # kappa=0 (from y=alpha in blocks 7-8) and omega=1e-300,
        # s = 0/1e-300 + log_alpha = log_alpha, and the contribution
        # to the quadratic form is omega * r^2 ≈ 0.
        _Z_MASK_EPS = 1e-300
        z1 = state.z == 1
        psi = eta_cnt - np.log(state.alpha)
        h = np.maximum(y + state.alpha, 1e-3)
        psi_clamped = np.clip(psi, -20.0, 20.0)
        omega_full = np.full(n, _Z_MASK_EPS, dtype=np.float64)
        if np.any(z1):
            omega_full[z1] = sample_polyagamma(h[z1], psi_clamped[z1], rng=rng)
        state.omega_cnt = omega_full
        cnt_state.omega = state.omega_cnt

        # Block 7: β | ω^cnt, ρ, α, y, z
        # Build X̃ = A_ρ^{-1} X
        try:
            solver = _make_solver(
                state.rho,
                cnt_cache.W_csc,
                n,
                cholmod_solver=cholmod_solver,
            )
            Xtilde = solver.solve(X)
        except (RuntimeError, ValueError):
            Xtilde = X  # fallback

        # For z=0 observations: omega=0 zeros out the precision contribution,
        # but kappa = 0.5*(y - alpha) is still nonzero (y=0 → kappa=-alpha/2).
        # We must also zero out kappa by setting y=alpha for z=0 obs.
        # This makes the RHS contribution from z=0 obs exactly zero:
        #   kappa + omega*log_alpha = 0 + 0 = 0
        y_for_beta = np.where(z1, y, state.alpha)
        state.beta = _sample_beta_cnt(
            state.beta,
            Xtilde,
            state.omega_cnt,
            y_for_beta,
            state.alpha,
            cnt_priors,
            rng=rng,
            rho=state.rho,
            intercept_col=intercept_col,
        )
        cnt_state.beta = state.beta

        # Block 8: ρ | ω^cnt, α, y, z
        # Pass y_for_rho where z=0 obs have y=alpha (kappa=0).
        # When omega=0 and kappa=0, s = 0/0 + log_alpha = nan.
        # We handle this by passing y=alpha for z=0 obs AND by
        # patching the rho log-density to handle nan in s.
        # Since omega=0 for z=0 obs, they contribute omega*r=0 to
        # the quadratic form regardless of s, but nan propagates.
        y_for_rho = np.where(z1, y, state.alpha)
        state.rho, _ = _sample_rho_cnt(
            cnt_state,
            cnt_cache,
            y_for_rho,
            X,
            cnt_priors,
            rng=rng,
            sweep_idx=i,
            tune=tune,
            cholmod_solver=cholmod_solver,
            intercept_col=intercept_col,
        )
        cnt_state.rho = state.rho

        # Block 9: α | y, η^cnt, z
        # Only z=1 observations contribute to the NB log-likelihood
        # for the alpha draw.  We pass only the z=1 subset to avoid
        # z=0 observations (where y=0) contaminating the NB ll.
        if np.any(z1):
            alpha_state = _NBGibbsState(
                eta=eta_cnt[z1],
                beta=state.beta,
                sigma2=1.0,  # unused by _sample_alpha
                rho=state.rho,
                alpha=state.alpha,
                omega=state.omega_cnt[z1],
            )
            alpha_priors = _NBGibbsPriors(
                alpha_sigma=priors.alpha_sigma,
                alpha_nu=priors.alpha_nu,
            )
            state.alpha = _sample_alpha(alpha_state, y[z1], alpha_priors, rng=rng)
        # else: all z=0 → no count data; keep current alpha
        cnt_state.alpha = state.alpha

        # ── Store post-warmup draws ───────────────────────────────────
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                lam_samples[idx] = state.lam
                gamma_samples[idx] = state.gamma
                rho_samples[idx] = state.rho
                beta_samples[idx] = state.beta
                alpha_samples[idx] = state.alpha

                # Log-likelihood: NB for z=1, logit for all
                ll_nb = _nb_loglik_pointwise(y, eta_cnt, state.alpha)
                ll_nb = np.where(state.z == 1, ll_nb, 0.0)
                # Logit log-lik for the selection equation
                from ..logit._core import _logit_loglik_pointwise

                ll_sel = _logit_loglik_pointwise(d, state.eta_sel)
                log_lik_samples[idx] = ll_nb + ll_sel

                # Posterior mean corridor activation probability
                pi_mean_samples[idx] = float(np.mean(_expit(state.eta_sel)))

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune, accept=None)

    return {
        "lam": lam_samples,
        "gamma": gamma_samples,
        "rho": rho_samples,
        "beta": beta_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
        "pi_mean": pi_mean_samples,
    }
