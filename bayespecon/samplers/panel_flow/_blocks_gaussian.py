r"""Block samplers for the Gaussian panel flow Gibbs sampler.

Seven Gibbs blocks for the separable SAR panel flow model with
eigenbasis FFBS. All blocks follow the pattern:
``take (state, cache, rng) → return new value``.

Blocks 1–5 are exact draws (conjugate). Blocks 6–7 are collapsed
1-D slice samplers that use the marginal Kalman filter log-likelihood
(integrating over :math:`\eta`) for improved mixing.

See Also
--------
bayespecon.samplers.panel_flow._eigenbasis
    Scalar Kalman filter, FFBS, and eigenbasis transforms.
bayespecon.samplers.panel_flow._state
    PanelGaussianState, PanelGaussianCache, PanelGaussianPriors.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm as _truncnorm

from .._utils._slice import SliceWidthState, slice_sample_1d_adaptive
from ._eigenbasis import (
    ffbs_backward_pass,
    kf_forward_pass,
    kf_log_likelihood,
    transform_from_eigenbasis,
    transform_to_eigenbasis,
)
from ._state import PanelGaussianCache, PanelGaussianState

# ---------------------------------------------------------------------------
# Block 1: η_{1:T} via FFBS
# ---------------------------------------------------------------------------


def _sample_eta_panel(
    y: np.ndarray,
    X: np.ndarray,
    state: PanelGaussianState,
    cache: PanelGaussianCache,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Joint draw of :math:`\eta_{1:T}` via Forward Filtering Backward Sampling.

    Algorithm
    ---------
    1. Compute residuals: :math:`r_t = y_t - X_t \beta`
    2. Transform to eigenbasis: :math:`\tilde y_t = (V \otimes V)^\top r_t`
    3. Forward pass: scalar Kalman filter for all :math:`n^2` modes
    4. Backward pass: FFBS simulation smoother
    5. Transform back: :math:`\eta_t = (V \otimes V) \tilde\eta_t + X_t \beta`

    Parameters
    ----------
    y : ndarray of shape (n², T)
        Observed flows in column-major per period.
    X : ndarray of shape (n², T, k) or (n², k)
        Covariate array.
    state : PanelGaussianState
        Current Gibbs state.
    cache : PanelGaussianCache
        Precomputed constants.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    eta : ndarray of shape (n², T)
        Posterior draw of the latent field in the spatial basis.
    ytilde : ndarray of shape (n², T)
        Transformed observations (reused by the :math:`\rho` slice samplers).
    """
    # 1. Residuals
    if cache.time_invariant_X:
        Xbeta = X @ state.beta  # (n²,)
        r = y - Xbeta[:, np.newaxis]  # (n², T) - (n², 1) → (n², T)
    else:
        r = y - np.einsum("ijk,k->ij", X, state.beta)  # (n², T)

    # 2. Transform to eigenbasis
    ytilde = transform_to_eigenbasis(r, cache.V, cache.VkronV)

    # 3. Modal variances
    q_modes = state.get_modal_variances(cache.eigs_W)

    # 4. Forward pass
    kf_out = kf_forward_pass(ytilde, q_modes, state.gamma, state.sigma2_y)

    # 5. Backward pass
    eta_tilde = ffbs_backward_pass(kf_out, state.gamma, q_modes, rng)

    # 6. Transform back to spatial basis
    eta = transform_from_eigenbasis(eta_tilde, cache.V, cache.VkronV)

    # 7. Add back Xβ
    if cache.time_invariant_X:
        eta = eta + Xbeta[:, np.newaxis]
    else:
        eta = eta + np.einsum("ijk,k->ij", X, state.beta)

    return eta, ytilde


# ---------------------------------------------------------------------------
# Block 2: β (conjugate Gaussian)
# ---------------------------------------------------------------------------


def _sample_beta_panel(
    eta: np.ndarray,
    X: np.ndarray,
    state: PanelGaussianState,
    cache: PanelGaussianCache,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Conjugate Gaussian draw for :math:`\beta`.

    Regresses :math:`\eta_t` on :math:`X_t` across all :math:`T` periods:

    .. math::

        \eta_t = X_t \beta + \xi_t, \quad \xi_t \sim N(0, \sigma^2_u I)

    This is consistent with the FFBS block which draws
    :math:`\eta_t = X_t\beta + (V \otimes V)\tilde\xi_t` where
    :math:`\tilde\xi_t` follows a scalar AR(1) in the eigenbasis.
    The variance :math:`\sigma^2_u I` is an approximation that is
    exact when :math:`\rho_d = \rho_o = 0`.

    Posterior:

    .. math::

        V_1^{-1} = V_\beta^{-1} + \frac{1}{\sigma^2_u} \sum_t X_t^\top X_t

        m_1 = V_1 \left( V_\beta^{-1} \beta_0
               + \frac{1}{\sigma^2_u} \sum_t X_t^\top \eta_t \right)

    Parameters
    ----------
    eta : ndarray of shape (n², T)
        Latent field draws.
    X : ndarray of shape (n², T, k) or (n², k)
        Covariate array.
    state : PanelGaussianState
        Current Gibbs state.
    cache : PanelGaussianCache
        Precomputed constants.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta : ndarray of shape (k,)
        New draw of regression coefficients.
    """
    sigma2_u = state.sigma2_u

    # Accumulate sufficient statistics: Σ_t X_t'η_t
    if cache.time_invariant_X and cache.XtX is not None:
        XtX_sum = cache.T * cache.XtX
        XtEta = X.T @ eta.sum(axis=1)  # (k, n²) @ (n²,) → (k,)
    else:
        XtX_sum = np.zeros((len(state.beta), len(state.beta)))
        XtEta = np.zeros(len(state.beta))
        for t in range(cache.T):
            Xt_slice = X[:, t, :]  # (n², k)
            XtX_sum += Xt_slice.T @ Xt_slice
            XtEta += Xt_slice.T @ eta[:, t]

    # Posterior precision and mean
    prec_post = cache.beta_prior_prec + XtX_sum / sigma2_u
    cov_post = np.linalg.inv(prec_post)
    mean_post = cov_post @ (
        cache.beta_prior_prec @ cache.beta_prior_mean + XtEta / sigma2_u
    )

    # Draw
    beta_new = rng.multivariate_normal(mean_post, cov_post)
    return beta_new


# ---------------------------------------------------------------------------
# Block 3: σ²_u (conjugate Inverse-Gamma)
# ---------------------------------------------------------------------------


def _sample_sigma2_u(
    eta: np.ndarray,
    X: np.ndarray,
    state: PanelGaussianState,
    cache: PanelGaussianCache,
    rng: np.random.Generator,
) -> float:
    r"""Conjugate Inverse-Gamma draw for :math:`\sigma^2_u`.

    Uses eigenbasis innovations for consistency with the FFBS block.
    The latent field in the eigenbasis is
    :math:`\tilde\xi_t = (V \otimes V)^\top (\eta_t - X_t\beta)`,
    and the innovation is
    :math:`\tilde\omega_t = \tilde\xi_t - \gamma \tilde\xi_{t-1}`
    with :math:`\text{Var}(\tilde\omega_{m,t}) = q_m = \sigma^2_u / g_m^2`
    where :math:`g_m = (1-\rho_d\lambda_i)(1-\rho_o\lambda_j)`.

    The SSR is computed in the eigenbasis, weighted by :math:`g_m^2`:

    .. math::

        \text{SSR}_u = \sum_{t=1}^{T} \sum_{m=1}^{n^2}
        g_m^2 (\tilde\xi_{m,t} - \gamma \tilde\xi_{m,t-1})^2

    Parameters
    ----------
    eta : ndarray of shape (n², T)
        Latent field draws.
    X : ndarray of shape (n², T, k) or (n², k)
        Covariate array.
    state : PanelGaussianState
        Current Gibbs state.
    cache : PanelGaussianCache
        Precomputed constants.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    sigma2_u : float
        New draw of innovation variance.
    """
    gamma = state.gamma
    beta = state.beta

    # Compute ξ_t = η_t - Xβ in eigenbasis
    if cache.time_invariant_X:
        Xbeta = X @ beta  # (n²,)
        xi = eta - Xbeta[:, np.newaxis]  # (n², T)
    else:
        xi = eta - np.einsum("ijk,k->ij", X, beta)

    # Transform to eigenbasis
    xi_tilde = transform_to_eigenbasis(xi, cache.V, cache.VkronV)

    # Compute gains² for weighting
    gains_d = 1.0 - state.rho_d * cache.eigs_W
    gains_o = 1.0 - state.rho_o * cache.eigs_W
    gains2 = np.outer(gains_d**2, gains_o**2).ravel()  # (n²,)

    # SSR = Σ_t gains² * (ξ̃_t - γ·ξ̃_{t-1})²
    ssr_u = 0.0
    for t in range(cache.T):
        if t == 0:
            # At t=0, ξ̃_0 is drawn from stationary distribution
            # Innovation = ξ̃_0 (no lag), weighted by gains²
            # But the stationary variance is q/(1-γ²), so the
            # contribution to SSR is gains² * ξ̃_0² * (1-γ²)
            ssr_u += np.sum(gains2 * xi_tilde[:, t] ** 2 * (1.0 - gamma**2))
        else:
            innov = xi_tilde[:, t] - gamma * xi_tilde[:, t - 1]
            ssr_u += np.sum(gains2 * innov**2)

    # Inverse-Gamma posterior
    shape_post = cache.a_u + 0.5 * cache.n**2 * cache.T
    scale_post = cache.b_u + 0.5 * ssr_u

    # Draw from IG(shape, scale) = 1 / Gamma(shape, 1/scale)
    sigma2_u = 1.0 / rng.gamma(shape_post, 1.0 / scale_post)
    return sigma2_u


# ---------------------------------------------------------------------------
# Block 4: σ²_y (conjugate Inverse-Gamma)
# ---------------------------------------------------------------------------


def _sample_sigma2_y(
    y: np.ndarray,
    eta: np.ndarray,
    state: PanelGaussianState,
    cache: PanelGaussianCache,
    rng: np.random.Generator,
) -> float:
    r"""Conjugate Inverse-Gamma draw for :math:`\sigma^2_y`.

    .. math::

        \sigma^2_y \mid \cdot \sim
        \text{IG}\!\left(a_y + \frac{n^2 T}{2},\;
        b_y + \frac{\text{SSR}_y}{2}\right)

    where :math:`\text{SSR}_y = \sum_t \|y_t - \eta_t\|^2`.

    Parameters
    ----------
    y : ndarray of shape (n², T)
        Observed flows.
    eta : ndarray of shape (n², T)
        Latent field draws.
    state : PanelGaussianState
        Current Gibbs state.
    cache : PanelGaussianCache
        Precomputed constants.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    sigma2_y : float
        New draw of observation variance.
    """
    resid = y - eta
    ssr_y = np.dot(resid.ravel(), resid.ravel())

    shape_post = cache.a_y + 0.5 * cache.n**2 * cache.T
    scale_post = cache.b_y + 0.5 * ssr_y

    sigma2_y = 1.0 / rng.gamma(shape_post, 1.0 / scale_post)
    return sigma2_y


# ---------------------------------------------------------------------------
# Block 5: γ (conjugate Gaussian, truncated to (-1, 1))
# ---------------------------------------------------------------------------


def _sample_gamma(
    eta: np.ndarray,
    X: np.ndarray,
    state: PanelGaussianState,
    cache: PanelGaussianCache,
    rng: np.random.Generator,
) -> float:
    r"""Conjugate Gaussian draw for :math:`\gamma`, truncated to :math:`(-1, 1)`.

    Uses eigenbasis innovations for consistency with the FFBS block.
    The AR(1) regression in the eigenbasis is:

    .. math::

        \tilde\xi_{m,t} = \gamma \cdot \tilde\xi_{m,t-1} + \tilde\omega_{m,t}

    Sufficient statistics (weighted by gains² for proper scaling):

    .. math::

        S_{zz} = \sum_{t=2}^{T} \sum_m g_m^2 \tilde\xi_{m,t-1}^2, \quad
        S_{rz} = \sum_{t=2}^{T} \sum_m g_m^2 \tilde\xi_{m,t-1}
                 \cdot \tilde\xi_{m,t}

    Posterior (before truncation):

    .. math::

        \sigma^2_{\gamma,\text{post}} = \left(
            \frac{1}{\sigma^2_{\gamma,\text{prior}}}
            + \frac{S_{zz}}{\sigma^2_u}
        \right)^{-1}

        \mu_{\gamma,\text{post}} = \sigma^2_{\gamma,\text{post}}
        \left(
            \frac{\mu_{\gamma,\text{prior}}}{\sigma^2_{\gamma,\text{prior}}}
            + \frac{S_{rz}}{\sigma^2_u}
        \right)

    Parameters
    ----------
    eta : ndarray of shape (n², T)
        Latent field draws.
    X : ndarray of shape (n², T, k) or (n², k)
        Covariate array.
    state : PanelGaussianState
        Current Gibbs state.
    cache : PanelGaussianCache
        Precomputed constants.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    gamma : float
        New draw of temporal AR(1) parameter, :math:`\gamma \in (-1, 1)`.
    """
    beta = state.beta
    sigma2_u = state.sigma2_u

    # Compute ξ_t = η_t - Xβ in eigenbasis
    if cache.time_invariant_X:
        Xbeta = X @ beta
        xi = eta - Xbeta[:, np.newaxis]
    else:
        xi = eta - np.einsum("ijk,k->ij", X, beta)

    xi_tilde = transform_to_eigenbasis(xi, cache.V, cache.VkronV)

    # Compute gains² for weighting
    gains_d = 1.0 - state.rho_d * cache.eigs_W
    gains_o = 1.0 - state.rho_o * cache.eigs_W
    gains2 = np.outer(gains_d**2, gains_o**2).ravel()  # (n²,)

    # Accumulate S_zz and S_rz in eigenbasis
    S_zz = 0.0
    S_rz = 0.0
    for t in range(1, cache.T):
        # Weighted: gains² * ξ̃_{t-1} · ξ̃_{t-1} and gains² * ξ̃_{t-1} · ξ̃_t
        S_zz += np.sum(gains2 * xi_tilde[:, t - 1] ** 2)
        S_rz += np.sum(gains2 * xi_tilde[:, t - 1] * xi_tilde[:, t])

    # Posterior (before truncation)
    prior_prec = 1.0 / cache.gamma_prior_var
    post_prec = prior_prec + S_zz / sigma2_u
    post_var = 1.0 / post_prec
    post_mean = post_var * (cache.gamma_prior_mean * prior_prec + S_rz / sigma2_u)

    # Truncated normal draw on (-1, 1)
    a_trunc = (-1.0 - post_mean) / np.sqrt(post_var)
    b_trunc = (1.0 - post_mean) / np.sqrt(post_var)
    gamma_new = _truncnorm.rvs(
        a_trunc, b_trunc, loc=post_mean, scale=np.sqrt(post_var), random_state=rng
    )
    return float(gamma_new)


# ---------------------------------------------------------------------------
# Block 6: ρ_d (collapsed 1-D slice sampler)
# ---------------------------------------------------------------------------


def _sample_rho_d_panel(
    state: PanelGaussianState,
    cache: PanelGaussianCache,
    ytilde: np.ndarray,
    width_state: SliceWidthState,
    rng: np.random.Generator,
) -> tuple[float, SliceWidthState]:
    r"""Collapsed 1-D slice sampler for :math:`\rho_d`.

    Uses the marginal Kalman filter log-likelihood (integrating over
    :math:`\eta`) for improved mixing over conditioning on the current
    :math:`\eta` draw.

    .. math::

        \log p(\rho_d \mid \text{rest}) \propto
        \log p(y_{1:T} \mid \rho_d, \rho_o, \gamma, \sigma^2_u, \sigma^2_y)
        + \log p(\rho_d)

    where the marginal log-likelihood comes from the scalar Kalman filter
    at cost :math:`O(n^2 T)` per evaluation.

    Parameters
    ----------
    state : PanelGaussianState
        Current Gibbs state.
    cache : PanelGaussianCache
        Precomputed constants.
    ytilde : ndarray of shape (n², T)
        Transformed observations (from Block 1).
    width_state : SliceWidthState
        Mutable state for adaptive slice width.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    rho_d : float
        New draw of destination autoregressive parameter.
    width_state : SliceWidthState
        Updated slice width state.
    """
    rho_lower, rho_upper = cache.rho_bounds

    def log_density(rho_d_prop: float) -> float:
        ll = kf_log_likelihood(
            rho_d_prop,
            state.rho_o,
            state.gamma,
            state.sigma2_u,
            state.sigma2_y,
            cache.eigs_W,
            ytilde,
        )
        # Uniform prior on [rho_lower, rho_upper] → log_prior = 0
        return ll

    rho_d_new, _, _, _ = slice_sample_1d_adaptive(
        log_density,
        state.rho_d,
        rho_lower,
        rho_upper,
        width_state=width_state,
        rng=rng,
    )
    return rho_d_new, width_state


# ---------------------------------------------------------------------------
# Block 7: ρ_o (collapsed 1-D slice sampler)
# ---------------------------------------------------------------------------


def _sample_rho_o_panel(
    state: PanelGaussianState,
    cache: PanelGaussianCache,
    ytilde: np.ndarray,
    width_state: SliceWidthState,
    rng: np.random.Generator,
) -> tuple[float, SliceWidthState]:
    r"""Collapsed 1-D slice sampler for :math:`\rho_o`.

    Identical structure to :func:`_sample_rho_d_panel` but varying
    :math:`\rho_o` while holding :math:`\rho_d` fixed.

    Parameters
    ----------
    state : PanelGaussianState
        Current Gibbs state.
    cache : PanelGaussianCache
        Precomputed constants.
    ytilde : ndarray of shape (n², T)
        Transformed observations (from Block 1).
    width_state : SliceWidthState
        Mutable state for adaptive slice width.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    rho_o : float
        New draw of origin autoregressive parameter.
    width_state : SliceWidthState
        Updated slice width state.
    """
    rho_lower, rho_upper = cache.rho_bounds

    def log_density(rho_o_prop: float) -> float:
        ll = kf_log_likelihood(
            state.rho_d,
            rho_o_prop,
            state.gamma,
            state.sigma2_u,
            state.sigma2_y,
            cache.eigs_W,
            ytilde,
        )
        return ll

    rho_o_new, _, _, _ = slice_sample_1d_adaptive(
        log_density,
        state.rho_o,
        rho_lower,
        rho_upper,
        width_state=width_state,
        rng=rng,
    )
    return rho_o_new, width_state
