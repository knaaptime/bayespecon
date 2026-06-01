r"""Scalar Kalman filter, FFBS backward sampler, and eigenbasis transforms.

The key insight: because the observation matrix is the identity and the
observation noise is isotropic (:math:`H = \sigma^2_y I`), the
:math:`n^2`-dimensional Kalman filter decomposes into :math:`n^2`
independent scalar filters in the eigenbasis of the spatial precision
matrix :math:`Q_{\text{space}}`.

Each mode is a scalar AR(1) process with its own innovation variance
:math:`q_m`. The forward pass and backward sampler are fully vectorised
over modes, giving :math:`O(n^2 T)` cost.

References
----------
Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
Space Methods*. Oxford University Press.

See Also
--------
bayespecon.samplers.panel_flow._state
    KFOutput namedtuple used as forward-pass return type.
"""

from __future__ import annotations

import numpy as np

from ._state import KFOutput

# ---------------------------------------------------------------------------
# Diffuse initialisation threshold
# ---------------------------------------------------------------------------

_DIFFUSE_GAMMA_THRESH = 0.99
"""When |γ| exceeds this, use diffuse initialisation p_{0|0} = 1e6."""

_DIFFUSE_VAR = 1e6
"""Diffuse initial variance for near-unit-root modes."""


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def kf_forward_pass(
    ytilde: np.ndarray,
    q_modes: np.ndarray,
    gamma: float,
    sigma2_y: float,
) -> KFOutput:
    r"""Run the scalar Kalman filter forward pass over all :math:`n^2` modes.

    Each mode evolves as a scalar AR(1):

    .. math::

        \tilde\eta_{m,t} = \gamma \tilde\eta_{m,t-1} + \tilde\xi_{m,t},
        \quad \tilde\xi_{m,t} \sim N(0, q_m)

    with observation

    .. math::

        \tilde y_{m,t} = \tilde\eta_{m,t} + \tilde\varepsilon_{m,t},
        \quad \tilde\varepsilon_{m,t} \sim N(0, \sigma^2_y)

    All :math:`n^2` modes are filtered independently and simultaneously
    via vectorised operations.

    Parameters
    ----------
    ytilde : ndarray of shape (n², T)
        Observations transformed to the eigenbasis.
    q_modes : ndarray of shape (n²,)
        Modal innovation variances.
    gamma : float
        Temporal AR(1) coefficient, :math:`\gamma \in (-1, 1)`.
    sigma2_y : float
        Observation variance.

    Returns
    -------
    KFOutput
        Filtered means, filtered variances, predicted variances, and
        marginal log-likelihood.

    Notes
    -----
    Initialisation uses the stationary variance
    :math:`p_{m,0|0} = q_m / (1 - \gamma^2)` when :math:`|\gamma| < 0.99`,
    and a diffuse prior :math:`p_{m,0|0} = 10^6` otherwise.
    """
    n2, T = ytilde.shape

    # Output arrays
    filtered_means = np.empty((n2, T))
    filtered_vars = np.empty((n2, T))
    pred_vars = np.empty((n2, T))

    # Initialisation
    if abs(gamma) < _DIFFUSE_GAMMA_THRESH:
        p = q_modes / (1.0 - gamma**2)  # stationary variance
    else:
        p = np.full(n2, _DIFFUSE_VAR)

    m = np.zeros(n2)

    log_lik = 0.0
    gamma2 = gamma * gamma

    for t in range(T):
        # Predict
        if t > 0:
            m = gamma * m
            p = gamma2 * p + q_modes

        pred_vars[:, t] = p

        # Update
        v = ytilde[:, t] - m  # innovation
        s = p + sigma2_y  # innovation variance (scalar per mode)
        k = p / s  # Kalman gain

        filtered_means[:, t] = m + k * v
        filtered_vars[:, t] = (1.0 - k) * p

        # Log-likelihood contribution
        log_lik += -0.5 * np.sum(np.log(s) + v * v / s)

        # Prepare for next step
        m = filtered_means[:, t]
        p = filtered_vars[:, t]

    return KFOutput(
        filtered_means=filtered_means,
        filtered_vars=filtered_vars,
        pred_vars=pred_vars,
        log_likelihood=log_lik,
    )


# ---------------------------------------------------------------------------
# Backward sampler (FFBS)
# ---------------------------------------------------------------------------


def ffbs_backward_pass(
    kf_out: KFOutput,
    gamma: float,
    q_modes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Forward Filtering Backward Sampling (FFBS) for all :math:`n^2` modes.

    Draws a joint sample from the smoothing distribution
    :math:`p(\tilde\eta_{1:T} \mid \tilde y_{1:T}, \theta)` by sampling
    backwards from :math:`t = T` to :math:`t = 1`.

    Parameters
    ----------
    kf_out : KFOutput
        Output from :func:`kf_forward_pass`.
    gamma : float
        Temporal AR(1) coefficient.
    q_modes : ndarray of shape (n²,)
        Modal innovation variances (unused directly but kept for API
        consistency; the backward pass uses stored forward quantities).
    rng : numpy.random.Generator
        Random state for drawing the backward samples.

    Returns
    -------
    eta_tilde : ndarray of shape (n², T)
        Drawn latent field in the eigenbasis.
    """
    n2, T = kf_out.filtered_means.shape

    eta_tilde = np.empty((n2, T))

    # Terminal draw at t = T
    eta_tilde[:, T - 1] = rng.normal(
        kf_out.filtered_means[:, T - 1],
        np.sqrt(kf_out.filtered_vars[:, T - 1]),
    )

    # Backward pass
    for t in range(T - 2, -1, -1):
        # Smoother gain (scalar per mode)
        J = kf_out.filtered_vars[:, t] * gamma / kf_out.pred_vars[:, t + 1]

        # Conditional mean and variance
        mu = kf_out.filtered_means[:, t] + J * (
            eta_tilde[:, t + 1] - gamma * kf_out.filtered_means[:, t]
        )
        sigma2 = kf_out.filtered_vars[:, t] - J**2 * kf_out.pred_vars[:, t + 1]

        # Clamp negative variances from numerical noise
        sigma2 = np.maximum(sigma2, 1e-12)

        eta_tilde[:, t] = rng.normal(mu, np.sqrt(sigma2))

    return eta_tilde


# ---------------------------------------------------------------------------
# Log-likelihood only (for collapsed slice samplers)
# ---------------------------------------------------------------------------


def kf_log_likelihood(
    rho_d: float,
    rho_o: float,
    gamma: float,
    sigma2_u: float,
    sigma2_y: float,
    eigs_W: np.ndarray,
    ytilde: np.ndarray,
) -> float:
    r"""Compute the marginal log-likelihood via the scalar Kalman filter.

    Used by the collapsed :math:`\rho` and :math:`\gamma` slice samplers
    to evaluate the marginal posterior without drawing :math:`\eta`.

    Parameters
    ----------
    rho_d : float
        Destination autoregressive parameter.
    rho_o : float
        Origin autoregressive parameter.
    gamma : float
        Temporal AR(1) coefficient.
    sigma2_u : float
        Innovation variance.
    sigma2_y : float
        Observation variance.
    eigs_W : ndarray of shape (n,)
        Eigenvalues of the spatial weights matrix :math:`W`.
    ytilde : ndarray of shape (n², T)
        Observations transformed to the eigenbasis.

    Returns
    -------
    log_lik : float
        Marginal log-likelihood
        :math:`\log p(y_{1:T} \mid \rho_d, \rho_o, \gamma, \sigma^2_u, \sigma^2_y)`.
    """
    # Compute modal variances for the proposed (rho_d, rho_o)
    gains_d = 1.0 - rho_d * eigs_W
    gains_o = 1.0 - rho_o * eigs_W
    q_modes = sigma2_u / np.outer(gains_d**2, gains_o**2).ravel()

    kf_out = kf_forward_pass(ytilde, q_modes, gamma, sigma2_y)
    return kf_out.log_likelihood


# ---------------------------------------------------------------------------
# Eigenbasis transforms
# ---------------------------------------------------------------------------


def transform_to_eigenbasis(
    v: np.ndarray,
    V: np.ndarray,
    VkronV: np.ndarray | None = None,
) -> np.ndarray:
    r"""Transform from the spatial basis to the eigenbasis: :math:`(V \otimes V)^\top v`.

    Parameters
    ----------
    v : ndarray of shape (n²,) or (n², T)
        Vector(s) in the spatial basis.
    V : ndarray of shape (n, n)
        Eigenvector matrix of :math:`W`.
    VkronV : ndarray of shape (n², n²) or None
        Precomputed :math:`V \otimes V`. If None, uses implicit
        Kronecker matvec via :func:`kron_At_matvec`.

    Returns
    -------
    v_tilde : ndarray, same shape as *v*
        Transformed vector(s) in the eigenbasis.
    """
    if VkronV is not None:
        if v.ndim == 1:
            return VkronV.T @ v
        else:
            return VkronV.T @ v

    from ..panel._kronecker import kron_At_matvec

    if v.ndim == 1:
        return kron_At_matvec(v, V.T, V.T)
    else:
        # Apply column-wise for (n², T) array
        result = np.empty_like(v)
        for t in range(v.shape[1]):
            result[:, t] = kron_At_matvec(v[:, t], V.T, V.T)
        return result


def transform_from_eigenbasis(
    v: np.ndarray,
    V: np.ndarray,
    VkronV: np.ndarray | None = None,
) -> np.ndarray:
    r"""Transform from the eigenbasis to the spatial basis: :math:`(V \otimes V) v`.

    Parameters
    ----------
    v : ndarray of shape (n²,) or (n², T)
        Vector(s) in the eigenbasis.
    V : ndarray of shape (n, n)
        Eigenvector matrix of :math:`W`.
    VkronV : ndarray of shape (n², n²) or None
        Precomputed :math:`V \otimes V`. If None, uses implicit
        Kronecker matvec via :func:`kron_matvec`.

    Returns
    -------
    v_spatial : ndarray, same shape as *v*
        Transformed vector(s) in the spatial basis.
    """
    if VkronV is not None:
        if v.ndim == 1:
            return VkronV @ v
        else:
            return VkronV @ v

    from ..panel._kronecker import kron_matvec

    if v.ndim == 1:
        return kron_matvec(v, V, V)
    else:
        result = np.empty_like(v)
        for t in range(v.shape[1]):
            result[:, t] = kron_matvec(v[:, t], V, V)
        return result
