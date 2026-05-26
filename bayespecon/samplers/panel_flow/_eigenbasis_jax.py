r"""JAX-native scalar Kalman filter, FFBS backward sampler, and eigenbasis transforms.

Replaces the Python loops in :mod:`~bayespecon.samplers.panel_flow._eigenbasis`
with ``jax.lax.scan`` for full JIT compilation. All operations are
vectorised over the :math:`n^2` modes.

The key difference from the numpy version: the forward pass uses
``jax.lax.scan`` instead of a Python ``for`` loop, and the backward
pass uses ``jax.lax.scan`` with reverse ordering. This eliminates
Python dispatch overhead and enables XLA optimisation of the entire
FFBS pipeline.

See Also
--------
bayespecon.samplers.panel_flow._eigenbasis
    Numpy-based KF/FFBS (reference implementation).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Diffuse initialisation threshold
# ---------------------------------------------------------------------------

_DIFFUSE_GAMMA_THRESH = 0.99
_DIFFUSE_VAR = 1e6


# ---------------------------------------------------------------------------
# Forward pass using jax.lax.scan
# ---------------------------------------------------------------------------


def kf_forward_pass_jax(
    ytilde: jax.Array,
    q_modes: jax.Array,
    gamma: float | jax.Array,
    sigma2_y: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""JAX-native scalar Kalman filter forward pass.

    Uses ``jax.lax.scan`` to iterate over time steps, fully JIT-compatible.

    Parameters
    ----------
    ytilde : jax.Array of shape (n², T)
        Observations transformed to the eigenbasis.
    q_modes : jax.Array of shape (n²,)
        Modal innovation variances.
    gamma : float or jax.Array (scalar)
        Temporal AR(1) coefficient.
    sigma2_y : float or jax.Array (scalar)
        Observation variance.

    Returns
    -------
    filtered_means : jax.Array of shape (n², T)
    filtered_vars : jax.Array of shape (n², T)
    pred_vars : jax.Array of shape (n², T)
    log_likelihood : jax.Array (scalar)
    """
    n2, T = ytilde.shape
    gamma2 = gamma * gamma

    # Initialisation
    p_init = jnp.where(
        jnp.abs(gamma) < _DIFFUSE_GAMMA_THRESH,
        q_modes / (1.0 - gamma2),
        jnp.full(n2, _DIFFUSE_VAR),
    )
    m_init = jnp.zeros(n2)

    # Carry: (m, p, log_lik)
    init_carry = (m_init, p_init, jnp.float64(0.0))

    def step(carry, yt):
        m, p, log_lik = carry
        # Predict (skip for t=0 handled by init)
        m_pred = gamma * m
        p_pred = gamma2 * p + q_modes

        # Update
        v = yt - m_pred  # innovation
        s = p_pred + sigma2_y  # innovation variance
        k = p_pred / s  # Kalman gain
        m_new = m_pred + k * v
        p_new = (1.0 - k) * p_pred

        # Log-likelihood contribution
        ll_inc = -0.5 * jnp.sum(jnp.log(s) + v * v / s)

        new_carry = (m_new, p_new, log_lik + ll_inc)
        output = (m_new, p_new, p_pred)
        return new_carry, output

    # For t=0, we need to skip the predict step (use init values directly)
    # We handle this by using the init carry for the first step
    def first_step(carry, yt):
        m, p, log_lik = carry
        # No predict for t=0 — use initial p directly
        s = p + sigma2_y
        v = yt - m
        k = p / s
        m_new = m + k * v
        p_new = (1.0 - k) * p

        ll_inc = -0.5 * jnp.sum(jnp.log(s) + v * v / s)

        new_carry = (m_new, p_new, log_lik + ll_inc)
        output = (m_new, p_new, p)
        return new_carry, output

    # Run first step separately, then scan the rest
    first_carry, first_output = first_step(init_carry, ytilde[:, 0])

    # Scan remaining steps (T-1 steps)
    # When T=1, ytilde[:, 1:] has shape (n², 0) and scan returns empty arrays
    _, (means_rest, vars_rest, preds_rest) = jax.lax.scan(
        step, first_carry, ytilde[:, 1:].T
    )  # scan over (T-1, n²)

    # Concatenate first step with rest
    filtered_means = jnp.concatenate(
        [first_output[0][:, jnp.newaxis], means_rest.T], axis=1
    )
    filtered_vars = jnp.concatenate(
        [first_output[1][:, jnp.newaxis], vars_rest.T], axis=1
    )
    pred_vars = jnp.concatenate([first_output[2][:, jnp.newaxis], preds_rest.T], axis=1)

    return filtered_means, filtered_vars, pred_vars, first_carry[2]


# ---------------------------------------------------------------------------
# FFBS backward pass using jax.lax.scan
# ---------------------------------------------------------------------------


def ffbs_backward_pass_jax(
    filtered_means: jax.Array,
    filtered_vars: jax.Array,
    pred_vars: jax.Array,
    gamma: float | jax.Array,
    key: jax.Array,
) -> jax.Array:
    r"""JAX-native FFBS backward sampler.

    Uses ``jax.lax.scan`` with reverse ordering to draw the backward
    samples. All :math:`n^2` modes are sampled simultaneously.

    Parameters
    ----------
    filtered_means : jax.Array of shape (n², T)
    filtered_vars : jax.Array of shape (n², T)
    pred_vars : jax.Array of shape (n², T)
    gamma : float or jax.Array (scalar)
    key : jax.Array
        JAX PRNG key.

    Returns
    -------
    eta_tilde : jax.Array of shape (n², T)
        Drawn latent field in the eigenbasis.
    """
    n2, T = filtered_means.shape

    # Terminal draw at t = T-1
    key_term, key_scan = jax.random.split(key)
    eta_T = (
        jax.random.normal(key_term, shape=(n2,), dtype=jnp.float64)
        * jnp.sqrt(filtered_vars[:, -1])
        + filtered_means[:, -1]
    )

    # Backward scan from t = T-2 to t = 0
    # Carry: eta_{t+1} (the draw from the next period)
    init_carry = eta_T

    # We need per-step keys for the backward pass
    # Split keys for each backward step
    n_backward = T - 1
    keys = jax.random.split(key_scan, n_backward)

    def backward_step_keyed(eta_next, inputs):
        t_idx, step_key = inputs
        J = filtered_vars[:, t_idx] * gamma / pred_vars[:, t_idx + 1]
        mu = filtered_means[:, t_idx] + J * (
            eta_next - gamma * filtered_means[:, t_idx]
        )
        sigma2 = jnp.maximum(
            filtered_vars[:, t_idx] - J**2 * pred_vars[:, t_idx + 1],
            1e-12,
        )
        eta_t = (
            jax.random.normal(step_key, shape=(n2,), dtype=jnp.float64)
            * jnp.sqrt(sigma2)
            + mu
        )
        return eta_t, eta_t

    # Indices from T-2 down to 0
    t_indices = jnp.arange(T - 2, -1, -1)

    # When n_backward=0, scan over empty arrays returns empty output
    _, eta_backward = jax.lax.scan(
        backward_step_keyed,
        init_carry,
        (t_indices, keys),
    )
    # eta_backward has shape (T-1, n²), need to reverse and transpose
    eta_backward = jnp.flip(eta_backward, axis=0).T  # (n², T-1)
    eta_tilde = jnp.concatenate([eta_backward, eta_T[:, jnp.newaxis]], axis=1)

    return eta_tilde


# ---------------------------------------------------------------------------
# Log-likelihood only (for collapsed slice samplers)
# ---------------------------------------------------------------------------


def kf_log_likelihood_jax(
    rho_d: float | jax.Array,
    rho_o: float | jax.Array,
    gamma: float | jax.Array,
    sigma2_u: float | jax.Array,
    sigma2_y: float | jax.Array,
    eigs_W: jax.Array,
    ytilde: jax.Array,
) -> jax.Array:
    r"""JAX-native marginal log-likelihood via the scalar Kalman filter.

    Used by the collapsed :math:`\rho` and :math:`\gamma` slice samplers.
    Fully differentiable via JAX autodiff.

    Parameters
    ----------
    rho_d, rho_o : float or jax.Array (scalar)
        Spatial autoregressive parameters.
    gamma : float or jax.Array (scalar)
        Temporal AR(1) coefficient.
    sigma2_u : float or jax.Array (scalar)
        Innovation variance.
    sigma2_y : float or jax.Array (scalar)
        Observation variance.
    eigs_W : jax.Array of shape (n,)
        Eigenvalues of :math:`W`.
    ytilde : jax.Array of shape (n², T)
        Transformed observations.

    Returns
    -------
    log_lik : jax.Array (scalar)
        Marginal log-likelihood.
    """
    gains_d = 1.0 - rho_d * eigs_W
    gains_o = 1.0 - rho_o * eigs_W
    q_modes = sigma2_u / jnp.outer(gains_d**2, gains_o**2).ravel()

    _, _, _, log_lik = kf_forward_pass_jax(ytilde, q_modes, gamma, sigma2_y)
    return log_lik


# ---------------------------------------------------------------------------
# Eigenbasis transforms (JAX)
# ---------------------------------------------------------------------------


def transform_to_eigenbasis_jax(
    v: jax.Array,
    V: jax.Array,
    VkronV: jax.Array | None = None,
) -> jax.Array:
    r"""Transform to eigenbasis: :math:`(V \otimes V)^\top v`."""
    if VkronV is not None:
        if v.ndim == 1:
            return VkronV.T @ v
        else:
            return VkronV.T @ v

    # For JAX, we need a JAX-native kron_At_matvec
    # Fall back to explicit (V⊗V)ᵀ v = vec(Vᵀ X V) where X = reshape(v)
    n = V.shape[0]
    if v.ndim == 1:
        X = v.reshape(n, n, order="F")
        return (V.T @ X @ V).ravel(order="F")
    else:
        # (n², T) — apply column-wise
        def transform_col(col):
            X = col.reshape(n, n, order="F")
            return (V.T @ X @ V).ravel(order="F")

        return jax.vmap(transform_col, in_axes=1, out_axes=1)(v)


def transform_from_eigenbasis_jax(
    v: jax.Array,
    V: jax.Array,
    VkronV: jax.Array | None = None,
) -> jax.Array:
    r"""Transform from eigenbasis: :math:`(V \otimes V) v`."""
    if VkronV is not None:
        if v.ndim == 1:
            return VkronV @ v
        else:
            return VkronV @ v

    n = V.shape[0]
    if v.ndim == 1:
        X = v.reshape(n, n, order="F")
        return (V @ X @ V.T).ravel(order="F")
    else:

        def transform_col(col):
            X = col.reshape(n, n, order="F")
            return (V @ X @ V.T).ravel(order="F")

        return jax.vmap(transform_col, in_axes=1, out_axes=1)(v)
