r"""JAX-native full-JIT Gibbs step for the Gaussian panel flow sampler.

Composes all Gibbs blocks (η FFBS, β, σ²_u, σ²_y, γ, ρ_d, ρ_o) into a
single ``@eqx.filter_jit``-compiled function, eliminating Python→JAX
dispatch overhead entirely.

The key insight (same as ``negbin/_jax.py``): each individual JAX
operation is fast (~0.05ms), but the Python→JAX dispatch overhead
per call is ~30ms.  By composing all blocks into a single JIT-compiled
function, we pay the dispatch cost only once per Gibbs iteration
instead of 7+ times.

Architecture
------------
Uses the ``_make_gibbs_step_with_data()`` closure pattern from
``negbin/_jax.py``: data and precomputed constants are bound into the
closure, and the returned ``gibbs_step(state, key)`` function is
``@eqx.filter_jit``-compiled.

The ρ slice samplers use ``jax.lax.fori_loop`` for the stepping-out
phase and ``jax.lax.while_loop`` for the shrinkage phase, keeping the
entire sampler inside a single XLA kernel.

See Also
--------
bayespecon.samplers.panel_flow._blocks_gaussian
    Numpy-based block samplers (reference implementation).
bayespecon.samplers.negbin._jax
    Full-JIT NB Gibbs sampler with the same closure pattern.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayespecon._jax_dispatch import ensure_x64


def _check_jax_available() -> None:
    """Raise ImportError if JAX or equinox is not installed."""
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        raise ImportError(
            "JAX is required for the full-JIT panel flow Gibbs sampler. "
            "Install with: pip install jax"
        )
    if importlib.util.find_spec("equinox") is None:
        raise ImportError(
            "equinox is required for the full-JIT panel flow Gibbs sampler. "
            "Install with: pip install equinox"
        )


def _slice_sample_1d_jax(
    log_density,
    x0,
    lower,
    upper,
    key,
    w=1.0,
    max_steps_out=10,
    max_shrink=50,
):
    """JAX-native univariate slice sampler (Neal 2003).

    Uses ``jax.lax.fori_loop`` for stepping-out and
    ``jax.lax.while_loop`` for shrinkage, fully JIT-compatible.

    Parameters
    ----------
    log_density : callable
        Log-density function (JAX-compatible).
    x0 : jax.Array (scalar)
        Current state.
    lower, upper : jax.Array (scalar)
        Support bounds.
    key : jax.Array
        JAX PRNG key.
    w : float
        Step-out width.
    max_steps_out : int
        Maximum stepping-out iterations per side.
    max_shrink : int
        Maximum shrinkage iterations.

    Returns
    -------
    x_new : jax.Array (scalar)
        New sample.
    """
    w_jax = jnp.float64(w)

    # Evaluate log-density at current point
    log_y0 = log_density(x0)

    # Draw vertical level
    key_u, key_init, key_shrink = jax.random.split(key, 3)
    log_u = log_y0 + jnp.log(jax.random.uniform(key_u, dtype=jnp.float64))

    # Initialise interval [L, R]
    u_rand = jax.random.uniform(key_init, dtype=jnp.float64)
    L = jnp.maximum(x0 - u_rand * w_jax, lower)
    R = jnp.minimum(L + w_jax, upper)

    # Stepping out — left
    def step_out_left(i, L_carry):
        L_new = jnp.maximum(L_carry - w_jax, lower)
        L_out = jnp.where(log_density(L_new) > log_u, L_new, L_carry)
        return L_out

    L = jax.lax.fori_loop(0, max_steps_out, step_out_left, L)

    # Stepping out — right
    def step_out_right(i, R_carry):
        R_new = jnp.minimum(R_carry + w_jax, upper)
        R_out = jnp.where(log_density(R_new) > log_u, R_new, R_carry)
        return R_out

    R = jax.lax.fori_loop(0, max_steps_out, step_out_right, R)

    # Shrinkage via jax.lax.while_loop
    # State: (x_prop, L, R, key, converged)
    init_shrink = (x0, L, R, key_shrink, jnp.bool_(False))

    def shrink_cond(state):
        _, _, _, _, converged = state
        return ~converged

    def shrink_body(state):
        x_prop, L_s, R_s, key_s, _ = state
        key_s, key_new = jax.random.split(key_s)
        x_prop = L_s + jax.random.uniform(key_new, dtype=jnp.float64) * (R_s - L_s)
        accept = log_density(x_prop) > log_u
        L_s = jnp.where((x_prop < x0) & ~accept, x_prop, L_s)
        R_s = jnp.where((x_prop >= x0) & ~accept, x_prop, R_s)
        converged = accept | ((R_s - L_s) < 1e-12)
        return (x_prop, L_s, R_s, key_s, converged)

    x_new, _, _, _, _ = jax.lax.while_loop(shrink_cond, shrink_body, init_shrink)
    return x_new


def _make_gibbs_step_with_data(
    y_jax: jax.Array,
    X_jax: jax.Array,
    eigs_W_jax: jax.Array,
    V_jax: jax.Array,
    VkronV_jax: jax.Array | None,
    n: int,
    T: int,
    k: int,
    beta_prior_mean: jax.Array,
    beta_prior_prec: jax.Array,
    a_u: float,
    b_u: float,
    a_y: float,
    b_y: float,
    gamma_prior_mean: float,
    gamma_prior_var: float,
    rho_lower: float,
    rho_upper: float,
    time_invariant_X: bool,
    XtX_jax: jax.Array | None,
    slice_w_rho: float = 0.1,
    slice_max_steps_out: int = 10,
    slice_max_shrink: int = 50,
):
    """Build a JIT-compiled Gibbs step with data bound into the closure.

    Parameters
    ----------
    y_jax : jax.Array of shape (n², T)
        Observed flows.
    X_jax : jax.Array of shape (n², k) or (n², T, k)
        Covariate array.
    eigs_W_jax : jax.Array of shape (n,)
        Eigenvalues of W.
    V_jax : jax.Array of shape (n, n)
        Eigenvectors of W.
    VkronV_jax : jax.Array of shape (n², n²) or None
        Precomputed (V⊗V).
    n, T, k : int
        Dimensions.
    beta_prior_mean : jax.Array of shape (k,)
    beta_prior_prec : jax.Array of shape (k, k)
    a_u, b_u, a_y, b_y : float
        Inverse-Gamma hyperparameters.
    gamma_prior_mean, gamma_prior_var : float
    rho_lower, rho_upper : float
        Stability bounds for ρ.
    time_invariant_X : bool
    XtX_jax : jax.Array of shape (k, k) or None
    slice_w_rho : float
        Initial slice width for ρ samplers.
    slice_max_steps_out : int
        Maximum stepping-out iterations.
    slice_max_shrink : int
        Maximum shrinkage iterations.

    Returns
    -------
    gibbs_step : callable
        JIT-compiled function with signature::

            gibbs_step(state, key) -> (new_state, log_lik)
    """
    import equinox as eqx

    from ._eigenbasis_jax import (
        ffbs_backward_pass_jax,
        kf_forward_pass_jax,
        kf_log_likelihood_jax,
        transform_from_eigenbasis_jax,
        transform_to_eigenbasis_jax,
    )
    from ._state_jax import JAXPanelGaussianState

    ensure_x64()

    n2 = n * n

    # Convert constants to JAX arrays
    a_u_jax = jnp.float64(a_u)
    b_u_jax = jnp.float64(b_u)
    a_y_jax = jnp.float64(a_y)
    b_y_jax = jnp.float64(b_y)
    gamma_prior_mean_jax = jnp.float64(gamma_prior_mean)
    gamma_prior_var_jax = jnp.float64(gamma_prior_var)
    rho_lower_jax = jnp.float64(rho_lower)
    rho_upper_jax = jnp.float64(rho_upper)
    n2T_half = jnp.float64(n2 * T / 2.0)

    @eqx.filter_jit
    def gibbs_step(state, key):
        """One complete Gibbs sweep: η → β → σ²_u → σ²_y → γ → ρ_d → ρ_o.

        Parameters
        ----------
        state : JAXPanelGaussianState
        key : jax.random.PRNGKey

        Returns
        -------
        new_state : JAXPanelGaussianState
        log_lik : jax.Array (scalar)
        """
        (
            key_eta,
            key_beta,
            key_su,
            key_sy,
            key_gamma,
            key_rho_d,
            key_rho_o,
        ) = jax.random.split(key, 7)

        rho_d = state.rho_d
        rho_o = state.rho_o
        gamma = state.gamma
        sigma2_u = state.sigma2_u
        sigma2_y = state.sigma2_y
        beta = state.beta

        # ── Block 1: η via FFBS ──
        gains_d = 1.0 - rho_d * eigs_W_jax
        gains_o = 1.0 - rho_o * eigs_W_jax
        q_modes = sigma2_u / jnp.outer(gains_d**2, gains_o**2).ravel()

        if time_invariant_X:
            Xbeta = X_jax @ beta  # (n²,)
            r = y_jax - Xbeta[:, jnp.newaxis]  # (n², T)
        else:
            Xbeta = jnp.einsum("ijk,k->ij", X_jax, beta)
            r = y_jax - Xbeta

        ytilde = transform_to_eigenbasis_jax(r, V_jax, VkronV_jax)
        fm, fv, pv, _ = kf_forward_pass_jax(ytilde, q_modes, gamma, sigma2_y)
        eta_tilde = ffbs_backward_pass_jax(fm, fv, pv, gamma, key_eta)
        eta = transform_from_eigenbasis_jax(eta_tilde, V_jax, VkronV_jax)

        if time_invariant_X:
            eta = eta + Xbeta[:, jnp.newaxis]
        else:
            eta = eta + Xbeta

        # ── Block 2: β (conjugate normal) ──
        # Regression: η_t = Xβ + ξ_t, ξ_t ~ N(0, σ²_u I) (approx)
        # Sufficient stats: Σ_t X'η_t and Σ_t X'X

        if time_invariant_X and XtX_jax is not None:
            XtX_sum = T * XtX_jax
            XtEta = X_jax.T @ eta.sum(axis=1)  # (k,)
        else:

            def beta_scan_step(carry, t):
                XtX_acc, XtEta_acc = carry
                Xt_slice = X_jax[:, t, :]
                XtX_new = XtX_acc + Xt_slice.T @ Xt_slice
                XtEta_new = XtEta_acc + Xt_slice.T @ eta[:, t]
                return (XtX_new, XtEta_new), None

            init_carry_beta = (jnp.zeros((k, k)), jnp.zeros(k))
            (XtX_sum, XtEta), _ = jax.lax.scan(
                beta_scan_step, init_carry_beta, jnp.arange(T)
            )

        prec_post = beta_prior_prec + XtX_sum / sigma2_u
        cov_post = jnp.linalg.inv(prec_post)
        mean_post = cov_post @ (beta_prior_prec @ beta_prior_mean + XtEta / sigma2_u)
        beta_new = jax.random.multivariate_normal(key_beta, mean_post, cov_post)

        # ── Block 3: σ²_u (conjugate IG) ──
        # Use eigenbasis innovations for consistency with FFBS
        # ξ_t = η_t - Xβ in eigenbasis, weighted by gains²
        if time_invariant_X:
            Xbeta_new = X_jax @ beta_new
            xi = eta - Xbeta_new[:, jnp.newaxis]
        else:
            Xbeta_new = jnp.einsum("ijk,k->ij", X_jax, beta_new)
            xi = eta - Xbeta_new

        xi_tilde = transform_to_eigenbasis_jax(xi, V_jax, VkronV_jax)

        # Compute gains²
        gains_d = 1.0 - rho_d * eigs_W_jax
        gains_o = 1.0 - rho_o * eigs_W_jax
        gains2 = jnp.outer(gains_d**2, gains_o**2).ravel()

        # SSR via scan: Σ_t gains² * (ξ̃_t - γ·ξ̃_{t-1})²
        # At t=0: gains² * ξ̃_0² * (1-γ²) (from stationary distribution)
        def ssr_u_scan_step(carry, t):
            ssr_acc, xi_prev = carry
            innov = jnp.where(t > 0, xi_tilde[:, t] - gamma * xi_prev, xi_tilde[:, t])
            weight = jnp.where(t > 0, gains2, gains2 * (1.0 - gamma**2))
            ssr_new = ssr_acc + jnp.sum(weight * innov**2)
            return (ssr_new, xi_tilde[:, t]), None

        init_carry_su = (jnp.float64(0.0), jnp.zeros(n2))
        (ssr_u, _), _ = jax.lax.scan(ssr_u_scan_step, init_carry_su, jnp.arange(T))

        shape_u = a_u_jax + n2T_half
        scale_u = b_u_jax + 0.5 * ssr_u
        sigma2_u_new = (
            1.0 / jax.random.gamma(key_su, shape_u, dtype=jnp.float64) * (1.0 / scale_u)
        )

        # ── Block 4: σ²_y (conjugate IG) ──
        resid_y = y_jax - eta
        ssr_y = jnp.sum(resid_y**2)
        shape_y = a_y_jax + n2T_half
        scale_y = b_y_jax + 0.5 * ssr_y
        sigma2_y_new = (
            1.0 / jax.random.gamma(key_sy, shape_y, dtype=jnp.float64) * (1.0 / scale_y)
        )

        # ── Block 5: γ (conjugate normal, truncated) ──
        # AR(1) regression in eigenbasis: ξ̃_t = γ·ξ̃_{t-1} + ω̃_t
        # Sufficient stats weighted by gains²
        # Recompute xi_tilde with new beta
        if time_invariant_X:
            xi_new = eta - Xbeta_new[:, jnp.newaxis]
        else:
            xi_new = eta - Xbeta_new
        xi_tilde_new = transform_to_eigenbasis_jax(xi_new, V_jax, VkronV_jax)

        def gamma_scan_step(carry, t):
            S_zz_acc, S_rz_acc, xi_prev = carry
            S_zz_new = jnp.where(
                t > 0, S_zz_acc + jnp.sum(gains2 * xi_prev**2), S_zz_acc
            )
            S_rz_new = jnp.where(
                t > 0,
                S_rz_acc + jnp.sum(gains2 * xi_prev * xi_tilde_new[:, t]),
                S_rz_acc,
            )
            return (S_zz_new, S_rz_new, xi_tilde_new[:, t]), None

        init_carry_gamma = (jnp.float64(0.0), jnp.float64(0.0), jnp.zeros(n2))
        (S_zz, S_rz, _), _ = jax.lax.scan(
            gamma_scan_step, init_carry_gamma, jnp.arange(T)
        )

        prior_prec = 1.0 / gamma_prior_var_jax
        post_prec = prior_prec + S_zz / sigma2_u_new
        post_var = 1.0 / post_prec
        post_mean = post_var * (gamma_prior_mean_jax * prior_prec + S_rz / sigma2_u_new)

        # Truncated normal via inverse CDF of standard normal
        # Φ^{-1}(u · Φ(b) + (1-u) · Φ(a)) where a,b are standardised bounds
        from jax.scipy.special import ndtr, ndtri

        a_std = (-1.0 - post_mean) / jnp.sqrt(post_var)
        b_std = (1.0 - post_mean) / jnp.sqrt(post_var)
        Phi_a = ndtr(a_std)
        Phi_b = ndtr(b_std)
        u = jax.random.uniform(key_gamma, dtype=jnp.float64)
        p = Phi_a + u * (Phi_b - Phi_a)
        # Clamp to avoid NaN from ndtri at exactly 0 or 1
        p = jnp.clip(p, 1e-10, 1.0 - 1e-10)
        gamma_new = post_mean + jnp.sqrt(post_var) * ndtri(p)

        # ── Block 6: ρ_d (collapsed slice) ──
        # Recompute ytilde with new beta
        if time_invariant_X:
            Xbeta_new = X_jax @ beta_new
            r_new = y_jax - Xbeta_new[:, jnp.newaxis]
        else:
            Xbeta_new = jnp.einsum("ijk,k->ij", X_jax, beta_new)
            r_new = y_jax - Xbeta_new
        ytilde_new = transform_to_eigenbasis_jax(r_new, V_jax, VkronV_jax)

        def log_density_rho_d(rho_d_prop):
            return kf_log_likelihood_jax(
                rho_d_prop,
                rho_o,
                gamma_new,
                sigma2_u_new,
                sigma2_y_new,
                eigs_W_jax,
                ytilde_new,
            )

        rho_d_new = _slice_sample_1d_jax(
            log_density_rho_d,
            rho_d,
            rho_lower_jax,
            rho_upper_jax,
            key_rho_d,
            w=slice_w_rho,
            max_steps_out=slice_max_steps_out,
            max_shrink=slice_max_shrink,
        )

        # ── Block 7: ρ_o (collapsed slice) ──
        def log_density_rho_o(rho_o_prop):
            return kf_log_likelihood_jax(
                rho_d_new,
                rho_o_prop,
                gamma_new,
                sigma2_u_new,
                sigma2_y_new,
                eigs_W_jax,
                ytilde_new,
            )

        rho_o_new = _slice_sample_1d_jax(
            log_density_rho_o,
            rho_o,
            rho_lower_jax,
            rho_upper_jax,
            key_rho_o,
            w=slice_w_rho,
            max_steps_out=slice_max_steps_out,
            max_shrink=slice_max_shrink,
        )

        # ── Compute log-likelihood at new parameters ──
        log_lik = kf_log_likelihood_jax(
            rho_d_new,
            rho_o_new,
            gamma_new,
            sigma2_u_new,
            sigma2_y_new,
            eigs_W_jax,
            ytilde_new,
        )

        new_state = JAXPanelGaussianState(
            eta=eta,
            beta=beta_new,
            sigma2_u=sigma2_u_new,
            sigma2_y=sigma2_y_new,
            rho_d=rho_d_new,
            rho_o=rho_o_new,
            gamma=gamma_new,
        )

        return new_state, log_lik

    return gibbs_step
