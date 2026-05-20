"""JAX-accelerated slice sampler for 1-D distributions.

Implements Neal (2003) slice sampling as a pure JAX function, enabling
JIT compilation and eliminating Python dispatch overhead when the
log-density is also a JAX function.

The key advantage over the Python slice sampler is that the entire
stepping-out and shrinkage loop runs inside a single XLA kernel,
with no Python→JAX dispatch between log-density evaluations.

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def jax_slice_sample_1d(
    log_density,
    x0,
    lower,
    upper,
    *,
    key,
    w=1.0,
    max_steps_out=50,
    max_shrink_iters=200,
):
    """Draw one sample from a univariate distribution via slice sampling.

    JAX-accelerated version of ``slice_sample_1d`` that runs the entire
    stepping-out and shrinkage loop inside a single JIT-compiled function.

    Parameters
    ----------
    log_density : callable
        Log-density function that takes a JAX scalar and returns a JAX
        scalar. Must be JIT-compatible (no Python side effects).
    x0 : float or jax.numpy scalar
        Current state (starting point).
    lower, upper : float or jax.numpy scalar
        Support bounds for the variable.
    key : jax.random.PRNGKey
        JAX random key for reproducibility.
    w : float, default 1.0
        Initial step-out width for interval expansion.
    max_steps_out : int, default 50
        Maximum number of stepping-out iterations per side.
    max_shrink_iters : int, default 200
        Maximum number of shrinkage iterations (safety limit).

    Returns
    -------
    x_new : jax.numpy scalar
        New sample point.
    log_density_new : jax.numpy scalar
        Log-density evaluated at ``x_new``.
    """
    # Split key for all random draws needed
    key, key_u, key_Lu, key_Ru = jax.random.split(key, 4)

    # Evaluate log-density at current point
    log_y0 = log_density(x0)

    # Draw vertical level: log(u) where u ~ Uniform(0, f(x0))
    log_u = log_y0 + jnp.log(jax.random.uniform(key_u))

    # --- Stepping out ---
    # Initialise interval [L, R] around x0
    u_L = jax.random.uniform(key_Lu)
    L = jnp.maximum(x0 - u_L * w, lower)
    R = jnp.minimum(L + w, upper)

    # Step out left: expand L until log_density(L) <= log_u or L hits lower bound
    def _step_left_cond(carry):
        L, ld_L, i = carry
        return (ld_L > log_u) & (i < max_steps_out) & (L > lower)

    def _step_left_body(carry):
        L, ld_L, i = carry
        L_new = jnp.maximum(L - w, lower)
        ld_L_new = log_density(L_new)
        return L_new, ld_L_new, i + 1

    L, _, _ = jax.lax.while_loop(
        _step_left_cond,
        _step_left_body,
        (L, log_density(L), jnp.array(0)),
    )

    # Step out right: expand R until log_density(R) <= log_u or R hits upper bound
    def _step_right_cond(carry):
        R, ld_R, i = carry
        return (ld_R > log_u) & (i < max_steps_out) & (R < upper)

    def _step_right_body(carry):
        R, ld_R, i = carry
        R_new = jnp.minimum(R + w, upper)
        ld_R_new = log_density(R_new)
        return R_new, ld_R_new, i + 1

    R, _, _ = jax.lax.while_loop(
        _step_right_cond,
        _step_right_body,
        (R, log_density(R), jnp.array(0)),
    )

    # --- Shrinkage ---
    # Sample from [L, R] and shrink until we find a point above log_u.
    # Start with ld_init = -inf so the loop always executes at least once.
    # Use fold_in for deterministic key splitting per iteration.
    def _shrink_cond(carry):
        L, R, x_new, ld_new, i = carry
        accepted = ld_new > log_u
        collapsed = (R - L) < 1e-15
        return (~accepted) & (~collapsed) & (i < max_shrink_iters)

    def _shrink_body(carry):
        L, R, x_new, ld_new, i = carry
        # Generate proposal using fold_in for deterministic key splitting
        key_i = jax.random.fold_in(key, i)
        x_prop = L + jax.random.uniform(key_i) * (R - L)
        ld_prop = log_density(x_prop)

        # Accept if above slice
        accepted = ld_prop > log_u

        # Shrink interval
        L_new = jnp.where(x_prop < x0, x_prop, L)
        R_new = jnp.where(x_prop >= x0, x_prop, R)

        # Update if accepted
        x_out = jnp.where(accepted, x_prop, x_new)
        ld_out = jnp.where(accepted, ld_prop, ld_new)

        return L_new, R_new, x_out, ld_out, i + 1

    # Initial values for shrinkage: start with -inf log-density so the
    # loop always executes at least once (proposing a new point from [L,R]).
    x_init = x0
    ld_init = jnp.float64(-jnp.inf)

    # Run shrinkage loop
    _, _, x_new, ld_new, _ = jax.lax.while_loop(
        _shrink_cond,
        _shrink_body,
        (L, R, x_init, ld_init, jnp.array(0)),
    )

    return x_new, ld_new