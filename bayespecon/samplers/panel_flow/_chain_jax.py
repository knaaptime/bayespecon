r"""JAX-native chain runner for the Gaussian panel flow Gibbs sampler.

Uses ``jax.lax.scan`` for the iteration loop and ``jax.vmap`` for
parallel chains, keeping the entire sampler inside XLA.

Architecture
------------
1. ``_make_chain_runner()`` builds a closure that binds data and
   precomputed constants, returning a JIT-compiled chain function.
2. The chain function uses ``jax.lax.scan`` to iterate the Gibbs step.
3. ``run_gaussian_panel_flow_chain_jax()`` handles setup, runs chains
   (optionally in parallel via ``jax.vmap``), and converts the output
   to :class:`~bayespecon.samplers.panel_flow._state.PanelGaussianTrace`.

See Also
--------
bayespecon.samplers.panel_flow._chain
    Numpy-based chain runner (reference implementation).
bayespecon.samplers.negbin._jax
    Full-JIT NB Gibbs sampler with the same pattern.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class _ChainOutput(NamedTuple):
    """Raw output from a single JAX chain scan."""

    beta: jax.Array  # (n_draws, k)
    sigma2_u: jax.Array  # (n_draws,)
    sigma2_y: jax.Array  # (n_draws,)
    rho_d: jax.Array  # (n_draws,)
    rho_o: jax.Array  # (n_draws,)
    gamma: jax.Array  # (n_draws,)
    log_lik: jax.Array  # (n_draws,)


def _make_chain_runner(gibbs_step, n_warmup, n_draws):
    """Build a JIT-compiled chain runner.

    Parameters
    ----------
    gibbs_step : callable
        JIT-compiled Gibbs step from ``_make_gibbs_step_with_data()``.
    n_warmup : int
        Number of warmup iterations (discarded).
    n_draws : int
        Number of post-warmup draws to keep.

    Returns
    -------
    run_chain : callable
        JIT-compiled function with signature::

            run_chain(init_state, key) -> _ChainOutput
    """

    def run_chain(init_state, key):
        """Run a single chain: warmup + sampling.

        Uses two separate ``jax.lax.scan`` calls — one for warmup
        (output discarded) and one for sampling (output collected).
        """
        # Split keys for warmup and sampling
        key_warmup, key_sample = jax.random.split(key)
        warmup_keys = jax.random.split(key_warmup, n_warmup)
        sample_keys = jax.random.split(key_sample, n_draws)

        # ── Warmup ──
        def warmup_step(state, key):
            new_state, _ = gibbs_step(state, key)
            return new_state, None

        final_warmup_state, _ = jax.lax.scan(warmup_step, init_state, warmup_keys)

        # ── Sampling ──
        def sample_step(state, key):
            new_state, log_lik = gibbs_step(state, key)
            return new_state, (
                new_state.beta,
                new_state.sigma2_u,
                new_state.sigma2_y,
                new_state.rho_d,
                new_state.rho_o,
                new_state.gamma,
                log_lik,
            )

        final_state, chain_output = jax.lax.scan(
            sample_step, final_warmup_state, sample_keys
        )

        return _ChainOutput(*chain_output)

    return run_chain


def run_gaussian_panel_flow_chain_jax(
    y,
    W,
    X,
    n_draws=2000,
    n_warmup=1000,
    *,
    priors=None,
    rho_init=0.0,
    gamma_init=0.5,
    chains=4,
    seed=42,
    VkronV_threshold=500,
    store_eta=False,
    eta_thin=1,
):
    r"""Run the Gaussian panel flow Gibbs sampler using JAX.

    This is the JAX-native entry point that uses ``@eqx.filter_jit``
    compiled Gibbs steps and ``jax.lax.scan`` for iteration loops.
    Multiple chains are run in parallel via ``jax.vmap``.

    Parameters
    ----------
    y : array_like of shape (n, n, T) or (n², T)
        Observed flow matrix over time.
    W : array_like of shape (n, n)
        Row-standardised spatial weights matrix.
    X : array_like of shape (n², k) or (n², T, k)
        Covariate array. If time-invariant, shape (n², k).
    n_draws : int
        Number of post-warmup draws per chain.
    n_warmup : int
        Number of warmup iterations per chain.
    priors : PanelGaussianPriors, optional
        Prior hyperparameters. Defaults to ``PanelGaussianPriors()``.
    rho_init : float
        Initial value for both ρ_d and ρ_o.
    gamma_init : float
        Initial value for γ.
    chains : int
        Number of parallel chains.
    seed : int
        Random seed.
    VkronV_threshold : int
        If n² > threshold, do not form (V⊗V) explicitly.
    store_eta : bool
        Whether to store η draws (expensive in memory).
    eta_thin : int
        Thinning interval for η draws.

    Returns
    -------
    trace : PanelGaussianTrace
        Collected posterior samples.
    """
    from ._blocks_gaussian_jax import _make_gibbs_step_with_data
    from ._chain import _validate_and_reshape_y
    from ._state import PanelGaussianPriors, PanelGaussianTrace
    from ._state_jax import JAXPanelGaussianState

    jax.config.update("jax_enable_x64", True)

    # ── Validate and prepare data ──
    y = np.asarray(y, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    n = W.shape[0]
    y = _validate_and_reshape_y(y, n)
    T = y.shape[1]
    n2 = n * n

    # Determine if X is time-invariant
    if X.ndim == 2:
        time_invariant_X = True
        k = X.shape[1]
    else:
        time_invariant_X = False
        k = X.shape[2]

    # Priors
    if priors is None:
        priors = PanelGaussianPriors()
    beta_prior_mean = np.zeros(k, dtype=np.float64)
    beta_prior_prec = np.eye(k, dtype=np.float64) / priors.beta_sigma**2

    # ── Eigenbasis of W ──
    eigs_W, V = np.linalg.eigh(W)

    # ── (V⊗V) ──
    VkronV = None
    if n2 <= VkronV_threshold:
        VkronV = np.kron(V, V)

    # ── Precompute XtX if time-invariant ──
    XtX = X.T @ X if time_invariant_X else None

    # ── Convert to JAX arrays ──
    y_jax = jnp.array(y)
    eigs_W_jax = jnp.array(eigs_W)
    V_jax = jnp.array(V)
    VkronV_jax = jnp.array(VkronV) if VkronV is not None else None
    X_jax = jnp.array(X)
    beta_prior_mean_jax = jnp.array(beta_prior_mean)
    beta_prior_prec_jax = jnp.array(beta_prior_prec)
    XtX_jax = jnp.array(XtX) if XtX is not None else None

    # ── Stability bounds for ρ ──
    rho_lower = 1.0 / np.min(eigs_W[eigs_W < 0]) if np.any(eigs_W < 0) else -1.0
    rho_upper = 1.0 / np.max(eigs_W[eigs_W > 0]) if np.any(eigs_W > 0) else 1.0

    # ── Build Gibbs step ──
    gibbs_step = _make_gibbs_step_with_data(
        y_jax=y_jax,
        X_jax=X_jax,
        eigs_W_jax=eigs_W_jax,
        V_jax=V_jax,
        VkronV_jax=VkronV_jax,
        n=n,
        T=T,
        k=k,
        beta_prior_mean=beta_prior_mean_jax,
        beta_prior_prec=beta_prior_prec_jax,
        a_u=priors.sigma2_alpha,
        b_u=priors.sigma2_beta,
        a_y=priors.sigma2_y_alpha,
        b_y=priors.sigma2_y_beta,
        gamma_prior_mean=0.0,
        gamma_prior_var=priors.gamma_prior_var,
        rho_lower=rho_lower,
        rho_upper=rho_upper,
        time_invariant_X=time_invariant_X,
        XtX_jax=XtX_jax,
    )

    # ── Build chain runner ──
    run_chain = _make_chain_runner(gibbs_step, n_warmup, n_draws)

    # ── Initial state ──
    def make_init_state(seed_offset=0):
        return JAXPanelGaussianState(
            eta=jnp.zeros((n2, T)),
            beta=jnp.zeros(k),
            sigma2_u=jnp.float64(1.0),
            sigma2_y=jnp.float64(1.0),
            rho_d=jnp.float64(rho_init),
            rho_o=jnp.float64(rho_init),
            gamma=jnp.float64(gamma_init),
        )

    # ── Run chains ──
    master_key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(master_key, chains)
    init_states = [make_init_state(i) for i in range(chains)]

    # Run chains (sequentially for now; vmap requires careful PyTree handling)
    chain_outputs = []
    for i in range(chains):
        out = run_chain(init_states[i], chain_keys[i])
        chain_outputs.append(out)

    # ── Convert to PanelGaussianTrace ──
    all_beta = np.stack([np.asarray(out.beta) for out in chain_outputs])
    all_sigma2_u = np.stack([np.asarray(out.sigma2_u) for out in chain_outputs])
    all_sigma2_y = np.stack([np.asarray(out.sigma2_y) for out in chain_outputs])
    all_rho_d = np.stack([np.asarray(out.rho_d) for out in chain_outputs])
    all_rho_o = np.stack([np.asarray(out.rho_o) for out in chain_outputs])
    all_gamma = np.stack([np.asarray(out.gamma) for out in chain_outputs])
    all_log_lik = np.stack([np.asarray(out.log_lik) for out in chain_outputs])

    return PanelGaussianTrace(
        beta=all_beta,  # (chains, n_draws, k)
        sigma2_u=all_sigma2_u,  # (chains, n_draws)
        sigma2_y=all_sigma2_y,
        rho_d=all_rho_d,
        rho_o=all_rho_o,
        gamma=all_gamma,
        loglik=all_log_lik,
        eta=None,  # Not stored in JAX path for now
    )
