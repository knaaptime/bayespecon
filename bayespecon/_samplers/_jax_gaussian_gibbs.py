r"""JAX-accelerated full-JIT Gibbs sampler for Gaussian spatial models.

Composes the 3-block Gibbs step (β, σ², ρ/λ) into a single
``@eqx.filter_jit``-compiled function, eliminating Python→JAX
dispatch overhead entirely.

Architecture
------------
The sampler uses:

- **β draw**: Conjugate normal via Cholesky factorisation.
  O(k³) but fast for k ≤ ~2000.
- **σ² draw**: Conjugate inverse-Gamma (direct, no solve needed).
- **ρ/λ draw**: Metropolis-adjusted Langevin algorithm (MALA) with
  JAX-native logdet ``log|I-ρW|`` (eigenvalue, Chebyshev, or trace
  polynomial).  The proposal uses JAX autodiff for the exact gradient
  of the collapsed (SAR) or conditional (SEM) log-density.

For SAR/SDM, the ρ log-density is *collapsed* (β and σ² integrated out):

.. math::

    \log p(\rho \mid y) = \log|I - \rho W|
    - \frac{n-k}{2}\log\text{RSS}(\rho) + \text{const}

where RSS(ρ) uses the Woodbury form:

.. math::

    \text{RSS}(\rho) = r^\top r - (X^\top r)^\top (X^\top X)^{-1}(X^\top r),
    \quad r = y - \rho W y

For SEM/SDEM, the λ log-density is *un-collapsed* (conditional on
current β and σ²):

.. math::

    \log p(\lambda \mid \beta, \sigma^2, y) = \log|I - \lambda W|
    - \frac{1}{2\sigma^2}\|(I - \lambda W)(y - X\beta)\|^2 + \text{const}

MALA proposal
~~~~~~~~~~~~~
The MALA update for ρ/λ uses ``jax.value_and_grad`` on the log-density:

.. math::

    \rho^* = \rho + \frac{\varepsilon^2}{2}\,\nabla\log p(\rho\mid\cdot)
    + \varepsilon\,z, \qquad z\sim\mathcal N(0,1)

The Metropolis–Hastings acceptance ratio includes the asymmetric
proposal density correction:

.. math::

    \alpha_{\text{MALA}} = \frac{p(\rho^*\mid\cdot)\,q(\rho\mid\rho^*)}
    {p(\rho\mid\cdot)\,q(\rho^*\mid\rho)},

where the forward and reverse proposal densities are Gaussian with
mean shifted by the gradient drift.

Limitations
-----------
- O(k³) dense Cholesky limits scalability to k ≤ ~2000.
- For SEM/SDEM, the un-collapsed λ log-density may mix more slowly
  than a collapsed version, but the penalty is small for a scalar
  parameter.

References
----------
Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of
Langevin distributions and their discrete approximations.
*Bernoulli*, 2(4), 341–363.

LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial
Econometrics*. CRC Press.
"""

from __future__ import annotations

import numpy as np


def _check_jax_available() -> None:
    """Raise ImportError if JAX or equinox is not installed."""
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        raise ImportError(
            "JAX is required for the full-JIT Gaussian Gibbs sampler. "
            "Install with: pip install jax"
        )
    if importlib.util.find_spec("equinox") is None:
        raise ImportError(
            "equinox is required for the full-JIT Gaussian Gibbs sampler. "
            "Install with: pip install equinox"
        )


# ---------------------------------------------------------------------------
# JAX Gaussian Gibbs state (equinox Module)
# ---------------------------------------------------------------------------


def _make_jax_gaussian_gibbs_state():
    """Create the JAXGaussianGibbsState equinox Module class.

    Returns the class (not an instance) so that JAX/equinox imports
    are deferred until first use.
    """
    import equinox as eqx
    import jax.numpy as jnp

    class JAXGaussianGibbsState(eqx.Module):
        """Immutable state for the JAX Gaussian Gibbs sampler.

        Parameters
        ----------
        beta : jax.Array of shape (k,)
            Regression coefficients.
        sigma2 : jax.Array
            Residual variance σ² (scalar).
        rho : jax.Array
            Spatial autoregressive parameter (ρ for SAR/SDM,
            λ for SEM/SDEM) (scalar).
        """

        beta: jnp.ndarray
        sigma2: jnp.ndarray
        rho: jnp.ndarray

    return JAXGaussianGibbsState


# ---------------------------------------------------------------------------
# JIT-compiled Gibbs step builder
# ---------------------------------------------------------------------------


def _make_gaussian_gibbs_step(
    y_jax,
    X_jax,
    Wy_jax,
    W_dense_jax,
    n,
    k,
    logdet_jax,
    XtX_jax,
    XtX_inv_jax,
    priors,
    model_type: str,
    mala_step_size: float = 0.05,
    use_mala: bool = True,
):
    """Build a JIT-compiled Gaussian Gibbs step with data bound into the closure.

    Creates a ``@eqx.filter_jit``-compiled function that performs one
    complete 3-block Gibbs sweep (β, σ², ρ/λ MALA/RW-MH) in a single
    XLA kernel call, eliminating all Python→JAX dispatch overhead.

    Parameters
    ----------
    y_jax : jax.numpy.ndarray of shape (n,)
        Response vector (JAX array).
    X_jax : jax.numpy.ndarray of shape (n, k)
        Design matrix (JAX array).
    Wy_jax : jax.numpy.ndarray of shape (n,) or None
        W @ y (precomputed, for SAR/SDM).  None for SEM/SDEM.
    W_dense_jax : jax.numpy.ndarray of shape (n, n) or None
        Dense W matrix (for SEM/SDEM residual filtering).  None for
        SAR/SDM.
    n : int
        Number of spatial units.
    k : int
        Number of regression coefficients.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.  Built by :func:`~bayespecon.logdet.make_logdet_jax_fn`.
    XtX_jax : jax.numpy.ndarray of shape (k, k)
        Precomputed X^T X.
    XtX_inv_jax : jax.numpy.ndarray of shape (k, k)
        Precomputed (X^T X)^{-1}.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    mala_step_size : float, default 0.05
        Step size for MALA proposal (or RW-MH proposal sd).
    use_mala : bool, default True
        If True, use MALA (gradient-guided proposals) for the ρ/λ update.
        If False, use random-walk Metropolis–Hastings.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key) -> (new_state, accept)

        where ``state`` is a ``JAXGaussianGibbsState`` and ``key`` is a
        JAX PRNG key.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    JAXGaussianGibbsState = _make_jax_gaussian_gibbs_state()

    is_sar = model_type in ("sar", "sdm")

    # Convert constants to JAX arrays
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    sigma_sigma_jax = jnp.float64(priors.sigma_sigma)
    rho_lower_jax = jnp.float64(priors.rho_lower)
    rho_upper_jax = jnp.float64(priors.rho_upper)
    mala_eps = jnp.float64(mala_step_size)

    # Prior precision for beta
    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)

    @eqx.filter_jit
    def gibbs_step(state, key):
        """One complete 3-block Gibbs sweep: β → σ² → ρ/λ (MALA).

        Parameters
        ----------
        state : JAXGaussianGibbsState
            Current state.
        key : jax.random.PRNGKey
            JAX random key.

        Returns
        -------
        new_state : JAXGaussianGibbsState
            Updated state.
        accept : bool
            Whether the MALA/MH step for ρ/λ was accepted.
        """
        sigma2 = state.sigma2
        rho = state.rho  # holds λ for SEM/SDEM

        key_beta, key_sigma2, key_rho = jax.random.split(key, 3)

        # ── Block 1: β | ρ, σ², y — conjugate normal ──
        if is_sar:
            r = y_jax - rho * Wy_jax
        else:
            r = y_jax  # SEM: residuals are just y (no ρWy term)

        Sigma_beta_inv = XtX_jax / sigma2 + beta_prior_prec
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_jax.T @ r / sigma2
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = jnp.linalg.solve(L_beta.T, jnp.linalg.solve(L_beta, rhs_beta))
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + jnp.linalg.solve(L_beta.T, z_beta)

        # ── Block 2: σ² | β, ρ/λ, y — conjugate inverse-Gamma ──
        if is_sar:
            resid = y_jax - rho * Wy_jax - X_jax @ beta_new
        else:
            resid_raw = y_jax - X_jax @ beta_new
            resid = resid_raw - rho * (W_dense_jax @ resid_raw)

        a_post = jnp.float64(n / 2.0 + 1.0)  # from HalfNormal prior
        b_post = jnp.float64(resid @ resid / 2.0 + sigma_sigma_jax**2 / 2.0)
        sigma2_inv = jax.random.gamma(key_sigma2, a_post) / b_post
        sigma2_new = jnp.maximum(1.0 / sigma2_inv, 1e-10)

        # ── Block 3: ρ/λ — MALA or RW-MH ──
        key_rho_proposal, key_rho_accept = jax.random.split(key_rho)

        if is_sar:
            # Collapsed log-density: log p(ρ | y) = log|I-ρW| - (n-k)/2 * log RSS(ρ)
            def log_density_spatial(param_val):
                r = y_jax - param_val * Wy_jax
                Xtr = X_jax.T @ r
                rss = r @ r - Xtr @ XtX_inv_jax @ Xtr
                logdet = logdet_jax(param_val)
                log_prior = jnp.where(
                    (param_val >= rho_lower_jax) & (param_val <= rho_upper_jax),
                    0.0,
                    -jnp.inf,
                )
                return logdet - 0.5 * (n - k) * jnp.log(rss) + log_prior

            log_density_fn = log_density_spatial
        else:
            # Un-collapsed conditional: log p(λ | β, σ², y)
            def log_density_spatial(param_val):
                resid_raw = y_jax - X_jax @ beta_new
                eps = resid_raw - param_val * (W_dense_jax @ resid_raw)
                ss = eps @ eps
                logdet = logdet_jax(param_val)
                log_prior = jnp.where(
                    (param_val >= rho_lower_jax) & (param_val <= rho_upper_jax),
                    0.0,
                    -jnp.inf,
                )
                return logdet - 0.5 * ss / sigma2_new + log_prior

            log_density_fn = log_density_spatial

        if use_mala:
            # Gradient + value via JAX autodiff
            val_and_grad = jax.value_and_grad(log_density_fn)
            log_density_current, g_current = val_and_grad(rho)

            # MALA proposal: ρ* = ρ + (ε²/2) ∇log p(ρ) + ε z
            eps = mala_eps
            drift = (eps**2 / 2.0) * g_current
            noise = eps * jax.random.normal(key_rho_proposal, dtype=jnp.float64)
            rho_proposed = rho + drift + noise
            rho_proposed = jnp.clip(rho_proposed, rho_lower_jax, rho_upper_jax)

            # MALA acceptance ratio includes proposal density
            log_density_proposed, g_proposed = val_and_grad(rho_proposed)
            log_p_fwd = (
                -0.5 * ((rho_proposed - rho - (eps**2 / 2.0) * g_current) / eps) ** 2
            )
            log_p_rev = (
                -0.5 * ((rho - rho_proposed - (eps**2 / 2.0) * g_proposed) / eps) ** 2
            )

            log_alpha = (
                log_density_proposed - log_density_current + log_p_rev - log_p_fwd
            )
        else:
            # RW-MH proposal
            rho_proposed = rho + mala_eps * jax.random.normal(
                key_rho_proposal, dtype=jnp.float64
            )
            rho_proposed = jnp.clip(rho_proposed, rho_lower_jax, rho_upper_jax)

            log_density_proposed = log_density_fn(rho_proposed)
            log_density_current = log_density_fn(rho)
            log_alpha = log_density_proposed - log_density_current

        u = jax.random.uniform(key_rho_accept, dtype=jnp.float64)
        accept = jnp.log(u) < log_alpha

        rho_new = jnp.where(accept, rho_proposed, rho)
        rho_new = jnp.clip(rho_new, rho_lower_jax, rho_upper_jax)

        new_state = JAXGaussianGibbsState(
            beta=beta_new,
            sigma2=sigma2_new,
            rho=rho_new,
        )
        return new_state, accept

    return gibbs_step


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_chain_jax_gaussian(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    Wy: np.ndarray | None,
    logdet_jax,
    logdet_vec_fn,
    priors,
    init,
    draws: int,
    tune: int,
    thin: int = 1,
    rng=None,
    model_type: str = "sar",
    mala_step_size: float = 0.05,
    use_mala: bool = True,
):
    """Run one chain of the full-JIT JAX Gaussian Gibbs sampler.

    Creates a JIT-compiled Gibbs step function and runs it in a Python
    loop, handling warmup, thinning, MALA step-size adaptation, and
    storage.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Spatial weights matrix.
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed, for SAR/SDM).  None for SEM/SDEM.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.  Built by :func:`~bayespecon.logdet.make_logdet_jax_fn`.
    logdet_vec_fn : callable
        Vectorized numpy logdet callable for post-chain LL computation.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    init : GaussianGibbsState
        Initial state (from OLS warm start).
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int, default 1
        Keep every thin-th draw.
    rng : numpy.random.Generator, optional
        Random state.
    model_type : str, default "sar"
        One of "sar", "sem", "sdm", "sdem".
    mala_step_size : float, default 0.05
        Initial MALA step size (or RW-MH proposal sd).
    use_mala : bool, default True
        If True, use MALA for the ρ/λ update.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho`` (or ``lam``), ``beta``,
        ``sigma``, and ``log_lik``.  Each array has shape
        ``(n_keep, ...)`` where n_keep = draws // thin.
        Also includes ``mh_accept_rate`` (float).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from ._gaussian_loglik import (
        sar_pointwise_loglik_vectorized,
        sem_pointwise_loglik_vectorized,
    )

    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws
    is_sar = model_type in ("sar", "sdm")

    # Convert data to JAX arrays
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)
    XtX_inv_jax = jnp.asarray(np.linalg.inv(X.T @ X), dtype=jnp.float64)

    if is_sar:
        Wy_jax = jnp.asarray(Wy, dtype=jnp.float64)
        W_dense_jax = None
    else:
        Wy_jax = None
        W_dense_jax = jnp.asarray(W_sparse.toarray(), dtype=jnp.float64)

    # Initialize JAX state
    JAXGaussianGibbsState = _make_jax_gaussian_gibbs_state()
    state = JAXGaussianGibbsState(
        beta=jnp.asarray(init.beta, dtype=jnp.float64),
        sigma2=jnp.float64(init.sigma2),
        rho=jnp.float64(init.rho),
    )

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)

    # ── MALA step-size adaptation ──
    adapted_step_size = mala_step_size

    # Build the initial JIT-compiled step function
    gibbs_step = _make_gaussian_gibbs_step(
        y_jax=y_jax,
        X_jax=X_jax,
        Wy_jax=Wy_jax,
        W_dense_jax=W_dense_jax,
        n=n,
        k=k,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        XtX_inv_jax=XtX_inv_jax,
        priors=priors,
        model_type=model_type,
        mala_step_size=adapted_step_size,
        use_mala=use_mala,
    )

    # Warmup the JIT function (first call triggers compilation)
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    _ = gibbs_step(state, warmup_key)

    # Run the chain
    accept_count = 0
    warmup_accept_count = 0
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, accept = gibbs_step(state, step_key)
        accept_count += int(accept)

        # Track warmup acceptance for step-size adaptation
        if i < tune:
            warmup_accept_count += int(accept)

        # At end of warmup, adapt MALA step size and recompile
        if use_mala and i == tune - 1 and tune > 0:
            warmup_rate = warmup_accept_count / tune
            # Simple multiplicative adaptation targeting ~57.4% acceptance
            # (optimal for MALA; see Roberts & Tweedie 1996).
            target_rate = 0.574
            if 0.0 < warmup_rate < 1.0:
                adapt_factor = min(max(target_rate / warmup_rate, 0.5), 2.0)
                adapted_step_size = mala_step_size * adapt_factor
            # Recompile with adapted step size
            gibbs_step = _make_gaussian_gibbs_step(
                y_jax=y_jax,
                X_jax=X_jax,
                Wy_jax=Wy_jax,
                W_dense_jax=W_dense_jax,
                n=n,
                k=k,
                logdet_jax=logdet_jax,
                XtX_jax=XtX_jax,
                XtX_inv_jax=XtX_inv_jax,
                priors=priors,
                model_type=model_type,
                mala_step_size=adapted_step_size,
                use_mala=use_mala,
            )
            key, warmup_key = jax.random.split(key)
            _ = gibbs_step(state, warmup_key)

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = float(state.rho)
                beta_samples[idx] = np.asarray(state.beta)
                sigma_samples[idx] = np.sqrt(float(state.sigma2))

    # ── Post-chain: vectorized pointwise log-likelihood ──
    # Compute after the chain completes using stored posterior draws,
    # avoiding per-draw logdet calls inside the JIT step.
    if is_sar:
        log_lik = sar_pointwise_loglik_vectorized(
            rho_draws=rho_samples,
            beta_draws=beta_samples,
            sigma_draws=sigma_samples,
            y=y,
            X=X,
            Wy=Wy,
            logdet_vec_fn=logdet_vec_fn,
            n=n,
        )
    else:
        log_lik = sem_pointwise_loglik_vectorized(
            lam_draws=rho_samples,
            beta_draws=beta_samples,
            sigma_draws=sigma_samples,
            y=y,
            X=X,
            W_sparse=W_sparse,
            logdet_vec_fn=logdet_vec_fn,
            n=n,
        )

    # Name the spatial parameter appropriately
    param_name = "rho" if is_sar else "lam"
    result = {
        param_name: rho_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "log_lik": log_lik,
        "mh_accept_rate": accept_count / total_iters,
    }

    return result
