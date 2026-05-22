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

For SEM/SDEM, the λ log-density is also *collapsed* (β and σ² integrated out):

.. math::

    \log p(\lambda \mid y) = \log|I - \lambda W|
    - \frac{1}{2}\log|X^{*\top} X^*|
    - \frac{n-k}{2}\log\text{RSS}(\lambda) + \text{const}

where :math:`y^* = (I - \lambda W)y`, :math:`X^* = (I - \lambda W)X`, and
:math:`\text{RSS}(\lambda) = y^{*\top} y^* - y^{*\top} X^* (X^{*\top} X^*)^{-1} X^{*\top} y^*`.
The extra term :math:`-\frac{1}{2}\log|X^{*\top} X^*|` appears because :math:`X^*`
depends on :math:`\lambda` (unlike SAR where :math:`X` is fixed).

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

_JAX_GAUSSIAN_GIBBS_STATE_CLS = None


def _make_jax_gaussian_gibbs_state():
    """Create (or return cached) JAXGaussianGibbsState equinox Module class.

    Returns the class (not an instance) so that JAX/equinox imports
    are deferred until first use.  The class is cached as a module-level
    singleton to ensure that all callers share the **same** Python type,
    which is required by ``jax.lax.scan`` (carry input and output must
    have identical pytree structure).
    """
    global _JAX_GAUSSIAN_GIBBS_STATE_CLS

    if _JAX_GAUSSIAN_GIBBS_STATE_CLS is not None:
        return _JAX_GAUSSIAN_GIBBS_STATE_CLS

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

    _JAX_GAUSSIAN_GIBBS_STATE_CLS = JAXGaussianGibbsState
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
    use_mala: bool = True,
):
    """Build a JIT-compiled Gaussian Gibbs step with data bound into the closure.

    Creates a ``@eqx.filter_jit``-compiled function that performs one
    complete 3-block Gibbs sweep (β, σ², ρ/λ MALA/RW-MH) in a single
    XLA kernel call, eliminating all Python→JAX dispatch overhead.

    The step size ``mala_eps`` is a **runtime argument** to the
    returned function, not a closure variable.  This enables
    ``jax.lax.scan`` (no Python-side recompilation for adaptation)
    and ``jax.vmap`` across chains.

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
    use_mala : bool, default True
        If True, use MALA (gradient-guided proposals) for the ρ/λ update.
        If False, use random-walk Metropolis–Hastings.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key, mala_eps) -> (new_state, accept)

        where ``state`` is a ``JAXGaussianGibbsState``, ``key`` is a
        JAX PRNG key, and ``mala_eps`` is a ``jnp.float64`` step size.
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
    jnp.float64(priors.sigma_sigma)
    rho_lower_jax = jnp.float64(priors.rho_lower)
    rho_upper_jax = jnp.float64(priors.rho_upper)

    # Prior precision for beta
    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)

    @eqx.filter_jit
    def gibbs_step(state, key, mala_eps):
        """One complete 3-block Gibbs sweep: β → σ² → ρ/λ (MALA).

        Parameters
        ----------
        state : JAXGaussianGibbsState
            Current state.
        key : jax.random.PRNGKey
            JAX random key.
        mala_eps : jax.numpy.float64
            Step size for MALA proposal (or RW-MH proposal sd).
            Passed as a runtime argument to avoid recompilation
            when adapting step size.

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
            X_eff = X_jax
            XtX_eff = XtX_jax
        else:
            # SEM: transform y and X by (I - λW)
            r = y_jax - rho * (W_dense_jax @ y_jax)
            X_eff = X_jax - rho * (W_dense_jax @ X_jax)
            XtX_eff = X_eff.T @ X_eff

        Sigma_beta_inv = XtX_eff / sigma2 + beta_prior_prec
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_eff.T @ r / sigma2
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

        # Weakly informative Jeffreys prior: p(σ²) ∝ 1/σ²
        # Approximated as Inv-Γ(ε, ε) with ε = 1e-3
        EPS = jnp.float64(1e-3)
        a_post = jnp.float64(n / 2.0 + EPS)
        b_post = jnp.float64(resid @ resid / 2.0 + EPS)
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
            # Collapsed log-density for SEM/SDEM:
            # log p(λ | y) = log|I-λW| - 0.5*log|X*ᵀX*| - (n-k)/2 * log RSS(λ)
            def log_density_spatial(param_val):
                y_star = y_jax - param_val * (W_dense_jax @ y_jax)
                X_star = X_jax - param_val * (W_dense_jax @ X_jax)
                XtX_star = X_star.T @ X_star
                Xty_star = X_star.T @ y_star
                yty_star = y_star @ y_star

                # RSS = y*ᵀy* - y*ᵀX* (X*ᵀX*)⁻¹ X*ᵀy*
                # Use solve for numerical stability
                XtX_star_inv_Xty = jnp.linalg.solve(XtX_star, Xty_star)
                rss = yty_star - Xty_star @ XtX_star_inv_Xty
                rss = jnp.maximum(rss, 1e-300)

                logdet = logdet_jax(param_val)
                logdet_XtX = jnp.linalg.slogdet(XtX_star)[1]

                log_prior = jnp.where(
                    (param_val >= rho_lower_jax) & (param_val <= rho_upper_jax),
                    0.0,
                    -jnp.inf,
                )
                return (
                    logdet - 0.5 * logdet_XtX - 0.5 * (n - k) * jnp.log(rss) + log_prior
                )

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
# jax.lax.scan-based chain runner
# ---------------------------------------------------------------------------


def _run_chain_jax_gibbs_scanned(
    gibbs_step,
    init_state,
    key,
    n_iters,
    mala_eps,
):
    """Run a single chain using ``jax.lax.scan``.

    All outputs are JAX arrays — no Python side effects.  This
    function is compatible with ``jax.vmap`` for vectorized
    multi-chain execution.

    Parameters
    ----------
    gibbs_step : callable
        JIT-compiled step function with signature
        ``(state, key, mala_eps) -> (new_state, accept)``.
    init_state : JAXGaussianGibbsState
        Initial state.
    key : jax.random.PRNGKey
        JAX random key.
    n_iters : int
        Number of iterations to run.
    mala_eps : jax.numpy.float64
        Step size for MALA/RW-MH proposals.

    Returns
    -------
    final_state : JAXGaussianGibbsState
        State after ``n_iters`` steps.
    rhos : jax.numpy.ndarray of shape (n_iters,)
        Trace of ρ/λ values.
    betas : jax.numpy.ndarray of shape (n_iters, k)
        Trace of β values.
    sigma2s : jax.numpy.ndarray of shape (n_iters,)
        Trace of σ² values.
    accept_rate : jax.numpy.float64
        Fraction of MALA/MH steps accepted.
    """
    import jax
    import jax.numpy as jnp

    def scan_body(carry, _):
        state, key, accept_sum = carry
        key, step_key = jax.random.split(key)
        state, accept = gibbs_step(state, step_key, mala_eps)
        return (
            (state, key, accept_sum + accept),
            (state.rho, state.beta, state.sigma2, accept),
        )

    (final_state, _, total_accept), (rhos, betas, sigma2s, accepts) = jax.lax.scan(
        scan_body,
        (init_state, key, jnp.float64(0.0)),
        None,
        length=n_iters,
    )

    accept_rate = total_accept / jnp.float64(n_iters)
    return final_state, rhos, betas, sigma2s, accept_rate


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
    progressbar: bool = True,
    chain_id: int = 0,
    progress_manager: object | None = None,
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

    # Build the JIT-compiled step function (mala_eps is a runtime arg)
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
        use_mala=use_mala,
    )

    key = jax.random.PRNGKey(rng.integers(2**31))

    # ── Phase 1: Warmup via jax.lax.scan ──
    if tune > 0:
        key, warmup_key = jax.random.split(key)
        state, _, _, _, warmup_accept_rate = _run_chain_jax_gibbs_scanned(
            gibbs_step,
            state,
            warmup_key,
            tune,
            jnp.float64(adapted_step_size),
        )

        # Adapt MALA step size in Python (no recompilation needed)
        if use_mala:
            warmup_rate = float(warmup_accept_rate)
            target_rate = 0.574
            if 0.0 < warmup_rate < 1.0:
                adapt_factor = min(max(target_rate / warmup_rate, 0.5), 2.0)
                adapted_step_size = mala_step_size * adapt_factor

        # Update progress bar for warmup phase
        if progress_manager is not None:
            for i in range(tune):
                progress_manager.update(chain_id, i, tuning=True, accept=None)
            progress_manager.refresh()

    # ── Phase 2: Post-warmup draws via jax.lax.scan ──
    key, draw_key = jax.random.split(key)
    state, rhos, betas, sigma2s, draw_accept_rate = _run_chain_jax_gibbs_scanned(
        gibbs_step,
        state,
        draw_key,
        draws,
        jnp.float64(adapted_step_size),
    )

    # Apply thinning and convert to NumPy
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    rho_samples = np.asarray(rhos[thin_slice])
    beta_samples = np.asarray(betas[thin_slice])
    sigma_samples = np.sqrt(np.asarray(sigma2s[thin_slice]))

    # Update progress bar for draw phase
    if progress_manager is not None:
        for i in range(tune, total_iters):
            progress_manager.update(chain_id, i, tuning=False, accept=None)
        # Set the aggregate MALA/MH accept rate for this chain
        progress_manager.set_accept_rate(chain_id, float(draw_accept_rate))
        progress_manager.refresh()

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
        "mh_accept_rate": float(draw_accept_rate),
    }

    return result


# ---------------------------------------------------------------------------
# Vectorized multi-chain runner (jax.vmap)
# ---------------------------------------------------------------------------


def run_chains_jax_gibbs_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    Wy: np.ndarray | None,
    logdet_jax,
    logdet_vec_fn,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    model_type: str = "sar",
    mala_step_size: float = 0.05,
    use_mala: bool = True,
    progressbar: bool = True,
) -> list[dict]:
    """Run multiple JAX Gibbs chains via ``jax.vmap``.

    All chains run in parallel on a single device via vectorized
    map.  This avoids Python multiprocessing entirely and is the
    recommended way to run multiple JAX chains.

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
        JAX-native logdet function.
    logdet_vec_fn : callable
        Vectorized numpy logdet callable for post-chain LL computation.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    inits : list of GaussianGibbsState
        Per-chain initial states.
    draws : int
        Number of post-warmup draws per chain.
    tune : int
        Number of warmup draws per chain.
    thin : int, default 1
        Keep every thin-th draw.
    jax_seeds : list of int, optional
        Per-chain JAX PRNG seeds.
    model_type : str, default "sar"
        One of "sar", "sem", "sdm", "sdem".
    mala_step_size : float, default 0.05
        Initial MALA step size.
    use_mala : bool, default True
        If True, use MALA for the ρ/λ update.
    progressbar : bool, default True
        Show per-chain progress bars.

    Returns
    -------
    list of dict
        One dict per chain, each containing posterior sample arrays
        and ``mh_accept_rate``.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from ._gaussian_loglik import (
        sar_pointwise_loglik_vectorized,
        sem_pointwise_loglik_vectorized,
    )

    chains = len(inits)
    n, k = X.shape
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

    # Build the JIT-compiled step function (shared across all chains)
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
        use_mala=use_mala,
    )

    JAXGaussianGibbsState = _make_jax_gaussian_gibbs_state()

    # Convert NumPy initial states to JAX states, then batch into a
    # vmappable pytree by stacking each leaf.
    jax_inits = [
        JAXGaussianGibbsState(
            beta=jnp.asarray(init.beta, dtype=jnp.float64),
            sigma2=jnp.float64(init.sigma2),
            rho=jnp.float64(init.rho),
        )
        for init in inits
    ]
    init_states = jax.tree.map(lambda *a: jnp.stack(a), *jax_inits)

    # Batch PRNG keys
    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(jax_seeds[0])
    keys = jax.random.split(master_key, chains)

    from ._progress import GibbsProgressBarManager

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type=model_type,
    ) as pm:
        # Record start times for all chains (they run simultaneously via vmap)
        if pm is not None:
            for c in range(chains):
                pm.start_chain(c)

        # ── Phase 1: Warmup via vmap ──
        if tune > 0:
            warmup_keys = keys

            def run_warmup(state, key):
                return _run_chain_jax_gibbs_scanned(
                    gibbs_step, state, key, tune, jnp.float64(mala_step_size)
                )

            final_states, _, _, _, warmup_rates = jax.vmap(run_warmup)(
                init_states, warmup_keys
            )

            # Adapt step size per-chain using individual warmup acceptance rates
            if use_mala:
                target_rate = 0.574
                # Per-chain adaptation: each chain gets its own step size
                # based on its own acceptance rate during warmup.
                # Chains with acceptance rate outside (0, 1) keep the
                # initial step size (no adaptation).
                safe_rates = jnp.where(
                    (warmup_rates > 0) & (warmup_rates < 1),
                    warmup_rates,
                    jnp.float64(target_rate),
                )
                adapt_factors = jnp.clip(target_rate / safe_rates, 0.5, 2.0)
                adapted_step_sizes = jnp.float64(mala_step_size) * adapt_factors
            else:
                adapted_step_sizes = jnp.full(chains, jnp.float64(mala_step_size))

            # Update progress bar for warmup phase (all chains at once)
            if pm is not None:
                for c in range(chains):
                    for i in range(tune):
                        pm.update(c, i, tuning=True, accept=None)
                pm.refresh()
        else:
            final_states = init_states
            adapted_step_sizes = jnp.full(chains, jnp.float64(mala_step_size))

        # ── Phase 2: Post-warmup draws via vmap ──
        draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

        def run_draws(state, key, step_size):
            return _run_chain_jax_gibbs_scanned(
                gibbs_step, state, key, draws, step_size
            )

        _, rhos, betas, sigma2s, accept_rates = jax.vmap(run_draws)(
            final_states, draw_keys, adapted_step_sizes
        )

        # Update progress bar for draw phase (all chains at once)
        if pm is not None:
            for c in range(chains):
                for i in range(tune, tune + draws):
                    pm.update(c, i, tuning=False, accept=None)
                # Set the aggregate MALA/MH accept rate for this chain
                pm.set_accept_rate(c, float(accept_rates[c]))
            pm.refresh()

    # Convert to NumPy and assemble per-chain results
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        rho_c = np.asarray(rhos[c][thin_slice])
        beta_c = np.asarray(betas[c][thin_slice])
        sigma_c = np.sqrt(np.asarray(sigma2s[c][thin_slice]))

        # Pointwise log-likelihood
        if is_sar:
            log_lik = sar_pointwise_loglik_vectorized(
                rho_draws=rho_c,
                beta_draws=beta_c,
                sigma_draws=sigma_c,
                y=y,
                X=X,
                Wy=Wy,
                logdet_vec_fn=logdet_vec_fn,
                n=n,
            )
        else:
            log_lik = sem_pointwise_loglik_vectorized(
                lam_draws=rho_c,
                beta_draws=beta_c,
                sigma_draws=sigma_c,
                y=y,
                X=X,
                W_sparse=W_sparse,
                logdet_vec_fn=logdet_vec_fn,
                n=n,
            )

        param_name = "rho" if is_sar else "lam"
        results.append(
            {
                param_name: rho_c,
                "beta": beta_c,
                "sigma": sigma_c,
                "log_lik": log_lik,
                "mh_accept_rate": float(accept_rates[c]),
            }
        )

    return results
