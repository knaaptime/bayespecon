r"""JAX-accelerated full-JIT Gibbs sampler for SAR-logit.

Composes all Gibbs blocks (PG ω, η, β, ρ slice) into a single
``@jax.jit``-compiled function, eliminating Python→JAX dispatch overhead
entirely.

Architecture
------------
The sampler uses:

- **PG sampling**: Alternating-series method with K terms using
  ``jax.random.gamma`` for Gamma(1, 1) draws.  Since h = 1 for binary
  logit, the Devroye method would also be valid, but the alternating-
  series method is fully JIT-compatible.
- **η and β draws**: Dense Cholesky factorisation via ``jnp.linalg.cholesky``
  and ``jnp.linalg.solve``.  O(n³) but fast for n ≤ ~2000.
- **ρ draw**: Neal's stepping-out slice sampler with Lanczos-based
  ``log|P|`` estimation.  A fixed Lanczos key is used for all
  log-density evaluations within a single slice step.

Limitations
-----------
- O(n³) dense Cholesky limits scalability to n ≤ ~2000.
- PG sum-of-exponentials has ~2% mean bias (acceptable for MCMC).
- Lanczos log|P| estimation is stochastic but uses a fixed key within
  each slice step for consistency.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.

Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

import numpy as np

from ._core import JAXLogitGibbsState


def _check_jax_available() -> None:
    """Raise ImportError if JAX or equinox is not installed."""
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        raise ImportError(
            "JAX is required for the full-JIT Gibbs sampler. "
            "Install with: pip install jax"
        )
    if importlib.util.find_spec("equinox") is None:
        raise ImportError(
            "equinox is required for the full-JIT Gibbs sampler. "
            "Install with: pip install equinox"
        )


def _make_gibbs_step_with_data(
    y_jax,
    X_jax,
    W_dense_jax,
    n,
    k,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    XtX_jax,
    priors,
    pg_n_terms,
    n_probes,
    lanczos_deg,
):
    """Build a JIT-compiled Gibbs step with data bound into the closure.

    This function creates a ``@jax.jit``-compiled function that performs
    one complete Gibbs sweep (ω, η, β, ρ slice) in a single XLA kernel
    call, eliminating all Python→JAX dispatch overhead.

    Parameters
    ----------
    y_jax : jax.numpy.ndarray of shape (n,)
        Binary response vector (JAX array).
    X_jax : jax.numpy.ndarray of shape (n, k)
        Design matrix (JAX array).
    W_dense_jax : jax.numpy.ndarray of shape (n, n)
        Original row-standardised W matrix (JAX array).
    n : int
        Number of spatial units.
    k : int
        Number of regression coefficients.
    W_sym_dense : jax.numpy.ndarray of shape (n, n)
        Dense (W + W^T).
    WtW_dense : jax.numpy.ndarray of shape (n, n)
        Dense W^T W.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.
    XtX_jax : jax.numpy.ndarray of shape (k, k)
        Precomputed X^T X.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    pg_n_terms : int
        Number of alternating-series terms for the PG approximation.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key) -> (new_state, accept)

        where ``state`` is a :class:`JAXLogitGibbsState`
        and ``key`` is a JAX PRNG key.  ``accept`` is always ``True``
        (slice sampling has no rejection step).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    jax.config.update("jax_enable_x64", True)

    # Convert constants to JAX arrays
    W_sym = jnp.asarray(W_sym_dense, dtype=jnp.float64)
    WtW = jnp.asarray(WtW_dense, dtype=jnp.float64)
    # Prior hyperparameters
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    rho_lower_jax = jnp.float64(priors.rho_lower)
    rho_upper_jax = jnp.float64(priors.rho_upper)

    # Prior precision for beta
    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)

    # PG term indices
    k_idx = jnp.arange(pg_n_terms, dtype=jnp.float64)
    pi = jnp.pi
    pi2 = pi * pi

    # For logit: h = 1 always, kappa = y - 0.5
    kappa = y_jax - 0.5

    @eqx.filter_jit
    def gibbs_step(state, key):
        """One complete Gibbs sweep: ω → η → β → ρ (slice).

        Parameters
        ----------
        state : JAXLogitGibbsState
            Current state.
        key : jax.random.PRNGKey
            JAX random key.

        Returns
        -------
        new_state : JAXLogitGibbsState
            Updated state.
        accept : bool
            Always True (slice sampling has no rejection step).
        """
        eta = state.eta
        beta = state.beta
        rho = state.rho

        key_omega, key_eta, key_beta, key_rho = jax.random.split(key, 4)

        # ── Block 1: ω ~ PG(1, η) — alternating-series with Gamma draws ──
        # For logit: h = 1 always
        z = eta  # tilting parameters
        Z = jnp.abs(z) / 2.0
        denominators = (k_idx + 0.5) ** 2 + (Z[:, None] / pi) ** 2  # (n, K)
        g = jax.random.gamma(
            key_omega,
            jnp.ones((n, pg_n_terms)),  # h = 1 for all observations
            dtype=jnp.float64,
        )  # (n, K) Gamma(1, 1) draws
        pg1 = jnp.sum(g / denominators, axis=1) / (2.0 * pi2)  # (n,)
        omega_new = jnp.maximum(pg1, 1e-6)

        # ── Block 2: η | ω, ρ, β — dense Cholesky solve (σ² = 1) ──
        # P = I + diag(ω) - ρ(W+W^T) + ρ²W^TW  (no 1/σ² scaling)
        P_diag = jnp.ones(n) + omega_new
        P = jnp.diag(P_diag) - rho * W_sym + rho**2 * WtW
        P = P + 1e-6 * jnp.eye(n)  # regularisation for numerical stability

        Xbeta = X_jax @ beta
        # RHS: Xbeta - ρ W'Xbeta + κ  (σ² = 1)
        rhs = Xbeta - rho * (W_dense_jax.T @ Xbeta) + kappa

        L = jnp.linalg.cholesky(P)
        P_inv_rhs = cho_solve((L, True), rhs)
        z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
        eta_new = P_inv_rhs + solve_triangular(L.T, z_eta, lower=False)

        # ── Block 3: β | η, ρ — conjugate normal (σ² = 1) ──
        A_rho_eta = eta_new - rho * W_dense_jax @ eta_new
        # Σ_β⁻¹ = Λ₀⁻¹ + X^TX  (no 1/σ² scaling)
        Sigma_beta_inv = beta_prior_prec + XtX_jax
        # rhs = Λ₀⁻¹μ₀ + X^T A_ρη  (no 1/σ² scaling)
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_jax.T @ A_rho_eta
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = cho_solve((L_beta, True), rhs_beta)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)

        # ── Block 4: ρ — slice sampling (collapsed, η integrated out) ──
        # Uses Neal's stepping-out slice sampler.  log_density is exact
        # (dense Cholesky), so the slice density is deterministic.
        key_rho_slice, key_rho_u, key_rho_L, key_rho_R, key_rho_shrink = (
            jax.random.split(key_rho, 5)
        )

        def log_density_rho(rho_val):
            """Collapsed log-density of ρ (η integrated out, σ² = 1).

            Uses one dense Cholesky factorisation of P_r to obtain both
            log|P_r| (= 2 Σ log diag(L)) and the quadratic form
            rhs' P_r⁻¹ rhs (= ‖L⁻¹ rhs‖²).  This is exact and BLAS-optimised
            — strictly faster than iterative Lanczos+CG for the dense
            regime this sampler targets.
            """
            # P = I + diag(ω) - ρ(W+W^T) + ρ²W^TW  (σ² = 1)
            P_diag_r = jnp.ones(n) + omega_new
            P_r = jnp.diag(P_diag_r) - rho_val * W_sym + rho_val**2 * WtW
            P_r = P_r + 1e-6 * jnp.eye(n)

            Xbeta_r = X_jax @ beta_new
            # RHS: Xbeta - ρ W'Xbeta + κ  (σ² = 1)
            rhs_r = Xbeta_r - rho_val * (W_dense_jax.T @ Xbeta_r) + kappa

            L_r = jnp.linalg.cholesky(P_r)
            log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))
            v = solve_triangular(L_r, rhs_r, lower=True)
            quad_r = v @ v

            logdet_W = logdet_jax(rho_val)

            log_prior = jnp.where(
                (rho_val >= rho_lower_jax) & (rho_val <= rho_upper_jax), 0.0, -jnp.inf
            )
            return logdet_W - 0.5 * log_det_P + 0.5 * quad_r + log_prior

        # Slice sampling for ρ
        log_y0 = log_density_rho(rho)
        log_u = log_y0 + jnp.log(jax.random.uniform(key_rho_u, dtype=jnp.float64))

        # Step-out width
        w_rho = jnp.float64(0.2)

        # Stepping out: expand [L, R] until log_density < log_u at endpoints
        u_rand = jax.random.uniform(key_rho_L, dtype=jnp.float64)
        L = jnp.maximum(rho - u_rand * w_rho, rho_lower_jax)
        R = jnp.minimum(L + w_rho, rho_upper_jax)

        max_steps = 50

        # Step out left
        def step_left(carry):
            L_val, steps = carry
            return (jnp.maximum(L_val - w_rho, rho_lower_jax), steps + 1)

        def should_step_left(carry):
            L_val, steps = carry
            return (
                (L_val > rho_lower_jax)
                & (log_density_rho(L_val) > log_u)
                & (steps < max_steps)
            )

        L_final, _ = jax.lax.while_loop(should_step_left, step_left, (L, jnp.int32(0)))

        # Step out right
        def step_right(carry):
            R_val, steps = carry
            return (jnp.minimum(R_val + w_rho, rho_upper_jax), steps + 1)

        def should_step_right(carry):
            R_val, steps = carry
            return (
                (R_val < rho_upper_jax)
                & (log_density_rho(R_val) > log_u)
                & (steps < max_steps)
            )

        R_final, _ = jax.lax.while_loop(
            should_step_right, step_right, (R, jnp.int32(0))
        )

        # Shrinkage: sample from [L, R] and shrink until accepted
        def shrink_cond(carry):
            _, _, _, _, done = carry
            return ~done

        def shrink_body(carry):
            L_val, R_val, key_val, _, done_val = carry
            key_val, subkey = jax.random.split(key_val)
            x_new = L_val + jax.random.uniform(subkey, dtype=jnp.float64) * (
                R_val - L_val
            )
            log_dens_new = log_density_rho(x_new)
            accepted = log_dens_new > log_u
            L_new = jnp.where(x_new < rho, x_new, L_val)
            R_new = jnp.where(x_new >= rho, x_new, R_val)
            collapsed = (R_new - L_new) < 1e-15
            done = accepted | collapsed
            x_best = jnp.where(accepted, x_new, rho)
            return (L_new, R_new, key_val, x_best, done)

        _, _, _, rho_new, _ = jax.lax.while_loop(
            shrink_cond,
            shrink_body,
            (L_final, R_final, key_rho_shrink, rho, jnp.bool_(False)),
        )

        rho_new = jnp.clip(rho_new, rho_lower_jax, rho_upper_jax)

        # Accept flag is always True for slice sampling
        accept = jnp.bool_(True)

        new_state = JAXLogitGibbsState(
            eta=eta_new,
            beta=beta_new,
            rho=rho_new,
            omega=omega_new,
        )
        return new_state, accept

    return gibbs_step


def _logit_loglik_pointwise_jax(y, eta):
    """Compute pointwise logit log-likelihood (numpy, for storage).

    Uses the numerically stable log-sum-exp trick.
    """
    import jax.numpy as jnp

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    eta_jax = jnp.asarray(eta, dtype=jnp.float64)
    # log p(y|η) = y*η - max(η, 0) - log(1 + exp(-|η|))
    result = (
        y_jax * eta_jax
        - jnp.maximum(eta_jax, 0)
        - jnp.log1p(jnp.exp(-jnp.abs(eta_jax)))
    )
    return np.asarray(result)


def run_chain_jax(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    init,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    progress_manager=None,
    chain_id: int = 0,
):
    """Run one chain of the full-JIT JAX Gibbs sampler for SAR-logit.

    Creates a JIT-compiled Gibbs step function and runs it in a Python
    loop, handling warmup, thinning, and storage.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Spatial weights matrix (used to extract W_dense).
    W_sym_dense : jax.numpy.ndarray of shape (n, n)
        Dense (W + W^T).
    WtW_dense : jax.numpy.ndarray of shape (n, n)
        Dense W^T W.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    init : LogitGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int
        Keep every thin-th draw.
    return_eta : bool
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.
    pg_n_terms : int
        Number of alternating-series terms for the PG approximation.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
        Also includes ``mh_accept_rate`` (always 1.0 for slice sampling).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    # Convert data to JAX arrays
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)

    # Extract W_dense (original row-standardised W, not W_sym)
    W_dense = W_sparse.toarray()
    W_dense_jax = jnp.asarray(W_dense, dtype=jnp.float64)

    # Initialize state
    state = JAXLogitGibbsState(
        eta=jnp.asarray(init.eta, dtype=jnp.float64),
        beta=jnp.asarray(init.beta, dtype=jnp.float64),
        rho=jnp.float64(init.rho),
        omega=jnp.asarray(init.omega, dtype=jnp.float64),
    )

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64)
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Build the JIT-compiled step function
    gibbs_step = _make_gibbs_step_with_data(
        y_jax=y_jax,
        X_jax=X_jax,
        W_dense_jax=W_dense_jax,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        priors=priors,
        pg_n_terms=pg_n_terms,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
    )

    # Warmup the JIT function (first call triggers compilation)
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    _ = gibbs_step(state, warmup_key)

    # Run the chain
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key)

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = float(state.rho)
                beta_samples[idx] = np.asarray(state.beta)
                eta_np = np.asarray(state.eta)
                eta_norm_samples[idx] = float(eta_np @ eta_np)
                if return_eta:
                    eta_samples[idx] = eta_np
                log_lik_samples[idx] = _logit_loglik_pointwise_jax(y, eta_np)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune, accept=None)

    if progress_manager is not None:
        progress_manager.refresh()

    result = {
        "rho": rho_samples,
        "beta": beta_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
        "mh_accept_rate": 1.0,  # slice sampling always accepts
    }
    if return_eta:
        result["eta"] = eta_samples

    return result


# ===========================================================================
# SEM-logit JAX Gibbs sampler
# ===========================================================================


def _make_gibbs_step_with_data_sem(
    y_jax,
    X_jax,
    W_dense_jax,
    n,
    k,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    pg_n_terms,
    n_probes,
    lanczos_deg,
):
    """Build a JIT-compiled SEM-logit Gibbs step with data bound into the closure.

    Same 4-block structure as SAR-logit (ω, η, β, λ) but with SEM-specific
    rhs and β block formulas.

    Key differences from SAR-logit:
    - η rhs: A_λ'A_λXβ + κ  (not A_λ'Xβ + κ)
    - β block: uses X* = A_λX, η* = A_λη  (not X, A_ρη)
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from ._core import JAXSEMLogitGibbsState

    jax.config.update("jax_enable_x64", True)

    W_sym = jnp.asarray(W_sym_dense, dtype=jnp.float64)
    WtW = jnp.asarray(WtW_dense, dtype=jnp.float64)
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    lam_lower_jax = jnp.float64(priors.lam_lower)
    lam_upper_jax = jnp.float64(priors.lam_upper)

    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)

    k_idx = jnp.arange(pg_n_terms, dtype=jnp.float64)
    pi = jnp.pi
    pi2 = pi * pi

    kappa = y_jax - 0.5

    @eqx.filter_jit
    def gibbs_step(state, key):
        """One complete SEM-logit Gibbs sweep: ω → η → β → λ (slice)."""
        eta = state.eta
        beta = state.beta
        lam = state.lam

        key_omega, key_eta, key_beta, key_lam = jax.random.split(key, 4)

        # ── Block 1: ω ~ PG(1, η) — identical to SAR-logit ──
        z = eta
        Z = jnp.abs(z) / 2.0
        denominators = (k_idx + 0.5) ** 2 + (Z[:, None] / pi) ** 2
        g = jax.random.gamma(
            key_omega,
            jnp.ones((n, pg_n_terms)),
            dtype=jnp.float64,
        )
        pg1 = jnp.sum(g / denominators, axis=1) / (2.0 * pi2)
        omega_new = jnp.maximum(pg1, 1e-6)

        # ── Block 2: η | ω, β, λ — SEM-specific rhs ──
        # P = I + diag(ω) - λ(W+W^T) + λ²W^TW
        P_diag = jnp.ones(n) + omega_new
        P = jnp.diag(P_diag) - lam * W_sym + lam**2 * WtW
        P = P + 1e-6 * jnp.eye(n)

        Xbeta = X_jax @ beta
        # SEM rhs: A_λ'A_λXβ + κ = Xβ - λ(W+W')Xβ + λ²W'WXβ + κ
        WsymXbeta = W_sym @ Xbeta
        WtWXbeta = WtW @ Xbeta
        rhs = Xbeta - lam * WsymXbeta + lam**2 * WtWXbeta + kappa

        L = jnp.linalg.cholesky(P)
        P_inv_rhs = cho_solve((L, True), rhs)
        z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
        eta_new = P_inv_rhs + solve_triangular(L.T, z_eta, lower=False)

        # ── Block 3: β | η, λ — SEM-style transformed data ──
        # X* = (I - λW)X,  η* = (I - λW)η
        A_lam_eta = eta_new - lam * W_dense_jax @ eta_new
        X_star = X_jax - lam * (W_dense_jax @ X_jax)
        eta_star = A_lam_eta

        XstXs = X_star.T @ X_star
        Sigma_beta_inv = beta_prior_prec + XstXs
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_star.T @ eta_star
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = cho_solve((L_beta, True), rhs_beta)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)

        # ── Block 4: λ — slice sampling (collapsed, η integrated out) ──
        key_lam_slice, key_lam_u, key_lam_L, key_lam_R, key_lam_shrink = (
            jax.random.split(key_lam, 5)
        )

        def log_density_lam(lam_val):
            """Collapsed log-density of λ (η integrated out, σ² = 1).

            Uses one dense Cholesky of P_r to obtain log|P_r| and the
            quadratic form exactly.
            """
            P_diag_r = jnp.ones(n) + omega_new
            P_r = jnp.diag(P_diag_r) - lam_val * W_sym + lam_val**2 * WtW
            P_r = P_r + 1e-6 * jnp.eye(n)

            Xbeta_r = X_jax @ beta_new
            WsymXbeta_r = W_sym @ Xbeta_r
            WtWXbeta_r = WtW @ Xbeta_r
            rhs_r = Xbeta_r - lam_val * WsymXbeta_r + lam_val**2 * WtWXbeta_r + kappa

            L_r = jnp.linalg.cholesky(P_r)
            log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))
            v = solve_triangular(L_r, rhs_r, lower=True)
            quad_r = v @ v

            logdet_W = logdet_jax(lam_val)

            # Missing term from SEM prior: -½Xβ'A_λ'A_λXβ
            # = -½Xβ'Xβ + ½λXβ'W_symXβ - ½λ²Xβ'W'WXβ
            # The -½Xβ'Xβ is constant in λ and drops out.
            xbeta_correction = 0.5 * lam_val * (
                Xbeta_r @ WsymXbeta_r
            ) - 0.5 * lam_val**2 * (Xbeta_r @ WtWXbeta_r)

            log_prior = jnp.where(
                (lam_val >= lam_lower_jax) & (lam_val <= lam_upper_jax), 0.0, -jnp.inf
            )
            return (
                logdet_W - 0.5 * log_det_P + 0.5 * quad_r + xbeta_correction + log_prior
            )

        # Slice sampling for λ
        log_y0 = log_density_lam(lam)
        log_u = log_y0 + jnp.log(jax.random.uniform(key_lam_u, dtype=jnp.float64))

        w_lam = jnp.float64(0.2)

        u_rand = jax.random.uniform(key_lam_L, dtype=jnp.float64)
        L = jnp.maximum(lam - u_rand * w_lam, lam_lower_jax)
        R = jnp.minimum(L + w_lam, lam_upper_jax)

        max_steps = 50

        def step_left(carry):
            L_val, steps = carry
            return (jnp.maximum(L_val - w_lam, lam_lower_jax), steps + 1)

        def should_step_left(carry):
            L_val, steps = carry
            return (
                (L_val > lam_lower_jax)
                & (log_density_lam(L_val) > log_u)
                & (steps < max_steps)
            )

        L_final, _ = jax.lax.while_loop(should_step_left, step_left, (L, jnp.int32(0)))

        def step_right(carry):
            R_val, steps = carry
            return (jnp.minimum(R_val + w_lam, lam_upper_jax), steps + 1)

        def should_step_right(carry):
            R_val, steps = carry
            return (
                (R_val < lam_upper_jax)
                & (log_density_lam(R_val) > log_u)
                & (steps < max_steps)
            )

        R_final, _ = jax.lax.while_loop(
            should_step_right, step_right, (R, jnp.int32(0))
        )

        def shrink_cond(carry):
            _, _, _, _, done = carry
            return ~done

        def shrink_body(carry):
            L_val, R_val, key_val, _, done_val = carry
            key_val, subkey = jax.random.split(key_val)
            x_new = L_val + jax.random.uniform(subkey, dtype=jnp.float64) * (
                R_val - L_val
            )
            log_dens_new = log_density_lam(x_new)
            accepted = log_dens_new > log_u
            L_new = jnp.where(x_new < lam, x_new, L_val)
            R_new = jnp.where(x_new >= lam, x_new, R_val)
            collapsed = (R_new - L_new) < 1e-15
            done = accepted | collapsed
            x_best = jnp.where(accepted, x_new, lam)
            return (L_new, R_new, key_val, x_best, done)

        _, _, _, lam_new, _ = jax.lax.while_loop(
            shrink_cond,
            shrink_body,
            (L_final, R_final, key_lam_shrink, lam, jnp.bool_(False)),
        )

        lam_new = jnp.clip(lam_new, lam_lower_jax, lam_upper_jax)

        accept = jnp.bool_(True)

        new_state = JAXSEMLogitGibbsState(
            eta=eta_new,
            beta=beta_new,
            lam=lam_new,
            omega=omega_new,
        )
        return new_state, accept

    return gibbs_step


def run_chain_jax_sem(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    init,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    progress_manager=None,
    chain_id: int = 0,
):
    """Run one chain of the full-JIT JAX Gibbs sampler for SEM-logit.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Spatial weights matrix.
    W_sym_dense : jax.numpy.ndarray of shape (n, n)
        Dense (W + W^T).
    WtW_dense : jax.numpy.ndarray of shape (n, n)
        Dense W^T W.
    logdet_jax : callable
        JAX-native function ``(lam) -> jax.numpy.ndarray`` computing
        log|I - lam*W|.
    priors : SEMLogitGibbsPriors
        Prior hyperparameters.
    init : SEMLogitGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int
        Keep every thin-th draw.
    return_eta : bool
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.
    pg_n_terms : int
        Number of alternating-series terms for the PG approximation.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``lam``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
        Also includes ``mh_accept_rate`` (always 1.0 for slice sampling).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from ._core import JAXSEMLogitGibbsState

    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    # Convert data to JAX arrays
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)

    # Extract W_dense (original row-standardised W, not W_sym)
    W_dense = W_sparse.toarray()
    W_dense_jax = jnp.asarray(W_dense, dtype=jnp.float64)

    # Initialize state
    state = JAXSEMLogitGibbsState(
        eta=jnp.asarray(init.eta, dtype=jnp.float64),
        beta=jnp.asarray(init.beta, dtype=jnp.float64),
        lam=jnp.float64(init.lam),
        omega=jnp.asarray(init.omega, dtype=jnp.float64),
    )

    # Pre-allocate storage
    lam_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64)
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Build the JIT-compiled step function
    gibbs_step = _make_gibbs_step_with_data_sem(
        y_jax=y_jax,
        X_jax=X_jax,
        W_dense_jax=W_dense_jax,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        priors=priors,
        pg_n_terms=pg_n_terms,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
    )

    # Warmup the JIT function
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    _ = gibbs_step(state, warmup_key)

    # Run the chain
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key)

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                lam_samples[idx] = float(state.lam)
                beta_samples[idx] = np.asarray(state.beta)
                eta_np = np.asarray(state.eta)
                eta_norm_samples[idx] = float(eta_np @ eta_np)
                if return_eta:
                    eta_samples[idx] = eta_np
                log_lik_samples[idx] = _logit_loglik_pointwise_jax(y, eta_np)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune, accept=None)

    if progress_manager is not None:
        progress_manager.refresh()

    result = {
        "lam": lam_samples,
        "beta": beta_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
        "mh_accept_rate": 1.0,
    }
    if return_eta:
        result["eta"] = eta_samples

    return result


# ===========================================================================
# Vectorized multi-chain runners (jax.vmap)
# ===========================================================================
#
# These runners execute all chains in parallel on a single device via
# ``jax.vmap`` over a ``jax.lax.scan``-ed chain.  This is the JAX-native
# equivalent of multiprocessing and is dramatically faster than calling
# ``run_chain_jax`` once per chain in a Python loop, because (a) the
# Gibbs step JITs only once and (b) all chains run as one fused XLA
# program with no per-iteration Python overhead.
#
# Mirrors the pattern in ``bayespecon.samplers.gaussian._jax``
# (``run_chains_jax_gibbs_vectorized``).


def _logit_loglik_pointwise_jax_op(y_jax, eta):
    """Pointwise logit log-likelihood as a pure-JAX op (vmap-safe)."""
    import jax.numpy as jnp

    return y_jax * eta - jnp.maximum(eta, 0) - jnp.log1p(jnp.exp(-jnp.abs(eta)))


def _run_chain_logit_warmup(gibbs_step, init_state, key, n_iters):
    """Run ``n_iters`` Gibbs steps and return only the final state + key.

    Uses :func:`jax.lax.fori_loop` so no per-iteration traces are
    materialised — memory cost is independent of ``n_iters``.
    The final PRNG key is returned so chunked runs can resume from a
    deterministic point without breaking the chain.
    """
    import jax

    def body(_, carry):
        state, k = carry
        k, step_key = jax.random.split(k)
        state, _ = gibbs_step(state, step_key)
        return (state, k)

    final_state, final_key = jax.lax.fori_loop(0, n_iters, body, (init_state, key))
    return final_state, final_key


def _run_chain_logit_draws_sar(gibbs_step, y_jax, init_state, key, n_iters):
    """Scan ``n_iters`` steps for SAR-logit, storing per-iter traces.

    Returns the final state, the final PRNG key, and stacked traces of
    ``rho``, ``beta``, ``eta_norm``, and per-observation ``log_lik``.
    """
    import jax

    def body(carry, _):
        state, k = carry
        k, step_key = jax.random.split(k)
        state, _ = gibbs_step(state, step_key)
        log_lik = _logit_loglik_pointwise_jax_op(y_jax, state.eta)
        eta_norm = state.eta @ state.eta
        return (state, k), (state.rho, state.beta, eta_norm, log_lik)

    (final_state, final_key), (rhos, betas, eta_norms, log_liks) = jax.lax.scan(
        body, (init_state, key), None, length=n_iters
    )
    return final_state, final_key, rhos, betas, eta_norms, log_liks


def _run_chain_logit_draws_sem(gibbs_step, y_jax, init_state, key, n_iters):
    """Scan ``n_iters`` steps for SEM-logit; ``lam`` instead of ``rho``."""
    import jax

    def body(carry, _):
        state, k = carry
        k, step_key = jax.random.split(k)
        state, _ = gibbs_step(state, step_key)
        log_lik = _logit_loglik_pointwise_jax_op(y_jax, state.eta)
        eta_norm = state.eta @ state.eta
        return (state, k), (state.lam, state.beta, eta_norm, log_lik)

    (final_state, final_key), (lams, betas, eta_norms, log_liks) = jax.lax.scan(
        body, (init_state, key), None, length=n_iters
    )
    return final_state, final_key, lams, betas, eta_norms, log_liks


def _stack_chain_inits(inits, state_cls, scalar_field):
    """Convert a list of per-chain states into one vmap-able pytree.

    ``scalar_field`` names the scalar parameter (``"rho"`` or ``"lam"``)
    so the helper works for both SAR and SEM state classes.
    """
    import jax.numpy as jnp

    jax_inits = [
        state_cls(
            eta=jnp.asarray(init.eta, dtype=jnp.float64),
            beta=jnp.asarray(init.beta, dtype=jnp.float64),
            omega=jnp.asarray(init.omega, dtype=jnp.float64),
            **{scalar_field: jnp.float64(getattr(init, scalar_field))},
        )
        for init in inits
    ]
    import jax

    return jax.tree.map(lambda *a: jnp.stack(a), *jax_inits)


def run_chains_jax_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    progressbar: bool = True,
) -> list[dict]:
    """Run multiple SAR-logit Gibbs chains in parallel via ``jax.vmap``.

    All chains execute together on a single device as one fused XLA
    program — there is no Python loop over chains, and the Gibbs step
    is JIT-compiled only once.

    Parameters
    ----------
    y, X, W_sparse, W_sym_dense, WtW_dense, logdet_jax, priors :
        Same meaning as in :func:`run_chain_jax`.
    inits : list of LogitGibbsState
        Per-chain initial states.  Length determines the number of
        chains.
    draws, tune, thin :
        Same meaning as in :func:`run_chain_jax`.
    jax_seeds : list of int, optional
        Per-chain JAX PRNG seeds.  Defaults to ``range(chains)``.
    pg_n_terms, n_probes, lanczos_deg :
        Same meaning as in :func:`run_chain_jax`.
    progressbar : bool, default True
        Show per-chain progress bars.  Because all chains advance in
        lock-step under vmap, the bar is updated in bulk at the end of
        each phase.

    Returns
    -------
    list of dict
        One dict per chain with keys ``rho``, ``beta``, ``log_lik``,
        ``eta_norm``, ``mh_accept_rate``.

    Notes
    -----
    ``return_eta=True`` is not supported by this runner; use the
    sequential :func:`run_chain_jax` path if you need the full latent
    field.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    chains = len(inits)
    n, k = X.shape

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)
    W_dense_jax = jnp.asarray(W_sparse.toarray(), dtype=jnp.float64)

    gibbs_step = _make_gibbs_step_with_data(
        y_jax=y_jax,
        X_jax=X_jax,
        W_dense_jax=W_dense_jax,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        priors=priors,
        pg_n_terms=pg_n_terms,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
    )

    init_states = _stack_chain_inits(inits, JAXLogitGibbsState, "rho")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._progress import GibbsProgressBarManager

    # Chunk both phases into ~20 segments so the progress bar advances
    # smoothly and Python regains control between segments.  The chunk
    # size is a Python constant so JIT compiles each kernel once and
    # reuses it across chunks.
    warmup_chunk = max(1, tune // 20) if tune > 0 else 1
    draws_chunk = max(1, draws // 20) if draws > 0 else 1

    warmup_step = jax.jit(
        lambda s, k: jax.vmap(
            lambda s_, k_: _run_chain_logit_warmup(gibbs_step, s_, k_, warmup_chunk)
        )(s, k)
    )
    draws_step = jax.jit(
        lambda s, k: jax.vmap(
            lambda s_, k_: _run_chain_logit_draws_sar(
                gibbs_step, y_jax, s_, k_, draws_chunk
            )
        )(s, k)
    )

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sar_logit",
    ) as pm:
        if pm is not None:
            for c in range(chains):
                pm.start_chain(c)

        # ── Phase 1: warmup (no traces stored) ──
        state = init_states
        keys = warmup_keys
        iter_done = 0
        while iter_done < tune:
            step = min(warmup_chunk, tune - iter_done)
            if step == warmup_chunk:
                state, keys = warmup_step(state, keys)
            else:
                # Final short chunk: take a one-off JIT hit rather than
                # padding (padding would change the chain).
                state, keys = jax.vmap(
                    lambda s_, k_: _run_chain_logit_warmup(gibbs_step, s_, k_, step)
                )(state, keys)
            jax.block_until_ready(state.rho)
            iter_done += step
            if pm is not None:
                for c in range(chains):
                    pm.update(c, iter_done - 1, tuning=True, accept=None)

        final_warm_states = state

        # ── Phase 2: post-warmup draws (stacked traces) ──
        draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)
        state = final_warm_states
        keys = draw_keys
        rho_chunks: list[np.ndarray] = []
        beta_chunks: list[np.ndarray] = []
        eta_chunks: list[np.ndarray] = []
        ll_chunks: list[np.ndarray] = []
        iter_done = 0
        while iter_done < draws:
            step = min(draws_chunk, draws - iter_done)
            if step == draws_chunk:
                state, keys, rhos_c, betas_c, eta_c, ll_c = draws_step(state, keys)
            else:
                state, keys, rhos_c, betas_c, eta_c, ll_c = jax.vmap(
                    lambda s_, k_: _run_chain_logit_draws_sar(
                        gibbs_step, y_jax, s_, k_, step
                    )
                )(state, keys)
            rho_chunks.append(np.asarray(rhos_c))
            beta_chunks.append(np.asarray(betas_c))
            eta_chunks.append(np.asarray(eta_c))
            ll_chunks.append(np.asarray(ll_c))
            iter_done += step
            if pm is not None:
                for c in range(chains):
                    pm.update(c, tune + iter_done - 1, tuning=False, accept=None)

        rhos = np.concatenate(rho_chunks, axis=1)
        betas = np.concatenate(beta_chunks, axis=1)
        eta_norms = np.concatenate(eta_chunks, axis=1)
        log_liks = np.concatenate(ll_chunks, axis=1)

    # Thin and pack as per-chain dicts
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "rho": rhos[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy(),
                "mh_accept_rate": 1.0,
            }
        )
    return results


def run_chains_jax_sem_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    progressbar: bool = True,
) -> list[dict]:
    """Run multiple SEM-logit Gibbs chains in parallel via ``jax.vmap``.

    See :func:`run_chains_jax_vectorized` for the SAR-logit analogue
    and shared design rationale.  Returns per-chain dicts keyed on
    ``lam`` rather than ``rho``.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from ._core import JAXSEMLogitGibbsState

    chains = len(inits)
    n, k = X.shape

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    W_dense_jax = jnp.asarray(W_sparse.toarray(), dtype=jnp.float64)

    gibbs_step = _make_gibbs_step_with_data_sem(
        y_jax=y_jax,
        X_jax=X_jax,
        W_dense_jax=W_dense_jax,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        priors=priors,
        pg_n_terms=pg_n_terms,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
    )

    init_states = _stack_chain_inits(inits, JAXSEMLogitGibbsState, "lam")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._progress import GibbsProgressBarManager

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sem_logit",
    ) as pm:
        if pm is not None:
            for c in range(chains):
                pm.start_chain(c)

        warmup_chunk = max(1, tune // 20) if tune > 0 else 1
        draws_chunk = max(1, draws // 20) if draws > 0 else 1

        warmup_step = jax.jit(
            lambda s, k: jax.vmap(
                lambda s_, k_: _run_chain_logit_warmup(gibbs_step, s_, k_, warmup_chunk)
            )(s, k)
        )
        draws_step = jax.jit(
            lambda s, k: jax.vmap(
                lambda s_, k_: _run_chain_logit_draws_sem(
                    gibbs_step, y_jax, s_, k_, draws_chunk
                )
            )(s, k)
        )

        # ── Phase 1: warmup ──
        state = init_states
        keys = warmup_keys
        iter_done = 0
        while iter_done < tune:
            step = min(warmup_chunk, tune - iter_done)
            if step == warmup_chunk:
                state, keys = warmup_step(state, keys)
            else:
                state, keys = jax.vmap(
                    lambda s_, k_: _run_chain_logit_warmup(gibbs_step, s_, k_, step)
                )(state, keys)
            jax.block_until_ready(state.lam)
            iter_done += step
            if pm is not None:
                for c in range(chains):
                    pm.update(c, iter_done - 1, tuning=True, accept=None)

        final_warm_states = state

        # ── Phase 2: post-warmup draws ──
        draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)
        state = final_warm_states
        keys = draw_keys
        lam_chunks: list[np.ndarray] = []
        beta_chunks: list[np.ndarray] = []
        eta_chunks: list[np.ndarray] = []
        ll_chunks: list[np.ndarray] = []
        iter_done = 0
        while iter_done < draws:
            step = min(draws_chunk, draws - iter_done)
            if step == draws_chunk:
                state, keys, lams_c, betas_c, eta_c, ll_c = draws_step(state, keys)
            else:
                state, keys, lams_c, betas_c, eta_c, ll_c = jax.vmap(
                    lambda s_, k_: _run_chain_logit_draws_sem(
                        gibbs_step, y_jax, s_, k_, step
                    )
                )(state, keys)
            lam_chunks.append(np.asarray(lams_c))
            beta_chunks.append(np.asarray(betas_c))
            eta_chunks.append(np.asarray(eta_c))
            ll_chunks.append(np.asarray(ll_c))
            iter_done += step
            if pm is not None:
                for c in range(chains):
                    pm.update(c, tune + iter_done - 1, tuning=False, accept=None)

        lams = np.concatenate(lam_chunks, axis=1)
        betas = np.concatenate(beta_chunks, axis=1)
        eta_norms = np.concatenate(eta_chunks, axis=1)
        log_liks = np.concatenate(ll_chunks, axis=1)

    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "lam": lams[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy(),
                "mh_accept_rate": 1.0,
            }
        )
    return results
