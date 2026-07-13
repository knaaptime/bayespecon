r"""JAX-accelerated full-JIT Gibbs sampler for SAR-logit.

Composes all Gibbs blocks (PG ω, η, β, ρ slice) into a single
``@jax.jit``-compiled function, eliminating Python→JAX dispatch overhead
entirely.

Architecture
------------
The sampler uses:

- **PG sampling**: Exact sum-of-exponentials method via
  :func:`jax_polyagamma` with ``method="exp"``.  Since h = 1 for
  binary logit, the Exp method is exact (no truncation bias) and
  fast (single exponential draw per observation).
- **η and β draws**: Dense Cholesky factorisation via ``jnp.linalg.cholesky``
  and ``jnp.linalg.solve``.  O(n³) but fast for n ≤ ~2000.
- **ρ/λ draw**: 1-D slice sampler on the doubly-collapsed log-density.
  For SAR-logit, the density marginalises out both η and β via a
  multi-RHS Cholesky solve.  For SEM-logit, the density marginalises
  out η but conditions on β.  Slice sampling avoids the expensive
  gradient computation that MALA requires (autodiff through Cholesky
  + solve is ~3× the forward cost).

Limitations
-----------
- O(n³) dense Cholesky limits scalability to n ≤ ~2000.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.
"""

from __future__ import annotations

import numpy as np

from .._utils._jax_polyagamma import jax_polyagamma
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


def _build_w_bcoo(W_sparse):
    """Build ``(W, Wᵀ)`` as JAX BCOO sparse matrices — never densify W."""
    from jax.experimental import sparse as jsparse

    W_bcoo = jsparse.BCOO.from_scipy_sparse(W_sparse.tocsr())
    Wt_bcoo = jsparse.BCOO.from_scipy_sparse(W_sparse.T.tocsr())
    return W_bcoo, Wt_bcoo


def _make_gibbs_step_with_data(
    y_jax,
    X_jax,
    W_bcoo,
    Wt_bcoo,
    n,
    k,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    XtX_jax,
    priors,
    n_probes,
    lanczos_deg,
    cholmodjax_pattern=None,
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
    W_bcoo : jax.experimental.sparse.BCOO of shape (n, n)
        Row-standardised W as a sparse BCOO matrix (for ``W @ x``).
    Wt_bcoo : jax.experimental.sparse.BCOO of shape (n, n)
        Transpose ``Wᵀ`` as a sparse BCOO matrix (for ``Wᵀ @ x``).
    n : int
        Number of spatial units.
    k : int
        Number of regression coefficients.
    W_sym_dense : jax.numpy.ndarray of shape (n, n) or None
        Dense (W + W^T).  Only needed for the dense-Cholesky fallback.
    WtW_dense : jax.numpy.ndarray of shape (n, n) or None
        Dense W^T W.  Only needed for the dense-Cholesky fallback.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.
    XtX_jax : jax.numpy.ndarray of shape (k, k)
        Precomputed X^T X.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key, slice_width) -> (new_state, accept)

        where ``state`` is a :class:`JAXLogitGibbsState`,
        ``key`` is a JAX PRNG key, and ``slice_width`` is a ``jnp.float64``
        stepping-out width for the ρ slice sampler.  ``accept`` is always
        ``1.0`` (slice sampling always accepts).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from .._utils._jax_slice import jax_slice_sample_1d

    jax.config.update("jax_enable_x64", True)

    use_cholmodjax = cholmodjax_pattern is not None

    # Sparse W matvecs — never densify W: W @ x and Wᵀ @ x go through BCOO.
    def W_matvec(x):
        return W_bcoo @ x

    def Wt_matvec(x):
        return Wt_bcoo @ x

    # Dense (W+Wᵀ) and WᵀW are only needed to assemble the dense P in the
    # no-cholmodjax fallback; skip building them on the sparse path.
    if not use_cholmodjax:
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
    1.0 / beta_sigma2_jax  # diagonal of V₀⁻¹
    V0_inv_b0 = beta_mu_jax / beta_sigma2_jax  # V₀⁻¹ b₀

    # For logit: h = 1 always, kappa = y - 0.5
    kappa = y_jax - 0.5

    # Precompute W^T X for the β-marginalised density (sparse, O(nnz·k))
    WtX = Wt_bcoo @ X_jax  # (n, k)

    # ── cholmodjax setup (optional sparse SPD Cholesky path) ──
    if use_cholmodjax:
        _Ai = jnp.asarray(cholmodjax_pattern["Ai"], dtype=jnp.int32)
        _Aj = jnp.asarray(cholmodjax_pattern["Aj"], dtype=jnp.int32)
        _W_sym_vals = jnp.asarray(cholmodjax_pattern["W_sym_vals"], dtype=jnp.float64)
        _WtW_vals = jnp.asarray(cholmodjax_pattern["WtW_vals"], dtype=jnp.float64)
        _diag_idx = jnp.asarray(cholmodjax_pattern["diag_idx"], dtype=jnp.int32)
        _nnz = len(cholmodjax_pattern["Ai"])
        _n_static = int(cholmodjax_pattern["n"])
        # Factor-once closures: η-draw and each ρ-density eval do exactly ONE
        # numeric factorization via cholmodjax 0.4 sample_gaussian / factor_solve
        # (0.3 fallback inside).
        from .._utils._cholmodjax_utils import make_cholmodjax_ops

        _eta_sample, _solve_logdet = make_cholmodjax_ops(_Ai, _Aj, _n_static)

        def _assemble_Ax(omega, rho_val):
            """Assemble COO values for P = I + diag(ω) − ρ(W+Wᵀ) + ρ²WᵀW."""
            Ax = -rho_val * _W_sym_vals + rho_val**2 * _WtW_vals
            diag_vals = jnp.zeros(_nnz, dtype=jnp.float64)
            diag_vals = diag_vals.at[_diag_idx].set(1.0 + omega)
            return Ax + diag_vals

    @eqx.filter_jit
    def gibbs_step(state, key, slice_width):
        """One complete Gibbs sweep: ω → η → β → ρ (slice).

        Parameters
        ----------
        state : JAXLogitGibbsState
            Current state.
        key : jax.random.PRNGKey
            JAX random key.
        slice_width : jax.numpy.float64
            Stepping-out width for the ρ slice sampler.

        Returns
        -------
        new_state : JAXLogitGibbsState
            Updated state.
        accept : jax.numpy.float64
            Always 1.0 (slice sampling always accepts).
        """
        eta = state.eta
        beta = state.beta
        rho = state.rho

        key_omega, key_eta, key_beta, key_rho = jax.random.split(key, 4)

        # ── Block 1: ω ~ PG(1, η) — exact Exp method (h = 1) ──
        omega_new = jax_polyagamma(jnp.ones(n), eta, key=key_omega, method="exp")

        # ── Block 2: η | ω, ρ, β — Cholesky solve (σ² = 1) ──
        # P = I + diag(ω) - ρ(W+W^T) + ρ²W^TW  (no 1/σ² scaling)
        Xbeta = X_jax @ beta
        # RHS: Xbeta - ρ W'Xbeta + κ  (σ² = 1); W'Xbeta = (WᵀX)β reuses WtX.
        rhs = Xbeta - rho * (WtX @ beta) + kappa

        if use_cholmodjax:
            # Sparse SPD Cholesky via cholmodjax — mean + draw ~ N(P⁻¹ rhs, P⁻¹)
            # from ONE factorization.
            Ax = _assemble_Ax(omega_new, rho)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = _eta_sample(Ax, rhs, z_eta)
        else:
            P_diag = jnp.ones(n) + omega_new
            P = jnp.diag(P_diag) - rho * W_sym + rho**2 * WtW
            P = P + 1e-6 * jnp.eye(n)  # regularisation for numerical stability

            L = jnp.linalg.cholesky(P)
            P_inv_rhs = cho_solve((L, True), rhs)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = P_inv_rhs + solve_triangular(L.T, z_eta, lower=False)

        # ── Block 3: β | η, ρ — conjugate normal (σ² = 1) ──
        A_rho_eta = eta_new - rho * W_matvec(eta_new)
        # Σ_β⁻¹ = Λ₀⁻¹ + X^TX  (no 1/σ² scaling)
        Sigma_beta_inv = beta_prior_prec + XtX_jax
        # rhs = Λ₀⁻¹μ₀ + X^T A_ρη  (no 1/σ² scaling)
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_jax.T @ A_rho_eta
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = cho_solve((L_beta, True), rhs_beta)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)

        # ── Block 4: ρ — 1-D slice sampler (doubly-collapsed density) ──
        #
        # The doubly-collapsed log-density (η AND β integrated out) is:
        #
        #   log p(ρ | ω, y) = log|I - ρW| - ½ log|P_η|
        #                     - ½ log|Σ_β*⁻¹| + ½ κ' P_η⁻¹ κ
        #                     + ½ m_β*' Σ_β*⁻¹ m_β*
        #
        # Slice sampling avoids the expensive gradient computation that
        # MALA requires (autodiff through Cholesky + solve is ~3× the
        # forward cost).  Each slice candidate needs only a forward
        # evaluation — no backward pass.

        def log_density_rho(rho_val):
            """Doubly-collapsed log-density of ρ (η and β integrated out).

            Uses one Cholesky of P_η with (k+1) right-hand sides
            to obtain both the κ quadratic form and the β-marginal
            terms.
            """
            # u = X - ρ W^T X  — (n, k)
            u = X_jax - rho_val * WtX

            # Multi-RHS: P_η [z | M] = [κ | u]
            rhs_stack = jnp.column_stack([kappa, u])  # (n, k+1)

            if use_cholmodjax:
                Ax_r = _assemble_Ax(omega_new, rho_val)
                # Multi-RHS solve + logdet from one factorization (factor_solve).
                sol, log_det_P = _solve_logdet(Ax_r, rhs_stack)  # sol: (n, k+1)
            else:
                # P_η = I + diag(ω) - ρ(W+W^T) + ρ²W^TW  (σ² = 1)
                P_diag_r = jnp.ones(n) + omega_new
                P_r = jnp.diag(P_diag_r) - rho_val * W_sym + rho_val**2 * WtW
                P_r = P_r + 1e-6 * jnp.eye(n)

                L_r = jnp.linalg.cholesky(P_r)
                sol = cho_solve((L_r, True), rhs_stack)  # (n, k+1)
                log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))

            z_vec = sol[:, 0]  # P_η⁻¹ κ  — (n,)
            M_mat = sol[:, 1:]  # P_η⁻¹ u  — (n, k)

            # Σ_β*⁻¹ = X^TX + V₀⁻¹ - u^T M  — (k, k)
            Sig_beta_inv = XtX_jax + beta_prior_prec - u.T @ M_mat
            Sig_beta_inv = Sig_beta_inv + 1e-10 * jnp.eye(k)  # regularise

            L_sig = jnp.linalg.cholesky(Sig_beta_inv)
            log_det_Sig_inv = 2.0 * jnp.sum(jnp.log(jnp.diag(L_sig)))

            # m_β* = Σ_β* (u^T z + V₀⁻¹ b₀)
            rhs_b = u.T @ z_vec + V0_inv_b0
            m_b = cho_solve((L_sig, True), rhs_b)

            # Quadratic forms
            quad_kappa = kappa @ z_vec
            quad_beta = rhs_b @ m_b

            logdet_W = logdet_jax(rho_val)

            log_prior = jnp.where(
                (rho_val >= rho_lower_jax) & (rho_val <= rho_upper_jax), 0.0, -jnp.inf
            )
            return (
                logdet_W
                - 0.5 * log_det_P
                - 0.5 * log_det_Sig_inv
                + 0.5 * quad_kappa
                + 0.5 * quad_beta
                + log_prior
            )

        # ── Slice sampling for ρ (shared JAX helper) ──
        rho_new, _ = jax_slice_sample_1d(
            log_density_rho,
            rho,
            rho_lower_jax,
            rho_upper_jax,
            key=key_rho,
            w=slice_width,
        )

        new_state = JAXLogitGibbsState(
            eta=eta_new,
            beta=beta_new,
            rho=rho_new,
            omega=omega_new,
        )
        return new_state, jnp.float64(1.0)  # slice always accepts

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
    loop, handling warmup, thinning, and storage.  The ρ update uses a
    1-D slice sampler on the doubly-collapsed log-density (η and β
    integrated out).

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
        Ignored (kept for API compatibility).  PG draws now use the
        exact sum-of-exponentials method which does not require
        truncation.
    n_probes : int
        Ignored (kept for API compatibility).
    lanczos_deg : int
        Ignored (kept for API compatibility).
    progress_manager : optional
        Progress bar manager.
    chain_id : int, default 0
        Chain identifier for progress reporting.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
        Also includes ``mh_accept_rate`` (always 1.0; slice sampling
        always accepts).
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

    # Sparse W as BCOO — never densify W for the O(nnz) matvecs.
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

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
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
    )

    # Warmup the JIT function (first call triggers compilation)
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    slice_width = jnp.float64(0.2)
    _ = gibbs_step(state, warmup_key, slice_width)

    # Run the chain
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key, slice_width)

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
        "mh_accept_rate": 1.0,  # slice always accepts
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
    W_bcoo,
    Wt_bcoo,
    n,
    k,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    n_probes,
    lanczos_deg,
    cholmodjax_pattern=None,
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

    from .._utils._jax_slice import jax_slice_sample_1d
    from ._core import JAXSEMLogitGibbsState

    jax.config.update("jax_enable_x64", True)

    use_cholmodjax = cholmodjax_pattern is not None

    # Sparse W matvecs — never densify W.  (W+Wᵀ)@x and (WᵀW)@x compose the
    # base W @ x / Wᵀ @ x BCOO matvecs, so no dense W_sym/WtW is materialised.
    def W_matvec(x):
        return W_bcoo @ x

    def Wt_matvec(x):
        return Wt_bcoo @ x

    def Wsym_matvec(x):
        return W_matvec(x) + Wt_matvec(x)

    def WtW_matvec(x):
        return Wt_matvec(W_matvec(x))

    # Dense (W+Wᵀ) and WᵀW are only needed to assemble the dense P in the
    # no-cholmodjax fallback; skip building them on the sparse path.
    if not use_cholmodjax:
        W_sym = jnp.asarray(W_sym_dense, dtype=jnp.float64)
        WtW = jnp.asarray(WtW_dense, dtype=jnp.float64)
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    lam_lower_jax = jnp.float64(priors.lam_lower)
    lam_upper_jax = jnp.float64(priors.lam_upper)

    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)
    beta_mu_jax / beta_sigma2_jax  # V₀⁻¹ b₀

    kappa = y_jax - 0.5

    # ── cholmodjax setup (optional sparse SPD Cholesky path) ──
    if use_cholmodjax:
        _Ai = jnp.asarray(cholmodjax_pattern["Ai"], dtype=jnp.int32)
        _Aj = jnp.asarray(cholmodjax_pattern["Aj"], dtype=jnp.int32)
        _W_sym_vals = jnp.asarray(cholmodjax_pattern["W_sym_vals"], dtype=jnp.float64)
        _WtW_vals = jnp.asarray(cholmodjax_pattern["WtW_vals"], dtype=jnp.float64)
        _diag_idx = jnp.asarray(cholmodjax_pattern["diag_idx"], dtype=jnp.int32)
        _nnz = len(cholmodjax_pattern["Ai"])
        _n_static = int(cholmodjax_pattern["n"])
        # Factor-once closures (cholmodjax 0.4 sample_gaussian / factor_solve).
        from .._utils._cholmodjax_utils import make_cholmodjax_ops

        _eta_sample, _solve_logdet = make_cholmodjax_ops(_Ai, _Aj, _n_static)

        def _assemble_Ax(omega, lam_val):
            """Assemble COO values for P = I + diag(ω) − λ(W+Wᵀ) + λ²WᵀW."""
            Ax = -lam_val * _W_sym_vals + lam_val**2 * _WtW_vals
            diag_vals = jnp.zeros(_nnz, dtype=jnp.float64)
            diag_vals = diag_vals.at[_diag_idx].set(1.0 + omega)
            return Ax + diag_vals

    @eqx.filter_jit
    def gibbs_step(state, key, slice_width):
        """One complete SEM-logit Gibbs sweep: ω → η → β → λ (slice)."""
        eta = state.eta
        beta = state.beta
        lam = state.lam

        key_omega, key_eta, key_beta, key_lam = jax.random.split(key, 4)

        # ── Block 1: ω ~ PG(1, η) — exact Exp method (h = 1) ──
        omega_new = jax_polyagamma(jnp.ones(n), eta, key=key_omega, method="exp")

        # ── Block 2: η | ω, β, λ — SEM-specific rhs ──
        # P = I + diag(ω) - λ(W+W^T) + λ²W^TW
        Xbeta = X_jax @ beta
        # SEM rhs: A_λ'A_λXβ + κ = Xβ - λ(W+W')Xβ + λ²W'WXβ + κ
        WsymXbeta = Wsym_matvec(Xbeta)
        WtWXbeta = WtW_matvec(Xbeta)
        rhs = Xbeta - lam * WsymXbeta + lam**2 * WtWXbeta + kappa

        if use_cholmodjax:
            # Mean + draw ~ N(P⁻¹ rhs, P⁻¹) from ONE factorization.
            Ax = _assemble_Ax(omega_new, lam)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = _eta_sample(Ax, rhs, z_eta)
        else:
            P_diag = jnp.ones(n) + omega_new
            P = jnp.diag(P_diag) - lam * W_sym + lam**2 * WtW
            P = P + 1e-6 * jnp.eye(n)

            L = jnp.linalg.cholesky(P)
            P_inv_rhs = cho_solve((L, True), rhs)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = P_inv_rhs + solve_triangular(L.T, z_eta, lower=False)

        # ── Block 3: β | η, λ — SEM-style transformed data ──
        # X* = (I - λW)X,  η* = (I - λW)η
        A_lam_eta = eta_new - lam * W_matvec(eta_new)
        X_star = X_jax - lam * W_matvec(X_jax)
        eta_star = A_lam_eta

        XstXs = X_star.T @ X_star
        Sigma_beta_inv = beta_prior_prec + XstXs
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_star.T @ eta_star
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = cho_solve((L_beta, True), rhs_beta)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)

        # ── Block 4: λ — 1-D slice sampler (η-collapsed density) ──
        #
        # The collapsed log-density (η integrated out, β conditioned) is:
        #
        #   log p(λ | β, ω, y) = log|I - λW| - ½ log|P_η|
        #                       + ½ rhs^T P_η⁻¹ rhs - ½ Xβ^T A_λ^T A_λ Xβ
        #
        # Slice sampling avoids the expensive gradient computation that
        # MALA requires (autodiff through Cholesky + solve is ~3× the
        # forward cost).

        # Hoist the λ-independent matvecs out of the slice density: Xβ and the
        # (W+Wᵀ)Xβ, WᵀWXβ products are constant across every stepping-out /
        # shrink evaluation, so a single set of sparse matvecs per sweep
        # replaces recomputing them dozens of times.
        Xbeta_r = X_jax @ beta_new
        WsymXbeta_r = Wsym_matvec(Xbeta_r)
        WtWXbeta_r = WtW_matvec(Xbeta_r)

        def log_density_lam(lam_val):
            """Collapsed log-density of λ (η integrated out, β conditioned).

            Uses one Cholesky of P_r to obtain log|P_r| and the
            quadratic form exactly.
            """
            rhs_r = Xbeta_r - lam_val * WsymXbeta_r + lam_val**2 * WtWXbeta_r + kappa

            if use_cholmodjax:
                Ax_r = _assemble_Ax(omega_new, lam_val)
                # Solve + logdet from one factorization (factor_solve).
                m, log_det_P = _solve_logdet(Ax_r, rhs_r)
                quad_r = rhs_r @ m
            else:
                P_diag_r = jnp.ones(n) + omega_new
                P_r = jnp.diag(P_diag_r) - lam_val * W_sym + lam_val**2 * WtW
                P_r = P_r + 1e-6 * jnp.eye(n)

                L_r = jnp.linalg.cholesky(P_r)
                log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))
                v = solve_triangular(L_r, rhs_r, lower=True)
                quad_r = v @ v

            logdet_W = logdet_jax(lam_val)

            # SEM correction: -½Xβ'A_λ'A_λXβ
            # = ½λXβ'(W+W')Xβ - ½λ²Xβ'W'WXβ + const
            xbeta_correction = 0.5 * lam_val * (
                Xbeta_r @ WsymXbeta_r
            ) - 0.5 * lam_val**2 * (Xbeta_r @ WtWXbeta_r)

            log_prior = jnp.where(
                (lam_val >= lam_lower_jax) & (lam_val <= lam_upper_jax), 0.0, -jnp.inf
            )
            return (
                logdet_W - 0.5 * log_det_P + 0.5 * quad_r + xbeta_correction + log_prior
            )

        # ── Slice sampling for λ (shared JAX helper) ──
        lam_new, _ = jax_slice_sample_1d(
            log_density_lam,
            lam,
            lam_lower_jax,
            lam_upper_jax,
            key=key_lam,
            w=slice_width,
        )

        new_state = JAXSEMLogitGibbsState(
            eta=eta_new,
            beta=beta_new,
            lam=lam_new,
            omega=omega_new,
        )
        return new_state, jnp.float64(1.0)  # slice always accepts

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
        Ignored (kept for API compatibility).  PG draws now use the
        exact sum-of-exponentials method which does not require
        truncation.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``lam``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
        Also includes ``mh_accept_rate`` (always 1.0; slice sampling
        always accepts).
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

    # Sparse W as BCOO — never densify W for the O(nnz) matvecs.
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

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
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
    )

    # Warmup the JIT function
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    slice_width = jnp.float64(0.2)
    _ = gibbs_step(state, warmup_key, slice_width)

    # Run the chain
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key, slice_width)

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
        "mh_accept_rate": 1.0,  # slice always accepts
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


def _run_chain_logit_warmup(gibbs_step, init_state, key, n_iters, slice_width):
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
        state, _ = gibbs_step(state, step_key, slice_width)
        return (state, k)

    final_state, final_key = jax.lax.fori_loop(0, n_iters, body, (init_state, key))
    return final_state, final_key


def _run_chain_logit_draws_sar(
    gibbs_step, y_jax, init_state, key, n_iters, slice_width
):
    """Scan ``n_iters`` steps for SAR-logit, storing per-iter traces.

    Returns the final state, the final PRNG key, and stacked traces of
    ``rho``, ``beta``, ``eta_norm``, and per-observation ``log_lik``.
    """
    import jax
    import jax.numpy as jnp

    def body(carry, _):
        state, k, accept_sum = carry
        k, step_key = jax.random.split(k)
        state, accept = gibbs_step(state, step_key, slice_width)
        log_lik = _logit_loglik_pointwise_jax_op(y_jax, state.eta)
        eta_norm = state.eta @ state.eta
        return (state, k, accept_sum + accept), (
            state.rho,
            state.beta,
            eta_norm,
            log_lik,
            accept,
        )

    (
        (final_state, final_key, total_accept),
        (rhos, betas, eta_norms, log_liks, accepts),
    ) = jax.lax.scan(body, (init_state, key, jnp.float64(0.0)), None, length=n_iters)
    accept_rate = total_accept / jnp.float64(n_iters)
    return final_state, final_key, rhos, betas, eta_norms, log_liks, accept_rate


def _run_chain_logit_draws_sem(
    gibbs_step, y_jax, init_state, key, n_iters, slice_width
):
    """Scan ``n_iters`` steps for SEM-logit; ``lam`` instead of ``rho``."""
    import jax
    import jax.numpy as jnp

    def body(carry, _):
        state, k, accept_sum = carry
        k, step_key = jax.random.split(k)
        state, accept = gibbs_step(state, step_key, slice_width)
        log_lik = _logit_loglik_pointwise_jax_op(y_jax, state.eta)
        eta_norm = state.eta @ state.eta
        return (state, k, accept_sum + accept), (
            state.lam,
            state.beta,
            eta_norm,
            log_lik,
            accept,
        )

    (
        (final_state, final_key, total_accept),
        (lams, betas, eta_norms, log_liks, accepts),
    ) = jax.lax.scan(body, (init_state, key, jnp.float64(0.0)), None, length=n_iters)
    accept_rate = total_accept / jnp.float64(n_iters)
    return final_state, final_key, lams, betas, eta_norms, log_liks, accept_rate


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
    cholmodjax_pattern=None,
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
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

    gibbs_step = _make_gibbs_step_with_data(
        y_jax=y_jax,
        X_jax=X_jax,
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        cholmodjax_pattern=cholmodjax_pattern,
    )

    init_states = _stack_chain_inits(inits, JAXLogitGibbsState, "rho")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._progress import GibbsProgressBarManager

    # Slice width for ρ — fixed at 0.2 (slice is robust to width choice)
    slice_width_arr = jnp.full(chains, jnp.float64(0.2))

    # ── Phase 1: warmup — fori_loop ──
    # The iteration count must be a Python int (not a JAX traced value)
    # because jax.lax.fori_loop requires a concrete length.
    adapt_window = max(50, tune // 10) if tune > 0 else 50

    def _make_warmup_fn(n_iters: int):
        """Create a JIT-compiled warmup function with baked-in iteration count."""
        return jax.jit(
            lambda s, k, w: jax.vmap(
                lambda s_, k_, w_: _run_chain_logit_warmup(
                    gibbs_step, s_, k_, n_iters, w_
                )
            )(s, k, w)
        )

    # Pre-compile the main warmup function for the standard window size
    warmup_fn = _make_warmup_fn(adapt_window)
    _warmup_cache: dict[int, object] = {adapt_window: warmup_fn}

    def _get_warmup_fn(n_iters: int):
        if n_iters not in _warmup_cache:
            _warmup_cache[n_iters] = _make_warmup_fn(n_iters)
        return _warmup_cache[n_iters]

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

        # ── Phase 1: warmup ──
        state = init_states
        keys = warmup_keys
        iter_done = 0
        while iter_done < tune:
            step = min(adapt_window, tune - iter_done)
            fn = _get_warmup_fn(step)
            state, keys = fn(state, keys, slice_width_arr)
            jax.block_until_ready(state.rho)
            iter_done += step

            if pm is not None:
                for c in range(chains):
                    pm.update(c, iter_done - 1, tuning=True, accept=None)

        final_warm_states = state

        # ── Phase 2: post-warmup draws — single scan ──
        draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

        draws_fn = jax.jit(
            lambda s, k, w: jax.vmap(
                lambda s_, k_, w_: _run_chain_logit_draws_sar(
                    gibbs_step, y_jax, s_, k_, draws, w_
                )
            )(s, k, w)
        )

        state, keys, rhos, betas, eta_norms, log_liks, accept_rates = draws_fn(
            final_warm_states, draw_keys, slice_width_arr
        )
        jax.block_until_ready(state.rho)

        if pm is not None:
            for c in range(chains):
                pm.update(c, tune + draws - 1, tuning=False, accept=None)

    # Convert to numpy
    rhos = np.asarray(rhos)
    betas = np.asarray(betas)
    eta_norms = np.asarray(eta_norms)
    log_liks = np.asarray(log_liks)
    accept_rates = np.asarray(accept_rates)

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
                "mh_accept_rate": float(accept_rates[c]),
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
    cholmodjax_pattern=None,
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
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

    gibbs_step = _make_gibbs_step_with_data_sem(
        y_jax=y_jax,
        X_jax=X_jax,
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        cholmodjax_pattern=cholmodjax_pattern,
    )

    init_states = _stack_chain_inits(inits, JAXSEMLogitGibbsState, "lam")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._progress import GibbsProgressBarManager

    # Slice width for λ — fixed at 0.2 (slice is robust to width choice)
    slice_width_arr = jnp.full(chains, jnp.float64(0.2))
    adapt_window = max(50, tune // 10) if tune > 0 else 50

    def _make_warmup_fn(n_iters: int):
        """Create a JIT-compiled warmup function with baked-in iteration count."""
        return jax.jit(
            lambda s, k, w: jax.vmap(
                lambda s_, k_, w_: _run_chain_logit_warmup(
                    gibbs_step, s_, k_, n_iters, w_
                )
            )(s, k, w)
        )

    warmup_fn = _make_warmup_fn(adapt_window)
    _warmup_cache: dict[int, object] = {adapt_window: warmup_fn}

    def _get_warmup_fn(n_iters: int):
        if n_iters not in _warmup_cache:
            _warmup_cache[n_iters] = _make_warmup_fn(n_iters)
        return _warmup_cache[n_iters]

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

        # ── Phase 1: warmup ──
        state = init_states
        keys = warmup_keys
        iter_done = 0
        while iter_done < tune:
            step = min(adapt_window, tune - iter_done)
            fn = _get_warmup_fn(step)
            state, keys = fn(state, keys, slice_width_arr)
            jax.block_until_ready(state.lam)
            iter_done += step

            if pm is not None:
                for c in range(chains):
                    pm.update(c, iter_done - 1, tuning=True, accept=None)

        final_warm_states = state

        # ── Phase 2: post-warmup draws — single scan ──
        draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

        draws_fn = jax.jit(
            lambda s, k, w: jax.vmap(
                lambda s_, k_, w_: _run_chain_logit_draws_sem(
                    gibbs_step, y_jax, s_, k_, draws, w_
                )
            )(s, k, w)
        )

        state, keys, lams, betas, eta_norms, log_liks, accept_rates = draws_fn(
            final_warm_states, draw_keys, slice_width_arr
        )
        jax.block_until_ready(state.lam)

        if pm is not None:
            for c in range(chains):
                pm.update(c, tune + draws - 1, tuning=False, accept=None)

    # Convert to numpy
    lams = np.asarray(lams)
    betas = np.asarray(betas)
    eta_norms = np.asarray(eta_norms)
    log_liks = np.asarray(log_liks)
    accept_rates = np.asarray(accept_rates)

    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "lam": lams[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy(),
                "mh_accept_rate": float(accept_rates[c]),
            }
        )
    return results
