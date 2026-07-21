r"""JAX-accelerated Gibbs sampler for reduced-form SAR Negative Binomial.

The reduced-form model has no latent η field and no σ² parameter:

.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
    \mu = \exp\{(I - \rho W)^{-1} X\beta\}

This is the canonical spatial-econometric SAR-NB specification (LeSage &
Pace 2009).

Architecture
------------
The sampler uses a Python loop that calls a JIT-compiled Gibbs step
for each iteration.  PG draws use ``jax.pure_callback`` to call the
exact C extension ``random_polyagamma``, which produces exact PG(h, z)
draws for any h (integer or non-integer).  This eliminates the
systematic ~0.5–1% mean bias of the Gamma-series approximation that
caused α to collapse over many Gibbs iterations.

A ``jax.lax.scan``-based runner is not used because ``pure_callback``
requires a host round-trip per call, which defeats the purpose of scan.
The Python loop compiles the Gibbs step once and calls it repeatedly,
amortising the JIT overhead.

- **PG sampling**: Exact draws via ``jax.pure_callback`` calling the
  C extension ``random_polyagamma``.
- **ρ draw**: JAX slice sampling with a shift-invert Krylov basis.
  The basis is built once per sweep at the current ρ via LU
  factorisation; each slice-candidate density evaluation is a cheap
  O(m·n·k) Horner polynomial — no linear solve, no autodiff needed.
- **β draw**: Conjugate Gaussian with intercept reparameterisation
  δ₀ = β₀/(1−ρ) to break the ρ–β₀ posterior correlation.
- **α draw**: JAX-compiled slice sampling on log(α).

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.
"""

from __future__ import annotations

import numpy as np

from bayespecon._jax_dispatch import ensure_x64

from .._utils._jax_polyagamma import jax_polyagamma

# ---------------------------------------------------------------------------
# Krylov basis: build and evaluate
# ---------------------------------------------------------------------------


def _build_sparse_ctx(W_sparse, n) -> dict:
    """Build the sparse klujax solve context for ``I − ρW`` (never densify W).

    Returns the constant COO pattern of ``I − ρW`` with aligned value vectors
    (so ``Ax(ρ) = eye_vals − ρ·w_vals``), a cached klujax symbolic
    factorisation, and a BCOO ``W`` for sparse matvecs.  The symbolic
    analysis (AMD ordering + elimination tree) is done once and reused for
    every numeric factorisation across the whole run.
    """
    import klujax
    from jax.experimental import sparse as jsparse

    from ._flow_jax import build_sar_pattern

    pat = build_sar_pattern(W_sparse, n)
    pat["symbolic"] = klujax.analyze(pat["Ai"], pat["Aj"], n)
    pat["W_bcoo"] = jsparse.BCOO.from_scipy_sparse(W_sparse.tocsr())
    return pat


def _make_sparse_solvers(sparse_ctx):
    """Build klujax solve closures over a sparse context (W never densified).

    Returns ``(factor_at, solve_num, matvec_W, solve_at)`` where

    - ``factor_at(rho)`` → numeric klujax factorisation of ``I − ρW`` (reusable
      across right-hand sides via ``solve_num``),
    - ``solve_num(numeric, rhs)`` → ``(I − ρW)⁻¹ rhs`` reusing a factorisation,
    - ``matvec_W(v)`` → ``W @ v`` via BCOO,
    - ``solve_at(rho, rhs)`` → one-shot ``(I − ρW)⁻¹ rhs`` (factor + solve).

    All closures close over the cached symbolic factorisation, so the AMD
    ordering / elimination tree is computed once for the whole run.
    """
    import jax.numpy as jnp
    import klujax

    Ai = jnp.asarray(sparse_ctx["Ai"])
    Aj = jnp.asarray(sparse_ctx["Aj"])
    eye_vals = jnp.asarray(sparse_ctx["eye_vals"])
    w_vals = jnp.asarray(sparse_ctx["w_vals"])
    symbolic = sparse_ctx["symbolic"]
    W_bcoo = sparse_ctx["W_bcoo"]

    def factor_at(rho):
        return klujax.factor(Ai, Aj, eye_vals - rho * w_vals, symbolic)

    def solve_num(numeric, rhs):
        return klujax.solve_with_numeric(numeric, rhs, symbolic)

    def matvec_W(v):
        return W_bcoo @ v

    def solve_at(rho, rhs):
        return solve_num(factor_at(rho), rhs)

    return factor_at, solve_num, matvec_W, solve_at


def _build_krylov_basis_jax(numeric_c, X_jax, solve_num, matvec_W, n, k, degree):
    """Build a shift-invert Krylov basis at ρ_c in JAX (sparse).

    Given a *single* klujax numeric factorisation ``numeric_c`` of
    ``A_c = I − ρ_c W`` (built once by the caller and reused), solves (m+1)
    RHS to build ``V_stack[j] = A_c⁻¹ (W V_{j-1})`` for ``j = 1..m`` with
    ``V_0 = A_c⁻¹ X``.  ``W`` is never densified: the ``W @ V_j`` products go
    through the sparse ``matvec_W`` (BCOO) and the solves through
    ``solve_num`` (klujax ``solve_with_numeric``, reusing ``numeric_c``).

    Parameters
    ----------
    numeric_c : klujax numeric handle
        Numeric factorisation of ``I − ρ_c W`` at the basis centre ρ_c.
    X_jax : jax.numpy.ndarray, shape (n, k)
        Design matrix.
    solve_num : callable ``(numeric, rhs) -> A_c⁻¹ rhs``
        klujax ``solve_with_numeric`` closure (reuses one factorisation).
    matvec_W : callable ``(v) -> W @ v``
        Sparse (BCOO) matvec.
    n, k : int
        Spatial units and regression coefficients.
    degree : int
        Krylov degree m (number of correction terms beyond V_0).

    Returns
    -------
    V_stack : jax.numpy.ndarray, shape (m+1, n, k)
        Krylov basis vectors.
    """
    import jax.numpy as jnp

    m = degree
    V_stack = jnp.empty((m + 1, n, k), dtype=jnp.float64)

    # V_0 = A_c⁻¹ X
    V_stack = V_stack.at[0].set(solve_num(numeric_c, X_jax))

    # V_{j+1} = A_c⁻¹ (W @ V_j)
    for j in range(m):
        Wv = matvec_W(V_stack[j])
        V_stack = V_stack.at[j + 1].set(solve_num(numeric_c, Wv))

    return V_stack


def _eval_U_from_basis_jax(V_stack, drho):
    """Evaluate U(ρ_c + Δρ) ≈ Σ (Δρ)ʲ V_j via Horner's method.

    Parameters
    ----------
    V_stack : jax.numpy.ndarray, shape (m+1, n, k)
        Krylov basis vectors.
    drho : jax.numpy.ndarray (scalar)
        Δρ = ρ − ρ_c.

    Returns
    -------
    U : jax.numpy.ndarray, shape (n, k)
        Approximate (I − ρW)⁻¹ X.
    """

    m = V_stack.shape[0] - 1
    # Horner: V_m + drho*(V_{m-1} + drho*(... + drho*V_0))
    result = V_stack[m]
    for j in range(m - 1, -1, -1):
        result = V_stack[j] + drho * result
    return result


# ---------------------------------------------------------------------------
# β-marginalised ρ log-density (Krylov-accelerated)
# ---------------------------------------------------------------------------


def _rho_log_density_marginal_jax(
    rho_val,
    V_stack,
    rho_basis,
    omega,
    y_jax,
    alpha,
    V0_inv_diag,
    mu0,
    intercept_col,
    krylov_dmax,
    X_jax=None,
    solve_at=None,
):
    """β-marginalised log-density of ρ for the reduced form.

    Evaluates U(ρ) via the Krylov basis when |Δρ| ≤ dmax,
    otherwise falls back to a direct sparse klujax solve (``solve_at``).
    This matches the NumPy path's direct-solve fallback and ensures the
    slice sampler can explore the full ρ support.

    The density is:

    .. math::

        \\log p(\\rho \\mid \\omega, \\alpha, y) =
        -\\frac{1}{2} \\log|M| - \\frac{1}{2}(r^T \\Omega r - w^T w)
        + \\text{Jacobian}

    where U = (I−ρW)⁻¹X, M = V₀⁻¹ + UᵀΩU, s = κ/ω + log(α),
    r = s − Uμ₀, w = L⁻¹v, v = UᵀΩr.

    No log|I−ρW| term — it cancels when β is marginalised out.
    """
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    k = V_stack.shape[2]

    # Evaluate U(ρ) via Krylov basis when within radius, else direct solve
    drho = rho_val - rho_basis
    use_basis = jnp.abs(drho) <= krylov_dmax

    U_krylov = _eval_U_from_basis_jax(V_stack, drho)

    # Direct sparse solve fallback (klujax; correct for any ρ)
    has_fallback = (X_jax is not None) and (solve_at is not None)
    if has_fallback:
        U_direct = solve_at(rho_val, X_jax)
        U = jnp.where(use_basis, U_krylov, U_direct)
    else:
        U = U_krylov

    # Intercept reparameterisation: δ₀ = β₀/(1−ρ)
    reparam = (intercept_col >= 0) & (jnp.abs(rho_val) > 1e-8)
    scale = 1.0 - rho_val

    # Apply reparameterisation: replace intercept column of U
    U_rp = jnp.where(
        reparam,
        U.at[:, intercept_col].set(1.0),
        U,
    )
    V0_inv_diag_rp = jnp.where(
        reparam,
        V0_inv_diag.at[intercept_col].set(V0_inv_diag[intercept_col] * scale * scale),
        V0_inv_diag,
    )
    mu0_rp = jnp.where(
        reparam,
        mu0.at[intercept_col].set(mu0[intercept_col] / scale),
        mu0,
    )

    # Working quantities
    log_alpha = jnp.log(alpha)
    kappa = 0.5 * (y_jax - alpha)
    s = kappa / omega + log_alpha
    r = s - U_rp @ mu0_rp

    # M = V₀⁻¹ + UᵀΩU  (k × k)
    Uw = U_rp * omega[:, None]  # (n, k)
    M = U_rp.T @ Uw
    M = M + jnp.diag(V0_inv_diag_rp)

    v = Uw.T @ r  # UᵀΩr

    # Cholesky of M
    M_reg = M + 1e-10 * jnp.eye(k)
    L_M = jnp.linalg.cholesky(M_reg)
    w = solve_triangular(L_M, v, lower=True)
    quad_pen = w @ w
    rOr = jnp.dot(r, omega * r)
    log_det_M = 2.0 * jnp.sum(jnp.log(jnp.diag(L_M)))

    result = -0.5 * log_det_M - 0.5 * (rOr - quad_pen)

    # Jacobian for intercept reparameterisation
    result = jnp.where(reparam, result + jnp.log(scale), result)

    # Reject if outside Krylov radius AND no fallback available
    if not has_fallback:
        result = jnp.where(use_basis, result, -jnp.inf)

    return result


# ---------------------------------------------------------------------------
# JAX slice sampler for ρ
# ---------------------------------------------------------------------------


def _slice_sample_rho_jax(
    rho_current,
    V_stack,
    rho_basis,
    omega,
    y_jax,
    alpha,
    V0_inv_diag,
    mu0,
    intercept_col,
    rho_lower,
    rho_upper,
    krylov_dmax,
    slice_width,
    key,
    X_jax=None,
    solve_at=None,
):
    """1-D slice sampler for ρ using jax.lax.while_loop.

    Parameters
    ----------
    rho_current : jax.numpy.ndarray (scalar)
        Current ρ value.
    V_stack, rho_basis : Krylov basis at current ρ.
    omega, y_jax, alpha, V0_inv_diag, mu0, intercept_col :
        Passed to the log-density.
    rho_lower, rho_upper : float
        Support bounds.
    krylov_dmax : float
        Maximum |Δρ| for Krylov approximation.
    slice_width : jax.numpy.ndarray (scalar)
        Stepping-out width for the slice sampler.
    key : jax.random.PRNGKey
        JAX random key.
    X_jax, solve_at :
        Passed to the log-density for the direct sparse-solve fallback
        when candidates are outside the Krylov radius.

    Returns
    -------
    rho_new : jax.numpy.ndarray (scalar)
        New ρ value.
    """
    import jax
    import jax.numpy as jnp

    def log_density(rho_val):
        return _rho_log_density_marginal_jax(
            rho_val,
            V_stack,
            rho_basis,
            omega,
            y_jax,
            alpha,
            V0_inv_diag,
            mu0,
            intercept_col,
            krylov_dmax,
            X_jax=X_jax,
            solve_at=solve_at,
        )

    log_y0 = log_density(rho_current)

    # Draw vertical level
    key, subkey = jax.random.split(key)
    log_u = log_y0 + jnp.log(jax.random.uniform(subkey, dtype=jnp.float64))

    # Stepping out: initialise [L, R]
    key, subkey = jax.random.split(key)
    u_rand = jax.random.uniform(subkey, dtype=jnp.float64)
    w = slice_width
    L = jnp.maximum(rho_current - u_rand * w, rho_lower)
    R = jnp.minimum(L + w, rho_upper)

    # Step out left
    def step_out_left_cond(carry):
        L_val, _ = carry
        return (L_val > rho_lower) & (log_density(L_val) > log_u)

    def step_out_left_body(carry):
        L_val, _ = carry
        return (jnp.maximum(L_val - w, rho_lower), jnp.float64(0.0))

    L_final, _ = jax.lax.while_loop(
        step_out_left_cond, step_out_left_body, (L, jnp.float64(0.0))
    )

    # Step out right
    def step_out_right_cond(carry):
        R_val, _ = carry
        return (R_val < rho_upper) & (log_density(R_val) > log_u)

    def step_out_right_body(carry):
        R_val, _ = carry
        return (jnp.minimum(R_val + w, rho_upper), jnp.float64(0.0))

    R_final, _ = jax.lax.while_loop(
        step_out_right_cond, step_out_right_body, (R, jnp.float64(0.0))
    )

    # Shrinkage
    def shrink_cond(carry):
        _, _, _, _, done = carry
        return ~done

    def shrink_body(carry):
        L_val, R_val, key_val, x_best, _ = carry
        key_val, subkey = jax.random.split(key_val)
        x_new = L_val + jax.random.uniform(subkey, dtype=jnp.float64) * (R_val - L_val)
        log_dens_new = log_density(x_new)
        accepted = log_dens_new > log_u
        L_new = jnp.where(x_new < rho_current, x_new, L_val)
        R_new = jnp.where(x_new >= rho_current, x_new, R_val)
        collapsed = (R_new - L_new) < 1e-15
        done = accepted | collapsed
        x_best = jnp.where(accepted, x_new, x_best)
        return (L_new, R_new, key_val, x_best, done)

    _, _, _, rho_new, _ = jax.lax.while_loop(
        shrink_cond,
        shrink_body,
        (L_final, R_final, key, rho_current, jnp.bool_(False)),
    )

    return rho_new


# ---------------------------------------------------------------------------
# Core Gibbs step builder
# ---------------------------------------------------------------------------


def _make_reduced_gibbs_step(
    y_jax,
    X_jax,
    sparse_ctx,
    n,
    k,
    priors,
    intercept_col=0,
    krylov_degree=8,
    krylov_dmax=0.15,
):
    """Build a JIT-compiled reduced-form Gibbs step (ω → ρ → β → α).

    The PG ω draw uses ``jax.pure_callback`` to call the exact C
    extension ``random_polyagamma``, which produces exact PG(h, z)
    draws for any h (integer or non-integer).  This eliminates the
    systematic bias of the Gamma-series approximation that caused
    α to collapse over many Gibbs iterations.

    Parameters
    ----------
    y_jax : jax.numpy.ndarray of shape (n,)
        Response vector (JAX array).
    X_jax : jax.numpy.ndarray of shape (n, k)
        Design matrix (JAX array).
    sparse_ctx : dict
        Sparse klujax context from :func:`_build_sparse_ctx`: keys ``Ai``,
        ``Aj``, ``eye_vals``, ``w_vals`` (aligned COO of ``I − ρW``),
        ``symbolic`` (cached klujax symbolic factorisation) and ``W_bcoo``
        (BCOO ``W`` for matvecs).  ``W`` is never densified.
    n : int
        Number of spatial units.
    k : int
        Number of regression coefficients.
    priors : ReducedGibbsPriors
        Prior hyperparameters.
    intercept_col : int, default 0
        Column index of the intercept in X. Set to -1 to disable
        the reparameterisation.
    krylov_degree : int, default 8
        Krylov basis degree m for the shift-invert polynomial
        approximation of (I − ρW)⁻¹X.
    krylov_dmax : float, default 0.15
        Maximum |Δρ| for which the Krylov basis is used.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key, slice_width) -> (new_state, accept)

        where ``state`` is a dict with keys ``beta``, ``rho``,
        ``alpha``, ``omega``, ``key`` is a JAX PRNG key,
        ``slice_width`` is a ``jnp.float64`` stepping-out width,
        and ``accept`` is always ``jnp.float64(1.0)`` (slice
        sampling always accepts).
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    ensure_x64()

    # ── Sparse klujax solve closures (W is never densified) ──
    _factor_at, _solve_num, _matvec_W, _solve_at = _make_sparse_solvers(sparse_ctx)

    # Prior hyperparameters
    beta_mu = priors.beta_mu
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = jnp.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / jnp.asarray(beta_sigma, dtype=jnp.float64) ** 2
    if np.isscalar(beta_mu):
        mu0 = jnp.full(k, float(beta_mu))
    else:
        mu0 = jnp.asarray(beta_mu, dtype=jnp.float64)

    rho_lower_jax = jnp.float64(priors.rho_lower)
    rho_upper_jax = jnp.float64(priors.rho_upper)
    alpha_sigma_jax = jnp.float64(priors.alpha_sigma)
    alpha_nu_jax = jnp.float64(priors.alpha_nu)

    _intercept_col = intercept_col
    _krylov_degree = krylov_degree
    _krylov_dmax = jnp.float64(krylov_dmax)

    @jax.jit
    def gibbs_step(state, key, slice_width):
        """One reduced-form Gibbs sweep: ω → ρ (slice+Krylov) → β → α.

        Parameters
        ----------
        state : dict
            Current state with keys ``beta``, ``rho``, ``alpha``,
            ``omega``.
        key : jax.random.PRNGKey
            JAX random key.
        slice_width : jax.numpy.float64
            Stepping-out width for the ρ slice sampler.

        Returns
        -------
        new_state : dict
            Updated state.
        accept : jax.numpy.float64
            Always 1.0 (slice sampling always accepts).
        """
        beta = state["beta"]
        rho = state["rho"]
        alpha = state["alpha"]

        key_rho, key_beta, key_alpha = jax.random.split(key, 3)

        # ── Block 0: ω ~ PG(y + α, η) ──
        # Factorise A = (I − ρW) once at the current ρ (sparse klujax); reuse
        # it for both η and the Krylov basis (W never densified).
        numeric_c = _factor_at(rho)
        eta = _solve_num(numeric_c, X_jax @ beta)
        key, key_pg = jax.random.split(key)
        h = jnp.maximum(y_jax + alpha, 1e-3)
        z = jnp.clip(eta - jnp.log(alpha), -20.0, 20.0)
        omega = jax_polyagamma(h, z, key=key_pg, method="callback")

        # ── Block 1: ρ — slice sampling with Krylov basis ──
        V_stack = _build_krylov_basis_jax(
            numeric_c, X_jax, _solve_num, _matvec_W, n, k, _krylov_degree
        )

        rho_new = _slice_sample_rho_jax(
            rho_current=rho,
            V_stack=V_stack,
            rho_basis=rho,
            omega=omega,
            y_jax=y_jax,
            alpha=alpha,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            intercept_col=_intercept_col,
            rho_lower=rho_lower_jax,
            rho_upper=rho_upper_jax,
            krylov_dmax=_krylov_dmax,
            slice_width=slice_width,
            key=key_rho,
            X_jax=X_jax,
            solve_at=_solve_at,
        )

        # ── Block 2: β | ρ, ω, α, y — conjugate normal ──
        drho_new = rho_new - rho
        use_basis = jnp.abs(drho_new) <= _krylov_dmax
        Xtilde_krylov = _eval_U_from_basis_jax(V_stack, drho_new)
        Xtilde_direct = _solve_at(rho_new, X_jax)
        Xtilde = jnp.where(use_basis, Xtilde_krylov, Xtilde_direct)

        reparam_beta = (_intercept_col >= 0) & (jnp.abs(rho_new) > 1e-8)
        scale_beta = 1.0 - rho_new

        Xtilde_rp = jnp.where(
            reparam_beta,
            Xtilde.at[:, _intercept_col].set(1.0),
            Xtilde,
        )
        V0_inv_diag_rp_beta = jnp.where(
            reparam_beta,
            V0_inv_diag.at[_intercept_col].set(
                V0_inv_diag[_intercept_col] * scale_beta * scale_beta
            ),
            V0_inv_diag,
        )
        mu0_rp_beta = jnp.where(
            reparam_beta,
            mu0.at[_intercept_col].set(mu0[_intercept_col] / scale_beta),
            mu0,
        )

        kappa = 0.5 * (y_jax - alpha)
        log_alpha_val = jnp.log(alpha)

        Xt_omega = Xtilde_rp * omega[:, None]
        Sigma_beta_inv = Xt_omega.T @ Xtilde_rp
        Sigma_beta_inv = Sigma_beta_inv + jnp.diag(V0_inv_diag_rp_beta)

        rhs = (
            Xtilde_rp.T @ (kappa + omega * log_alpha_val)
            + V0_inv_diag_rp_beta * mu0_rp_beta
        )

        Sigma_beta_inv_reg = Sigma_beta_inv + 1e-10 * jnp.eye(k)
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv_reg)
        m_beta = cho_solve((L_beta, True), rhs)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        delta = solve_triangular(L_beta.T, z_beta, lower=False)
        beta_draw = m_beta + delta

        beta_new = jnp.where(
            reparam_beta,
            beta_draw.at[_intercept_col].set(beta_draw[_intercept_col] * scale_beta),
            beta_draw,
        )

        # ── Block 3: α | y, η — JAX slice sampling ──
        eta_new = Xtilde @ beta_new
        alpha_new = _sample_alpha_jax_reduced(
            eta_new, y_jax, alpha, alpha_sigma_jax, alpha_nu_jax, key_alpha
        )

        new_state = {
            "beta": beta_new,
            "rho": rho_new,
            "alpha": alpha_new,
            "omega": omega,
        }
        return new_state, jnp.float64(1.0)

    return gibbs_step


def _sample_alpha_jax_reduced(eta, y_jax, alpha_current, alpha_sigma, alpha_nu, key):
    """Sample α using JAX-compiled slice sampling for the reduced form.

    Parameters
    ----------
    eta : jax.numpy.ndarray
        Current latent field (n,).
    y_jax : jax.numpy.ndarray
        Integer response vector.
    alpha_current : jax.numpy.ndarray
        Current α value (scalar).
    alpha_sigma : jax.numpy.ndarray
        Prior scale for α (Half-Student-t).
    alpha_nu : jax.numpy.ndarray
        Half-Student-t degrees of freedom for α.
    key : jax.random.PRNGKey
        JAX random key.

    Returns
    -------
    jax.numpy.ndarray
        New α value (scalar JAX array).
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jax_gammaln

    log_alpha = jnp.log(alpha_current)

    def log_density(log_a):
        """Log-density on the log(α) scale."""
        a = jnp.exp(log_a)
        mu = jnp.exp(eta)
        # NB log-likelihood
        log_lik = (
            jax_gammaln(y_jax + a)
            - jax_gammaln(a)
            + y_jax * jnp.log(jnp.maximum(mu / (mu + a), 1e-300))
            + a * jnp.log(jnp.maximum(a / (mu + a), 1e-300))
        )
        total_log_lik = jnp.sum(log_lik)
        # Half-Student-t prior on α
        log_prior = (
            -0.5
            * (alpha_nu + 1.0)
            * jnp.log1p((a * a) / (alpha_nu * alpha_sigma * alpha_sigma))
        )
        # Jacobian: d(α)/d(log α) = α, so log|J| = log(α) = log_a
        return log_a + total_log_lik + log_prior

    # Log-density at current point
    log_y0 = log_density(log_alpha)

    # Draw vertical level
    key, subkey = jax.random.split(key)
    log_u = log_y0 + jnp.log(jax.random.uniform(subkey, dtype=jnp.float64))

    # Slice bounds
    w = jnp.float64(1.0)
    lower_bound = jnp.float64(-4.0)
    upper_bound = jnp.float64(4.0)

    # Stepping out
    key, subkey = jax.random.split(key)
    u_rand = jax.random.uniform(subkey, dtype=jnp.float64)
    L = jnp.maximum(log_alpha - u_rand * w, lower_bound)
    R = jnp.minimum(L + w, upper_bound)

    # Step out left
    def step_out_left(carry):
        L_val, _ = carry
        L_new = jnp.maximum(L_val - w, lower_bound)
        return (L_new, jnp.float64(0.0))

    def should_step_left(carry):
        L_val, _ = carry
        return (L_val > lower_bound) & (log_density(L_val) > log_u)

    L_final, _ = jax.lax.while_loop(
        should_step_left, step_out_left, (L, jnp.float64(0.0))
    )

    # Step out right
    def step_out_right(carry):
        R_val, _ = carry
        R_new = jnp.minimum(R_val + w, upper_bound)
        return (R_new, jnp.float64(0.0))

    def should_step_right(carry):
        R_val, _ = carry
        return (R_val < upper_bound) & (log_density(R_val) > log_u)

    R_final, _ = jax.lax.while_loop(
        should_step_right, step_out_right, (R, jnp.float64(0.0))
    )

    # Shrinkage
    def shrink_while_cond(carry):
        _, _, _, _, done = carry
        return ~done

    def shrink_while_body(carry):
        L_val, R_val, key_val, x_best, _ = carry
        key_val, subkey = jax.random.split(key_val)
        x_new = L_val + jax.random.uniform(subkey, dtype=jnp.float64) * (R_val - L_val)
        log_dens_new = log_density(x_new)
        accepted = log_dens_new > log_u
        L_new = jnp.where(x_new < log_alpha, x_new, L_val)
        R_new = jnp.where(x_new >= log_alpha, x_new, R_val)
        collapsed = (R_new - L_new) < 1e-15
        done = accepted | collapsed
        x_best = jnp.where(accepted, x_new, x_best)
        return (L_new, R_new, key_val, x_best, done)

    _, _, _, log_alpha_new, _ = jax.lax.while_loop(
        shrink_while_cond,
        shrink_while_body,
        (L_final, R_final, key, log_alpha, jnp.bool_(False)),
    )

    return jnp.exp(log_alpha_new)


def _nb_loglik_pointwise_jax(y_jax, eta, alpha):
    """Pointwise NB log-likelihood as a pure-JAX op (vmap-safe)."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jax_gammaln

    mu = jnp.exp(eta)
    log_mu_ratio = jnp.log(jnp.maximum(mu / (mu + alpha), 1e-300))
    log_alpha_ratio = jnp.log(jnp.maximum(alpha / (mu + alpha), 1e-300))
    return (
        jax_gammaln(y_jax + alpha)
        - jax_gammaln(alpha)
        + y_jax * log_mu_ratio
        + alpha * log_alpha_ratio
    )


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def _run_chain_reduced(gibbs_step, init_state, key, n_iters, slice_width):
    """Run ``n_iters`` Gibbs steps in a Python loop.

    Uses a Python loop instead of ``jax.lax.scan`` because the PG draw
    uses ``jax.pure_callback``, which requires a host round-trip per
    iteration and cannot be used inside ``jax.lax.scan``.

    The Python loop calls the JIT-compiled ``gibbs_step`` once per
    iteration, which amortises the JIT overhead and only incurs one
    host round-trip per step for the PG draw.

    Parameters
    ----------
    gibbs_step : callable
        JIT-compiled Gibbs step ``(state, key, slice_width) -> (state, accept)``.
    init_state : dict
        Initial state with keys ``beta``, ``rho``, ``alpha``, ``omega``.
    key : jax.random.PRNGKey
        PRNG key.
    n_iters : int
        Number of iterations.
    slice_width : jax.numpy.float64
        Stepping-out width for the ρ slice sampler.

    Returns
    -------
    final_state : dict
        Final state after ``n_iters`` steps.
    final_key : jax.random.PRNGKey
        Final PRNG key.
    traces : tuple of ndarray
        Stacked traces of ``rho``, ``beta``, ``alpha``.
    """
    import jax

    state = init_state
    rho_list = []
    beta_list = []
    alpha_list = []

    for _ in range(n_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key, slice_width)
        rho_list.append(np.asarray(state["rho"]))
        beta_list.append(np.asarray(state["beta"]))
        alpha_list.append(np.asarray(state["alpha"]))

    final_state = state
    final_key = key
    traces = (
        np.stack(rho_list),
        np.stack(beta_list),
        np.stack(alpha_list),
    )
    return final_state, final_key, traces


def run_chains_jax_reduced(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    progressbar: bool = True,
    intercept_col: int = 0,
    krylov_degree: int = 8,
    krylov_dmax: float = 0.15,
    slice_width: float = 0.2,
) -> list[dict]:
    """Run multiple reduced-form SAR-NB Gibbs chains using JAX.

    Uses ``jax.pure_callback`` to call the exact C extension for PG
    draws, which produces exact PG(h, z) draws for any h (integer or
    non-integer).  This eliminates the systematic bias of the
    Gamma-series approximation that caused α to collapse over many
    Gibbs iterations.

    Parameters
    ----------
    y : ndarray, shape (n,)
    X : ndarray, shape (n, k)
    W_sparse : scipy.sparse matrix
    priors : ReducedGibbsPriors
    inits : list of ReducedGibbsState
    draws, tune : int
    thin : int, default 1
    jax_seeds : list of int, optional
    progressbar : bool, default True
    intercept_col : int, default 0
    krylov_degree : int, default 8
    krylov_dmax : float, default 0.15
    slice_width : float, default 0.2
        Stepping-out width for the ρ slice sampler.

    Returns
    -------
    list of dict
        One dict per chain with keys ``rho``, ``beta``, ``alpha``,
        ``log_lik``.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    from .._utils._progress import GibbsProgressBarManager

    chains = len(inits)
    n, k = X.shape

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    sparse_ctx = _build_sparse_ctx(W_sparse, n)

    slice_width_jax = jnp.float64(slice_width)

    if jax_seeds is None:
        jax_seeds = list(range(chains))

    chain_results = []

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sar_negbin",
    ) as pm:
        for c in range(chains):
            if pm is not None:
                pm.start_chain(c)

            key = jax.random.PRNGKey(jax_seeds[c])

            # Build the Gibbs step
            gibbs_step = _make_reduced_gibbs_step(
                y_jax=y_jax,
                X_jax=X_jax,
                sparse_ctx=sparse_ctx,
                n=n,
                k=k,
                priors=priors,
                intercept_col=intercept_col,
                krylov_degree=krylov_degree,
                krylov_dmax=krylov_dmax,
            )

            # Initialise state
            init_state = {
                "beta": jnp.asarray(inits[c].beta, dtype=jnp.float64),
                "rho": jnp.float64(inits[c].rho),
                "alpha": jnp.float64(inits[c].alpha),
                "omega": jnp.asarray(inits[c].omega, dtype=jnp.float64),
            }

            # ── Phase 1: warmup (no traces stored) ──
            if tune > 0:
                warmup_chunk = max(1, tune // 20)
                state = init_state
                iter_done = 0
                while iter_done < tune:
                    step = min(warmup_chunk, tune - iter_done)
                    state, key, _ = _run_chain_reduced(
                        gibbs_step, state, key, step, slice_width_jax
                    )
                    iter_done += step
                    if pm is not None:
                        pm.update(c, iter_done, tuning=True, accept=None)

            # ── Phase 2: draws (traces stored) ──
            draws_chunk = max(1, draws // 20)
            rho_list = []
            beta_list = []
            alpha_list = []
            iter_done = 0
            while iter_done < draws:
                step = min(draws_chunk, draws - iter_done)
                state, key, traces = _run_chain_reduced(
                    gibbs_step, state, key, step, slice_width_jax
                )
                rho_list.append(np.asarray(traces[0]))
                beta_list.append(np.asarray(traces[1]))
                alpha_list.append(np.asarray(traces[2]))
                iter_done += step
                if pm is not None:
                    pm.update(c, tune + iter_done, tuning=False, accept=None)

            if pm is not None:
                pm.update(c, tune + draws, tuning=False, accept=None)

            # Stack and thin
            rho_samples = np.concatenate(rho_list)
            beta_samples = np.concatenate(beta_list)
            alpha_samples = np.concatenate(alpha_list)
            if thin > 1:
                rho_samples = rho_samples[::thin]
                beta_samples = beta_samples[::thin]
                alpha_samples = alpha_samples[::thin]

            # Compute pointwise log-likelihood (NumPy, sparse solves)
            n_keep = rho_samples.shape[0]
            log_lik = np.empty((n_keep, n), dtype=np.float64)
            # Prefer klujax (cached symbolic analysis) over dense solve loop
            import scipy.sparse as _sp
            from scipy.special import gammaln

            from ..._jax_dispatch import _klujax_available

            if _klujax_available():
                import klujax

                I_coo = _sp.eye(n, format="coo")
                W_coo = W_sparse.tocoo()
                all_rows = np.concatenate([I_coo.row, W_coo.row])
                all_cols = np.concatenate([I_coo.col, W_coo.col])
                shape = (n, n)
                const_coo = _sp.coo_matrix(
                    (
                        np.concatenate([np.ones(I_coo.nnz), np.zeros(W_coo.nnz)]),
                        (all_rows, all_cols),
                    ),
                    shape=shape,
                )
                const_coo.sum_duplicates()
                w_coo = _sp.coo_matrix(
                    (
                        np.concatenate([np.zeros(I_coo.nnz), W_coo.data]),
                        (all_rows, all_cols),
                    ),
                    shape=shape,
                )
                w_coo.sum_duplicates()
                Ai = np.asarray(const_coo.row, dtype=np.int32)
                Aj = np.asarray(const_coo.col, dtype=np.int32)
                const_vals = np.asarray(const_coo.data, dtype=np.float64)
                w_vals = np.asarray(w_coo.data, dtype=np.float64)
                symbolic = klujax.analyze(Ai, Aj, n)
                for i in range(n_keep):
                    rho_i = rho_samples[i]
                    beta_i = beta_samples[i]
                    Ax = const_vals - rho_i * w_vals
                    eta_i = np.asarray(
                        klujax.solve_with_symbol(Ai, Aj, Ax, X @ beta_i, symbolic),
                        dtype=np.float64,
                    )
                    alpha_i = alpha_samples[i]
                    mu = np.exp(eta_i)
                    log_lik[i] = (
                        gammaln(y + alpha_i)
                        - gammaln(alpha_i)
                        + y * np.log(np.maximum(mu / (mu + alpha_i), 1e-300))
                        + alpha_i * np.log(np.maximum(alpha_i / (mu + alpha_i), 1e-300))
                    )
            else:
                from ..._ops._backend import _solve_sparse_vector

                I_sp = _sp.eye(n, format="csc")
                for i in range(n_keep):
                    rho_i = rho_samples[i]
                    beta_i = beta_samples[i]
                    A_csc = (I_sp - rho_i * W_sparse).tocsc()
                    eta_i = _solve_sparse_vector(A_csc, X @ beta_i)
                    alpha_i = alpha_samples[i]
                    mu = np.exp(eta_i)
                    log_lik[i] = (
                        gammaln(y + alpha_i)
                        - gammaln(alpha_i)
                        + y * np.log(np.maximum(mu / (mu + alpha_i), 1e-300))
                        + alpha_i * np.log(np.maximum(alpha_i / (mu + alpha_i), 1e-300))
                    )

            chain_results.append(
                {
                    "rho": rho_samples,
                    "beta": beta_samples,
                    "alpha": alpha_samples,
                    "log_lik": log_lik,
                }
            )

    return chain_results
