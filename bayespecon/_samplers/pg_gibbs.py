"""Pólya–Gamma Gibbs sampler for structural-form SAR-NB.

Orchestrates the 6-block Gibbs sweep:
  1. ω | η, α          (PG augmentation)
  2. η | ω, ρ, β, σ²  (spatial-normal draw)
  3. β | η, ρ, σ²      (conjugate normal)
  4. σ² | η, ρ, β      (conjugate inverse-gamma)
  5. ρ | β, σ², ω, y   (collapsed 1-D slice, η integrated out)
  6. α | y, η           (1-D slice on log-likelihood)

The structural form parameterises the latent log-mean as
``η = ρ W η + X β + ν`` with ``ν ~ N(0, σ² I)``, and augments the NB
likelihood with Pólya–Gamma auxiliary variables to obtain fully
conjugate Gibbs updates for η, β, and σ².

The ρ update uses a **collapsed** (marginal) posterior that integrates
out η, avoiding the slow mixing that arises from conditioning on the
current η draw.  The collapsed log-density is:

    log p(ρ | ·) = log|I - ρW| - ½ log|P_η| + ½ rhs^T P_η⁻¹ rhs

where P_η = A_ρ^T A_ρ / σ² + diag(ω) and rhs = A_ρ^T Xβ / σ² + κ.

**Backend dispatch**: The sampler supports three computational paths,
selected via ``gibbs_method`` in :meth:`SARNegBinLatent.fit`:

- ``"factorize"``: CHOLMOD/splu factorisation (exact, O(nnz^{1.5})).
- ``"iterative"``: CG + Lanczos + Chebyshev (approximate, avoids
  factorisation).
- ``"jax_dense"``: JAX dense matvec + vmap (3–4× faster for single
  draws, 20–27× per-draw when batching Chebyshev draws).  Requires
  JAX with float64 enabled.  Viable for n ≤ ~10 000 on machines
  with ≥ 32 GB RAM (the dense matrices need ~800 MB at n=10 000).

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.

Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._jax_dispatch import _eqx_available
from ._polyagamma import sample_polyagamma
from ._slice import (
    SliceWidthState,
    slice_sample_1d,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from ._spatial_normal import (
    CholmodFactor,
    cg_solve,
    chebyshev_sample,
    jax_build_P_dense,
    jax_chebyshev_sample,
    lanczos_logdet,
    sample_spatial_normal,
)

# ---------------------------------------------------------------------------
# Data classes for state, priors, and precomputed cache
# ---------------------------------------------------------------------------


@dataclass
class GibbsState:
    """Mutable state carried through one Gibbs sweep (Python-loop path).

    All arrays are numpy arrays; scalars are Python floats.
    For the JAX-dense path, use :class:`JAXGibbsState` instead.
    """

    eta: np.ndarray  # (n,) latent field
    beta: np.ndarray  # (k,) regression coefficients
    sigma2: float  # residual variance σ²
    rho: float  # spatial autoregressive parameter
    alpha: float  # NB dispersion parameter
    omega: np.ndarray  # (n,) PG auxiliary variables

    def to_jax(self) -> "JAXGibbsState":
        """Convert to a JAX-compatible :class:`JAXGibbsState`."""
        import jax.numpy as jnp

        return JAXGibbsState(
            eta=jnp.asarray(self.eta, dtype=jnp.float64),
            beta=jnp.asarray(self.beta, dtype=jnp.float64),
            sigma2=jnp.float64(self.sigma2),
            rho=jnp.float64(self.rho),
            alpha=jnp.float64(self.alpha),
            omega=jnp.asarray(self.omega, dtype=jnp.float64),
        )


if _eqx_available():
    import equinox as eqx
    import jax

    class JAXGibbsState(eqx.Module):
        """JAX-compatible Gibbs sampler state (used by the JAX-dense path).

        An ``equinox.Module`` that holds JAX arrays and is automatically
        registered as a PyTree, so it can be passed through ``@jax.jit``
        and ``@eqx.filter_jit`` boundaries without manual registration.

        For the Python-loop path, use :class:`GibbsState` instead.
        """

        eta: jax.Array
        beta: jax.Array
        sigma2: jax.Array
        rho: jax.Array
        alpha: jax.Array
        omega: jax.Array

        def to_numpy(self) -> GibbsState:
            """Convert to a numpy-based :class:`GibbsState`."""
            return GibbsState(
                eta=np.asarray(self.eta),
                beta=np.asarray(self.beta),
                sigma2=float(self.sigma2),
                rho=float(self.rho),
                alpha=float(self.alpha),
                omega=np.asarray(self.omega),
            )

else:

    class JAXGibbsState:  # type: ignore[no-redef]
        """Stub when equinox is not installed — should never be instantiated."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "equinox is required for the JAX-dense Gibbs sampler path. "
                "Install with: pip install equinox"
            )


@dataclass
class GibbsPriors:
    """Prior hyperparameters for the SAR-NB Gibbs sampler.

    All priors are weakly informative by default, matching the
    ``SARNegativeBinomial`` defaults.
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    sigma_sigma: float = 10.0  # HalfNormal scale for σ
    alpha_sigma: float = 10.0  # HalfNormal scale for α
    rho_lower: float = -0.999
    rho_upper: float = 0.999


class GibbsCache(NamedTuple):
    """Precomputed data that doesn't change across sweeps.

    Some fields (like ``XtX``) are constant; others (like the
    logdet callable) depend on ρ but are precomputed once at
    model construction time.

    When CHOLMOD is available, ``cholmod_factor`` holds a
    ``CholmodFactor`` with the symbolic analysis for the precision
    matrix sparsity pattern.  This allows ``cholesky_inplace`` to
    skip the symbolic phase on each ρ candidate in the slice sampler,
    giving a 5–9× speedup over ``splu``.

    The precomputed matrix pieces ``W_sym_over_s2`` and ``WtW_over_s2``
    allow the precision matrix P to be constructed as
    ``P = base - ρ * W_sym_over_s2 + ρ² * WtW_over_s2``
    instead of the expensive ``A_ρ^T @ A_ρ / σ²`` product.

    When ``solve_method="cg"`` or ``logdet_P_method="lanczos"``, the
    decoupled (iterative) path is used in the ρ slice sampler:
    CG replaces the factorisation-based solve, and Lanczos-based
    stochastic estimation replaces the factorisation-based log|P_η|.
    This avoids the O(nnz^{1.5}) factorisation cost for large n.

    When ``solve_method="jax_dense"``, the JAX-accelerated path is
    used: dense matvec + vmap over Lanczos probes and Chebyshev
    draws.  This gives 3–4× speedup for single draws and 20–27×
    per-draw when batching Chebyshev draws, for n ≤ ~10 000.
    """

    W_sparse: sp.csr_matrix
    XtX: np.ndarray  # (k, k) = X^T X
    logdet_fn: Callable[[float], float]  # log|I - rho*W| callable
    rho_lower: float
    rho_upper: float
    cholmod_factor: CholmodFactor | None = None
    W_sym_over_s2: sp.csr_matrix | None = None  # (W + W^T), divided by σ² at runtime
    WtW_over_s2: sp.csr_matrix | None = None  # W^T W, divided by σ² at runtime
    solve_method: str = "cholmod"  # "cholmod" | "splu" | "cg" | "jax_dense"
    logdet_P_method: str = "cholmod"  # "cholmod" | "lanczos" | "jax_dense"
    sample_method: str = "cholmod"  # "cholmod" | "splu" | "chebyshev" | "jax_dense"
    lanczos_n_probes: int = 10  # probe vectors for Lanczos logdet
    lanczos_deg: int = 30  # Lanczos iteration depth
    chebyshev_degree: int = 30  # Chebyshev polynomial degree for η draw
    # JAX dense backend fields (only used when solve_method="jax_dense")
    W_sym_dense: object | None = None  # jax.numpy.ndarray (n, n): W + W^T
    WtW_dense: object | None = None  # jax.numpy.ndarray (n, n): W^T W
    W_eigs: object | None = None  # jax.numpy.ndarray (n,): eigenvalues of W (legacy)
    logdet_jax: object | None = None  # callable (rho) -> jax.numpy.ndarray
    # Mode-finding for ρ slice sampler (JAX-dense only)
    rho_mode_update_freq: int = 10  # recompute mode every N sweeps (0 = never)
    rho_mode_w_factor: float = 2.0  # slice width = factor / sqrt(-Hessian)
    # Adaptive width for ρ slice sampler (all backends)
    rho_adaptive_width: bool = True  # enable adaptive width tuning
    rho_slice_width_state: SliceWidthState | None = None  # mutable state


# ---------------------------------------------------------------------------
# Gibbs block samplers
# ---------------------------------------------------------------------------


def _sample_omega(
    y: np.ndarray,
    alpha: float,
    eta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 1: Draw ω | η, α — Pólya–Gamma augmentation.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Integer response vector.
    alpha : float
        NB dispersion parameter.
    eta : ndarray of shape (n,)
        Current latent field.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    omega : ndarray of shape (n,)
        PG(y + alpha, eta) draws.
    """
    h = y + alpha  # shape parameters — must be > 0 for PG
    # Guard against numerical zeros (alpha can be very small during
    # early iterations when y_i = 0)
    h = np.maximum(h, 1e-6)
    z = eta  # tilting parameters
    return sample_polyagamma(h, z, rng=rng)


def _sample_eta(
    state: GibbsState,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    *,
    rng: np.random.Generator,
    cache: GibbsCache | None = None,
) -> tuple[np.ndarray, CholmodFactor | spla.SuperLU]:
    """Block 2: Draw η | ω, ρ, β, σ² — spatial-normal draw.

    The conditional posterior is

        η | · ~ N(m_η, Σ_η)

    where Σ_η⁻¹ = A_ρ^T A_ρ / σ² + diag(ω) and
    m_η = Σ_η (A_ρ^T Xβ / σ² + κ) with κ = (y - α) / 2.

    Parameters
    ----------
    state : GibbsState
        Current state (uses eta, beta, sigma2, rho, omega, alpha).
    y : ndarray of shape (n,)
        Integer response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Spatial weights matrix.
    rng : numpy.random.Generator
        Random state.
    cache : GibbsCache, optional
        If provided, uses precomputed matrix pieces (W_sym, WtW)
        to avoid the expensive A_ρ^T @ A_ρ sparse product.

    Returns
    -------
    eta_new : ndarray of shape (n,)
        New draw of the latent field.
    factor : CholmodFactor or SuperLU
        Factorisation (for potential reuse within the sweep).
    """
    n = X.shape[0]
    rho = state.rho
    sigma2 = state.sigma2
    omega = state.omega
    beta = state.beta

    # Precision: P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
    # Use precomputed pieces if available to avoid A_rho^T @ A_rho product.
    if (
        cache is not None
        and cache.W_sym_over_s2 is not None
        and cache.WtW_over_s2 is not None
    ):
        P = (
            sp.eye(n, format="csr") / sigma2
            + sp.diags(omega, format="csr")
            - rho * cache.W_sym_over_s2 / sigma2
            + rho**2 * cache.WtW_over_s2 / sigma2
        )
    else:
        A_rho = sp.eye(n, format="csr") - rho * W_sparse
        AtA = A_rho.T @ A_rho / sigma2
        P = AtA + sp.diags(omega, format="csr")

    # Mean term: P @ m = A_rho^T Xβ / σ² + κ  where κ = (y - α) / 2
    # A_rho^T = I - ρ W^T, so A_rho^T Xβ / σ² = Xβ/σ² - ρ W^T Xβ / σ²
    Xbeta = X @ beta
    kappa = (y - state.alpha) / 2.0
    rhs = Xbeta / sigma2 - rho * W_sparse.T @ Xbeta / sigma2 + kappa

    # Dispatch based on sample_method
    sample_method = cache.sample_method if cache is not None else "cholmod"

    if sample_method == "jax_dense":
        # JAX Chebyshev path: build dense P, use vmap over draws
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        omega_jax = jnp.asarray(omega)
        P_dense = jax_build_P_dense(
            rho,
            sigma2,
            omega_jax,
            cache.W_sym_dense,
            cache.WtW_dense,
        )
        rhs_jax = jnp.asarray(rhs)
        # Create JAX PRNG key from numpy RNG (avoids pickle issues with
        # joblib worker processes — JAX keys can't be serialized)
        _jax_key_eta = jax.random.PRNGKey(rng.integers(2**31))
        draw = jax_chebyshev_sample(
            P_dense,
            rhs_jax,
            key=_jax_key_eta,
            degree=cache.chebyshev_degree,
        )
        return draw.x, draw.factor
    elif sample_method == "chebyshev":
        degree = cache.chebyshev_degree if cache is not None else 30
        draw = chebyshev_sample(P, rhs, rng=rng, degree=degree)
    else:
        draw = sample_spatial_normal(P, rhs, rng=rng)
    return draw.x, draw.factor


def _sample_beta(
    state: GibbsState,
    X: np.ndarray,
    XtX: np.ndarray,
    priors: GibbsPriors,
    A_rho_eta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 3: Draw β | η, ρ, σ² — conjugate normal.

    Prior: β ~ N(μ_β, Σ_β) with Σ_β = diag(σ_β²).
    Likelihood: A_ρ η - Xβ ~ N(0, σ² I).

    Posterior: β | · ~ N(m_β, Σ_β') where
        Σ_β'⁻¹ = Σ_β⁻¹ + X^T X / σ²
        m_β = Σ_β' (Σ_β⁻¹ μ_β + X^T A_ρ η / σ²)

    Parameters
    ----------
    state : GibbsState
        Current state (uses sigma2, beta).
    X : ndarray of shape (n, k)
        Design matrix.
    XtX : ndarray of shape (k, k)
        Precomputed X^T X.
    priors : GibbsPriors
        Prior hyperparameters.
    A_rho_eta : ndarray of shape (n,)
        A_ρ @ η = (I - ρW) @ η.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta_new : ndarray of shape (k,)
        New draw of regression coefficients.
    """
    k = X.shape[1]
    sigma2 = state.sigma2

    # Prior precision and mean
    beta_mu = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,))
    beta_sigma2 = np.broadcast_to(
        np.asarray(priors.beta_sigma, dtype=np.float64) ** 2, (k,)
    )

    # Posterior precision: Sigma_beta_inv = diag(1/sigma_beta^2) + XtX / sigma2
    Sigma_beta_inv = np.diag(1.0 / beta_sigma2) + XtX / sigma2

    # Posterior mean: m_beta = Sigma_beta @ (mu_beta / sigma_beta^2 + Xt @ A_rho_eta / sigma2)
    rhs = beta_mu / beta_sigma2 + X.T @ A_rho_eta / sigma2

    # Solve Sigma_beta_inv @ m_beta = rhs for posterior mean
    m_beta = np.linalg.solve(Sigma_beta_inv, rhs)

    # Sample: beta = m_beta + L^{-T} z where L L^T = Sigma_beta_inv
    # Cov(beta) = L^{-T} L^{-1} = (L L^T)^{-1} = Sigma_beta_inv^{-1} ✓
    L = np.linalg.cholesky(Sigma_beta_inv)
    z = rng.standard_normal(k)
    beta_new = m_beta + np.linalg.solve(L.T, z)

    return beta_new


def _sample_sigma2(
    state: GibbsState,
    priors: GibbsPriors,
    A_rho_eta: np.ndarray,
    Xbeta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> float:
    """Block 4: Draw σ² | η, ρ, β — conjugate inverse-gamma.

    Prior: σ ~ HalfNormal(σ_σ), which on the σ² scale contributes
    p(σ²) ∝ σ^{-1} exp(-σ²/(2σ_σ²)), adding 0.5 to the InvGamma
    shape and σ_σ^{-2}/2 to the rate.

    Posterior: σ² | · ~ InvGamma(a_post, b_post) where
        a_post = n/2 + 1/2 + 1 = (n + 3)/2
        b_post = ||A_ρ η - Xβ||²/2 + 1/(2σ_σ²)

    Parameters
    ----------
    state : GibbsState
        Current state (uses sigma2).
    priors : GibbsPriors
        Prior hyperparameters (uses sigma_sigma).
    A_rho_eta : ndarray of shape (n,)
        A_ρ @ η = (I - ρW) @ η.
    Xbeta : ndarray of shape (n,)
        X @ β.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    sigma2_new : float
        New draw of residual variance.
    """
    n = len(A_rho_eta)
    r = A_rho_eta - Xbeta  # residual

    # HalfNormal(σ_σ) prior on σ → InvGamma contribution:
    # shape += 0.5, rate += 1/(2*σ_σ²)
    sigma_sigma = priors.sigma_sigma
    a_post = n / 2.0 + 0.5 + 1.0  # (n + 3) / 2
    b_post = float(r @ r) / 2.0 + 1.0 / (2.0 * sigma_sigma**2)

    # Draw sigma2 ~ InvGamma(a_post, b_post)
    # InvGamma(a, b) has density ∝ x^{-(a+1)} exp(-b/x)
    # Equivalently: 1/sigma2 ~ Gamma(a, 1/b)
    sigma2_new = 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)

    return sigma2_new


def _find_rho_mode_newton(
    log_density_jax: Callable,
    rho_lower: float,
    rho_upper: float,
    x0: float,
    *,
    max_iter: int = 10,
    tol: float = 1e-5,
) -> tuple[float, float, float]:
    r"""Find the mode of the ρ log-density via Newton's method with JAX autodiff.

    Uses ``jax.grad`` and ``jax.hessian`` for exact first/second derivatives,
    giving quadratic convergence near the mode.  Typically needs 4–6
    iterations vs. ~30 for Brent's method.

    The Newton update for maximisation is:

    .. math::

        \rho_{k+1} = \rho_k - \frac{f'(\rho_k)}{f''(\rho_k)},

    where :math:`f(\rho)=\log p(\rho\mid\cdot)`.  Because we seek a
    maximum, :math:`f''(\rho_k)<0` and the step moves in the direction
    of the gradient.

    Falls back to bounded Brent if the Hessian is non-negative (non-concave
    region) or if Newton steps diverge.

    Parameters
    ----------
    log_density_jax : callable
        JAX function ``rho -> log p(rho | ·)`` returning a JAX scalar.
    rho_lower, rho_upper : float
        Support bounds for ρ.
    x0 : float
        Initial guess (typically the current ρ).
    max_iter : int, default 10
        Maximum Newton iterations.
    tol : float, default 1e-5
        Stopping tolerance for :math:`|f'(\rho)|`.

    Returns
    -------
    rho_mode : float
        Mode of the conditional posterior.
    log_dens_mode : float
        Log-density at the mode.
    hessian : float
        Second derivative at the mode (negative for a maximum).
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize_scalar

    grad_fn = jax.grad(log_density_jax)
    hess_fn = jax.hessian(log_density_jax)

    rho = float(x0)
    converged = False
    for _ in range(max_iter):
        g = float(grad_fn(jnp.float64(rho)))
        h = float(hess_fn(jnp.float64(rho)))

        # If not concave, abort and fall back to Brent
        if h >= 0:
            break

        # Newton step for maximisation: rho -= f' / f''
        step = g / h  # h < 0, so step moves in direction of gradient
        rho -= step

        # Clip to bounds
        rho = max(rho_lower, min(rho_upper, rho))

        if abs(g) < tol or abs(step) < tol:
            converged = True
            break

    if not converged:
        # Fall back to bounded Brent
        def _py_logdens(r: float) -> float:
            return float(log_density_jax(jnp.float64(r)))

        result = minimize_scalar(
            lambda r: -_py_logdens(r),
            bounds=(rho_lower, rho_upper),
            method="bounded",
            options={"xatol": tol, "maxiter": 50},
        )
        rho = float(result.x)

    log_dens_mode = float(log_density_jax(jnp.float64(rho)))
    hessian = float(hess_fn(jnp.float64(rho)))
    return rho, log_dens_mode, hessian


def _sample_rho(
    state: GibbsState,
    cache: GibbsCache,
    priors: GibbsPriors,
    y: np.ndarray,
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    log_density_current: float | None = None,
    sweep_idx: int = 0,
    tune: int = 0,
) -> tuple[float, float]:
    """Block 5: Draw ρ | β, σ², ω, α, y — collapsed 1-D slice sampler.

    Uses the **marginal** (collapsed) posterior for ρ that integrates
    out η, avoiding the slow mixing that arises from conditioning on
    the current η draw.  The collapsed log-density is:

    .. math::

        \\log p(\\rho \\mid \\cdot) = \\log|I - \\rho W|
            - \\tfrac{1}{2} \\log|P_\\eta|
            + \\tfrac{1}{2} \\mathit{rhs}^T P_\\eta^{-1} \\mathit{rhs}

    where :math:`P_\\eta = A_\\rho^T A_\\rho / \\sigma^2 + \\mathrm{diag}(\\omega)`
    and :math:`\\mathit{rhs} = A_\\rho^T X\\beta / \\sigma^2 + \\kappa`
    with :math:`\\kappa = (y - \\alpha) / 2`.

    Each ρ evaluation requires computing log|P_η| and solving
    P_η m = rhs.  By default, both are done via CHOLMOD/splu
    factorisation (O(nnz^{1.5})).  When ``cache.solve_method="cg"``
    and/or ``cache.logdet_P_method="lanczos"``, the decoupled
    (iterative) path is used instead, avoiding the factorisation
    cost entirely for large n with high fill-in.

    Parameters
    ----------
    state : GibbsState
        Current state (uses rho, sigma2, beta, omega, alpha).
    cache : GibbsCache
        Precomputed data (W_sparse, logdet_fn, rho bounds).
    priors : GibbsPriors
        Prior hyperparameters (rho bounds).
    y : ndarray of shape (n,)
        Integer response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    rng : numpy.random.Generator
        Random state.
    log_density_current : float, optional
        Cached log-density at current ρ. If provided, avoids one
        logdet + factorisation evaluation.

    Returns
    -------
    rho_new : float
        New draw of ρ.
    log_density_new : float
        Log-density at the new ρ (for caching).
    """
    sigma2 = state.sigma2
    n = len(y)
    W = cache.W_sparse
    logdet_fn = cache.logdet_fn
    rho_lower = priors.rho_lower
    rho_upper = priors.rho_upper
    omega = state.omega
    Xbeta = X @ state.beta
    kappa = (y - state.alpha) / 2.0

    cholmod_factor = cache.cholmod_factor
    W_sym_over_s2 = cache.W_sym_over_s2
    WtW_over_s2 = cache.WtW_over_s2
    solve_method = cache.solve_method
    logdet_P_method = cache.logdet_P_method

    # Precompute the base precision matrix (changes each Gibbs iteration
    # because omega changes, but is constant across ρ candidates within
    # one slice-sampler step).
    # P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
    base = sp.eye(n, format="csr") / sigma2 + sp.diags(omega, format="csr")

    # Precompute σ²-scaled matrix pieces (constant across ρ candidates)
    Wsym_s2 = W_sym_over_s2 / sigma2 if W_sym_over_s2 is not None else None
    WtW_s2 = WtW_over_s2 / sigma2 if WtW_over_s2 is not None else None

    # Precompute W^T @ Xbeta / sigma2 (constant across ρ candidates)
    WtXbeta_over_s2 = W.T @ Xbeta / sigma2
    Xbeta_over_s2 = Xbeta / sigma2

    # Lanczos RNG: use a child seed so the slice sampler's ρ path
    # is reproducible but doesn't interfere with the main Gibbs RNG.
    _lanczos_rng = np.random.default_rng(rng.integers(2**31))

    # JAX dense backend: precompute dense components for P construction
    use_jax = solve_method == "jax_dense"
    if use_jax:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        from bayespecon._samplers._spatial_normal import _jax_log_density_core

        # Convert omega to JAX array (changes each Gibbs iteration,
        # but is constant across ρ candidates within one slice step)
        omega_jax = jnp.asarray(omega)
        # Create JAX PRNG key from numpy RNG (avoids pickle issues with
        # joblib worker processes — JAX keys can't be serialized)
        _jax_key = jax.random.PRNGKey(rng.integers(2**31))
        # Precompute constant JAX arrays for the log-density closure
        _Xbeta_over_s2_jax = jnp.asarray(Xbeta_over_s2)
        _WtXbeta_over_s2_jax = jnp.asarray(WtXbeta_over_s2)
        _kappa_jax = jnp.asarray(kappa)
        # JIT-compile the log-density core function once
        _jax_logdens_fn = jax.jit(
            lambda rho, key: _jax_log_density_core(
                rho=rho,
                sigma2=sigma2,
                omega=omega_jax,
                W_sym_dense=cache.W_sym_dense,
                WtW_dense=cache.WtW_dense,
                logdet_jax=cache.logdet_jax,
                Xbeta_over_s2=_Xbeta_over_s2_jax,
                WtXbeta_over_s2=_WtXbeta_over_s2_jax,
                kappa=_kappa_jax,
                key=key,
                n_probes=cache.lanczos_n_probes,
                lanczos_deg=cache.lanczos_deg,
                cg_tol=1e-8,
                cg_maxiter=n,
            ),
        )
        # Warm up the JIT-compiled function (first call triggers compilation)
        _warmup_key = jax.random.fold_in(_jax_key, 0)
        _ = float(_jax_logdens_fn(jnp.float64(state.rho), _warmup_key))

    def log_density(rho: float) -> float:
        """Collapsed log-density of ρ (η integrated out)."""
        if use_jax:
            # --- JAX dense path (JIT-compiled) ---
            # Split key for this evaluation (deterministic from rho)
            _jax_key_step = jax.random.fold_in(_jax_key, hash(rho) % (2**31))
            result = _jax_logdens_fn(jnp.float64(rho), _jax_key_step)
            return float(result)

        # --- scipy sparse path ---
        # log|I - rho*W|
        logdet = logdet_fn(rho)

        # Right-hand side (constant form regardless of solve method)
        rhs = Xbeta_over_s2 - rho * WtXbeta_over_s2 + kappa

        # Precision: P = base - rho * Wsym_s2 + rho^2 * WtW_s2
        if Wsym_s2 is not None and WtW_s2 is not None:
            P = base - rho * Wsym_s2 + rho**2 * WtW_s2
        else:
            A_rho = sp.eye(n, format="csr") - rho * W
            AtA = A_rho.T @ A_rho / sigma2
            P = AtA + sp.diags(omega, format="csr")

        # --- log|P_η| ---
        if logdet_P_method == "lanczos":
            log_det_P = lanczos_logdet(
                P,
                n_probes=cache.lanczos_n_probes,
                lanczos_deg=cache.lanczos_deg,
                rng=_lanczos_rng,
            )
        elif cholmod_factor is not None:
            cholmod_factor.factorize(P)
            log_det_P = cholmod_factor.logdet()
        else:
            P_csc = sp.csc_matrix(P)
            lu = spla.splu(P_csc, permc_spec="MMD_AT_PLUS_A")
            log_det_P = np.sum(np.log(np.abs(lu.U.diagonal())))

        # --- Solve P m = rhs ---
        if solve_method == "cg":
            m = cg_solve(P, rhs)
        elif cholmod_factor is not None and solve_method == "cholmod":
            m = cholmod_factor.solve(rhs)
        elif solve_method == "splu":
            P_csc = sp.csc_matrix(P)
            lu = spla.splu(P_csc, permc_spec="MMD_AT_PLUS_A")
            m = lu.solve(rhs)
        else:
            if logdet_P_method != "lanczos" and cholmod_factor is not None:
                m = cholmod_factor.solve(rhs)
            else:
                P_csc = sp.csc_matrix(P)
                lu = spla.splu(P_csc, permc_spec="MMD_AT_PLUS_A")
                m = lu.solve(rhs)

        # Quadratic form: rhs^T P^{-1} rhs = rhs^T m
        quad = float(rhs @ m)

        # Uniform prior on [rho_lower, rho_upper] → log p(rho) = 0
        return logdet - 0.5 * log_det_P + 0.5 * quad

    # Cache log-density at current x0 to avoid redundant evaluation
    # inside the slice sampler (which always evaluates log_density(x0))
    x0 = state.rho
    if log_density_current is not None:
        log_dens_x0 = log_density_current
    else:
        log_dens_x0 = log_density(x0)

    # --- Mode-centered slice sampling (JAX dense only, burn-in only) ---
    # Mode-finding is expensive (~50 Lanczos/CG evals) but helps mixing
    # during burn-in. After burn-in, the mode is stable — skip it.
    in_burnin = sweep_idx < tune
    use_mode_centered = use_jax and cache.rho_mode_update_freq > 0 and in_burnin
    if use_mode_centered and (sweep_idx % cache.rho_mode_update_freq == 0):
        # For mode-finding, use the exact dense-Cholesky log-density
        # (no stochastic Lanczos/CG) — it's 15× faster for n ≤ ~500.
        from bayespecon._samplers._spatial_normal import _jax_log_density_core_exact

        _jax_logdens_exact_fn = jax.jit(
            lambda rho: _jax_log_density_core_exact(
                rho=rho,
                sigma2=sigma2,
                omega=omega_jax,
                W_sym_dense=cache.W_sym_dense,
                WtW_dense=cache.WtW_dense,
                logdet_jax=cache.logdet_jax,
                Xbeta_over_s2=_Xbeta_over_s2_jax,
                WtXbeta_over_s2=_WtXbeta_over_s2_jax,
                kappa=_kappa_jax,
            ),
        )

        def _jax_logdens_scalar(rho):
            return _jax_logdens_exact_fn(jnp.float64(rho))

        rho_mode, log_dens_mode, hessian = _find_rho_mode_newton(
            _jax_logdens_scalar,
            rho_lower=rho_lower,
            rho_upper=rho_upper,
            x0=x0,
        )

        # Proposal width from curvature: σ = 1 / sqrt(-H), w = factor * σ
        if hessian < 0:
            sigma = 1.0 / np.sqrt(-hessian)
            w = float(cache.rho_mode_w_factor * sigma)
        else:
            w = 0.2  # fallback if Hessian is non-negative
        w = min(w, rho_upper - rho_lower)

        # Mode-centered slice sampling
        rho_new, log_density_new = slice_sample_1d(
            log_density=log_density,
            x0=rho_mode,
            lower=rho_lower,
            upper=rho_upper,
            w=w,
            rng=rng,
        )
    else:
        # --- Adaptive width slice sampling ---
        # The width state is created in GibbsCache and mutated in-place.
        # SliceWidthState is a mutable dataclass, so updates to its
        # fields (w, L, R) persist across calls.
        width_state = cache.rho_slice_width_state
        if width_state is None:
            # Fallback: should not happen if GibbsCache is constructed
            # correctly, but provides a safe default.
            width_state = SliceWidthState(w=0.2)

        rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
            log_density=log_density,
            x0=x0,
            lower=rho_lower,
            upper=rho_upper,
            width_state=width_state,
            rng=rng,
            log_density_x0=log_dens_x0,
        )

        # Update width (only during burn-in; freeze after)
        if cache.rho_adaptive_width and sweep_idx < tune:
            update_slice_width(width_state, steps_left, steps_right)

    return rho_new, log_density_new


def _sample_alpha(
    state: GibbsState,
    y: np.ndarray,
    priors: GibbsPriors,
    *,
    rng: np.random.Generator,
) -> float:
    """Block 6: Draw α | y, η — 1-D slice on log(α).

    Log-density on the log(α) scale:
        log p(log α | y, η) = log α + Σ_i log NB(y_i | exp(η_i), α) + log p(α)

    where log α is the Jacobian from the change of variables α = exp(log α),
    and p(α) is the HalfNormal(σ_α) prior.

    Parameters
    ----------
    state : GibbsState
        Current state (uses alpha, eta).
    y : ndarray of shape (n,)
        Integer response vector.
    priors : GibbsPriors
        Prior hyperparameters (alpha_sigma).
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    alpha_new : float
        New draw of α.
    """
    alpha_sigma = priors.alpha_sigma
    eta = state.eta
    log_alpha = np.log(state.alpha)

    def log_density(log_a: float) -> float:
        """Log-density on the log(α) scale."""
        alpha = np.exp(log_a)
        if alpha <= 0:
            return -np.inf

        # NB log-likelihood: sum_i log NB(y_i | mu_i, alpha)
        # where mu_i = exp(eta_i)
        mu = np.exp(eta)
        # scipy.stats.nbinom.logpmf uses (n, p) parameterisation
        # NB(y | mu, alpha) = Gamma-Poisson mixture
        # log p(y | mu, alpha) = log Gamma(y + alpha) - log Gamma(alpha)
        #   + y * log(mu / (mu + alpha)) + alpha * log(alpha / (mu + alpha))
        #   - log(y!)
        # Using scipy's nbinom: n=alpha, p=alpha/(mu+alpha)
        from scipy.special import gammaln

        log_lik = (
            gammaln(y + alpha)
            - gammaln(alpha)
            + y * np.log(mu / (mu + alpha))
            + alpha * np.log(alpha / (mu + alpha))
        )
        total_log_lik = np.sum(log_lik)

        # HalfNormal prior on alpha: p(alpha) = (2/(pi*sigma^2))^{1/2} exp(-alpha^2/(2*sigma^2))
        # On log(alpha) scale: log p(alpha) + log(alpha) [Jacobian]
        # = -alpha^2 / (2*sigma^2) + log(alpha) + const
        log_prior = -(alpha**2) / (2.0 * alpha_sigma**2)

        # Jacobian: d(alpha)/d(log_alpha) = alpha, so log|J| = log(alpha) = log_a
        return log_a + total_log_lik + log_prior

    # Slice sample on log(alpha) with bounds
    log_alpha_new, _ = slice_sample_1d(
        log_density=log_density,
        x0=log_alpha,
        lower=-10.0,  # alpha > exp(-10) ≈ 4.5e-5
        upper=10.0,  # alpha < exp(10) ≈ 22026
        w=0.5,
        rng=rng,
    )

    return np.exp(log_alpha_new)


# ---------------------------------------------------------------------------
# NB log-likelihood (for InferenceData)
# ---------------------------------------------------------------------------


def _nb_loglik_pointwise(
    y: np.ndarray,
    eta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute pointwise NB log-likelihood.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Integer response.
    eta : ndarray of shape (n,)
        Latent log-mean.
    alpha : float
        NB dispersion parameter.

    Returns
    -------
    log_lik : ndarray of shape (n,)
        Pointwise log-likelihood values.
    """
    from scipy.special import gammaln

    mu = np.exp(eta)
    return (
        gammaln(y + alpha)
        - gammaln(alpha)
        + y * np.log(mu / (mu + alpha))
        + alpha * np.log(alpha / (mu + alpha))
    )


# ---------------------------------------------------------------------------
# Main chain runner
# ---------------------------------------------------------------------------


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: GibbsPriors,
    cache: GibbsCache,
    init: GibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Run one chain of the PG-Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Integer response vector.
    X : ndarray of shape (n, k)
        Design matrix (including intercept column if desired).
    W_sparse : csr_matrix of shape (n, n)
        Row-standardised spatial weights matrix.
    priors : GibbsPriors
        Prior hyperparameters.
    cache : GibbsCache
        Precomputed data (XtX, logdet_fn, etc.).
    init : GibbsState
        Initial state for the chain.
    draws : int
        Number of post-warmup draws to keep.
    tune : int
        Number of warmup (burn-in) draws.
    thin : int, default 1
        Keep every ``thin``-th draw after warmup.
    return_eta : bool, default False
        If True, store the full latent field η in the posterior.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, ``sigma``,
        ``alpha``, and optionally ``eta``. Each array has shape
        ``(draws // thin, ...)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws

    # Pre-allocate storage for post-warmup draws
    n_keep = draws // thin if thin > 0 else draws
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64)
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)  # ||η||² for diagnostics
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Copy initial state (we mutate state in-place)
    state = GibbsState(
        eta=init.eta.copy(),
        beta=init.beta.copy(),
        sigma2=init.sigma2,
        rho=init.rho,
        alpha=init.alpha,
        omega=init.omega.copy(),
    )

    # Precompute X^T X (already in cache)
    XtX = cache.XtX

    # Track log-density at current rho for caching
    log_density_rho = None

    for i in range(total_iters):
        # --- Block 1: ω | η, α ---
        state.omega = _sample_omega(y, state.alpha, state.eta, rng=rng)

        # --- Block 2: η | ω, ρ, β, σ² ---
        state.eta, _ = _sample_eta(state, y, X, W_sparse, rng=rng, cache=cache)

        # Recompute A_rho_eta and Xbeta with new eta
        A_rho = sp.eye(n, format="csr") - state.rho * W_sparse
        A_rho_eta = A_rho @ state.eta
        Xbeta = X @ state.beta

        # --- Block 3: β | η, ρ, σ² ---
        state.beta = _sample_beta(state, X, XtX, priors, A_rho_eta, rng=rng)
        Xbeta = X @ state.beta  # recompute with new beta

        # --- Block 4: σ² | η, ρ, β ---
        state.sigma2 = _sample_sigma2(state, priors, A_rho_eta, Xbeta, rng=rng)

        # --- Block 5: ρ | β, σ², ω, α, y (collapsed, η integrated out) ---
        state.rho, log_density_rho = _sample_rho(
            state,
            cache,
            priors,
            y,
            X,
            rng=rng,
            log_density_current=log_density_rho,
            sweep_idx=i,
            tune=tune,
        )

        # --- Block 6: α | y, η ---
        state.alpha = _sample_alpha(state, y, priors, rng=rng)

        # --- Store post-warmup draws ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = state.rho
                beta_samples[idx] = state.beta
                sigma_samples[idx] = np.sqrt(state.sigma2)  # store σ, not σ²
                alpha_samples[idx] = state.alpha
                log_lik_samples[idx] = _nb_loglik_pointwise(y, state.eta, state.alpha)
                eta_norm_samples[idx] = float(state.eta @ state.eta)
                if return_eta:
                    eta_samples[idx] = state.eta

    result = {
        "rho": rho_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
    }
    if return_eta:
        result["eta"] = eta_samples

    return result
