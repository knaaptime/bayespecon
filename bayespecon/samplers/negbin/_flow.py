r"""Pólya–Gamma Gibbs sampler for structural-form SAR flow models.

Orchestrates the 7-block (separable) or 8-block (non-separable) Gibbs sweep:

Separable (:math:`\rho_w = -\rho_d \rho_o`):
  1. :math:`\omega \mid \eta, \alpha` (PG augmentation)
  2. :math:`\eta \mid \omega, \rho_d, \rho_o, \beta, \sigma^2` (Kronecker spatial-normal)
  3. :math:`\beta \mid \eta, \rho_d, \rho_o, \sigma^2` (conjugate normal)
  4. :math:`\sigma^2 \mid \eta, \rho_d, \rho_o, \beta` (conjugate inverse-gamma)
  5. :math:`\rho_d \mid \beta, \sigma^2, \omega, y` (collapsed 1-D slice, :math:`\eta` integrated out)
  6. :math:`\rho_o \mid \beta, \sigma^2, \omega, y` (collapsed 1-D slice, :math:`\eta` integrated out)
  7. :math:`\alpha \mid y, \eta` (1-D slice on :math:`\log(\alpha)`)

Non-separable (3 free :math:`\rho`):
  Same as above but with three :math:`\rho` blocks (5, 6, 7) and
  :math:`\alpha` as block 8. The :math:`\eta` draw uses general sparse
  :math:`N \times N` precision rather than Kronecker structure.

The structural form parameterises the latent log-mean as

.. math::

    \eta = \rho_d W_d \eta + \rho_o W_o \eta + \rho_w W_w \eta
        + X\beta + \nu, \quad \nu \sim N(0, \sigma^2 I_N)

where :math:`N = n^2` and :math:`W_d = I_n \otimes W`,
:math:`W_o = W \otimes I_n`, :math:`W_w = W \otimes W`.

For the separable constraint :math:`\rho_w = -\rho_d \rho_o`, the system
matrix factors as :math:`A = L_o \otimes L_d` where
:math:`L_k = I_n - \rho_k W`, enabling :math:`O(n^3)` Kronecker matvec
instead of :math:`O(n^6)`.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.

LeSage, J. P., & Pace, R. K. (2008). *Spatial Econometric Modeling of
Origin-Destination Flows*. Journal of Regional Science, 48(5), 941–967.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from .._utils._spatial_normal import (
    cg_solve,
    chebyshev_sample,
    lanczos_logdet,
)
from ..panel._kronecker import (
    kron_At_matvec,
    kron_eigenvalue_bounds,
    kron_logdet_A,
    kron_matvec,
    kron_P_matvec,
)

# ---------------------------------------------------------------------------
# Data classes for state, priors, and precomputed cache
# ---------------------------------------------------------------------------


@dataclass
class FlowGibbsState:
    """Mutable state carried through one Gibbs sweep.

    All arrays are numpy arrays; scalars are Python floats.

    Parameters
    ----------
    eta : ndarray of shape (N,)
        Latent field, where :math:`N = n^2`.
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma2 : float
        Residual variance :math:`\\sigma^2`.
    rho_d : float
        Destination autoregressive parameter.
    rho_o : float
        Origin autoregressive parameter.
    rho_w : float or None
        Network autoregressive parameter. ``None`` for the separable model
        (where :math:`\rho_w = -\rho_d \rho_o` is deterministic).
    alpha : float
        NB dispersion parameter.
    omega : ndarray of shape (N,)
        PG auxiliary variables.
    """

    eta: np.ndarray
    beta: np.ndarray
    sigma2: float
    rho_d: float
    rho_o: float
    rho_w: float | None
    alpha: float
    omega: np.ndarray


@dataclass
class FlowGibbsPriors:
    """Prior hyperparameters for the flow Gibbs sampler.

    All priors are weakly informative by default, matching the
    ``GaussianGibbsPriors`` convention.
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    sigma2_alpha: float = 2.0  # InverseGamma shape for σ²
    sigma2_beta: float = 1.0  # InverseGamma scale for σ²
    alpha_sigma: float = 10.0  # HalfNormal scale for α
    rho_lower: float = -0.999
    rho_upper: float = 0.999


class FlowGibbsCache(NamedTuple):
    """Precomputed data that doesn't change across sweeps (separable variant).

    Stores the :math:`n \\times n` weight matrix and its eigenvalues
    (for Kronecker logdet), plus the :math:`N \\times N` weight matrices
    (for effects computation in the model class).

    This cache is **immutable** and safe to share across chains.
    Per-chain mutable state (slice-sampler widths) lives in
    :class:`FlowGibbsSliceState`.
    """

    W_sparse: sp.csr_matrix  # (n, n) base weight matrix
    W_dense: np.ndarray  # (n, n) dense copy of W (avoids repeated toarray())
    W_eigs: np.ndarray | None  # (n,) eigenvalues of W (for logdet)
    n: int  # number of spatial units (N = n²)
    XtX: np.ndarray  # (k, k) = X^T X
    logdet_fn: Callable[[float], float]  # log|I - rho*W| callable
    rho_lower: float
    rho_upper: float
    rho_adaptive_width: bool = True


@dataclass
class FlowGibbsSliceState:
    """Per-chain mutable state for adaptive slice-sampler widths (separable).

    Created fresh for each chain so that chains don't contaminate
    each other's width adaptation.
    """

    rho_d: SliceWidthState = field(default_factory=lambda: SliceWidthState(w=0.2))
    rho_o: SliceWidthState = field(default_factory=lambda: SliceWidthState(w=0.2))


class FlowGibbsCacheNS(NamedTuple):
    """Precomputed data for the non-separable (3-ρ) variant.

    Stores the :math:`N \\times N` sparse weight matrices and
    precomputed trace polynomials for the flow log-determinant.

    This cache is **immutable** and safe to share across chains.
    Per-chain mutable state (slice-sampler widths) lives in
    :class:`FlowGibbsSliceStateNS`.
    """

    Wd: sp.csr_matrix  # (N, N) destination weight matrix
    Wo: sp.csr_matrix  # (N, N) origin weight matrix
    Ww: sp.csr_matrix  # (N, N) network weight matrix
    Wd_sym: sp.csr_matrix  # (N, N) Wd + Wd^T
    Wo_sym: sp.csr_matrix  # (N, N) Wo + Wo^T
    Ww_sym: sp.csr_matrix  # (N, N) Ww + Ww^T
    WdWd: sp.csr_matrix  # (N, N) Wd^T @ Wd
    WoWo: sp.csr_matrix  # (N, N) Wo^T @ Wo
    WwWw: sp.csr_matrix  # (N, N) Ww^T @ Ww
    # Cross-product terms for P construction
    WdWo: sp.csr_matrix  # (N, N) Wd^T @ Wo + Wo^T @ Wd
    WdWw: sp.csr_matrix  # (N, N) Wd^T @ Ww + Ww^T @ Wd
    WoWw: sp.csr_matrix  # (N, N) Wo^T @ Ww + Ww^T @ Wo
    N: int  # N = n²
    XtX: np.ndarray  # (k, k) = X^T X
    # Flow logdet callable: logdet_fn(rho_d, rho_o, rho_w) -> float
    logdet_fn: Callable[[float, float, float], float]
    rho_lower: float
    rho_upper: float
    rho_adaptive_width: bool = True


@dataclass
class FlowGibbsSliceStateNS:
    """Per-chain mutable state for adaptive slice-sampler widths (non-separable).

    Created fresh for each chain so that chains don't contaminate
    each other's width adaptation.
    """

    rho_d: SliceWidthState = field(default_factory=lambda: SliceWidthState(w=0.2))
    rho_o: SliceWidthState = field(default_factory=lambda: SliceWidthState(w=0.2))
    rho_w: SliceWidthState = field(default_factory=lambda: SliceWidthState(w=0.2))


# ---------------------------------------------------------------------------
# Shared Gibbs blocks (identical for both variants)
# ---------------------------------------------------------------------------


def _sample_omega_flow(
    y: np.ndarray,
    alpha: float,
    eta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Block 1: Draw :math:`\omega \mid \eta, \alpha` — Pólya–Gamma augmentation.

    Identical to the single-ρ version but operates on :math:`N = n^2`
    observations instead of :math:`n`.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Integer response vector.
    alpha : float
        NB dispersion parameter.
    eta : ndarray of shape (N,)
        Current latent field.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    omega : ndarray of shape (N,)
        :math:`\mathrm{PG}(y + \alpha, \eta)` draws.
    """
    h = y + alpha
    h = np.maximum(h, 1e-6)
    z = eta
    return sample_polyagamma(h, z, rng=rng)


def _sample_beta_flow(
    state: FlowGibbsState,
    X: np.ndarray,
    XtX: np.ndarray,
    priors: FlowGibbsPriors,
    A_eta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Block 3: Draw :math:`\beta \mid \eta, \rho, \sigma^2` — conjugate normal.

    Prior: :math:`\beta \sim N(\mu_\beta, \Sigma_\beta)` with
    :math:`\Sigma_\beta = \mathrm{diag}(\sigma_\beta^2)`.
    Likelihood: :math:`A_\rho\,\eta - X\beta \sim N(0, \sigma^2 I)`.

    Posterior: :math:`\beta \mid \cdot \sim N(m_\beta, \Sigma_\beta')` where

    .. math::

        \Sigma_\beta'^{-1} = \Sigma_\beta^{-1} + X^\top X / \sigma^2

        m_\beta = \Sigma_\beta' (\Sigma_\beta^{-1} \mu_\beta
            + X^\top A_\rho\,\eta / \sigma^2)

    Parameters
    ----------
    state : FlowGibbsState
        Current state (uses sigma2, beta).
    X : ndarray of shape (N, k)
        Design matrix.
    XtX : ndarray of shape (k, k)
        Precomputed :math:`X^\top X`.
    priors : FlowGibbsPriors
        Prior hyperparameters.
    A_eta : ndarray of shape (N,)
        :math:`A\,\eta` (system matrix applied to latent field).
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta_new : ndarray of shape (k,)
        New draw of regression coefficients.
    """
    k = X.shape[1]
    sigma2 = state.sigma2

    beta_mu = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,))
    beta_sigma2 = np.broadcast_to(
        np.asarray(priors.beta_sigma, dtype=np.float64) ** 2, (k,)
    )

    Sigma_beta_inv = np.diag(1.0 / beta_sigma2) + XtX / sigma2
    rhs = beta_mu / beta_sigma2 + X.T @ A_eta / sigma2
    m_beta = np.linalg.solve(Sigma_beta_inv, rhs)

    L = np.linalg.cholesky(Sigma_beta_inv)
    z = rng.standard_normal(k)
    beta_new = m_beta + np.linalg.solve(L.T, z)

    return beta_new


def _sample_sigma2_flow(
    state: FlowGibbsState,
    priors: FlowGibbsPriors,
    A_eta: np.ndarray,
    Xbeta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> float:
    r"""Block 4: Draw :math:`\sigma^2 \mid \eta, \rho, \beta` — conjugate inverse-gamma.

    Prior: :math:`\sigma^2 \sim \mathrm{InvGamma}(\alpha_\sigma, \beta_\sigma)`,
    which is conjugate with the Gaussian likelihood.

    Posterior: :math:`\sigma^2 \mid \cdot \sim \mathrm{InvGamma}(a_{\mathrm{post}}, b_{\mathrm{post}})`

    .. math::

        a_{\mathrm{post}} = \alpha_\sigma + N/2

        b_{\mathrm{post}} = \beta_\sigma + \|A_\rho\,\eta - X\beta\|^2 / 2

    Parameters
    ----------
    state : FlowGibbsState
        Current state (uses sigma2).
    priors : FlowGibbsPriors
        Prior hyperparameters (uses sigma2_alpha, sigma2_beta).
    A_eta : ndarray of shape (N,)
        :math:`A\,\eta`.
    Xbeta : ndarray of shape (N,)
        :math:`X\,\beta`.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    sigma2_new : float
        New draw of residual variance.
    """
    N = len(A_eta)
    r = A_eta - Xbeta

    a_post = priors.sigma2_alpha + N / 2.0
    b_post = priors.sigma2_beta + float(r @ r) / 2.0

    sigma2_new = 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)
    return sigma2_new


def _sample_alpha_flow(
    state: FlowGibbsState,
    y: np.ndarray,
    priors: FlowGibbsPriors,
    *,
    rng: np.random.Generator,
) -> float:
    r"""Block last: Draw :math:`\alpha \mid y, \eta` — 1-D slice on :math:`\log(\alpha)`.

    Log-density on the :math:`\log(\alpha)` scale:

    .. math::

        \log p(\log\alpha \mid y, \eta) = \log\alpha
            + \sum_i \log \mathrm{NB}(y_i \mid \exp(\eta_i), \alpha)
            + \log p(\alpha)

    where :math:`\log\alpha` is the Jacobian from the change of variables
    :math:`\alpha = \exp(\log\alpha)`, and :math:`p(\alpha)` is the
    :math:`\mathrm{HalfNormal}(\sigma_\alpha)` prior.

    Identical to the single-ρ version but operates on :math:`N = n^2`
    observations.

    Parameters
    ----------
    state : FlowGibbsState
        Current state (uses alpha, eta).
    y : ndarray of shape (N,)
        Integer response vector.
    priors : FlowGibbsPriors
        Prior hyperparameters (alpha_sigma).
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    alpha_new : float
        New draw of :math:`\alpha`.
    """
    alpha_sigma = priors.alpha_sigma
    eta = state.eta
    log_alpha = np.log(state.alpha)

    def log_density(log_a: float) -> float:
        alpha = np.exp(log_a)
        if alpha <= 0:
            return -np.inf

        mu = np.exp(eta)
        from scipy.special import gammaln

        log_lik = (
            gammaln(y + alpha)
            - gammaln(alpha)
            + y * np.log(mu / (mu + alpha))
            + alpha * np.log(alpha / (mu + alpha))
        )
        total_log_lik = np.sum(log_lik)

        log_prior = -(alpha**2) / (2.0 * alpha_sigma**2)
        return log_a + total_log_lik + log_prior

    log_alpha_new, _ = slice_sample_1d(
        log_density=log_density,
        x0=log_alpha,
        lower=-10.0,
        upper=10.0,
        w=0.5,
        rng=rng,
    )

    return np.exp(log_alpha_new)


# ---------------------------------------------------------------------------
# NB log-likelihood (for InferenceData)
# ---------------------------------------------------------------------------


def _nb_loglik_pointwise_flow(
    y: np.ndarray,
    eta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute pointwise NB log-likelihood for flow models.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Integer response.
    eta : ndarray of shape (N,)
        Latent log-mean.
    alpha : float
        NB dispersion parameter.

    Returns
    -------
    log_lik : ndarray of shape (N,)
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
# Separable (Kronecker) variant: η draw and ρ draws
# ---------------------------------------------------------------------------


def _sample_eta_separable(
    state: FlowGibbsState,
    y: np.ndarray,
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    cache: FlowGibbsCache | None = None,
    chebyshev_degree: int = 30,
) -> np.ndarray:
    r"""Block 2 (separable): Draw :math:`\eta \mid \omega, \rho_d, \rho_o, \beta, \sigma^2`.

    The conditional posterior is

    .. math::

        \eta \mid \cdot \sim N(m_\eta, \Sigma_\eta)

    where :math:`\Sigma_\eta^{-1} = P_\eta` and

    .. math::

        P_\eta = (L_d^\top L_d \otimes L_o^\top L_o) / \sigma^2
            + \mathrm{diag}(\omega)

        m_\eta = P_\eta^{-1}\,\mathit{rhs},
        \quad \mathit{rhs} = (L_o^\top \otimes L_d^\top)\,X\beta / \sigma^2 + \kappa

    with :math:`\kappa = (y - \alpha) / 2`.

    Uses Chebyshev polynomial approximation with Kronecker-structured
    matvec for :math:`O(n^3)` per iteration instead of :math:`O(n^6)`.

    Parameters
    ----------
    state : FlowGibbsState
        Current state.
    y : ndarray of shape (N,)
        Integer response vector.
    X : ndarray of shape (N, k)
        Design matrix.
    rng : numpy.random.Generator
        Random state.
    cache : FlowGibbsCache, optional
        Precomputed data (includes ``W_dense``).
    chebyshev_degree : int, default 30
        Degree of Chebyshev polynomial approximation.

    Returns
    -------
    eta_new : ndarray of shape (N,)
        New draw of the latent field.
    """
    n = cache.n
    N = n * n
    rho_d = state.rho_d
    rho_o = state.rho_o
    sigma2 = state.sigma2
    omega = state.omega
    beta = state.beta

    # Build L_d and L_o as dense n×n matrices for Kronecker matvec
    I_n = np.eye(n)
    Ld = I_n - rho_d * cache.W_dense
    Lo = I_n - rho_o * cache.W_dense
    LdtLd = Ld.T @ Ld
    LotLo = Lo.T @ Lo

    # Right-hand side: rhs = A^T Xβ / σ² + κ
    Xbeta = X @ beta
    kappa = (y - state.alpha) / 2.0
    rhs = kron_At_matvec(Xbeta / sigma2, Ld.T, Lo.T) + kappa

    # Build precision matvec for Chebyshev sampler (O(n³) per call)
    def P_matvec(v: np.ndarray) -> np.ndarray:
        return kron_P_matvec(v, LdtLd, LotLo, omega, sigma2)

    # Compute eigenvalue bounds from Kronecker structure (O(n²))
    lambda_min, lambda_max = kron_eigenvalue_bounds(LdtLd, LotLo, omega, sigma2)

    # Build LinearOperator for P
    N = n * n
    P_op = spla.LinearOperator((N, N), matvec=P_matvec)

    # Draw via Chebyshev polynomial approximation
    draw = chebyshev_sample(
        P_op,
        rhs,
        rng=rng,
        degree=chebyshev_degree,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )
    return draw.x


def _sample_rho_separable(
    state: FlowGibbsState,
    cache: FlowGibbsCache,
    priors: FlowGibbsPriors,
    y: np.ndarray,
    X: np.ndarray,
    *,
    which_rho: str,
    rng: np.random.Generator,
    slice_state: SliceWidthState | None = None,
    log_density_current: float | None = None,
    sweep_idx: int = 0,
    tune: int = 0,
) -> tuple[float, float]:
    r"""Collapsed 1-D slice sampler for :math:`\rho_d` or :math:`\rho_o`.

    Uses the marginal (collapsed) posterior that integrates out :math:`\eta`,
    avoiding slow mixing from conditioning on the current :math:`\eta` draw.

    For :math:`\rho_d` (with :math:`\rho_o` fixed):

    .. math::

        \log p(\rho_d \mid \cdot) = \log|A|
            - \tfrac{1}{2}\log|P_\eta|
            + \tfrac{1}{2}\,\mathit{rhs}^\top P_\eta^{-1}\,\mathit{rhs}

    where :math:`\log|A| = n\,\log|I - \rho_d W| + n\,\log|I - \rho_o W|`
    and :math:`P_\eta` uses the Kronecker-structured matvec.

    Parameters
    ----------
    state : FlowGibbsState
        Current state.
    cache : FlowGibbsCache
        Precomputed data (includes ``W_dense``).
    priors : FlowGibbsPriors
        Prior hyperparameters.
    y : ndarray of shape (N,)
        Integer response vector.
    X : ndarray of shape (N, k)
        Design matrix.
    which_rho : {"rho_d", "rho_o"}
        Which parameter to sample.
    rng : numpy.random.Generator
        Random state.
    log_density_current : float, optional
        Cached log-density at current value.
    sweep_idx : int
        Current sweep index (for adaptive width).
    tune : int
        Number of warmup sweeps.

    Returns
    -------
    rho_new : float
        New draw of the sampled parameter.
    log_density_new : float
        Log-density at the new value (for caching).
    """
    n = cache.n
    sigma2 = state.sigma2
    omega = state.omega
    Xbeta = X @ state.beta
    kappa = (y - state.alpha) / 2.0
    logdet_fn = cache.logdet_fn

    # Determine which rho to sample and fix the other
    if which_rho == "rho_d":
        rho_var = state.rho_d
        rho_fixed = state.rho_o
    elif which_rho == "rho_o":
        rho_var = state.rho_o
        rho_fixed = state.rho_d
    else:
        raise ValueError(f"which_rho must be 'rho_d' or 'rho_o', got {which_rho!r}")

    rho_lower = priors.rho_lower
    rho_upper = priors.rho_upper

    I_n = np.eye(n)
    W_dense = cache.W_dense  # Use cached dense W (avoids repeated toarray())

    def log_density(rho_var_val: float) -> float:
        """Collapsed log-density for rho_d or rho_o."""
        if which_rho == "rho_d":
            rho_d_val = rho_var_val
            rho_o_val = rho_fixed
        else:
            rho_d_val = rho_fixed
            rho_o_val = rho_var_val

        # 1. Jacobian: log|A| = n * log|I - rho_d W| + n * log|I - rho_o W|
        logdet_A = kron_logdet_A(rho_d_val, rho_o_val, n, logdet_fn)

        # 2. Build L_d, L_o and their products
        Ld = I_n - rho_d_val * W_dense
        Lo = I_n - rho_o_val * W_dense
        LdtLd = Ld.T @ Ld
        LotLo = Lo.T @ Lo

        # 3. Right-hand side: A^T Xβ / σ² + κ
        rhs = kron_At_matvec(Xbeta / sigma2, Ld.T, Lo.T) + kappa

        # 4. Build Kronecker matvec LinearOperator for P (avoids N×N sparse)
        N = n * n
        P_op = spla.LinearOperator(
            (N, N),
            matvec=lambda v: kron_P_matvec(v, LdtLd, LotLo, omega, sigma2),
        )

        # 5. Eigenvalue bounds from Kronecker structure (O(n²))
        lambda_min, lambda_max = kron_eigenvalue_bounds(LdtLd, LotLo, omega, sigma2)

        # 6. log|P_η| via Lanczos with LinearOperator
        _lanczos_rng = np.random.default_rng(rng.integers(2**31))
        log_det_P = lanczos_logdet(
            P_op,
            n_probes=5,
            lanczos_deg=20,
            rng=_lanczos_rng,
        )

        # 7. Solve P m = rhs via CG with LinearOperator
        m = cg_solve(P_op, rhs)

        # 8. Quadratic form
        quad = float(rhs @ m)

        return logdet_A - 0.5 * log_det_P + 0.5 * quad

    # Slice sampling
    x0 = rho_var
    if log_density_current is not None:
        log_dens_x0 = log_density_current
    else:
        log_dens_x0 = log_density(x0)

    width_state = slice_state or SliceWidthState(w=0.2)

    rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
        log_density=log_density,
        x0=x0,
        lower=rho_lower,
        upper=rho_upper,
        width_state=width_state,
        rng=rng,
        log_density_x0=log_dens_x0,
    )

    if cache.rho_adaptive_width and sweep_idx < tune:
        update_slice_width(width_state, steps_left, steps_right)

    return rho_new, log_density_new


# ---------------------------------------------------------------------------
# Non-separable (3-ρ) variant: η draw and ρ draws
# ---------------------------------------------------------------------------


def _sample_eta_nonseparable(
    state: FlowGibbsState,
    y: np.ndarray,
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    cache: FlowGibbsCacheNS | None = None,
    chebyshev_degree: int = 30,
) -> np.ndarray:
    r"""Block 2 (non-separable): Draw :math:`\eta` via sparse N×N precision.

    The conditional posterior is

    .. math::

        \eta \mid \cdot \sim N(m_\eta, \Sigma_\eta)

    where :math:`\Sigma_\eta^{-1} = P_\eta = A^\top A / \sigma^2 + \mathrm{diag}(\omega)`
    and :math:`A = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w`.

    This is the same structure as the single-ρ sampler but with an
    :math:`N \times N` precision matrix instead of :math:`n \times n`.

    Parameters
    ----------
    state : FlowGibbsState
        Current state.
    y : ndarray of shape (N,)
        Integer response vector.
    X : ndarray of shape (N, k)
        Design matrix.
    rng : numpy.random.Generator
        Random state.
    cache : FlowGibbsCacheNS, optional
        Precomputed data.
    chebyshev_degree : int, default 30
        Degree of Chebyshev polynomial approximation.

    Returns
    -------
    eta_new : ndarray of shape (N,)
        New draw of the latent field.
    """
    N = X.shape[0]
    rho_d = state.rho_d
    rho_o = state.rho_o
    rho_w = state.rho_w
    sigma2 = state.sigma2
    omega = state.omega
    beta = state.beta

    # Build P = A^T A / sigma2 + diag(omega) from precomputed pieces
    # A = I_N - rho_d * Wd - rho_o * Wo - rho_w * Ww
    # A^T A = I_N/sigma2 + rho_d^2 * WdWd/sigma2 + ... + cross terms
    #       - rho_d * Wd_sym/sigma2 - rho_o * Wo_sym/sigma2 - rho_w * Ww_sym/sigma2
    #       + rho_d*rho_o * WdWo/sigma2 + rho_d*rho_w * WdWw/sigma2 + rho_o*rho_w * WoWw/sigma2
    P = (
        sp.eye(N, format="csr") / sigma2
        + sp.diags(omega, format="csr")
        - rho_d * cache.Wd_sym / sigma2
        - rho_o * cache.Wo_sym / sigma2
        - rho_w * cache.Ww_sym / sigma2
        + rho_d**2 * cache.WdWd / sigma2
        + rho_o**2 * cache.WoWo / sigma2
        + rho_w**2 * cache.WwWw / sigma2
        + rho_d * rho_o * cache.WdWo / sigma2
        + rho_d * rho_w * cache.WdWw / sigma2
        + rho_o * rho_w * cache.WoWw / sigma2
    )

    # Right-hand side: A^T Xβ / σ² + κ
    # A^T = I - rho_d * Wd^T - rho_o * Wo^T - rho_w * Ww^T
    Xbeta = X @ beta
    kappa = (y - state.alpha) / 2.0
    rhs = (
        Xbeta / sigma2
        - rho_d * cache.Wd.T @ Xbeta / sigma2
        - rho_o * cache.Wo.T @ Xbeta / sigma2
        - rho_w * cache.Ww.T @ Xbeta / sigma2
        + kappa
    )

    # Dispatch to Chebyshev sampler
    draw = chebyshev_sample(P, rhs, rng=rng, degree=chebyshev_degree)
    return draw.x


def _sample_rho_nonseparable(
    state: FlowGibbsState,
    cache: FlowGibbsCacheNS,
    priors: FlowGibbsPriors,
    y: np.ndarray,
    X: np.ndarray,
    *,
    which_rho: str,
    rng: np.random.Generator,
    slice_state: SliceWidthState | None = None,
    log_density_current: float | None = None,
    sweep_idx: int = 0,
    tune: int = 0,
) -> tuple[float, float]:
    r"""Collapsed 1-D slice sampler for :math:`\rho_d`, :math:`\rho_o`, or :math:`\rho_w`.

    Uses the marginal (collapsed) posterior that integrates out :math:`\eta`.
    The collapsed log-density is:

    .. math::

        \log p(\rho_k \mid \cdot) = \log|A|
            - \tfrac{1}{2}\log|P_\eta|
            + \tfrac{1}{2}\,\mathit{rhs}^\top P_\eta^{-1}\,\mathit{rhs}

    where :math:`A = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w` and
    :math:`P_\eta = A^\top A / \sigma^2 + \mathrm{diag}(\omega)`.

    Parameters
    ----------
    state : FlowGibbsState
        Current state.
    cache : FlowGibbsCacheNS
        Precomputed data.
    priors : FlowGibbsPriors
        Prior hyperparameters.
    y : ndarray of shape (N,)
        Integer response vector.
    X : ndarray of shape (N, k)
        Design matrix.
    which_rho : {"rho_d", "rho_o", "rho_w"}
        Which parameter to sample.
    rng : numpy.random.Generator
        Random state.
    log_density_current : float, optional
        Cached log-density at current value.
    sweep_idx : int
        Current sweep index (for adaptive width).
    tune : int
        Number of warmup sweeps.

    Returns
    -------
    rho_new : float
        New draw of the sampled parameter.
    log_density_new : float
        Log-density at the new value (for caching).
    """
    N = cache.N
    sigma2 = state.sigma2
    omega = state.omega
    Xbeta = X @ state.beta
    kappa = (y - state.alpha) / 2.0
    logdet_fn = cache.logdet_fn

    rho_d = state.rho_d
    rho_o = state.rho_o
    rho_w = state.rho_w

    rho_lower = priors.rho_lower
    rho_upper = priors.rho_upper

    # Select which rho to vary
    if which_rho == "rho_d":
        rho_var = rho_d
    elif which_rho == "rho_o":
        rho_var = rho_o
    elif which_rho == "rho_w":
        rho_var = rho_w
    else:
        raise ValueError(
            f"which_rho must be 'rho_d', 'rho_o', or 'rho_w', got {which_rho!r}"
        )

    # Precompute base precision (constant across rho candidates within one slice step)
    # P = I_N/σ² + diag(ω) + terms that depend on rho_d, rho_o, rho_w
    base = sp.eye(N, format="csr") / sigma2 + sp.diags(omega, format="csr")

    # Precompute σ²-scaled matrix pieces
    Wd_s2 = cache.Wd_sym / sigma2
    Wo_s2 = cache.Wo_sym / sigma2
    Ww_s2 = cache.Ww_sym / sigma2
    WdWd_s2 = cache.WdWd / sigma2
    WoWo_s2 = cache.WoWo / sigma2
    WwWw_s2 = cache.WwWw / sigma2
    WdWo_s2 = cache.WdWo / sigma2
    WdWw_s2 = cache.WdWw / sigma2
    WoWw_s2 = cache.WoWw / sigma2

    # Precompute W^T @ Xbeta / sigma2 for each weight matrix
    WdXbeta_s2 = cache.Wd.T @ Xbeta / sigma2
    WoXbeta_s2 = cache.Wo.T @ Xbeta / sigma2
    WwXbeta_s2 = cache.Ww.T @ Xbeta / sigma2
    Xbeta_s2 = Xbeta / sigma2

    _lanczos_rng = np.random.default_rng(rng.integers(2**31))

    def log_density(rho_var_val: float) -> float:
        """Collapsed log-density for rho_d, rho_o, or rho_w."""
        # Set the varying rho
        if which_rho == "rho_d":
            rd, ro, rw = rho_var_val, rho_o, rho_w
        elif which_rho == "rho_o":
            rd, ro, rw = rho_d, rho_var_val, rho_w
        else:
            rd, ro, rw = rho_d, rho_o, rho_var_val

        # 1. Jacobian: log|A| via flow logdet
        logdet_A = logdet_fn(rd, ro, rw)

        # 2. Right-hand side: A^T Xβ / σ² + κ
        # A^T = I - rho_d Wd^T - rho_o Wo^T - rho_w Ww^T
        rhs = Xbeta_s2 - rd * WdXbeta_s2 - ro * WoXbeta_s2 - rw * WwXbeta_s2 + kappa

        # 3. Precision: P = base - rho_d*Wd_s2 - rho_o*Wo_s2 - rho_w*Ww_s2
        #    + rho_d^2*WdWd_s2 + rho_o^2*WoWo_s2 + rho_w^2*WwWw_s2
        #    + rho_d*rho_o*WdWo_s2 + rho_d*rho_w*WdWw_s2 + rho_o*rho_w*WoWw_s2
        P = (
            base
            - rd * Wd_s2
            - ro * Wo_s2
            - rw * Ww_s2
            + rd**2 * WdWd_s2
            + ro**2 * WoWo_s2
            + rw**2 * WwWw_s2
            + rd * ro * WdWo_s2
            + rd * rw * WdWw_s2
            + ro * rw * WoWw_s2
        )

        # 4. log|P_η| via Lanczos
        log_det_P = lanczos_logdet(
            P,
            n_probes=5,
            lanczos_deg=20,
            rng=_lanczos_rng,
        )

        # 5. Solve P m = rhs via CG
        m = cg_solve(P, rhs)

        # 6. Quadratic form
        quad = float(rhs @ m)

        return logdet_A - 0.5 * log_det_P + 0.5 * quad

    # Slice sampling
    x0 = rho_var
    if log_density_current is not None:
        log_dens_x0 = log_density_current
    else:
        log_dens_x0 = log_density(x0)

    width_state = slice_state or SliceWidthState(w=0.2)

    rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
        log_density=log_density,
        x0=x0,
        lower=rho_lower,
        upper=rho_upper,
        width_state=width_state,
        rng=rng,
        log_density_x0=log_dens_x0,
    )

    if cache.rho_adaptive_width and sweep_idx < tune:
        update_slice_width(width_state, steps_left, steps_right)

    return rho_new, log_density_new


# ---------------------------------------------------------------------------
# Main chain runners
# ---------------------------------------------------------------------------


def run_flow_chain_separable(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: FlowGibbsPriors,
    cache: FlowGibbsCache,
    init: FlowGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng: np.random.Generator | None = None,
    chebyshev_degree: int = 30,
) -> dict[str, np.ndarray]:
    """Run one chain of the separable flow PG-Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Integer response vector.
    X : ndarray of shape (N, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Row-standardised spatial weights matrix.
    priors : FlowGibbsPriors
        Prior hyperparameters.
    cache : FlowGibbsCache
        Precomputed data.
    init : FlowGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws to keep.
    tune : int
        Number of warmup draws.
    thin : int, default 1
        Keep every ``thin``-th draw.
    return_eta : bool, default False
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.
    chebyshev_degree : int, default 30
        Chebyshev polynomial degree for η draw.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho_d``, ``rho_o``, ``beta``,
        ``sigma``, ``alpha``, and optionally ``eta``.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, k = X.shape
    n = cache.n
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    # Pre-allocate storage
    rho_d_samples = np.empty(n_keep, dtype=np.float64)
    rho_o_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, N), dtype=np.float64)
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, N), dtype=np.float64) if return_eta else None

    # Copy initial state
    state = FlowGibbsState(
        eta=init.eta.copy(),
        beta=init.beta.copy(),
        sigma2=init.sigma2,
        rho_d=init.rho_d,
        rho_o=init.rho_o,
        rho_w=None,  # deterministic: rho_w = -rho_d * rho_o
        alpha=init.alpha,
        omega=init.omega.copy(),
    )

    XtX = cache.XtX
    log_density_rho_d = None
    log_density_rho_o = None

    # Per-chain slice state (mutable, not shared across chains)
    slice_state = FlowGibbsSliceState()

    I_n = np.eye(n)

    for i in range(total_iters):
        # --- Block 1: ω | η, α ---
        state.omega = _sample_omega_flow(y, state.alpha, state.eta, rng=rng)

        # --- Block 2: η | ω, ρ_d, ρ_o, β, σ² (Kronecker) ---
        state.eta = _sample_eta_separable(
            state,
            y,
            X,
            rng=rng,
            cache=cache,
            chebyshev_degree=chebyshev_degree,
        )

        # Recompute Aη and Xβ with new eta
        Ld = I_n - state.rho_d * cache.W_dense
        Lo = I_n - state.rho_o * cache.W_dense
        A_eta = kron_matvec(state.eta, Ld, Lo)
        Xbeta = X @ state.beta

        # --- Block 3: β | η, ρ, σ² ---
        state.beta = _sample_beta_flow(state, X, XtX, priors, A_eta, rng=rng)
        Xbeta = X @ state.beta

        # --- Block 4: σ² | η, ρ, β ---
        state.sigma2 = _sample_sigma2_flow(state, priors, A_eta, Xbeta, rng=rng)

        # --- Block 5: ρ_d | · (collapsed, η integrated out) ---
        state.rho_d, log_density_rho_d = _sample_rho_separable(
            state,
            cache,
            priors,
            y,
            X,
            which_rho="rho_d",
            rng=rng,
            slice_state=slice_state.rho_d,
            log_density_current=log_density_rho_d,
            sweep_idx=i,
            tune=tune,
        )
        # Update deterministic rho_w
        state.rho_w = -state.rho_d * state.rho_o

        # --- Block 6: ρ_o | · (collapsed, η integrated out) ---
        state.rho_o, log_density_rho_o = _sample_rho_separable(
            state,
            cache,
            priors,
            y,
            X,
            which_rho="rho_o",
            rng=rng,
            slice_state=slice_state.rho_o,
            log_density_current=log_density_rho_o,
            sweep_idx=i,
            tune=tune,
        )
        # Update deterministic rho_w
        state.rho_w = -state.rho_d * state.rho_o

        # --- Block 7: α | y, η ---
        state.alpha = _sample_alpha_flow(state, y, priors, rng=rng)

        # --- Store post-warmup draws ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_d_samples[idx] = state.rho_d
                rho_o_samples[idx] = state.rho_o
                beta_samples[idx] = state.beta
                sigma_samples[idx] = np.sqrt(state.sigma2)
                alpha_samples[idx] = state.alpha
                log_lik_samples[idx] = _nb_loglik_pointwise_flow(
                    y, state.eta, state.alpha
                )
                eta_norm_samples[idx] = float(state.eta @ state.eta)
                if return_eta:
                    eta_samples[idx] = state.eta

    result = {
        "rho_d": rho_d_samples,
        "rho_o": rho_o_samples,
        "rho_w": -rho_d_samples * rho_o_samples,  # deterministic
        "beta": beta_samples,
        "sigma": sigma_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
    }
    if return_eta:
        result["eta"] = eta_samples

    return result


def run_flow_chain_nonseparable(
    y: np.ndarray,
    X: np.ndarray,
    priors: FlowGibbsPriors,
    cache: FlowGibbsCacheNS,
    init: FlowGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng: np.random.Generator | None = None,
    chebyshev_degree: int = 30,
) -> dict[str, np.ndarray]:
    """Run one chain of the non-separable (3-ρ) flow PG-Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Integer response vector.
    X : ndarray of shape (N, k)
        Design matrix.
    priors : FlowGibbsPriors
        Prior hyperparameters.
    cache : FlowGibbsCacheNS
        Precomputed data.
    init : FlowGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws to keep.
    tune : int
        Number of warmup draws.
    thin : int, default 1
        Keep every ``thin``-th draw.
    return_eta : bool, default False
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.
    chebyshev_degree : int, default 30
        Chebyshev polynomial degree for η draw.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho_d``, ``rho_o``, ``rho_w``,
        ``beta``, ``sigma``, ``alpha``, and optionally ``eta``.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    # Pre-allocate storage
    rho_d_samples = np.empty(n_keep, dtype=np.float64)
    rho_o_samples = np.empty(n_keep, dtype=np.float64)
    rho_w_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, N), dtype=np.float64)
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, N), dtype=np.float64) if return_eta else None

    # Copy initial state
    state = FlowGibbsState(
        eta=init.eta.copy(),
        beta=init.beta.copy(),
        sigma2=init.sigma2,
        rho_d=init.rho_d,
        rho_o=init.rho_o,
        rho_w=init.rho_w,
        alpha=init.alpha,
        omega=init.omega.copy(),
    )

    XtX = cache.XtX
    log_density_rho_d = None
    log_density_rho_o = None
    log_density_rho_w = None

    # Per-chain slice state (mutable, not shared across chains)
    slice_state = FlowGibbsSliceStateNS()

    for i in range(total_iters):
        # --- Block 1: ω | η, α ---
        state.omega = _sample_omega_flow(y, state.alpha, state.eta, rng=rng)

        # --- Block 2: η | ω, ρ_d, ρ_o, ρ_w, β, σ² (sparse N×N) ---
        state.eta = _sample_eta_nonseparable(
            state,
            y,
            X,
            rng=rng,
            cache=cache,
            chebyshev_degree=chebyshev_degree,
        )

        # Recompute Aη and Xβ with new eta
        I_N = sp.eye(N, format="csr")
        A = (
            I_N
            - state.rho_d * cache.Wd
            - state.rho_o * cache.Wo
            - state.rho_w * cache.Ww
        )
        A_eta = A @ state.eta
        Xbeta = X @ state.beta

        # --- Block 3: β | η, ρ, σ² ---
        state.beta = _sample_beta_flow(state, X, XtX, priors, A_eta, rng=rng)
        Xbeta = X @ state.beta

        # --- Block 4: σ² | η, ρ, β ---
        state.sigma2 = _sample_sigma2_flow(state, priors, A_eta, Xbeta, rng=rng)

        # --- Block 5: ρ_d | · (collapsed) ---
        state.rho_d, log_density_rho_d = _sample_rho_nonseparable(
            state,
            cache,
            priors,
            y,
            X,
            which_rho="rho_d",
            rng=rng,
            slice_state=slice_state.rho_d,
            log_density_current=log_density_rho_d,
            sweep_idx=i,
            tune=tune,
        )

        # --- Block 6: ρ_o | · (collapsed) ---
        state.rho_o, log_density_rho_o = _sample_rho_nonseparable(
            state,
            cache,
            priors,
            y,
            X,
            which_rho="rho_o",
            rng=rng,
            slice_state=slice_state.rho_o,
            log_density_current=log_density_rho_o,
            sweep_idx=i,
            tune=tune,
        )

        # --- Block 7: ρ_w | · (collapsed) ---
        state.rho_w, log_density_rho_w = _sample_rho_nonseparable(
            state,
            cache,
            priors,
            y,
            X,
            which_rho="rho_w",
            rng=rng,
            slice_state=slice_state.rho_w,
            log_density_current=log_density_rho_w,
            sweep_idx=i,
            tune=tune,
        )

        # --- Block 8: α | y, η ---
        state.alpha = _sample_alpha_flow(state, y, priors, rng=rng)

        # --- Store post-warmup draws ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_d_samples[idx] = state.rho_d
                rho_o_samples[idx] = state.rho_o
                rho_w_samples[idx] = state.rho_w
                beta_samples[idx] = state.beta
                sigma_samples[idx] = np.sqrt(state.sigma2)
                alpha_samples[idx] = state.alpha
                log_lik_samples[idx] = _nb_loglik_pointwise_flow(
                    y, state.eta, state.alpha
                )
                eta_norm_samples[idx] = float(state.eta @ state.eta)
                if return_eta:
                    eta_samples[idx] = state.eta

    result = {
        "rho_d": rho_d_samples,
        "rho_o": rho_o_samples,
        "rho_w": rho_w_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
    }
    if return_eta:
        result["eta"] = eta_samples

    return result
