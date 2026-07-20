r"""Chain runner for the Gaussian panel flow Gibbs sampler.

Provides :func:`run_gaussian_panel_flow_chain` — the top-level entry
point that handles setup, warmup, sampling, and trace assembly.

See Also
--------
bayespecon.samplers.panel_flow._blocks_gaussian
    Individual Gibbs block samplers.
bayespecon.samplers.panel_flow._state
    Dataclasses for state, cache, priors, and trace.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ..._logdet import make_logdet_numpy_fn
from .._utils._slice import SliceWidthState
from ._blocks_gaussian import (
    _sample_beta_panel,
    _sample_eta_panel,
    _sample_gamma,
    _sample_rho_d_panel,
    _sample_rho_o_panel,
    _sample_sigma2_u,
    _sample_sigma2_y,
)
from ._state import (
    PanelGaussianCache,
    PanelGaussianPriors,
    PanelGaussianState,
    PanelGaussianTrace,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_and_reshape_y(y: np.ndarray, n: int) -> np.ndarray:
    """Reshape y to (n², T) column-major if needed."""
    if y.ndim == 3 and y.shape[0] == n and y.shape[1] == n:
        T = y.shape[2]
        return y.reshape(n * n, T, order="F")
    if y.ndim == 2 and y.shape[0] == n * n:
        return y
    raise ValueError(f"y must be (n², T) or (n, n, T); got shape {y.shape} with n={n}")


def _validate_X(
    X: np.ndarray, n2: int, T: int
) -> tuple[np.ndarray, bool, np.ndarray | None]:
    """Validate X and determine if time-invariant. Returns (X, time_invariant, XtX)."""
    if X.ndim == 2:
        if X.shape[0] != n2:
            raise ValueError(f"X must have {n2} rows when 2-D; got {X.shape[0]}")
        XtX = X.T @ X
        return X, True, XtX
    if X.ndim == 3:
        if X.shape[0] != n2 or X.shape[1] != T:
            raise ValueError(f"X must be (n², T, k) when 3-D; got {X.shape}")
        return X, False, None
    raise ValueError(f"X must be 2-D or 3-D; got {X.ndim}-D")


def _ols_init_beta(y: np.ndarray, X: np.ndarray, time_invariant_X: bool) -> np.ndarray:
    """OLS initialization for β from pooled data."""
    if time_invariant_X:
        # Stack y and X across T
        # y: (n², T), X: (n², k)
        # Simple OLS: β = (X'X)^{-1} X'y_mean
        y_mean = y.mean(axis=1)
        return np.linalg.lstsq(X, y_mean, rcond=None)[0]
    else:
        # Stack: y_flat (n²*T,), X_flat (n²*T, k)
        y.shape[1]
        y_flat = y.ravel(order="F")
        X_flat = X.reshape(-1, X.shape[2], order="F")
        return np.linalg.lstsq(X_flat, y_flat, rcond=None)[0]


# ---------------------------------------------------------------------------
# Main chain runner
# ---------------------------------------------------------------------------


def run_gaussian_panel_flow_chain(
    y: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    n_draws: int,
    n_warmup: int,
    *,
    priors: PanelGaussianPriors | None = None,
    rho_init: float = 0.0,
    gamma_init: float = 0.0,
    ffbs_method: str = "eigenbasis",
    VkronV_threshold: int = 400,
    store_eta: bool = True,
    eta_thin: int = 1,
    seed: int | None = None,
) -> PanelGaussianTrace:
    r"""Run a Kronecker eigenbasis FFBS Gibbs chain for the Gaussian panel flow model.

    Setup (one-time costs)
    ---------------------
    1. Validate and reshape :math:`y` to :math:`(n^2, T)` column-major
    2. Validate :math:`W` — check symmetry, convert to dense
    3. Eigendecomposition: :math:`W = V \Lambda V^\top` via ``np.linalg.eigh``
    4. Precompute :math:`(V \otimes V)` if :math:`n^2 \leq` ``VkronV_threshold``
    5. Precompute :math:`X^\top X` if time-invariant :math:`X`
    6. Build :class:`PanelGaussianCache`
    7. Initialize :class:`PanelGaussianState`
    8. Initialize :class:`SliceWidthState` for :math:`\rho_d, \rho_o`

    Warmup
    ------
    Run ``n_warmup`` iterations. Adapt slice widths for :math:`\rho_d`,
    :math:`\rho_o`. Discard all draws.

    Sampling
    --------
    Run ``n_draws`` iterations. Store draws according to ``store_eta``
    and ``eta_thin``.

    Parameters
    ----------
    y : ndarray of shape (n², T) or (n, n, T)
        Observed OD flows. If 3-D, reshaped to column-major :math:`(n^2, T)`.
    W : ndarray of shape (n, n)
        Spatial weights matrix. Must be symmetric.
    X : ndarray of shape (n², T, k) or (n², k)
        Covariate array. If 2-D, covariates are time-invariant.
    n_draws : int
        Number of post-warmup draws.
    n_warmup : int
        Number of warmup draws (discarded).
    priors : PanelGaussianPriors, optional
        Prior hyperparameters. Uses defaults if None.
    rho_init : float, default 0.0
        Initial value for :math:`\rho_d` and :math:`\rho_o`.
    gamma_init : float, default 0.0
        Initial value for :math:`\gamma`.
    ffbs_method : str, default "eigenbasis"
        FFBS method. Only ``"eigenbasis"`` is currently supported.
    VkronV_threshold : int, default 400
        Use explicit :math:`(V \otimes V)` matrix only when :math:`n^2`
        is at most this value.
    store_eta : bool, default True
        Whether to store the latent field draws.
    eta_thin : int, default 1
        Store every ``eta_thin``-th draw of :math:`\eta`.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PanelGaussianTrace
        Posterior draws for all parameters.
    """
    if priors is None:
        priors = PanelGaussianPriors()

    rng = np.random.default_rng(seed)

    # --- Setup ---
    n = W.shape[0]
    n2 = n * n

    # Validate W
    if W.shape != (n, n):
        raise ValueError(f"W must be square; got shape {W.shape}")
    if not np.allclose(W, W.T, atol=1e-10):
        warnings.warn(
            "W is not symmetric; symmetrising via (W + Wᵀ)/2. "
            "The eigenbasis decomposition requires symmetric W.",
            UserWarning,
            stacklevel=2,
        )
        W = (W + W.T) / 2.0
    W_dense = np.asarray(W, dtype=np.float64)

    # Eigendecomposition
    eigs_W, V = np.linalg.eigh(W_dense)

    # Validate and reshape y
    y = _validate_and_reshape_y(np.asarray(y, dtype=np.float64), n)
    T = y.shape[1]

    # Validate X
    X, time_invariant_X, XtX = _validate_X(np.asarray(X, dtype=np.float64), n2, T)
    k = X.shape[-1]

    # Precompute (V⊗V) if small enough
    VkronV = None
    if n2 <= VkronV_threshold:
        VkronV = np.kron(V, V)

    # Logdet function — shared eigenvalue evaluator from _logdet (the
    # sampler's rho bounds keep 1 - rho*lambda positive, so this matches the
    # old local np.maximum-clamped form exactly within bounds).
    logdet_fn = make_logdet_numpy_fn(None, eigs=eigs_W, method="eigenvalue")

    # Stability bounds for ρ
    eig_max = np.max(np.abs(eigs_W))
    rho_lower = max(-0.999, -1.0 / eig_max + 0.001)
    rho_upper = min(0.999, 1.0 / eig_max - 0.001)

    # Prior arrays
    beta_prior_mean = np.full(k, priors.beta_mu, dtype=np.float64)
    beta_prior_prec = np.eye(k) / float(priors.beta_sigma) ** 2

    # Build cache
    cache = PanelGaussianCache(
        y=y,
        X=X,
        n=n,
        T=T,
        W_dense=W_dense,
        eigs_W=eigs_W,
        V=V,
        VkronV=VkronV,
        logdet_fn=logdet_fn,
        beta_prior_mean=beta_prior_mean,
        beta_prior_prec=beta_prior_prec,
        a_u=priors.sigma2_alpha,
        b_u=priors.sigma2_beta,
        a_y=priors.sigma2_y_alpha,
        b_y=priors.sigma2_y_beta,
        gamma_prior_mean=0.0,
        gamma_prior_var=priors.gamma_prior_var,
        rho_bounds=(rho_lower, rho_upper),
        XtX=XtX,
        time_invariant_X=time_invariant_X,
        ffbs_method=ffbs_method,
        VkronV_threshold=VkronV_threshold,
    )

    # Initialize state
    beta_init = _ols_init_beta(y, X, time_invariant_X)
    sigma2_u_init = 1.0
    sigma2_y_init = 1.0
    # Initialize η from y + small noise
    eta_init = y + rng.normal(0, 0.1, size=y.shape)

    state = PanelGaussianState(
        eta=eta_init,
        beta=beta_init,
        sigma2_u=sigma2_u_init,
        sigma2_y=sigma2_y_init,
        rho_d=rho_init,
        rho_o=rho_init,
        gamma=gamma_init,
    )

    # Slice width states for ρ_d, ρ_o
    rho_d_width = SliceWidthState(w=0.1)
    rho_o_width = SliceWidthState(w=0.1)

    # --- Warmup ---
    _log.info("Starting warmup (%d iterations)", n_warmup)
    ytilde = None
    for i in range(n_warmup):
        state.eta, ytilde = _sample_eta_panel(y, X, state, cache, rng)
        state.beta = _sample_beta_panel(state.eta, X, state, cache, rng)
        state.sigma2_u = _sample_sigma2_u(state.eta, X, state, cache, rng)
        state.sigma2_y = _sample_sigma2_y(y, state.eta, state, cache, rng)
        state.gamma = _sample_gamma(state.eta, X, state, cache, rng)
        state.rho_d, rho_d_width = _sample_rho_d_panel(
            state, cache, ytilde, rho_d_width, rng
        )
        state.rho_o, rho_o_width = _sample_rho_o_panel(
            state, cache, ytilde, rho_o_width, rng
        )
        state.invalidate_cache()

        if (i + 1) % 100 == 0:
            _log.info(
                "Warmup %d/%d: rho_d=%.4f rho_o=%.4f gamma=%.4f "
                "sigma2_u=%.4f sigma2_y=%.4f",
                i + 1,
                n_warmup,
                state.rho_d,
                state.rho_o,
                state.gamma,
                state.sigma2_u,
                state.sigma2_y,
            )

    # --- Sampling ---
    _log.info("Starting sampling (%d iterations)", n_draws)

    # Allocate trace arrays
    if store_eta:
        n_eta_draws = (n_draws + eta_thin - 1) // eta_thin
        eta_trace = np.empty((n_eta_draws, n2, T))
    else:
        eta_trace = None
    beta_trace = np.empty((n_draws, k))
    sigma2_u_trace = np.empty(n_draws)
    sigma2_y_trace = np.empty(n_draws)
    rho_d_trace = np.empty(n_draws)
    rho_o_trace = np.empty(n_draws)
    gamma_trace = np.empty(n_draws)
    loglik_trace = np.empty(n_draws)

    from ._eigenbasis import kf_log_likelihood

    eta_draw_idx = 0
    for i in range(n_draws):
        state.eta, ytilde = _sample_eta_panel(y, X, state, cache, rng)
        state.beta = _sample_beta_panel(state.eta, X, state, cache, rng)
        state.sigma2_u = _sample_sigma2_u(state.eta, X, state, cache, rng)
        state.sigma2_y = _sample_sigma2_y(y, state.eta, state, cache, rng)
        state.gamma = _sample_gamma(state.eta, X, state, cache, rng)
        state.rho_d, rho_d_width = _sample_rho_d_panel(
            state, cache, ytilde, rho_d_width, rng
        )
        state.rho_o, rho_o_width = _sample_rho_o_panel(
            state, cache, ytilde, rho_o_width, rng
        )
        state.invalidate_cache()

        # Store draws
        beta_trace[i] = state.beta
        sigma2_u_trace[i] = state.sigma2_u
        sigma2_y_trace[i] = state.sigma2_y
        rho_d_trace[i] = state.rho_d
        rho_o_trace[i] = state.rho_o
        gamma_trace[i] = state.gamma

        # Log-likelihood at current parameters
        loglik_trace[i] = kf_log_likelihood(
            state.rho_d,
            state.rho_o,
            state.gamma,
            state.sigma2_u,
            state.sigma2_y,
            cache.eigs_W,
            ytilde,
        )

        # Store η (with thinning)
        if store_eta and (i % eta_thin == 0):
            eta_trace[eta_draw_idx] = state.eta
            eta_draw_idx += 1

        if (i + 1) % 100 == 0:
            _log.info(
                "Draw %d/%d: rho_d=%.4f rho_o=%.4f gamma=%.4f",
                i + 1,
                n_draws,
                state.rho_d,
                state.rho_o,
                state.gamma,
            )

    # Trim eta trace if thinned
    if store_eta and eta_draw_idx < n_eta_draws:
        eta_trace = eta_trace[:eta_draw_idx]

    return PanelGaussianTrace(
        eta=eta_trace,
        beta=beta_trace,
        sigma2_u=sigma2_u_trace,
        sigma2_y=sigma2_y_trace,
        rho_d=rho_d_trace,
        rho_o=rho_o_trace,
        gamma=gamma_trace,
        loglik=loglik_trace,
    )
