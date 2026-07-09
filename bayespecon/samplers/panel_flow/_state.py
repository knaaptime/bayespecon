r"""Data classes for the Gaussian panel flow Gibbs sampler.

Provides mutable state, immutable cache, prior specification, and
posterior trace dataclasses for the Kronecker eigenbasis FFBS Gibbs
sampler for Gaussian spatial interaction panel models.

See Also
--------
bayespecon.samplers.panel_flow._eigenbasis
    Scalar Kalman filter, FFBS backward pass, and eigenbasis transforms.
bayespecon.samplers.negbin._flow
    Cross-sectional flow Gibbs sampler with the same state/cache pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np

from ...models.priors import PanelGaussianPriors  # noqa: F401

# ---------------------------------------------------------------------------
# Kalman filter output
# ---------------------------------------------------------------------------


class KFOutput(NamedTuple):
    """Output from the scalar Kalman filter forward pass.

    All arrays have the mode dimension first: shape ``(n², ...)``.

    Attributes
    ----------
    filtered_means : ndarray of shape (n², T)
        Filtered means :math:`m_{m,t|t}` for each mode and time step.
    filtered_vars : ndarray of shape (n², T)
        Filtered variances :math:`p_{m,t|t}` for each mode and time step.
    pred_vars : ndarray of shape (n², T)
        One-step-ahead prediction variances :math:`p_{m,t|t-1}`.
        Stored for use in the FFBS backward smoother gain computation.
    log_likelihood : float
        Marginal log-likelihood :math:`\\log p(y_{1:T} \\mid \\theta)`
        accumulated over the forward pass.
    """

    filtered_means: np.ndarray
    filtered_vars: np.ndarray
    pred_vars: np.ndarray
    log_likelihood: float


# ---------------------------------------------------------------------------
# Mutable state
# ---------------------------------------------------------------------------


@dataclass
class PanelGaussianState:
    r"""Mutable state for one Gibbs sweep of the panel flow sampler.

    All arrays are numpy arrays; scalars are Python floats.

    Parameters
    ----------
    eta : ndarray of shape (n², T)
        Latent field in the **original** (spatial) basis.
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma2_u : float
        Innovation variance :math:`\sigma^2_u`.
    sigma2_y : float
        Observation variance :math:`\sigma^2_y`.
    rho_d : float
        Destination autoregressive parameter.
    rho_o : float
        Origin autoregressive parameter.
    gamma : float
        Temporal AR(1) parameter, :math:`\gamma \in (-1, 1)`.
    """

    eta: np.ndarray
    beta: np.ndarray
    sigma2_u: float
    sigma2_y: float
    rho_d: float
    rho_o: float
    gamma: float

    # Cached derived quantities — invalidated when ρ changes
    _Ld: np.ndarray | None = field(default=None, repr=False)
    _Lo: np.ndarray | None = field(default=None, repr=False)
    _q_modes: np.ndarray | None = field(default=None, repr=False)

    def get_filter_matrices(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Return spatial filter matrices :math:`(L_d, L_o)`, computing if not cached.

        Parameters
        ----------
        W : ndarray of shape (n, n)
            Spatial weights matrix.

        Returns
        -------
        Ld : ndarray of shape (n, n)
            :math:`L_d = I - \rho_d W`.
        Lo : ndarray of shape (n, n)
            :math:`L_o = I - \rho_o W`.
        """
        if self._Ld is None:
            n = W.shape[0]
            self._Ld = np.eye(n) - self.rho_d * W
            self._Lo = np.eye(n) - self.rho_o * W
        return self._Ld, self._Lo

    def get_modal_variances(
        self, eigs_W: np.ndarray, sigma2_u: float | None = None
    ) -> np.ndarray:
        r"""Return modal innovation variances, computing if not cached.

        .. math::

            q_{ij} = \sigma^2_u / [(1 - \rho_d \lambda_i)^2 (1 - \rho_o \lambda_j)^2]

        Parameters
        ----------
        eigs_W : ndarray of shape (n,)
            Eigenvalues of the spatial weights matrix :math:`W`.
        sigma2_u : float, optional
            Innovation variance. If None, uses ``self.sigma2_u``.

        Returns
        -------
        q_modes : ndarray of shape (n²,)
            Modal innovation variances in column-major order.
        """
        if sigma2_u is None:
            sigma2_u = self.sigma2_u
        if self._q_modes is None:
            gains_d = 1.0 - self.rho_d * eigs_W
            gains_o = 1.0 - self.rho_o * eigs_W
            self._q_modes = sigma2_u / np.outer(gains_d**2, gains_o**2).ravel()
        return self._q_modes

    def invalidate_cache(self) -> None:
        """Invalidate cached derived quantities (call when ρ changes)."""
        self._Ld = None
        self._Lo = None
        self._q_modes = None


# ---------------------------------------------------------------------------
# Immutable cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelGaussianCache:
    r"""Precomputed constants for the panel Gaussian Gibbs sampler.

    Immutable and safe to share across chains. Per-chain mutable state
    (slice-sampler widths) is passed separately.

    Parameters
    ----------
    y : ndarray of shape (n², T)
        Observed flows in column-major per period.
    X : ndarray of shape (n², T, k) or (n², k)
        Covariate array. If 2-D, covariates are time-invariant.
    n : int
        Number of spatial units (:math:`N = n^2`).
    T : int
        Number of time periods.
    W_dense : ndarray of shape (n, n)
        Dense spatial weights matrix.
    eigs_W : ndarray of shape (n,)
        Eigenvalues of :math:`W`.
    V : ndarray of shape (n, n)
        Eigenvectors of :math:`W`.
    VkronV : ndarray of shape (n², n²) or None
        Precomputed :math:`V \otimes V`. ``None`` when :math:`n^2`
        exceeds ``VkronV_threshold`` (use implicit Kronecker matvec
        instead).
    logdet_fn : callable
        Function ``logdet_fn(rho) -> float`` computing
        :math:`\log|I_n - \rho W|` for a single scalar :math:`\rho`.
    beta_prior_mean : ndarray of shape (k,)
        Prior mean for :math:`\beta`.
    beta_prior_prec : ndarray of shape (k, k)
        Prior precision matrix :math:`\Lambda_0^{-1}` for :math:`\beta`.
    a_u : float
        Inverse-Gamma shape for :math:`\sigma^2_u`.
    b_u : float
        Inverse-Gamma scale for :math:`\sigma^2_u`.
    a_y : float
        Inverse-Gamma shape for :math:`\sigma^2_y`.
    b_y : float
        Inverse-Gamma scale for :math:`\sigma^2_y`.
    gamma_prior_mean : float
        Prior mean for :math:`\gamma`.
    gamma_prior_var : float
        Prior variance for :math:`\gamma`.
    rho_bounds : tuple of (float, float)
        ``(rho_lower, rho_upper)`` stability bounds.
    XtX : ndarray of shape (k, k) or None
        Precomputed :math:`X^\top X` if covariates are time-invariant.
    time_invariant_X : bool
        ``True`` if :math:`X_t = X` for all :math:`t`.
    ffbs_method : str
        FFBS method: ``"eigenbasis"`` (default) or ``"sparse"`` (fallback).
    VkronV_threshold : int
        Use explicit :math:`V \otimes V` matrix only when :math:`n^2`
        is at most this value.
    """

    y: np.ndarray
    X: np.ndarray
    n: int
    T: int
    W_dense: np.ndarray
    eigs_W: np.ndarray
    V: np.ndarray
    VkronV: np.ndarray | None
    logdet_fn: Callable[[float], float]
    beta_prior_mean: np.ndarray
    beta_prior_prec: np.ndarray
    a_u: float
    b_u: float
    a_y: float
    b_y: float
    gamma_prior_mean: float
    gamma_prior_var: float
    rho_bounds: tuple[float, float]
    XtX: np.ndarray | None
    time_invariant_X: bool
    ffbs_method: str = "eigenbasis"
    VkronV_threshold: int = 400


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Posterior trace
# ---------------------------------------------------------------------------


@dataclass
class PanelGaussianTrace:
    """Posterior draws from the Gaussian panel flow Gibbs sampler.

    Parameters
    ----------
    eta : ndarray of shape (n_draws, n², T) or None
        Latent field draws. ``None`` when ``store_eta=False``.
    beta : ndarray of shape (n_draws, k)
        Regression coefficient draws.
    sigma2_u : ndarray of shape (n_draws,)
        Innovation variance draws.
    sigma2_y : ndarray of shape (n_draws,)
        Observation variance draws.
    rho_d : ndarray of shape (n_draws,)
        Destination autoregressive parameter draws.
    rho_o : ndarray of shape (n_draws,)
        Origin autoregressive parameter draws.
    gamma : ndarray of shape (n_draws,)
        Temporal AR(1) parameter draws.
    loglik : ndarray of shape (n_draws,)
        Marginal log-likelihood from the Kalman filter at each draw.
    """

    eta: np.ndarray | None
    beta: np.ndarray
    sigma2_u: np.ndarray
    sigma2_y: np.ndarray
    rho_d: np.ndarray
    rho_o: np.ndarray
    gamma: np.ndarray
    loglik: np.ndarray
