"""Spatial Durbin Error Model (SDEM).

y = X @ beta1 + W @ X @ beta2 + u,
u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Combines spatially lagged covariates (SLX) with a spatially autocorrelated
error process (SEM). No spatial lag on y, so rho is absent.
Jacobian log|I - lambda*W| is required for the error process.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .base import SpatialModel
from ..logdet import make_logdet_fn


class SDEM(SpatialModel):
    """Bayesian Spatial Durbin Error Model.

    .. math::
        y = X\\beta_1 + WX\\beta_2 + u, \\quad u = \\lambda Wu + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method
        See :class:`SpatialModel`.

    Priors (``priors`` dict keys)
    ------------------------------
    lam_lower, lam_upper : float, default -1, 1
    beta_mu : float, default 0
    beta_sigma : float, default 1e6
    sigma_sigma : float, default 10
    """

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SDEM regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        Z = np.hstack([self._X, self._WX])  # (n, 2k)

        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(self._W_dense, method=self.logdet_method,
                                   rho_min=lam_lower, rho_max=lam_upper)
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # epsilon = (I - lambda*W)(y - Z@beta) = resid - lambda*(W @ resid)
            resid = self._y - pt.dot(Z, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            # PyMC does not allow symbolic expressions as `observed`; encode
            # the transformed-error likelihood directly as a Potential.
            logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
            pm.Potential("eps_loglik", logp_eps)

            # Jacobian for error process
            pm.Potential("jacobian", logdet_fn(lam))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SDEM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # For SDEM: spatial multiplier applies to errors, not X.
        # dy/dX_k = beta1_k, dy/dWX_k = beta2_k (local impacts only)
        # Direct = beta1, Indirect = beta2 (interpreted as neighbour effect),
        # Total = beta1 + beta2.
        beta = self._posterior_mean("beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]
        return {
            "direct": beta1[self._wx_column_indices],
            "indirect": beta2,
            "total": beta1[self._wx_column_indices] + beta2,
            "feature_names": self._wx_feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X, self._WX])
        return Z @ beta
