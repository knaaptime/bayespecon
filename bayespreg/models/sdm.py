"""Spatial Durbin Model (SDM).

y = rho * W @ y + X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

Combines a spatial lag on y (SAR) with spatially lagged covariates (SLX).
Jacobian log|I - rho*W| is required as in the SAR model.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .base import SpatialModel
from ..logdet import make_logdet_fn


class SDM(SpatialModel):
    """Bayesian Spatial Durbin Model.

    .. math::
        y = \\rho Wy + X\\beta_1 + WX\\beta_2 + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method, w_vars
        See :class:`SpatialModel`. Use ``w_vars`` to restrict which X columns
        are spatially lagged.

    Priors (``priors`` dict keys)
    ------------------------------
    rho_lower, rho_upper : float, default -1, 1
    beta_mu : float, default 0
    beta_sigma : float, default 1e6
    sigma_sigma : float, default 10
    """

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SDM regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        k = self._X.shape[1]
        Z = np.hstack([self._X, self._WX])  # (n, 2k)

        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(self._W_dense, method=self.logdet_method,
                                   rho_min=rho_lower, rho_max=rho_upper)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho * self._Wy + pt.dot(Z, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SDM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # SDM impacts: S_k = (I - rho*W)^{-1} (beta1_k*I + beta2_k*W)
        # Direct   = mean diagonal of S_k
        # Total    = mean row sum of S_k
        # Indirect = total - direct
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]

        n = self._W_dense.shape[0]
        W = self._W_dense
        M = np.linalg.inv(np.eye(n) - rho * W)  # (I - rho*W)^{-1}

        direct = np.array([
            np.diag(M @ (beta1[j] * np.eye(n) + b2 * W)).mean()
            for j, b2 in zip(self._wx_column_indices, beta2)
        ])
        total = np.array([
            (M @ (beta1[j] * np.eye(n) + b2 * W)).mean()
            for j, b2 in zip(self._wx_column_indices, beta2)
        ])
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._wx_feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X, self._WX])
        return rho * self._Wy + Z @ beta
