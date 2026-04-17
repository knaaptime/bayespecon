"""Spatial Lag X (SLX) Model.

y = X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

No spatial lag on y, so no Jacobian adjustment is needed and NUTS
converges without difficulty.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .base import SpatialModel


class SLX(SpatialModel):
    """Bayesian SLX (Spatial Lag X) model.

    .. math::
        y = X\\beta_1 + WX\\beta_2 + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method, w_vars
        See :class:`SpatialModel`. Use ``w_vars`` to restrict which X columns
        are spatially lagged.

    Priors (``priors`` dict keys)
    ------------------------------
    beta_mu : float, default 0
        Prior mean for all beta coefficients.
    beta_sigma : float, default 1e6
        Prior std for all beta coefficients (diffuse Normal).
    sigma_sigma : float, default 10
        Scale for HalfNormal prior on sigma.
    """

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SLX regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        n, k = self._X.shape
        # Combine X and WX into one design matrix; beta covers both
        Z = np.hstack([self._X, self._WX])  # (n, 2k)

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = pt.dot(Z, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SLX direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta = self._posterior_mean("beta")
        beta1 = beta[:k]   # coefficients on X
        beta2 = beta[k:k + kw]   # coefficients on WX (excluding intercept-like terms)

        # For SLX: dy/dX_k = beta1_k * I + beta2_k * W
        # Direct = beta1 (diagonal mean of derivative matrix = beta1)
        # Indirect = beta2 * (mean row sum of W) = beta2 * 1 (row-standardised W)
        W = self._W_dense
        direct = beta1[self._wx_column_indices]
        indirect = beta2 * W.mean(axis=1).mean()  # scalar row mean (=1 for row-standardised)
        total = direct + indirect

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._wx_feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        k = self._X.shape[1]
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X, self._WX])
        if beta.shape[0] != 2 * k:
            raise ValueError("Unexpected beta dimension for SLX fitted mean.")
        return Z @ beta
