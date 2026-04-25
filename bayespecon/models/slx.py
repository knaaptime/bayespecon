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

    Notes
    -----
    The ``priors`` dict supports the following keys:

    - ``beta_mu`` (float, default 0): Prior mean for all beta coefficients.
    - ``beta_sigma`` (float, default 1e6): Prior std for all beta coefficients (diffuse Normal).
    - ``sigma_sigma`` (float, default 10): Scale for HalfNormal prior on sigma.
    - ``nu_lam`` (float, default 1/30): Rate for Exponential prior on
      :math:`\\nu` (only used when ``robust=True``).

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
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
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
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

        # For SLX: S_k = d y / d X_k = beta1_k * I + beta2_k * W
        # Direct   = mean(diag(S_k))
        # Total    = mean(row_sums(S_k))
        # Indirect = Total - Direct
        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        direct = beta1[self._wx_column_indices] + beta2 * mean_diag_w
        total = beta1[self._wx_column_indices] + beta2 * mean_row_sum_w
        indirect = total - direct

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


