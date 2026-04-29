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
        See :class:`~bayespecon.models.base.SpatialModel`. Use ``w_vars`` to restrict which X columns
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

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_lag_test"],
            ).bayesian_lm_lag_test(m),
            "LM-Lag",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_error_test"],
            ).bayesian_lm_error_test(m),
            "LM-Error",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_lag_sdm_test"],
            ).bayesian_robust_lm_lag_sdm_test(m),
            "Robust-LM-Lag-SDM",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_error_sdem_test"],
            ).bayesian_robust_lm_error_sdem_test(m),
            "Robust-LM-Error-SDEM",
        ),
    ]

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SLX regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        if not self._wx_column_indices:
            raise ValueError(
                "SLX requires at least one regressor to spatially lag, but no "
                "WX columns were selected. Pass `w_vars=[...]` to choose which "
                "regressors receive a spatial lag, or fit an OLS model instead."
            )
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

        Notes
        -----
        Unlike SAR / SDM, the SLX model does **not** invert
        :math:`(I - \\rho W)`: the partial-derivative matrix for
        covariate :math:`k` is simply

        .. math::

            S_k \\;=\\; \\beta_{1k}\\, I \\;+\\; \\beta_{2k}\\, W,

        a *linear* (W-direct) impact rather than the full Leontief
        multiplier :math:`(I - \\rho W)^{-1}` that arises in SAR-style
        models.  Consequently SLX impacts are exact functions of the
        posterior :math:`(\\beta_1, \\beta_2)` draws and the row-sum /
        diagonal summaries of :math:`W` — no global feedback loop is
        implied.  This is the key conceptual difference from
        SDM/SAR/SDEM impact reporting (LeSage & Pace 2009, ch. 2;
        Halleck Vega & Elhorst 2015).

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta = self._posterior_mean("beta")
        beta1 = beta[:k]  # coefficients on X
        beta2 = beta[k : k + kw]  # coefficients on WX (excluding intercept-like terms)

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

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SLX model (no :math:`\\rho`), the impact measures for
        covariate :math:`k` are:

        .. math::
            S_k^{(g)} = \\beta_{1j}^{(g)} I + \\beta_{2k}^{(g)} W

            \\text{Direct}_k^{(g)} = \\overline{\\text{diag}}(S_k^{(g)})
            = \\beta_{1j}^{(g)} + \\beta_{2k}^{(g)} \\overline{\\text{diag}}(W)

            \\text{Total}_k^{(g)} = \\overline{\\text{rowsum}}(S_k^{(g)})
            = \\beta_{1j}^{(g)} + \\beta_{2k}^{(g)} \\overline{\\text{rowsum}}(W)

            \\text{Indirect}_k^{(g)} = \\text{Total}_k^{(g)} - \\text{Direct}_k^{(g)}

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each
            of shape ``(G, k_wx)``.
        """
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k+k_wx)
        beta_draws.shape[0]
        k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1_draws = beta_draws[:, :k]  # (G, k)
        beta2_draws = beta_draws[:, k : k + kw]  # (G, kw)

        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        wx_idx = self._wx_column_indices
        direct_samples = np.column_stack(
            [
                beta1_draws[:, j] + beta2_draws[:, idx] * mean_diag_w
                for idx, j in enumerate(wx_idx)
            ]
        )  # (G, kw)
        total_samples = np.column_stack(
            [
                beta1_draws[:, j] + beta2_draws[:, idx] * mean_row_sum_w
                for idx, j in enumerate(wx_idx)
            ]
        )  # (G, kw)
        indirect_samples = total_samples - direct_samples  # (G, kw)

        return direct_samples, indirect_samples, total_samples

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
