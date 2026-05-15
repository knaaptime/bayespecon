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
from .priors import SLXPriors


class SLX(SpatialModel):
    """Bayesian SLX (Spatial Lag X) model.

    Adds spatial lags of the regressors :math:`X` to a standard linear
    model, without any spatial dependence in :math:`y`:

    .. math::
        y = X\\beta + WX\\theta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The sampled coefficient vector stacks the local and lagged-regressor
    blocks as :math:`[\\beta, \\theta]`. There is no Jacobian term
    because there is no spatial lag of :math:`y`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. Intercept is included by default; suppress with
        ``"y ~ x - 1"``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Dependent variable of shape ``(n,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode. DataFrame columns are
        preserved as feature names.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``. Used to construct the
        ``WX`` block. Accepts a :class:`libpysal.graph.Graph` or any
        :class:`scipy.sparse` matrix; legacy ``libpysal.weights.W`` is
        not accepted (use ``w.sparse`` or
        ``libpysal.graph.Graph.from_W(w)``). Should be row-standardised.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` (float, default 0.0): Normal prior mean for all
          coefficients :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for all
          coefficients (diffuse).
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\\sigma`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        Accepted for API consistency with the spatial-lag/error models
        but unused (SLX has no spatial Jacobian).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See *Robust
        regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``. SLX requires at least one
        WX column; if filtering eliminates all of them a ValueError is
        raised.

    Notes
    -----
    Direct effects equal :math:`\\beta` and indirect effects equal
    :math:`\\theta`. There is no global spillover multiplier because
    :math:`y` has no spatial autoregression.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)`
    with rate ``nu_lam`` (default 1/30, mean ≈ 30).
    """

    _priors_cls = SLXPriors

    _spatial_diagnostics_tests = [
        (
            SpatialModel._lazy_lm_test(
                "bayespecon.diagnostics.lmtests", "bayesian_lm_lag_test"
            ),
            "LM-Lag",
        ),
        (
            SpatialModel._lazy_lm_test(
                "bayespecon.diagnostics.lmtests", "bayesian_lm_error_test"
            ),
            "LM-Error",
        ),
        (
            SpatialModel._lazy_lm_test(
                "bayespecon.diagnostics.lmtests", "bayesian_robust_lm_lag_sdm_test"
            ),
            "Robust-LM-Lag-SDM",
        ),
        (
            SpatialModel._lazy_lm_test(
                "bayespecon.diagnostics.lmtests", "bayesian_robust_lm_error_sdem_test"
            ),
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
        posterior :math:`(\\beta, \\theta)` draws and the row-sum /
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
        from ..diagnostics.lmtests import _get_posterior_draws

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
        direct_samples = beta1_draws[:, wx_idx] + mean_diag_w * beta2_draws  # (G, kw)
        total_samples = beta1_draws[:, wx_idx] + mean_row_sum_w * beta2_draws  # (G, kw)
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
