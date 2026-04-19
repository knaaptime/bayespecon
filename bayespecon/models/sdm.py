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

        logdet_fn = make_logdet_fn(self._W_eigs.real, method=self.logdet_method,
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

        eigs = self._W_eigs
        inv_eigs = 1.0 / (1.0 - rho * eigs)
        mean_diag_M = float(np.mean(inv_eigs.real))
        mean_diag_MW = float(np.mean((eigs * inv_eigs).real))
        if self._is_row_std:
            mean_row_sum_M = 1.0 / (1.0 - rho)
            mean_row_sum_MW = mean_row_sum_M
        else:
            ones = np.ones(self._W_sparse.shape[0])
            A = np.eye(self._W_sparse.shape[0]) - rho * self._W_sparse.toarray()
            M_ones = np.linalg.solve(A, ones)
            mean_row_sum_M = float(M_ones.mean())
            mean_row_sum_MW = float((self._W_sparse.toarray() @ M_ones).mean())
        direct = np.array([
            beta1[j] * mean_diag_M + b2 * mean_diag_MW
            for j, b2 in zip(self._wx_column_indices, beta2)
        ])
        total = np.array([
            beta1[j] * mean_row_sum_M + b2 * mean_row_sum_MW
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

    # ------------------------------------------------------------------
    # Spatial specification tests
    # ------------------------------------------------------------------

    def lm_error_test(self) -> "DiagnosticResult":
        """LM test for omitted spatial error autocorrelation.

        Tests whether SDM residuals show additional spatial error structure,
        suggesting a SARAR model may be more appropriate.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_error"``

            ``statistic`` : float — LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        Notes
        -----
        H\\ :sub:`0`: no spatial error autocorrelation in OLS residuals.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        from ..stats.core import lmerror
        raw = lmerror(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_error", raw, "lm")

    def lm_rho_test(self) -> "DiagnosticResult":
        """LM test for the SAR spatial autoregressive parameter :math:`\\rho`.

        Tests H\\ :sub:`0`: :math:`\\rho = 0` from OLS residuals.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_rho"``

            ``statistic`` : float — LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        from ..stats.core import lmrho
        raw = lmrho(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_rho", raw, "lmrho")

    def lm_rho_robust_test(self) -> "DiagnosticResult":
        """Robust LM test for the SAR parameter :math:`\\rho`.

        Robust to the presence of spatial error autocorrelation under
        the alternative.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_rho_robust"``

            ``statistic`` : float — robust LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        References
        ----------
        .. [1] Anselin et al. (1996). *Regional Science and Urban Economics*,
               26(1), 77–104.
        """
        from ..stats.core import lmrhorob
        raw = lmrhorob(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_rho_robust", raw, "lmrhorob")

    def spatial_specification_tests(self) -> dict:
        """Run a battery of spatial specification tests on OLS residuals.

        Combines Moran's I, LM-error, LM-:math:`\\rho`, and its robust
        version.

        Returns
        -------
        dict[str, DiagnosticResult]
            Keys: ``"moran"``, ``"lm_error"``, ``"lm_rho"``,
            ``"lm_rho_robust"``.

        See Also
        --------
        moran_test : Moran's I for residual spatial autocorrelation.
        lm_error_test : LM test for spatial error dependence.
        lm_rho_test : LM test for the SAR parameter.
        lm_rho_robust_test : Robust LM test for the SAR parameter.
        """
        return {
            "moran": self.moran_test(),
            "lm_error": self.lm_error_test(),
            "lm_rho": self.lm_rho_test(),
            "lm_rho_robust": self.lm_rho_robust_test(),
        }
