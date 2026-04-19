"""Spatial panel model classes analogous to MATLAB panel_g routines."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from ..logdet import make_logdet_fn
from .panel_base import SpatialPanelModel
from ..diagnostics import DiagnosticResult, hausman_fe_re_test


class OLSPanelFE(SpatialPanelModel):
    """Bayesian pooled/fixed-effects panel regression.

    Parameters
    ----------
    formula, data, y, X, W, unit_col, time_col, N, T, model, priors,
    logdet_method
        See :class:`~bayespecon.models.panel_base.SpatialPanelModel`.

    Notes
    -----
    This class mirrors MATLAB ``ols_panel_FE_g`` behavior through the shared
    :class:`~bayespecon.models.panel_base.SpatialPanelModel` transformation
    pipeline.
    """

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for pooled/FE panel regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = pt.dot(self._X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)
        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute direct/indirect/total effects for OLS panel model.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }

    # ------------------------------------------------------------------
    # Specification tests
    # ------------------------------------------------------------------

    def hausman_test(
        self,
        re_model,
        variance_tol: float = 1e-12,
    ) -> DiagnosticResult:
        """Hausman FE-vs-RE specification test against an ``OLSPanelRE`` fit.

        Parameters
        ----------
        re_model : OLSPanelRE
            Fitted random-effects panel model using the same outcome and
            regressors.
        variance_tol : float, default=1e-12
            Threshold used to drop FE coefficients with effectively zero
            transformed design variance (e.g., demeaned intercepts).

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"hausman_fe_re"``

            ``statistic`` : float — Hausman chi-square statistic.

            ``pvalue`` : float — chi-square p-value.

        Notes
        -----
        Coefficients are aligned by shared feature names. Any coefficient that
        is unidentified after FE transformation (near-zero variance column) is
        excluded automatically.
        """
        from .panel_re import OLSPanelRE

        self._require_fit()
        if not isinstance(re_model, OLSPanelRE):
            raise TypeError("re_model must be an instance of OLSPanelRE")
        re_model._require_fit()

        fe_names = list(self._beta_names())
        re_names = list(re_model._beta_names())
        common_names = [name for name in fe_names if name in re_names]
        if not common_names:
            raise ValueError("No shared coefficients found between FE and RE models")

        filtered_names: list[str] = []
        fe_idx: list[int] = []
        re_idx: list[int] = []
        for name in common_names:
            i_fe = fe_names.index(name)
            if float(np.var(self._X[:, i_fe])) <= variance_tol:
                continue
            filtered_names.append(name)
            fe_idx.append(i_fe)
            re_idx.append(re_names.index(name))

        if not filtered_names:
            raise ValueError(
                "No identifiable shared coefficients remain after FE variance filtering"
            )

        fe_draws = np.asarray(self._idata.posterior["beta"]).reshape(-1, len(fe_names))[:, fe_idx]
        re_draws = np.asarray(re_model._idata.posterior["beta"]).reshape(-1, len(re_names))[:, re_idx]

        beta_fe = fe_draws.mean(axis=0)
        beta_re = re_draws.mean(axis=0)
        cov_fe = np.atleast_2d(np.cov(fe_draws, rowvar=False))
        cov_re = np.atleast_2d(np.cov(re_draws, rowvar=False))

        return hausman_fe_re_test(
            beta_fe=beta_fe,
            cov_fe=cov_fe,
            beta_re=beta_re,
            cov_re=cov_re,
            coef_names=filtered_names,
        )

    def lm_error_test(self) -> DiagnosticResult:
        """LM test for spatial error dependence in a fixed-effects panel.

        Tests whether the Lee-Yu FE-transformed residuals show spatial
        error autocorrelation, suggesting a SEM panel model may be more
        appropriate than OLS panel.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_f_error"``

            ``statistic`` : float — LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        Notes
        -----
        H\\ :sub:`0`: no spatial error dependence in FE panel residuals.
        The Lee-Yu within-transformation is applied internally.

        References
        ----------
        .. [1] Lee, L., & Yu, J. (2010). Estimation of spatial autoregressive
               panel data models with fixed effects. *Journal of Econometrics*,
               154(2), 165–185.
        """
        from ..stats.panel import lm_f_err
        W_cs = self._W_sparse[:self._N, :self._N].toarray()
        raw = lm_f_err(self._y_raw, self._X_raw, W_cs, self._N)
        return self._wrap_stats_result("lm_f_error", raw, "lm")

    def lm_sar_test(self) -> DiagnosticResult:
        """LM test for SAR dependence in a fixed-effects panel.

        Tests whether a spatially lagged dependent variable :math:`\\rho Wy`
        is missing from the OLS panel specification, suggesting SAR panel.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_f_sar"``

            ``statistic`` : float — LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        Notes
        -----
        H\\ :sub:`0`: no omitted spatial lag in FE panel.

        References
        ----------
        .. [1] Lee, L., & Yu, J. (2010). *Journal of Econometrics*, 154(2),
               165–185.
        """
        from ..stats.panel import lm_f_sar
        W_cs = self._W_sparse[:self._N, :self._N].toarray()
        raw = lm_f_sar(self._y_raw, self._X_raw, W_cs, self._N)
        return self._wrap_stats_result("lm_f_sar", raw, "lm")

    def lm_joint_test(self) -> DiagnosticResult:
        """Joint LM test for both SAR and SEM effects in an FE panel.

        Tests whether either (or both) of a spatial lag on *y* and a
        spatial error process are present in the OLS panel specification.
        The statistic is asymptotically :math:`\\chi^2(2)` under H\\ :sub:`0`.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_f_joint"``

            ``statistic`` : float — joint LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(2)` null.

        Notes
        -----
        H\\ :sub:`0`: no SAR and no SEM dependence in the FE panel.

        References
        ----------
        .. [1] Lee, L., & Yu, J. (2010). *Journal of Econometrics*, 154(2),
               165–185.
        """
        from ..stats.panel import lm_f_joint
        W_cs = self._W_sparse[:self._N, :self._N].toarray()
        raw = lm_f_joint(self._y_raw, self._X_raw, W_cs, W_cs, self._N)
        return self._wrap_stats_result("lm_f_joint", raw, "lm")

    def spatial_specification_tests(self) -> dict:
        """Run panel spatial specification tests on Lee-Yu FE residuals.

        Combines the LM-error, LM-SAR, and joint LM tests.  Use the results
        to decide whether to adopt a SAR panel, SEM panel, or remain with
        OLS panel.

        Returns
        -------
        dict[str, DiagnosticResult]
            Keys: ``"lm_error"``, ``"lm_sar"``, ``"lm_joint"``.

        See Also
        --------
        lm_error_test : LM test for spatial error in the panel.
        lm_sar_test : LM test for spatial lag in the panel.
        lm_joint_test : Joint LM test for both SAR and SEM effects.
        """
        return {
            "lm_error": self.lm_error_test(),
            "lm_sar": self.lm_sar_test(),
            "lm_joint": self.lm_joint_test(),
        }


class SARPanelFE(SpatialPanelModel):
    """Bayesian spatial lag panel regression.

    Parameters
    ----------
    formula, data, y, X, W, unit_col, time_col, N, T, model, priors,
    logdet_method
        See :class:`~bayespecon.models.panel_base.SpatialPanelModel`.

    Notes
    -----
    This class mirrors MATLAB ``sar_panel_FE_g`` behavior under the common
    panel transformation framework.
    """

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SAR panel regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=rho_lower,
            rho_max=rho_upper,
            T=self._T,
        )

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = rho * self._Wy + pt.dot(self._X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)
            pm.Potential("jacobian", logdet_fn(rho))
        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        return rho * self._Wy + self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SAR panel direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        if self._is_row_std:
            mean_row_sum = 1.0 / (1.0 - rho)
        else:
            mean_row_sum = float(
                np.linalg.solve(
                    np.eye(self._W_sparse.shape[0]) - rho * self._W_sparse.toarray(),
                    np.ones(self._W_sparse.shape[0]),
                ).mean()
            )
        direct = mean_diag * beta
        total = mean_row_sum * beta
        indirect = total - direct
        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._feature_names,
        }

    # ------------------------------------------------------------------
    # Spatial specification tests
    # ------------------------------------------------------------------

    def lm_error_conditional_test(self) -> DiagnosticResult:
        """LM test for spatial error dependence, conditional on SAR.

        Returns
        -------
        DiagnosticResult
            Test result with LM statistic and p-value.
        """
        from ..stats.panel import lm_f_err_c

        W_cs = self._W_sparse[:self._N, :self._N].toarray()
        raw = lm_f_err_c(self._y_raw, self._X_raw, W_cs, W_cs, self._N)
        return self._wrap_stats_result("lm_f_error_c", raw, "lm")

    def lr_sar_test(self) -> DiagnosticResult:
        """LR test for the spatial lag parameter :math:`\\rho` in the FE panel.

        Returns
        -------
        DiagnosticResult
            Test result with LR statistic and p-value.
        """
        from ..stats.panel import lr_f_sar

        W_cs = self._W_sparse[:self._N, :self._N].toarray()
        raw = lr_f_sar(self._y_raw, self._X_raw, W_cs, self._N)
        return self._wrap_stats_result("lr_f_sar", raw, "lr")

    def spatial_specification_tests(self) -> dict:
        """Run panel spatial specification tests for the SAR panel model.

        Returns
        -------
        dict[str, DiagnosticResult]
            Keys: ``"lm_error_conditional"``, ``"lr_sar"``.
        """
        return {
            "lm_error_conditional": self.lm_error_conditional_test(),
            "lr_sar": self.lr_sar_test(),
        }


class SEMPanelFE(SpatialPanelModel):
    """Bayesian spatial error panel regression.

    Parameters
    ----------
    formula, data, y, X, W, unit_col, time_col, N, T, model, priors,
    logdet_method
        See :class:`~bayespecon.models.panel_base.SpatialPanelModel`.

    Notes
    -----
    This class mirrors MATLAB ``sem_panel_FE_g`` behavior under the common
    panel transformation framework.
    """

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SEM panel regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=lam_lower,
            rho_max=lam_upper,
            T=self._T,
        )

        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            resid = self._y - pt.dot(self._X, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
            pm.Potential("eps_loglik", logp_eps)
            pm.Potential("jacobian", logdet_fn(lam))
        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach pointwise log-likelihood for IC metrics."""
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            **sample_kwargs,
        )

        if "log_likelihood" in idata.groups() and "obs" in idata.log_likelihood:
            return idata

        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = lam.shape
        s = c * d
        n = self._y.shape[0]
        X = self._X
        W = self._W_dense

        lam_f = lam.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        resid = self._y[None, :] - beta_f @ X.T
        eps = resid - lam_f[:, None] * (resid @ W.T)

        ll = -0.5 * (
            (eps / sigma_f[:, None]) ** 2
            + np.log(2.0 * np.pi)
            + 2.0 * np.log(sigma_f[:, None])
        )

        eye_n = np.eye(n)
        jac = np.empty(s, dtype=float)
        for i in range(s):
            sign, logabsdet = np.linalg.slogdet(eye_n - lam_f[i] * W)
            if sign == 0:
                jac[i] = -np.inf
            else:
                jac[i] = logabsdet
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"))

        if "log_likelihood" in idata.groups():
            idata.log_likelihood["obs"] = ll_da
        else:
            idata.add_groups({"log_likelihood": xr.Dataset({"obs": ll_da})})
        return idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SEM panel direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }

    # ------------------------------------------------------------------
    # Spatial specification tests
    # ------------------------------------------------------------------

    def lm_sar_conditional_test(self) -> DiagnosticResult:
        """LM test for SAR dependence, conditional on SEM.

        Returns
        -------
        DiagnosticResult
            Test result with LM statistic and p-value.
        """
        from ..stats.panel import lm_f_sar_c

        W_cs = self._W_sparse[:self._N, :self._N].toarray()
        raw = lm_f_sar_c(self._y_raw, self._X_raw, W_cs, W_cs, self._N)
        return self._wrap_stats_result("lm_f_sar_c", raw, "lm")

    def lr_error_test(self) -> DiagnosticResult:
        """LR test for the spatial error parameter :math:`\\lambda` in the FE panel.

        Returns
        -------
        DiagnosticResult
            Test result with LR statistic and p-value.
        """
        from ..stats.panel import lr_f_err

        W_cs = self._W_sparse[:self._N, :self._N].toarray()
        raw = lr_f_err(self._y_raw, self._X_raw, W_cs, self._N)
        return self._wrap_stats_result("lr_f_error", raw, "lr")

    def spatial_specification_tests(self) -> dict:
        """Run panel spatial specification tests for the SEM panel model.

        Returns
        -------
        dict[str, DiagnosticResult]
            Keys: ``"lm_sar_conditional"``, ``"lr_error"``.
        """
        return {
            "lm_sar_conditional": self.lm_sar_conditional_test(),
            "lr_error": self.lr_error_test(),
        }


class SDMPanelFE(SpatialPanelModel):
    """Bayesian spatial Durbin panel regression.

    Parameters
    ----------
    formula, data, y, X, W, unit_col, time_col, N, T, model, priors,
    logdet_method
        See :class:`~bayespecon.models.panel_base.SpatialPanelModel`.

    Notes
    -----
    This class mirrors MATLAB ``sdm_panel_FE_g`` behavior by augmenting the
    design matrix with spatially lagged regressors ``W X``.
    """

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SDM panel regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        Z = np.hstack([self._X, self._WX])

        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=rho_lower,
            rho_max=rho_upper,
            T=self._T,
        )

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = rho * self._Wy + pt.dot(Z, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)
            pm.Potential("jacobian", logdet_fn(rho))
        return model

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

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SDM panel direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
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


class SDEMPanelFE(SpatialPanelModel):
    """Bayesian spatial Durbin error panel regression.

    Parameters
    ----------
    formula, data, y, X, W, unit_col, time_col, N, T, model, priors,
    logdet_method
        See :class:`~bayespecon.models.panel_base.SpatialPanelModel`.

    Notes
    -----
    This class mirrors MATLAB ``sdem_panel_FE_g`` behavior using transformed
    residual likelihood and Jacobian adjustment.
    """

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SDEM panel regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        Z = np.hstack([self._X, self._WX])

        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=lam_lower,
            rho_max=lam_upper,
            T=self._T,
        )

        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            resid = self._y - pt.dot(Z, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
            pm.Potential("eps_loglik", logp_eps)
            pm.Potential("jacobian", logdet_fn(lam))
        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach pointwise log-likelihood for IC metrics."""
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            **sample_kwargs,
        )

        if "log_likelihood" in idata.groups() and "obs" in idata.log_likelihood:
            return idata

        Z = np.hstack([self._X, self._WX])
        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = lam.shape
        s = c * d
        n = self._y.shape[0]
        W = self._W_dense

        lam_f = lam.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        resid = self._y[None, :] - beta_f @ Z.T
        eps = resid - lam_f[:, None] * (resid @ W.T)

        ll = -0.5 * (
            (eps / sigma_f[:, None]) ** 2
            + np.log(2.0 * np.pi)
            + 2.0 * np.log(sigma_f[:, None])
        )

        eye_n = np.eye(n)
        jac = np.empty(s, dtype=float)
        for i in range(s):
            sign, logabsdet = np.linalg.slogdet(eye_n - lam_f[i] * W)
            if sign == 0:
                jac[i] = -np.inf
            else:
                jac[i] = logabsdet
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"))

        if "log_likelihood" in idata.groups():
            idata.log_likelihood["obs"] = ll_da
        else:
            idata.add_groups({"log_likelihood": xr.Dataset({"obs": ll_da})})
        return idata

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

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SDEM panel direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        beta = self._posterior_mean("beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]
        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])
        direct = beta1[self._wx_column_indices] + beta2 * mean_diag_w
        total = beta1[self._wx_column_indices] + beta2 * mean_row_sum_w
        return {
            "direct": direct,
            "indirect": total - direct,
            "total": total,
            "feature_names": self._wx_feature_names,
        }
