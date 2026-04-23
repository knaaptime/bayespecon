"""Spatial panel model classes analogous to MATLAB panel_g routines."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from ..logdet import make_logdet_fn
from .panel_base import SpatialPanelModel


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

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach Jacobian-corrected log-likelihood.

        The SAR panel model uses ``pm.Normal("obs", observed=y)`` which
        auto-captures the Gaussian log-likelihood, plus a ``pm.Potential``
        Jacobian term that is not captured.  When ``log_likelihood=True``
        is requested, the Jacobian correction is added post-sampling.
        """
        idata_kwargs = idata_kwargs or {}
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )
        if idata_kwargs.get("log_likelihood", False):
            self._attach_jacobian_corrected_log_likelihood(idata, "rho", T=self._T)
        return idata

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
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach pointwise log-likelihood for IC metrics.

        The SEM panel model uses ``pm.Potential`` for both the Gaussian
        error log-likelihood and the Jacobian, so neither is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, using eigenvalue-based Jacobian for efficiency.
        """
        idata_kwargs = idata_kwargs or {}
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

        if "log_likelihood" in idata.groups() and "obs" in idata.log_likelihood:
            return idata

        X = self._X
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

        resid = self._y[None, :] - beta_f @ X.T
        eps = resid - lam_f[:, None] * (resid @ W.T)

        ll = -0.5 * (
            (eps / sigma_f[:, None]) ** 2
            + np.log(2.0 * np.pi)
            + 2.0 * np.log(sigma_f[:, None])
        )

        # Eigenvalue-based Jacobian: log|I - lam*W| * T / n
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_f]) * self._T  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
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

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach Jacobian-corrected log-likelihood.

        The SDM panel model uses ``pm.Normal("obs", observed=y)`` which
        auto-captures the Gaussian log-likelihood, plus a ``pm.Potential``
        Jacobian term that is not captured.  When ``log_likelihood=True``
        is requested, the Jacobian correction is added post-sampling.
        """
        idata_kwargs = idata_kwargs or {}
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )
        if idata_kwargs.get("log_likelihood", False):
            self._attach_jacobian_corrected_log_likelihood(idata, "rho", T=self._T)
        return idata

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
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach pointwise log-likelihood for IC metrics.

        The SDEM panel model uses ``pm.Potential`` for both the Gaussian
        error log-likelihood and the Jacobian, so neither is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, using eigenvalue-based Jacobian for efficiency.
        """
        idata_kwargs = idata_kwargs or {}
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
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

        # Eigenvalue-based Jacobian: log|I - lam*W| * T / n
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_f]) * self._T  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
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
