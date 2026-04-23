"""Dynamic spatial panel model classes inspired by MATLAB dynamic panel routines."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..logdet import make_logdet_fn
from .panel_base import SpatialPanelModel


class _DynamicPanelMixin:
    """Shared helpers for dynamic panel classes based on lagged stacked arrays."""

    _y_dyn: np.ndarray
    _y_lag: np.ndarray
    _Wy_dyn: np.ndarray
    _Wy_lag: np.ndarray
    _X_dyn: np.ndarray
    _WX_dyn: np.ndarray
    _Z_dyn: np.ndarray
    _n_time_eff: int

    def _prepare_dynamic_design(self) -> None:
        if hasattr(self, "_Z_dyn"):
            return

        if self._T < 2:
            raise ValueError("Dynamic panel models require T >= 2.")

        T, N = self._T, self._N
        k = self._X.shape[1]
        kw = self._WX.shape[1]

        y2 = self._y.reshape(T, N)
        wy2 = self._Wy.reshape(T, N)
        X3 = self._X.reshape(T, N, k)

        self._y_dyn = y2[1:, :].reshape(-1)
        self._y_lag = y2[:-1, :].reshape(-1)
        self._Wy_dyn = wy2[1:, :].reshape(-1)
        self._Wy_lag = wy2[:-1, :].reshape(-1)
        self._X_dyn = X3[1:, :, :].reshape(-1, k)

        if kw > 0:
            WX3 = self._WX.reshape(T, N, kw)
            self._WX_dyn = WX3[1:, :, :].reshape(-1, kw)
            self._Z_dyn = np.hstack([self._X_dyn, self._WX_dyn])
        else:
            self._WX_dyn = np.empty((self._X_dyn.shape[0], 0), dtype=float)
            self._Z_dyn = self._X_dyn

        self._n_time_eff = T - 1

    def _beta_names(self) -> list[str]:
        if self._wx_feature_names:
            return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]
        return self._feature_names


class DLMPanelFE(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic non-spatial panel model with lagged dependent variable.

    Model:
    ``y_t = phi * y_{t-1} + X_t * beta + W*X_t * gamma + e_t``.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = phi * self._y_lag + pt.dot(self._Z_dyn, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        return phi * self._y_lag + self._Z_dyn @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        self._prepare_dynamic_design()
        beta = self._posterior_mean("beta")
        k = self._X_dyn.shape[1]
        kw = self._WX_dyn.shape[1]
        beta1 = beta[:k]

        if kw == 0:
            direct = beta1.copy()
            indirect = np.zeros_like(beta1)
            total = beta1.copy()
            names = self._feature_names
        else:
            beta2 = beta[k:k + kw]
            mean_diag_w = float(self._W_sparse.diagonal().mean())
            mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])
            direct = beta1[self._wx_column_indices] + beta2 * mean_diag_w
            total = beta1[self._wx_column_indices] + beta2 * mean_row_sum_w
            indirect = total - direct
            names = self._wx_feature_names

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": names,
        }


class SDMRPanelFE(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic restricted spatial Durbin model for panels.

    Model:
    ``y_t = phi*y_{t-1} + rho*W*y_t - rho*phi*W*y_{t-1} + X_t*beta + W*X_t*gamma + e_t``.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        rho_lower = self.priors.get("rho_lower", -0.95)
        rho_upper = self.priors.get("rho_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=rho_lower,
            rho_max=rho_upper,
            T=self._n_time_eff,
        )

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = (
                phi * self._y_lag
                + rho * self._Wy_dyn
                - rho * phi * self._Wy_lag
                + pt.dot(self._Z_dyn, beta)
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)
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

        The SDMR panel model uses ``pm.Normal("obs", observed=y)`` which
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
            self._attach_jacobian_corrected_log_likelihood(
                idata, "rho", T=self._n_time_eff
            )
        return idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        rho = float(self._posterior_mean("rho"))
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        return (
            phi * self._y_lag
            + rho * self._Wy_dyn
            - rho * phi * self._Wy_lag
            + self._Z_dyn @ beta
        )

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        self._prepare_dynamic_design()
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        k = self._X_dyn.shape[1]
        kw = self._WX_dyn.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]

        if kw == 0:
            direct = beta1.copy()
            indirect = np.zeros_like(beta1)
            total = beta1.copy()
            names = self._feature_names
        else:
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
            names = self._wx_feature_names

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": names,
        }


class SDMUPanelFE(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic unrestricted spatial Durbin model for panels.

    Model:
    ``y_t = phi*y_{t-1} + rho*W*y_t + theta*W*y_{t-1} + X_t*beta + W*X_t*gamma + e_t``.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        rho_lower = self.priors.get("rho_lower", -0.95)
        rho_upper = self.priors.get("rho_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        theta_lower = self.priors.get("theta_lower", -0.95)
        theta_upper = self.priors.get("theta_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=rho_lower,
            rho_max=rho_upper,
            T=self._n_time_eff,
        )

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            theta = pm.Uniform("theta", lower=theta_lower, upper=theta_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = (
                phi * self._y_lag
                + rho * self._Wy_dyn
                + theta * self._Wy_lag
                + pt.dot(self._Z_dyn, beta)
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)
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

        The SDMU panel model uses ``pm.Normal("obs", observed=y)`` which
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
            self._attach_jacobian_corrected_log_likelihood(
                idata, "rho", T=self._n_time_eff
            )
        return idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        rho = float(self._posterior_mean("rho"))
        phi = float(self._posterior_mean("phi"))
        theta = float(self._posterior_mean("theta"))
        beta = self._posterior_mean("beta")
        return (
            phi * self._y_lag
            + rho * self._Wy_dyn
            + theta * self._Wy_lag
            + self._Z_dyn @ beta
        )

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        # Effects reported for contemporaneous X/WX terms.
        self._prepare_dynamic_design()
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        k = self._X_dyn.shape[1]
        kw = self._WX_dyn.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]

        if kw == 0:
            direct = beta1.copy()
            indirect = np.zeros_like(beta1)
            total = beta1.copy()
            names = self._feature_names
        else:
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
            names = self._wx_feature_names

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": names,
        }
