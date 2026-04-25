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
    _W_dense_dyn_cache: np.ndarray | None

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
        self._W_dense_dyn_cache = None

    @property
    def _W_dense_dyn(self) -> np.ndarray:
        """Dense (N*(T-1)) x (N*(T-1)) block-diagonal W for dynamic period."""
        if self._W_dense_dyn_cache is None:
            from scipy.sparse import block_diag
            Wn = self._W_sparse.toarray() if hasattr(self._W_sparse, "toarray") else np.asarray(self._W_sparse)
            self._W_dense_dyn_cache = np.kron(np.eye(self._n_time_eff), Wn)
        return self._W_dense_dyn_cache

    def _beta_names(self) -> list[str]:
        if self._wx_feature_names:
            return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]
        return self._feature_names


class DLMPanelFE(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic non-spatial panel model with lagged dependent variable.

    Model:
    ``y_t = phi * y_{t-1} + X_t * beta + W*X_t * gamma + e_t``.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
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
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
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

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
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
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
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

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
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
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
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


class SARPanelDEDynamic(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic spatial lag panel model (SAR with lagged dependent variable).

    Model:
    ``y_t = phi*y_{t-1} + rho*W*y_t + X_t*beta + e_t``.

    This is the dynamic analogue of :class:`SARPanelFE`, adding a
    time-lagged dependent variable but no WX terms (no Durbin component).
    The Jacobian ``|I - rho*W|^(T-1)`` accounts for the contemporaneous
    spatial lag.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
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

            mu = phi * self._y_lag + rho * self._Wy_dyn + pt.dot(self._X_dyn, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
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
        """Sample posterior and attach Jacobian-corrected log-likelihood."""
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
        return phi * self._y_lag + rho * self._Wy_dyn + self._X_dyn @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        self._prepare_dynamic_design()
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


class SEMPanelDEDynamic(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic spatial error panel model (SEM with lagged dependent variable).

    Model:
    ``y_t = phi*y_{t-1} + X_t*beta + u_t,  u_t = lambda*W*u_t + e_t``.

    This is the dynamic analogue of :class:`SEMPanelFE`. The likelihood
    uses transformed residuals ``(I - lambda*W)(y_t - phi*y_{t-1} - X_t*beta)``
    with a Jacobian ``|I - lambda*W|^(T-1)``.

    **Robust regression**

    When ``robust=True``, the spatially-filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)(y_t - \\phi y_{t-1} - X_t \\beta) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        lam_lower = self.priors.get("lam_lower", -0.95)
        lam_upper = self.priors.get("lam_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=lam_lower,
            rho_max=lam_upper,
            T=self._n_time_eff,
        )

        W_pt = pt.as_tensor_variable(self._W_dense_dyn)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            resid = self._y_dyn - phi * self._y_lag - pt.dot(self._X_dyn, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_eps = pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps).sum()
            else:
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
        """Sample posterior and attach pointwise log-likelihood for IC metrics."""
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

        import xarray as xr

        X = self._X_dyn
        lam = idata.posterior["lam"].values
        phi = idata.posterior["phi"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = lam.shape
        s = c * d
        n = self._y_dyn.shape[0]
        W = self._W_dense_dyn

        lam_f = lam.reshape(s)
        phi_f = phi.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        resid = self._y_dyn[None, :] - phi_f[:, None] * self._y_lag[None, :] - beta_f @ X.T
        eps = resid - lam_f[:, None] * (resid @ W.T)

        if self.robust:
            nu_f = idata.posterior["nu"].values.reshape(s)
            from scipy.special import gammaln
            ll = (
                gammaln((nu_f[:, None] + 1) / 2)
                - gammaln(nu_f[:, None] / 2)
                - 0.5 * np.log(nu_f[:, None] * np.pi)
                - np.log(sigma_f[:, None])
                - ((nu_f[:, None] + 1) / 2)
                * np.log1p((eps / sigma_f[:, None]) ** 2 / nu_f[:, None])
            )
        else:
            ll = -0.5 * (
                (eps / sigma_f[:, None]) ** 2
                + np.log(2.0 * np.pi)
                + 2.0 * np.log(sigma_f[:, None])
            )

        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_f]) * self._n_time_eff
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        return phi * self._y_lag + self._X_dyn @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        self._prepare_dynamic_design()
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }


class SDEMPanelDEDynamic(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic spatial Durbin error panel model (SDEM with lagged dependent variable).

    Model:
    ``y_t = phi*y_{t-1} + X_t*beta + W*X_t*theta + u_t,  u_t = lambda*W*u_t + e_t``.

    This is the dynamic analogue of :class:`SDEMPanelFE`. The likelihood
    uses transformed residuals with a Jacobian ``|I - lambda*W|^(T-1)``.

    **Robust regression**

    When ``robust=True``, the spatially-filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)(y_t - \\phi y_{t-1} - X_t \\beta_1 - WX_t \\beta_2) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    def _beta_names(self) -> list[str]:
        if self._wx_feature_names:
            return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]
        return self._feature_names

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        Z = np.hstack([self._X_dyn, self._WX_dyn])

        lam_lower = self.priors.get("lam_lower", -0.95)
        lam_upper = self.priors.get("lam_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=lam_lower,
            rho_max=lam_upper,
            T=self._n_time_eff,
        )

        W_pt = pt.as_tensor_variable(self._W_dense_dyn)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            resid = self._y_dyn - phi * self._y_lag - pt.dot(Z, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_eps = pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps).sum()
            else:
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
        """Sample posterior and attach pointwise log-likelihood for IC metrics."""
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

        import xarray as xr

        Z = np.hstack([self._X_dyn, self._WX_dyn])
        lam = idata.posterior["lam"].values
        phi = idata.posterior["phi"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = lam.shape
        s = c * d
        n = self._y_dyn.shape[0]
        W = self._W_dense_dyn

        lam_f = lam.reshape(s)
        phi_f = phi.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        resid = self._y_dyn[None, :] - phi_f[:, None] * self._y_lag[None, :] - beta_f @ Z.T
        eps = resid - lam_f[:, None] * (resid @ W.T)

        if self.robust:
            nu_f = idata.posterior["nu"].values.reshape(s)
            from scipy.special import gammaln
            ll = (
                gammaln((nu_f[:, None] + 1) / 2)
                - gammaln(nu_f[:, None] / 2)
                - 0.5 * np.log(nu_f[:, None] * np.pi)
                - np.log(sigma_f[:, None])
                - ((nu_f[:, None] + 1) / 2)
                * np.log1p((eps / sigma_f[:, None]) ** 2 / nu_f[:, None])
            )
        else:
            ll = -0.5 * (
                (eps / sigma_f[:, None]) ** 2
                + np.log(2.0 * np.pi)
                + 2.0 * np.log(sigma_f[:, None])
            )

        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_f]) * self._n_time_eff
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X_dyn, self._WX_dyn])
        return phi * self._y_lag + Z @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        self._prepare_dynamic_design()
        beta = self._posterior_mean("beta")
        k = self._X_dyn.shape[1]
        kw = self._WX_dyn.shape[1]
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


class SLXPanelDEDynamic(_DynamicPanelMixin, SpatialPanelModel):
    """Dynamic SLX panel model (SLX with lagged dependent variable).

    Model:
    ``y_t = phi*y_{t-1} + X_t*beta_1 + W*X_t*beta_2 + e_t``.

    This is the dynamic analogue of :class:`SLXPanelFE`. No spatial lag
    on y, so no Jacobian adjustment is needed.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    def _beta_names(self) -> list[str]:
        if self._wx_feature_names:
            return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]
        return self._feature_names

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        Z = np.hstack([self._X_dyn, self._WX_dyn])

        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = phi * self._y_lag + pt.dot(Z, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X_dyn, self._WX_dyn])
        return phi * self._y_lag + Z @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        self._prepare_dynamic_design()
        beta = self._posterior_mean("beta")
        k = self._X_dyn.shape[1]
        kw = self._WX_dyn.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]

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
