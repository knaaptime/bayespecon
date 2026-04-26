"""Spatial Tobit model classes.

Implements left-censored (default at 0) Bayesian spatial Tobit variants:

- ``SARTobit``: spatial autoregressive Tobit
- ``SEMTobit``: spatial error Tobit
- ``SDMTobit``: spatial Durbin Tobit

All classes use latent-data augmentation for censored observations.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..logdet import make_logdet_fn
from .base import SpatialModel


class _SpatialTobitBase(SpatialModel):
    """Shared helpers for spatial Tobit models."""

    def __init__(self, *args, censoring: float = 0.0, **kwargs):
        self.censoring = float(censoring)
        super().__init__(*args, **kwargs)
        self._censored_mask = self._y <= self.censoring + 1e-12
        self._censored_idx = np.where(self._censored_mask)[0]

    def _latent_y_tensor(self) -> pt.TensorVariable:
        """Create latent y* tensor where censored values are sampled."""
        y_lat = pt.as_tensor_variable(self._y.astype(np.float64))
        n_cens = int(self._censored_idx.size)
        if n_cens > 0:
            censor_sigma = float(self.priors.get("censor_sigma", 10.0))
            y_cens_gap = pm.HalfNormal("y_cens_gap", sigma=censor_sigma, shape=n_cens)
            y_cens = self.censoring - y_cens_gap
            y_lat = pt.set_subtensor(y_lat[self._censored_idx], y_cens)
        return y_lat

    def _posterior_latent_y_mean(self) -> np.ndarray:
        """Posterior mean of latent y* on the observed index set."""
        y_lat = self._y.copy().astype(float)
        if self._censored_idx.size > 0 and "y_cens_gap" in self._idata.posterior:
            gap_hat = self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            y_lat[self._censored_idx] = self.censoring - np.asarray(gap_hat, dtype=float)
        return y_lat


class SARTobit(_SpatialTobitBase):
    """Bayesian spatial autoregressive Tobit model.

    .. math::
        y^* = \\rho W y^* + X\\beta + \\varepsilon,\\quad \\varepsilon \\sim N(0,\\sigma^2 I),

    with observed outcome

    .. math::
        y = \\max(c, y^*)

    where ``c`` is the left-censoring point (default: ``0``).

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t.  For uncensored observations the density becomes:

    .. math::

        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =
        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)

    and for censored observations the probability becomes:

    .. math::

        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)

    where :math:`T_\\nu` is the Student-t CDF with :math:`\\nu` degrees of
    freedom, and :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30.
    """
    _spatial_diagnostics_tests = [
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_error_test"],
        ).bayesian_lm_error_test(m), "LM-Error"),
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_wx_test"],
        ).bayesian_lm_wx_test(m), "LM-WX"),
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_robust_lm_wx_test"],
        ).bayesian_robust_lm_wx_test(m), "Robust-LM-WX"),
    ]
    def _build_pymc_model(self) -> pm.Model:
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
        )
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_lat = self._latent_y_tensor()
            resid = y_lat - rho * pt.dot(W_pt, y_lat) - pt.dot(self._X, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_resid = pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), resid).sum()
            else:
                logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
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
        ni = self._nonintercept_indices
        direct = mean_diag * beta[ni]
        total = mean_row_sum * beta[ni]
        indirect = total - direct
        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._nonintercept_feature_names,
        }

    def _compute_spatial_effects_posterior(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws
        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            if self._is_row_std:
                mean_row_sum = 1.0 / (1.0 - rho_draws)
            else:
                n = self._W_sparse.shape[0]
                W_dense = self._W_dense
                ones = np.ones(n)
                G = rho_draws.shape[0]
                mean_row_sum = np.empty(G)
                for g in range(G):
                    A = np.eye(n) - rho_draws[g] * W_dense
                    mean_row_sum[g] = np.linalg.solve(A, ones).mean()
            ni = self._nonintercept_indices
            direct_samples = mean_diag[:, None] * beta_draws[:, ni]
            total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMTobit):
            beta_draws = _get_posterior_draws(idata, "beta")
            ni = self._nonintercept_indices
            direct_samples = beta_draws[:, ni].copy()
            indirect_samples = np.zeros_like(direct_samples)
            total_samples = direct_samples.copy()

        elif isinstance(self, SDMTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            k = self._X.shape[1]
            kw = self._WX.shape[1]
            beta1_draws = beta_draws[:, :k]
            beta2_draws = beta_draws[:, k:k + kw]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag_M = np.mean(inv_eigs, axis=1)
            mean_diag_MW = np.mean((eigs * inv_eigs).real, axis=1)
            if self._is_row_std:
                mean_row_sum_M = 1.0 / (1.0 - rho_draws)
                mean_row_sum_MW = mean_row_sum_M
            else:
                n = self._W_sparse.shape[0]
                W_dense = self._W_dense
                ones = np.ones(n)
                G = rho_draws.shape[0]
                mean_row_sum_M = np.empty(G)
                mean_row_sum_MW = np.empty(G)
                for g in range(G):
                    A = np.eye(n) - rho_draws[g] * W_dense
                    M_ones = np.linalg.solve(A, ones)
                    mean_row_sum_M[g] = M_ones.mean()
                    mean_row_sum_MW[g] = (W_dense @ M_ones).mean()
            wx_idx = self._wx_column_indices
            direct_samples = np.column_stack([
                beta1_draws[:, j] * mean_diag_M + beta2_draws[:, idx] * mean_diag_MW
                for idx, j in enumerate(wx_idx)
            ])
            total_samples = np.column_stack([
                beta1_draws[:, j] * mean_row_sum_M + beta2_draws[:, idx] * mean_row_sum_MW
                for idx, j in enumerate(wx_idx)
            ])
            indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        return rho * (self._W_dense @ y_lat) + self._X @ beta

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

        The SAR Tobit model uses ``pm.Potential`` for both the residual
        log-likelihood and the Jacobian, so nothing is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, using the Tobit censoring formula:

        - Uncensored: log N(y | mu, sigma^2)
        - Censored:   log Phi((c - mu) / sigma)
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

        import pytensor
        import xarray as xr
        from scipy.stats import norm

        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        X = self._X
        W = self._W_dense
        censored = self._censored_mask
        censoring = self.censoring

        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        # Get posterior mean of latent y* for computing mu
        y_lat = self._posterior_latent_y_mean()
        mu = rho_f[:, None] * (W @ y_lat)[None, :] + beta_f @ X.T  # (s, n)

        # Tobit pointwise log-likelihood
        ll = np.empty((s, n), dtype=np.float64)
        uncens = ~censored
        if self.robust:
            nu_f = idata.posterior["nu"].values.reshape(s)
            from scipy.special import gammaln
            from scipy.stats import t as t_dist
            # Uncensored: log t(y | mu, sigma, nu)
            ll[:, uncens] = (
                gammaln((nu_f[:, None] + 1) / 2)
                - gammaln(nu_f[:, None] / 2)
                - 0.5 * np.log(nu_f[:, None] * np.pi)
                - np.log(sigma_f[:, None])
                - ((nu_f[:, None] + 1) / 2)
                * np.log1p(
                    ((self._y[uncens][None, :] - mu[:, uncens]) / sigma_f[:, None]) ** 2
                    / nu_f[:, None]
                )
            )
            # Censored: log T_cdf((c - mu) / sigma, nu)
            ll[:, censored] = t_dist.logcdf(
                (censoring - mu[:, censored]) / sigma_f[:, None],
                df=nu_f[:, None],
            )
        else:
            # Uncensored: log N(y | mu, sigma^2)
            ll[:, uncens] = -0.5 * (
                ((self._y[uncens][None, :] - mu[:, uncens]) / sigma_f[:, None]) ** 2
                + np.log(2.0 * np.pi)
                + 2.0 * np.log(sigma_f[:, None])
            )
            # Censored: log Phi((c - mu) / sigma)
            ll[:, censored] = norm.logcdf(
                (censoring - mu[:, censored]) / sigma_f[:, None]
            )

        # Eigenvalue-based Jacobian: log|I - rho*W| / n (pure numpy)
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - rv * eigs))) for rv in rho_f])  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata


class SEMTobit(_SpatialTobitBase):
    """Bayesian spatial error Tobit model.

    .. math::
        y^* = X\\beta + u,\\quad u = \\lambda W u + \\varepsilon,
        \\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome ``y = max(c, y*)``.

    **Robust regression**

    When ``robust=True``, the spatially-filtered error distribution is
    changed from Normal to Student-t.  For uncensored observations:

    .. math::

        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =
        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)

    and for censored observations:

    .. math::

        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)

    where :math:`T_\\nu` is the Student-t CDF and
    :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    """
    _spatial_diagnostics_tests = [
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_lag_test"],
        ).bayesian_lm_lag_test(m), "LM-Lag"),
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_wx_sem_test"],
        ).bayesian_lm_wx_sem_test(m), "LM-WX"),
    ]
    def _build_pymc_model(self) -> pm.Model:
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
        )
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_lat = self._latent_y_tensor()
            resid = y_lat - pt.dot(self._X, beta)
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

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        beta = self._posterior_mean("beta")
        ni = self._nonintercept_indices
        return {
            "direct": beta[ni].copy(),
            "indirect": np.zeros(len(ni)),
            "total": beta[ni].copy(),
            "feature_names": self._nonintercept_feature_names,
        }

    def _compute_spatial_effects_posterior(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws
        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            if self._is_row_std:
                mean_row_sum = 1.0 / (1.0 - rho_draws)
            else:
                n = self._W_sparse.shape[0]
                W_dense = self._W_dense
                ones = np.ones(n)
                G = rho_draws.shape[0]
                mean_row_sum = np.empty(G)
                for g in range(G):
                    A = np.eye(n) - rho_draws[g] * W_dense
                    mean_row_sum[g] = np.linalg.solve(A, ones).mean()
            ni = self._nonintercept_indices
            direct_samples = mean_diag[:, None] * beta_draws[:, ni]
            total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMTobit):
            beta_draws = _get_posterior_draws(idata, "beta")
            ni = self._nonintercept_indices
            direct_samples = beta_draws[:, ni].copy()
            indirect_samples = np.zeros_like(direct_samples)
            total_samples = direct_samples.copy()

        elif isinstance(self, SDMTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            k = self._X.shape[1]
            kw = self._WX.shape[1]
            beta1_draws = beta_draws[:, :k]
            beta2_draws = beta_draws[:, k:k + kw]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag_M = np.mean(inv_eigs, axis=1)
            mean_diag_MW = np.mean((eigs * inv_eigs).real, axis=1)
            if self._is_row_std:
                mean_row_sum_M = 1.0 / (1.0 - rho_draws)
                mean_row_sum_MW = mean_row_sum_M
            else:
                n = self._W_sparse.shape[0]
                W_dense = self._W_dense
                ones = np.ones(n)
                G = rho_draws.shape[0]
                mean_row_sum_M = np.empty(G)
                mean_row_sum_MW = np.empty(G)
                for g in range(G):
                    A = np.eye(n) - rho_draws[g] * W_dense
                    M_ones = np.linalg.solve(A, ones)
                    mean_row_sum_M[g] = M_ones.mean()
                    mean_row_sum_MW[g] = (W_dense @ M_ones).mean()
            wx_idx = self._wx_column_indices
            direct_samples = np.column_stack([
                beta1_draws[:, j] * mean_diag_M + beta2_draws[:, idx] * mean_diag_MW
                for idx, j in enumerate(wx_idx)
            ])
            total_samples = np.column_stack([
                beta1_draws[:, j] * mean_row_sum_M + beta2_draws[:, idx] * mean_row_sum_MW
                for idx, j in enumerate(wx_idx)
            ])
            indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        beta = self._posterior_mean("beta")
        return self._X @ beta

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

        The SEM Tobit model uses ``pm.Potential`` for both the error
        log-likelihood and the Jacobian, so nothing is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, using the Tobit censoring formula:

        - Uncensored: log N(y | mu, sigma^2)
        - Censored:   log Phi((c - mu) / sigma)

        where mu = X @ beta and the spatial filtering is absorbed into
        the Jacobian.
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

        import pytensor
        import pytensor.tensor as pt_ll
        import xarray as xr
        from scipy.stats import norm

        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = lam.shape
        s = c * d
        n = self._y.shape[0]
        X = self._X
        censored = self._censored_mask
        censoring = self.censoring

        lam_f = lam.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        # mu = X @ beta (SEM: spatial error, mean is just X@beta)
        mu = beta_f @ X.T  # (s, n)

        # Tobit pointwise log-likelihood
        ll = np.empty((s, n), dtype=np.float64)
        uncens = ~censored
        if self.robust:
            nu_f = idata.posterior["nu"].values.reshape(s)
            from scipy.special import gammaln
            from scipy.stats import t as t_dist
            ll[:, uncens] = (
                gammaln((nu_f[:, None] + 1) / 2)
                - gammaln(nu_f[:, None] / 2)
                - 0.5 * np.log(nu_f[:, None] * np.pi)
                - np.log(sigma_f[:, None])
                - ((nu_f[:, None] + 1) / 2)
                * np.log1p(
                    ((self._y[uncens][None, :] - mu[:, uncens]) / sigma_f[:, None]) ** 2
                    / nu_f[:, None]
                )
            )
            ll[:, censored] = t_dist.logcdf(
                (censoring - mu[:, censored]) / sigma_f[:, None],
                df=nu_f[:, None],
            )
        else:
            ll[:, uncens] = -0.5 * (
                ((self._y[uncens][None, :] - mu[:, uncens]) / sigma_f[:, None]) ** 2
                + np.log(2.0 * np.pi)
                + 2.0 * np.log(sigma_f[:, None])
            )
            ll[:, censored] = norm.logcdf(
                (censoring - mu[:, censored]) / sigma_f[:, None]
            )

        # Eigenvalue-based Jacobian: log|I - lam*W| / n (pure numpy)
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_f])  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata


class SDMTobit(_SpatialTobitBase):
    """Bayesian spatial Durbin Tobit model.

    .. math::
        y^* = \\rho Wy^* + X\\beta_1 + WX\\beta_2 + \\varepsilon,
        \\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome ``y = max(c, y*)``.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t.  For uncensored observations the density becomes:

    .. math::

        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =
        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)

    and for censored observations:

    .. math::

        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)

    where :math:`T_\\nu` is the Student-t CDF and
    :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    """
    _spatial_diagnostics_tests = [
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_error_test"],
        ).bayesian_lm_error_test(m), "LM-Error"),
    ]
    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
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
        )
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_lat = self._latent_y_tensor()
            resid = y_lat - rho * pt.dot(W_pt, y_lat) - pt.dot(Z, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_resid = pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), resid).sum()
            else:
                logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
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

    def _compute_spatial_effects_posterior(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws
        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            if self._is_row_std:
                mean_row_sum = 1.0 / (1.0 - rho_draws)
            else:
                n = self._W_sparse.shape[0]
                W_dense = self._W_dense
                ones = np.ones(n)
                G = rho_draws.shape[0]
                mean_row_sum = np.empty(G)
                for g in range(G):
                    A = np.eye(n) - rho_draws[g] * W_dense
                    mean_row_sum[g] = np.linalg.solve(A, ones).mean()
            ni = self._nonintercept_indices
            direct_samples = mean_diag[:, None] * beta_draws[:, ni]
            total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMTobit):
            beta_draws = _get_posterior_draws(idata, "beta")
            ni = self._nonintercept_indices
            direct_samples = beta_draws[:, ni].copy()
            indirect_samples = np.zeros_like(direct_samples)
            total_samples = direct_samples.copy()

        elif isinstance(self, SDMTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            k = self._X.shape[1]
            kw = self._WX.shape[1]
            beta1_draws = beta_draws[:, :k]
            beta2_draws = beta_draws[:, k:k + kw]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag_M = np.mean(inv_eigs, axis=1)
            mean_diag_MW = np.mean((eigs * inv_eigs).real, axis=1)
            if self._is_row_std:
                mean_row_sum_M = 1.0 / (1.0 - rho_draws)
                mean_row_sum_MW = mean_row_sum_M
            else:
                n = self._W_sparse.shape[0]
                W_dense = self._W_dense
                ones = np.ones(n)
                G = rho_draws.shape[0]
                mean_row_sum_M = np.empty(G)
                mean_row_sum_MW = np.empty(G)
                for g in range(G):
                    A = np.eye(n) - rho_draws[g] * W_dense
                    M_ones = np.linalg.solve(A, ones)
                    mean_row_sum_M[g] = M_ones.mean()
                    mean_row_sum_MW[g] = (W_dense @ M_ones).mean()
            wx_idx = self._wx_column_indices
            direct_samples = np.column_stack([
                beta1_draws[:, j] * mean_diag_M + beta2_draws[:, idx] * mean_diag_MW
                for idx, j in enumerate(wx_idx)
            ])
            total_samples = np.column_stack([
                beta1_draws[:, j] * mean_row_sum_M + beta2_draws[:, idx] * mean_row_sum_MW
                for idx, j in enumerate(wx_idx)
            ])
            indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        Z = np.hstack([self._X, self._WX])
        return rho * (self._W_dense @ y_lat) + Z @ beta

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

        The SDM Tobit model uses ``pm.Potential`` for both the residual
        log-likelihood and the Jacobian, so nothing is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, using the Tobit censoring formula:

        - Uncensored: log N(y | mu, sigma^2)
        - Censored:   log Phi((c - mu) / sigma)

        where mu = rho*Wy* + Z@beta.
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

        import pytensor
        import pytensor.tensor as pt_ll
        import xarray as xr
        from scipy.stats import norm

        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        Z = np.hstack([self._X, self._WX])
        W = self._W_dense
        censored = self._censored_mask
        censoring = self.censoring

        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        # Get posterior mean of latent y* for computing mu
        y_lat = self._posterior_latent_y_mean()
        mu = rho_f[:, None] * (W @ y_lat)[None, :] + beta_f @ Z.T  # (s, n)

        # Tobit pointwise log-likelihood
        ll = np.empty((s, n), dtype=np.float64)
        uncens = ~censored
        if self.robust:
            nu_f = idata.posterior["nu"].values.reshape(s)
            from scipy.special import gammaln
            from scipy.stats import t as t_dist
            ll[:, uncens] = (
                gammaln((nu_f[:, None] + 1) / 2)
                - gammaln(nu_f[:, None] / 2)
                - 0.5 * np.log(nu_f[:, None] * np.pi)
                - np.log(sigma_f[:, None])
                - ((nu_f[:, None] + 1) / 2)
                * np.log1p(
                    ((self._y[uncens][None, :] - mu[:, uncens]) / sigma_f[:, None]) ** 2
                    / nu_f[:, None]
                )
            )
            ll[:, censored] = t_dist.logcdf(
                (censoring - mu[:, censored]) / sigma_f[:, None],
                df=nu_f[:, None],
            )
        else:
            ll[:, uncens] = -0.5 * (
                ((self._y[uncens][None, :] - mu[:, uncens]) / sigma_f[:, None]) ** 2
                + np.log(2.0 * np.pi)
                + 2.0 * np.log(sigma_f[:, None])
            )
            ll[:, censored] = norm.logcdf(
                (censoring - mu[:, censored]) / sigma_f[:, None]
            )

        # Eigenvalue-based Jacobian: log|I - rho*W| / n (pure numpy)
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - rv * eigs))) for rv in rho_f])  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata
