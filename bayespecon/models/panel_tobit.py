"""Spatial panel Tobit models.

Implements left-censored (default at 0) panel spatial Tobit variants:

- ``SARPanelTobit``: spatial lag panel Tobit
- ``SEMPanelTobit``: spatial error panel Tobit

Notes
-----
These classes force ``model=0`` (pooled transform) because within transforms
are not compatible with censoring augmentation on the observed scale.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..logdet import make_logdet_fn
from .panel_base import SpatialPanelModel


class _PanelTobitBase(SpatialPanelModel):
    """Shared helpers for panel Tobit classes."""

    def __init__(self, *args, censoring: float = 0.0, **kwargs):
        kwargs.pop("model", None)
        self.censoring = float(censoring)
        super().__init__(*args, model=0, **kwargs)
        self._censored_mask = self._y <= self.censoring + 1e-12
        self._censored_idx = np.where(self._censored_mask)[0]

    def _latent_y_tensor(self) -> pt.TensorVariable:
        y_lat = pt.as_tensor_variable(self._y.astype(np.float64))
        n_cens = int(self._censored_idx.size)
        if n_cens > 0:
            censor_sigma = float(self.priors.get("censor_sigma", 10.0))
            y_cens_gap = pm.HalfNormal("y_cens_gap", sigma=censor_sigma, shape=n_cens)
            y_cens = self.censoring - y_cens_gap
            y_lat = pt.set_subtensor(y_lat[self._censored_idx], y_cens)
        return y_lat

    def _posterior_latent_y_mean(self) -> np.ndarray:
        y_lat = self._y.copy().astype(float)
        if self._censored_idx.size > 0 and "y_cens_gap" in self._idata.posterior:
            gap_hat = self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            y_lat[self._censored_idx] = self.censoring - np.asarray(gap_hat, dtype=float)
        return y_lat


class SARPanelTobit(_PanelTobitBase):
    """Bayesian spatial lag panel Tobit model.

    .. math::
        y^* = \\rho W y^* + X\\beta + \\varepsilon,\\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome

    .. math::
        y = \\max(c, y^*)

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
            T=self._T,
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

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        return rho * (self._W_dense @ y_lat) + self._X @ beta

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
        direct = mean_diag * beta
        total = mean_row_sum * beta
        indirect = total - direct
        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._feature_names,
        }

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

        The SAR panel Tobit model uses ``pm.Potential`` for both the
        residual log-likelihood and the Jacobian, so nothing is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, using the Tobit censoring formula:

        - Uncensored: log N(y | mu, sigma^2)
        - Censored:   log Phi((c - mu) / sigma)

        where mu = rho*Wy* + X@beta.
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

        # Eigenvalue-based Jacobian: log|I - rho*W| * T / n (pure numpy)
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - rv * eigs))) for rv in rho_f]) * self._T  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata


class SEMPanelTobit(_PanelTobitBase):
    """Bayesian spatial error panel Tobit model.

    .. math::
        y^* = X\\beta + u,\\quad u = \\lambda W u + \\varepsilon,
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
            T=self._T,
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

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        beta = self._posterior_mean("beta")
        return self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }

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

        The SEM panel Tobit model uses ``pm.Potential`` for both the error
        log-likelihood and the Jacobian, so nothing is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, using the Tobit censoring formula:

        - Uncensored: log N(y | mu, sigma^2)
        - Censored:   log Phi((c - mu) / sigma)

        where mu = X@beta and the spatial filtering is absorbed into
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

        # Eigenvalue-based Jacobian: log|I - lam*W| * T / n (pure numpy)
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_f]) * self._T  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata
