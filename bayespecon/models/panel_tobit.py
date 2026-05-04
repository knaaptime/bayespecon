"""Spatial panel Tobit models.

Implements left-censored (default at 0) panel spatial Tobit variants:

- ``SARPanelTobit``: spatial lag panel Tobit
- ``SEMPanelTobit``: spatial error panel Tobit

Notes
-----
The Tobit specification splits the observation vector into two pieces:

* **Uncensored** observations enter the likelihood directly through the
  Gaussian density evaluated at the observed value.
* **Censored** observations have unknown latent values
  :math:`y^*_i < c` (with :math:`c` the censoring threshold).  These are
  augmented in the PyMC model via a half-Normal *gap* parameter
  ``y_cens_gap`` such that
  :math:`y^*_i = c - \\text{gap}_i` with :math:`\\text{gap}_i \\geq 0`,
  which trades the analytic ``Φ((c-μ)/σ)`` factor of a marginal Tobit
  likelihood for tractable posterior sampling on the joint
  :math:`(\\theta, y^*_{\\text{cens}})` space (Albert & Chib, 1993;
  Chib, 1992).

These classes force ``model=0`` (pooled transform) because within
transformations are not compatible with the censoring augmentation on
the *observed* scale: subtracting unit means would mix censored and
uncensored values inside the Gaussian likelihood.

References
----------
Chib, S. (1992). Bayes inference in the Tobit censored regression
model. *Journal of Econometrics*, 51(1–2), 79–99.

Albert, J.H. & Chib, S. (1993). Bayesian analysis of binary and
polychotomous response data. *Journal of the American Statistical
Association*, 88(422), 669–679.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import sparse as pts

from ._sampler import use_jax_likelihood
from .base import _write_log_likelihood_to_idata
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
            gap_hat = (
                self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            )
            y_lat[self._censored_idx] = self.censoring - np.asarray(
                gap_hat, dtype=float
            )
        return y_lat


class SARPanelTobit(_PanelTobitBase):
    """Bayesian spatial lag panel Tobit model.

    .. math::
        y^* = \\rho W y^* + X\\beta + \\varepsilon,\\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome

    .. math::
        y = \\max(c, y^*)

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``, ``unit_col``,
        ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked observed outcome of shape ``(N*T,)``. Required in
        matrix mode. Values at or below ``censoring`` are treated as
        left-censored.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be
        row-standardised.
    unit_col, time_col : str, optional
        Column names identifying the unit and time period in ``data``.
        Required in formula mode.
    N, T : int, optional
        Cross-sectional and time dimensions. Required in matrix mode.
    censoring : float, default 0.0
        Left-censoring threshold ``c``. Observations with
        ``y <= censoring`` are treated as censored and the latent
        ``y*`` is sampled from a HalfNormal gap below ``c``.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\\beta`.
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\\sigma`.
        - ``censor_sigma`` (float, default 10.0): HalfNormal prior
          std for the latent gap below the censoring threshold.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`; auto-selected when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used.

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

        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_lat = self._latent_y_tensor()
            resid = (
                y_lat
                - rho * pts.structured_dot(W_pt, y_lat[:, None]).flatten()
                - pt.dot(self._X, beta)
            )
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_resid = pm.logp(
                    pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), resid
                ).sum()
            else:
                logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        return rho * self._sparse_panel_lag(y_lat) + self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        ni = self._nonintercept_indices
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        mean_row_sum = float(self._batch_mean_row_sum(np.array([rho]))[0])
        direct = mean_diag * beta[ni]
        total = mean_row_sum * beta[ni]
        indirect = total - direct
        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._nonintercept_feature_names,
        }

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelTobit):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples

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

        from scipy.stats import norm

        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values

        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        X = self._X
        censored = self._censored_mask
        censoring = self.censoring

        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)

        # Get posterior mean of latent y* for computing mu
        y_lat = self._posterior_latent_y_mean()
        Wy_lat = self._sparse_panel_lag(y_lat)
        mu = rho_f[:, None] * Wy_lat[None, :] + beta_f @ X.T  # (s, n)

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

        # Jacobian (respects logdet_method)
        jac = self._logdet_numpy_vec_fn(rho_f) * self._T  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata


class SEMPanelTobit(_PanelTobitBase):
    """Bayesian spatial error panel Tobit model.

    .. math::
        y^* = X\\beta + u,\\quad u = \\lambda W u + \\varepsilon,
        \\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome ``y = max(c, y*)``.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``, ``unit_col``,
        ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked observed outcome of shape ``(N*T,)``. Required in
        matrix mode. Values at or below ``censoring`` are treated as
        left-censored.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be
        row-standardised.
    unit_col, time_col : str, optional
        Column names identifying the unit and time period in ``data``.
        Required in formula mode.
    N, T : int, optional
        Cross-sectional and time dimensions. Required in matrix mode.
    censoring : float, default 0.0
        Left-censoring threshold ``c``.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\\beta`.
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\\sigma`.
        - ``censor_sigma`` (float, default 10.0): HalfNormal prior
          std for the latent gap below the censoring threshold.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used.

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

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse

        n_obs = int(self._y.shape[0])
        # ``_logdet_pytensor_fn`` already includes the T multiplier.
        inv_n = 1.0 / n_obs
        jax_logp = use_jax_likelihood(nuts_sampler)
        n_cens = int(self._censored_idx.size)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            if self.robust:
                self._add_nu_prior(model)

            if jax_logp:
                # Materialise censored data-augmentation RV up front so the
                # closure can reconstruct the latent y from ``value``.
                if n_cens > 0:
                    censor_sigma = float(self.priors.get("censor_sigma", 10.0))
                    y_cens_gap = pm.HalfNormal(
                        "y_cens_gap", sigma=censor_sigma, shape=n_cens
                    )
                else:
                    y_cens_gap = None

                X_const = pt.as_tensor_variable(self._X)
                cens_idx_const = (
                    pt.as_tensor_variable(self._censored_idx) if n_cens > 0 else None
                )
                censoring_const = self.censoring

                def _eps(value, lam_, beta_, gap_):
                    y_lat = value
                    if gap_ is not None:
                        y_cens = censoring_const - gap_
                        y_lat = pt.set_subtensor(y_lat[cens_idx_const], y_cens)
                    resid = y_lat - pt.dot(X_const, beta_)
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = model["nu"]

                    if n_cens > 0:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_, gap_, nu_):
                            eps = _eps(value, lam_, beta_, gap_)
                            log_dens = pm.logp(
                                pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            y_cens_gap,
                            nu,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
                    else:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_, nu_):
                            eps = _eps(value, lam_, beta_, None)
                            log_dens = pm.logp(
                                pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            nu,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
                else:
                    if n_cens > 0:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_, gap_):
                            eps = _eps(value, lam_, beta_, gap_)
                            log_dens = pm.logp(
                                pm.Normal.dist(mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            y_cens_gap,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
                    else:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_):
                            eps = _eps(value, lam_, beta_, None)
                            log_dens = pm.logp(
                                pm.Normal.dist(mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
            else:
                y_lat = self._latent_y_tensor()
                resid = y_lat - pt.dot(self._X, beta)
                eps = resid - lam * pts.structured_dot(W_pt, resid[:, None]).flatten()
                if self.robust:
                    nu = model["nu"]
                    logp_eps = pm.logp(
                        pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps
                    ).sum()
                else:
                    logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
                pm.Potential("eps_loglik", logp_eps)
                pm.Potential("jacobian", logdet_fn(lam))

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        beta = self._posterior_mean("beta")
        return self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        ni = self._nonintercept_indices
        beta = self._posterior_mean("beta")
        return {
            "direct": beta[ni].copy(),
            "indirect": np.zeros_like(beta[ni]),
            "total": beta[ni].copy(),
            "feature_names": self._nonintercept_feature_names,
        }

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelTobit):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples

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

        # Jacobian (respects logdet_method)
        jac = self._logdet_numpy_vec_fn(lam_f) * self._T  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata
