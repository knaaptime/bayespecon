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
from pytensor import sparse as pts

from .base import SpatialModel


class _SpatialTobitBase(SpatialModel):
    """Shared helpers for spatial Tobit models."""

    def __init__(self, *args, censoring: float = 0.0, **kwargs):
        self.censoring = float(censoring)
        super().__init__(*args, **kwargs)
        # Censored observations are those at (or below) the censoring point.
        # Use exact comparison: ``y_i == censoring`` is the definition of a
        # left-censored observation; an arbitrary ``+ 1e-12`` slack would
        # silently mark uncensored values that happen to be just above the
        # threshold as censored.
        self._censored_mask = self._y <= self.censoring
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
            gap_hat = (
                self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            )
            y_lat[self._censored_idx] = self.censoring - np.asarray(
                gap_hat, dtype=float
            )
        return y_lat


class SARTobit(_SpatialTobitBase):
    """Bayesian spatial autoregressive Tobit model.

    .. math::
        y^* = \\rho W y^* + X\\beta + \\varepsilon,\\quad \\varepsilon \\sim N(0,\\sigma^2 I),

    with observed outcome

    .. math::
        y = \\max(c, y^*)

    where ``c`` is the left-censoring point (default ``0``). Censored
    observations contribute their CDF to the likelihood; uncensored
    observations contribute the density of :math:`y^*` evaluated at
    :math:`y`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Observed (censored) response of shape ``(n,)``. Required in
        matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``; see :class:`SAR` for
        accepted formats.
    censoring : float, default 0.0
        Left-censoring threshold ``c``. Observations with
        ``y <= censoring`` are treated as censored and the latent
        :math:`y^*` is sampled.
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
        - ``censor_sigma`` (float, default 10.0): HalfNormal scale for
          the latent ``y_cens_gap`` shifting censored draws below ``c``.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects ``"eigenvalue"`` for ``n <= 2000`` else
        ``"chebyshev"``.
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t. For uncensored observations the density becomes:

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
                fromlist=["bayesian_lm_wx_test"],
            ).bayesian_lm_wx_test(m),
            "LM-WX",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_wx_test"],
            ).bayesian_robust_lm_wx_test(m),
            "Robust-LM-WX",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_error_sar_test"],
            ).bayesian_robust_lm_error_sar_test(m),
            "Robust-LM-Error",
        ),
    ]

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

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        mean_row_sum = float(self._batch_mean_row_sum(np.array([rho]))[0])
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

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
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
            beta2_draws = beta_draws[:, k : k + kw]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag_M = np.mean(inv_eigs, axis=1)
            mean_diag_MW = np.mean((eigs * inv_eigs).real, axis=1)
            mean_row_sum_M = self._batch_mean_row_sum(rho_draws)
            mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)
            wx_idx = self._wx_column_indices
            direct_samples = (
                mean_diag_M[:, None] * beta1_draws[:, wx_idx]
                + mean_diag_MW[:, None] * beta2_draws
            )
            total_samples = (
                mean_row_sum_M[:, None] * beta1_draws[:, wx_idx]
                + mean_row_sum_MW[:, None] * beta2_draws
            )
            indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Reported fitted mean: ``max(c, E[y* | X, params])``.

        The structural latent mean for SAR-Tobit is
        :math:`E[y^* \\mid X] = (I - \\rho W)^{-1} X \\beta`, evaluated at the
        posterior mean of :math:`(\\rho, \\beta)`. Censored observations are
        reported at the censoring point ``c`` (consistent with the
        observation rule ``y = max(c, y*)``).
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        n = self._y.shape[0]
        A = np.eye(n) - rho * self._W_dense
        structural = np.linalg.solve(A, self._X @ beta)
        return np.maximum(self.censoring, structural)

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

        # Structural latent mean per draw: mu = (I - rho W)^{-1} X beta.
        # We deliberately do NOT plug the posterior mean of the latent y*
        # into mu: the pointwise observed-data likelihood must be a function
        # of the model parameters only (not of latent quantities), otherwise
        # marginal-likelihood / WAIC / LOO comparisons are biased.
        I_n = np.eye(n)
        Xb = beta_f @ X.T  # (s, n)
        mu = np.empty((s, n), dtype=np.float64)
        for i in range(s):
            mu[i] = np.linalg.solve(I_n - rho_f[i] * W, Xb[i])

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

        # Jacobian (respects logdet_method)
        jac = self._logdet_numpy_vec_fn(rho_f)  # (n_draws,)
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

    with observed outcome ``y = max(c, y*)``. Censored observations
    contribute their CDF; uncensored observations contribute the
    spatially-filtered density of :math:`y^*`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Observed (censored) response. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``; see :class:`SAR` for
        accepted formats.
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
        - ``censor_sigma`` (float, default 10.0): HalfNormal scale for
          the latent ``y_cens_gap``.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.

    Notes
    -----
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
                fromlist=["bayesian_lm_wx_sem_test"],
            ).bayesian_lm_wx_sem_test(m),
            "LM-WX",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_lag_sem_test"],
            ).bayesian_robust_lm_lag_sem_test(m),
            "Robust-LM-Lag",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_wx_sem_test"],
            ).bayesian_robust_lm_wx_sem_test(m),
            "Robust-LM-WX",
        ),
    ]

    def _build_pymc_model(self) -> pm.Model:
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_lat = self._latent_y_tensor()
            resid = y_lat - pt.dot(self._X, beta)
            eps = resid - lam * pts.structured_dot(W_pt, resid[:, None]).flatten()
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_eps = pm.logp(
                    pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps
                ).sum()
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

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
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
            beta2_draws = beta_draws[:, k : k + kw]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag_M = np.mean(inv_eigs, axis=1)
            mean_diag_MW = np.mean((eigs * inv_eigs).real, axis=1)
            mean_row_sum_M = self._batch_mean_row_sum(rho_draws)
            mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)
            wx_idx = self._wx_column_indices
            direct_samples = (
                mean_diag_M[:, None] * beta1_draws[:, wx_idx]
                + mean_diag_MW[:, None] * beta2_draws
            )
            total_samples = (
                mean_row_sum_M[:, None] * beta1_draws[:, wx_idx]
                + mean_row_sum_MW[:, None] * beta2_draws
            )
            indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Reported fitted mean: ``max(c, E[y* | X, params])``.

        For SEM-Tobit the structural latent mean is
        :math:`E[y^* \\mid X] = X\\beta` (the spatial filter operates on the
        error term and integrates out). Censored entries are reported at
        the censoring point.
        """
        beta = self._posterior_mean("beta")
        return np.maximum(self.censoring, self._X @ beta)

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

        # Jacobian (respects logdet_method)
        jac = self._logdet_numpy_vec_fn(lam_f)  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata


class SDMTobit(_SpatialTobitBase):
    """Bayesian spatial Durbin Tobit model.

    .. math::
        y^* = \\rho Wy^* + X\\beta + WX\\theta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome ``y = max(c, y*)``. The sampled coefficient
    vector stacks the local and lagged-regressor blocks as
    :math:`[\\beta, \\theta]`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Observed (censored) response. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``.
    censoring : float, default 0.0
        Left-censoring threshold ``c``.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`[\\beta, \\theta]`.
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\\sigma`.
        - ``censor_sigma`` (float, default 10.0): HalfNormal scale for
          the latent ``y_cens_gap``.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`; auto-selected when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
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
                fromlist=["bayesian_robust_lm_error_sdm_test"],
            ).bayesian_robust_lm_error_sdm_test(m),
            "Robust-LM-Error",
        ),
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
                - pt.dot(Z, beta)
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

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1, beta2 = beta[:k], beta[k : k + kw]

        eigs = self._W_eigs
        inv_eigs = 1.0 / (1.0 - rho * eigs)
        mean_diag_M = float(np.mean(inv_eigs.real))
        mean_diag_MW = float(np.mean((eigs * inv_eigs).real))
        rho_arr = np.array([rho])
        mean_row_sum_M = float(self._batch_mean_row_sum(rho_arr)[0])
        mean_row_sum_MW = float(self._batch_mean_row_sum_MW(rho_arr)[0])
        direct = np.array(
            [
                beta1[j] * mean_diag_M + b2 * mean_diag_MW
                for j, b2 in zip(self._wx_column_indices, beta2)
            ]
        )
        total = np.array(
            [
                beta1[j] * mean_row_sum_M + b2 * mean_row_sum_MW
                for j, b2 in zip(self._wx_column_indices, beta2)
            ]
        )
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
        """Compute posterior samples of direct, indirect, and total effects."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
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
            beta2_draws = beta_draws[:, k : k + kw]
            eigs = self._W_eigs.real.astype(np.float64)
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag_M = np.mean(inv_eigs, axis=1)
            mean_diag_MW = np.mean((eigs * inv_eigs).real, axis=1)
            mean_row_sum_M = self._batch_mean_row_sum(rho_draws)
            mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)
            wx_idx = self._wx_column_indices
            direct_samples = (
                mean_diag_M[:, None] * beta1_draws[:, wx_idx]
                + mean_diag_MW[:, None] * beta2_draws
            )
            total_samples = (
                mean_row_sum_M[:, None] * beta1_draws[:, wx_idx]
                + mean_row_sum_MW[:, None] * beta2_draws
            )
            indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Reported fitted mean: ``max(c, E[y* | X, params])``.

        The structural latent mean for SDM-Tobit is
        :math:`E[y^* \\mid X] = (I - \\rho W)^{-1} (X\\beta + WX\\theta)`,
        evaluated at posterior means; censored entries are reported at
        the censoring point.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X, self._WX])
        n = self._y.shape[0]
        A = np.eye(n) - rho * self._W_dense
        structural = np.linalg.solve(A, Z @ beta)
        return np.maximum(self.censoring, structural)

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

        # Structural latent mean per draw: mu = (I - rho W)^{-1} Z beta.
        # See SARTobit.fit for rationale (must not depend on posterior y*).
        I_n = np.eye(n)
        Zb = beta_f @ Z.T  # (s, n)
        mu = np.empty((s, n), dtype=np.float64)
        for i in range(s):
            mu[i] = np.linalg.solve(I_n - rho_f[i] * W, Zb[i])

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
        jac = self._logdet_numpy_vec_fn(rho_f)  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata
