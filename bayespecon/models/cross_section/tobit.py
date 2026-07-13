"""Spatial Tobit model classes.

Implements left-censored (default at 0) Bayesian spatial Tobit variants:

- ``SARTobit``: spatial autoregressive Tobit
- ``SEMTobit``: spatial error Tobit
- ``SDMTobit``: spatial Durbin Tobit

All classes use latent-data augmentation for censored observations.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor import sparse as pts

from ..._lazy_deps import az, pm
from .._base._shared import _tobit_pointwise_loglik, _write_log_likelihood_to_idata
from ..base import SpatialModel
from ..priors import SARTobitPriors, SDMTobitPriors, SEMTobitPriors


class _SpatialTobitBase(SpatialModel):
    """Shared helpers for spatial Tobit models."""

    def __init__(self, *args, censoring: float = 0.0, **kwargs):
        self.censoring = float(censoring)
        super().__init__(*args, **kwargs)
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

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior samples of direct, indirect, and total effects.

        Effects are reported on the latent (uncensored) scale, so they share
        the linear SAR/SEM/SDM impact structure of the corresponding Gaussian
        model.  The diagonal traces ride the resolvent identities
        ``tr(S)/n = 1 - (rho/n)*g`` and ``tr(SW)/n = -g/n`` and need no
        eigendecomposition.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            mean_diag = self._batch_mean_diag(rho_draws)
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
            mean_diag_M = self._batch_mean_diag(rho_draws)
            mean_diag_MW = self._batch_mean_diag_MW(rho_draws)
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

        else:
            raise NotImplementedError(
                f"Spatial effects not implemented for {type(self).__name__}."
            )

        return direct_samples, indirect_samples, total_samples


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

    _priors_cls = SARTobitPriors

    def _build_pymc_model(self) -> pm.Model:
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1000000.0)
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

    def _postprocess_idata(self, idata: az.InferenceData) -> az.InferenceData:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SAR Tobit model expresses both the residual log-likelihood and the
        Jacobian via ``pm.Potential``, so PyMC captures no pointwise
        ``log_likelihood``.  We rebuild it here from the Tobit censoring
        formula (Normal or Student-t density for uncensored observations, the
        matching left-tail log-CDF for censored ones) with the structural
        latent mean :math:`\\mu = (I - \\rho W)^{-1} X\\beta`.
        """
        if "log_likelihood" in idata.groups() and "obs" in idata.log_likelihood:
            return idata
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        W_sp = self._W_sparse
        n = self._y.shape[0]
        Xb = beta_f @ self._X.T
        mu = np.empty((s, n), dtype=np.float64)
        # Prefer klujax (cached symbolic analysis, fast batched solve)
        # over dense np.linalg.solve loop.
        from ..._jax_dispatch import _klujax_available

        if _klujax_available():
            import klujax
            import scipy.sparse as _sp

            # Fixed COO pattern for (I - ρW): only values change with ρ
            I_coo = _sp.eye(n, format="coo")
            W_coo = W_sp.tocoo()
            # Merge I and W patterns
            all_rows = np.concatenate([I_coo.row, W_coo.row])
            all_cols = np.concatenate([I_coo.col, W_coo.col])
            shape = (n, n)
            const_coo = _sp.coo_matrix(
                (np.concatenate([np.ones(I_coo.nnz), np.zeros(W_coo.nnz)]),
                 (all_rows, all_cols)), shape=shape)
            const_coo.sum_duplicates()
            w_coo = _sp.coo_matrix(
                (np.concatenate([np.zeros(I_coo.nnz), W_coo.data]),
                 (all_rows, all_cols)), shape=shape)
            w_coo.sum_duplicates()
            Ai = np.asarray(const_coo.row, dtype=np.int32)
            Aj = np.asarray(const_coo.col, dtype=np.int32)
            const_vals = np.asarray(const_coo.data, dtype=np.float64)
            w_vals = np.asarray(w_coo.data, dtype=np.float64)
            symbolic = klujax.analyze(Ai, Aj, n)
            for i in range(s):
                Ax = const_vals - rho_f[i] * w_vals
                mu[i] = np.asarray(
                    klujax.solve_with_symbol(Ai, Aj, Ax, Xb[i], symbolic),
                    dtype=np.float64,
                )
        else:
            from ..._ops._backend import _solve_sparse_vector

            I_sp = _sp.eye(n, format="csc")
            for i in range(s):
                A_csc = (I_sp - rho_f[i] * W_sp).tocsc()
                mu[i] = _solve_sparse_vector(A_csc, Xb[i])
        nu_f = idata.posterior["nu"].values.reshape(s) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        jac = self._logdet_numpy_vec_fn(rho_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
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

    _priors_cls = SEMTobitPriors

    def _build_pymc_model(self) -> pm.Model:
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1000000.0)
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

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Reported fitted mean: ``max(c, E[y* | X, params])``.

        For SEM-Tobit the structural latent mean is
        :math:`E[y^* \\mid X] = X\\beta` (the spatial filter operates on the
        error term and integrates out). Censored entries are reported at
        the censoring point.
        """
        beta = self._posterior_mean("beta")
        return np.maximum(self.censoring, self._X @ beta)

    def _postprocess_idata(self, idata: az.InferenceData) -> az.InferenceData:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SEM Tobit model expresses both the error log-likelihood and the
        Jacobian via ``pm.Potential``, so PyMC captures no pointwise
        ``log_likelihood``.  We rebuild it here from the Tobit censoring
        formula with structural latent mean :math:`\\mu = X\\beta` (the spatial
        filter operates on the error term and is absorbed into the Jacobian).
        """
        if "log_likelihood" in idata.groups() and "obs" in idata.log_likelihood:
            return idata
        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = lam.shape
        s = c * d
        n = self._y.shape[0]
        lam_f = lam.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        mu = beta_f @ self._X.T
        nu_f = idata.posterior["nu"].values.reshape(s) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        jac = self._logdet_numpy_vec_fn(lam_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
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

    _priors_cls = SDMTobitPriors

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        Z = np.hstack([self._X, self._WX])
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1000000.0)
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

    def _postprocess_idata(self, idata: az.InferenceData) -> az.InferenceData:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SDM Tobit model expresses both the residual log-likelihood and the
        Jacobian via ``pm.Potential``, so PyMC captures no pointwise
        ``log_likelihood``.  We rebuild it here from the Tobit censoring
        formula with structural latent mean
        :math:`\\mu = (I - \\rho W)^{-1} (X\\beta + WX\\theta)`, where the
        stacked design ``Z = [X, WX]`` carries the Durbin lags.
        """
        if "log_likelihood" in idata.groups() and "obs" in idata.log_likelihood:
            return idata
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        Z = np.hstack([self._X, self._WX])
        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        W_sp = self._W_sparse
        Zb = beta_f @ Z.T
        mu = np.empty((s, n), dtype=np.float64)
        # Prefer klujax (cached symbolic analysis) over dense solve loop
        from ..._jax_dispatch import _klujax_available

        if _klujax_available():
            import klujax
            import scipy.sparse as _sp

            I_coo = _sp.eye(n, format="coo")
            W_coo = W_sp.tocoo()
            all_rows = np.concatenate([I_coo.row, W_coo.row])
            all_cols = np.concatenate([I_coo.col, W_coo.col])
            shape = (n, n)
            const_coo = _sp.coo_matrix(
                (
                    np.concatenate([np.ones(I_coo.nnz), np.zeros(W_coo.nnz)]),
                    (all_rows, all_cols),
                ),
                shape=shape,
            )
            const_coo.sum_duplicates()
            w_coo = _sp.coo_matrix(
                (
                    np.concatenate([np.zeros(I_coo.nnz), W_coo.data]),
                    (all_rows, all_cols),
                ),
                shape=shape,
            )
            w_coo.sum_duplicates()
            Ai = np.asarray(const_coo.row, dtype=np.int32)
            Aj = np.asarray(const_coo.col, dtype=np.int32)
            const_vals = np.asarray(const_coo.data, dtype=np.float64)
            w_vals = np.asarray(w_coo.data, dtype=np.float64)
            symbolic = klujax.analyze(Ai, Aj, n)
            for i in range(s):
                Ax = const_vals - rho_f[i] * w_vals
                mu[i] = np.asarray(
                    klujax.solve_with_symbol(Ai, Aj, Ax, Zb[i], symbolic),
                    dtype=np.float64,
                )
        else:
            from ..._ops._backend import _solve_sparse_vector

            I_sp = _sp.eye(n, format="csc")
            for i in range(s):
                A_csc = (I_sp - rho_f[i] * W_sp).tocsc()
                mu[i] = _solve_sparse_vector(A_csc, Zb[i])
        nu_f = idata.posterior["nu"].values.reshape(s) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        jac = self._logdet_numpy_vec_fn(rho_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata
