"""Spatial Durbin Error Model (SDEM).

y = X @ beta1 + W @ X @ beta2 + u,
u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Combines spatially lagged covariates (SLX) with a spatially autocorrelated
error process (SEM). No spatial lag on y, so rho is absent.
Jacobian log|I - lambda*W| is required for the error process.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base import SpatialModel


class SDEM(SpatialModel):
    """Bayesian Spatial Durbin Error Model.

    Combines spatial lags of the regressors :math:`X` with a spatial
    autoregressive disturbance:

    .. math::
        y = X\\beta + WX\\theta + u,
        \\quad u = \\lambda Wu + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The sampled coefficient vector stacks the local and lagged-regressor
    blocks as :math:`[\\beta, \\theta]`. The likelihood includes the
    spatial Jacobian :math:`\\log|I - \\lambda W|`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. Intercept is included by default.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Dependent variable of shape ``(n,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode. DataFrame columns are
        preserved as feature names.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``. Accepts a
        :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
        matrix. The legacy :class:`libpysal.weights.W` object is **not**
        accepted; pass ``w.sparse`` or ``libpysal.graph.Graph.from_W(w)``.
        Should be row-standardised; a :class:`UserWarning` is raised
        otherwise.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of the
          Uniform prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of the
          Uniform prior on :math:`\\lambda`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`[\\beta, \\theta]`.
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\\sigma`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`. ``None`` (default)
        auto-selects ``"eigenvalue"`` for ``n <= 2000`` else
        ``"chebyshev"``. Other options: ``"exact"``, ``"dense_grid"``,
        ``"sparse_grid"``, ``"spline"``, ``"mc"``, ``"ilu"``.
    robust : bool, default False
        If True, replace the Normal disturbance with Student-t. See
        *Robust regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``.

    Notes
    -----
    Because the spatial autoregression enters only through the
    disturbance, direct effects equal :math:`\\beta` and indirect
    effects equal :math:`\\theta` (no global spillover multiplier).

    **Robust regression**

    When ``robust=True``, the spatially-filtered innovation is
    Student-t:

    .. math::

        \\varepsilon = (I - \\lambda W)(y - X\\beta - WX\\theta)
        \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)`
    with rate ``nu_lam`` (default 1/30, mean ≈ 30).
    """

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_lag_sdem_test"],
            ).bayesian_lm_lag_sdem_test(m),
            "LM-Lag-SDEM",
        ),
    ]

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> "az.InferenceData":
        """Draw samples from the posterior. Accepts ``idata_kwargs`` for ArviZ compatibility.

        Parameters
        ----------
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` for InferenceData creation. If contains
            ``log_likelihood: True``, the complete pointwise log-likelihood
            (including the Jacobian correction) is attached to the output.
        Other parameters as in :class:`~bayespecon.models.base.SpatialModel`.

        Notes
        -----
        The log-likelihood for the SDEM model is:

        .. math::
            \\log p(y \\mid \\theta) =
            \\sum_{i=1}^{n} \\log \\mathcal{N}(\\varepsilon_i \\mid 0, \\sigma^2)
            + \\log |I - \\lambda W |

        where :math:`\\varepsilon = (I - \\lambda W)(y - Z\\beta)` and
        :math:`Z = [X, WX]`.

        Because the SDEM model uses ``pm.Potential`` for both the Gaussian
        error log-likelihood and the Jacobian, neither term is auto-captured
        in the ``log_likelihood`` group by PyMC.  We compute the complete
        pointwise log-likelihood manually after sampling:

        .. math::
            \\ell_i = -\\frac{1}{2}\\left(\\frac{\\varepsilon_i}{\\sigma}\\right)^2
            - \\log(\\sigma) - \\frac{1}{2}\\log(2\\pi)
            + \\frac{1}{n} \\log |I - \\lambda W |
        """
        from ._sampler import (
            prepare_compile_kwargs,
            prepare_idata_kwargs,
        )

        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")

        model = self._build_pymc_model()
        self._pymc_model = model
        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        sample_kwargs = prepare_compile_kwargs(sample_kwargs, nuts_sampler)
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs=idata_kwargs,
                nuts_sampler=nuts_sampler,
                **sample_kwargs,
            )

        # --- Compute complete pointwise log-likelihood ---
        # SDEM uses pm.Potential for both Gaussian and Jacobian terms,
        # so nothing is auto-captured. We recompute from posterior draws.
        if compute_log_likelihood:
            idata = self._idata
            n = self._y.shape[0]
            Z = np.hstack([self._X, self._WX])  # (n, 2k)
            W = self._W_dense

            lam_draws = idata.posterior["lam"].values.reshape(-1)  # (n_draws,)
            beta_draws = idata.posterior["beta"].values.reshape(
                -1, Z.shape[1]
            )  # (n_draws, 2k)
            sigma_draws = idata.posterior["sigma"].values.reshape(-1)  # (n_draws,)

            # Residuals: resid = y - Z @ beta.T  => (n_draws, n)
            resid = self._y[None, :] - (beta_draws @ Z.T)  # (n_draws, n)

            # Spatially filtered residuals: eps = resid - lam * W @ resid
            eps = resid - lam_draws[:, None] * (resid @ W.T)  # (n_draws, n)

            # Pointwise log-likelihood for eps
            if self.robust:
                nu_draws = idata.posterior["nu"].values.reshape(-1)  # (n_draws,)
                from scipy.special import gammaln

                ll_gauss = (
                    gammaln((nu_draws[:, None] + 1) / 2)
                    - gammaln(nu_draws[:, None] / 2)
                    - 0.5 * np.log(nu_draws[:, None] * np.pi)
                    - np.log(sigma_draws[:, None])
                    - ((nu_draws[:, None] + 1) / 2)
                    * np.log1p((eps / sigma_draws[:, None]) ** 2 / nu_draws[:, None])
                )  # (n_draws, n)
            else:
                ll_gauss = (
                    -0.5 * (eps / sigma_draws[:, None]) ** 2
                    - np.log(sigma_draws[:, None])
                    - 0.5 * np.log(2 * np.pi)
                )  # (n_draws, n)

            # Jacobian contribution per draw: log|I - lam*W| / n (respects logdet_method)
            jacobian = self._logdet_numpy_vec_fn(lam_draws)  # (n_draws,)
            ll_jac = jacobian[:, None] / n  # (n_draws, 1) broadcast to (n_draws, n)

            ll_total = ll_gauss + ll_jac  # (n_draws, n)

            # Reshape to (chains, draws, n)
            n_chains = idata.posterior.sizes["chain"]
            n_draws_per_chain = idata.posterior.sizes["draw"]
            ll_array = ll_total.reshape(n_chains, n_draws_per_chain, n)

            # Attach to idata — use explicit Dataset creation to ensure
            # "obs" is a data variable, not a coordinate.
            ll_da = xr.DataArray(
                ll_array, dims=("chain", "draw", "obs_dim"), name="obs"
            )
            ll_ds = xr.Dataset({"obs": ll_da})
            idata["log_likelihood"] = ll_ds

        return self._idata

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SDEM regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        if not self._wx_column_indices:
            raise ValueError(
                "SDEM requires at least one WX column. Pass `w_vars=[...]` to "
                "choose which regressors receive a spatial lag, or fit a SEM "
                "model instead."
            )
        Z = np.hstack([self._X, self._WX])  # (n, 2k)

        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = self._logdet_pytensor_fn

        # Precompute W @ Z so the spatial filter can be expressed as
        #   eps = (y - lam*Wy) - (Z - lam*WZ)@beta
        # avoiding any sparse matvec inside the NUTS gradient loop.
        if not hasattr(self, "_WZ_sdem_cache") or self._WZ_sdem_cache is None:
            self._WZ_sdem_cache = np.asarray(self._W_sparse @ Z, dtype=np.float64)
        WZ = self._WZ_sdem_cache

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_star = self._y - lam * self._Wy
            Z_star = Z - lam * WZ
            eps = y_star - pt.dot(Z_star, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_eps = pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps)
            else:
                logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps)
            pm.Potential("eps_loglik", logp_eps.sum())
            pm.Potential("jacobian", logdet_fn(lam))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SDEM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # For SDEM (no y-lag), impacts match SLX form:
        # S_k = d y / d X_k = beta1_k * I + beta2_k * W
        # Direct   = mean(diag(S_k))
        # Total    = mean(row_sums(S_k))
        # Indirect = Total - Direct
        beta = self._posterior_mean("beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1, beta2 = beta[:k], beta[k : k + kw]
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

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SDEM model (no :math:`\\rho` on :math:`y`), the impact
        measures are identical in form to SLX:

        .. math::
            S_k^{(g)} = \\beta_{1j}^{(g)} I + \\beta_{2k}^{(g)} W

        The spatial error parameter :math:`\\lambda` does not affect the
        partial derivatives of :math:`y` with respect to :math:`X`.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each
            of shape ``(G, k_wx)``.
        """
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k+k_wx)
        beta_draws.shape[0]
        k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1_draws = beta_draws[:, :k]  # (G, k)
        beta2_draws = beta_draws[:, k : k + kw]  # (G, kw)

        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        wx_idx = self._wx_column_indices
        direct_samples = beta1_draws[:, wx_idx] + mean_diag_w * beta2_draws  # (G, kw)
        total_samples = beta1_draws[:, wx_idx] + mean_row_sum_w * beta2_draws  # (G, kw)
        indirect_samples = total_samples - direct_samples  # (G, kw)

        return direct_samples, indirect_samples, total_samples

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
