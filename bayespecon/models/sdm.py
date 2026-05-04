"""Spatial Durbin Model (SDM).

y = rho * W @ y + X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

Combines a spatial lag on y (SAR) with spatially lagged covariates (SLX).
Jacobian log|I - rho*W| is required as in the SAR model.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._sampler import prepare_compile_kwargs, prepare_idata_kwargs
from .base import (
    SpatialModel,
    _pointwise_gaussian_loglik,
    _write_log_likelihood_to_idata,
)


class SDM(SpatialModel):
    """Bayesian Spatial Durbin Model.

    Combines a spatial lag of :math:`y` with spatial lags of the
    regressors :math:`X`:

    .. math::
        y = \\rho Wy + X\\beta + WX\\theta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The sampled coefficient vector stacks the local and lagged-regressor
    blocks as :math:`[\\beta, \\theta]`. The likelihood includes the
    spatial Jacobian :math:`\\log|I - \\rho W|`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. Intercept is included by default; suppress with
        ``"y ~ x - 1"``.
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

        - ``rho_lower`` (float, default -1.0): Lower bound of the
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of the
          Uniform prior on :math:`\\rho`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`[\\beta, \\theta]`.
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\\sigma`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects ``"eigenvalue"`` for ``n <= 2000`` else
        ``"chebyshev"``. Other options: ``"exact"``, ``"dense_grid"``,
        ``"sparse_grid"``, ``"spline"``, ``"mc"``, ``"ilu"``.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See *Robust
        regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``. SDM requires at least one
        WX column; if filtering eliminates all of them a ValueError is
        raised.

    Notes
    -----
    Direct, indirect and total effects of :math:`X` on :math:`y`
    incorporate both the local and lagged-X blocks via the spatial
    multiplier :math:`(I - \\rho W)^{-1}` and are reported by
    :meth:`spatial_effects`.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)`
    with rate ``nu_lam`` (default 1/30, mean ≈ 30).
    """

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_error_sdm_test"],
            ).bayesian_lm_error_sdm_test(m),
            "LM-Error-SDM",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_error_sdm_test"],
            ).bayesian_robust_lm_error_sdm_test(m),
            "Robust-LM-Error-SDM",
        ),
    ]

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self, compute_log_likelihood: bool = False) -> pm.Model:
        """Construct the PyMC model for SDM regression.

        Parameters
        ----------
        compute_log_likelihood : bool, default False
            If True, store pointwise log-likelihood (not used in SDM since
            the Jacobian correction is applied post-sampling in ``fit()``).

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        if not self._wx_column_indices:
            raise ValueError(
                "SDM requires at least one WX column. Pass `w_vars=[...]` to "
                "choose which regressors receive a spatial lag, or fit a SAR "
                "model instead."
            )
        self._X.shape[1]
        Z = np.hstack([self._X, self._WX])  # (n, 2k)

        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho * self._Wy + pt.dot(Z, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

            # Jacobian: log|I - rho*W|  (respects logdet_method via self._logdet_pytensor_fn)
            pm.Potential("jacobian", self._logdet_pytensor_fn(rho))

        return model

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
        """
        Draw samples from the posterior. Accepts ``idata_kwargs`` for ArviZ compatibility.

        Parameters
        ----------
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` for InferenceData creation. If contains
            ``log_likelihood: True``, the complete pointwise log-likelihood
            (including the Jacobian correction) is attached to the output.
        Other parameters as in :class:`~bayespecon.models.base.SpatialModel`.

        Notes
        -----
        The log-likelihood for the SDM model is:

        .. math::
            \\log p(y \\mid \\theta) =
            \\sum_{i=1}^{n} \\log \\mathcal{N}(y_i \\mid \\mu_i, \\sigma^2)
            + \\log |I - \\rho W |

        where :math:`\\mu = \\rho W y + Z \\beta` and
        :math:`Z = [X, WX]`.

        As with the SAR model, ``pm.Normal`` with ``observed`` auto-captures
        the Gaussian part, while the Jacobian :math:`\\log |I - \\rho W|` is
        added via ``pm.Potential`` and is absent from the ``log_likelihood``
        group.  To enable WAIC/LOO and Bayes factor comparison, we correct
        the pointwise log-likelihood after sampling:

        .. math::
            \\ell_i = -\\frac{1}{2}\\left(\\frac{y_i - \\mu_i}{\\sigma}\\right)^2
            + \\frac{1}{n} \\log |I - \\rho W |
        """

        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")

        model = self._build_pymc_model(compute_log_likelihood=compute_log_likelihood)
        self._pymc_model = model
        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
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

        # --- Correct log_likelihood: add Jacobian contribution ---
        # The pm.Normal("obs") auto-captures the Gaussian part, but the
        # Jacobian log|I - rho*W| (added via pm.Potential) is absent.
        # We recompute the complete pointwise LL and overwrite the group.
        if compute_log_likelihood and hasattr(self, "_idata"):
            idata = self._idata
            n = self._y.shape[0]
            Z = np.hstack([self._X, self._WX])  # (n, 2k)

            rho_draws = idata.posterior["rho"].values.reshape(-1)  # (n_draws,)
            beta_draws = idata.posterior["beta"].values.reshape(
                -1, Z.shape[1]
            )  # (n_draws, 2k)
            sigma_draws = idata.posterior["sigma"].values.reshape(-1)  # (n_draws,)
            nu_draws = idata.posterior["nu"].values.reshape(-1) if self.robust else None

            mu = rho_draws[:, None] * self._Wy[None, :] + (
                beta_draws @ Z.T
            )  # (n_draws, n)
            resid = self._y[None, :] - mu  # (n_draws, n)

            ll_gauss = _pointwise_gaussian_loglik(resid, sigma_draws, nu_draws)
            jacobian = self._logdet_numpy_vec_fn(rho_draws)  # (n_draws,)
            ll_total = ll_gauss + jacobian[:, None] / n  # (n_draws, n)

            n_chains = idata.posterior.sizes["chain"]
            n_draws_per_chain = idata.posterior.sizes["draw"]
            _write_log_likelihood_to_idata(
                idata, ll_total.reshape(n_chains, n_draws_per_chain, n)
            )

        return self._idata

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SDM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # SDM impacts: S_k = (I - rho*W)^{-1} (beta1_k*I + beta2_k*W)
        # Direct   = mean diagonal of S_k
        # Total    = mean row sum of S_k
        # Indirect = total - direct
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
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SDM model, the impact measures for covariate :math:`k` are:

        .. math::
            S_k^{(g)} = (I - \\rho^{(g)} W)^{-1}
            (\\beta_{1j}^{(g)} I + \\beta_{2k}^{(g)} W)

            \\text{Direct}_k^{(g)} = \\overline{\\text{diag}}(S_k^{(g)})

            \\text{Total}_k^{(g)} = \\overline{\\text{rowsum}}(S_k^{(g)})

            \\text{Indirect}_k^{(g)} = \\text{Total}_k^{(g)} - \\text{Direct}_k^{(g)}

        where :math:`j` is the index of covariate :math:`k` in :math:`X`,
        and :math:`\\beta, \\theta` are the coefficients on :math:`X` and
        :math:`WX` respectively.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each
            of shape ``(G, k_wx)`` where *G* is the total number of posterior
            draws and *k_wx* is the number of spatially lagged covariates.

        References
        ----------
        LeSage, J.P. & Pace, R.K. (2009). *Introduction to Spatial
        Econometrics*. Chapman & Hall/CRC.  Sections 2.7 and 5.6 derive
        the SDM impact decomposition above; the lagged-covariate term
        :math:`\\theta W` enters because :math:`WX` is included as a
        regressor block in the SDM design.
        """
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws
        from ..diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")  # (G,)
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k+k_wx)
        rho_draws.shape[0]
        k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1_draws = beta_draws[:, :k]  # (G, k)
        beta2_draws = beta_draws[:, k : k + kw]  # (G, kw)

        eigs = self._W_eigs.real.astype(np.float64)  # (n,)

        # Chunk over draws to avoid an O(G*n) intermediate.
        mean_diag_M = _chunked_eig_means(rho_draws, eigs)  # (G,)
        mean_diag_MW = _chunked_eig_means(rho_draws, eigs, weights=eigs)  # (G,)

        mean_row_sum_M = self._batch_mean_row_sum(rho_draws)  # (G,)
        mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)  # (G,)

        # For each lagged covariate k (with index j in X):
        # Direct_k = beta1_j * mean_diag_M + beta2_k * mean_diag_MW
        # Total_k  = beta1_j * mean_row_sum_M + beta2_k * mean_row_sum_MW
        wx_idx = self._wx_column_indices
        direct_samples = (
            mean_diag_M[:, None] * beta1_draws[:, wx_idx]
            + mean_diag_MW[:, None] * beta2_draws
        )  # (G, kw)
        total_samples = (
            mean_row_sum_M[:, None] * beta1_draws[:, wx_idx]
            + mean_row_sum_MW[:, None] * beta2_draws
        )  # (G, kw)
        indirect_samples = total_samples - direct_samples  # (G, kw)

        return direct_samples, indirect_samples, total_samples

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
