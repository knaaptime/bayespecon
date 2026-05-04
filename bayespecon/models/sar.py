"""Spatial Autoregressive (SAR / Spatial Lag) Model.

y = rho * W @ y + X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)

The full likelihood includes the Jacobian |I - rho*W|, added via
pm.Potential so that NUTS samples from the correct posterior.
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


class SAR(SpatialModel):
    """Bayesian Spatial Autoregressive (Spatial Lag) model.

    Models a contemporaneous spatial dependence in the dependent
    variable via the autoregressive parameter :math:`\\rho`:

    .. math::
        y = \\rho Wy + X\\beta + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The likelihood includes the spatial Jacobian :math:`\\log|I - \\rho W|`
    so that posterior inference on :math:`\\rho` is exact.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. An intercept is included by default; suppress with
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
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\\beta`.
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\\sigma`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects ``"eigenvalue"`` for ``n <= 2000`` else
        ``"chebyshev"``. Other options: ``"exact"`` (symbolic det,
        slow for ``n > 500``), ``"dense_grid"``, ``"sparse_grid"``,
        ``"spline"``, ``"mc"``, ``"ilu"``.
    robust : bool, default False
        If True, replace the Normal error with Student-t for robustness
        to heavy-tailed outliers. See *Robust regression* below.
    w_vars : list of str, optional
        Accepted for API consistency with SLX/SDM/SDEM but unused
        (SAR has no ``WX`` term). If supplied without effect on this
        model.

    Notes
    -----
    Direct, indirect and total effects of :math:`X` on :math:`y` are
    derived from the spatial multiplier :math:`(I - \\rho W)^{-1}` and
    are reported by :meth:`spatial_effects`.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)`
    with rate ``nu_lam`` (default 1/30, mean ≈ 30, favouring near-Normal
    tails). The lower bound of 2 ensures the variance exists.
    """

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_error_from_sar_test"],
            ).bayesian_lm_error_from_sar_test(m),
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

    def _build_pymc_model(self, compute_log_likelihood: bool = False) -> pm.Model:
        """Construct the PyMC model for SAR regression.

        Parameters
        ----------
        compute_log_likelihood : bool, default False
            If True, store pointwise log-likelihood (not used in SAR since
            the Jacobian correction is applied post-sampling in ``fit()``).

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        self._X.shape[1]

        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # mu = rho*Wy + X@beta  (Wy is fixed observed data here)
            mu = rho * self._Wy + pt.dot(self._X, beta)
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
        The log-likelihood for the SAR model is:

        .. math::
            \\log p(y \\mid \\theta) =
            \\sum_{i=1}^{n} \\log \\mathcal{N}(y_i \\mid \\mu_i, \\sigma^2)
            + \\log |I - \\rho W |

        The ``pm.Normal`` with ``observed=self._y`` automatically captures
        the first term (the Gaussian log-likelihood) in ``log_likelihood``.
        However, the Jacobian term :math:`\\log |I - \\rho W|` is added via
        ``pm.Potential`` and does **not** appear in the auto-computed
        ``log_likelihood`` group.

        For correct WAIC/LOO computation (and therefore Bayes factor
        comparison via bridge sampling), we construct the complete
        pointwise log-likelihood manually after sampling:

        .. math::
            \\ell_i = -\\frac{1}{2}\\left(\\frac{y_i - \\mu_i}{\\sigma}\\right)^2
            + \\frac{1}{n} \\log |I - \\rho W |

        where :math:`\\mu_i = \\rho (Wy)_i + x_i' \\beta` and the Jacobian
        contribution is divided by :math:`n` so that
        :math:`\\sum_{i=1}^{n} \\ell_i` equals the total log-likelihood
        used for sampling.
        """

        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")

        # Build model with log_likelihood computation if requested
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

            rho_draws = idata.posterior["rho"].values.reshape(-1)  # (n_draws,)
            beta_draws = idata.posterior["beta"].values.reshape(
                -1, self._X.shape[1]
            )  # (n_draws, k)
            sigma_draws = idata.posterior["sigma"].values.reshape(-1)  # (n_draws,)
            nu_draws = idata.posterior["nu"].values.reshape(-1) if self.robust else None

            mu = rho_draws[:, None] * self._Wy[None, :] + (
                beta_draws @ self._X.T
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
        """Compute SAR direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # SAR impacts: S(W) = (I - rho*W)^{-1}
        # Direct = mean diagonal of S * beta_k
        # Indirect = (mean row sum of S - 1) * beta_k
        # Total = mean row sum of S * beta_k
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
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SAR model, the impact measures for covariate :math:`k` are:

        .. math::
            \\text{Direct}_k^{(g)} = \\overline{\\text{diag}}(S^{(g)}) \\times \\beta_k^{(g)}

            \\text{Total}_k^{(g)} = \\overline{\\text{rowsum}}(S^{(g)}) \\times \\beta_k^{(g)}

            \\text{Indirect}_k^{(g)} = \\text{Total}_k^{(g)} - \\text{Direct}_k^{(g)}

        where :math:`S^{(g)} = (I - \\rho^{(g)} W)^{-1}` and the overline
        denotes the average over diagonal elements or row sums.

        The eigenvalue decomposition is used for efficiency:
        :math:`\\overline{\\text{diag}}(S) = \\frac{1}{n} \\sum_i \\frac{1}{1 - \\rho \\omega_i}`
        where :math:`\\omega_i` are eigenvalues of :math:`W`.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each
            of shape ``(G, k)`` where *G* is the total number of posterior
            draws and *k* is the number of covariates.

        References
        ----------
        LeSage, J.P. & Pace, R.K. (2009). *Introduction to Spatial
        Econometrics*. Chapman & Hall/CRC.  Sections 2.7 and 5.6 derive
        the impact decomposition above and motivate the trace-based
        scalar summaries used here.
        """
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws
        from ..diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")  # (G,)
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k)
        rho_draws.shape[0]
        beta_draws.shape[1]

        eigs = self._W_eigs.real.astype(np.float64)  # (n,)

        # For each draw g, compute mean_diag and mean_rowsum of S = (I - rho*W)^{-1}
        # Using eigenvalues: diag(S) has entries 1/(1 - rho*omega_i)
        # mean_diag = (1/n) * sum_i 1/(1 - rho*omega_i)
        # mean_rowsum: if W is row-standardized, mean_rowsum = 1/(1-rho)
        #              otherwise, use eigenvalue decomposition (vectorised)
        # Computed in chunks over draws to bound memory at O(chunk*n) rather
        # than O(G*n).
        mean_diag = _chunked_eig_means(rho_draws, eigs)  # (G,)

        mean_row_sum = self._batch_mean_row_sum(rho_draws)  # (G,)

        # Direct = mean_diag * beta_k, Total = mean_row_sum * beta_k
        # Exclude intercept from effects (it has no meaningful spatial interpretation)
        ni = self._nonintercept_indices
        direct_samples = mean_diag[:, None] * beta_draws[:, ni]  # (G, k-1)
        total_samples = mean_row_sum[:, None] * beta_draws[:, ni]  # (G, k-1)
        indirect_samples = total_samples - direct_samples  # (G, k-1)

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
        return rho * self._Wy + self._X @ beta
