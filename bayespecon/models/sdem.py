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

from ..diagnostics.lmtests import SDEM_SUITE
from ._sampler import (
    prepare_compile_kwargs,
    prepare_idata_kwargs,
    use_jax_likelihood,
)
from .base import (
    SpatialModel,
    _pointwise_gaussian_loglik,
    _write_log_likelihood_to_idata,
)
from .priors import SDEMPriors


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
        - ``sigma2_alpha`` (float, default 2.0): Shape of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): Scale of the
          InverseGamma prior on :math:`\\sigma^2`.
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

    _priors_cls = SDEMPriors

    _spatial_diagnostics_tests = SDEM_SUITE.tests

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        idata_kwargs: Optional[dict] = None,
        sampler: str = "gibbs",
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        **sample_kwargs,
    ) -> "az.InferenceData":
        """Draw samples from the posterior.

        Parameters
        ----------
        draws : int, default 2000
            Number of posterior samples per chain (after tuning).
        tune : int, default 1000
            Number of tuning (burn-in) steps per chain.
        chains : int, default 4
            Number of parallel chains.
        target_accept : float, default 0.9
            Target acceptance rate for NUTS.
        random_seed : int, optional
            Seed for reproducibility.
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` for InferenceData creation. If contains
            ``log_likelihood: True``, the complete pointwise log-likelihood
            (including the Jacobian correction) is attached to the output.
            Only used when ``sampler="nuts"``.
        sampler : str, default "nuts"
            Sampling method:

            - ``"nuts"``: NUTS via PyMC (default).
            - ``"gibbs"``: 3-block Gibbs sampler (β conjugate normal,
              σ² conjugate Inv-Γ, λ conditional slice).  The design
              matrix is Z = [X, WX] and β covers both direct and
              indirect coefficients.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup.  Only used when
            ``sampler="gibbs"``.
        n_jobs : int, default -1
            Number of parallel workers for Gibbs chains.  ``-1`` uses
            all CPUs.  When ``n_jobs=1``, chains run sequentially with
            progress bars.  When ``n_jobs>1`` (or ``-1``), chains run
            in parallel via ``joblib``.  Only used when
            ``sampler="gibbs"`` with ``gibbs_method="numpy"``.
        progressbar : bool, default True
            Show per-chain progress bars.  Only used when
            ``sampler="gibbs"``.
        **sample_kwargs
            Additional keyword arguments forwarded to ``pm.sample``.
            Only used when ``sampler="nuts"``.

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
        error log-likelihood and the Jacobian on the default (C / Numba)
        backend, neither term is auto-captured in the ``log_likelihood``
        group by PyMC.  We compute the complete pointwise log-likelihood
        manually after sampling:

        .. math::
            \\ell_i = -\\frac{1}{2}\\left(\\frac{\\varepsilon_i}{\\sigma}\\right)^2
            - \\log(\\sigma) - \\frac{1}{2}\\log(2\\pi)
            + \\frac{1}{n} \\log |I - \\lambda W |

        On JAX backends (``nuts_sampler="numpyro"`` or ``"blackjax"``) the
        same per-observation density is registered via :class:`pymc.CustomDist`
        so PyMC populates ``log_likelihood`` natively.
        """
        if sampler == "gibbs":
            return self._fit_gibbs(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                thin=thin,
                n_jobs=n_jobs,
                progressbar=progressbar,
                gibbs_method=sample_kwargs.pop("gibbs_method", "numpy"),
                mala_step_size=sample_kwargs.pop("mala_step_size", 0.05),
                use_mala=sample_kwargs.pop("use_mala", True),
                use_slice=sample_kwargs.pop("use_slice", True),
                slice_width=sample_kwargs.pop("slice_width", None),
                chain_method=sample_kwargs.pop("chain_method", None),
            )
        elif sampler != "nuts":
            raise ValueError(f"sampler must be 'nuts' or 'gibbs', got '{sampler}'")

        # --- NUTS path (default) ---
        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")

        model = self._build_pymc_model(nuts_sampler=nuts_sampler)
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
                progressbar=progressbar,
                **sample_kwargs,
            )

        # --- Compute complete pointwise log-likelihood ---
        # On the default (pymc/numba) backend SDEM uses pm.Potential for both
        # Gaussian and Jacobian terms, so nothing is auto-captured.  On JAX
        # backends the model is built via pm.CustomDist with an observed RV,
        # so PyMC has already populated ``log_likelihood`` natively.
        needs_manual_loglik = compute_log_likelihood and not use_jax_likelihood(
            nuts_sampler
        )
        if needs_manual_loglik:
            idata = self._idata
            n = self._y.shape[0]
            Z = np.hstack([self._X, self._WX])  # (n, 2k)
            W = self._W_dense

            lam_draws = idata.posterior["lam"].values.reshape(-1)  # (n_draws,)
            beta_draws = idata.posterior["beta"].values.reshape(
                -1, Z.shape[1]
            )  # (n_draws, 2k)
            sigma_draws = idata.posterior["sigma"].values.reshape(-1)  # (n_draws,)
            nu_draws = idata.posterior["nu"].values.reshape(-1) if self.robust else None

            resid = self._y[None, :] - (beta_draws @ Z.T)  # (n_draws, n)
            eps = resid - lam_draws[:, None] * (resid @ W.T)  # (n_draws, n)

            ll_gauss = _pointwise_gaussian_loglik(eps, sigma_draws, nu_draws)
            jacobian = self._logdet_numpy_vec_fn(lam_draws)  # (n_draws,)
            ll_total = ll_gauss + jacobian[:, None] / n  # (n_draws, n)

            n_chains = idata.posterior.sizes["chain"]
            n_draws_per_chain = idata.posterior.sizes["draw"]
            _write_log_likelihood_to_idata(
                idata, ll_total.reshape(n_chains, n_draws_per_chain, n)
            )

        return self._idata

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        gibbs_method: str = "numpy",
        mala_step_size: float = 0.05,
        use_mala: bool = True,
        use_slice: bool = True,
        slice_width: float | None = None,
        chain_method: str | None = None,
    ) -> "az.InferenceData":
        """Sample posterior via 3-block Gaussian Gibbs.

        The SDEM model is equivalent to SEM with Z = [X, WX] as the
        design matrix.  The β block covers both direct and indirect
        coefficients.

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup (burn-in) draws per chain.
        chains : int, default 4
            Number of independent chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup.
        n_jobs : int, default -1
            Number of parallel workers for the NumPy path. ``-1`` uses
            all CPUs.  When ``n_jobs=1``, chains run sequentially with
            progress bars.  When ``n_jobs>1`` (or ``-1``), chains run
            in parallel via ``joblib``.  Ignored for the JAX path
            (use ``chain_method`` instead).
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "numpy"
            Execution backend: ``"numpy"`` for Python-loop Gibbs with
            adaptive slice sampling, or ``"jax"`` for full-JIT Gibbs
            with MALA for λ.  The JAX path requires JAX and equinox.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
            Ignored when ``use_slice=True``.
        use_mala : bool, default True
            If True, use MALA for the λ update in the JAX path.
            Ignored when ``use_slice=True``.
        use_slice : bool, default True
            If True, use slice sampling for the λ update in the
            JAX path.  Slice sampling gives much better ESS per sample
            than MALA.  Ignored when ``gibbs_method="numpy"``.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.  If None, defaults
            to ``(rho_upper - rho_lower) * 0.1``.  Ignored when
            ``use_slice=False`` or ``gibbs_method="numpy"``.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.
            ``"vectorized"`` uses ``jax.vmap`` for JAX-native
            parallelism (all chains on one device).  ``"sequential"``
            runs chains one after another with progress bars.
            ``"parallel"`` is not supported for the JAX path.
            If None, defaults to ``"vectorized"`` when
            ``gibbs_method="jax"``.  Ignored for the NumPy path
            (use ``n_jobs`` to control parallelism instead).

        Returns
        -------
        az.InferenceData
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.

        Raises
        ------
        NotImplementedError
            If the model uses a robust (Student-t) likelihood.
        """
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        from .._samplers._gaussian_gibbs import GaussianGibbsPriors
        from .._samplers._gibbs_estimation import GaussianSEMGibbs

        Z = np.hstack([self._X, self._WX])  # (n, 2k)
        feature_names = list(self._feature_names) + [
            f"W*{name}" for name in self._wx_feature_names
        ]
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            Z, feature_names
        )

        priors = GaussianGibbsPriors(
            beta_mu=self.priors.get("beta_mu", default_beta_mu),
            beta_sigma=self.priors.get("beta_sigma", default_beta_sigma),
            sigma2_alpha=self.priors.get("sigma2_alpha", 2.0),
            sigma2_beta=self.priors.get("sigma2_beta", float(np.var(self._y))),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        gibbs = GaussianSEMGibbs(
            y=self._y,
            X=Z,
            W_sparse=self._W_sparse,
            priors=priors,
            logdet_fn=self._logdet_numpy_fn,
            logdet_vec_fn=self._logdet_numpy_vec_fn,
            feature_names=feature_names,
            model_type="sdem",
            W_eigs=self._W_eigs.real.astype(np.float64)
            if self._resolved_logdet_method == "eigenvalue"
            else None,
            logdet_method=self.logdet_method,
        )

        self._idata = gibbs.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            n_jobs=n_jobs,
            progressbar=progressbar,
            gibbs_method=gibbs_method,
            mala_step_size=mala_step_size,
            use_mala=use_mala,
            use_slice=use_slice,
            slice_width=slice_width,
            chain_method=chain_method,
        )
        return self._idata

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        """Construct the PyMC model for SDEM regression.

        Parameters
        ----------
        nuts_sampler :
            Resolved sampler name (``"pymc"``, ``"blackjax"``, ``"numpyro"``,
            ``"nutpie"``).  When the sampler is JAX-backed (``"blackjax"`` /
            ``"numpyro"``), the likelihood is registered via
            :class:`pymc.CustomDist` with an observed RV so PyMC's JAX path
            can capture ``log_likelihood`` natively.  Otherwise the
            (benchmarked) :func:`pymc.Potential` formulation is used.

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
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            Z,
            list(self._feature_names)
            + [f"W*{name}" for name in self._wx_feature_names],
        )
        beta_mu = self.priors.get("beta_mu", default_beta_mu)
        beta_sigma = self.priors.get("beta_sigma", default_beta_sigma)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y)))

        logdet_fn = self._logdet_pytensor_fn

        # Precompute W @ Z so the spatial filter can be expressed as
        #   eps = (y - lam*Wy) - (Z - lam*WZ)@beta
        # avoiding any sparse matvec inside the NUTS gradient loop.
        if not hasattr(self, "_WZ_sdem_cache") or self._WZ_sdem_cache is None:
            self._WZ_sdem_cache = np.asarray(self._W_sparse @ Z, dtype=np.float64)
        WZ = self._WZ_sdem_cache

        n_obs = int(self._y.shape[0])
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            if self.robust:
                self._add_nu_prior(model)

            if jax_logp:
                # JAX path: register an observed RV via pm.CustomDist so PyMC
                # can capture ``log_likelihood`` natively.  The Jacobian
                # ``log|I - lam*W|`` is a scalar in lam; we distribute it
                # evenly as ``logdet/n`` per observation so the *sum* of the
                # per-point log-likelihood reproduces the joint log-density
                # (matches the manual NumPy fallback's convention so
                # loo/waic numbers are unchanged across backends).
                Wy_const = pt.as_tensor_variable(self._Wy)
                Z_const = pt.as_tensor_variable(Z)
                WZ_const = pt.as_tensor_variable(WZ)
                inv_n = 1.0 / n_obs

                if self.robust:
                    nu = model["nu"]

                    def sdem_logp(value, lam_, beta_, sigma_, nu_):
                        y_star = value - lam_ * Wy_const
                        Z_star = Z_const - lam_ * WZ_const
                        eps = y_star - pt.dot(Z_star, beta_)
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
                        logp=sdem_logp,
                        observed=self._y,
                    )
                else:

                    def sdem_logp(value, lam_, beta_, sigma_):
                        y_star = value - lam_ * Wy_const
                        Z_star = Z_const - lam_ * WZ_const
                        eps = y_star - pt.dot(Z_star, beta_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        logp=sdem_logp,
                        observed=self._y,
                    )
            else:
                # Default (C / Numba) path: benchmarked pm.Potential
                # formulation.  Log-likelihood is recomputed manually after
                # sampling because pm.Potential terms are not captured by
                # ``compute_log_likelihood``.
                y_star = self._y - lam * self._Wy
                Z_star = Z - lam * WZ
                eps = y_star - pt.dot(Z_star, beta)
                if self.robust:
                    nu = model["nu"]
                    logp_eps = pm.logp(
                        pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps
                    )
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
        from ..diagnostics.lmtests import _get_posterior_draws

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
