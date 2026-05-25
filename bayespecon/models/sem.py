"""Spatial Error Model (SEM).

y = X @ beta + u,  u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Equivalently: (I - lambda*W)(y - X@beta) = epsilon
Likelihood: epsilon ~ N(0, sigma^2 I), plus Jacobian log|I - lambda*W|.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .._backends.sampler_helpers import use_jax_likelihood
from .base import SpatialModel
from .priors import SEMPriors


class SEM(SpatialModel):
    """Bayesian Spatial Error Model.

    Spatial dependence enters through the disturbance via the
    autoregressive parameter :math:`\\lambda`:

    .. math::
        y = X\\beta + u, \\quad u = \\lambda Wu + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The likelihood includes the spatial Jacobian
    :math:`\\log|I - \\lambda W|` so that posterior inference on
    :math:`\\lambda` is exact.

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

        - ``lam_lower`` (float, default -1.0): Lower bound of the
          Uniform prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of the
          Uniform prior on :math:`\\lambda`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\\beta`.
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

    Notes
    -----
    Because spatial dependence enters only through the disturbance,
    direct effects equal :math:`\\beta` and indirect effects are zero.

    **Robust regression**

    When ``robust=True``, the spatially-filtered innovation is
    Student-t:

    .. math::

        \\varepsilon = (I - \\lambda W)(y - X\\beta) \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)`
    with rate ``nu_lam`` (default 1/30, mean ≈ 30). The lower bound of 2
    ensures the variance exists.
    """

    _priors_cls = SEMPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = "lam"
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = "GaussianSEMGibbs"
    _model_type: str = "sem"

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
              σ² conjugate Inv-Γ, λ conditional slice).
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
        The log-likelihood for the SEM model is:

        .. math::
            \\log p(y \\mid \\theta) =
            \\sum_{i=1}^{n} \\log \\mathcal{N}(\\varepsilon_i \\mid 0, \\sigma^2)
            + \\log |I - \\lambda W |

        where :math:`\\varepsilon = (I - \\lambda W)(y - X\\beta)`.

        Because the SEM model uses ``pm.Potential`` for both the Gaussian
        error log-likelihood and the Jacobian, neither term is auto-captured
        in the ``log_likelihood`` group by PyMC.  We compute the complete
        pointwise log-likelihood manually after sampling:

        .. math::
            \\ell_i = -\\frac{1}{2}\\left(\\frac{\\varepsilon_i}{\\sigma}\\right)^2
            - \\log(\\sigma) - \\frac{1}{2}\\log(2\\pi)
            + \\frac{1}{n} \\log |I - \\lambda W |
        """
        if sampler == "gibbs":
            return self._fit_gibbs_dispatch(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                thin=thin,
                n_jobs=n_jobs,
                progressbar=progressbar,
                sample_kwargs=sample_kwargs,
            )
        elif sampler != "nuts":
            raise ValueError(f"sampler must be 'nuts' or 'gibbs', got '{sampler}'")

        # --- NUTS path (default) ---
        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")

        _, compute_log_likelihood = self._fit_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
            idata_kwargs=idata_kwargs,
            compute_log_likelihood=compute_log_likelihood,
            sample_kwargs=sample_kwargs,
        )

        if compute_log_likelihood:
            self._reconstruct_cross_sectional_log_likelihood(nuts_sampler=nuts_sampler)

        return self._idata

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        """Construct the PyMC model for SEM regression.

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
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            self._X, list(self._feature_names)
        )
        beta_mu = self.priors.get("beta_mu", default_beta_mu)
        beta_sigma = self.priors.get("beta_sigma", default_beta_sigma)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y)))

        logdet_fn = self._logdet_pytensor_fn

        # Precompute W @ X once so the spatial filter
        #   eps = (I - lam*W)(y - X@beta) = (y - lam*Wy) - (X - lam*WX_all)@beta
        # avoids any sparse matvec inside the NUTS gradient loop. ``_Wy`` is
        # already cached in :class:`SpatialModel`; ``WX_all`` is materialised
        # here for the full design matrix (vs. ``self._WX`` which only covers
        # ``w_vars`` columns).
        if not hasattr(self, "_WX_all_cache") or self._WX_all_cache is None:
            self._WX_all_cache = np.asarray(self._W_sparse @ self._X, dtype=np.float64)
        WX_all = self._WX_all_cache

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
                # per-point log-likelihood reproduces the model's joint
                # log-density (matches the manual NumPy fallback's
                # convention, so loo/waic numbers are unchanged).
                Wy_const = pt.as_tensor_variable(self._Wy)
                X_const = pt.as_tensor_variable(self._X)
                WX_const = pt.as_tensor_variable(WX_all)
                inv_n = 1.0 / n_obs

                if self.robust:
                    nu = model["nu"]

                    def sem_logp(value, lam_, beta_, sigma_, nu_):
                        y_star = value - lam_ * Wy_const
                        X_star = X_const - lam_ * WX_const
                        eps = y_star - pt.dot(X_star, beta_)
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
                        logp=sem_logp,
                        observed=self._y,
                    )
                else:

                    def sem_logp(value, lam_, beta_, sigma_):
                        y_star = value - lam_ * Wy_const
                        X_star = X_const - lam_ * WX_const
                        eps = y_star - pt.dot(X_star, beta_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        logp=sem_logp,
                        observed=self._y,
                    )
            else:
                # Default (C / Numba) path: the benchmarked pm.Potential
                # formulation.  Log-likelihood is recomputed manually after
                # sampling in fit() because pm.Potential terms are not
                # captured by ``compute_log_likelihood``.
                y_star = self._y - lam * self._Wy
                X_star = self._X - lam * WX_all
                eps = y_star - pt.dot(X_star, beta)
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
        """Compute SEM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # For SEM, spatial multiplier does not apply to X directly.
        # Direct = beta, indirect = 0, total = beta.
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
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SEM model, the spatial multiplier does not apply to :math:`X`
        directly, so:

        .. math::
            \\text{Direct}_k^{(g)} = \\beta_k^{(g)}, \\quad
            \\text{Indirect}_k^{(g)} = 0, \\quad
            \\text{Total}_k^{(g)} = \\beta_k^{(g)}

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each
            of shape ``(G, k)`` where *k* is the number of covariates.
        """
        from ..diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k)

        ni = self._nonintercept_indices
        direct_samples = beta_draws[:, ni].copy()
        indirect_samples = np.zeros_like(direct_samples)
        total_samples = direct_samples.copy()

        return direct_samples, indirect_samples, total_samples

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta
