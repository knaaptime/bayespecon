"""Bayesian spatial panel models with unit random effects.

Analogues of the legacy ``prandom`` (non-spatial GLS random effects) and the
LeSage/Pace spatial panel routines, cast as hierarchical Bayesian models.

Model structure for all three classes
--------------------------------------
.. math::
    y_{it} = \\mu_{it} + \\alpha_i + \\varepsilon_{it}

where :math:`\\mu_{it}` is the spatial or non-spatial mean depending on the model.

    \\alpha_i \\sim N(0, \\sigma_\\alpha^2), \\quad
    \\varepsilon_{it} \\sim N(0, \\sigma^2)

Data convention
---------------
Observations must be stacked time-first (time period changes slowest),
so that observation index ``i`` belongs to unit ``i % N``.  This matches
the convention used by all other panel classes in this package.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import sparse as pts

from ..._backends.sampler_helpers import use_jax_likelihood
from ..panel_base import SpatialPanelModel
from ..priors import (
    PanelOLSREPriors,
    PanelSARREPriors,
    PanelSDEMREPriors,
    PanelSEMREPriors,
)


class OLSPanelRE(SpatialPanelModel):
    """Bayesian random effects panel regression (non-spatial).

    .. math::
        y_{it} = X_{it}\\beta + \\alpha_i + \\varepsilon_{it}

    where :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)` are unit-level
    random effects and :math:`\\varepsilon_{it} \\sim N(0, \\sigma^2)`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Accepts a
        :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
        matrix; legacy ``libpysal.weights.W`` is not accepted (use
        ``w.sparse``). Should be row-standardised. Unused in the RE
        likelihood but required by the base class for consistency
        (e.g. computing spatial lags for SDM/SDEM variants).
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_sigma`` (float, default 10.0): HalfNormal
          prior std for :math:`\\sigma_\\alpha`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    Data are **not** demeaned — the random effects absorb the unit-level
    mean structure probabilistically.  This is the Bayesian analogue of
    the classical GLS random-effects estimator in ``prandom.m``.

    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    _priors_cls = PanelOLSREPriors

    def __init__(self, **kwargs):
        kwargs.pop("model", None)  # RE always uses raw (pooled) data
        kwargs["effects"] = 0  # pooled — no FE transform
        super().__init__(**kwargs)
        # obs i → unit i % N  (time-first stacking)
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for non-spatial random effects panel.

        Returns
        -------
        pymc.Model
        """
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y)))
        sigma_alpha_sigma = self.priors.get("sigma_alpha_sigma", 10.0)

        unit_idx = self._unit_idx

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=sigma_alpha_sigma)
            alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")

            mu = pt.dot(self._X, beta) + alpha[unit_idx]
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values including unit random effects.

        Returns
        -------
        np.ndarray
        """
        beta = self._posterior_mean("beta")
        alpha = self._posterior_mean("alpha")
        return self._X @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects(self) -> dict:
        """Direct/indirect/total effects (no spatial multiplier).

        Returns
        -------
        dict
        """
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
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelRE):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            eigs = self._W_eigs
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        elif isinstance(self, OLSPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples


class SARPanelRE(SpatialPanelModel):
    """Bayesian spatial lag panel model with unit random effects.

    .. math::
        y_{it} = \\rho (Wy)_{it} + X_{it}\\beta + \\alpha_i + \\varepsilon_{it}

    where :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)` are unit-level
    random effects and :math:`\\varepsilon_{it} \\sim N(0, \\sigma^2)`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
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
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_sigma`` (float, default 10.0): HalfNormal
          prior std for :math:`\\sigma_\\alpha`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`; auto-selected
        (``"eigenvalue"`` for ``N <= 2000`` else ``"chebyshev"``) when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    _priors_cls = PanelSARREPriors

    def __init__(self, **kwargs):
        kwargs.pop("model", None)
        kwargs["effects"] = 0  # pooled
        super().__init__(**kwargs)
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SAR panel with random effects.

        Returns
        -------
        pymc.Model
        """
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y)))
        sigma_alpha_sigma = self.priors.get("sigma_alpha_sigma", 10.0)

        logdet_fn = self._logdet_pytensor_fn
        unit_idx = self._unit_idx

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=sigma_alpha_sigma)
            alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")

            mu = rho * self._Wy + pt.dot(self._X, beta) + alpha[unit_idx]
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
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
        """Sample posterior via 5-block RE Gibbs (β, σ², α, σ_α², ρ).

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
            Number of parallel workers. ``-1`` uses all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.

        Returns
        -------
        az.InferenceData
        """
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        from ...samplers.panel import GaussianSARREGibbs, REGibbsPriors

        priors = REGibbsPriors(
            beta_mu=self.priors.get("beta_mu", 0.0),
            beta_sigma=self.priors.get("beta_sigma", 1e6),
            sigma_alpha_sigma=self.priors.get("sigma_alpha_sigma", 10.0),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        gibbs = GaussianSARREGibbs(
            y=self._y,
            X=self._X,
            W_sparse=self._W_sparse_NT,
            Wy=self._Wy,
            priors=priors,
            logdet_fn=self._logdet_numpy_fn,
            logdet_vec_fn=self._logdet_numpy_vec_fn,
            feature_names=list(self._feature_names),
            N=self._N,
            T=self._T,
            unit_idx=self._unit_idx,
            W_eigs=(
                self._W_eigs if self._resolved_logdet_method == "eigenvalue" else None
            ),
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
        )
        return self._idata

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        sampler: str = "gibbs",
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        **sample_kwargs,
    ):
        """Sample posterior for SAR panel RE model.

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup draws per chain (NUTS) or burn-in draws
            (Gibbs).
        chains : int, default 4
            Number of independent chains.
        target_accept : float, default 0.9
            NUTS target acceptance probability. Ignored for Gibbs.
        random_seed : int or None
            Seed for reproducibility.
        idata_kwargs : dict or None
            Extra kwargs for InferenceData (NUTS only).
        sampler : str, default "gibbs"
            Sampler to use: ``"gibbs"`` for 5-block Gibbs or ``"nuts"``
            for PyMC NUTS.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup (Gibbs only).
        n_jobs : int, default -1
            Number of parallel workers (Gibbs only).
        progressbar : bool, default True
            Show per-chain progress bars (Gibbs only).
        **sample_kwargs
            Extra keyword arguments forwarded to PyMC (NUTS only).

        Returns
        -------
        az.InferenceData
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
            self._reconstruct_panel_log_likelihood(
                spatial_param="rho",
                nuts_sampler=nuts_sampler,
                T_eff=self._T,
            )

        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values.

        Returns
        -------
        np.ndarray
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        alpha = self._posterior_mean("alpha")
        return rho * self._Wy + self._X @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects(self) -> dict:
        """SAR direct/indirect/total effects at posterior mean rho.

        Returns
        -------
        dict
        """
        ni = self._nonintercept_indices
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        mean_row_sum = float(self._batch_mean_row_sum(np.array([rho]))[0])
        direct = mean_diag * beta[ni]
        total = mean_row_sum * beta[ni]
        return {
            "direct": direct,
            "indirect": total - direct,
            "total": total,
            "feature_names": self._nonintercept_feature_names,
        }

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelRE):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            eigs = self._W_eigs
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        elif isinstance(self, OLSPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples


class SEMPanelRE(SpatialPanelModel):
    """Bayesian spatial error panel model with unit random effects.

    .. math::
        y_{it} = X_{it}\\beta + \\alpha_i + u_{it}, \\quad
        u_{it} = \\lambda (Wu)_{it} + \\varepsilon_{it}

    Equivalently the spatially-filtered residual is i.i.d.:

    .. math::
        \\varepsilon_{it} = (I - \\lambda W)(y - X\\beta - \\alpha)_{it}
        \\sim N(0, \\sigma^2)

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
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
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_sigma`` (float, default 10.0): HalfNormal
          prior std for :math:`\\sigma_\\alpha`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.

    mundlak : bool, default False
        If True, use the Mundlak (1978) correlated random effects
        specification.  This augments the design matrix with unit-level
        time-averages of the regressors, modelling the correlation
        between :math:`\\alpha_i` and :math:`X` explicitly.  See
        *Mundlak / Correlated Random Effects* below for details.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.

    **Identification of λ in SEM-RE models**

    The spatial error parameter :math:`\\lambda` is **weakly identified**
    when random effects :math:`\\alpha_i` are present.  The random effects
    absorb spatial correlation across units, making it difficult for the
    data to distinguish between :math:`\\lambda` (spatial error dependence)
    and :math:`\\sigma_\\alpha^2` (between-unit variance).  Both Gibbs and
    NUTS samplers will tend to estimate :math:`\\lambda` near zero even when
    the true value is moderate, because the posterior genuinely
    concentrates there.  This is a model identification issue, not a
    sampler bug.

    Possible remedies include:

    - Use fixed effects (``SEMPanelFE``) instead of random effects
    - Use a Spatial Durbin model (``SDMPanelRE``) that includes WX terms
    - Use longer panels (:math:`T \\to \\infty`) which provide more
      information to separate :math:`\\lambda` from :math:`\\alpha`
    - Use the Mundlak specification (``mundlak=True``) to test for
      RE-regressor correlation, though note this does not resolve
      the :math:`\\lambda` identification issue itself

    **Mundlak / Correlated Random Effects**

    The Mundlak (1978) approach models the correlation between
    :math:`\\alpha_i` and the regressors by decomposing the random effect:

    .. math::

        \\alpha_i = \\bar{X}_i \\gamma + \\eta_i, \\quad \\eta_i \\sim N(0, \\sigma_\\eta^2)

    where :math:`\\bar{X}_i = T^{-1} \\sum_t X_{it}` are the unit-level
    means of the time-varying regressors.  Substituting into the model
    yields an augmented regression with :math:`[X, \\bar{X}]` as
    regressors, where :math:`\\gamma` is estimated alongside
    :math:`\\beta` and the residual random effect :math:`\\eta_i`
    captures only orthogonal unit heterogeneity.

    **Important**: The Mundlak specification addresses RE-regressor
    correlation but does **not** resolve the :math:`\\lambda`
    identification issue described above.  Even with Mundlak
    augmentation, :math:`\\lambda` remains weakly identified because
    :math:`\\eta_i` can still absorb spatial correlation.  The Mundlak
    approach is primarily useful for:

    - Testing whether :math:`\\alpha_i` is correlated with regressors
      (LR test of :math:`\\gamma = 0`)
    - Obtaining consistent :math:`\\beta` estimates when RE are
      correlated with :math:`X`
    - Reducing :math:`\\sigma_\\alpha^2` by absorbing the explained
      between-unit variation into :math:`\\gamma`

    Following Baltagi (2023), the Mundlak approach does *not* yield
    the same estimates as fixed effects for spatial models (unlike
    the non-spatial case), but MLE/Gibbs estimation remains valid.

    **Robust regression**

    When ``robust=True``, the spatially-filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} = (I - \\lambda W)(y - X\\beta - \\alpha_i) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    _priors_cls = PanelSEMREPriors

    def __init__(self, mundlak: bool = False, **kwargs):
        kwargs.pop("model", None)
        kwargs["effects"] = 0  # pooled
        super().__init__(**kwargs)
        self._unit_idx = np.arange(self._N * self._T) % self._N
        self._mundlak = mundlak

        if mundlak:
            self._build_mundlak_augmentation()

    def _build_mundlak_augmentation(self):
        """Compute unit-level means of X and augment the design matrix.

        The Mundlak (1978) approach models correlated random effects as:

            α_i = X̄_i γ + η_i

        where X̄_i = T⁻¹ Σ_t X_{it} are unit-level time averages.
        Substituting into the model yields an augmented regression with
        [X, X̄_expanded] as regressors, where γ is estimated alongside β
        and the residual random effect η_i captures only orthogonal
        unit heterogeneity.

        This addresses RE-regressor correlation but does NOT resolve
        the α-λ identification issue in SEM-RE models — η_i can still
        absorb spatial correlation.  The Mundlak approach is primarily
        useful for testing RE-regressor correlation (LR test of γ=0)
        and obtaining consistent β estimates when RE are correlated
        with X.

        Following Baltagi (2023), the Mundlak approach does *not* yield
        the same estimates as fixed effects for spatial models (unlike
        the non-spatial case), but MLE/Gibbs estimation remains valid.

        Note: Constant/intercept columns are excluded from the Mundlak
        means because their unit-level averages are collinear with the
        original intercept.
        """
        X = self._X
        N, _T = self._N, self._T
        unit_idx = self._unit_idx

        # Identify non-constant columns for Mundlak means
        # Constant columns have unit means equal to the constant itself,
        # creating perfect collinearity with the original intercept.
        nonconst_idx = self._nonintercept_indices
        if len(nonconst_idx) == 0:
            # No time-varying regressors — Mundlak has nothing to add
            return

        # Compute unit-level means for non-constant columns only
        counts = np.bincount(unit_idx, minlength=N)
        X_bar = np.zeros((N, len(nonconst_idx)))
        for j_idx, j in enumerate(nonconst_idx):
            X_bar[:, j_idx] = (
                np.bincount(unit_idx, weights=X[:, j], minlength=N) / counts
            )

        # Expand to observation level: repeat each unit's means T times
        X_bar_expanded = X_bar[unit_idx]  # shape (NT, len(nonconst_idx))

        # Store original X and feature names for reference
        self._X_original = X.copy()
        self._feature_names_original = list(self._feature_names)

        # Augment X: [X, X̄_expanded]
        self._X = np.column_stack([X, X_bar_expanded])

        # Augment feature names (only for non-constant columns)
        mundlak_names = [
            f"mundlak_{self._feature_names_original[j]}" for j in nonconst_idx
        ]
        self._feature_names = list(self._feature_names_original) + mundlak_names

    @property
    def mundlak(self) -> bool:
        """Whether the Mundlak correlated RE specification is active."""
        return self._mundlak

    @property
    def mundlak_names(self) -> list[str] | None:
        """Names of the Mundlak augmentation columns, or None if inactive."""
        if not self._mundlak:
            return None
        k_orig = len(self._feature_names_original)
        return list(self._feature_names[k_orig:])

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        """Construct the PyMC model for SEM panel with random effects.

        Parameters
        ----------
        nuts_sampler :
            Resolved sampler.  When JAX-backed, the likelihood is registered
            via :class:`pymc.CustomDist` so PyMC's JAX path captures
            ``log_likelihood`` natively; otherwise the
            :func:`pymc.Potential` formulation is used.

        Returns
        -------
        pymc.Model
        """
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y)))
        sigma_alpha_sigma = self.priors.get("sigma_alpha_sigma", 10.0)

        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        unit_idx = self._unit_idx

        n_obs = int(self._y.shape[0])
        inv_n = 1.0 / n_obs  # _logdet_pytensor_fn already includes T multiplier
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=sigma_alpha_sigma)
            alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")

            if self.robust:
                self._add_nu_prior(model)

            if jax_logp:
                X_const = pt.as_tensor_variable(self._X)
                y_const = pt.as_tensor_variable(self._y)
                unit_idx_const = pt.as_tensor_variable(unit_idx)

                def _eps(lam_, beta_, alpha_):
                    resid = y_const - pt.dot(X_const, beta_) - alpha_[unit_idx_const]
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = model["nu"]

                    def sempanel_re_logp(value, lam_, beta_, sigma_, alpha_, nu_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        nu,
                        logp=sempanel_re_logp,
                        observed=self._y,
                    )
                else:

                    def sempanel_re_logp(value, lam_, beta_, sigma_, alpha_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        logp=sempanel_re_logp,
                        observed=self._y,
                    )
            else:
                # epsilon = (I - lam*W)(y - X@beta - alpha_expanded)
                #         = resid - lam * W @ resid
                resid = self._y - pt.dot(self._X, beta) - alpha[unit_idx]
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

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
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
        """Sample posterior via 5-block RE Gibbs (β, σ², α, σ_α², λ).

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
            Number of parallel workers. ``-1`` uses all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.

        Returns
        -------
        az.InferenceData
        """
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        from ...samplers.panel import GaussianSEMREGibbs, REGibbsPriors

        priors = REGibbsPriors(
            beta_mu=self.priors.get("beta_mu", 0.0),
            beta_sigma=self.priors.get("beta_sigma", 1e6),
            sigma_alpha_sigma=self.priors.get("sigma_alpha_sigma", 10.0),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        gibbs = GaussianSEMREGibbs(
            y=self._y,
            X=self._X,
            W_sparse=self._W_sparse_NT,
            priors=priors,
            logdet_fn=self._logdet_numpy_fn,
            logdet_vec_fn=self._logdet_numpy_vec_fn,
            feature_names=list(self._feature_names),
            N=self._N,
            T=self._T,
            unit_idx=self._unit_idx,
            W_eigs=(
                self._W_eigs if self._resolved_logdet_method == "eigenvalue" else None
            ),
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
        )
        return self._idata

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        sampler: str = "gibbs",
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        **sample_kwargs,
    ):
        """Sample posterior for SEM panel RE model.

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup draws per chain (NUTS) or burn-in draws
            (Gibbs).
        chains : int, default 4
            Number of independent chains.
        target_accept : float, default 0.9
            NUTS target acceptance probability. Ignored for Gibbs.
        random_seed : int or None
            Seed for reproducibility.
        idata_kwargs : dict or None
            Extra kwargs for InferenceData (NUTS only).
        sampler : str, default "gibbs"
            Sampler to use: ``"gibbs"`` for 5-block Gibbs or ``"nuts"``
            for PyMC NUTS.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup (Gibbs only).
        n_jobs : int, default -1
            Number of parallel workers (Gibbs only).
        progressbar : bool, default True
            Show per-chain progress bars (Gibbs only).
        **sample_kwargs
            Extra keyword arguments forwarded to PyMC (NUTS only).

        Returns
        -------
        az.InferenceData
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
            self._reconstruct_panel_log_likelihood(
                spatial_param="lam",
                nuts_sampler=nuts_sampler,
                T_eff=self._T,
            )

        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values (on the observed y scale).

        Returns
        -------
        np.ndarray
        """
        beta = self._posterior_mean("beta")
        alpha = self._posterior_mean("alpha")
        return self._X @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects(self) -> dict:
        """SEM direct/indirect/total effects (error model: no y-multiplier).

        Returns
        -------
        dict
        """
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
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelRE):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            eigs = self._W_eigs
            inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])
            mean_diag = np.mean(inv_eigs, axis=1)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        elif isinstance(self, OLSPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples


class SDEMPanelRE(SpatialPanelModel):
    """Bayesian spatial Durbin error panel model with unit random effects.

    .. math::
        y_{it} = X_{it}\\beta + (WX)_{it}\\theta + \\alpha_i + u_{it}, \\quad
        u_{it} = \\lambda (Wu)_{it} + \\varepsilon_{it}

    Combines the SDEM mean structure (covariates plus their spatial lags)
    with random unit effects :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)`
    and a spatially-correlated error term governed by :math:`\\lambda`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Used to construct the
        ``WX`` block and the spatial filter on the disturbance.
        Should be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_sigma`` (float, default 10.0): HalfNormal
          prior std for :math:`\\sigma_\\alpha`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. At least one column must be
        lagged; if no WX columns remain a :class:`ValueError` is
        raised. Pass a subset to restrict which variables receive a
        spatial lag.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.
    """

    _has_wx_in_beta = True

    _priors_cls = PanelSDEMREPriors

    def __init__(self, **kwargs):
        kwargs.pop("model", None)
        kwargs["effects"] = 0  # pooled
        super().__init__(**kwargs)
        if not self._wx_column_indices:
            raise ValueError(
                "SDEMPanelRE requires at least one WX column. Pass "
                "`w_vars=[...]` to choose which regressors receive a spatial "
                "lag, or fit an SEMPanelRE model instead."
            )
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        """Construct the PyMC model for SDEM panel with random effects.

        Parameters
        ----------
        nuts_sampler :
            Resolved sampler.  When JAX-backed, the likelihood is registered
            via :class:`pymc.CustomDist` so PyMC's JAX path captures
            ``log_likelihood`` natively; otherwise the
            :func:`pymc.Potential` formulation is used.
        """
        Z = np.hstack([self._X, self._WX])

        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y)))
        sigma_alpha_sigma = self.priors.get("sigma_alpha_sigma", 10.0)

        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        unit_idx = self._unit_idx

        n_obs = int(self._y.shape[0])
        inv_n = 1.0 / n_obs  # _logdet_pytensor_fn already includes T multiplier
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=sigma_alpha_sigma)
            alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")

            if self.robust:
                self._add_nu_prior(model)

            if jax_logp:
                Z_const = pt.as_tensor_variable(Z)
                y_const = pt.as_tensor_variable(self._y)
                unit_idx_const = pt.as_tensor_variable(unit_idx)

                def _eps(lam_, beta_, alpha_):
                    resid = y_const - pt.dot(Z_const, beta_) - alpha_[unit_idx_const]
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = model["nu"]

                    def sdempanel_re_logp(value, lam_, beta_, sigma_, alpha_, nu_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        nu,
                        logp=sdempanel_re_logp,
                        observed=self._y,
                    )
                else:

                    def sdempanel_re_logp(value, lam_, beta_, sigma_, alpha_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        logp=sdempanel_re_logp,
                        observed=self._y,
                    )
            else:
                resid = self._y - pt.dot(Z, beta) - alpha[unit_idx]
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
        """Sample posterior and attach pointwise log-likelihood for IC metrics."""
        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
        progressbar = sample_kwargs.pop("progressbar", True)

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
            self._reconstruct_panel_log_likelihood(
                spatial_param="lam",
                nuts_sampler=nuts_sampler,
                T_eff=self._T,
            )

        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        beta = self._posterior_mean("beta")
        alpha = self._posterior_mean("alpha")
        Z = np.hstack([self._X, self._WX])
        return Z @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects(self) -> dict:
        """SDEM-style direct/indirect/total effects (no rho multiplier)."""
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
        """Posterior samples of direct/indirect/total effects (SDEM form)."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1_draws = beta_draws[:, :k]
        beta2_draws = beta_draws[:, k : k + kw]

        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        wx_idx = self._wx_column_indices
        direct_samples = beta1_draws[:, wx_idx] + mean_diag_w * beta2_draws
        total_samples = beta1_draws[:, wx_idx] + mean_row_sum_w * beta2_draws
        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples
