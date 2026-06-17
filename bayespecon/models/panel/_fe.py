"""Spatial panel model classes."""
from __future__ import annotations
import arviz as az
import numpy as np
from .._mixins import GaussianLikelihoodMixin
from ..panel_base import SpatialPanelModel
from ..priors import PanelOLSPriors, PanelSARPriors, PanelSDEMPriors, PanelSDMPriors, PanelSEMPriors, PanelSLXPriors

class OLSPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian pooled and fixed-effects linear panel regression.

    Implements the Gaussian panel model

    .. math::

        y_{it} = x_{it}'\\beta + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2),

    where the included effects depend on ``model``: ``0`` pooled,
    ``1`` unit effects, ``2`` time effects, ``3`` two-way effects. The
    within transformation is handled by
    :class:`~bayespecon.models.panel_base.SpatialPanelModel` before the
    likelihood is evaluated.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)`` in unit-major order.
        Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix of shape ``(N*T, k)``. Required in
        matrix mode. DataFrame columns are preserved as feature names.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)`` (preferred) or
        ``(N*T, N*T)`` block-diagonal. Accepted for API consistency
        with the other panel models but does not enter the OLS
        likelihood; required if downstream Bayesian LM diagnostics
        will be run.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`\\beta`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        Accepted for API consistency; unused in OLSPanelFE (no
        spatial Jacobian).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    This is the aspatial baseline for panel LM diagnostics and panel model
    comparison. The spatial weights object ``W`` is accepted for API
    consistency but does not enter the likelihood.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails. The lower bound of 2 ensures the
    variance exists.
    """
    _priors_cls = PanelOLSPriors
    _jacobian_param: str | None = None
    _gibbs_class: str | None = None

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean('beta')
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            return self._X[:, ni] @ beta
        return self._X @ beta

class SARPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial-lag panel regression.

    Implements

    .. math::

        y_{it} = \\rho \\sum_j w_{ij} y_{jt} + x_{it}'\\beta + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2),

    with the same pooled, unit-effect, time-effect, or two-way panel
    transformation selected by ``model`` as in :class:`OLSPanelFE`.

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
        Spatial weights of shape ``(N, N)`` (preferred) or
        ``(N*T, N*T)``. Accepts a :class:`libpysal.graph.Graph` or any
        :class:`scipy.sparse` matrix; legacy ``libpysal.weights.W`` is
        not accepted (use ``w.sparse``). Should be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`\\beta`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects ``"eigenvalue"`` for ``N <= 2000`` else
        ``"chebyshev"``.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    The likelihood combines the Gaussian observation density with the
    spatial Jacobian term associated with :math:`I - \\rho W`.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails. The lower bound of 2 ensures the
    variance exists.
    """
    _priors_cls = PanelSARPriors
    _jacobian_param: str | None = 'rho'
    _gibbs_class: str | None = 'GaussianSARGibbs'

    def _fit_gibbs(self, draws: int=2000, tune: int=1000, chains: int=4, random_seed: int | None=None, thin: int=1, n_jobs: int=-1, progressbar: bool=True, gibbs_method: str='jax', mala_step_size: float=0.05, use_mala: bool=True, use_slice: bool=True, slice_width: float | None=None, chain_method: str | None=None) -> 'az.InferenceData':
        """Sample posterior via 3-block Gaussian Gibbs.

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
            Number of parallel workers for the NumPy path.
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "jax"
            Execution backend: ``"jax"`` for full-JIT Gibbs (default, falls back to ``"numpy"`` when JAX is not installed), or ``"numpy"`` for Python-loop Gibbs.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
        use_mala : bool, default True
            If True, use MALA for the ρ update in the JAX path.
            Ignored when ``use_slice=True``.
        use_slice : bool, default True
            If True, use slice sampling for the ρ/λ update in the
            JAX path.  Slice sampling gives much better ESS per sample
            than MALA.  Ignored when ``gibbs_method="numpy"``.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.  If None, defaults
            to ``(rho_upper - rho_lower) * 0.1``.  Ignored when
            ``use_slice=False`` or ``gibbs_method="numpy"``.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.

        Returns
        -------
        az.InferenceData
        """
        if self.robust:
            raise NotImplementedError("Gibbs sampling is not yet supported for robust (Student-t) models. Use sampler='nuts' (the default).")
        from ...samplers.gaussian import GaussianGibbsPriors, GaussianSARGibbs
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(self._X, list(self._feature_names))
        priors = GaussianGibbsPriors(beta_mu=self.priors.get('beta_mu', default_beta_mu), beta_sigma=self.priors.get('beta_sigma', default_beta_sigma), sigma2_alpha=self.priors.get('sigma2_alpha', 2.0), sigma2_beta=self.priors.get('sigma2_beta', float(np.var(self._y))), rho_lower=self._logdet_bounds.rho_min, rho_upper=self._logdet_bounds.rho_max)
        gibbs = GaussianSARGibbs(y=self._y, X=self._X, W_sparse=self._W_sparse_NT, Wy=self._Wy, priors=priors, logdet_fn=self._logdet_numpy_fn, logdet_vec_fn=self._logdet_numpy_vec_fn, feature_names=list(self._feature_names), model_type='sar', W_eigs=self._W_eigs if self._resolved_logdet_method == 'eigenvalue' else None, logdet_method=self.logdet_method, T=self._T)
        self._idata = gibbs.fit(draws=draws, tune=tune, chains=chains, random_seed=random_seed, thin=thin, n_jobs=n_jobs, progressbar=progressbar, gibbs_method=gibbs_method, mala_step_size=mala_step_size, use_mala=use_mala, use_slice=use_slice, slice_width=slice_width, chain_method=chain_method)
        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        rho = float(self._posterior_mean('rho'))
        beta = self._posterior_mean('beta')
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            return rho * self._Wy + self._X[:, ni] @ beta
        return rho * self._Wy + self._X @ beta

class SEMPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial-error panel regression.

    Implements

    .. math::

        y_{it} = x_{it}'\\beta + \\alpha_i + \\tau_t + u_{it},
        \\qquad u_{it} = \\lambda \\sum_j w_{ij} u_{jt} + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    Spatial dependence enters through the disturbance, while the panel
    transformation selected by ``model`` absorbs pooled, unit, time, or
    two-way effects before likelihood evaluation.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Should
        be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`\\beta`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`. ``None`` (default)
        auto-selects ``"eigenvalue"`` for ``N <= 2000`` else
        ``"chebyshev"``.
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.

    Notes
    -----
    Direct effects equal :math:`\\beta`; indirect effects are zero
    because spatial dependence enters only through the disturbance.

    **Robust regression**

    When ``robust=True``, the spatially filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)\\bigl(y_t - X_t \\beta - \\alpha\\bigr) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails. The lower bound of 2 ensures the
    variance exists.
    """
    _priors_cls = PanelSEMPriors
    _jacobian_param: str | None = 'lam'
    _gibbs_class: str | None = 'GaussianSEMGibbs'

    def _fit_gibbs(self, draws: int=2000, tune: int=1000, chains: int=4, random_seed: int | None=None, thin: int=1, n_jobs: int=-1, progressbar: bool=True, gibbs_method: str='jax', mala_step_size: float=0.05, use_mala: bool=True, use_slice: bool=True, slice_width: float | None=None, chain_method: str | None=None) -> 'az.InferenceData':
        """Sample posterior via 3-block Gaussian Gibbs.

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
            Number of parallel workers for the NumPy path.
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "jax"
            Execution backend: ``"jax"`` for full-JIT Gibbs (default, falls back to ``"numpy"`` when JAX is not installed), or ``"numpy"`` for Python-loop Gibbs.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
        use_mala : bool, default True
            If True, use MALA for the λ update in the JAX path.
            Ignored when ``use_slice=True``.
        use_slice : bool, default True
            If True, use slice sampling for the ρ/λ update in the
            JAX path.  Slice sampling gives much better ESS per sample
            than MALA.  Ignored when ``gibbs_method="numpy"``.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.  If None, defaults
            to ``(rho_upper - rho_lower) * 0.1``.  Ignored when
            ``use_slice=False`` or ``gibbs_method="numpy"``.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.

        Returns
        -------
        az.InferenceData
        """
        if self.robust:
            raise NotImplementedError("Gibbs sampling is not yet supported for robust (Student-t) models. Use sampler='nuts' (the default).")
        from ...samplers.gaussian import GaussianGibbsPriors, GaussianSEMGibbs
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(self._X, list(self._feature_names))
        priors = GaussianGibbsPriors(beta_mu=self.priors.get('beta_mu', default_beta_mu), beta_sigma=self.priors.get('beta_sigma', default_beta_sigma), sigma2_alpha=self.priors.get('sigma2_alpha', 2.0), sigma2_beta=self.priors.get('sigma2_beta', float(np.var(self._y))), rho_lower=self._logdet_bounds.rho_min, rho_upper=self._logdet_bounds.rho_max)
        gibbs = GaussianSEMGibbs(y=self._y, X=self._X, W_sparse=self._W_sparse_NT, priors=priors, logdet_fn=self._logdet_numpy_fn, logdet_vec_fn=self._logdet_numpy_vec_fn, feature_names=list(self._feature_names), model_type='sem', W_eigs=self._W_eigs if self._resolved_logdet_method == 'eigenvalue' else None, logdet_method=self.logdet_method, T=self._T)
        self._idata = gibbs.fit(draws=draws, tune=tune, chains=chains, random_seed=random_seed, thin=thin, n_jobs=n_jobs, progressbar=progressbar, gibbs_method=gibbs_method, mala_step_size=mala_step_size, use_mala=use_mala, use_slice=use_slice, slice_width=slice_width, chain_method=chain_method)
        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean('beta')
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            return self._X[:, ni] @ beta
        return self._X @ beta

class SDMPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial Durbin panel regression.

    Implements

    .. math::

        y_{it} = \\rho \\sum_j w_{ij} y_{jt} + x_{it}'\\beta
        + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    The coefficient vector sampled by the model stacks the local and
    lagged-regressor blocks as :math:`[\\beta, \\theta]`.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Should
        be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`; auto-selected when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails. The lower bound of 2 ensures the
    variance exists.
    """
    _has_wx_in_beta: bool = True
    _jacobian_param: str | None = 'rho'
    _gibbs_class: str | None = 'GaussianSARGibbs'

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f'W*{name}' for name in self._wx_feature_names]
    _priors_cls = PanelSDMPriors

    def _fit_gibbs(self, draws: int=2000, tune: int=1000, chains: int=4, random_seed: int | None=None, thin: int=1, n_jobs: int=-1, progressbar: bool=True, gibbs_method: str='jax', mala_step_size: float=0.05, use_mala: bool=True, use_slice: bool=True, slice_width: float | None=None, chain_method: str | None=None) -> 'az.InferenceData':
        """Sample posterior via 3-block Gaussian Gibbs.

        The SDM panel model is equivalent to SAR panel with Z = [X, WX]
        as the design matrix.  The β block covers both direct and
        indirect coefficients.

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
            Number of parallel workers for the NumPy path.
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "jax"
            Execution backend: ``"jax"`` for full-JIT Gibbs (default, falls back to ``"numpy"`` when JAX is not installed), or ``"numpy"`` for Python-loop Gibbs.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
        use_mala : bool, default True
            If True, use MALA for the ρ update in the JAX path.
            Ignored when ``use_slice=True``.
        use_slice : bool, default True
            If True, use slice sampling for the ρ/λ update in the
            JAX path.  Slice sampling gives much better ESS per sample
            than MALA.  Ignored when ``gibbs_method="numpy"``.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.  If None, defaults
            to ``(rho_upper - rho_lower) * 0.1``.  Ignored when
            ``use_slice=False`` or ``gibbs_method="numpy"``.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.

        Returns
        -------
        az.InferenceData
        """
        if self.robust:
            raise NotImplementedError("Gibbs sampling is not yet supported for robust (Student-t) models. Use sampler='nuts' (the default).")
        from ...samplers.gaussian import GaussianGibbsPriors, GaussianSARGibbs
        Z = np.hstack([self._X, self._WX])
        feature_names = list(self._feature_names) + [f'W*{name}' for name in self._wx_feature_names]
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(Z, feature_names)
        priors = GaussianGibbsPriors(beta_mu=self.priors.get('beta_mu', default_beta_mu), beta_sigma=self.priors.get('beta_sigma', default_beta_sigma), sigma2_alpha=self.priors.get('sigma2_alpha', 2.0), sigma2_beta=self.priors.get('sigma2_beta', float(np.var(self._y))), rho_lower=self._logdet_bounds.rho_min, rho_upper=self._logdet_bounds.rho_max)
        gibbs = GaussianSARGibbs(y=self._y, X=Z, W_sparse=self._W_sparse_NT, Wy=self._Wy, priors=priors, logdet_fn=self._logdet_numpy_fn, logdet_vec_fn=self._logdet_numpy_vec_fn, feature_names=feature_names, model_type='sdm', W_eigs=self._W_eigs if self._resolved_logdet_method == 'eigenvalue' else None, logdet_method=self.logdet_method, T=self._T)
        self._idata = gibbs.fit(draws=draws, tune=tune, chains=chains, random_seed=random_seed, thin=thin, n_jobs=n_jobs, progressbar=progressbar, gibbs_method=gibbs_method, mala_step_size=mala_step_size, use_mala=use_mala, use_slice=use_slice, slice_width=slice_width, chain_method=chain_method)
        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        rho = float(self._posterior_mean('rho'))
        beta = self._posterior_mean('beta')
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            Z = np.hstack([self._X[:, ni], self._WX])
        else:
            Z = np.hstack([self._X, self._WX])
        return rho * self._Wy + Z @ beta

class SDEMPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial Durbin error panel regression.

    Implements

    .. math::

        y_{it} = x_{it}'\\beta + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\alpha_i + \\tau_t + u_{it},
        \\qquad u_{it} = \\lambda \\sum_j w_{ij} u_{jt} + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    The sampled coefficient vector stacks the local and lagged-covariate
    blocks as :math:`[\\beta, \\theta]`.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Should
        be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the spatially filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)\\bigl(y_t - X_t \\beta - (W X_t)\\theta - \\alpha\\bigr) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails. The lower bound of 2 ensures the
    variance exists.
    """
    _has_wx_in_beta: bool = True
    _jacobian_param: str | None = 'lam'
    _gibbs_class: str | None = 'GaussianSEMGibbs'

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f'W*{name}' for name in self._wx_feature_names]
    _priors_cls = PanelSDEMPriors

    def _fit_gibbs(self, draws: int=2000, tune: int=1000, chains: int=4, random_seed: int | None=None, thin: int=1, n_jobs: int=-1, progressbar: bool=True, gibbs_method: str='jax', mala_step_size: float=0.05, use_mala: bool=True, use_slice: bool=True, slice_width: float | None=None, chain_method: str | None=None) -> 'az.InferenceData':
        """Sample posterior via 3-block Gaussian Gibbs.

        The SDEM panel model is equivalent to SEM panel with Z = [X, WX]
        as the design matrix.  The β block covers both direct and
        indirect coefficients.

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
            Number of parallel workers for the NumPy path.
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "jax"
            Execution backend: ``"jax"`` for full-JIT Gibbs (default, falls back to ``"numpy"`` when JAX is not installed), or ``"numpy"`` for Python-loop Gibbs.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
        use_mala : bool, default True
            If True, use MALA for the λ update in the JAX path.
            Ignored when ``use_slice=True``.
        use_slice : bool, default True
            If True, use slice sampling for the ρ/λ update in the
            JAX path.  Slice sampling gives much better ESS per sample
            than MALA.  Ignored when ``gibbs_method="numpy"``.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.  If None, defaults
            to ``(rho_upper - rho_lower) * 0.1``.  Ignored when
            ``use_slice=False`` or ``gibbs_method="numpy"``.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.

        Returns
        -------
        az.InferenceData
        """
        if self.robust:
            raise NotImplementedError("Gibbs sampling is not yet supported for robust (Student-t) models. Use sampler='nuts' (the default).")
        from ...samplers.gaussian import GaussianGibbsPriors, GaussianSEMGibbs
        Z = np.hstack([self._X, self._WX])
        feature_names = list(self._feature_names) + [f'W*{name}' for name in self._wx_feature_names]
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(Z, feature_names)
        priors = GaussianGibbsPriors(beta_mu=self.priors.get('beta_mu', default_beta_mu), beta_sigma=self.priors.get('beta_sigma', default_beta_sigma), sigma2_alpha=self.priors.get('sigma2_alpha', 2.0), sigma2_beta=self.priors.get('sigma2_beta', float(np.var(self._y))), rho_lower=self._logdet_bounds.rho_min, rho_upper=self._logdet_bounds.rho_max)
        gibbs = GaussianSEMGibbs(y=self._y, X=Z, W_sparse=self._W_sparse_NT, priors=priors, logdet_fn=self._logdet_numpy_fn, logdet_vec_fn=self._logdet_numpy_vec_fn, feature_names=feature_names, model_type='sdem', W_eigs=self._W_eigs if self._resolved_logdet_method == 'eigenvalue' else None, logdet_method=self.logdet_method, T=self._T)
        self._idata = gibbs.fit(draws=draws, tune=tune, chains=chains, random_seed=random_seed, thin=thin, n_jobs=n_jobs, progressbar=progressbar, gibbs_method=gibbs_method, mala_step_size=mala_step_size, use_mala=use_mala, use_slice=use_slice, slice_width=slice_width, chain_method=chain_method)
        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean('beta')
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            Z = np.hstack([self._X[:, ni], self._WX])
        else:
            Z = np.hstack([self._X, self._WX])
        return Z @ beta

class SLXPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian SLX panel regression.

    Implements

    .. math::

        y_{it} = x_{it}'\\beta + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    There is no contemporaneous spatial lag on :math:`y`, so no Jacobian
    adjustment is required. The coefficient vector stacks the local and
    lagged-covariate blocks as :math:`[\\beta, \\theta]`.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Used
        to construct the ``WX`` block. Should be row-standardised.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu_lam`` (float, default 1/30): Rate of TruncExp(lower=2)
          prior on :math:`\\nu` (only used when ``robust=True``).

    logdet_method : str, optional
        Accepted for API consistency; unused (SLX has no spatial
        Jacobian).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails. The lower bound of 2 ensures the
    variance exists.
    """
    _has_wx_in_beta: bool = True
    _jacobian_param: str | None = None
    _gibbs_class: str | None = None

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f'W*{name}' for name in self._wx_feature_names]
    _priors_cls = PanelSLXPriors

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean('beta')
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            Z = np.hstack([self._X[:, ni], self._WX])
        else:
            Z = np.hstack([self._X, self._WX])
        return Z @ beta