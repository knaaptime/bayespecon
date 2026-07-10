"""Spatial panel model classes."""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin
from ..panel_base import SpatialPanelModel
from ..priors import (
    PanelOLSPriors,
    PanelSARPriors,
    PanelSDEMPriors,
    PanelSDMPriors,
    PanelSEMPriors,
    PanelSLXPriors,
)


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
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            return self._X[:, ni] @ beta
        return self._X @ beta

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        OLS panel has no spatial structure: Direct = beta, Indirect = 0.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k)

        ni = self._beta_nonintercept_indices
        direct_samples = beta_draws[:, ni].copy()
        indirect_samples = np.zeros_like(direct_samples)
        total_samples = direct_samples.copy()

        return direct_samples, indirect_samples, total_samples


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
    _jacobian_param: str | None = "rho"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSARGibbs"
    _model_type: str = "sar"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            return rho * self._Wy + self._X[:, ni] @ beta
        return rho * self._Wy + self._X @ beta

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SAR panel impacts use the same eigenvalue-based formulas as
        cross-sectional SAR, applied per draw.
        """
        from ...diagnostics.lmtests import _get_posterior_draws
        from ...diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")  # (G,)
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k)
        rho_draws.shape[0]

        eigs = self._W_eigs
        mean_diag = _chunked_eig_means(rho_draws, eigs)  # (G,)

        mean_row_sum = self._batch_mean_row_sum(rho_draws)  # (G,)

        # Exclude intercept from effects (it has no meaningful spatial interpretation)
        ni = self._beta_nonintercept_indices
        direct_samples = mean_diag[:, None] * beta_draws[:, ni]  # (G, k_ni)
        total_samples = mean_row_sum[:, None] * beta_draws[:, ni]  # (G, k_ni)
        indirect_samples = total_samples - direct_samples  # (G, k_ni)

        return direct_samples, indirect_samples, total_samples


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
    _jacobian_param: str | None = "lam"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSEMGibbs"
    _model_type: str = "sem"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            return self._X[:, ni] @ beta
        return self._X @ beta

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SEM panel has no spatial multiplier on X: Direct = beta, Indirect = 0.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k)

        ni = self._beta_nonintercept_indices
        direct_samples = beta_draws[:, ni].copy()
        indirect_samples = np.zeros_like(direct_samples)
        total_samples = direct_samples.copy()

        return direct_samples, indirect_samples, total_samples


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
    _jacobian_param: str | None = "rho"

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    _priors_cls = PanelSDMPriors
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSARGibbs"
    _model_type: str = "sdm"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            Z = np.hstack([self._X[:, ni], self._WX])
        else:
            Z = np.hstack([self._X, self._WX])
        return rho * self._Wy + Z @ beta

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SDM panel impacts use the same eigenvalue-based formulas as
        cross-sectional SDM, applied per draw.
        """
        from ...diagnostics.lmtests import _get_posterior_draws
        from ...diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")  # (G,)
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k+k_wx)
        rho_draws.shape[0]
        if self._intercept_dropped:
            k = len(self._beta_nonintercept_indices)
        else:
            k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1_draws = beta_draws[:, :k]  # (G, k)
        beta2_draws = beta_draws[:, k : k + kw]  # (G, kw)

        eigs = self._W_eigs
        mean_diag_M = _chunked_eig_means(rho_draws, eigs)  # (G,)
        mean_diag_MW = _chunked_eig_means(rho_draws, eigs, weights=eigs)  # (G,)

        mean_row_sum_M = self._batch_mean_row_sum(rho_draws)  # (G,)
        mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)  # (G,)

        wx_idx = self._beta_wx_column_indices
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
    _jacobian_param: str | None = "lam"

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    _priors_cls = PanelSDEMPriors
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSEMGibbs"
    _model_type: str = "sdem"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            Z = np.hstack([self._X[:, ni], self._WX])
        else:
            Z = np.hstack([self._X, self._WX])
        return Z @ beta

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SDEM panel impacts match SLX form (no rho multiplier).
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k+k_wx)
        beta_draws.shape[0]
        if self._intercept_dropped:
            k = len(self._beta_nonintercept_indices)
        else:
            k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1_draws = beta_draws[:, :k]  # (G, k)
        beta2_draws = beta_draws[:, k : k + kw]  # (G, kw)

        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        wx_idx = self._beta_wx_column_indices
        direct_samples = beta1_draws[:, wx_idx] + mean_diag_w * beta2_draws  # (G, kw)
        total_samples = beta1_draws[:, wx_idx] + mean_row_sum_w * beta2_draws  # (G, kw)
        indirect_samples = total_samples - direct_samples  # (G, kw)

        return direct_samples, indirect_samples, total_samples


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
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    _priors_cls = PanelSLXPriors

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        if self._intercept_dropped:
            ni = self._beta_nonintercept_indices
            Z = np.hstack([self._X[:, ni], self._WX])
        else:
            Z = np.hstack([self._X, self._WX])
        return Z @ beta

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SLX panel impacts are linear in beta (no rho multiplier).
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k+k_wx)
        beta_draws.shape[0]
        if self._intercept_dropped:
            k = len(self._beta_nonintercept_indices)
        else:
            k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1_draws = beta_draws[:, :k]  # (G, k)
        beta2_draws = beta_draws[:, k : k + kw]  # (G, kw)

        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        wx_idx = self._beta_wx_column_indices
        direct_samples = beta1_draws[:, wx_idx] + mean_diag_w * beta2_draws  # (G, kw)
        total_samples = beta1_draws[:, wx_idx] + mean_row_sum_w * beta2_draws  # (G, kw)
        indirect_samples = total_samples - direct_samples  # (G, kw)

        return direct_samples, indirect_samples, total_samples
