"""Spatial Autoregressive (SAR / Spatial Lag) Model.

y = rho * W @ y + X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)

The full likelihood includes the Jacobian |I - rho*W|, added via
pm.Potential so that NUTS samples from the correct posterior.
"""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin
from ..base import SpatialModel
from ..priors import SARPriors


class SAR(GaussianLikelihoodMixin, SpatialModel):
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
        - ``sigma2_alpha`` (float, default 2.0): Shape of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): Scale of the
          InverseGamma prior on :math:`\\sigma^2`.
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

    _priors_cls = SARPriors
    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy",)
    _jacobian_param: str | None = "rho"
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = "GaussianSARGibbs"
    _model_type: str = "sar"

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
        from ...diagnostics.lmtests import _get_posterior_draws
        from ...diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")  # (G,)
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k)
        rho_draws.shape[0]
        beta_draws.shape[1]

        eigs = self._W_eigs  # (n,)

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
