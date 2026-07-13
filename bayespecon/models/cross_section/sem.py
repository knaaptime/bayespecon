"""Spatial Error Model (SEM).

y = X @ beta + u,  u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Equivalently: (I - lambda*W)(y - X@beta) = epsilon
Likelihood: epsilon ~ N(0, sigma^2 I), plus Jacobian log|I - lambda*W|.
"""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin
from ..base import SpatialModel
from ..priors import SEMPriors


class SEM(GaussianLikelihoodMixin, SpatialModel):
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
        auto-selects by size: ``"eigenvalue"`` for ``n <= 500``; for
        ``500 < n <= 20000``, ``"cheb_cholesky"`` (exact, sparse Cholesky
        at Chebyshev nodes) when ``W`` is symmetric else ``"aaa"`` (AAA
        rational approximation); ``"cheb_stochastic"`` for ``n > 20000``.
        Explicit opt-ins: ``"chebyshev"`` (Barry-Pace) and ``"slq"``
        (stochastic Lanczos quadrature).
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
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "cross_section")

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SEM model the spatial multiplier does not apply to :math:`X`
        directly, so :math:`\\text{Direct}_k = \\beta_k`,
        :math:`\\text{Indirect}_k = 0`, and :math:`\\text{Total}_k = \\beta_k`.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each of
            shape ``(G, k)``.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

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
