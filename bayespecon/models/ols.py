"""Bayesian OLS (non-spatial) cross-sectional regression model.

y = X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)

This model contains no spatial structure of its own.  It is the natural
baseline from which Bayesian spatial specification tests are run to
determine which spatial model — SAR, SEM, SLX, etc. — is most appropriate.
W is optional at construction time.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..diagnostics.lmtests import OLS_SUITE
from .base import SpatialModel
from .priors import OLSPriors


class OLS(SpatialModel):
    """Bayesian ordinary least squares cross-sectional regression.

    .. math::
        y = X\\beta + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    This model places diffuse Normal priors on the coefficient vector
    :math:`\\beta` and a HalfNormal prior on the noise standard deviation
    :math:`\\sigma`.

    ``W`` is **optional**.  If supplied, Bayesian LM tests can be run on
    the OLS posterior to guide model selection.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.  The ``nu_lam`` rate can be controlled via
    ``priors={"nu_lam": value}``.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula string, e.g. ``"price ~ poverty + income"``.
        If provided, ``data`` must also be supplied.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source when using formula mode.
    y : array-like, optional
        Dependent variable of shape ``(n,)``.  Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Predictor matrix.  Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix, optional
        Spatial weights matrix of shape ``(n, n)``.  Not used during
        estimation; required for Bayesian LM specification tests.
    priors : dict, optional
        Override default priors.  Supported keys:

        - ``beta_mu`` (array or float, optional): Prior mean for
          :math:`\beta`. Default is Gelman et al. (2008) weakly-informative
          mean (``mean(y)`` for intercept/constant columns, 0 for slopes).
        - ``beta_sigma`` (array or float, optional): Prior std for
          :math:`\beta`. Default is Gelman et al. (2008) weakly-informative
          scale: ``2.5 * sd(y)`` for intercept/constant columns and
          ``2.5 * sd(y) / sd(x_j)`` for each slope.
        - ``sigma2_alpha`` (float, default 2.0): Shape of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): Scale of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``nu_lam`` (float, default 1/30): Rate for Exponential prior on
          :math:`\\nu` (only used when ``robust=True``).

    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal.
    """

    _priors_cls = OLSPriors

    _spatial_diagnostics_tests = OLS_SUITE.tests

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for Bayesian OLS regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object with Normal or Student-t
            likelihood depending on ``self.robust``.
        """
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            self._X, list(self._feature_names)
        )
        beta_mu = self.priors.get("beta_mu", default_beta_mu)
        beta_sigma = self.priors.get("beta_sigma", default_beta_sigma)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y)))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            mu = pt.dot(self._X, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Not applicable — OLS has no spatial lag structure.

        Raises
        ------
        NotImplementedError
            Always raised; use Bayesian LM diagnostics instead to
            assess spatial structure after estimation.
        """
        raise NotImplementedError(
            "OLS has no spatial lag structure and therefore no spatial effects. "
            "Use Bayesian LM diagnostics to assess which spatial model "
            "is appropriate, then refit with SAR, SEM, SLX, SDM, or SDEM."
        )

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Not applicable — OLS has no spatial lag structure.

        Raises
        ------
        NotImplementedError
            Always raised; use Bayesian LM diagnostics instead.
        """
        raise NotImplementedError(
            "OLS has no spatial lag structure and therefore no spatial effects. "
            "Use Bayesian LM diagnostics to assess which spatial model "
            "is appropriate, then refit with SAR, SEM, SLX, SDM, or SDEM."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values ``X @ E[beta | data]``.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta
