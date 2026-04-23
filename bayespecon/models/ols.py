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

from .base import SpatialModel


class OLS(SpatialModel):
    """Bayesian ordinary least squares cross-sectional regression.

    .. math::
        y = X\\beta + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    This model places diffuse Normal priors on the coefficient vector
    :math:`\\beta` and a HalfNormal prior on the noise standard deviation
    :math:`\\sigma`.

    ``W`` is **optional**.  If supplied, Bayesian LM tests can be run on
    the OLS posterior to guide model selection.

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

        - ``beta_mu`` (float, default 0): Prior mean for :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Prior std for :math:`\\beta`.
        - ``sigma_sigma`` (float, default 10): Scale for HalfNormal prior
          on :math:`\\sigma`.
    logdet_method : str, optional
        Unused for OLS (no spatial lag); kept for API compatibility with
        :class:`SpatialModel`.
    """

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for Bayesian OLS regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object with Normal likelihood.
        """
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = pt.dot(self._X, beta)
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

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values ``X @ E[beta | data]``.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta
