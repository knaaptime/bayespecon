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

        - ``beta_mu`` (float, default 0): Prior mean for :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Prior std for :math:`\\beta`.
        - ``sigma_sigma`` (float, default 10): Scale for HalfNormal prior
          on :math:`\\sigma`.
        - ``nu_lam`` (float, default 1/30): Rate for Exponential prior on
          :math:`\\nu` (only used when ``robust=True``).
    logdet_method : str, optional
        Unused for OLS (no spatial lag); kept for API compatibility with
        :class:`~bayespecon.models.base.SpatialModel`.
    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal.
    """

    _spatial_diagnostics_tests = [
        # Lazy import avoids circular dependency; resolved at call time.
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_lag_test"],
        ).bayesian_lm_lag_test(m), "LM-Lag"),
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_error_test"],
        ).bayesian_lm_error_test(m), "LM-Error"),
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_sdm_joint_test"],
        ).bayesian_lm_sdm_joint_test(m), "LM-SDM-Joint"),
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_slx_error_joint_test"],
        ).bayesian_lm_slx_error_joint_test(m), "LM-SLX-Error-Joint"),
    ]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for Bayesian OLS regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object with Normal or Student-t
            likelihood depending on ``self.robust``.
        """
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
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

    def _compute_spatial_effects_posterior(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
