"""Bayesian spatial panel models with unit random effects.

Analogues of the MATLAB ``prandom`` (non-spatial GLS random effects) and the
LeSage/Pace spatial panel routines, cast as hierarchical Bayesian models.

Model structure for all three classes
--------------------------------------
.. math::
    y_{it} = \\text{(spatial/non-spatial mean)} + \\alpha_i + \\varepsilon_{it}

    \\alpha_i \\sim N(0, \\sigma_\\alpha^2), \\quad
    \\varepsilon_{it} \\sim N(0, \\sigma^2)

Data convention
---------------
Observations must be stacked time-first (time period changes slowest),
so that observation index ``i`` belongs to unit ``i % N``.  This matches
the convention used by all other panel classes in this package.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..logdet import make_logdet_fn
from .panel_base import SpatialPanelModel


class OLSPanelRE(SpatialPanelModel):
    """Bayesian random effects panel regression (non-spatial).

    .. math::
        y_{it} = X_{it}\\beta + \\alpha_i + \\varepsilon_{it}

    where :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)` are unit-level
    random effects and :math:`\\varepsilon_{it} \\sim N(0, \\sigma^2)`.

    Parameters
    ----------
    formula, data, y, X
        Either formula mode (formula + data) or matrix mode (y + X).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Accepts a
        :class:`libpysal.graph.Graph` (the modern libpysal graph API) or any
        :class:`scipy.sparse` matrix.  The legacy :class:`libpysal.weights.W`
        object is **not** accepted directly; pass ``w.sparse`` or convert with
        ``libpysal.graph.Graph.from_W(w)``.
        Unused in the RE likelihood but required by the base class for
        consistency (e.g. computing spatial lags for SDM/SDEM variants).
        W should be row-standardised; a :class:`UserWarning` is raised if not.
    unit_col, time_col
        Required in formula mode.
    N, T
        Required in matrix mode.
    priors : dict, optional
        Override default priors. Supported keys:

        ``beta_mu`` (default 0), ``beta_sigma`` (default 1e6),
        ``sigma_sigma`` (default 10), ``sigma_alpha_sigma`` (default 10).

    Notes
    -----
    Data are **not** demeaned — the random effects absorb the unit-level
    mean structure probabilistically.  This is the Bayesian analogue of
    the classical GLS random-effects estimator in ``prandom.m``.
    """

    def __init__(self, **kwargs):
        kwargs.pop("model", None)  # RE always uses raw (pooled) data
        super().__init__(model=0, **kwargs)
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
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        sigma_alpha_sigma = self.priors.get("sigma_alpha_sigma", 10.0)

        unit_idx = self._unit_idx

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=sigma_alpha_sigma)
            alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")

            mu = pt.dot(self._X, beta) + alpha[unit_idx]
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
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }


class SARPanelRE(SpatialPanelModel):
    """Bayesian spatial lag panel model with unit random effects.

    .. math::
        y_{it} = \\rho (Wy)_{it} + X_{it}\\beta + \\alpha_i + \\varepsilon_{it}

    where :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)` are unit-level
    random effects and :math:`\\varepsilon_{it} \\sim N(0, \\sigma^2)`.

    Parameters
    ----------
    formula, data, y, X, W, unit_col, time_col, N, T, priors, logdet_method
        See :class:`~bayespecon.models.panel_base.SpatialPanelModel`.
        ``model`` is forced to 0 (no within-transform demeaning).

    Priors (``priors`` dict keys)
    ------------------------------
    rho_lower, rho_upper : float, default -1, 1
    beta_mu : float, default 0
    beta_sigma : float, default 1e6
    sigma_sigma : float, default 10
    sigma_alpha_sigma : float, default 10
    """

    def __init__(self, **kwargs):
        kwargs.pop("model", None)
        super().__init__(model=0, **kwargs)
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
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        sigma_alpha_sigma = self.priors.get("sigma_alpha_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real, method=self.logdet_method,
            rho_min=rho_lower, rho_max=rho_upper,
            T=self._T,
        )
        unit_idx = self._unit_idx

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=sigma_alpha_sigma)
            alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")

            mu = rho * self._Wy + pt.dot(self._X, beta) + alpha[unit_idx]
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)
            pm.Potential("jacobian", logdet_fn(rho))

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
        """Sample posterior and attach Jacobian-corrected log-likelihood.

        The SAR panel RE model uses ``pm.Normal("obs", observed=y)`` which
        auto-captures the Gaussian log-likelihood, plus a ``pm.Potential``
        Jacobian term that is not captured.  When ``log_likelihood=True``
        is requested, the Jacobian correction is added post-sampling.
        """
        idata_kwargs = idata_kwargs or {}
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )
        if idata_kwargs.get("log_likelihood", False):
            self._attach_jacobian_corrected_log_likelihood(idata, "rho", T=self._T)
        return idata

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
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        if self._is_row_std:
            mean_row_sum = 1.0 / (1.0 - rho)
        else:
            mean_row_sum = float(
                np.linalg.solve(
                    np.eye(self._W_sparse.shape[0]) - rho * self._W_sparse.toarray(),
                    np.ones(self._W_sparse.shape[0]),
                ).mean()
            )
        direct = mean_diag * beta
        total = mean_row_sum * beta
        return {
            "direct": direct,
            "indirect": total - direct,
            "total": total,
            "feature_names": self._feature_names,
        }


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
    formula, data, y, X, W, unit_col, time_col, N, T, priors, logdet_method
        See :class:`~bayespecon.models.panel_base.SpatialPanelModel`.
        ``model`` is forced to 0.

    Priors (``priors`` dict keys)
    ------------------------------
    lam_lower, lam_upper : float, default -1, 1
    beta_mu : float, default 0
    beta_sigma : float, default 1e6
    sigma_sigma : float, default 10
    sigma_alpha_sigma : float, default 10
    """

    def __init__(self, **kwargs):
        kwargs.pop("model", None)
        super().__init__(model=0, **kwargs)
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SEM panel with random effects.

        Returns
        -------
        pymc.Model
        """
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        sigma_alpha_sigma = self.priors.get("sigma_alpha_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real, method=self.logdet_method,
            rho_min=lam_lower, rho_max=lam_upper,
            T=self._T,
        )
        W_pt = pt.as_tensor_variable(self._W_dense)
        unit_idx = self._unit_idx

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=sigma_alpha_sigma)
            alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")

            # epsilon = (I - lam*W)(y - X@beta - alpha_expanded)
            #         = resid - lam * W @ resid
            resid = self._y - pt.dot(self._X, beta) - alpha[unit_idx]
            eps = resid - lam * pt.dot(W_pt, resid)
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
        """Sample posterior and attach pointwise log-likelihood for IC metrics.

        The SEM panel RE model uses ``pm.Potential`` for both the Gaussian
        error log-likelihood and the Jacobian, so neither is auto-captured.
        We compute the complete pointwise log-likelihood manually after
        sampling, including the random effects ``alpha[unit_idx]``.
        """
        idata_kwargs = idata_kwargs or {}
        idata = super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

        if "log_likelihood" in idata.groups() and "obs" in idata.log_likelihood:
            return idata

        import pytensor
        import pytensor.tensor as pt_ll
        import xarray as xr

        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        alpha = idata.posterior["alpha"].values

        c, d = lam.shape
        s = c * d
        n = self._y.shape[0]
        X = self._X
        W = self._W_dense
        unit_idx = self._unit_idx

        lam_f = lam.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        alpha_f = alpha.reshape(s, alpha.shape[-1])  # (s, N)

        # resid = y - X@beta - alpha[unit_idx]
        resid = self._y[None, :] - beta_f @ X.T - alpha_f[:, unit_idx]
        eps = resid - lam_f[:, None] * (resid @ W.T)

        ll = -0.5 * (
            (eps / sigma_f[:, None]) ** 2
            + np.log(2.0 * np.pi)
            + 2.0 * np.log(sigma_f[:, None])
        )

        # Eigenvalue-based Jacobian: log|I - lam*W| * T / n (pure numpy)
        eigs = self._W_eigs.real.astype(np.float64)
        jac = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_f]) * self._T  # (n_draws,)
        ll = ll + jac[:, None] / n

        ll = ll.reshape(c, d, n)
        ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": ll_da})
        return idata

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
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }
