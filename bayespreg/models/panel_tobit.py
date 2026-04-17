"""Spatial panel Tobit models.

Implements left-censored (default at 0) panel spatial Tobit variants:

- ``SARPanelTobit``: spatial lag panel Tobit
- ``SEMPanelTobit``: spatial error panel Tobit

Notes
-----
These classes force ``model=0`` (pooled transform) because within transforms
are not compatible with censoring augmentation on the observed scale.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..logdet import make_logdet_fn
from .panel_base import SpatialPanelModel


class _PanelTobitBase(SpatialPanelModel):
    """Shared helpers for panel Tobit classes."""

    def __init__(self, *args, censoring: float = 0.0, **kwargs):
        kwargs.pop("model", None)
        self.censoring = float(censoring)
        super().__init__(*args, model=0, **kwargs)
        self._censored_mask = self._y <= self.censoring + 1e-12
        self._censored_idx = np.where(self._censored_mask)[0]

    def _latent_y_tensor(self) -> pt.TensorVariable:
        y_lat = pt.as_tensor_variable(self._y.astype(np.float64))
        n_cens = int(self._censored_idx.size)
        if n_cens > 0:
            censor_sigma = float(self.priors.get("censor_sigma", 10.0))
            y_cens_gap = pm.HalfNormal("y_cens_gap", sigma=censor_sigma, shape=n_cens)
            y_cens = self.censoring - y_cens_gap
            y_lat = pt.set_subtensor(y_lat[self._censored_idx], y_cens)
        return y_lat

    def _posterior_latent_y_mean(self) -> np.ndarray:
        y_lat = self._y.copy().astype(float)
        if self._censored_idx.size > 0 and "y_cens_gap" in self._idata.posterior:
            gap_hat = self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            y_lat[self._censored_idx] = self.censoring - np.asarray(gap_hat, dtype=float)
        return y_lat


class SARPanelTobit(_PanelTobitBase):
    r"""Bayesian spatial lag panel Tobit model.

    .. math::
        y^* = \rho W y^* + X\beta + \varepsilon,\quad \varepsilon \sim N(0,\sigma^2 I)

    with observed outcome

    .. math::
        y = \max(c, y^*)
    """

    def _build_pymc_model(self) -> pm.Model:
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_dense,
            method=self.logdet_method,
            rho_min=rho_lower,
            rho_max=rho_upper,
        )
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_lat = self._latent_y_tensor()
            resid = y_lat - rho * pt.dot(W_pt, y_lat) - pt.dot(self._X, beta)
            logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        return rho * (self._W_dense @ y_lat) + self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        n = self._W_dense.shape[0]
        S = np.linalg.inv(np.eye(n) - rho * self._W_dense)
        direct = np.diag(S).mean() * beta
        total = S.mean(axis=1).mean() * beta
        indirect = total - direct
        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._feature_names,
        }


class SEMPanelTobit(_PanelTobitBase):
    r"""Bayesian spatial error panel Tobit model.

    .. math::
        y^* = X\beta + u,\quad u = \lambda W u + \varepsilon,
        \quad \varepsilon \sim N(0,\sigma^2 I)

    with observed outcome ``y = max(c, y*)``.
    """

    def _build_pymc_model(self) -> pm.Model:
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_dense,
            method=self.logdet_method,
            rho_min=lam_lower,
            rho_max=lam_upper,
        )
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            y_lat = self._latent_y_tensor()
            resid = y_lat - pt.dot(self._X, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
            pm.Potential("eps_loglik", logp_eps)
            pm.Potential("jacobian", logdet_fn(lam))

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        beta = self._posterior_mean("beta")
        return self._X @ beta

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }
