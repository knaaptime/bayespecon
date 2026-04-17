"""Spatial Tobit model classes.

Implements left-censored (default at 0) Bayesian spatial Tobit variants:

- ``SARTobit``: spatial autoregressive Tobit
- ``SEMTobit``: spatial error Tobit
- ``SDMTobit``: spatial Durbin Tobit

All classes use latent-data augmentation for censored observations.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..logdet import make_logdet_fn
from .base import SpatialModel


class _SpatialTobitBase(SpatialModel):
    """Shared helpers for spatial Tobit models."""

    def __init__(self, *args, censoring: float = 0.0, **kwargs):
        self.censoring = float(censoring)
        super().__init__(*args, **kwargs)
        self._censored_mask = self._y <= self.censoring + 1e-12
        self._censored_idx = np.where(self._censored_mask)[0]

    def _latent_y_tensor(self) -> pt.TensorVariable:
        """Create latent y* tensor where censored values are sampled."""
        y_lat = pt.as_tensor_variable(self._y.astype(np.float64))
        n_cens = int(self._censored_idx.size)
        if n_cens > 0:
            censor_sigma = float(self.priors.get("censor_sigma", 10.0))
            y_cens_gap = pm.HalfNormal("y_cens_gap", sigma=censor_sigma, shape=n_cens)
            y_cens = self.censoring - y_cens_gap
            y_lat = pt.set_subtensor(y_lat[self._censored_idx], y_cens)
        return y_lat

    def _posterior_latent_y_mean(self) -> np.ndarray:
        """Posterior mean of latent y* on the observed index set."""
        y_lat = self._y.copy().astype(float)
        if self._censored_idx.size > 0 and "y_cens_gap" in self._idata.posterior:
            gap_hat = self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            y_lat[self._censored_idx] = self.censoring - np.asarray(gap_hat, dtype=float)
        return y_lat


class SARTobit(_SpatialTobitBase):
    r"""Bayesian spatial autoregressive Tobit model.

    .. math::
        y^* = \rho W y^* + X\beta + \varepsilon,\quad \varepsilon \sim N(0,\sigma^2 I),

    with observed outcome

    .. math::
        y = \max(c, y^*)

    where ``c`` is the left-censoring point (default: ``0``).
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

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        return rho * (self._W_dense @ y_lat) + self._X @ beta


class SEMTobit(_SpatialTobitBase):
    r"""Bayesian spatial error Tobit model.

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

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        beta = self._posterior_mean("beta")
        return self._X @ beta


class SDMTobit(_SpatialTobitBase):
    r"""Bayesian spatial Durbin Tobit model.

    .. math::
        y^* = \rho Wy^* + X\beta_1 + WX\beta_2 + \varepsilon,
        \quad \varepsilon \sim N(0,\sigma^2 I)

    with observed outcome ``y = max(c, y*)``.
    """

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        Z = np.hstack([self._X, self._WX])

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
            resid = y_lat - rho * pt.dot(W_pt, y_lat) - pt.dot(Z, beta)
            logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]

        n = self._W_dense.shape[0]
        W = self._W_dense
        M = np.linalg.inv(np.eye(n) - rho * W)

        direct = np.array([
            np.diag(M @ (beta1[j] * np.eye(n) + b2 * W)).mean()
            for j, b2 in zip(self._wx_column_indices, beta2)
        ])
        total = np.array([
            (M @ (beta1[j] * np.eye(n) + b2 * W)).mean()
            for j, b2 in zip(self._wx_column_indices, beta2)
        ])
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._wx_feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        Z = np.hstack([self._X, self._WX])
        return rho * (self._W_dense @ y_lat) + Z @ beta
