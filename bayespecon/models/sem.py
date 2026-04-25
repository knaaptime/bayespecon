"""Spatial Error Model (SEM).

y = X @ beta + u,  u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Equivalently: (I - lambda*W)(y - X@beta) = epsilon
Likelihood: epsilon ~ N(0, sigma^2 I), plus Jacobian log|I - lambda*W|.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from ..logdet import make_logdet_fn
from .base import SpatialModel


class SEM(SpatialModel):
    """Bayesian Spatial Error Model.

    .. math::
        y = X\\beta + u, \\quad u = \\lambda Wu + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method
        See :class:`SpatialModel`.

    Notes
    -----
    The ``priors`` dict supports the following keys:

    - ``lam_lower, lam_upper`` (float, default -1, 1): Bounds for the Uniform prior on lambda.
    - ``beta_mu`` (float, default 0): Prior mean for beta.
    - ``beta_sigma`` (float, default 1e6): Prior std for beta.
    - ``sigma_sigma`` (float, default 10): Scale for HalfNormal prior on sigma.
    - ``nu_lam`` (float, default 1/30): Rate for Exponential prior on
      :math:`\\nu` (only used when ``robust=True``).

    **Robust regression**

    When ``robust=True``, the spatially-filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon = (I - \\lambda W)(y - X\\beta) \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> "az.InferenceData":
        """Draw samples from the posterior. Accepts ``idata_kwargs`` for ArviZ compatibility.

        Parameters
        ----------
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` for InferenceData creation. If contains
            ``log_likelihood: True``, the complete pointwise log-likelihood
            (including the Jacobian correction) is attached to the output.
        Other parameters as in :class:`SpatialModel`.

        Notes
        -----
        The log-likelihood for the SEM model is:

        .. math::
            \\log p(y \\mid \\theta) =
            \\sum_{i=1}^{n} \\log \\mathcal{N}(\\varepsilon_i \\mid 0, \\sigma^2)
            + \\log |I - \\lambda W |

        where :math:`\\varepsilon = (I - \\lambda W)(y - X\\beta)`.

        Because the SEM model uses ``pm.Potential`` for both the Gaussian
        error log-likelihood and the Jacobian, neither term is auto-captured
        in the ``log_likelihood`` group by PyMC.  We compute the complete
        pointwise log-likelihood manually after sampling:

        .. math::
            \\ell_i = -\\frac{1}{2}\\left(\\frac{\\varepsilon_i}{\\sigma}\\right)^2
            - \\log(\\sigma) - \\frac{1}{2}\\log(2\\pi)
            + \\frac{1}{n} \\log |I - \\lambda W |
        """
        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))

        model = self._build_pymc_model()
        self._pymc_model = model
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            )

        # --- Compute complete pointwise log-likelihood ---
        # SEM uses pm.Potential for both Gaussian and Jacobian terms,
        # so nothing is auto-captured. We recompute from posterior draws.
        if compute_log_likelihood:
            import pytensor

            idata = self._idata
            n = self._y.shape[0]
            W = self._W_dense

            lam_draws = idata.posterior["lam"].values.reshape(-1)       # (n_draws,)
            beta_draws = idata.posterior["beta"].values.reshape(-1, self._X.shape[1])  # (n_draws, k)
            sigma_draws = idata.posterior["sigma"].values.reshape(-1)   # (n_draws,)

            # Residuals: resid = y - X @ beta.T  => (n_draws, n)
            resid = self._y[None, :] - (beta_draws @ self._X.T)  # (n_draws, n)

            # Spatially filtered residuals: eps = resid - lam * W @ resid
            eps = resid - lam_draws[:, None] * (resid @ W.T)  # (n_draws, n)

            # Pointwise log-likelihood for eps
            if self.robust:
                nu_draws = idata.posterior["nu"].values.reshape(-1)  # (n_draws,)
                from scipy.special import gammaln
                ll_gauss = (
                    gammaln((nu_draws[:, None] + 1) / 2)
                    - gammaln(nu_draws[:, None] / 2)
                    - 0.5 * np.log(nu_draws[:, None] * np.pi)
                    - np.log(sigma_draws[:, None])
                    - ((nu_draws[:, None] + 1) / 2)
                    * np.log1p((eps / sigma_draws[:, None]) ** 2 / nu_draws[:, None])
                )  # (n_draws, n)
            else:
                ll_gauss = (
                    -0.5 * (eps / sigma_draws[:, None]) ** 2
                    - np.log(sigma_draws[:, None])
                    - 0.5 * np.log(2 * np.pi)
                )  # (n_draws, n)

            # Jacobian contribution per draw: log|I - lam*W| / n (pure numpy)
            eigs = self._W_eigs.real.astype(np.float64)
            jacobian = np.array([np.sum(np.log(np.abs(1.0 - lv * eigs))) for lv in lam_draws])  # (n_draws,)
            ll_jac = jacobian[:, None] / n  # (n_draws, 1) broadcast to (n_draws, n)

            ll_total = ll_gauss + ll_jac  # (n_draws, n)

            # Reshape to (chains, draws, n)
            n_chains = idata.posterior.sizes["chain"]
            n_draws_per_chain = idata.posterior.sizes["draw"]
            ll_array = ll_total.reshape(n_chains, n_draws_per_chain, n)

            # Attach to idata — use explicit Dataset creation to ensure
            # "obs" is a data variable, not a coordinate.
            ll_da = xr.DataArray(ll_array, dims=("chain", "draw", "obs_dim"), name="obs")
            ll_ds = xr.Dataset({"obs": ll_da})
            idata["log_likelihood"] = ll_ds

        return self._idata

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SEM regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(
            self._W_eigs.real,
            method=self.logdet_method,
            rho_min=lam_lower,
            rho_max=lam_upper,
        )
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            resid = self._y - pt.dot(self._X, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                logp_eps = pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps)
            else:
                logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps)
            pm.Potential("eps_loglik", logp_eps.sum())
            pm.Potential("jacobian", logdet_fn(lam))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SEM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # For SEM, spatial multiplier does not apply to X directly.
        # Direct = beta, indirect = 0, total = beta.
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta


