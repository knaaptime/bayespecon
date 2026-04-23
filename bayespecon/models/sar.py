"""Spatial Autoregressive (SAR / Spatial Lag) Model.

y = rho * W @ y + X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)

The full likelihood includes the Jacobian |I - rho*W|, added via
pm.Potential so that NUTS samples from the correct posterior.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from .base import SpatialModel


class SAR(SpatialModel):
    """Bayesian Spatial Autoregressive (Spatial Lag) model.

    .. math::
        y = \\rho Wy + X\\beta + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method, w_vars
        See :class:`SpatialModel` for descriptions.

    Notes
    -----
    The ``priors`` dict supports the following keys:

    - ``rho_lower, rho_upper`` (float, default -1, 1): Bounds for the Uniform prior on rho.
    - ``beta_mu`` (float, default 0): Prior mean for beta.
    - ``beta_sigma`` (float, default 1e6): Prior std for beta.
    - ``sigma_sigma`` (float, default 10): Prior std for sigma.
    """

    def _build_pymc_model(self, compute_log_likelihood: bool = False) -> pm.Model:
        """Construct the PyMC model for SAR regression.

        Parameters
        ----------
        compute_log_likelihood : bool, default False
            If True, store pointwise log-likelihood (not used in SAR since
            the Jacobian correction is applied post-sampling in ``fit()``).

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        k = self._X.shape[1]

        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # mu = rho*Wy + X@beta  (Wy is fixed observed data here)
            mu = rho * self._Wy + pt.dot(self._X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

            # Jacobian: log|I - rho*W|
            # Build the expression inline so pytensor can resolve rho as a graph input.
            # eigs is a constant (no gradient w.r.t. it), so it does not cause
            # MissingInputError — only rho does, and it's properly defined above.
            eigs = self._W_eigs.real.astype(np.float64)
            pm.Potential("jacobian", pt.sum(pt.log(pt.abs(1.0 - rho * pt.as_tensor_variable(eigs)))))

        return model

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
        r"""
        Draw samples from the posterior. Accepts ``idata_kwargs`` for ArviZ compatibility.

        Parameters
        ----------
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` for InferenceData creation. If contains
            ``log_likelihood: True``, the complete pointwise log-likelihood
            (including the Jacobian correction) is attached to the output.
        Other parameters as in :class:`SpatialModel`.

        Notes
        -----
        The log-likelihood for the SAR model is:

        .. math::
            \log p(y \mid \theta) =
            \sum_{i=1}^{n} \log \mathcal{N}(y_i \mid \mu_i, \sigma^2)
            + \log |I - \rho W |

        The ``pm.Normal`` with ``observed=self._y`` automatically captures
        the first term (the Gaussian log-likelihood) in ``log_likelihood``.
        However, the Jacobian term :math:`\log |I - \rho W|` is added via
        ``pm.Potential`` and does **not** appear in the auto-computed
        ``log_likelihood`` group.

        For correct WAIC/LOO computation (and therefore Bayes factor
        comparison via bridge sampling), we construct the complete
        pointwise log-likelihood manually after sampling:

        .. math::
            \ell_i = -\frac{1}{2}\left(\frac{y_i - \mu_i}{\sigma}\right)^2
            + \frac{1}{n} \log |I - \rho W |

        where :math:`\mu_i = \rho (Wy)_i + x_i' \beta` and the Jacobian
        contribution is divided by :math:`n` so that
        :math:`\sum_{i=1}^{n} \ell_i` equals the total log-likelihood
        used for sampling.
        """
        from ..logdet import make_logdet_fn

        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))

        # Build model with log_likelihood computation if requested
        model = self._build_pymc_model(compute_log_likelihood=compute_log_likelihood)
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

        # --- Correct log_likelihood: add Jacobian contribution ---
        # The pm.Normal("obs") auto-captures the Gaussian part, but the
        # Jacobian log|I - rho*W| (added via pm.Potential) is absent.
        # We recompute the complete pointwise LL and overwrite the group.
        if compute_log_likelihood and hasattr(self, "_idata"):
            import xarray as xr

            idata = self._idata
            n = self._y.shape[0]

            # Posterior draws: shape (chains, draws, ...)
            rho_draws = idata.posterior["rho"].values.reshape(-1)    # (n_draws,)
            beta_draws = idata.posterior["beta"].values.reshape(-1, self._X.shape[1])  # (n_draws, k)
            sigma_draws = idata.posterior["sigma"].values.reshape(-1)   # (n_draws,)

            # Residuals per draw: resid = y - rho*Wy - X@beta
            # Shapes: y (n,), Wy (n,), X (n, k)
            # mu = rho * Wy + X @ beta.T  => (n_draws, n)
            mu = rho_draws[:, None] * self._Wy[None, :] + (beta_draws @ self._X.T)  # (n_draws, n)
            resid = self._y[None, :] - mu  # (n_draws, n)

            # Pointwise Gaussian log-likelihood
            ll_gauss = -0.5 * (resid / sigma_draws[:, None]) ** 2 - np.log(sigma_draws[:, None]) - 0.5 * np.log(2 * np.pi)  # (n_draws, n)

            # Jacobian contribution per draw: log|I - rho*W| / n (pure numpy)
            eigs = self._W_eigs.real.astype(np.float64)
            jacobian = np.array([np.sum(np.log(np.abs(1.0 - rv * eigs))) for rv in rho_draws])  # (n_draws,)
            ll_jac = jacobian[:, None] / n  # (n_draws, 1) broadcast to (n_draws, n)

            ll_total = ll_gauss + ll_jac  # (n_draws, n)

            # Reshape to (chains, draws, n)
            n_chains = idata.posterior.sizes["chain"]
            n_draws_per_chain = idata.posterior.sizes["draw"]
            ll_array = ll_total.reshape(n_chains, n_draws_per_chain, n)

            # Attach to idata — replace the entire log_likelihood group with complete LL
            # (pm.sample auto-captures Gaussian LL; we replace with complete LL)
            #
            # Use explicit Dataset creation to ensure "obs" is a data variable, not a coordinate.
            # Assigning via idata["log_likelihood"] = xr.Dataset({"obs": da}) can silently
            # treat "obs" as a dimension coordinate if da.dims includes "obs" as a dim name,
            # causing data_vars to be empty and breaking az.loo() / az.waic().
            ll_da = xr.DataArray(ll_array, dims=("chain", "draw", "obs_dim"), name="obs")
            ll_ds = xr.Dataset({"obs": ll_da})
            idata["log_likelihood"] = ll_ds

        return self._idata

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
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._feature_names,
        }

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


