"""Spatial Durbin Model (SDM).

y = rho * W @ y + X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

Combines a spatial lag on y (SAR) with spatially lagged covariates (SLX).
Jacobian log|I - rho*W| is required as in the SAR model.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from .base import SpatialModel


class SDM(SpatialModel):
    """Bayesian Spatial Durbin Model.

    .. math::
        y = \\rho Wy + X\\beta_1 + WX\\beta_2 + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method, w_vars
        See :class:`~bayespecon.models.base.SpatialModel`. Use ``w_vars`` to restrict which X columns
        are spatially lagged.

    Notes
    -----
    The ``priors`` dict supports the following keys:

    - ``rho_lower, rho_upper`` (float, default -1, 1): Bounds for the Uniform prior on rho.
    - ``beta_mu`` (float, default 0): Prior mean for beta.
    - ``beta_sigma`` (float, default 1e6): Prior std for beta.
    - ``sigma_sigma`` (float, default 10): Scale for HalfNormal prior on sigma.
    - ``nu_lam`` (float, default 1/30): Rate for Exponential prior on
      :math:`\\nu` (only used when ``robust=True``).

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu \\sim \\mathrm{TruncExp}(\\lambda_\\nu, \\mathrm{lower}=2)` with rate ``nu_lam`` (default 1/30).
    The default ``nu_lam = 1/30`` gives a prior mean of approximately 30,
    favouring near-Normal tails.  The lower bound of 2 ensures the
    variance exists.
    """

    _spatial_diagnostics_tests = [
        (lambda m: __import__(
            "bayespecon.diagnostics.bayesian_lmtests",
            fromlist=["bayesian_lm_error_test"],
        ).bayesian_lm_error_test(m), "LM-Error"),
    ]

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self, compute_log_likelihood: bool = False) -> pm.Model:
        """Construct the PyMC model for SDM regression.

        Parameters
        ----------
        compute_log_likelihood : bool, default False
            If True, store pointwise log-likelihood (not used in SDM since
            the Jacobian correction is applied post-sampling in ``fit()``).

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        k = self._X.shape[1]
        Z = np.hstack([self._X, self._WX])  # (n, 2k)

        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho * self._Wy + pt.dot(Z, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
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
        """
        Draw samples from the posterior. Accepts ``idata_kwargs`` for ArviZ compatibility.

        Parameters
        ----------
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` for InferenceData creation. If contains
            ``log_likelihood: True``, the complete pointwise log-likelihood
            (including the Jacobian correction) is attached to the output.
        Other parameters as in :class:`~bayespecon.models.base.SpatialModel`.

        Notes
        -----
        The log-likelihood for the SDM model is:

        .. math::
            \\log p(y \\mid \\theta) =
            \\sum_{i=1}^{n} \\log \\mathcal{N}(y_i \\mid \\mu_i, \\sigma^2)
            + \\log |I - \\rho W |

        where :math:`\\mu = \\rho W y + Z \\beta` and
        :math:`Z = [X, WX]`.

        As with the SAR model, ``pm.Normal`` with ``observed`` auto-captures
        the Gaussian part, while the Jacobian :math:`\\log |I - \\rho W|` is
        added via ``pm.Potential`` and is absent from the ``log_likelihood``
        group.  To enable WAIC/LOO and Bayes factor comparison, we correct
        the pointwise log-likelihood after sampling:

        .. math::
            \\ell_i = -\\frac{1}{2}\\left(\\frac{y_i - \\mu_i}{\\sigma}\\right)^2
            + \\frac{1}{n} \\log |I - \\rho W |
        """
        from ..logdet import make_logdet_fn
        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))

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
            Z = np.hstack([self._X, self._WX])  # (n, 2k)

            # Posterior draws: shape (chains, draws, ...)
            rho_draws = idata.posterior["rho"].values.reshape(-1)    # (n_draws,)
            beta_draws = idata.posterior["beta"].values.reshape(-1, Z.shape[1])  # (n_draws, 2k)
            sigma_draws = idata.posterior["sigma"].values.reshape(-1)   # (n_draws,)

            # Mean: mu = rho * Wy + Z @ beta.T  => (n_draws, n)
            mu = rho_draws[:, None] * self._Wy[None, :] + (beta_draws @ Z.T)  # (n_draws, n)
            resid = self._y[None, :] - mu  # (n_draws, n)

            # Pointwise log-likelihood
            if self.robust:
                nu_draws = idata.posterior["nu"].values.reshape(-1)  # (n_draws,)
                from scipy.special import gammaln
                ll_gauss = (
                    gammaln((nu_draws[:, None] + 1) / 2)
                    - gammaln(nu_draws[:, None] / 2)
                    - 0.5 * np.log(nu_draws[:, None] * np.pi)
                    - np.log(sigma_draws[:, None])
                    - ((nu_draws[:, None] + 1) / 2)
                    * np.log1p((resid / sigma_draws[:, None]) ** 2 / nu_draws[:, None])
                )  # (n_draws, n)
            else:
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
        """Compute SDM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # SDM impacts: S_k = (I - rho*W)^{-1} (beta1_k*I + beta2_k*W)
        # Direct   = mean diagonal of S_k
        # Total    = mean row sum of S_k
        # Indirect = total - direct
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1, beta2 = beta[:k], beta[k:k + kw]

        eigs = self._W_eigs
        inv_eigs = 1.0 / (1.0 - rho * eigs)
        mean_diag_M = float(np.mean(inv_eigs.real))
        mean_diag_MW = float(np.mean((eigs * inv_eigs).real))
        if self._is_row_std:
            mean_row_sum_M = 1.0 / (1.0 - rho)
            mean_row_sum_MW = mean_row_sum_M
        else:
            ones = np.ones(self._W_sparse.shape[0])
            A = np.eye(self._W_sparse.shape[0]) - rho * self._W_sparse.toarray()
            M_ones = np.linalg.solve(A, ones)
            mean_row_sum_M = float(M_ones.mean())
            mean_row_sum_MW = float((self._W_sparse.toarray() @ M_ones).mean())
        direct = np.array([
            beta1[j] * mean_diag_M + b2 * mean_diag_MW
            for j, b2 in zip(self._wx_column_indices, beta2)
        ])
        total = np.array([
            beta1[j] * mean_row_sum_M + b2 * mean_row_sum_MW
            for j, b2 in zip(self._wx_column_indices, beta2)
        ])
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._wx_feature_names,
        }

    def _compute_spatial_effects_posterior(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SDM model, the impact measures for covariate :math:`k` are:

        .. math::
            S_k^{(g)} = (I - \\rho^{(g)} W)^{-1}
            (\\beta_{1j}^{(g)} I + \\beta_{2k}^{(g)} W)

            \\text{Direct}_k^{(g)} = \\overline{\\text{diag}}(S_k^{(g)})

            \\text{Total}_k^{(g)} = \\overline{\\text{rowsum}}(S_k^{(g)})

            \\text{Indirect}_k^{(g)} = \\text{Total}_k^{(g)} - \\text{Direct}_k^{(g)}

        where :math:`j` is the index of covariate :math:`k` in :math:`X`,
        and :math:`\\beta_1, \\beta_2` are the coefficients on :math:`X` and
        :math:`WX` respectively.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each
            of shape ``(G, k_wx)`` where *G* is the total number of posterior
            draws and *k_wx* is the number of spatially lagged covariates.
        """
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")  # (G,)
        beta_draws = _get_posterior_draws(idata, "beta")  # (G, k+k_wx)
        G = rho_draws.shape[0]
        k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1_draws = beta_draws[:, :k]  # (G, k)
        beta2_draws = beta_draws[:, k:k + kw]  # (G, kw)

        eigs = self._W_eigs.real.astype(np.float64)  # (n,)

        # Vectorised eigenvalue computation over draws
        inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])  # (G, n)
        mean_diag_M = np.mean(inv_eigs, axis=1)  # (G,)
        mean_diag_MW = np.mean(eigs[None, :] * inv_eigs, axis=1)  # (G,)

        if self._is_row_std:
            mean_row_sum_M = 1.0 / (1.0 - rho_draws)  # (G,)
            mean_row_sum_MW = mean_row_sum_M  # row sums of M*W = row sums of M for row-std W
        else:
            n = self._W_sparse.shape[0]
            W_dense = self._W_dense
            ones = np.ones(n)
            mean_row_sum_M = np.empty(G)
            mean_row_sum_MW = np.empty(G)
            for g in range(G):
                A = np.eye(n) - rho_draws[g] * W_dense
                M_ones = np.linalg.solve(A, ones)
                mean_row_sum_M[g] = M_ones.mean()
                mean_row_sum_MW[g] = (W_dense @ M_ones).mean()

        # For each lagged covariate k (with index j in X):
        # Direct_k = beta1_j * mean_diag_M + beta2_k * mean_diag_MW
        # Total_k  = beta1_j * mean_row_sum_M + beta2_k * mean_row_sum_MW
        wx_idx = self._wx_column_indices
        direct_samples = np.column_stack([
            beta1_draws[:, j] * mean_diag_M + beta2_draws[:, idx] * mean_diag_MW
            for idx, j in enumerate(wx_idx)
        ])  # (G, kw)
        total_samples = np.column_stack([
            beta1_draws[:, j] * mean_row_sum_M + beta2_draws[:, idx] * mean_row_sum_MW
            for idx, j in enumerate(wx_idx)
        ])  # (G, kw)
        indirect_samples = total_samples - direct_samples  # (G, kw)

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
        Z = np.hstack([self._X, self._WX])
        return rho * self._Wy + Z @ beta


