r"""Spatial autoregressive Negative Binomial (SAR-NB) model.

Count-outcome analogue of :class:`bayespecon.models.sar.SAR` with
NB2 observation noise.  The latent log-mean follows the SAR reduced form:

.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
    \log \mu_i = \left[(I - \rho W)^{-1} X \beta\right]_i.

The spatial Jacobian :math:`\log|I-\rho W|` is included
via ``pm.Potential`` and must be added to pointwise log-likelihood when
computing WAIC/LOO.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import SpatialModel


class SARNegativeBinomial(SpatialModel):
    """Bayesian SAR model with a Negative Binomial likelihood.

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method
        Same interface and semantics as :class:`bayespecon.models.sar.SAR`.
    robust : bool, default False
        Not supported for count outcomes. If True, ``NotImplementedError`` is raised.

    Notes
    -----
    The model uses the SAR-in-mean reduced form
    ``log(mu) = (I - rho * W)^{-1} X @ beta``, which is the count-outcome
    analogue of the Gaussian SAR reduced form.  The spatial Jacobian
    ``log|I - rho * W|`` is included via ``pm.Potential``.
    Overdispersion is captured by the NB2 parameter ``alpha``.
    """

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_error_test"],
            ).bayesian_lm_error_test(m),
            "LM-Error",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_wx_test"],
            ).bayesian_lm_wx_test(m),
            "LM-WX",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_wx_test"],
            ).bayesian_robust_lm_wx_test(m),
            "Robust-LM-WX",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError(
                "robust=True is not supported for SARNegativeBinomial."
            )

        y_round = np.round(self._y).astype(np.int64)
        if not np.allclose(self._y, y_round):
            raise ValueError(
                "SARNegativeBinomial requires integer-valued observations."
            )
        if np.any(y_round < 0):
            raise ValueError(
                "SARNegativeBinomial requires non-negative integer observations."
            )

        self._y_int = y_round
        self._y = y_round.astype(np.float64)
        self._Wy = np.asarray(self._W_sparse @ self._y, dtype=np.float64)

    def _build_pymc_model(self) -> pm.Model:
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)

            # SAR-in-mean reduced form: eta = (I - rho*W)^{-1} X beta
            # Uses SparseSARSolveOp for efficient differentiable sparse LU.
            from ..ops import SparseSARSolveOp

            _sar_solve_op = SparseSARSolveOp(self._W_sparse)
            Xbeta = pt.dot(self._X, beta)
            eta = _sar_solve_op(rho, Xbeta)
            mu = pm.Deterministic("mu", pt.exp(eta))
            pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=self._y_int)

            pm.Potential("jacobian", self._logdet_pytensor_fn(rho))
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
        """Sample posterior and attach Jacobian-corrected pointwise log-likelihood."""
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
        if "log_likelihood" in idata.groups():
            self._attach_jacobian_corrected_log_likelihood(idata, "rho")
        return idata

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute average direct/indirect/total impacts on the log-mean scale."""
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        mean_row_sum = float(self._batch_mean_row_sum(np.array([rho]))[0])
        ni = self._nonintercept_indices
        direct = mean_diag * beta[ni]
        total = mean_row_sum * beta[ni]
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._nonintercept_feature_names,
        }

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior impacts on the log-mean scale for each draw."""
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws
        from ..diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        eigs = self._W_eigs.real.astype(np.float64)
        mean_diag = _chunked_eig_means(rho_draws, eigs)
        mean_row_sum = self._batch_mean_row_sum(rho_draws)

        ni = self._nonintercept_indices
        direct_samples = mean_diag[:, None] * beta_draws[:, ni]
        total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
        indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def _compute_count_scale_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute posterior impacts on the expected-count scale for each draw.

        Notes
        -----
        For the SAR-NB model with

        .. math::

            \mu = \exp\{(I - \rho W)^{-1} X\beta\},

        the average partial-effect matrix for covariate :math:`x_r` on the
        response scale is

        .. math::

            \frac{\partial \mu}{\partial x_r'} =
            \operatorname{diag}(\mu) (I - \rho W)^{-1} \beta_r.

        Direct, indirect, and total effects are the average diagonal, the
        average off-diagonal sum, and their sum respectively. This is more
        expensive than the log-mean-scale formula because it requires the
        diagonal of the spatial multiplier for each posterior draw.
        """
        from ..diagnostics.bayesian_lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        n = self._X.shape[0]
        ni = self._nonintercept_indices
        n_draws = rho_draws.shape[0]
        n_effects = len(ni)

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        identity = np.eye(n, dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        eye_sparse = sp.eye(n, format="csc", dtype=np.float64)

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            system = eye_sparse - float(rho) * self._W_sparse.tocsc()
            lu = spla.splu(system)

            xbeta = self._X @ beta
            eta = lu.solve(np.asarray(xbeta, dtype=np.float64))
            mu = np.exp(np.clip(eta, -50.0, 50.0))

            multiplier_diag = np.diag(lu.solve(identity))
            if self._is_row_std:
                multiplier_row_sums = np.full(
                    n, 1.0 / (1.0 - float(rho)), dtype=np.float64
                )
            else:
                multiplier_row_sums = lu.solve(ones)

            direct_base = float(np.mean(mu * multiplier_diag))
            total_base = float(np.mean(mu * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    def spatial_effects(
        self,
        return_posterior_samples: bool = False,
        scale: str = "logmean",
    ):
        r"""Compute Bayesian inference for direct, indirect, and total impacts.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, also return the posterior draws for each effect type.
        scale : {"logmean", "count"}, default "logmean"
            Scale on which impacts are reported.

            ``"logmean"`` returns the current default impacts on the linear
            predictor scale :math:`\log \mu`.

            ``"count"`` returns impacts on the expected-count scale
            :math:`\mu = \exp(\eta)`. This is exact but more expensive because
            it requires the diagonal of the spatial multiplier for each
            posterior draw.
        """
        from ..diagnostics.spatial_effects import _build_effects_dataframe

        if scale == "logmean":
            return super().spatial_effects(
                return_posterior_samples=return_posterior_samples
            )
        if scale != "count":
            raise ValueError("scale must be either 'logmean' or 'count'.")

        self._require_fit()
        direct_samples, indirect_samples, total_samples = (
            self._compute_count_scale_spatial_effects_posterior()
        )

        k_effects = direct_samples.shape[1]
        if (
            hasattr(self, "_wx_feature_names")
            and len(self._wx_feature_names) == k_effects
        ):
            feature_names = list(self._wx_feature_names)
        elif (
            hasattr(self, "_nonintercept_feature_names")
            and len(self._nonintercept_feature_names) == k_effects
        ):
            feature_names = list(self._nonintercept_feature_names)
        else:
            feature_names = list(self._feature_names[:k_effects])

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type=self.__class__.__name__,
        )
        df.attrs["scale"] = scale

        if return_posterior_samples:
            posterior_samples = {
                "direct": direct_samples,
                "indirect": indirect_samples,
                "total": total_samples,
            }
            return df, posterior_samples
        return df

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted expected counts."""
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        Xbeta = self._X @ beta
        n = self._X.shape[0]
        A = sp.eye(n, format="csr", dtype=np.float64) - rho * self._W_sparse
        eta = sp.linalg.spsolve(A, Xbeta)
        return np.exp(eta)
