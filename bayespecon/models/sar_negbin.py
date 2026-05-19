r"""Spatial autoregressive Negative Binomial (SAR-NB) model.

Count-outcome analogue of :class:`bayespecon.models.sar.SAR` with
NB2 observation noise.  The latent log-mean follows the SAR reduced form:

.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
    \log \mu_i = \left[(I - \rho W)^{-1} X \beta\right]_i.

No spatial Jacobian is needed because the spatial filter
:math:`(I - \rho W)^{-1}` parameterizes the latent mean, not the
observed data.  Unlike the Gaussian SAR model — where
:math:`\varepsilon \mapsto y` is a change-of-variables requiring
:math:`\log|I - \rho W|` — the Negative Binomial likelihood is
specified directly on the observed counts given :math:`\mu`, so
there is no Jacobian correction.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp

from ..diagnostics.lmtests import SAR_NEGBIN_SUITE
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
    analogue of the Gaussian SAR reduced form.  Unlike the Gaussian SAR,
    no spatial Jacobian ``log|I - rho * W|`` is needed because the
    spatial filter parameterizes the latent mean rather than defining
    a change-of-variables from a latent error to the observed data.
    Overdispersion is captured by the NB2 parameter ``alpha``.
    """

    _spatial_diagnostics_tests = SAR_NEGBIN_SUITE.tests

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
        bounds = self._logdet_bounds
        rho_lower = bounds.rho_min
        rho_upper = bounds.rho_max
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)

            # SAR-in-mean reduced form: eta = (I - rho*W)^{-1} X beta
            # Uses SparseSARSolveOp for efficient differentiable sparse LU.
            # Pass shared eigendecomposition cache so the JAX eigen path
            # can reuse it instead of computing its own O(n³) decomposition.
            from ..ops import SparseSARSolveOp

            _sar_solve_op = SparseSARSolveOp(
                self._W_sparse, eigendecomposition=self._W_eigendecomposition
            )
            Xbeta = pt.dot(self._X, beta)
            eta = _sar_solve_op(rho, Xbeta)
            mu = pm.Deterministic("mu", pt.exp(eta))
            pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=self._y_int)

            # No Jacobian term is needed for the SAR-in-mean reduced form.
            # The spatial filter is applied to the latent mean parameter,
            # not to the observed data, so there is no change-of-variables
            # correction.  Including log|I - rho W| would incorrectly
            # bias rho toward zero.
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
        """Sample posterior.

        The Negative Binomial log-likelihood is auto-captured by PyMC
        and is already complete — no Jacobian correction is needed
        because the spatial filter parameterizes the mean, not the
        observed data.
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
        from ..diagnostics.lmtests import _get_posterior_draws
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

        This implementation uses the shared eigendecomposition cache
        (:attr:`_W_eigendecomposition`) to avoid per-draw sparse LU
        factorisation, reducing complexity from :math:`O(n^3)` per draw to
        :math:`O(n^2)` per draw.
        """
        from ..diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        n = self._X.shape[0]
        ni = self._nonintercept_indices
        n_draws = rho_draws.shape[0]
        n_effects = len(ni)

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        # Use shared eigendecomposition cache (complex128 throughout).
        # Row-standardised W is generally non-symmetric, so V and Vinv
        # are complex.  Taking .real prematurely drops imaginary parts and
        # produces wrong results for eta, diag, and row sums.
        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        eigs_c = decomp[0]  # complex128, (n,)
        V_c = decomp[1]     # complex128, (n, n)
        Vinv_c = decomp[2]  # complex128, (n, n)

        # Precompute Vinv @ X (complex128, (n, k)) — reused for every draw
        VinvX = Vinv_c @ self._X.astype(np.complex128)  # (n, k)

        # Precompute Vinv @ 1 (complex128, (n,)) for row sums
        ones_c = np.ones(n, dtype=np.complex128)
        Vinv_ones = Vinv_c @ ones_c  # (n,)

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            inv_eigs_c = 1.0 / (1.0 - float(rho) * eigs_c)  # complex128

            # eta = V @ diag(inv_eigs) @ Vinv @ X @ beta
            coeff = inv_eigs_c * (VinvX @ beta.astype(np.complex128))  # (n,)
            eta = (V_c @ coeff).real.astype(np.float64)  # (n,)
            mu = np.exp(np.clip(eta, -50.0, 50.0))

            # diag((I - rho W)^{-1}) = diag(V @ diag(inv_eigs) @ Vinv)
            # = sum_j V_{ij} * Vinv_{ji} / (1 - rho lambda_j)
            # (element-wise product of V and Vinv^T, weighted by inv_eigs)
            multiplier_diag = (
                (V_c * Vinv_c.T) @ inv_eigs_c
            ).real.astype(np.float64)  # (n,)

            if self._is_row_std:
                multiplier_row_sums = np.full(
                    n, 1.0 / (1.0 - float(rho)), dtype=np.float64
                )
            else:
                # row_sum_i = (V @ diag(inv_eigs) @ Vinv @ 1)_i
                multiplier_row_sums = (
                    V_c @ (inv_eigs_c * Vinv_ones)
                ).real.astype(np.float64)  # (n,)

            direct_base = float(np.mean(mu * multiplier_diag))
            total_base = float(np.mean(mu * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    @staticmethod
    def _hutchinson_diag(
        A_solve: callable,
        n: int,
        n_probes: int = 20,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        r"""Estimate the diagonal of :math:`A^{-1}` via Hutchinson's method.

        Given a callable ``A_solve(b)`` that returns :math:`A^{-1} b``, this
        estimates :math:`\operatorname{diag}(A^{-1})` using Rademacher
        (±1) probe vectors:

        .. math::

            \widehat{d_i} = \frac{1}{K} \sum_{k=1}^{K}
                z_{ki} \, [A^{-1} z_k]_i,

        where :math:`z_k \sim \operatorname{Rademacher}(\pm 1)`.
        With 20 probes the relative error is typically < 5%.

        Parameters
        ----------
        A_solve : callable
            Function ``(b) -> A^{-1} b`` where b is (n,).
        n : int
            Dimension of the matrix.
        n_probes : int
            Number of Rademacher probe vectors.
        rng : numpy random Generator, optional
            Random state for reproducibility.

        Returns
        -------
        np.ndarray, shape (n,)
            Estimated diagonal of :math:`A^{-1}`.
        """
        if rng is None:
            rng = np.random.default_rng()
        diag_est = np.zeros(n, dtype=np.float64)
        for _ in range(n_probes):
            z = rng.choice(np.array([-1.0, 1.0]), size=n)
            Az = A_solve(z)
            diag_est += z * Az
        return diag_est / n_probes

    def _compute_count_scale_spatial_effects_posterior_sparse(
        self,
        rho_draws: np.ndarray,
        beta_draws: np.ndarray,
        n: int,
        ni: list[int],
        n_draws: int,
        n_effects: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Count-scale spatial effects via sparse solves + Hutchinson.

        For large W where eigenvectors are unavailable, this method uses:

        - ``scipy.sparse.linalg.spsolve`` for :math:`\eta = (I - \rho W)^{-1} X\beta`
        - Hutchinson diagonal estimator for :math:`\operatorname{diag}((I - \rho W)^{-1})`
        - ``spsolve`` with ones vector for row sums (non-row-standardised W)

        Complexity is :math:`O(\text{nnz} \cdot n_{\text{probes}})` per draw
        instead of :math:`O(n^2)` per draw for the eigen path.

        Parameters
        ----------
        rho_draws : np.ndarray, shape (G,)
        beta_draws : np.ndarray, shape (G, k)
        n : int
        ni : list[int]
        n_draws : int
        n_effects : int

        Returns
        -------
        direct_samples : np.ndarray, shape (G, n_effects)
        total_samples : np.ndarray, shape (G, n_effects)
        """
        W = self._W_sparse
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        rng = np.random.default_rng(42)  # fixed seed for reproducibility

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            rho_f = float(rho)
            A = I_n - rho_f * W

            # eta = (I - rho W)^{-1} X beta
            Xbeta = self._X @ beta
            eta = sp.linalg.spsolve(A, Xbeta)
            mu = np.exp(np.clip(eta, -50.0, 50.0))

            # diag((I - rho W)^{-1}) via Hutchinson
            multiplier_diag = self._hutchinson_diag(
                lambda b: sp.linalg.spsolve(A, b),
                n,
                n_probes=20,
                rng=rng,
            )

            # Row sums of (I - rho W)^{-1}
            if self._is_row_std:
                multiplier_row_sums = np.full(
                    n, 1.0 / (1.0 - rho_f), dtype=np.float64
                )
            else:
                multiplier_row_sums = sp.linalg.spsolve(A, ones)

            direct_base = float(np.mean(mu * multiplier_diag))
            total_base = float(np.mean(mu * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        return direct_samples, total_samples

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
