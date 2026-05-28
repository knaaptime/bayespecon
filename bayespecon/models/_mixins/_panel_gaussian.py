"""Mixin providing a unified ``_build_pymc_model`` for Gaussian panel models.

The mixin is driven by declarative attributes already present on the model
class (``_jacobian_param``, ``_has_wx_in_beta``).  Subclasses only need to
set those attributes and inherit from both ``SpatialPanelModel`` and
``PanelGaussianLikelihoodMixin``; the mixin supplies the full
``_build_pymc_model`` implementation.

Three likelihood branches are dispatched based on ``_jacobian_param``:

* ``None``  (OLS, SLX):  ``mu = Z @ β``,  ``pm.Normal/StudentT``, no potential.
* ``"rho"`` (SAR, SDM):  ``mu = ρ·Wy + Z @ β``,  ``pm.Normal/StudentT``,
  ``pm.Potential("jacobian", logdet(ρ))``.
* ``"lam"`` (SEM, SDEM):  spatially-filtered residual via ``pm.CustomDist``
  (JAX path) or ``pm.Potential`` (default path), plus Jacobian.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..._backends.sampler_helpers import use_jax_likelihood


class PanelGaussianLikelihoodMixin:
    """Mixin that provides ``_build_pymc_model`` for Gaussian panel models.

    Expects the host class to also inherit from
    :class:`~bayespecon.models.panel_base.SpatialPanelModel` and to define
    the following declarative attributes:

    * ``_jacobian_param``: ``None``, ``"rho"``, or ``"lam"``
    * ``_has_wx_in_beta``: ``bool``
    * ``robust``: ``bool``
    * ``priors``: dict-like with prior hyper-parameters

    And to provide these methods (inherited from ``SpatialPanelModel``):

    * ``_gelman_default_beta_prior(design, names)`` → ``(mu, sigma)``
    * ``_model_coords()`` → dict of coordinate arrays
    * ``_add_nu_prior(model)`` → adds ``nu`` to a ``pm.Model``
    * ``_logdet_pytensor_fn``: pytensor log-determinant callable
    * ``_sparse_panel_lag(X)``: compute (I_T ⊗ W) @ X for panel models
    """

    # ------------------------------------------------------------------
    # Design matrix
    # ------------------------------------------------------------------

    def _panel_gaussian_design_matrix(self) -> np.ndarray:
        """Return the effective design matrix for the panel Gaussian likelihood.

        For models with ``_has_wx_in_beta=True`` (SLX, SDM, SDEM), this
        stacks ``[X, WX]``.  Otherwise, returns ``X`` alone.
        """
        if self._has_wx_in_beta:
            return np.hstack([self._X, self._WX])
        return self._X

    def _panel_gaussian_design_names(self) -> list[str]:
        """Return feature names aligned with the design matrix columns."""
        if self._has_wx_in_beta:
            return list(self._feature_names) + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Prior extraction
    # ------------------------------------------------------------------

    def _panel_gaussian_priors(self, Z: np.ndarray, names: list[str]):
        """Extract prior hyper-parameters with Gelman defaults.

        Returns
        -------
        dict
            Keys: ``beta_mu``, ``beta_sigma``, ``sigma2_alpha``, ``sigma2_beta``,
            plus any spatial-param priors (``rho_lower``, ``rho_upper``,
            ``lam_lower``, ``lam_upper``).
        """
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(Z, names)
        return {
            "beta_mu": self.priors.get("beta_mu", default_beta_mu),
            "beta_sigma": self.priors.get("beta_sigma", default_beta_sigma),
            "sigma2_alpha": self.priors.get("sigma2_alpha", 2.0),
            "sigma2_beta": self.priors.get("sigma2_beta", float(np.var(self._y))),
            "rho_lower": self.priors.get("rho_lower", -1.0),
            "rho_upper": self.priors.get("rho_upper", 1.0),
            "lam_lower": self.priors.get("lam_lower", -1.0),
            "lam_upper": self.priors.get("lam_upper", 1.0),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _build_pymc_model(
        self,
        *,
        compute_log_likelihood: bool = False,
        nuts_sampler: str = "pymc",
    ) -> pm.Model:
        """Build the PyMC model for a Gaussian panel spatial model.

        Dispatches to the appropriate branch based on ``_jacobian_param``:

        * ``None``  → :meth:`_build_pymc_model_no_jacobian`
        * ``"rho"`` → :meth:`_build_pymc_model_rho`
        * ``"lam"`` → :meth:`_build_pymc_model_lam`

        Parameters
        ----------
        compute_log_likelihood : bool, default False
            Passed through to SAR/SDM models.  Ignored by OLS/SLX/SEM/SDEM.
        nuts_sampler : str, default ``"pymc"``
            Passed through to SEM/SDEM models to select JAX vs Potential path.
            Ignored by OLS/SLX/SAR/SDM.
        """
        jacobian_param = self._jacobian_param
        if jacobian_param is None:
            return self._build_pymc_model_no_jacobian()
        elif jacobian_param == "rho":
            return self._build_pymc_model_rho(
                compute_log_likelihood=compute_log_likelihood,
            )
        elif jacobian_param == "lam":
            return self._build_pymc_model_lam(nuts_sampler=nuts_sampler)
        else:
            raise ValueError(
                f"Unknown _jacobian_param={jacobian_param!r}; "
                f"expected None, 'rho', or 'lam'."
            )

    # ------------------------------------------------------------------
    # WX validation
    # ------------------------------------------------------------------

    def _validate_wx_columns(self) -> None:
        """Raise ValueError if the model requires WX columns but has none.

        Models with ``_has_wx_in_beta=True`` (SLX, SDM, SDEM) require at
        least one WX column.  This check provides a clear error message
        rather than a cryptic shape mismatch later.
        """
        if self._has_wx_in_beta and not self._wx_column_indices:
            model_name = type(self).__name__
            if self._jacobian_param == "rho":
                alt = "SAR"
            elif self._jacobian_param == "lam":
                alt = "SEM"
            else:
                alt = "OLS"
            raise ValueError(
                f"{model_name} requires at least one WX column. "
                f"Pass `w_vars=[...]` to choose which regressors receive "
                f"a spatial lag, or fit an {alt} model instead."
            )

    # ------------------------------------------------------------------
    # Branch 1: No Jacobian (OLS, SLX)
    # ------------------------------------------------------------------

    def _build_pymc_model_no_jacobian(self) -> pm.Model:
        """Build PyMC model for OLS or SLX panel (no spatial autoregressive term)."""
        self._validate_wx_columns()
        Z = self._panel_gaussian_design_matrix()
        names = self._panel_gaussian_design_names()
        priors = self._panel_gaussian_priors(Z, names)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal(
                "beta",
                mu=priors["beta_mu"],
                sigma=priors["beta_sigma"],
                dims="coefficient",
            )
            sigma2 = pm.InverseGamma(
                "sigma2",
                alpha=priors["sigma2_alpha"],
                beta=priors["sigma2_beta"],
            )
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            mu = pt.dot(Z, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    # ------------------------------------------------------------------
    # Branch 2: ρ Jacobian (SAR, SDM)
    # ------------------------------------------------------------------

    def _build_pymc_model_rho(
        self,
        *,
        compute_log_likelihood: bool = False,
    ) -> pm.Model:
        """Build PyMC model for SAR or SDM panel (spatial lag with ρ Jacobian)."""
        self._validate_wx_columns()
        Z = self._panel_gaussian_design_matrix()
        names = self._panel_gaussian_design_names()
        priors = self._panel_gaussian_priors(Z, names)

        logdet_fn = self._logdet_pytensor_fn

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform(
                "rho",
                lower=priors["rho_lower"],
                upper=priors["rho_upper"],
            )
            beta = pm.Normal(
                "beta",
                mu=priors["beta_mu"],
                sigma=priors["beta_sigma"],
                dims="coefficient",
            )
            sigma2 = pm.InverseGamma(
                "sigma2",
                alpha=priors["sigma2_alpha"],
                beta=priors["sigma2_beta"],
            )
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            # mu = rho * Wy + Z @ beta
            mu = rho * self._Wy + pt.dot(Z, beta)
            if self.robust:
                self._add_nu_prior(model)
                nu = model["nu"]
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

            # Jacobian: log|I - rho*W|
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    # ------------------------------------------------------------------
    # Branch 3: λ Jacobian (SEM, SDEM)
    # ------------------------------------------------------------------

    def _build_pymc_model_lam(self, *, nuts_sampler: str = "pymc") -> pm.Model:
        """Build PyMC model for SEM or SDEM panel (spatial error with λ Jacobian)."""
        self._validate_wx_columns()
        Z = self._panel_gaussian_design_matrix()
        names = self._panel_gaussian_design_names()
        priors = self._panel_gaussian_priors(Z, names)

        logdet_fn = self._logdet_pytensor_fn

        # Precompute W @ Z so the spatial filter can be expressed as
        #   eps = (y - lam*Wy) - (Z - lam*WZ) @ beta
        # avoiding any sparse matvec inside the NUTS gradient loop.
        cache_attr = "_WZ_panel_cache"
        if not hasattr(self, cache_attr) or getattr(self, cache_attr) is None:
            setattr(
                self,
                cache_attr,
                np.asarray(self._sparse_panel_lag(Z), dtype=np.float64),
            )
        WZ = getattr(self, cache_attr)

        n_obs = int(self._y.shape[0])
        inv_n = 1.0 / n_obs
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform(
                "lam",
                lower=priors["lam_lower"],
                upper=priors["lam_upper"],
            )
            beta = pm.Normal(
                "beta",
                mu=priors["beta_mu"],
                sigma=priors["beta_sigma"],
                dims="coefficient",
            )
            sigma2 = pm.InverseGamma(
                "sigma2",
                alpha=priors["sigma2_alpha"],
                beta=priors["sigma2_beta"],
            )
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            if self.robust:
                self._add_nu_prior(model)

            if jax_logp:
                # JAX path: register an observed RV via pm.CustomDist so PyMC
                # can capture ``log_likelihood`` natively.
                Wy_const = pt.as_tensor_variable(self._Wy)
                Z_const = pt.as_tensor_variable(Z)
                WZ_const = pt.as_tensor_variable(WZ)

                if self.robust:
                    nu = model["nu"]

                    def _logp(value, lam_, beta_, sigma_, nu_):
                        y_star = value - lam_ * Wy_const
                        Z_star = Z_const - lam_ * WZ_const
                        eps = y_star - pt.dot(Z_star, beta_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_),
                            eps,
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        nu,
                        logp=_logp,
                        observed=self._y,
                    )
                else:

                    def _logp(value, lam_, beta_, sigma_):
                        y_star = value - lam_ * Wy_const
                        Z_star = Z_const - lam_ * WZ_const
                        eps = y_star - pt.dot(Z_star, beta_)
                        log_dens = pm.logp(
                            pm.Normal.dist(mu=0.0, sigma=sigma_),
                            eps,
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        logp=_logp,
                        observed=self._y,
                    )
            else:
                # Default (C / Numba) path: benchmarked pm.Potential formulation.
                y_star = self._y - lam * self._Wy
                Z_star = Z - lam * WZ
                eps = y_star - pt.dot(Z_star, beta)
                if self.robust:
                    nu = model["nu"]
                    logp_eps = pm.logp(
                        pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma),
                        eps,
                    )
                else:
                    logp_eps = pm.logp(
                        pm.Normal.dist(mu=0.0, sigma=sigma),
                        eps,
                    )
                pm.Potential("eps_loglik", logp_eps.sum())
                pm.Potential("jacobian", logdet_fn(lam))

        return model
