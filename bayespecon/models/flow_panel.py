"""Panel extensions for Bayesian spatial flow (origin-destination) models.

This module introduces a panel-flow base class and four panel model
variants that extend the cross-sectional flow models to balanced panel data.

The panel stack uses time-first ordering. For each period t, the response is
an n^2-length vectorized origin-destination flow array, and all periods are
stacked to length n^2 * T.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp
from libpysal.graph import Graph

from ..graph import _validate_graph, flow_weight_matrices
from ..logdet import (
    _flow_logdet_poly_coeffs,
    compute_flow_traces,
    flow_logdet_pytensor,
    make_flow_separable_logdet,
)
from ..ops import kron_solve_matrix
from .flow import (
    _build_flow_effect_masks,
    _compute_flow_effects_lesage,
)
from .panel_base import _demean_panel


class FlowPanelModel(ABC):
    """Abstract base class for balanced panel spatial flow models.

    Parameters
    ----------
    y : array-like
        Stacked panel response in one of these forms:
        - shape (T, n, n)
        - shape (T, n^2)
        - shape (n^2 * T,)
    G : libpysal.graph.Graph
        Row-standardized graph on n units.
    X : np.ndarray or pandas.DataFrame, shape (n^2 * T, p)
        Stacked panel design matrix in time-first order.
    T : int
        Number of panel periods.
    col_names : list[str], optional
        Feature names for X.
    k : int, optional
        Number of destination/origin covariate pairs used by flow effects.
        If omitted, inferred from column names with "dest_" prefix.
    model : int, default 0
        Fixed-effects transform mode:
        0 pooled, 1 pair FE, 2 time FE, 3 two-way FE.
    priors : dict, optional
        Prior overrides.
    logdet_method : str, default "traces"
        Flow log-determinant method.
    restrict_positive : bool, default True
        If True, use simplex-constrained rho parameters.
    robust : bool, default False
        If True, use Student-t observation errors.
    miter, titer, trace_riter, trace_seed
        Flow logdet approximation controls.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series],
        G: Graph,
        X: Union[np.ndarray, pd.DataFrame],
        T: int,
        col_names: Optional[list[str]] = None,
        k: Optional[int] = None,
        model: int = 0,
        priors: Optional[dict] = None,
        logdet_method: str = "traces",
        restrict_positive: bool = True,
        robust: bool = False,
        miter: int = 30,
        titer: int = 800,
        trace_riter: int = 50,
        trace_seed: Optional[int] = None,
    ):
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.restrict_positive = restrict_positive
        self.robust = robust
        self.model = int(model)
        if self.model not in (0, 1, 2, 3):
            raise ValueError("model must be one of {0,1,2,3}.")

        self.miter = miter
        self.titer = titer
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None
        self._approximation = None

        # Validate and extract n x n W
        self._W_sparse: sp.csr_matrix = _validate_graph(G)
        self._n: int = self._W_sparse.shape[0]
        self._N_flow: int = self._n * self._n

        # Validate T
        self._T: int = int(T)
        if self._T <= 0:
            raise ValueError(f"T must be positive, got {T}.")

        # Validate y
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 3:
            expected = (self._T, self._n, self._n)
            if y_arr.shape != expected:
                raise ValueError(
                    f"y with 3 dims must have shape {expected}, got {y_arr.shape}."
                )
            y_vec = y_arr.reshape(self._T, self._N_flow).reshape(-1)
        elif y_arr.ndim == 2:
            if y_arr.shape == (self._T, self._N_flow):
                y_vec = y_arr.reshape(-1)
            elif y_arr.shape == (self._n, self._n) and self._T == 1:
                y_vec = y_arr.ravel()
            else:
                raise ValueError(
                    "y with 2 dims must have shape (T, n^2) or (n, n) when T=1. "
                    f"Got {y_arr.shape}."
                )
        elif y_arr.ndim == 1:
            expected_len = self._N_flow * self._T
            if y_arr.shape[0] != expected_len:
                raise ValueError(
                    f"y vector must have length n^2*T={expected_len}, got {y_arr.shape[0]}."
                )
            y_vec = y_arr
        else:
            raise ValueError("y must be a 1-D, 2-D, or 3-D array.")
        self._y_raw = y_vec

        # Validate X
        if isinstance(X, pd.DataFrame):
            if col_names is None:
                col_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]

        expected_rows = self._N_flow * self._T
        if X_arr.shape[0] != expected_rows:
            raise ValueError(
                f"X must have n^2*T={expected_rows} rows, got {X_arr.shape[0]}."
            )
        self._X_raw = X_arr

        if col_names is not None:
            self._feature_names: list[str] = list(col_names)
        elif X_arr.shape[1] == 0:
            self._feature_names = []
        else:
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        if k is not None:
            self._k: int = int(k)
        else:
            dest_cols = [
                name for name in self._feature_names if name.startswith("dest_")
            ]
            self._k = len(dest_cols)

        # Demean panel data using N_flow panel units (OD pairs)
        self._y, self._X = _demean_panel(
            self._y_raw,
            self._X_raw,
            self._N_flow,
            self._T,
            self.model,
        )

        # Keep aliases matching flow model naming
        self._y_vec = self._y
        self._X_design = self._X

        # Build flow weight matrices on N_flow = n^2 system
        wms = flow_weight_matrices(G)
        self._Wd: sp.csr_matrix = wms["destination"]
        self._Wo: sp.csr_matrix = wms["origin"]
        self._Ww: sp.csr_matrix = wms["network"]

        # Cache region-shock masks for LeSage effects decomposition.
        self._dmask, self._omask, self._imask = _build_flow_effect_masks(self._n)

        # Spatial lags on demeaned/stationary panel stack
        self._Wd_y = self._sparse_flow_panel_lag(self._y, self._Wd)
        self._Wo_y = self._sparse_flow_panel_lag(self._y, self._Wo)
        self._Ww_y = self._sparse_flow_panel_lag(self._y, self._Ww)

        # Pre-compute logdet data for separable constraint: log|Lo⊗Ld| = n*f(ρ_d) + n*f(ρ_o).
        # Also keep _W_eigs for backward compatibility.
        self._W_eigs: Optional[np.ndarray] = None
        self._separable_logdet_fn = None
        _SEPARABLE_METHODS = {"separable", "eigenvalue", "chebyshev", "mc_poly"}
        if self.logdet_method in _SEPARABLE_METHODS:
            _method_key = (
                "eigenvalue"
                if self.logdet_method == "separable"
                else self.logdet_method
            )
            self._separable_logdet_fn = make_flow_separable_logdet(
                self._W_sparse,
                self._n,
                method=_method_key,
                miter=miter,
                riter=trace_riter,
                random_state=trace_seed,
            )
            if self.logdet_method in ("separable", "eigenvalue"):
                self._W_eigs = np.linalg.eigvals(
                    self._W_sparse.toarray().astype(np.float64)
                ).real

        # Precompute traces for unrestricted flow Jacobian when requested
        if self.logdet_method == "traces":
            self._traces = compute_flow_traces(
                self._W_sparse,
                miter=self.miter,
                riter=trace_riter,
                random_state=trace_seed,
            )
            (
                self._poly_a,
                self._poly_b,
                self._poly_c,
                self._poly_coeffs,
                self._miter_a,
                self._miter_b,
                self._miter_c,
                self._miter_coeffs,
            ) = _flow_logdet_poly_coeffs(self._traces, self._n, self.miter)
        else:
            self._traces = None

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return the PyMC model."""

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute posterior effects per draw."""

    def _posterior_var_names(
        self,
        model: pm.Model,
        *,
        store_lambda: bool,
    ) -> list[str]:
        names = [rv.name for rv in model.free_RVs]
        names.extend(
            var.name
            for var in model.deterministics
            if store_lambda or var.name != "lambda"
        )
        return list(dict.fromkeys(name for name in names if name is not None))

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior."""
        model = self._build_pymc_model()
        self._pymc_model = model
        self._approximation = None
        if "var_names" not in sample_kwargs and not store_lambda:
            sample_kwargs["var_names"] = self._posterior_var_names(
                model,
                store_lambda=False,
            )
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                **sample_kwargs,
            )
        return self._idata

    def fit_approx(
        self,
        draws: int = 2000,
        n: int = 10000,
        method: str = "advi",
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        **fit_kwargs,
    ) -> az.InferenceData:
        """Fit a variational approximation and return posterior draws."""
        method = method.lower()
        if method not in {"advi", "fullrank_advi"}:
            raise ValueError("fit_approx method must be 'advi' or 'fullrank_advi'.")

        model = self._build_pymc_model()
        self._pymc_model = model
        with model:
            self._approximation = pm.fit(
                n=n,
                method=method,
                random_seed=random_seed,
                **fit_kwargs,
            )
            self._idata = self._approximation.sample(
                draws=draws,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        if (
            not store_lambda
            and self._idata is not None
            and hasattr(self._idata, "posterior")
            and "lambda" in self._idata.posterior.data_vars
        ):
            self._idata.posterior = self._idata.posterior.drop_vars("lambda")

        return self._idata

    @property
    def inference_data(self) -> Optional[az.InferenceData]:
        """Return posterior draws from the most recent fit."""
        return self._idata

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the most recently built PyMC model."""
        return self._pymc_model

    @property
    def approximation(self):
        """Return the most recent PyMC variational approximation, if any."""
        return self._approximation

    def summary(self, var_names: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """Return posterior summary table via ArviZ."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        return az.summary(self._idata, var_names=var_names, **kwargs)

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        """Return named dimensions for PyMC model coordinates."""
        coords = {"coefficient": self._feature_names}
        if extra:
            coords.update(extra)
        return coords

    def _add_nu_prior(self):
        """Add Student-t degrees-of-freedom prior for robust models."""
        nu_lam = self.priors.get("nu_lam", 1.0 / 30.0)
        return pm.Truncated("nu", pm.Exponential.dist(lam=nu_lam), lower=2.0)

    def _assemble_A(self, rho_d: float, rho_o: float, rho_w: float) -> sp.csr_matrix:
        """Assemble A = I - rho_d*Wd - rho_o*Wo - rho_w*Ww for one period."""
        eye_n = sp.eye(self._N_flow, format="csr", dtype=np.float64)
        return eye_n - rho_d * self._Wd - rho_o * self._Wo - rho_w * self._Ww

    def _sparse_flow_panel_lag(
        self, v: np.ndarray, W_flow: sp.csr_matrix
    ) -> np.ndarray:
        """Apply panel flow lag I_T kron W_flow to time-first stacked vector."""
        chunks = v.reshape(self._T, self._N_flow)
        return np.asarray((W_flow @ chunks.T).T, dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    def spatial_effects(
        self,
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Summarise posterior origin/destination/intra/network/total effects.

        Wraps :meth:`_compute_spatial_effects_posterior` to produce a tidy
        DataFrame with posterior means, credible-interval bounds, and Bayesian
        p-values for each effect type.  Effects are evaluated on the
        one-period :math:`n^2 \\times n^2` flow system (time-invariant under
        static panel parameters).

        Parameters
        ----------
        draws : int, optional
            Maximum number of posterior draws to use.  Defaults to all.
        return_posterior_samples : bool, default False
            If True, also return the underlying posterior-draw arrays.
        ci : float, default 0.95
            Credible-interval coverage.

        Returns
        -------
        pandas.DataFrame, or (DataFrame, dict)
        """
        from ..diagnostics.spatial_effects import _compute_bayesian_pvalue

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        if self._k == 0:
            raise RuntimeError(
                "Cannot compute spatial effects: no `dest_*` columns detected "
                "in the design matrix.  Pass `k=` explicitly when constructing "
                "the model."
            )

        posterior = self._compute_spatial_effects_posterior(draws=draws)

        feature_names = [
            name[len("dest_") :] if name.startswith("dest_") else name
            for name in self._feature_names
            if name.startswith("dest_")
        ][: self._k]
        if len(feature_names) != self._k:
            feature_names = [f"x{i}" for i in range(self._k)]

        alpha = (1.0 - ci) / 2.0
        rows = []
        for effect_name, samples in posterior.items():
            means = samples.mean(axis=0)
            lower = np.quantile(samples, alpha, axis=0)
            upper = np.quantile(samples, 1.0 - alpha, axis=0)
            pvals = _compute_bayesian_pvalue(samples)
            for j, fname in enumerate(feature_names):
                rows.append(
                    {
                        "predictor": fname,
                        "effect": effect_name,
                        "mean": float(means[j]),
                        "ci_lower": float(lower[j]),
                        "ci_upper": float(upper[j]),
                        "bayes_pvalue": float(pvals[j]),
                    }
                )

        df = pd.DataFrame(rows).set_index(["predictor", "effect"])
        if return_posterior_samples:
            return df, posterior
        return df

    def _simulate_y_rep_period(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw a single posterior-predictive replicate for the full panel.

        Default Gaussian implementation: :math:`y_{rep,t} = A^{-1}(X_t \\beta + \\sigma\\varepsilon_t)`
        for each period ``t``, with a single sparse :math:`LU` factorisation
        of :math:`A` reused across periods.  Subclasses (Poisson variants)
        override this method.
        """
        N = self._N_flow
        T = self._T
        A = self._assemble_A(rho_d, rho_o, rho_w).tocsc()
        lu = sp.linalg.splu(A)
        Xb = self._X_design @ beta  # (N*T,)
        Xb_mat = Xb.reshape(T, N).T  # (N, T)
        if sigma is not None:
            noise = rng.normal(scale=float(sigma), size=(N, T))
            rhs = Xb_mat + noise
        else:
            rhs = Xb_mat
        out = lu.solve(rhs)  # (N, T)
        return out.T.reshape(-1)  # back to time-first stacked vector

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive samples ``y_rep`` for the full panel stack.

        Parameters
        ----------
        n_draws : int, optional
            Number of posterior draws to use.  Defaults to all.
        random_seed : int, optional
            Seed for the noise/Poisson sampler.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_draws, N_flow * T)`` with posterior-predictive
            flows in time-first stacked order.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        rho_d = post["rho_d"].values.reshape(-1)
        rho_o = post["rho_o"].values.reshape(-1)
        rho_w = post["rho_w"].values.reshape(-1)
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = len(rho_d)
        if n_draws is not None:
            total = min(int(n_draws), total)
            rho_d = rho_d[:total]
            rho_o = rho_o[:total]
            rho_w = rho_w[:total]
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep_period(
                float(rho_d[g]),
                float(rho_o[g]),
                float(rho_w[g]),
                beta_draws[g],
                sigma_g,
                rng,
            )
        return out

    # ------------------------------------------------------------------
    # Internal effects helpers
    # ------------------------------------------------------------------

    def _compute_flow_effects_from_draws(
        self,
        rho_d_draws: np.ndarray,
        rho_o_draws: np.ndarray,
        rho_w_draws: np.ndarray,
        beta_draws: np.ndarray,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute LeSage flow effects from posterior draws.

        Effects are computed using one-period :math:`n^2 \\times n^2` system
        matrices, which are time-invariant under static panel parameters.  See
        :func:`~bayespecon.models.flow._compute_flow_effects_lesage` for the
        decomposition.  One sparse :math:`LU` factorisation per draw covers all
        :math:`n` shock columns and all :math:`k` predictors.
        """
        n = self._n
        k = self._k

        dest_start = 2
        orig_start = 2 + k

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            rho_w_draws = rho_w_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        out = {
            key: np.zeros((n_draws_total, k), dtype=np.float64)
            for key in ("origin", "destination", "intra", "network", "total")
        }

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            rw = float(rho_w_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k]

            A = self._assemble_A(rd, ro, rw).tocsc()
            lu = sp.linalg.splu(A)

            def _solve(rhs: np.ndarray, _lu=lu) -> np.ndarray:
                return _lu.solve(rhs)

            res = _compute_flow_effects_lesage(
                _solve,
                self._dmask,
                self._omask,
                self._imask,
                beta_d_vec,
                beta_o_vec,
                n,
                k,
            )
            for key, arr in res.items():
                out[key][idx, :] = arr

        return out

    def _compute_flow_effects_kron(
        self,
        rho_d_draws: np.ndarray,
        rho_o_draws: np.ndarray,
        beta_draws: np.ndarray,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute LeSage flow effects via Kronecker-factored solve.

        Replaces the :math:`N\\times N` sparse factorisation in
        :meth:`_compute_flow_effects_from_draws` with two :math:`n\\times n`
        solves via :func:`~bayespecon.ops.kron_solve_matrix`, exploiting
        :math:`A = L_o \\otimes L_d`.
        """
        n = self._n
        k = self._k
        W = self._W_sparse.tocsr()
        I_n = sp.eye(n, format="csr", dtype=np.float64)

        dest_start = 2
        orig_start = 2 + k

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        out = {
            key: np.zeros((n_draws_total, k), dtype=np.float64)
            for key in ("origin", "destination", "intra", "network", "total")
        }

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k]

            Ld = (I_n - rd * W).tocsr()
            Lo = (I_n - ro * W).tocsr()

            def _solve(rhs: np.ndarray, _Lo=Lo, _Ld=Ld, _n=n) -> np.ndarray:
                return kron_solve_matrix(_Lo, _Ld, rhs, _n)

            res = _compute_flow_effects_lesage(
                _solve,
                self._dmask,
                self._omask,
                self._imask,
                beta_d_vec,
                beta_o_vec,
                n,
                k,
            )
            for key, arr in res.items():
                out[key][idx, :] = arr

        return out


class SARFlowPanel(FlowPanelModel):
    """Panel SAR flow model with three free rho parameters."""

    def _build_pymc_model(self) -> pm.Model:
        if self.logdet_method != "traces":
            raise ValueError("SARFlowPanel supports logdet_method='traces' only.")

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            if self.robust:
                nu = self._add_nu_prior()
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._T
                * flow_logdet_pytensor(
                    rho_d,
                    rho_o,
                    rho_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        rho_w_draws = idata.posterior["rho_w"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
        return self._compute_flow_effects_from_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws,
            draws=draws,
        )


class SARFlowSeparablePanel(FlowPanelModel):
    """Panel separable SAR flow model with rho_w = -rho_d * rho_o."""

    def __init__(self, y, G, X, **kwargs):
        # Normalize logdet_method: 'separable' is an alias for 'eigenvalue'.
        method = kwargs.pop("logdet_method", "eigenvalue")
        if method == "separable":
            method = "eigenvalue"
        _VALID = {"eigenvalue", "chebyshev", "mc_poly"}
        if method not in _VALID:
            raise ValueError(
                f"SARFlowSeparablePanel logdet_method must be one of {sorted(_VALID)}; "
                f"got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, G, X, **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SARFlowSeparablePanel requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            if self.robust:
                nu = self._add_nu_prior()
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._T * self._separable_logdet_fn(rho_d, rho_o),
            )

        return model

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
        return self._compute_flow_effects_kron(
            rho_d_draws,
            rho_o_draws,
            beta_draws,
            draws=draws,
        )


class PoissonSARFlowPanel(FlowPanelModel):
    """Panel Poisson SAR flow model with three free rho parameters.

    Notes
    -----
    This class currently supports pooled panels only (model=0). Applying
    within transforms to count outcomes would violate the Poisson support.
    """

    def __init__(self, y, G, X, **kwargs):
        model_mode = int(kwargs.get("model", 0))
        if model_mode != 0:
            raise ValueError(
                "PoissonSARFlowPanel currently supports model=0 only. "
                "Within-transformed FE panels are not valid for Poisson counts."
            )

        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonSARFlowPanel requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "PoissonSARFlowPanel requires non-negative integer observations."
            )

        super().__init__(y_arr.astype(np.float64), G, X, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.reshape(-1).astype(np.int64)

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import SparseFlowSolveMatrixOp

        if self.logdet_method != "traces":
            raise ValueError(
                "PoissonSARFlowPanel supports logdet_method='traces' only."
            )

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)

        N = self._N_flow  # n²
        T = self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")

            # Reshape Xb to (N, T) so that A^{-1} B can be solved with one
            # LU factorisation covering all T periods (SparseFlowSolveMatrixOp).
            Xb = pt.dot(X_t, beta)  # (N*T,)
            Xb_mat = pt.reshape(Xb, (T, N)).T  # (N, T), one column per period
            solve_op = SparseFlowSolveMatrixOp(self._Wd, self._Wo, self._Ww)
            eta_mat = solve_op(rho_d, rho_o, rho_w, Xb_mat)  # (N, T)
            eta = pt.reshape(eta_mat.T, (N * T,))  # (N*T,), back to time-first order
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                self._T
                * flow_logdet_pytensor(
                    rho_d,
                    rho_o,
                    rho_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        rho_w_draws = idata.posterior["rho_w"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
        return self._compute_flow_effects_from_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws,
            draws=draws,
        )

    def _simulate_y_rep_period(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Poisson posterior-predictive replicate for the full panel stack."""
        N = self._N_flow
        T = self._T
        A = self._assemble_A(rho_d, rho_o, rho_w).tocsc()
        lu = sp.linalg.splu(A)
        Xb = self._X_design @ beta
        Xb_mat = Xb.reshape(T, N).T
        eta_mat = lu.solve(Xb_mat)
        eta = eta_mat.T.reshape(-1)
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        return rng.poisson(lam).astype(np.float64)


class PoissonSARFlowSeparablePanel(FlowPanelModel):
    """Panel separable Poisson SAR flow model with rho_w = -rho_d * rho_o.

    Notes
    -----
    This class currently supports pooled panels only (model=0).
    """

    def __init__(self, y, G, X, **kwargs):
        model_mode = int(kwargs.get("model", 0))
        if model_mode != 0:
            raise ValueError(
                "PoissonSARFlowSeparablePanel currently supports model=0 only. "
                "Within-transformed FE panels are not valid for Poisson counts."
            )

        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonSARFlowSeparablePanel requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "PoissonSARFlowSeparablePanel requires non-negative integer observations."
            )

        method = kwargs.pop("logdet_method", "eigenvalue")
        if method == "separable":
            method = "eigenvalue"
        _VALID = {"eigenvalue", "chebyshev", "mc_poly"}
        if method not in _VALID:
            raise ValueError(
                f"PoissonSARFlowSeparablePanel logdet_method must be one of {sorted(_VALID)}; "
                f"got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y_arr.astype(np.float64), G, X, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.reshape(-1).astype(np.int64)

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import KroneckerFlowSolveMatrixOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "PoissonSARFlowSeparablePanel requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        n = self._n
        N = self._N_flow  # n²
        T = self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")

            # KroneckerFlowSolveMatrixOp: A = L_d ⊗ L_o (two n×n solves
            # covering all T columns — O(n³) regardless of T).
            # rho_w is kept as a Deterministic for trace reporting only.
            Xb = pt.dot(X_t, beta)  # (N*T,)
            Xb_mat = pt.reshape(Xb, (T, N)).T  # (N, T), one column per period
            solve_op = KroneckerFlowSolveMatrixOp(self._W_sparse, n)
            eta_mat = solve_op(rho_d, rho_o, Xb_mat)  # (N, T)
            eta = pt.reshape(eta_mat.T, (N * T,))  # (N*T,), back to time-first order
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                self._T * self._separable_logdet_fn(rho_d, rho_o),
            )

        return model

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
        return self._compute_flow_effects_kron(
            rho_d_draws,
            rho_o_draws,
            beta_draws,
            draws=draws,
        )

    def _simulate_y_rep_period(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,  # ignored; rho_w = -rho_d * rho_o
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Poisson posterior-predictive replicate using Kronecker solve."""
        N = self._N_flow
        T = self._T
        n = self._n
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - rho_d * self._W_sparse).tocsr()
        Lo = (I_n - rho_o * self._W_sparse).tocsr()
        Xb = self._X_design @ beta
        Xb_mat = Xb.reshape(T, N).T  # (N, T)
        eta_mat = kron_solve_matrix(Lo, Ld, Xb_mat, n)
        eta = eta_mat.T.reshape(-1)
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        return rng.poisson(lam).astype(np.float64)
