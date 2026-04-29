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

from ..graph import _validate_graph, flow_trace_blocks, flow_weight_matrices
from ..logdet import (
    _flow_logdet_poly_coeffs,
    compute_flow_traces,
    flow_logdet_numpy,
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
        symmetric_xo_xd: Optional[bool] = None,
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
            self._k_d: int = int(k)
            self._k_o: int = int(k)
        else:
            dest_cols = [
                name for name in self._feature_names if name.startswith("dest_")
            ]
            orig_cols = [
                name for name in self._feature_names if name.startswith("orig_")
            ]
            self._k_d = len(dest_cols)
            self._k_o = len(orig_cols)
            self._k = self._k_d  # backward compat alias

        # Locate β_intra slice for the Thomas-Agnan & LeSage (2014) intra
        # contribution.
        if self._k_d > 0:
            intra_cols = [
                i
                for i, name in enumerate(self._feature_names)
                if name.startswith("intra_")
            ]
            self._intra_idx: Optional[np.ndarray] = (
                np.asarray(intra_cols, dtype=np.int64) if intra_cols else None
            )
        else:
            self._intra_idx = None

        # Detect Xo == Xd symmetry on the (undemeaned) raw design.
        if (
            symmetric_xo_xd is None
            and self._k_d > 0
            and self._k_d == self._k_o
            and X_arr.shape[1] >= 2 + self._k_d + self._k_o
        ):
            dest_block = X_arr[:, 2 : 2 + self._k_d]
            orig_block = X_arr[:, 2 + self._k_d : 2 + self._k_d + self._k_o]
            self._symmetric_xo_xd: bool = bool(np.array_equal(dest_block, orig_block))
        else:
            self._symmetric_xo_xd = (
                bool(symmetric_xo_xd)
                if symmetric_xo_xd is not None
                else (self._k_d == self._k_o)
            )

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

        # Cache the symmetric 3x3 Kronecker trace matrix used by Bayesian
        # LM diagnostics on flow models: T[i,j] = tr(W_i' W_j) + tr(W_i W_j)
        # for (W_d, W_o, W_w).  Computed in O(nnz) from the n x n base graph.
        self._T_flow_traces: np.ndarray = flow_trace_blocks(self._W_sparse)

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
        idata_kwargs: Optional[dict] = None,
        sampler: Optional[str] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior."""
        from ._sampler import (
            prepare_compile_kwargs,
            prepare_idata_kwargs,
            resolve_sampler,
        )

        idata_kwargs = dict(idata_kwargs) if idata_kwargs else {}
        idata_kwargs.setdefault("log_likelihood", True)
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop(
            "nuts_sampler",
            resolve_sampler(
                sampler,
                requires_c_backend=getattr(self, "_requires_c_backend", False),
                model_name=type(self).__name__,
            ),
        )

        model = self._build_pymc_model()
        self._pymc_model = model
        self._approximation = None
        if "var_names" not in sample_kwargs and not store_lambda:
            sample_kwargs["var_names"] = self._posterior_var_names(
                model,
                store_lambda=False,
            )
        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        sample_kwargs = prepare_compile_kwargs(sample_kwargs, nuts_sampler)
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs=idata_kwargs,
                nuts_sampler=nuts_sampler,
                **sample_kwargs,
            )
        if compute_log_likelihood:
            self._attach_complete_log_likelihood(self._idata)
        return self._idata

    def fit_approx(
        self,
        draws: int = 2000,
        n: int = 10000,
        method: str = "advi",
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        compute_log_likelihood: bool = True,
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
            if compute_log_likelihood:
                pm.compute_log_likelihood(
                    self._idata,
                    extend_inferencedata=True,
                    progressbar=False,
                )

        if (
            not store_lambda
            and self._idata is not None
            and hasattr(self._idata, "posterior")
            and "lambda" in self._idata.posterior.data_vars
        ):
            self._idata.posterior = self._idata.posterior.drop_vars("lambda")

        if compute_log_likelihood:
            self._attach_complete_log_likelihood(self._idata)

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

    # ------------------------------------------------------------------
    # Class-level registry of Bayesian LM specification tests.
    # ------------------------------------------------------------------
    _spatial_diagnostics_tests: list[tuple] = []

    def spatial_diagnostics(self) -> pd.DataFrame:
        """Run Bayesian LM specification tests for flow panel models.

        See :meth:`bayespecon.models.base.SpatialModel.spatial_diagnostics`
        for the column schema.

        Raises
        ------
        RuntimeError
            If the model has not been fit yet.
        """
        from .base import SpatialModel

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        return SpatialModel._run_lm_diagnostics(self, self._spatial_diagnostics_tests)

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        """Return named dimensions for PyMC model coordinates."""
        coords = {"coefficient": self._feature_names}
        if extra:
            coords.update(extra)
        return coords

    # ------------------------------------------------------------------
    # Pointwise log-likelihood (with Jacobian correction for SAR variants)
    # ------------------------------------------------------------------

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        """Per-draw :math:`T \\cdot \\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|`.

        Returns ``None`` (the default) when no Jacobian correction is
        required — for OLS / Poisson panel baselines (``A = I_N``) and the
        Poisson SAR variants (Poisson observation density on counts is
        already captured exactly).  Subclasses with a Gaussian observation
        model and a ``pm.Potential("jacobian", T * log|A|)`` term must
        override this to return the per-draw value of that potential.
        """
        return None

    def _attach_complete_log_likelihood(self, idata) -> None:
        """Add Jacobian contribution to the pointwise log-likelihood.

        See :meth:`FlowModel._attach_complete_log_likelihood` for details.
        """
        if idata is None or not hasattr(idata, "log_likelihood"):
            return
        if "obs" not in idata.log_likelihood.data_vars:
            return

        jacobian_draws = self._compute_jacobian_log_det(idata.posterior)
        if jacobian_draws is None:
            return

        import xarray as xr

        ll_da = idata.log_likelihood["obs"]
        n_chains = ll_da.sizes["chain"]
        n_draws_per_chain = ll_da.sizes["draw"]
        n_obs = int(np.prod(ll_da.shape[2:]))

        ll_array = ll_da.values.reshape(n_chains * n_draws_per_chain, n_obs)
        jacobian_draws = np.asarray(jacobian_draws, dtype=np.float64).reshape(-1)
        if jacobian_draws.shape[0] != ll_array.shape[0]:
            raise RuntimeError(
                "Posterior draw count does not match log-likelihood shape: "
                f"{jacobian_draws.shape[0]} vs {ll_array.shape[0]}."
            )

        ll_array = ll_array + jacobian_draws[:, None] / n_obs
        ll_array = ll_array.reshape(n_chains, n_draws_per_chain, n_obs)

        new_da = xr.DataArray(ll_array, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": new_da})

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
        mode: str = "auto",
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Summarise posterior origin/destination/intra/network/total effects.

        See :meth:`bayespecon.models.flow.FlowModel.spatial_effects` for the
        ``mode`` semantics (auto / combined / separate destination-origin
        sides per Thomas-Agnan & LeSage 2014, §83.5.2).
        """
        from ..diagnostics.spatial_effects import _compute_bayesian_pvalue
        from .flow import _EFFECT_KEYS

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        if self._k == 0:
            raise RuntimeError(
                "Cannot compute spatial effects: no `dest_*` columns detected "
                "in the design matrix.  Pass `k=` explicitly when constructing "
                "the model."
            )
        if mode not in {"auto", "combined", "separate"}:
            raise ValueError(
                f"mode must be 'auto', 'combined', or 'separate'; got {mode!r}."
            )

        posterior = self._compute_spatial_effects_posterior(draws=draws)

        if mode == "auto":
            effective_mode = "combined" if self._symmetric_xo_xd else "separate"
        else:
            effective_mode = mode

        if effective_mode == "combined":
            display = [("combined", eff) for eff in _EFFECT_KEYS]
        else:
            display = [(side, eff) for side in ("dest", "orig") for eff in _EFFECT_KEYS]

        feature_names = [
            name[len("dest_") :] if name.startswith("dest_") else name
            for name in self._feature_names
            if name.startswith("dest_")
        ][: self._k_d]
        if len(feature_names) != self._k_d:
            feature_names = [f"x{i}" for i in range(self._k_d)]

        orig_feature_names = [
            name[len("orig_") :] if name.startswith("orig_") else name
            for name in self._feature_names
            if name.startswith("orig_")
        ][: self._k_o]
        if len(orig_feature_names) != self._k_o:
            orig_feature_names = [f"y{i}" for i in range(self._k_o)]

        # For combined mode: when k_d == k_o, combined effects are the sum
        # of dest and orig (same variables), so use dest names.
        # When k_d != k_o, combined effects are concatenated (different variables).
        if self._k_d == self._k_o:
            combined_feature_names = feature_names
        else:
            combined_feature_names = feature_names + orig_feature_names

        alpha = (1.0 - ci) / 2.0
        rows = []
        for side, effect_name in display:
            key = effect_name if side == "combined" else f"{side}_{effect_name}"
            samples = posterior[key]
            means = samples.mean(axis=0)
            lower = np.quantile(samples, alpha, axis=0)
            upper = np.quantile(samples, 1.0 - alpha, axis=0)
            pvals = _compute_bayesian_pvalue(samples)
            if side == "combined":
                fnames = combined_feature_names
            elif side == "dest":
                fnames = feature_names
            else:
                fnames = orig_feature_names
            for j, fname in enumerate(fnames):
                rows.append(
                    {
                        "predictor": fname,
                        "side": side,
                        "effect": effect_name,
                        "mean": float(means[j]),
                        "ci_lower": float(lower[j]),
                        "ci_upper": float(upper[j]),
                        "bayes_pvalue": float(pvals[j]),
                    }
                )

        df = pd.DataFrame(rows).set_index(["predictor", "side", "effect"])
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
        k_d = self._k_d
        k_o = self._k_o

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            rho_w_draws = rho_w_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        from .flow import _EFFECT_KEYS

        out: dict[str, np.ndarray] = {}
        for side in ("dest", "orig"):
            k_side = k_d if side == "dest" else k_o
            for eff in _EFFECT_KEYS:
                out[f"{side}_{eff}"] = np.zeros(
                    (n_draws_total, k_side), dtype=np.float64
                )
        k_combined = k_d + k_o if k_d != k_o else k_d
        for eff in _EFFECT_KEYS:
            out[eff] = np.zeros((n_draws_total, k_combined), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            rw = float(rho_w_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k_d]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k_o]
            beta_intra_vec = (
                beta_draws[idx, intra_start : intra_start + k_d] if has_intra else None
            )

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
                k_d,
                k_o=k_o,
                beta_intra=beta_intra_vec,
            )
            for key, arr in res.items():
                out[key][idx, : len(arr)] = arr

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
        k_d = self._k_d
        k_o = self._k_o
        W = self._W_sparse.tocsr()
        I_n = sp.eye(n, format="csr", dtype=np.float64)

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        from .flow import _EFFECT_KEYS

        out: dict[str, np.ndarray] = {}
        for side in ("dest", "orig"):
            k_side = k_d if side == "dest" else k_o
            for eff in _EFFECT_KEYS:
                out[f"{side}_{eff}"] = np.zeros(
                    (n_draws_total, k_side), dtype=np.float64
                )
        k_combined = k_d + k_o if k_d != k_o else k_d
        for eff in _EFFECT_KEYS:
            out[eff] = np.zeros((n_draws_total, k_combined), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k_d]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k_o]
            beta_intra_vec = (
                beta_draws[idx, intra_start : intra_start + k_d] if has_intra else None
            )

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
                k_d,
                k_o=k_o,
                beta_intra=beta_intra_vec,
            )
            for key, arr in res.items():
                out[key][idx, : len(arr)] = arr

        return out


class SARFlowPanel(FlowPanelModel):
    r"""Panel spatial-lag origin-destination flow model with unrestricted dependence.

    For each period :math:`t`, the vectorized flow matrix
    :math:`y_t \in \mathbb{R}^{N}` with :math:`N = n^2` satisfies

    .. math::

        y_t = \rho_d W_d y_t + \rho_o W_o y_t + \rho_w W_w y_t + X_t \beta + \varepsilon_t,
        \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N).

    The panel stack is time-first across :math:`T` periods. The ``model``
    argument controls pooled, pair fixed-effects, time fixed-effects, or
    two-way demeaning before the likelihood is evaluated. The Jacobian
    contribution scales as :math:`T \log |A(\rho_d, \rho_o, \rho_w)|`.
    """

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

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        rho_d = posterior["rho_d"].values.reshape(-1)
        rho_o = posterior["rho_o"].values.reshape(-1)
        rho_w = posterior["rho_w"].values.reshape(-1)
        log_det = flow_logdet_numpy(
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
        )
        return self._T * log_det

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
    r"""Panel separable spatial-lag flow model with :math:`\rho_w = -\rho_d \rho_o`.

    For each period :math:`t`,

    .. math::

        y_t = \rho_d W_d y_t + \rho_o W_o y_t - \rho_d \rho_o W_w y_t + X_t \beta + \varepsilon_t,
        \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N).

    Under the separability restriction,
    :math:`A = I_N - \rho_d W_d - \rho_o W_o + \rho_d \rho_o W_w`
    factorizes into Kronecker blocks, which enables the exact or
    approximated eigenvalue-based log-determinant used by this class.
    """

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

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        rho_d = np.asarray(posterior["rho_d"].values.reshape(-1), dtype=np.float64)
        rho_o = np.asarray(posterior["rho_o"].values.reshape(-1), dtype=np.float64)
        n = self._n
        if self._W_eigs is None:
            self._W_eigs = np.linalg.eigvals(
                self._W_sparse.toarray().astype(np.float64)
            ).real.astype(np.float64)
        eigs = self._W_eigs
        ld_d = np.log(np.abs(1.0 - rho_d[:, None] * eigs[None, :])).sum(axis=1)
        ld_o = np.log(np.abs(1.0 - rho_o[:, None] * eigs[None, :])).sum(axis=1)
        return self._T * n * (ld_d + ld_o)

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
    r"""Panel Poisson spatial-lag flow model with unrestricted dependence.

    The stacked panel counts satisfy

    .. math::

        y_{ij,t} \sim \operatorname{Poisson}(\lambda_{ij,t}),
        \qquad \log \boldsymbol{\lambda}_t = A(\rho_d, \rho_o, \rho_w)^{-1} X_t \beta,

    where

    .. math::

        A(\rho_d, \rho_o, \rho_w) = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w.

    Notes
    -----
    This class currently supports pooled panels only (``model=0``).
    Within transforms are not valid for Poisson counts because they break
    the non-negative integer support.
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
    r"""Panel separable Poisson spatial-lag flow model.

    The panel counts satisfy

    .. math::

        y_{ij,t} \sim \operatorname{Poisson}(\lambda_{ij,t}),
        \qquad \log \boldsymbol{\lambda}_t = A(\rho_d, \rho_o)^{-1} X_t \beta,

    with the separability restriction :math:`\rho_w = -\rho_d \rho_o` and

    .. math::

        A(\rho_d, \rho_o) = I_N - \rho_d W_d - \rho_o W_o + \rho_d \rho_o W_w.

    Notes
    -----
    This class currently supports pooled panels only (``model=0``).
    The separability restriction enables the Kronecker-factorized
    log-determinant used in estimation.
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


class OLSFlowPanel(FlowPanelModel):
    r"""Non-spatial Bayesian OD-flow gravity model for balanced panel data.

    Panel analogue of :class:`~bayespecon.models.flow.OLSFlow`: implements
    the conventional log-linear gravity specification of
    :cite:t:`thomasAgnan2014SpatialEconometric` (eq. 83.2) with no spatial
    lag terms,

    .. math::

        y_{t} = X_{t}\,\beta + \varepsilon_{t}, \quad
        \varepsilon_{t} \sim \mathcal{N}(0, \sigma^{2} I_{N}),

    on a balanced panel of :math:`T` periods, applying the same
    fixed-effects within transform (`model` argument) as the spatial panel
    flow models.  Provided as the canonical null model for Bayesian LM
    diagnostics on panel flow data.

    Parameters
    ----------
    y, G, X, T, col_names, k, model, priors, robust, symmetric_xo_xd
        See :class:`FlowPanelModel`.  *G* is required for API symmetry but
        the spatial weights are not used in estimation.

    Notes
    -----
    The ``priors`` dict supports ``beta_mu``, ``beta_sigma``,
    ``sigma_sigma``; spatial keys (``rho_*``) are ignored.  All
    log-determinant precomputation is skipped (``A = I_N`` with
    :math:`|A| = 1`).
    """

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_panel_lm_flow_dest_test"],
            ).bayesian_panel_lm_flow_dest_test(m),
            "Panel-LM-Flow-Dest",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_panel_lm_flow_orig_test"],
            ).bayesian_panel_lm_flow_orig_test(m),
            "Panel-LM-Flow-Orig",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_panel_lm_flow_network_test"],
            ).bayesian_panel_lm_flow_network_test(m),
            "Panel-LM-Flow-Network",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_panel_lm_flow_joint_test"],
            ).bayesian_panel_lm_flow_joint_test(m),
            "Panel-LM-Flow-Joint",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_panel_lm_flow_intra_test"],
            ).bayesian_panel_lm_flow_intra_test(m),
            "Panel-LM-Flow-Intra",
        ),
    ]

    def __init__(self, y, G, X, T, **kwargs):
        # Skip log-determinant precomputation: A = I_N has |A| = 1.
        kwargs.pop("logdet_method", None)
        kwargs.pop("restrict_positive", None)
        super().__init__(y, G, X, T, logdet_method="none", **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = pt.dot(X_t, beta)
            if self.robust:
                nu = self._add_nu_prior()
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

        return model

    def _simulate_y_rep_period(
        self,
        rho_d: float,  # unused
        rho_o: float,  # unused
        rho_w: float,  # unused
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-predictive replicate ``y_rep = X β + σ ε`` (full panel stack)."""
        Xb = self._X_design @ beta  # (N_flow * T,)
        if sigma is None:
            return Xb
        return Xb + rng.normal(scale=float(sigma), size=Xb.shape[0])

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flows for the OLS panel gravity model.

        Overrides the base implementation, which expects ``rho_d``,
        ``rho_o``, ``rho_w`` posterior arrays that this model does not
        sample.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep_period(
                0.0, 0.0, 0.0, beta_draws[g], sigma_g, rng
            )
        return out

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        r"""Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

        Identical to
        :meth:`bayespecon.models.flow.OLSFlow._compute_spatial_effects_posterior`:
        with :math:`A = I_N` the response to any shock equals the shock
        itself, so the per-region averages collapse to closed-form
        expressions in :math:`\beta_d`, :math:`\beta_o`, and
        :math:`\beta_{\text{intra}}`.  Effects are time-invariant under
        the static panel parameters of this model.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        # Local import avoids a circular import at module load time.
        from .flow import _EFFECT_KEYS

        idata = self._idata
        n = self._n
        k = self._k
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

        dest_start = 2
        orig_start = 2 + k
        intra_start = 2 + 2 * k
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k
        )

        n_draws_total = beta_draws.shape[0]
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            beta_draws = beta_draws[:n_draws_total]

        bd = beta_draws[:, dest_start : dest_start + k]
        bo = beta_draws[:, orig_start : orig_start + k]
        bi = (
            beta_draws[:, intra_start : intra_start + k]
            if has_intra
            else np.zeros((n_draws_total, k), dtype=np.float64)
        )

        zeros = np.zeros_like(bd)
        out: dict[str, np.ndarray] = {}
        out["dest_total"] = bd + bi / n
        out["dest_destination"] = bd * (n - 1) / n
        out["dest_intra"] = (bd + bi) / n
        out["dest_origin"] = zeros.copy()
        out["dest_network"] = zeros.copy()

        out["orig_total"] = bo
        out["orig_origin"] = bo * (n - 1) / n
        out["orig_intra"] = bo / n
        out["orig_destination"] = zeros.copy()
        out["orig_network"] = zeros.copy()

        for eff in _EFFECT_KEYS:
            out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]

        return out


class PoissonFlowPanel(OLSFlowPanel):
    r"""Non-spatial Bayesian OD-flow Poisson gravity model for balanced panel data.

    Panel analogue of :class:`~bayespecon.models.flow.PoissonFlow` and
    count analogue of :class:`OLSFlowPanel`.  Models stacked panel flow
    counts with a log-linear gravity mean and no spatial-lag terms,

    .. math::

        y_{ij,t} \sim \operatorname{Poisson}(\lambda_{ij,t}), \qquad
        \log \boldsymbol{\lambda}_{t} = X_{t}\beta,

    on a balanced panel of :math:`T` periods.  Provided as the canonical
    aspatial count baseline for panel flow data.

    Parameters
    ----------
    y, G, X, T, col_names, k, priors, symmetric_xo_xd
        See :class:`FlowPanelModel`.  *G* is required for API symmetry but
        the spatial weights are not used in estimation.

    Notes
    -----
    Currently supports pooled panels only (``model=0``).  Within-transformed
    fixed-effects panels are not valid for Poisson counts (they break the
    non-negative integer support), matching the restriction enforced by
    :class:`PoissonSARFlowPanel`.

    The ``priors`` dict supports ``beta_mu`` and ``beta_sigma``; spatial
    keys (``rho_*``), ``sigma_sigma``, and the ``robust`` flag are ignored
    (Poisson has no scale parameter to robustify).
    """

    def __init__(self, y, G, X, T, **kwargs):
        model_mode = int(kwargs.get("model", 0))
        if model_mode != 0:
            raise ValueError(
                "PoissonFlowPanel currently supports model=0 only. "
                "Within-transformed FE panels are not valid for Poisson counts."
            )

        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonFlowPanel requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "PoissonFlowPanel requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), G, X, T, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.reshape(-1).astype(np.int64)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            eta = pt.dot(X_t, beta)
            lam = pm.Deterministic("lambda", pt.exp(eta))
            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

        return model

    def _simulate_y_rep_period(
        self,
        rho_d: float,  # unused
        rho_o: float,  # unused
        rho_w: float,  # unused
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Poisson posterior-predictive replicate ``y_rep`` (full panel stack)."""
        eta = self._X_design @ beta  # (N_flow * T,)
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        return rng.poisson(lam).astype(np.float64)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for the panel Poisson gravity model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            out[g] = self._simulate_y_rep_period(
                0.0, 0.0, 0.0, beta_draws[g], None, rng
            )
        return out


# ---------------------------------------------------------------------------
# Panel SEM-Flow variants (spatial-error analogues of SARFlowPanel)
# ---------------------------------------------------------------------------


def _sparse_flow_panel_lag_matrix(
    M: np.ndarray, W_flow: sp.csr_matrix, T: int, N_flow: int
) -> np.ndarray:
    """Apply :math:`I_T \\otimes W_{flow}` to a stacked panel design matrix.

    Parameters
    ----------
    M : np.ndarray, shape ``(N_flow * T, p)``
        Time-first stacked design matrix.
    W_flow : scipy.sparse matrix, shape ``(N_flow, N_flow)``
        Flow weight matrix (one of ``W_d``, ``W_o``, ``W_w``).
    T, N_flow : int
        Panel dimensions.

    Returns
    -------
    np.ndarray, shape ``(N_flow * T, p)``
        ``W_flow`` applied to each period block independently.
    """
    p = M.shape[1] if M.ndim == 2 else 1
    chunks = M.reshape(T, N_flow, p)
    out = np.empty_like(chunks)
    for t in range(T):
        out[t] = W_flow @ chunks[t]
    return out.reshape(T * N_flow, p)


class _SEMFlowPanelMixin:
    """Shared init helper to precompute design-matrix lags for SEM panel models."""

    def _init_sem_lags(self) -> None:
        T = self._T
        N = self._N_flow
        # Lags of the (already-demeaned) design matrix.  Constants — no
        # parameter dependence, so we precompute once.
        self._Wd_X: np.ndarray = _sparse_flow_panel_lag_matrix(
            self._X.astype(np.float64), self._Wd, T, N
        )
        self._Wo_X: np.ndarray = _sparse_flow_panel_lag_matrix(
            self._X.astype(np.float64), self._Wo, T, N
        )
        self._Ww_X: np.ndarray = _sparse_flow_panel_lag_matrix(
            self._X.astype(np.float64), self._Ww, T, N
        )


class SEMFlowPanel(_SEMFlowPanelMixin, FlowPanelModel):
    r"""Panel spatial-error flow model with three free spatial parameters.

    Panel analogue of :class:`~bayespecon.models.flow.SEMFlow`: applies the
    Kronecker spatial filter (:math:`W_d`, :math:`W_o`, :math:`W_w`) to the
    disturbance rather than the dependent variable, period by period:

    .. math::

        y_t = X_t \beta + u_t, \qquad B u_t = \varepsilon_t,
        \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N).

    The Jacobian contribution scales as :math:`T \cdot \log|B|` — identical
    in form to :class:`SARFlowPanel`.  Marginal mean is :math:`X_t \beta`,
    so there are no :math:`X`-mediated spillovers; effects collapse to the
    closed-form expressions used by :class:`OLSFlowPanel`.
    """

    def __init__(self, y, G, X, T, **kwargs):
        super().__init__(y, G, X, T, **kwargs)
        self._init_sem_lags()

    def _build_pymc_model(self) -> pm.Model:
        if self.logdet_method != "traces":
            raise ValueError("SEMFlowPanel supports logdet_method='traces' only.")

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        Wd_X_t = pt.as_tensor_variable(self._Wd_X.astype(np.float64))
        Wo_X_t = pt.as_tensor_variable(self._Wo_X.astype(np.float64))
        Ww_X_t = pt.as_tensor_variable(self._Ww_X.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                lam_simplex = pm.Dirichlet("lam_simplex", a=np.ones(4))
                lam_d = pm.Deterministic("lam_d", lam_simplex[0])
                lam_o = pm.Deterministic("lam_o", lam_simplex[1])
                lam_w = pm.Deterministic("lam_w", lam_simplex[2])
            else:
                lam_lower = self.priors.get("lam_lower", -1.0)
                lam_upper = self.priors.get("lam_upper", 1.0)
                lam_d = pm.Uniform("lam_d", lower=lam_lower, upper=lam_upper)
                lam_o = pm.Uniform("lam_o", lower=lam_lower, upper=lam_upper)
                lam_w = pm.Uniform("lam_w", lower=lam_lower, upper=lam_upper)
                slack = 1.0 - lam_d - lam_o - lam_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # mu chosen so y - mu = By - BXβ = Bu (the whitened residual).
            mu = (
                lam_d * Wd_y_t
                + lam_o * Wo_y_t
                + lam_w * Ww_y_t
                + pt.dot(X_t, beta)
                - lam_d * pt.dot(Wd_X_t, beta)
                - lam_o * pt.dot(Wo_X_t, beta)
                - lam_w * pt.dot(Ww_X_t, beta)
            )
            if self.robust:
                nu = self._add_nu_prior()
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._T
                * flow_logdet_pytensor(
                    lam_d,
                    lam_o,
                    lam_w,
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

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        lam_d = posterior["lam_d"].values.reshape(-1)
        lam_o = posterior["lam_o"].values.reshape(-1)
        lam_w = posterior["lam_w"].values.reshape(-1)
        log_det = flow_logdet_numpy(
            lam_d,
            lam_o,
            lam_w,
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
        )
        return self._T * log_det

    def _simulate_y_rep_period(
        self,
        lam_d: float,
        lam_o: float,
        lam_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """SEM panel posterior-predictive: ``y_rep,t = X_t β + B^{-1} ε_t``."""
        N = self._N_flow
        T = self._T
        Xb = self._X_design @ beta  # (N*T,)
        if sigma is None:
            return Xb
        B = self._assemble_A(lam_d, lam_o, lam_w).tocsc()
        lu = sp.linalg.splu(B)
        eps = rng.normal(scale=float(sigma), size=(N, T))
        u = lu.solve(eps)  # (N, T)
        u_stacked = u.T.reshape(-1)  # back to time-first
        return Xb + u_stacked

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Closed-form effects (delegates to OLSFlowPanel logic)."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        return _ols_panel_effects(
            self._idata,
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )


class SEMFlowSeparablePanel(_SEMFlowPanelMixin, FlowPanelModel):
    r"""Panel separable spatial-error flow model with :math:`\lambda_w = -\lambda_d \lambda_o`.

    Panel analogue of :class:`~bayespecon.models.flow.SEMFlowSeparable` and
    spatial-error counterpart of :class:`SARFlowSeparablePanel`.  Uses the
    eigenvalue / Chebyshev factorisation of :math:`\log|B|` with the panel
    Jacobian scaling :math:`T \cdot \log|B|`.
    """

    def __init__(self, y, G, X, T, **kwargs):
        method = kwargs.pop("logdet_method", "eigenvalue")
        if method == "separable":
            method = "eigenvalue"
        _VALID = {"eigenvalue", "chebyshev", "mc_poly"}
        if method not in _VALID:
            raise ValueError(
                f"SEMFlowSeparablePanel logdet_method must be one of {sorted(_VALID)}; "
                f"got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, G, X, T, **kwargs)
        self._init_sem_lags()

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        lam_lower = self.priors.get("lam_lower", -0.999)
        lam_upper = self.priors.get("lam_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SEMFlowSeparablePanel requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        Wd_X_t = pt.as_tensor_variable(self._Wd_X.astype(np.float64))
        Wo_X_t = pt.as_tensor_variable(self._Wo_X.astype(np.float64))
        Ww_X_t = pt.as_tensor_variable(self._Ww_X.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            lam_d = pm.Uniform("lam_d", lower=lam_lower, upper=lam_upper)
            lam_o = pm.Uniform("lam_o", lower=lam_lower, upper=lam_upper)
            lam_w = pm.Deterministic("lam_w", -lam_d * lam_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = (
                lam_d * Wd_y_t
                + lam_o * Wo_y_t
                + lam_w * Ww_y_t
                + pt.dot(X_t, beta)
                - lam_d * pt.dot(Wd_X_t, beta)
                - lam_o * pt.dot(Wo_X_t, beta)
                - lam_w * pt.dot(Ww_X_t, beta)
            )
            if self.robust:
                nu = self._add_nu_prior()
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._T * self._separable_logdet_fn(lam_d, lam_o),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        lam_d = np.asarray(posterior["lam_d"].values.reshape(-1), dtype=np.float64)
        lam_o = np.asarray(posterior["lam_o"].values.reshape(-1), dtype=np.float64)
        n = self._n
        if self._W_eigs is None:
            self._W_eigs = np.linalg.eigvals(
                self._W_sparse.toarray().astype(np.float64)
            ).real.astype(np.float64)
        eigs = self._W_eigs
        ld_d = np.log(np.abs(1.0 - lam_d[:, None] * eigs[None, :])).sum(axis=1)
        ld_o = np.log(np.abs(1.0 - lam_o[:, None] * eigs[None, :])).sum(axis=1)
        return self._T * n * (ld_d + ld_o)

    def _simulate_y_rep_period(
        self,
        lam_d: float,
        lam_o: float,
        lam_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """SEM panel posterior-predictive using Kronecker solve for ``B^{-1}``."""
        N = self._N_flow
        T = self._T
        n = self._n
        Xb = self._X_design @ beta
        if sigma is None:
            return Xb
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - lam_d * self._W_sparse).tocsr()
        Lo = (I_n - lam_o * self._W_sparse).tocsr()
        eps = rng.normal(scale=float(sigma), size=(N, T))
        u = kron_solve_matrix(Lo, Ld, eps, n)
        return Xb + u.T.reshape(-1)

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        return _ols_panel_effects(
            self._idata,
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )


def _ols_panel_effects(
    idata: az.InferenceData,
    *,
    n: int,
    k_d: int,
    k_o: int,
    feature_names: list[str],
    intra_idx: Optional[np.ndarray],
    draws: Optional[int],
) -> dict[str, np.ndarray]:
    """Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

    Shared between :class:`OLSFlowPanel`, :class:`SEMFlowPanel`, and
    :class:`SEMFlowSeparablePanel` — all of which have :math:`\\mathbb{E}[y]
    = X\\beta` (no :math:`X`-mediated spillovers).
    """
    from .flow import _EFFECT_KEYS

    beta_draws = idata.posterior["beta"].values.reshape(-1, len(feature_names))

    dest_start = 2
    orig_start = 2 + k_d
    intra_start = 2 + k_d + k_o
    has_intra = intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d

    n_draws_total = beta_draws.shape[0]
    if draws is not None:
        n_draws_total = min(draws, n_draws_total)
        beta_draws = beta_draws[:n_draws_total]

    bd = beta_draws[:, dest_start : dest_start + k_d]
    bo = beta_draws[:, orig_start : orig_start + k_o]
    bi = (
        beta_draws[:, intra_start : intra_start + k_d]
        if has_intra
        else np.zeros((n_draws_total, k_d), dtype=np.float64)
    )

    zeros_d = np.zeros_like(bd)
    zeros_o = np.zeros_like(bo)
    out: dict[str, np.ndarray] = {}
    out["dest_total"] = bd + bi / n
    out["dest_destination"] = bd * (n - 1) / n
    out["dest_intra"] = (bd + bi) / n
    out["dest_origin"] = zeros_d.copy()
    out["dest_network"] = zeros_d.copy()

    out["orig_total"] = bo
    out["orig_origin"] = bo * (n - 1) / n
    out["orig_intra"] = bo / n
    out["orig_destination"] = zeros_o.copy()
    out["orig_network"] = zeros_o.copy()

    if k_d == k_o:
        for eff in _EFFECT_KEYS:
            out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]
    else:
        for eff in _EFFECT_KEYS:
            out[eff] = np.concatenate([out[f"dest_{eff}"], out[f"orig_{eff}"]], axis=1)

    return out
