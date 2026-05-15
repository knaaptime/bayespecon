"""Shared base class for cross-sectional and panel spatial models.

Holds API surface that is genuinely identical between
:class:`bayespecon.models.base.SpatialModel` and
:class:`bayespecon.models.panel_base.SpatialPanelModel`:

* Public methods: :meth:`fit`, :meth:`summary`, :meth:`fitted_values`,
  :meth:`residuals`, :meth:`spatial_effects`, and the
  :attr:`inference_data` / :attr:`pymc_model` properties.
* Diagnostics utilities: :meth:`_lazy_lm_test`, :meth:`_run_lm_diagnostics`
  and the class-level ``_spatial_diagnostics_tests`` registry.
* Naming and coordinate helpers: :meth:`_beta_names`,
  :meth:`_model_coords`, :meth:`_rename_summary_index`,
  :meth:`_spatial_lag_column_indices`, :attr:`_nonintercept_indices`,
  :attr:`_nonintercept_feature_names`.
* Internals: :meth:`_require_fit`, :meth:`_posterior_mean`,
  :meth:`_add_nu_prior`, :meth:`_attach_jacobian_corrected_log_likelihood`.

Subclasses override only the genuinely divergent pieces (input parsing,
spatial-lag computation, FE demeaning, diagnostics-test name prefixing).
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from ._sampler import prepare_compile_kwargs, prepare_idata_kwargs


class _SpatialModelBase(ABC):
    """Common ABC for cross-sectional and panel spatial models.

    This class is internal; user-facing code should subclass
    :class:`bayespecon.models.base.SpatialModel` or
    :class:`bayespecon.models.panel_base.SpatialPanelModel`.
    """

    # Class-level registry of Bayesian LM specification tests applicable to
    # this model.  Subclasses populate this with a tuple of
    # ``(callable, label)`` pairs.
    _spatial_diagnostics_tests: tuple = ()

    # Configuration hooks for :meth:`spatial_diagnostics` /
    # :meth:`spatial_diagnostics_decision` that differ between the
    # cross-sectional and panel families.  Subclasses override to switch
    # the test-name prefix used in the decision tree (``""`` vs
    # ``"Panel-"``), the predicate-name prefix passed to
    # :mod:`bayespecon.diagnostics._decision_trees`, and the spec
    # accessor (``get_spec`` vs ``get_panel_spec``).
    _decision_test_prefix: str = ""
    _decision_predicate_prefix: str = ""
    _decision_spec_attr: str = "get_spec"
    _diagnostics_require_W: bool = True

    # Name of the sparse-W attribute that :attr:`_W_pt_sparse` should
    # wrap as a PyTensor sparse variable.  Cross-sectional models point
    # this at ``_W_sparse`` (the n×n weight matrix); panel models
    # override to ``_W_sparse_NT`` (the (N·T)×(N·T) Kronecker form).
    _pt_sparse_attr: str = "_W_sparse"

    # ------------------------------------------------------------------
    # Abstract interface — implemented by leaf subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model. Subclasses implement this."""

    @abstractmethod
    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute model-specific impact measures at posterior mean."""

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)`` where each
            array has shape ``(G, k)`` or ``(G, k_wx)``.
        """

    @abstractmethod
    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters."""

    # ------------------------------------------------------------------
    # Feature / coefficient naming helpers
    # ------------------------------------------------------------------

    @property
    def _nonintercept_indices(self) -> list[int]:
        """Return indices of non-constant (non-intercept) columns in X.

        Used to exclude the intercept from impact measures since
        the intercept has no meaningful spatial effect interpretation.
        """
        indices: list[int] = []
        for j, name in enumerate(self._feature_names):
            column = self._X[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    @property
    def _nonintercept_feature_names(self) -> list[str]:
        """Return feature names for non-intercept columns."""
        return [self._feature_names[i] for i in self._nonintercept_indices]

    @staticmethod
    def _spatial_lag_column_indices(
        X: np.ndarray, feature_names: list[str]
    ) -> list[int]:
        """Return indices of regressors that should receive spatial lags.

        Constant columns are treated as intercept-like and excluded, which
        avoids adding redundant ``W * intercept`` terms to SLX/Durbin models.
        """
        indices: list[int] = []
        for j, name in enumerate(feature_names):
            column = X[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    def _beta_names(self) -> list[str]:
        """Return coefficient labels used for posterior summaries."""
        return list(self._feature_names)

    def _model_coords(self) -> dict[str, list[str]]:
        """Return PyMC coordinate labels for named dimensions."""
        return {"coefficient": self._beta_names()}

    @staticmethod
    def _rename_summary_index(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Strip the ``beta[...]`` wrapper from coefficient row labels."""
        renamed = []
        for label in summary_df.index.astype(str):
            if label.startswith("beta[") and label.endswith("]"):
                renamed.append(label[5:-1])
            else:
                renamed.append(label)
        out = summary_df.copy()
        out.index = renamed
        return out

    # ------------------------------------------------------------------
    # Robust (Student-t) prior helper
    # ------------------------------------------------------------------

    def _add_nu_prior(self, model: pm.Model) -> pm.Model:
        """Add the degrees-of-freedom prior for robust (Student-t) models.

        Called inside ``_build_pymc_model`` when ``self.robust`` is True.
        Uses an :math:`\\mathrm{Exp}(\\lambda_\\nu)` prior on ``nu`` with rate
        ``nu_lam`` (default 1/30, giving mean ≈ 30, favouring near-Normal
        tails). A lower bound of 2 is enforced so that the variance exists.
        """
        nu_lam = self.priors.get("nu_lam", 1.0 / 30.0)
        pm.Truncated("nu", pm.Exponential.dist(lam=nu_lam), lower=2.0)
        return model

    # ------------------------------------------------------------------
    # Public API: sampling + summaries
    # ------------------------------------------------------------------

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws : int
            Number of posterior samples per chain (after tuning).
        tune : int
            Number of tuning (burn-in) steps per chain.
        chains : int
            Number of parallel chains.
        target_accept : float
            Target acceptance rate for NUTS.
        random_seed : int, optional
            Seed for reproducibility.
        **sample_kwargs
            Additional keyword arguments forwarded to ``pm.sample``.  Pass
            ``nuts_sampler="blackjax"`` (or ``"numpyro"``, ``"nutpie"``) to
            select an alternative NUTS backend; defaults to PyMC's built-in
            sampler.

        Returns
        -------
        arviz.InferenceData
        """
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
        try:
            model = self._build_pymc_model(nuts_sampler=nuts_sampler)
        except TypeError:
            # Subclasses that don't accept ``nuts_sampler`` build the same
            # model on every backend.
            model = self._build_pymc_model()
        self._pymc_model = model
        if "idata_kwargs" in sample_kwargs:
            sample_kwargs["idata_kwargs"] = prepare_idata_kwargs(
                sample_kwargs["idata_kwargs"], model, nuts_sampler
            )
        sample_kwargs = prepare_compile_kwargs(sample_kwargs, nuts_sampler)
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                **sample_kwargs,
            )
        return self._idata

    @property
    def inference_data(self) -> Optional[az.InferenceData]:
        """Return the ArviZ InferenceData from the most recent fit."""
        return self._idata

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model object built for the most recent fit."""
        return self._pymc_model

    def summary(self, var_names: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """Return posterior summary table.

        Parameters
        ----------
        var_names : list, optional
            Variable names to include in the summary.
        **kwargs
            Additional arguments passed to :func:`arviz.summary`.

        Returns
        -------
        pandas.DataFrame
            Posterior summary statistics.
        """
        self._require_fit()
        summary_df = az.summary(self._idata, var_names=var_names, **kwargs)
        return self._rename_summary_index(summary_df)

    def fitted_values(self) -> np.ndarray:
        """Return fitted values at posterior mean parameters."""
        self._require_fit()
        return self._fitted_mean_from_posterior()

    def residuals(self) -> np.ndarray:
        """Return residuals on the observed (or transformed-panel) scale."""
        self._require_fit()
        return self._y - self.fitted_values()

    # ------------------------------------------------------------------
    # Log-likelihood Jacobian correction
    # ------------------------------------------------------------------

    def _attach_jacobian_corrected_log_likelihood(
        self,
        idata: az.InferenceData,
        spatial_param: str,
        T: int = 1,
    ) -> None:
        """Add Jacobian correction to the auto-captured log-likelihood group.

        For models that use ``pm.Normal("obs", observed=y)`` plus
        ``pm.Potential("jacobian", logdet_fn(rho))``, PyMC auto-captures
        the Gaussian part in the ``log_likelihood`` group but the Jacobian
        term is absent.  This method adds the per-observation Jacobian
        contribution ``log|I - ρW| * T / n`` to each pointwise LL value.

        Notes
        -----
        For panel models the full log-determinant of the spatial filter is
        :math:`T \\log|I_N - \\rho W|` because the stacked
        :math:`(N T) \\times (N T)` filter is
        :math:`I_T \\otimes (I_N - \\rho W)`, whose determinant is
        :math:`|I_N - \\rho W|^T` by the Kronecker product rule.  Dividing
        by :math:`n = N T` distributes that scalar Jacobian uniformly
        over the per-observation pointwise log-likelihood entries that
        ArviZ expects.  For dynamic panels the time dimension is ``T - 1``
        (one period is consumed by the lag), so callers pass
        ``T = T - 1`` here.

        Parameters
        ----------
        idata : arviz.InferenceData
            InferenceData with an existing ``log_likelihood`` group.
        spatial_param : str
            Name of the spatial autoregressive parameter (``"rho"`` or
            ``"lam"``) in the posterior.
        T : int, default 1
            Panel time-period multiplier for the Jacobian.
        """
        import xarray as xr

        if "log_likelihood" not in idata.groups():
            return

        n = self._y.shape[0]
        param_draws = idata.posterior[spatial_param].values.reshape(-1)  # (n_draws,)

        # Jacobian: log|I - param*W| * T (pure numpy, respects logdet_method)
        jacobian = self._logdet_numpy_vec_fn(param_draws) * T  # (n_draws,)
        ll_jac = jacobian[:, None] / n  # (n_draws, 1)

        # Add Jacobian to each variable in the log_likelihood group
        n_chains = idata.posterior.sizes["chain"]
        n_draws_per_chain = idata.posterior.sizes["draw"]
        ll_jac_3d = ll_jac.reshape(n_chains, n_draws_per_chain, 1)  # broadcast over obs

        new_vars = {}
        for var_name in list(idata.log_likelihood.data_vars):
            da = idata.log_likelihood[var_name]
            # Use numpy addition + broadcast to avoid xarray alignment issues
            # when the observation dimension name differs (e.g., "obs_dim_0" vs "obs_dim")
            new_vals = da.values + ll_jac_3d
            new_vars[var_name] = xr.DataArray(
                new_vals,
                dims=da.dims,
                coords={k: v for k, v in da.coords.items() if k != da.dims[-1]},
            )

        idata["log_likelihood"] = xr.Dataset(new_vars)

    # ------------------------------------------------------------------
    # Spatial effects (identical body for cross-section and panel)
    # ------------------------------------------------------------------

    def spatial_effects(
        self, return_posterior_samples: bool = False
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Compute Bayesian inference for direct, indirect, and total impacts.

        Computes impact measures for each posterior draw, then summarises
        the posterior distribution with means, 95% credible intervals, and
        Bayesian p-values.  This is the fully Bayesian analog of the
        simulation-based approach in :cite:t:`lesage2009IntroductionSpatial`
        and the asymptotic variance formulas in
        :cite:t:`arbia2020TestingImpact`.

        Models without a spatial lag on y do not exhibit global
        feedback propagation through :math:`(I-\\rho W)^{-1}`. However,
        models with spatially lagged covariates (SLX, SDEM) can still
        have non-zero neighbour spillovers captured in the indirect term.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, return a ``(DataFrame, dict)`` tuple where the
            dict contains the full posterior draws under keys
            ``"direct"``, ``"indirect"``, and ``"total"``.  Default
            ``False``.

        Returns
        -------
        pd.DataFrame or tuple of (pd.DataFrame, dict)
            If *return_posterior_samples* is ``False`` (default), returns
            a DataFrame indexed by feature names with columns for posterior
            means, credible-interval bounds, and Bayesian p-values.

            If *return_posterior_samples* is ``True``, returns
            ``(DataFrame, dict)`` where the dict has keys
            ``"direct"``, ``"indirect"``, ``"total"``, each mapping
            to a ``(G, k)`` array of posterior draws.
        """
        from ..diagnostics.spatial_effects import _build_effects_dataframe

        self._require_fit()
        direct_samples, indirect_samples, total_samples = (
            self._compute_spatial_effects_posterior()
        )

        # Determine feature names based on the shape of the posterior samples.
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

        model_type = self.__class__.__name__

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type=model_type,
        )

        if return_posterior_samples:
            posterior_samples = {
                "direct": direct_samples,
                "indirect": indirect_samples,
                "total": total_samples,
            }
            return df, posterior_samples
        return df

    # ------------------------------------------------------------------
    # LM diagnostics utilities (shared registry execution)
    # ------------------------------------------------------------------

    @staticmethod
    def _lazy_lm_test(module: str, name: str):
        """Return a callable that lazily imports ``name`` from ``module``.

        Used in ``_spatial_diagnostics_tests`` registries to avoid
        circular imports at module-load time.
        """

        def _fn(model):
            mod = importlib.import_module(module)
            return getattr(mod, name)(model)

        return _fn

    @staticmethod
    def _run_lm_diagnostics(model, tests) -> pd.DataFrame:
        """Execute a registry of LM tests and return a tidy DataFrame.

        Shared helper used by ``spatial_diagnostics`` in
        :class:`SpatialModel`, :class:`SpatialPanelModel`, and the flow
        model bases.
        """
        from ..diagnostics.lmtests import BayesianLMTestResult

        rows: dict[str, dict] = {}
        raw_results: dict[str, BayesianLMTestResult] = {}

        for test_fn, label in tests:
            try:
                result = test_fn(model)
                rows[label] = {
                    "statistic": result.mean,
                    "median": result.median,
                    "df": result.df,
                    "p_value": result.bayes_pvalue,
                    "ci_lower": result.credible_interval[0],
                    "ci_upper": result.credible_interval[1],
                }
                raw_results[label] = result
            except (ValueError, np.linalg.LinAlgError) as exc:
                rows[label] = {
                    "statistic": np.nan,
                    "median": np.nan,
                    "df": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "error": str(exc),
                }

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "test"

        idata = model._idata
        n_draws = int(idata.posterior.sizes.get("draw", 0))
        n_chains = int(idata.posterior.sizes.get("chain", 1))
        df.attrs["model_type"] = model.__class__.__name__
        df.attrs["n_draws"] = n_draws * n_chains
        df.attrs["_raw_results"] = raw_results

        return df

    def spatial_diagnostics(self) -> pd.DataFrame:
        """Run Bayesian LM specification tests and return a summary table.

        Iterates over the class-level ``_spatial_diagnostics_tests``
        registry and calls each test function on this fitted model,
        collecting the results into a tidy DataFrame.  The set of tests
        depends on the model type.

        Requires the model to have been fit (``.fit()`` called).  For
        cross-sectional models a spatial weights matrix ``W`` must also
        have been supplied at construction time.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by test name with columns ``statistic``
            (posterior mean), ``median``, ``df`` (degrees of freedom for
            the :math:`\\chi^2` reference), ``p_value`` (Bayesian p-value
            ``1 - chi2.cdf(mean, df)``), and ``ci_lower`` / ``ci_upper``
            (95% credible interval).  The DataFrame carries
            ``attrs["model_type"]`` and ``attrs["n_draws"]`` metadata.

        Raises
        ------
        RuntimeError
            If the model has not been fit yet.
        ValueError
            If a cross-sectional model was constructed without ``W``.

        See Also
        --------
        spatial_diagnostics_decision : Model-selection decision based on
            the test results.
        spatial_effects : Posterior inference for direct/indirect/total
            impacts.
        """
        self._require_fit()
        if self._diagnostics_require_W:
            self._require_W()
        return self._run_lm_diagnostics(self, self._spatial_diagnostics_tests)

    def spatial_diagnostics_decision(
        self, alpha: float = 0.05, format: str = "graphviz"
    ):
        """Return a model-selection decision from Bayesian LM test results.

        Implements the decision tree from :cite:t:`koley2024UseNot`
        (the Bayesian analogue of the classical ``stge_kb`` procedure
        in :cite:t:`anselin1996SimpleDiagnostic`), adapted for panel
        models following :cite:t:`elhorst2014SpatialEconometrics` when
        invoked on a panel subclass.  See the cross-sectional /
        panel-specific docstrings on the leaf classes for the full set
        of branches consulted.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the Bayesian p-values.
        format : {"graphviz", "ascii", "model"}, default "graphviz"
            Output format. ``"model"`` returns the recommended-model
            name string. ``"ascii"`` returns an indented box-drawing
            rendering of the full decision tree with the chosen path
            highlighted. ``"graphviz"`` returns a
            :class:`graphviz.Digraph` object that renders inline in
            Jupyter; if the optional ``graphviz`` package is not
            installed a :class:`UserWarning` is issued and the ASCII
            rendering is returned instead.

        Returns
        -------
        str or graphviz.Digraph
            Recommended model name when ``format="model"``, an ASCII
            tree string when ``format="ascii"``, or a
            ``graphviz.Digraph`` when ``format="graphviz"`` (with ASCII
            fallback on missing dep).

        See Also
        --------
        spatial_diagnostics : Compute the Bayesian LM test statistics.
        """
        from ..diagnostics import _decision_trees as _dt

        diag = self.spatial_diagnostics()
        model_type = self.__class__.__name__
        tprefix = self._decision_test_prefix
        pprefix = self._decision_predicate_prefix

        def _sig(test_name: str) -> bool:
            if test_name not in diag.index:
                return False
            pval = diag.loc[test_name, "p_value"]
            return not np.isnan(pval) and pval < alpha

        def _pval(label: str) -> float:
            return diag.loc[f"{tprefix}{label}", "p_value"]

        def _lag_le_error() -> bool:
            return _pval("LM-Lag") <= _pval("LM-Error")

        def _robust_lag_le_error() -> bool:
            # OLS-tree tie-break.  When both naive AND both robust tests
            # fire, route to the dominant single-channel model by the
            # smaller robust p-value.
            return _pval("Robust-LM-Lag") <= _pval("Robust-LM-Error")

        def _lag_sdm_le_error_sdem() -> bool:
            return _pval("Robust-LM-Lag-SDM") <= _pval("Robust-LM-Error-SDEM")

        def _lag_sem_le_wx_sem() -> bool:
            # SEM-tree tie-break (cross-sectional only).
            return _pval("Robust-LM-Lag") <= _pval("Robust-LM-WX")

        spec = getattr(_dt, self._decision_spec_attr)(model_type)
        decision, path = _dt.evaluate(
            spec,
            sig_lookup=_sig,
            predicate_lookup={
                f"{pprefix}lag_pval_le_error_pval": _lag_le_error,
                f"{pprefix}robust_lag_pval_le_error_pval": _robust_lag_le_error,
                f"{pprefix}lag_sdm_pval_le_error_sdem_pval": _lag_sdm_le_error_sdem,
                f"{pprefix}lag_sem_pval_le_wx_sem_pval": _lag_sem_le_wx_sem,
            },
        )

        p_values: dict[str, float] = {}
        for test_name in diag.index:
            pv = diag.loc[test_name, "p_value"]
            if not np.isnan(pv):
                p_values[str(test_name)] = float(pv)

        return _dt.render(
            spec,
            path,
            decision,
            p_values=p_values,
            alpha=alpha,
            fmt=format,
            title=f"{model_type} decision tree (alpha={alpha})",
        )

    def _require_W(self):
        """Raise if no spatial weights matrix was supplied.

        Default implementation is a no-op for subclasses (e.g. panel
        models) that always carry a ``W``; the cross-sectional base
        overrides this to enforce the check.
        """
        return None

    # ------------------------------------------------------------------
    # Shared W-derived helpers
    # ------------------------------------------------------------------

    @cached_property
    def _W_pt_sparse(self):
        """PyTensor sparse variable wrapping the model's sparse W.

        Cached so repeated PyMC model builds reuse the same symbolic
        sparse operator and avoid the ``O(n²)`` dense materialisation
        that ``pt.as_tensor_variable(self._W_dense)`` performs each
        time.  The sparse attribute being wrapped is controlled by the
        :attr:`_pt_sparse_attr` class hook (``"_W_sparse"`` for
        cross-section, ``"_W_sparse_NT"`` for panel).

        Use with :func:`pytensor.sparse.structured_dot` (vector inputs
        must first be reshaped to ``(n, 1)`` because the vector
        overload's backward pass is broken in PyTensor).
        """
        import scipy.sparse as _sp
        from pytensor import sparse as _pts

        W_attr = getattr(self, self._pt_sparse_attr)
        return _pts.as_sparse_variable(_sp.csc_matrix(W_attr))

    @cached_property
    def _T_ww(self) -> float:
        """Trace of W'W + W², cached on first access.

        Computed as ``||W||_F² + sum(W * W')`` using sparse operations,
        which is O(nnz) rather than O(n²).
        """
        from ..graph import sparse_trace_WtW_plus_WW

        return sparse_trace_WtW_plus_WW(self._W_sparse)

    def _batch_mean_row_sum(self, rho_draws: np.ndarray) -> np.ndarray:
        """Compute mean row sum of ``(I - ρW)^{-1}`` for each posterior draw.

        For row-standardised W this is the scalar ``1/(1 - ρ)``.
        For non-row-standardised W the eigenvalue decomposition is used:
        ``mean_row_sum = (1/n) * ones' V diag(1/(1-ρω)) V^{-1} ones``,
        where the vector ``c = V^{-1} ones`` is pre-computed once.

        Parameters
        ----------
        rho_draws : np.ndarray, shape (G,)
            Spatial autoregressive parameter draws.

        Returns
        -------
        np.ndarray, shape (G,)
            Mean row sum for each draw.
        """
        if self._is_row_std:
            return 1.0 / (1.0 - rho_draws)

        # Eigenvalue-based computation: precompute c = V^{-1} @ ones once.
        if not hasattr(self, "_eig_inv_ones"):
            W_dense = self._W_dense
            eigs, V = np.linalg.eig(W_dense)
            self._W_eigs_full = eigs.real.astype(np.float64)
            self._V_full = V.real.astype(np.float64)
            self._eig_inv_ones = np.linalg.solve(
                self._V_full, np.ones(W_dense.shape[0])
            )

        c = self._eig_inv_ones
        eigs = self._W_eigs_full
        V_col_sums = self._V_full.sum(axis=0)  # (n,)
        from ..diagnostics.spatial_effects import _chunked_eig_means

        return _chunked_eig_means(rho_draws, eigs, weights=V_col_sums * c)

    def _batch_mean_row_sum_MW(self, rho_draws: np.ndarray) -> np.ndarray:
        """Compute mean row sum of ``(I - ρW)^{-1} W`` for each posterior draw.

        For row-standardised W this equals ``1/(1 - ρ)`` (same as
        :meth:`_batch_mean_row_sum`) because row sums of ``M @ W`` equal
        row sums of ``M`` when W is row-standardised.

        For non-row-standardised W the eigenvalue decomposition is used:
        ``mean_row_sum_MW = (1/n) * ones' V diag(ω/(1-ρω)) V^{-1} ones``.

        Parameters
        ----------
        rho_draws : np.ndarray, shape (G,)
            Spatial autoregressive parameter draws.

        Returns
        -------
        np.ndarray, shape (G,)
            Mean row sum of ``M @ W`` for each draw.
        """
        if self._is_row_std:
            return 1.0 / (1.0 - rho_draws)

        # Ensure eigenvalue decomposition is available
        if not hasattr(self, "_eig_inv_ones"):
            _ = self._batch_mean_row_sum(rho_draws[:1])

        c = self._eig_inv_ones
        eigs = self._W_eigs_full
        V_col_sums = self._V_full.sum(axis=0)  # (n,)
        from ..diagnostics.spatial_effects import _chunked_eig_means

        return _chunked_eig_means(rho_draws, eigs, weights=eigs * V_col_sums * c)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fit(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

    def _posterior_mean(self, var: str) -> np.ndarray:
        return self._idata.posterior[var].mean(("chain", "draw")).to_numpy()
