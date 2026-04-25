"""Base class for Bayesian spatial regression models."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp
from formulaic import model_matrix
from libpysal.graph import Graph


def _parse_W(
    W: Union[Graph, sp.spmatrix],
    n: int,
) -> sp.csr_matrix:
    """Validate and normalise a spatial weights argument to CSR.

    Parameters
    ----------
    W :
        Either a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
        matrix.
    n :
        Expected number of spatial units (must match both dimensions of W).

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-compressed version of W.

    Raises
    ------
    TypeError
        If *W* is not a Graph or scipy sparse matrix.
    ValueError
        If *W* is not square or its size does not match *n*.

    Warns
    -----
    UserWarning
        If *W* does not appear to be row-standardised.
    """
    if isinstance(W, Graph):
        W_csr = W.sparse.tocsr().astype(np.float64)
        transform = getattr(W, "transformation", None)
        row_std = transform in ("r", "R")
    elif sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
        row_sums = np.asarray(W_csr.sum(axis=1)).ravel()
        row_std = bool(np.allclose(row_sums, 1.0, atol=1e-6))
    elif hasattr(W, "sparse") and hasattr(W, "transform"):
        # Legacy libpysal.weights.W object — not accepted directly.
        raise TypeError(
            "W appears to be a legacy libpysal.weights.W object. "
            "Convert it to a libpysal.graph.Graph first: "
            "Graph.from_W(w), or pass w.sparse (the scipy sparse matrix) directly."
        )
    else:
        raise TypeError(
            f"W must be a libpysal.graph.Graph or a scipy sparse matrix, "
            f"got {type(W).__name__}."
        )

    if W_csr.ndim != 2 or W_csr.shape[0] != W_csr.shape[1]:
        raise ValueError(
            f"W must be a square matrix, got shape {W_csr.shape}."
        )
    if W_csr.shape[0] != n:
        raise ValueError(
            f"W has shape {W_csr.shape} but data has {n} observations. "
            "W must be an n\u00d7n matrix."
        )
    if not row_std:
        warnings.warn(
            "W does not appear to be row-standardised (row sums \u2260 1). "
            "Most spatial models assume W is row-standardised; results may be "
            "unreliable otherwise. For a scipy sparse matrix normalise rows "
            "manually (divide each row by its sum). To use a libpysal.graph.Graph "
            "set its transformation attribute: "
            "graph = graph.transform('r').",
            UserWarning,
            stacklevel=3,
        )
    return W_csr, row_std


class SpatialModel(ABC):
    """Base class for Bayesian spatial regression models. Models follow the notation
    of :cite:p:`anselin1988SpatialEconometrics` and :cite:p:`lesage2009IntroductionSpatial`.
    The API supports both formula and matrix input modes.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula string, e.g. ``"price ~ poverty + rev_rating"``.
        If provided, ``data`` must also be supplied. An intercept is included
        by default; suppress with ``"y ~ x - 1"``.
    data : DataFrame or GeoDataFrame, optional
        Data source when using formula mode.
    y : array-like, optional
        Dependent variable. Required in matrix mode.
    X : array-like, optional
        Predictor matrix. Required in matrix mode. If a DataFrame, column
        names are preserved for labelling.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``. Accepts a
        :class:`libpysal.graph.Graph` (the modern libpysal graph API) or any
        :class:`scipy.sparse` matrix.  The legacy :class:`libpysal.weights.W`
        object is **not** accepted directly; pass ``w.sparse`` to use the
        underlying sparse matrix, or convert with
        ``libpysal.graph.Graph.from_W(w)``.
        W should be row-standardised; a :class:`UserWarning` is raised if not.
    priors : dict, optional
        Override default priors. Keys depend on the model subclass; see
        each model's docstring for supported keys.
    logdet_method : str
        How to compute ``log|I - rho*W|``. ``"eigenvalue"`` (default)
        pre-computes W's eigenvalues once and evaluates O(n) per step;
        ``"exact"`` uses symbolic pytensor det (slow for n > 500);
        ``"grid"`` uses spline interpolation (approximate);
        ``"full"`` uses sparse-LU grid (MATLAB-style ``lndetfull``);
        ``"int"`` uses sparse-LU + spline interpolation (``lndetint``);
        ``"mc"`` uses Monte Carlo trace approximation (``lndetmc``);
        ``"ichol"`` uses ILU-based approximation (``lndetichol`` analog).
    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal,
        yielding a model that is robust to heavy-tailed outliers. When
        ``robust=True``, a ``nu`` (degrees of freedom) parameter is added
        to the model with an :math:`\\mathrm{Exp}(\\lambda_\\nu)` prior (default
        ``nu_lam = 1/30``, mean ≈ 30). The ``nu`` prior can be controlled
        via the ``priors`` dict with key ``nu_lam``.
    w_vars : list of str, optional
        Names of X columns to spatially lag. Only relevant for models that
        include ``WX`` terms (SLX, SDM, SDEM and their panel/Tobit variants).
        By default all non-constant columns are lagged. Pass a subset to
        restrict which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        priors: Optional[dict] = None,
        logdet_method: str = "eigenvalue",
        robust: bool = False,
        w_vars: Optional[list] = None,
    ):
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.robust = robust
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None
        self._W_dense_cache: Optional[np.ndarray] = None

        if formula is not None:
            if data is None:
                raise ValueError("data must be provided when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
        elif y is not None and X is not None:
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
        else:
            raise ValueError(
                "Provide either (formula, data) or (y, X)."
            )

        if W is not None:
            # Validate W and store as CSR sparse matrix.
            # Dense conversion is deferred to _W_dense (lazy property).
            self._W_sparse, self._is_row_std = _parse_W(W, len(self._y))
            # Pre-compute eigenvalues of the N×N matrix once (O(n³)) so that
            # logdet and effect calculations can use O(n) eigenvalue formulas.
            self._W_eigs: np.ndarray = np.linalg.eigvals(
                self._W_sparse.toarray().astype(np.float64)
            )
            self._wx_column_indices = self._spatial_lag_column_indices(self._X, self._feature_names)
            if w_vars is not None:
                unknown = [v for v in w_vars if v not in self._feature_names]
                if unknown:
                    raise ValueError(
                        f"w_vars contains names not found in X columns: {unknown}. "
                        f"Available: {self._feature_names}"
                    )
                self._wx_column_indices = [
                    i for i in self._wx_column_indices if self._feature_names[i] in w_vars
                ]
            self._wx_feature_names = [self._feature_names[i] for i in self._wx_column_indices]
            # Pre-compute spatial lags using sparse matmul (no dense materialisation).
            self._Wy: np.ndarray = np.asarray(self._W_sparse @ self._y, dtype=np.float64)
            if self._wx_column_indices:
                self._WX = np.asarray(
                    self._W_sparse @ self._X[:, self._wx_column_indices], dtype=np.float64
                )
            else:
                self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
        else:
            # W-free mode: no spatial structure; spec tests require W to be supplied.
            self._W_sparse = None
            self._is_row_std = False
            self._W_eigs = None
            self._wx_column_indices: list[int] = []
            self._wx_feature_names: list[str] = []
            self._Wy = np.zeros(len(self._y), dtype=np.float64)
            self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
            if w_vars is not None:
                raise ValueError("w_vars requires a spatial weights matrix W.")

    @property
    def _W_dense(self) -> np.ndarray:
        """Dense weight matrix, materialised lazily on first access."""
        if self._W_dense_cache is None:
            self._W_dense_cache = np.asarray(
                self._W_sparse.toarray(), dtype=np.float64
            )
        return self._W_dense_cache

    # ------------------------------------------------------------------
    # Input parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_formula(formula: str, data: pd.DataFrame):
        """Parse formula/data inputs into ``y`` and ``X`` arrays.

        Parameters
        ----------
        formula : str
            Wilkinson-style formula containing ``lhs ~ rhs``.
        data : pandas.DataFrame
            Tabular data source referenced by the formula.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Dependent variable, design matrix, and feature names.
        """
        lhs_name, rhs = formula.split("~", 1)
        lhs_name = lhs_name.strip()
        rhs = rhs.strip()

        # Build RHS model matrix (handles intercept, interactions, transforms)
        X_mm = model_matrix(rhs, data)
        feature_names = list(X_mm.columns)
        X_arr = np.asarray(X_mm, dtype=np.float64)

        y_arr = np.asarray(data[lhs_name], dtype=np.float64)
        return y_arr, X_arr, feature_names

    @staticmethod
    def _parse_matrices(y, X):
        """Parse matrix-mode inputs and infer feature names.

        Parameters
        ----------
        y : array-like
            Dependent variable.
        X : array-like or pandas.DataFrame
            Predictor matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Dependent variable, design matrix, and feature names.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        return y_arr, X_arr, feature_names

    @staticmethod
    def _spatial_lag_column_indices(X: np.ndarray, feature_names: list[str]) -> list[int]:
        """Return indices of regressors that should receive spatial lags.

        Constant columns are treated as intercept-like and excluded, which
        avoids adding redundant ``W * intercept`` terms to SLX/Durbin models.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.
        feature_names : list[str]
            Column labels aligned with ``X``.

        Returns
        -------
        list[int]
            Column indices eligible for spatial lags.
        """
        indices: list[int] = []
        for j, name in enumerate(feature_names):
            column = X[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _add_nu_prior(self, model: pm.Model) -> pm.Model:
        """Add the degrees-of-freedom prior for robust (Student-t) models.

        Called inside ``_build_pymc_model`` when ``self.robust`` is True.
        Uses an :math:`\\mathrm{Exp}(\\lambda_\\nu)` prior on ``nu`` with rate ``nu_lam`` (default
        1/30, giving mean ≈ 30, favouring near-Normal tails). A lower
        bound of 2 is enforced so that the variance exists.

        Parameters
        ----------
        model : pymc.Model
            The model context in which to add the ``nu`` prior.

        Returns
        -------
        pymc.Model
            The same model context (``nu`` is added as a side effect).
        """
        nu_lam = self.priors.get("nu_lam", 1.0 / 30.0)
        pm.Truncated("nu", pm.Exponential.dist(lam=nu_lam), lower=2.0)
        return model

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model. Subclasses implement this."""

    def _beta_names(self) -> list[str]:
        """Return coefficient labels used for posterior summaries.

        Returns
        -------
        list[str]
            Coefficient labels aligned with the ``beta`` parameter.
        """
        return list(self._feature_names)

    def _model_coords(self) -> dict[str, list[str]]:
        """Return PyMC coordinate labels for named dimensions.

        Returns
        -------
        dict[str, list[str]]
            Coordinates passed to :class:`pymc.Model`.
        """
        return {"coefficient": self._beta_names()}

    @staticmethod
    def _rename_summary_index(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Strip the ``beta[...]`` wrapper from coefficient row labels.

        Parameters
        ----------
        summary_df : pandas.DataFrame
            ArviZ summary output.

        Returns
        -------
        pandas.DataFrame
            Summary with human-readable coefficient row labels.
        """
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
    # Public API
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
            Additional keyword arguments forwarded to ``pm.sample``.

        Returns
        -------
        arviz.InferenceData
        """
        model = self._build_pymc_model()
        self._pymc_model = model
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

        # Eigenvalue-based Jacobian: log|I - param*W| * T (pure numpy)
        eigs = self._W_eigs.real.astype(np.float64)
        jacobian = np.array([np.sum(np.log(np.abs(1.0 - v * eigs))) for v in param_draws]) * T  # (n_draws,)
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
                new_vals, dims=da.dims, coords={k: v for k, v in da.coords.items() if k != da.dims[-1]}
            )

        idata["log_likelihood"] = xr.Dataset(new_vars)

    @property
    def inference_data(self) -> Optional[az.InferenceData]:
        """Return the ArviZ InferenceData from the most recent fit.

        Returns
        -------
        arviz.InferenceData or None
            The inference data object, or ``None`` if the model has not
            been fit yet.
        """
        return self._idata

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model object built for the most recent fit.

        Returns
        -------
        pymc.Model or None
            The model object used by :meth:`fit`, or ``None`` if the instance
            has not been fit yet.
        """
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

    def spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute posterior direct, indirect, and total impacts.

        Models without a spatial lag on y do not exhibit global
        feedback propagation through :math:`(I-\\rho W)^{-1}`. However,
        models with spatially lagged covariates (SLX, SDEM) can still
        have non-zero neighbour spillovers captured in the indirect term.

        Returns
        -------
        dict with keys "direct", "indirect", "total", each shaped
        (n_features,) containing posterior mean impacts.
        """
        self._require_fit()
        return self._compute_spatial_effects()

    @abstractmethod
    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute model-specific impact measures.

        Returns
        -------
        dict
            Dictionary with direct, indirect, and total effects.
        """

    @abstractmethod
    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Fitted mean vector.
        """

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fit(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

    def _require_W(self):
        """Raise if no spatial weights matrix was supplied."""
        if self._W_sparse is None:
            raise ValueError(
                "This method requires a spatial weights matrix W. "
                "Pass W when constructing the model."
            )

    def _posterior_mean(self, var: str) -> np.ndarray:
        return self._idata.posterior[var].mean(("chain", "draw")).to_numpy()

    def fitted_values(self) -> np.ndarray:
        """Return fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        self._require_fit()
        return self._fitted_mean_from_posterior()

    def residuals(self) -> np.ndarray:
        """Return residuals on the observed scale.

        Returns
        -------
        np.ndarray
            Residual vector ``y - fitted_values``.
        """
        self._require_fit()
        return self._y - self.fitted_values()

    def __repr__(self) -> str:
        n, k = self._X.shape
        return f"{self.__class__.__name__}(n={n}, k={k}, features={self._feature_names})"
