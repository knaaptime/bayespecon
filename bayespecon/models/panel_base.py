"""Base classes and helpers for spatial panel models."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.sparse as sp
from formulaic import model_matrix
from libpysal.graph import Graph


def _demean_panel(y: np.ndarray, X: np.ndarray, N: int, T: int, model: int):
    """Apply panel demeaning transformation.

    Parameters
    ----------
    y : np.ndarray
        Stacked dependent variable of shape ``(N*T,)``.
    X : np.ndarray
        Stacked regressor matrix of shape ``(N*T, k)``.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    model : int
        Fixed-effects mode: 0 pooled, 1 unit FE, 2 time FE, 3 two-way FE.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Demeaned ``(y, X)`` arrays in stacked format.
    """
    y2 = y.reshape(T, N)
    X3 = X.reshape(T, N, X.shape[1])

    if model == 0:
        y_with = y2
        X_with = X3
    elif model == 1:
        y_with = y2 - y2.mean(axis=0, keepdims=True)
        X_with = X3 - X3.mean(axis=0, keepdims=True)
    elif model == 2:
        y_with = y2 - y2.mean(axis=1, keepdims=True)
        X_with = X3 - X3.mean(axis=1, keepdims=True)
    elif model == 3:
        y_i = y2.mean(axis=0, keepdims=True)
        y_t = y2.mean(axis=1, keepdims=True)
        y_g = y2.mean()
        y_with = y2 - y_i - y_t + y_g

        X_i = X3.mean(axis=0, keepdims=True)
        X_t = X3.mean(axis=1, keepdims=True)
        X_g = X3.mean(axis=(0, 1), keepdims=True)
        X_with = X3 - X_i - X_t + X_g
    else:
        raise ValueError("model must be one of {0,1,2,3}")

    return y_with.reshape(-1), X_with.reshape(-1, X.shape[1])


def _as_dense_W(W: Union[Graph, sp.spmatrix, np.ndarray], N: int, T: int) -> np.ndarray:
    """Convert graph/sparse/array weights into dense panel-compatible matrix.

    Parameters
    ----------
    W : Graph, scipy.sparse, or np.ndarray
        Either an ``N x N`` cross-sectional matrix or an ``(N*T) x (N*T)``
        block-diagonal panel matrix. Public APIs accept only Graph or sparse
        inputs; ndarray is supported here for internal use.
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    np.ndarray
        Dense panel weights matrix.
    """
    if isinstance(W, Graph):
        Wn = W.sparse.toarray().astype(float)
    elif sp.issparse(W):
        Wn = W.toarray().astype(float)
    else:
        Wn = np.asarray(W, dtype=float)

    if Wn.shape == (N, N):
        return np.kron(np.eye(T), Wn)
    if Wn.shape == (N * T, N * T):
        return Wn

    raise ValueError(
        f"W has shape {Wn.shape}; expected (N,N)=({N},{N}) or (N*T,N*T)=({N*T},{N*T})."
    )


def _parse_panel_W(
    W: Union[Graph, sp.spmatrix],
    N: int,
    T: int,
) -> sp.csr_matrix:
    """Validate W and return it as a CSR sparse matrix sized ``(N, N)``.

    Accepts a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
    matrix. Raises a :class:`ValueError` if the shape is incompatible with
    *N* (and optionally *T*). Issues a :class:`UserWarning` when *W* does not
    appear to be row-standardised.

    Returns the CSR representation of the ``N x N`` cross-sectional matrix;
    callers that need the full ``(N*T) x (N*T)`` Kronecker form should use
    :func:`_as_dense_W` or build the sparse Kronecker product themselves.
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
        raise ValueError(f"W must be square, got shape {W_csr.shape}.")

    if W_csr.shape[0] == N:
        pass  # N x N unit matrix — expected
    elif W_csr.shape[0] == N * T:
        # Caller passed the full block matrix; extract N x N block for storage.
        # We keep it as-is but raise if neither shape matches.
        pass
    else:
        raise ValueError(
            f"W has shape {W_csr.shape} but data has N={N} units (T={T} periods). "
            f"W must be ({N},{N}) or ({N*T},{N*T})."
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


class SpatialPanelModel(ABC):
    """Base class for static spatial panel models with FE transforms.

    Parameters
    ----------
    formula, data, y, X
        Either formula mode (formula + data) or matrix mode (y + X).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)`` (preferred) or the full
        ``(N*T, N*T)`` block-diagonal panel matrix. Accepts a
        :class:`libpysal.graph.Graph` (the modern libpysal graph API) or any
        :class:`scipy.sparse` matrix.  The legacy :class:`libpysal.weights.W`
        object is **not** accepted directly; pass ``w.sparse`` to use the
        underlying sparse matrix, or convert with
        ``libpysal.graph.Graph.from_W(w)``.
        W should be row-standardised; a :class:`UserWarning` is raised if not.
    unit_col, time_col
        Required in formula mode for robust panel sorting and N/T inference.
    N, T
        Required in matrix mode if not inferable.
    model
        0 pooled, 1 unit FE, 2 time FE, 3 two-way FE.
    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal,
        yielding a model that is robust to heavy-tailed outliers. When
        ``robust=True``, a ``nu`` (degrees of freedom) parameter is added
        to the model with an ``Exponential(lam=nu_lam)`` prior (default
        ``nu_lam = 1/30``, mean ≈ 30). The ``nu`` prior can be controlled
        via the ``priors`` dict with key ``nu_lam``.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        unit_col: Optional[str] = None,
        time_col: Optional[str] = None,
        N: Optional[int] = None,
        T: Optional[int] = None,
        model: int = 0,
        priors: Optional[dict] = None,
        logdet_method: str = "eigenvalue",
        robust: bool = False,
    ):
        if W is None:
            raise ValueError("W is required.")

        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.model = int(model)
        self.robust = robust
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None
        self._W_dense_cache: Optional[np.ndarray] = None

        if formula is not None:
            if data is None:
                raise ValueError("data is required with formula mode.")
            if unit_col is None or time_col is None:
                raise ValueError("unit_col and time_col are required in formula mode.")

            d = data.sort_values([time_col, unit_col]).reset_index(drop=True)
            lhs, rhs = formula.split("~", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()

            X_mm = model_matrix(rhs, d)
            self._feature_names = list(X_mm.columns)
            y_arr = np.asarray(d[lhs], dtype=float)
            X_arr = np.asarray(X_mm, dtype=float)

            units = d[unit_col].nunique()
            times = d[time_col].nunique()
            if units * times != len(d):
                raise ValueError("Data are not a balanced panel after sorting by time/unit.")
            self._N = units
            self._T = times
            self._panel_index = d[[time_col, unit_col]].copy()
        elif y is not None and X is not None:
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            if isinstance(X, pd.DataFrame):
                self._feature_names = list(X.columns)
                X_arr = X.to_numpy(dtype=float)
            else:
                X_arr = np.asarray(X, dtype=float)
                self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

            if N is None or T is None:
                raise ValueError("N and T are required in matrix mode.")
            self._N = int(N)
            self._T = int(T)
            self._panel_index = None
            if self._N * self._T != len(y_arr):
                raise ValueError("N*T must equal number of stacked observations.")
        else:
            raise ValueError("Provide either (formula,data,unit_col,time_col) or (y,X,N,T).")

        self._y_raw = y_arr
        self._X_raw = X_arr

        self._wx_column_indices = self._spatial_lag_column_indices(self._X_raw, self._feature_names)
        self._wx_feature_names = [self._feature_names[i] for i in self._wx_column_indices]

        # Validate W and store as N×N CSR. Dense expansion is deferred.
        self._W_sparse, self._is_row_std = _parse_panel_W(W, self._N, self._T)
        # Eigenvalues of the N×N matrix, pre-computed once (O(n³)).
        self._W_eigs: np.ndarray = np.linalg.eigvals(
            self._W_sparse.toarray().astype(np.float64)
        )

        self._y, self._X = _demean_panel(self._y_raw, self._X_raw, self._N, self._T, self.model)

        # Compute spatial lags with the sparse Kronecker block W⊗I_T, avoiding
        # full dense materialisation.  For an N×N unit matrix W, the panel lag
        # of a stacked vector v (length N*T, ordered T×N) is equivalent to
        # applying W row-wise within each time period.
        self._Wy = self._sparse_panel_lag(self._y)
        if self._wx_column_indices:
            self._WX = np.column_stack([
                self._sparse_panel_lag(self._X[:, j])
                for j in self._wx_column_indices
            ])
        else:
            self._WX = np.empty((self._X.shape[0], 0), dtype=float)

    def _sparse_panel_lag(self, v: np.ndarray) -> np.ndarray:
        """Apply the panel spatial lag W⊗I_T to a stacked vector *v*.

        For each of the T time periods the N-length slice is multiplied by the
        N×N sparse weight matrix, staying sparse until the final concatenation.
        """
        W = self._W_sparse
        N, T = self._N, self._T
        if W.shape[0] == N:
            # Stack is ordered (T, N); reshape and apply W period by period.
            chunks = v.reshape(T, N)  # (T, N)
            return np.asarray((W @ chunks.T).T, dtype=float).ravel()
        else:
            # Full (N*T)×(N*T) block matrix provided.
            return np.asarray(W @ v, dtype=float)

    @property
    def _W_dense(self) -> np.ndarray:
        """Dense (N*T)×(N*T) weight matrix, materialised lazily on first access."""
        if self._W_dense_cache is None:
            self._W_dense_cache = _as_dense_W(self._W_sparse, self._N, self._T)
        return self._W_dense_cache

    @staticmethod
    def _spatial_lag_column_indices(X: np.ndarray, feature_names: list[str]) -> list[int]:
        """Return indices of regressors that should receive spatial lags.

        Constant columns are treated as intercept-like and excluded, which
        avoids adding redundant ``W * intercept`` terms to SLX/Durbin models.

        Parameters
        ----------
        X : np.ndarray
            Raw panel design matrix before FE transformation.
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
        """Construct and return a pm.Model."""

    @abstractmethod
    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute direct/indirect/total effects."""

    @abstractmethod
    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values on transformed scale."""

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

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Sample the posterior for the panel model.

        Parameters
        ----------
        draws : int, default=2000
            Number of post-tuning draws per chain.
        tune : int, default=1000
            Number of tuning draws per chain.
        chains : int, default=4
            Number of chains.
        target_accept : float, default=0.9
            NUTS target acceptance probability.
        random_seed : int, optional
            Random seed used by PyMC.
        **sample_kwargs
            Extra keyword arguments forwarded to :func:`pymc.sample`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples and diagnostics.
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

    def _require_fit(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

    def _posterior_mean(self, var: str) -> np.ndarray:
        return self._idata.posterior[var].mean(("chain", "draw")).to_numpy()

    def summary(self, var_names: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """Return posterior summary table.

        Parameters
        ----------
        var_names : list, optional
            Variable names to include.
        **kwargs
            Additional arguments passed to :func:`arviz.summary`.

        Returns
        -------
        pandas.DataFrame
            Posterior summary table.
        """
        self._require_fit()
        summary_df = az.summary(self._idata, var_names=var_names, **kwargs)
        return self._rename_summary_index(summary_df)

    def fitted_values(self) -> np.ndarray:
        """Return fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Fitted values on transformed panel scale.
        """
        self._require_fit()
        return self._fitted_mean_from_posterior()

    def residuals(self) -> np.ndarray:
        """Return transformed residuals ``y - fitted``.

        Returns
        -------
        np.ndarray
            Residual vector on transformed panel scale.
        """
        self._require_fit()
        return self._y - self.fitted_values()

    def spatial_effects(self) -> dict[str, np.ndarray]:
        """Return direct, indirect, and total effects.

        Returns
        -------
        dict
            Dictionary containing effects and feature names.
        """
        self._require_fit()
        return self._compute_spatial_effects()

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(N={self._N}, T={self._T}, n={n}, "
            f"k={k}, model={self.model}, features={self._feature_names})"
        )


