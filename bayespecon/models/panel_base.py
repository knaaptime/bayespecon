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

from ..logdet import (
    _auto_logdet_method,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)
from .base import _is_row_standardized_csr


def _demean_panel(y: np.ndarray, X: np.ndarray, N: int, T: int, model: int):
    """Apply panel demeaning transformation.

    Implements the within-transformation for two-way fixed-effects panel
    models prior to the spatial filter.  In the SAR-FE setting we model

    .. math::

        y_{it} = \\rho \\sum_j W_{ij} y_{jt} + X_{it}\\beta + \\mu_i
                 + \\alpha_t + \\varepsilon_{it},

    and concentrate out the fixed effects by demeaning *both* sides of
    the equation before the spatial lag is applied.  Because :math:`W`
    operates only across units (within a period), the within-period
    demeaning commutes with :math:`W` (i.e. ``W (M_T y) = M_T (W y)``)
    so the order of "demean then apply :math:`W`" or "apply :math:`W`
    then demean" yields the same likelihood — a fact exploited in
    Lee & Yu (2010) and Elhorst (2014, ch. 3).  This is why
    :func:`bayespecon.models.panel.SARPanel` builds ``Wy`` from the
    *demeaned* ``y`` returned here without an additional Jacobian
    correction beyond the standard :math:`T\\,\\log|I_N - \\rho W|`
    panel Jacobian.

    References
    ----------
    Lee, L.-F. & Yu, J. (2010). Estimation of spatial autoregressive
    panel data models with fixed effects.  *Journal of Econometrics*,
    154(2), 165–185.

    Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional
    Data to Spatial Panels*. Springer.

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

    if model in (1, 3) and T < 2:
        raise ValueError(
            f"Unit fixed effects (model={model}) require T >= 2 to identify "
            "within-unit variation, but T=" + str(T) + " was supplied. "
            "Use model=0 (pooled) or model=2 (time FE) when T=1."
        )

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
        f"W has shape {Wn.shape}; expected (N,N)=({N},{N}) or (N*T,N*T)=({N * T},{N * T})."
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
        row_std = transform in ("r", "R") or _is_row_standardized_csr(W_csr)
    elif sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
        row_std = _is_row_standardized_csr(W_csr)
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
            f"W must be ({N},{N}) or ({N * T},{N * T})."
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
    priors : dict, optional
        Override default priors. Keys depend on the model subclass; see
        each model's docstring for supported keys.
    logdet_method : str, optional
        How to compute ``log|I - rho*W|``. When ``None``, the method is
        auto-selected from the cross-sectional ``N x N`` weights size via
        :func:`bayespecon.logdet._auto_logdet_method`: ``"eigenvalue"`` for
        ``N <= 2000`` and ``"chebyshev"`` otherwise.
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
        logdet_method: str | None = None,
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
                raise ValueError(
                    "Data are not a balanced panel after sorting by time/unit."
                )
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
            raise ValueError(
                "Provide either (formula,data,unit_col,time_col) or (y,X,N,T)."
            )

        self._y_raw = y_arr
        self._X_raw = X_arr

        self._wx_column_indices = self._spatial_lag_column_indices(
            self._X_raw, self._feature_names
        )
        self._wx_feature_names = [
            self._feature_names[i] for i in self._wx_column_indices
        ]

        # Validate W and store as N×N CSR. Dense expansion is deferred.
        self._W_sparse, self._is_row_std = _parse_panel_W(W, self._N, self._T)
        # Eigenvalues of the N×N matrix, pre-computed once (O(n³)).
        self._W_eigs: np.ndarray = np.linalg.eigvals(
            self._W_sparse.toarray().astype(np.float64)
        )

        self._y, self._X = _demean_panel(
            self._y_raw, self._X_raw, self._N, self._T, self.model
        )

        # Compute spatial lags with the sparse Kronecker block W⊗I_T, avoiding
        # full dense materialisation.  For an N×N unit matrix W, the panel lag
        # of a stacked vector v (length N*T, ordered T×N) is equivalent to
        # applying W row-wise within each time period.
        # Store a numpy logdet callable for post-sampling LL Jacobians.
        self._logdet_numpy_fn = make_logdet_numpy_fn(
            self._W_sparse, self._W_eigs.real, method=self.logdet_method
        )
        # Vectorized version: evaluates logdet over an array of rho draws in one call.
        self._logdet_numpy_vec_fn = make_logdet_numpy_vec_fn(
            self._W_sparse, self._W_eigs.real, method=self.logdet_method
        )
        # Store the correct W argument for logdet calls.
        # For eigenvalue method (explicit or auto-selected for n ≤ 2000),
        # pass 1-D eigenvalues to avoid O(n²) dense materialisation.
        # For other methods, pass the 2-D dense matrix.
        _resolved_logdet = (
            self.logdet_method
            if self.logdet_method is not None
            else _auto_logdet_method(self._W_sparse.shape[0])
        )
        self._W_for_logdet: np.ndarray = (
            self._W_eigs.real.astype(np.float64)
            if _resolved_logdet in ("eigenvalue", "chebyshev")
            else self._W_sparse.toarray().astype(np.float64)
        )
        # Store a pytensor logdet callable for use in _build_pymc_model.
        self._logdet_pytensor_fn = make_logdet_fn(
            self._W_for_logdet, method=self.logdet_method, T=self._T
        )

        self._Wy = self._sparse_panel_lag(self._y)
        if self._wx_column_indices:
            # Single batched sparse multiply across all WX columns, replacing
            # the per-column Python loop that previously paid an O(k_wx)
            # overhead.
            self._WX = self._sparse_panel_lag(self._X[:, self._wx_column_indices])
        else:
            self._WX = np.empty((self._X.shape[0], 0), dtype=float)

    def _sparse_panel_lag(self, v: np.ndarray) -> np.ndarray:
        """Apply the panel spatial lag W⊗I_T to a stacked vector or matrix.

        Accepts either a 1-D stacked vector of length ``N*T`` or a 2-D matrix
        ``(N*T, k)`` whose columns will all be lagged in a single batched
        sparse multiply.  Stays sparse until the final reshape.
        """
        W = self._W_sparse
        N, T = self._N, self._T
        v = np.asarray(v, dtype=float)
        if W.shape[0] == N:
            if v.ndim == 1:
                # Stack ordered (T, N); apply W per period in one matmul.
                chunks = v.reshape(T, N)  # (T, N)
                return np.asarray((W @ chunks.T).T, dtype=float).ravel()
            # 2-D path: (N*T, k) → reshape so all periods/columns become a
            # single dense block, perform ONE sparse matmul, then reshape back.
            k = v.shape[1]
            chunks = v.reshape(T, N, k)  # (T, N, k)
            mat = chunks.transpose(1, 0, 2).reshape(N, T * k)
            out = np.asarray(W @ mat, dtype=float)  # (N, T*k)
            return out.reshape(N, T, k).transpose(1, 0, 2).reshape(T * N, k)
        # Full (N*T)×(N*T) block matrix provided.
        if v.ndim == 1:
            return np.asarray(W @ v, dtype=float)
        return np.asarray(W @ v, dtype=float)

    @property
    def _W_dense(self) -> np.ndarray:
        """Dense (N*T)×(N*T) weight matrix, materialised lazily on first access."""
        if self._W_dense_cache is None:
            self._W_dense_cache = _as_dense_W(self._W_sparse, self._N, self._T)
        return self._W_dense_cache

    @property
    def _W_sparse_NT(self) -> "sp.csr_matrix":
        """Sparse (N*T)×(N*T) Kronecker-block weight matrix ``I_T ⊗ W_n``.

        Cached on first access. Used by symbolic (PyMC/PyTensor) likelihoods
        to avoid the O((N*T)²) memory footprint of :attr:`_W_dense` while
        still exposing a single linear operator that can be applied to a
        stacked panel residual vector.
        """
        if not hasattr(self, "_W_sparse_NT_cache") or self._W_sparse_NT_cache is None:
            W = self._W_sparse
            if W.shape[0] == self._N:
                # Force ``csr_matrix`` (not ``csr_array``) so the result is
                # accepted by :mod:`pytensor.sparse`, which currently only
                # supports the legacy ``scipy.sparse`` matrix API.
                self._W_sparse_NT_cache = sp.csr_matrix(
                    sp.kron(sp.eye(self._T, format="csr"), W, format="csr")
                )
            else:
                # Caller already supplied a full (N*T)×(N*T) matrix.
                self._W_sparse_NT_cache = sp.csr_matrix(W)
        return self._W_sparse_NT_cache

    @property
    def _W_pt_sparse(self):
        """PyTensor sparse variable wrapping :attr:`_W_sparse_NT`.

        Cached so that repeated PyMC model builds reuse the same symbolic
        sparse weight operator, avoiding redundant ``as_sparse_variable`` calls.
        """
        if not hasattr(self, "_W_pt_sparse_cache") or self._W_pt_sparse_cache is None:
            from pytensor import sparse as pts

            self._W_pt_sparse_cache = pts.as_sparse_variable(
                sp.csc_matrix(self._W_sparse_NT)
            )
        return self._W_pt_sparse_cache

    @property
    def _T_ww(self) -> float:
        """Trace of W'W + W², cached on first access.

        Computed as ``||W||_F² + sum(W * W')`` using sparse operations,
        which is O(nnz) rather than O(n²).
        """
        if not hasattr(self, "_T_ww_cache"):
            from ..graph import sparse_trace_WtW_plus_WW

            self._T_ww_cache = sparse_trace_WtW_plus_WW(self._W_sparse)
        return self._T_ww_cache

    def _batch_mean_row_sum(self, rho_draws: np.ndarray) -> np.ndarray:
        """Compute mean row sum of (I - rho*W)^{-1} for each posterior draw.

        For row-standardised W this is the scalar ``1/(1 - rho)``.
        For non-row-standardised W the eigenvalue decomposition is used:
        ``mean_row_sum = (1/n) * ones' V diag(1/(1-rho*omega)) V^{-1} ones``,
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
        inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])  # (G, n)
        V_col_sums = self._V_full.sum(axis=0)  # (n,)
        return (1.0 / len(c)) * (inv_eigs * c[None, :]) @ V_col_sums

    def _batch_mean_row_sum_MW(self, rho_draws: np.ndarray) -> np.ndarray:
        """Compute mean row sum of (I - rho*W)^{-1} W for each posterior draw.

        For row-standardised W this equals ``1/(1 - rho)`` (same as
        ``_batch_mean_row_sum``) because row sums of M@W = row sums of M
        when W is row-standardised.

        For non-row-standardised W the eigenvalue decomposition is used:
        ``mean_row_sum_MW = (1/n) * ones' V diag(omega/(1-rho*omega)) V^{-1} ones``.

        Parameters
        ----------
        rho_draws : np.ndarray, shape (G,)
            Spatial autoregressive parameter draws.

        Returns
        -------
        np.ndarray, shape (G,)
            Mean row sum of M@W for each draw.
        """
        if self._is_row_std:
            return 1.0 / (1.0 - rho_draws)

        # Ensure eigenvalue decomposition is available
        if not hasattr(self, "_eig_inv_ones"):
            _ = self._batch_mean_row_sum(rho_draws[:1])

        c = self._eig_inv_ones
        eigs = self._W_eigs_full
        inv_eigs = 1.0 / (1.0 - rho_draws[:, None] * eigs[None, :])  # (G, n)
        eig_weighted = eigs[None, :] * inv_eigs  # omega / (1 - rho*omega)
        V_col_sums = self._V_full.sum(axis=0)  # (n,)
        return (1.0 / len(c)) * (eig_weighted * c[None, :]) @ V_col_sums

    @property
    def _nonintercept_indices(self) -> list[int]:
        """Return indices of non-constant (non-intercept) columns in X.

        This is used to exclude the intercept from impact measures, since
        the intercept has no meaningful spatial effect interpretation.

        Returns
        -------
        list[int]
            Column indices of X that are not constant/intercept columns.
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
        """Return feature names for non-intercept columns.

        Returns
        -------
        list[str]
            Feature names excluding intercept/constant columns.
        """
        return [self._feature_names[i] for i in self._nonintercept_indices]

    @staticmethod
    def _spatial_lag_column_indices(
        X: np.ndarray, feature_names: list[str]
    ) -> list[int]:
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
        """Compute direct/indirect/total effects at posterior mean."""

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

        Notes
        -----
        The full panel log-determinant of the spatial filter is
        :math:`T \\log|I_N - \\rho W|` because the stacked
        :math:`(N T) \\times (N T)` filter is
        :math:`I_T \\otimes (I_N - \\rho W)`, whose determinant is
        :math:`|I_N - \\rho W|^T` by the Kronecker product rule.  Dividing
        by :math:`n = N T` distributes that scalar Jacobian uniformly
        over the per-observation pointwise log-likelihood entries that
        ArviZ expects, so quantities like LOO and WAIC are computed on a
        per-observation log density.  For dynamic panels the time
        dimension is ``T - 1`` (one period is consumed by the lag), so
        callers pass ``T = T - 1`` here.

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

    # ------------------------------------------------------------------
    # Class-level registry of applicable Bayesian LM specification tests.
    # ------------------------------------------------------------------
    _spatial_diagnostics_tests: list[tuple] = []

    def spatial_diagnostics(self) -> pd.DataFrame:
        """Run Bayesian LM specification tests and return a summary table.

        Iterates over the class-level ``_spatial_diagnostics_tests`` registry
        and calls each test function on this fitted model, collecting the
        results into a tidy DataFrame.  The set of tests depends on the
        model type — for example, an OLSPanelFE model runs Panel-LM-Lag,
        Panel-LM-Error, Panel-LM-SDM-Joint, and Panel-LM-SLX-Error-Joint.

        Requires the model to have been fit (``.fit()`` called) and a
        spatial weights matrix ``W`` to have been supplied at construction
        time.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by test name with columns:

            ==============  =====================================================
            Column          Description
            ==============  =====================================================
            statistic       Posterior mean of the LM statistic
            median          Posterior median of the LM statistic
            df              Degrees of freedom for the :math:`\\chi^2` reference
            p_value         Bayesian p-value: ``1 - chi2.cdf(mean, df)``
            ci_lower        Lower bound of 95% credible interval (2.5%)
            ci_upper        Upper bound of 95% credible interval (97.5%)
            ==============  =====================================================

            The DataFrame has ``attrs["model_type"]`` (class name) and
            ``attrs["n_draws"]`` (total posterior draws) metadata.

        Raises
        ------
        RuntimeError
            If the model has not been fit yet.

        See Also
        --------
        spatial_diagnostics_decision : Model-selection decision based on
            the test results.
        """
        from ..diagnostics.bayesian_lmtests import BayesianLMTestResult

        self._require_fit()

        rows: dict[str, dict] = {}
        raw_results: dict[str, BayesianLMTestResult] = {}

        for test_fn, label in self._spatial_diagnostics_tests:
            try:
                result = test_fn(self)
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

        idata = self._idata
        n_draws = int(idata.posterior.sizes.get("draw", 0))
        n_chains = int(idata.posterior.sizes.get("chain", 1))
        df.attrs["model_type"] = self.__class__.__name__
        df.attrs["n_draws"] = n_draws * n_chains
        df.attrs["_raw_results"] = raw_results

        return df

    def spatial_diagnostics_decision(self, alpha: float = 0.05) -> str:
        """Return a model-selection decision from Bayesian LM test results.

        Implements the decision tree from :cite:t:`koley2024UseNot`
        (the Bayesian analogue of the classical ``stge_kb`` procedure
        in :cite:t:`anselin1996SimpleDiagnostic`), adapted for panel models
        following :cite:t:`elhorst2014SpatialEconometrics`.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the Bayesian p-values.

        Returns
        -------
        str
            Recommended model name (e.g. ``"SARPanelFE"``, ``"SDMPanelFE"``).

        See Also
        --------
        spatial_diagnostics : Compute the Bayesian LM test statistics.

        References
        ----------
        :cite:t:`koley2024UseNot`, :cite:t:`anselin1996SimpleDiagnostic`,
        :cite:t:`elhorst2014SpatialEconometrics`
        """
        diag = self.spatial_diagnostics()

        def _sig(test_name: str) -> bool:
            if test_name not in diag.index:
                return False
            pval = diag.loc[test_name, "p_value"]
            return not np.isnan(pval) and pval < alpha

        model_type = self.__class__.__name__

        # Determine the "FE" or "RE" suffix for the recommended model
        suffix = "FE" if "FE" in model_type else "RE" if "RE" in model_type else ""

        # --- OLSPanelFE / OLSPanelRE decision tree ---
        if model_type.startswith("OLSPanel"):
            lag = _sig("Panel-LM-Lag")
            error = _sig("Panel-LM-Error")
            sdm_joint = _sig("Panel-LM-SDM-Joint")
            slx_err_joint = _sig("Panel-LM-SLX-Error-Joint")

            if sdm_joint or slx_err_joint:
                if lag and not error:
                    return f"SARPanel{suffix}"
                elif error and not lag:
                    return f"SEMPanel{suffix}"
                elif lag and error:
                    if (
                        diag.loc["Panel-LM-Lag", "p_value"]
                        <= diag.loc["Panel-LM-Error", "p_value"]
                    ):
                        return f"SARPanel{suffix}"
                    else:
                        return f"SEMPanel{suffix}"
                else:
                    return model_type
            else:
                if lag and not error:
                    return f"SARPanel{suffix}"
                elif error and not lag:
                    return f"SEMPanel{suffix}"
                elif lag and error:
                    if (
                        diag.loc["Panel-LM-Lag", "p_value"]
                        <= diag.loc["Panel-LM-Error", "p_value"]
                    ):
                        return f"SARPanel{suffix}"
                    else:
                        return f"SEMPanel{suffix}"
                else:
                    return model_type

        # --- SARPanelFE/RE decision tree ---
        elif model_type.startswith("SARPanel"):
            if _sig("Panel-LM-Error"):
                return f"SARARPanel{suffix}"
            elif _sig("Panel-Robust-LM-WX") or _sig("Panel-LM-WX"):
                return f"SDMPanel{suffix}"
            else:
                return model_type

        # --- SEMPanelFE/RE decision tree ---
        elif model_type.startswith("SEMPanel"):
            if _sig("Panel-LM-Lag"):
                return f"SARARPanel{suffix}"
            elif _sig("Panel-LM-WX"):
                return f"SDEMPanel{suffix}"
            else:
                return model_type

        # --- SLXPanelFE decision tree ---
        elif model_type.startswith("SLXPanel"):
            rlag = _sig("Panel-Robust-LM-Lag-SDM")
            rerr = _sig("Panel-Robust-LM-Error-SDEM")
            if rlag and rerr:
                return f"MANSARPanel{suffix}"
            elif rlag:
                return f"SDMPanel{suffix}"
            elif rerr:
                return f"SDEMPanel{suffix}"
            else:
                return model_type

        # --- SDMPanelFE decision tree ---
        elif model_type.startswith("SDMPanel"):
            if _sig("Panel-LM-Error"):
                return f"MANSARPanel{suffix}"
            else:
                return model_type

        # --- SDEMPanelFE decision tree ---
        elif model_type.startswith("SDEMPanel"):
            if _sig("Panel-LM-Lag"):
                return f"MANSARPanel{suffix}"
            else:
                return model_type

        else:
            return model_type

    def spatial_effects(
        self, return_posterior_samples: bool = False
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Compute Bayesian inference for direct, indirect, and total impacts.

        Computes impact measures for each posterior draw, then summarises
        the posterior distribution with means, 95% credible intervals, and
        Bayesian p-values.

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
        # Models with WX terms (SDM, SLX, SDEM) report effects only for
        # lagged covariates (k_wx columns), while models without WX terms
        # (SAR, SEM) report effects for non-intercept covariates.
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

        # Determine model type label
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

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(N={self._N}, T={self._T}, n={n}, "
            f"k={k}, model={self.model}, features={self._feature_names})"
        )
