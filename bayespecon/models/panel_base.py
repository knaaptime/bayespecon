"""Base classes and helpers for spatial panel models."""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.sparse as sp
from formulaic import model_matrix
from libpysal.graph import Graph

from .._backends import resolve_backend
from ..logdet import (
    LogdetBounds,
    LogDetMethodName,
    _auto_logdet_method,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
    resolve_logdet_bounds,
    resolve_logdet_method,
)
from ._common import _SpatialModelBase
from .base import _is_row_standardized_csr
from .priors import PanelOLSPriors, PriorsLike, priors_as_dict, resolve_priors


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


class SpatialPanelModel(_SpatialModelBase):
    """Base class for static spatial panel models with FE transforms.

    Holds the within-transformation, panel-aware sorting, and weights
    handling shared by all static fixed-effects panel model subclasses.
    Not instantiated directly.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode. Must contain
        the response, regressors, ``unit_col``, and ``time_col``.
    y : array-like, optional
        Stacked response of shape ``(N*T,)`` in unit-major order.
        Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix of shape ``(N*T, k)``. Required in matrix
        mode. DataFrame columns are preserved as feature names.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)`` (preferred — broadcast over
        time periods) or the full ``(N*T, N*T)`` block-diagonal panel
        matrix. Accepts a :class:`libpysal.graph.Graph` or any
        :class:`scipy.sparse` matrix. The legacy
        :class:`libpysal.weights.W` object is **not** accepted; pass
        ``w.sparse`` or ``libpysal.graph.Graph.from_W(w)``. Should be
        row-standardised; a :class:`UserWarning` is raised otherwise.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode for panel sorting and N/T inference.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable from ``W`` or the data shape.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE. The within transformation is
        applied to ``y`` and ``X`` before likelihood evaluation.
    priors : dict, optional
        Override default priors. Supported keys depend on the subclass;
        each subclass docstring lists its keys with defaults.
    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects from the cross-sectional ``N x N`` weights size:
        ``"eigenvalue"`` for ``N <= 2000`` else ``"chebyshev"``.
    robust : bool, default False
        If True, replace the Normal error with Student-t for robustness
        to heavy-tailed outliers. Adds a ``nu`` parameter with a
        ``TruncExp(lower=2)`` prior of rate ``nu_lam`` (default 1/30,
        mean ≈ 30). Override via ``priors={"nu_lam": value}``.
    w_vars : list of str, optional
        Names of X columns to spatially lag. Only relevant for
        subclasses that include ``WX`` terms (``SLXPanelFE``,
        ``SDMPanelFE``, ``SDEMPanelFE`` and their RE/dynamic
        analogues). By default all non-constant columns are lagged.
        Pass a subset, e.g. ``w_vars=["income", "density"]``.
    """

    # Emit a ResourceWarning before materializing very large dense panel
    # weight matrices. Tests may monkeypatch this value.
    _DENSE_W_WARN_BYTES: int = 100 * 1024 * 1024

    # Hooks consumed by `_SpatialModelBase.spatial_diagnostics` and
    # `spatial_diagnostics_decision` to route panel-flavored test names,
    # predicate keys, and decision-tree specs.
    _decision_test_prefix: str = "Panel-"
    _decision_predicate_prefix: str = "panel_"
    _decision_spec_attr: str = "get_panel_spec"
    _diagnostics_require_W: bool = False

    # Panel models wrap the full (N·T)×(N·T) Kronecker form for PyTensor,
    # not the N×N reduced weight matrix.
    _pt_sparse_attr: str = "_W_sparse_NT"

    #: The default :class:`PanelOLSPriors` accepts the shared
    #: ``beta_mu``/``beta_sigma``/``sigma_sigma`` keys. Subclasses with
    #: extra parameters (``rho``/``lam``/``phi``/random effects/censoring)
    #: override this with the matching :mod:`bayespecon.models.priors`
    #: dataclass so unknown keys raise ``TypeError`` at construction.
    _priors_cls: type = PanelOLSPriors

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
        priors: PriorsLike = None,
        logdet_method: LogDetMethodName | None = None,
        robust: bool = False,
        w_vars: Optional[list] = None,
        backend: str | None = None,
    ):
        if W is None:
            raise ValueError("W is required.")

        # Coerce ``priors`` to the subclass's typed dataclass; unknown keys
        # raise immediately. ``self.priors`` is kept as a plain dict view
        # so existing ``self.priors.get(...)`` calls in subclass
        # ``_build_pymc_model`` implementations continue to work.
        self.priors_obj = resolve_priors(priors, self._priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        self.backend = resolve_backend(backend)
        self.backend_name = self.backend.name
        # Validate ``logdet_method`` eagerly so invalid strings fail at
        # construction time rather than at first logdet access (which is
        # lazy via @cached_property).  ``resolve_logdet_method`` with
        # ``n=1`` only validates the string; auto-selection for ``None``
        # happens later once W is known.
        if logdet_method is not None:
            resolve_logdet_method(logdet_method, n=1)
        self.logdet_method = logdet_method
        self.model = int(model)
        self.robust = robust
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None

        if formula is not None:
            (
                y_arr,
                X_arr,
                self._feature_names,
                self._N,
                self._T,
                self._panel_index,
            ) = self._parse_panel_formula(formula, data, unit_col, time_col)
        elif y is not None and X is not None:
            y_arr, X_arr, self._feature_names = self._parse_panel_matrices(y, X)
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
        self._wx_feature_names = [
            self._feature_names[i] for i in self._wx_column_indices
        ]

        # Validate W and store as N×N CSR. Dense expansion, eigenvalues,
        # and logdet builders are all derived lazily via @cached_property
        # so init stays O(nnz) for chebyshev / sparse-grid methods that do
        # not require the eigendecomposition.
        self._W_sparse, self._is_row_std = _parse_panel_W(W, self._N, self._T)

        self._y, self._X = _demean_panel(
            self._y_raw, self._X_raw, self._N, self._T, self.model
        )

        # Resolve the logdet method up-front so the lazy property accessors
        # know whether eigenvalues are required.
        self._resolved_logdet_method = (
            self.logdet_method
            if self.logdet_method is not None
            else _auto_logdet_method(self._W_sparse.shape[0])
        )

    def _apply_spatial_lag(self, v: np.ndarray) -> np.ndarray:
        """Apply ``W_n ⊗ I_T`` to a stacked panel vector or matrix.

        Overrides the cross-sectional default to use the blockwise
        sparse multiply implemented by :meth:`_sparse_panel_lag`, which
        accepts either ``(N·T,)`` or ``(N·T, k)`` arrays and stays
        sparse internally.
        """
        return self._sparse_panel_lag(v)

    # ------------------------------------------------------------------
    # Input parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_panel_formula(
        formula: str,
        data: Optional[pd.DataFrame],
        unit_col: Optional[str],
        time_col: Optional[str],
    ):
        """Parse formula/data inputs into stacked panel arrays.

        Sorts ``data`` by ``(time_col, unit_col)`` to produce the
        time-major stacked layout the panel base assumes (so the
        ``T``-th block of ``N`` rows corresponds to time period ``T``).

        Returns
        -------
        tuple
            ``(y, X, feature_names, N, T, panel_index)`` where
            ``panel_index`` is the sorted ``[time_col, unit_col]``
            DataFrame.
        """
        if data is None:
            raise ValueError("data is required with formula mode.")
        if unit_col is None or time_col is None:
            raise ValueError("unit_col and time_col are required in formula mode.")

        d = data.sort_values([time_col, unit_col]).reset_index(drop=True)
        lhs, rhs = formula.split("~", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        X_mm = model_matrix(rhs, d)
        feature_names = list(X_mm.columns)
        y_arr = np.asarray(d[lhs], dtype=float)
        X_arr = np.asarray(X_mm, dtype=float)

        units = d[unit_col].nunique()
        times = d[time_col].nunique()
        if units * times != len(d):
            raise ValueError(
                "Data are not a balanced panel after sorting by time/unit."
            )
        panel_index = d[[time_col, unit_col]].copy()
        return y_arr, X_arr, feature_names, units, times, panel_index

    @staticmethod
    def _parse_panel_matrices(y, X):
        """Parse matrix-mode inputs and infer feature names.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Stacked dependent variable, design matrix, and feature names.
            The caller is responsible for validating ``N * T == len(y)``.
        """
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        return y_arr, X_arr, feature_names

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

    def _batch_sparse_lag(
        self,
        resid: np.ndarray,
        T_eff: int | None = None,
    ) -> np.ndarray:
        """Apply panel spatial lag to a batch of stacked residual draws.

        Parameters
        ----------
        resid : np.ndarray
            Residual draws with shape ``(n_draws, N*T_eff)``.
        T_eff : int, optional
            Effective time periods in the stacked residual layout. Defaults to
            ``self._T``. Dynamic panel paths may pass ``T-1``.

        Returns
        -------
        np.ndarray
            Spatially lagged residuals with the same shape as ``resid``.
        """
        R = np.asarray(resid, dtype=np.float64)
        if R.ndim != 2:
            raise ValueError(
                f"resid must be 2D (n_draws, N*T_eff), got shape {R.shape}."
            )

        N = int(self._N)
        Te = int(self._T if T_eff is None else T_eff)
        expected = N * Te
        if R.shape[1] != expected:
            raise ValueError(
                "resid second dimension must equal N*T_eff; "
                f"got {R.shape[1]} and expected {expected} (N={N}, T_eff={Te})."
            )

        W = self._W_sparse
        if W.shape[0] == N:
            # Reshape (draws, N*T_eff) -> (draws*T_eff, N), apply one sparse
            # matrix multiply, then reshape back.
            draws = R.shape[0]
            R_flat = R.reshape(draws * Te, N)
            WR_flat = np.asarray(W @ R_flat.T, dtype=np.float64).T
            return WR_flat.reshape(draws, Te * N)

        # Full panel matrix path (N*T x N*T) if supplied by caller.
        if W.shape[0] != expected:
            raise ValueError(
                f"W has shape {W.shape}; expected ({N},{N}) or ({expected},{expected}) "
                "for the provided N and T_eff."
            )
        return np.asarray(W @ R.T, dtype=np.float64).T

    @cached_property
    def _W_eigs(self) -> np.ndarray:
        """Eigenvalues of the N×N spatial weights matrix, computed lazily.

        Cached on first access to keep init O(nnz) when chebyshev / sparse-grid
        log-determinants are used (those methods do not need the full
        eigendecomposition).
        """
        return np.linalg.eigvals(self._W_sparse.toarray().astype(np.float64))

    @cached_property
    def _W_for_logdet(self):
        """Argument passed to ``make_logdet_fn`` — eigenvalues or sparse W.

        Computed lazily so that init never forces an eigendecomposition for
        chebyshev / sparse-grid methods.
        """
        if self._resolved_logdet_method == "eigenvalue":
            return self._W_eigs.real.astype(np.float64)
        return self._W_sparse

    @cached_property
    def _logdet_bounds(self) -> LogdetBounds:
        """Resolved rho/lambda bounds for the logdet builder and prior.

        Combines the user's ``priors`` (``rho_lower/upper`` or
        ``lam_lower/upper``) with method-aware defaults so that
        positive-only methods (``sparse_spline``, ``grid_mc``)
        automatically restrict the support to ``[1e-5, 1.0]`` unless the
        user supplies an explicit override.
        """
        eigs = (
            self._W_eigs.real.astype(np.float64)
            if self._resolved_logdet_method == "eigenvalue"
            and self._W_eigs is not None
            else None
        )
        return resolve_logdet_bounds(
            self.logdet_method,
            n=int(self._W_sparse.shape[0]),
            eigs=eigs,
            priors=self.priors,
        )

    @cached_property
    def _logdet_numpy_fn(self):
        """Pure-numpy ``(rho) -> float`` logdet evaluator (lazy)."""
        eigs = (
            self._W_eigs.real
            if self._resolved_logdet_method == "eigenvalue"
            else None
        )
        b = self._logdet_bounds
        return make_logdet_numpy_fn(
            self._W_sparse,
            eigs,
            method=self.logdet_method,
            rho_min=b.rho_min,
            rho_max=b.rho_max,
        )

    @cached_property
    def _logdet_numpy_vec_fn(self):
        """Vectorised pure-numpy logdet evaluator (lazy)."""
        eigs = (
            self._W_eigs.real
            if self._resolved_logdet_method == "eigenvalue"
            else None
        )
        b = self._logdet_bounds
        return make_logdet_numpy_vec_fn(
            self._W_sparse,
            eigs,
            method=self.logdet_method,
            rho_min=b.rho_min,
            rho_max=b.rho_max,
        )

    @cached_property
    def _logdet_pytensor_fn(self):
        """PyTensor logdet evaluator used inside ``_build_pymc_model`` (lazy)."""
        b = self._logdet_bounds
        return make_logdet_fn(
            self._W_for_logdet,
            method=self.logdet_method,
            rho_min=b.rho_min,
            rho_max=b.rho_max,
            T=self._T,
        )

    @cached_property
    def _W_dense(self) -> np.ndarray:
        """Dense (N*T)×(N*T) weight matrix, materialised lazily on first access."""
        # If W is N x N, dense panel matrix is (N*T) x (N*T); otherwise
        # caller supplied full panel matrix already.
        n_nt = (
            self._N * self._T
            if self._W_sparse.shape[0] == self._N
            else int(self._W_sparse.shape[0])
        )
        nbytes = n_nt * n_nt * 8
        if nbytes > int(self._DENSE_W_WARN_BYTES):
            warnings.warn(
                f"Materialising a dense panel weight matrix of size {n_nt}x{n_nt} "
                f"(~{nbytes / 1024**2:.0f} MB).",
                ResourceWarning,
                stacklevel=2,
            )
        return _as_dense_W(self._W_sparse, self._N, self._T)

    @cached_property
    def _W_sparse_NT(self) -> "sp.csr_matrix":
        """Sparse (N*T)×(N*T) Kronecker-block weight matrix ``I_T ⊗ W_n``.

        Cached on first access. Used by symbolic (PyMC/PyTensor) likelihoods
        to avoid the O((N*T)²) memory footprint of :attr:`_W_dense` while
        still exposing a single linear operator that can be applied to a
        stacked panel residual vector.
        """
        W = self._W_sparse
        if W.shape[0] == self._N:
            # Force ``csr_matrix`` (not ``csr_array``) so the result is
            # accepted by :mod:`pytensor.sparse`, which currently only
            # supports the legacy ``scipy.sparse`` matrix API.
            return sp.csr_matrix(
                sp.kron(sp.eye(self._T, format="csr"), W, format="csr")
            )
        # Caller already supplied a full (N*T)×(N*T) matrix.
        return sp.csr_matrix(W)

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(N={self._N}, T={self._T}, n={n}, "
            f"k={k}, model={self.model}, features={self._feature_names})"
        )
