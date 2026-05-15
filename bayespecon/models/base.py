"""Base class for Bayesian spatial regression models."""

from __future__ import annotations

import warnings
import weakref
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
    _auto_logdet_method,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)
from ._common import _SpatialModelBase
from .priors import OLSPriors, PriorsLike, priors_as_dict, resolve_priors

# Global eigenvalue cache keyed by ``id(graph)``.  We cannot use
# ``WeakKeyDictionary`` because :class:`libpysal.graph.Graph` is intentionally
# unhashable (different graphs with identical contents must still be treated as
# distinct keys).  Instead we register a :func:`weakref.finalize` callback on
# each Graph that drops the entry once the Graph is garbage-collected.
_EIG_CACHE: dict[int, np.ndarray] = {}


def _store_eigs(graph: Graph, eigs: np.ndarray) -> None:
    """Insert ``eigs`` into :data:`_EIG_CACHE` with a finalize-based eviction."""
    key = id(graph)
    if key in _EIG_CACHE:
        return
    _EIG_CACHE[key] = eigs
    try:
        weakref.finalize(graph, _EIG_CACHE.pop, key, None)
    except TypeError:
        # Graph not weakref-able for some reason; fall back to a leaky cache
        # entry rather than failing the model construction.
        pass


def _is_row_standardized_csr(W_csr: sp.csr_matrix) -> bool:
    """Return True when each row sum is numerically close to one."""
    row_sums = np.asarray(W_csr.sum(axis=1)).ravel()
    return bool(np.allclose(row_sums, 1.0, atol=1e-6))


def _parse_W(
    W: Union[Graph, sp.spmatrix],
    n: int,
) -> Graph:
    """Validate and normalise a spatial weights argument to a libpysal.graph.Graph.

    Parameters
    ----------
    W :
        Either a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse` matrix.
    n :
        Expected number of spatial units (must match both dimensions of W).

    Returns
    -------
    Graph
        Row-standardised libpysal.graph.Graph object.

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
        G = W
    elif sp.issparse(W):
        # Convert sparse matrix to Graph. Validate shape up-front so that
        # ``Graph.from_sparse`` only sees square inputs.
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be a square matrix, got shape {W.shape}.")
        G = Graph.from_sparse(W)
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

    W_csr = G.sparse.tocsr().astype(np.float64)
    if W_csr.ndim != 2 or W_csr.shape[0] != W_csr.shape[1]:
        raise ValueError(f"W must be a square matrix, got shape {W_csr.shape}.")
    if W_csr.shape[0] != n:
        raise ValueError(
            f"W has shape {W_csr.shape} but data has {n} observations. "
            "W must be an n\u00d7n matrix."
        )
    transform = getattr(G, "transformation", None)
    row_std = transform in ("r", "R") or _is_row_standardized_csr(W_csr)
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
    return G

def _pointwise_gaussian_loglik(
    eps: np.ndarray,
    sigma_draws: np.ndarray,
    nu_draws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute pointwise Gaussian or Student-t log-likelihood.

    Parameters
    ----------
    eps : np.ndarray
        Residual matrix of shape ``(n_draws, n_obs)``.
    sigma_draws : np.ndarray
        Posterior scale draws of shape ``(n_draws,)``.
    nu_draws : np.ndarray, optional
        Student-t degrees-of-freedom draws of shape ``(n_draws,)``.
        When ``None``, computes Gaussian log-likelihood.

    Returns
    -------
    np.ndarray
        Pointwise log-likelihood with shape ``(n_draws, n_obs)``.
    """
    eps = np.asarray(eps, dtype=np.float64)
    sigma = np.asarray(sigma_draws, dtype=np.float64).reshape(-1)
    if eps.ndim != 2:
        raise ValueError(f"eps must be 2D (n_draws, n_obs), got shape {eps.shape}.")
    if sigma.shape[0] != eps.shape[0]:
        raise ValueError(
            "sigma_draws length must equal eps first dimension; "
            f"got {sigma.shape[0]} and {eps.shape[0]}."
        )

    sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    sigma_2d = sigma[:, None]

    if nu_draws is None:
        return (
            -0.5 * (eps / sigma_2d) ** 2 - np.log(sigma_2d) - 0.5 * np.log(2.0 * np.pi)
        )

    nu = np.asarray(nu_draws, dtype=np.float64).reshape(-1)
    if nu.shape[0] != eps.shape[0]:
        raise ValueError(
            "nu_draws length must equal eps first dimension; "
            f"got {nu.shape[0]} and {eps.shape[0]}."
        )
    nu = np.maximum(nu, np.finfo(np.float64).tiny)
    from scipy import stats

    return stats.t.logpdf(eps, df=nu[:, None], loc=0.0, scale=sigma_2d)


def _write_log_likelihood_to_idata(
    idata: az.InferenceData,
    ll_array: np.ndarray,
) -> None:
    """Write a complete pointwise log-likelihood array to InferenceData.

    Parameters
    ----------
    idata : az.InferenceData
        Target inference data object to mutate in place.
    ll_array : np.ndarray
        Array with shape ``(chain, draw, obs)``.
    """
    import xarray as xr

    ll = np.asarray(ll_array, dtype=np.float64)
    if ll.ndim != 3:
        raise ValueError(
            f"ll_array must be 3D (chain, draw, obs), got shape {ll.shape}."
        )

    ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
    idata["log_likelihood"] = xr.Dataset({"obs": ll_da})


class SpatialModel(_SpatialModelBase):
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
        How to compute ``log|I - rho*W|``. ``"eigenvalue"`` (default for
        ``n <= 2000``) pre-computes W's eigenvalues once and evaluates
        O(n) per step; ``"exact"`` uses symbolic pytensor det (slow for
        ``n > 500``); ``"grid_dense"`` uses dense eigenvalue grid +
        cubic-spline interpolation (MATLAB-style ``lndetfull`` for dense
        W); ``"grid_sparse"`` uses sparse-LU grid + cubic-spline
        interpolation (``lndetfull`` style for large sparse W);
        ``"sparse_spline"`` uses sparse-LU + spline on ``[max(rho_min, 0),
        rho_max]`` (``lndetint`` style); ``"grid_mc"`` uses Monte Carlo
        trace approximation (``lndetmc``); ``"grid_ilu"`` uses ILU-based
        approximation (``lndetichol`` analog); ``"chebyshev"`` (default
        for ``n > 2000``) uses a Chebyshev polynomial approximation
        evaluated via Clenshaw's algorithm.
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

    #: Subclasses override this to declare their typed priors dataclass.
    #: The default :class:`OLSPriors` accepts the shared
    #: ``beta_mu / beta_sigma / sigma_sigma / nu_lam`` keys used by every
    #: cross-sectional Gaussian model.
    _priors_cls: type = OLSPriors

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        priors: PriorsLike = None,
        logdet_method: str | None = None,
        robust: bool = False,
        w_vars: Optional[list] = None,
        backend: str | None = None,
    ):
        # Coerce ``priors`` to the subclass's typed dataclass; unknown keys
        # raise immediately.  ``self.priors`` is kept as a plain dict view
        # so existing ``self.priors.get(...)`` calls in subclass
        # ``_build_pymc_model`` implementations continue to work.
        self.priors_obj = resolve_priors(priors, self._priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        # Resolve the probabilistic-programming backend up-front so invalid
        # names fail at construction time rather than during ``fit()``.  The
        # backend object itself is currently advisory; the existing PyMC fit
        # paths still call the helpers in ``_sampler`` directly.
        self.backend = resolve_backend(backend)
        self.backend_name = self.backend.name
        self.logdet_method = logdet_method
        self.robust = robust
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None

        if formula is not None:
            if data is None:
                raise ValueError("data must be provided when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
        elif y is not None and X is not None:
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
        else:
            raise ValueError("Provide either (formula, data) or (y, X).")

        if W is not None:
            # Validate W and store as a libpysal.graph.Graph.  All matrix
            # representations (_W_sparse, _W_dense, _W_eigs, _Wy, _WX) are
            # derived lazily via @cached_property on first access.
            self._graph: Optional[Graph] = _parse_W(W, len(self._y))
            transform = getattr(self._graph, "transformation", None)
            self._is_row_std = transform in ("r", "R") or _is_row_standardized_csr(
                self._graph.sparse.tocsr().astype(np.float64)
            )
            self._wx_column_indices = self._spatial_lag_column_indices(
                self._X, self._feature_names
            )
            if w_vars is not None:
                unknown = [v for v in w_vars if v not in self._feature_names]
                if unknown:
                    raise ValueError(
                        f"w_vars contains names not found in X columns: {unknown}. "
                        f"Available: {self._feature_names}"
                    )
                self._wx_column_indices = [
                    i
                    for i in self._wx_column_indices
                    if self._feature_names[i] in w_vars
                ]
            self._wx_feature_names = [
                self._feature_names[i] for i in self._wx_column_indices
            ]
            # Resolve the logdet method up-front so the lazy property
            # accessors know whether eigenvalues are required.  The numpy
            # / pytensor logdet callables themselves and their W argument
            # are deferred to ``@cached_property`` so init never forces an
            # eigendecomposition for chebyshev / sparse-grid methods that
            # do not require it.
            self._resolved_logdet_method = (
                self.logdet_method
                if self.logdet_method is not None
                else _auto_logdet_method(self._W_sparse.shape[0])
            )
        else:
            # W-free mode: no spatial structure; spec tests require W to be supplied.
            self._graph = None
            self._is_row_std = False
            self._wx_column_indices: list[int] = []
            self._wx_feature_names: list[str] = []
            if w_vars is not None:
                raise ValueError("w_vars requires a spatial weights matrix W.")

    # -----------------------------------------------------------------
    # Lazy matrix representations of self._graph.  All keyed off the
    # underlying Graph object; eigenvalues are shared across model
    # instances that point at the same Graph via the module-level
    # _EIG_CACHE WeakKeyDictionary.
    # -----------------------------------------------------------------

    @cached_property
    def _W_sparse(self) -> Optional[sp.csr_matrix]:
        """CSR sparse representation of the row-standardised W."""
        if self._graph is None:
            return None
        return self._graph.sparse.tocsr().astype(np.float64)

    @cached_property
    def _W_dense(self) -> Optional[np.ndarray]:
        """Dense weight matrix, materialised lazily on first access."""
        if self._graph is None:
            return None
        return np.asarray(self._W_sparse.toarray(), dtype=np.float64)

    @cached_property
    def _W_eigs(self) -> Optional[np.ndarray]:
        """Eigenvalues of W, cached per Graph in :data:`_EIG_CACHE`."""
        if self._graph is None:
            return None
        cached = _EIG_CACHE.get(id(self._graph))
        if cached is not None:
            return cached
        eigs = np.linalg.eigvals(self._W_dense)
        _store_eigs(self._graph, eigs)
        return eigs

    # ------------------------------------------------------------------
    # Lazy log-determinant evaluators.
    #
    # Built on first access so that constructing a model with a
    # chebyshev or sparse-grid logdet method never forces the O(n³)
    # eigendecomposition that the eigenvalue method needs.  Mirrors the
    # caching style used by ``SpatialPanelModel``.
    # ------------------------------------------------------------------

    @cached_property
    def _W_for_logdet(self):
        """Argument passed to :func:`make_logdet_fn` — eigenvalues or dense W."""
        if self._resolved_logdet_method in ("eigenvalue", "chebyshev"):
            return self._W_eigs.real.astype(np.float64)
        return self._W_dense

    @cached_property
    def _logdet_numpy_fn(self):
        """Pure-numpy ``(rho) -> float`` logdet evaluator (lazy)."""
        eigs = (
            self._W_eigs.real
            if self._resolved_logdet_method == "eigenvalue"
            else None
        )
        return make_logdet_numpy_fn(self._W_sparse, eigs, method=self.logdet_method)

    @cached_property
    def _logdet_numpy_vec_fn(self):
        """Vectorised pure-numpy logdet evaluator (lazy)."""
        eigs = (
            self._W_eigs.real
            if self._resolved_logdet_method == "eigenvalue"
            else None
        )
        return make_logdet_numpy_vec_fn(
            self._W_sparse, eigs, method=self.logdet_method
        )

    @cached_property
    def _logdet_pytensor_fn(self):
        """PyTensor logdet evaluator used inside ``_build_pymc_model`` (lazy)."""
        return make_logdet_fn(self._W_for_logdet, method=self.logdet_method)

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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_W(self):
        """Raise if no spatial weights matrix was supplied."""
        if self._W_sparse is None:
            raise ValueError(
                "This method requires a spatial weights matrix W. "
                "Pass W when constructing the model."
            )

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(n={n}, k={k}, features={self._feature_names})"
        )
