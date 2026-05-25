"""Base class for Bayesian spatial regression models."""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.sparse as sp
from formulaic import model_matrix
from libpysal.graph import Graph

if TYPE_CHECKING:
    from .._backends import ProbabilisticBackend

from .._backends.sampler_helpers import (
    prepare_compile_kwargs,
    prepare_idata_kwargs,
    use_jax_likelihood,
)
from .._logdet import (
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
    resolve_logdet_bounds,
)
from .._logdet._config import _auto_logdet_method


def gelman_default_beta_prior(
    y: np.ndarray,
    design: np.ndarray,
    feature_names: list[str],
    scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Weakly-informative default prior on regression coefficients.

    Follows Gelman, Jakulin, Pittau & Su (2008) by setting per-column
    prior scales from ``sd(y)`` and ``sd(x_j)``.  For each column ``j``
    of ``design``:

    * **Intercept-like** (named ``"intercept"`` or numerically constant):
      ``mu_j = mean(y)``, ``sigma_j = scale * sd(y)``.
    * **Slope**:
      ``mu_j = 0``, ``sigma_j = scale * sd(y) / sd(x_j)``.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    design : ndarray, shape (n, p)
        Effective design matrix used by ``beta`` in the model
        (i.e. ``X`` for SAR/SEM/OLS; ``[X, WX]`` for SDM/SDEM/SLX).
    feature_names : list[str]
        Column labels aligned with ``design``.  Used to detect
        intercept-like columns named ``"intercept"``.
    scale : float, default 2.5
        Multiplier on the standardised prior scale.

    Returns
    -------
    beta_mu : ndarray, shape (p,)
    beta_sigma : ndarray, shape (p,)

    References
    ----------
    Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008).
    *A weakly informative default prior distribution for logistic and
    other regression models.* Annals of Applied Statistics, 2(4),
    1360-1383.
    """
    sd_y = float(np.std(y))
    if sd_y <= 0.0:
        sd_y = 1.0
    mean_y = float(np.mean(y))
    p = design.shape[1]
    beta_mu = np.zeros(p, dtype=np.float64)
    beta_sigma = np.empty(p, dtype=np.float64)
    for j in range(p):
        col = design[:, j]
        name = feature_names[j] if j < len(feature_names) else ""
        is_named_intercept = name.lower() == "intercept"
        is_constant = np.allclose(col, col[0])
        if is_named_intercept or is_constant:
            beta_mu[j] = mean_y
            beta_sigma[j] = scale * sd_y
        else:
            sd_col = float(np.std(col))
            beta_sigma[j] = scale * sd_y / sd_col if sd_col > 0.0 else scale * sd_y
    return beta_mu, beta_sigma


def _is_row_standardized_csr(W_csr: sp.csr_matrix) -> bool:
    """Return True when each row sum is numerically close to one."""
    row_sums = np.asarray(W_csr.sum(axis=1)).ravel()
    return bool(np.allclose(row_sums, 1.0, atol=1e-6))


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
        row_std = transform in ("r", "R") or _is_row_standardized_csr(W_csr)
    elif sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
        row_std = _is_row_standardized_csr(W_csr)
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
        raise ValueError(f"W must be a square matrix, got shape {W_csr.shape}.")
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
        How to compute ``log|I - rho*W|``. ``"eigenvalue"`` (default for
        ``n <= 500``) pre-computes W's eigenvalues once and evaluates
        O(n) per step; ``"exact"`` uses symbolic pytensor det (slow for
        ``n > 500``); ``"grid_dense"`` uses dense eigenvalue grid +
        cubic-spline interpolation (MATLAB-style ``lndetfull`` for dense
        W); ``"grid_sparse"`` uses sparse-LU grid + cubic-spline
        interpolation (``lndetfull`` style for large sparse W);
        ``"sparse_spline"`` uses sparse-LU + spline on
        ``[max(rho_min, 0), rho_max]`` (``lndetint`` style); ``"grid_mc"``
        uses Monte Carlo trace approximation (``lndetmc``); ``"grid_ilu"``
        uses ILU-based approximation (``lndetichol`` analog);
        ``"chebyshev"`` (default for ``n > 500``) uses a Chebyshev
        polynomial approximation evaluated via Clenshaw's algorithm.
        For large ``n`` the Chebyshev coefficients are built from a
        stochastic trace estimator selected by ``trace_estimator``.
    trace_estimator : {"hutchinson", "hutchpp", "xtrace"}, default "hutchpp"
        Stochastic trace estimator used to build the Chebyshev
        coefficients when an eigendecomposition is unavailable.  Ignored
        for non-Chebyshev methods.  See
        ``docs/source/user-guide/logdet_profiling.ipynb`` for the
        cost/accuracy frontier.
    trace_k : int, optional
        Number of probe vectors for the trace estimator.  Defaults:
        ``30`` (hutchinson), ``50`` (hutchpp), ``25`` (xtrace).
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

    Attributes
    ----------
    _spatial_params : tuple[str, ...]
        Spatial autoregressive parameters in the model (e.g. ``("rho",)``
        for SAR, ``("lam",)`` for SEM).  Empty for OLS and SLX.
    _lag_terms : tuple[str, ...]
        Lagged terms present in the model specification (e.g. ``("Wy",)``
        for SAR, ``("WX",)`` for SLX, ``("Wy", "WX")`` for SDM).
    _jacobian_param : str or None
        Name of the parameter that appears in the Jacobian determinant
        ``log|I - param * W|``.  ``"rho"`` for SAR/SDM, ``"lam"`` for
        SEM/SDEM, ``None`` for OLS/SLX (no Jacobian).
    _has_wx_in_beta : bool
        Whether the ``beta`` coefficient vector includes WX coefficients
        (i.e. whether the design matrix is ``[X, WX]`` rather than just
        ``X``).  True for SLX, SDM, SDEM.
    _gibbs_class : str or None
        Fully-qualified class name of the Gibbs sampler for this model
        (e.g. ``"GaussianSARGibbs"``), or ``None`` if no Gibbs sampler
        exists.  Used to look up the sampler at runtime to avoid circular
        imports.
    _model_type : str
        Short lowercase model name used as the ``model_type`` argument to
        the Gibbs sampler (e.g. ``"sar"``, ``"sdm"``).  Also used for
        InferenceData coordinate labels.
    """

    # --- Declarative model metadata ----------------------------------------
    # Subclasses override these to declare their spatial structure.
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = ""

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        priors: Optional[Union[dict, Any]] = None,
        logdet_method: str | None = None,
        robust: bool = False,
        w_vars: Optional[list] = None,
        backend: Optional[Union[str, "ProbabilisticBackend"]] = None,
        trace_estimator: str = "hutchpp",
        trace_k: int | None = None,
    ):
        # Resolve typed priors (dataclass) and dict view.
        from .priors import BasePriors, priors_as_dict, resolve_priors

        _priors_cls = getattr(self.__class__, "_priors_cls", BasePriors)
        self.priors_obj = resolve_priors(priors, _priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        self.logdet_method = logdet_method
        self.trace_estimator = trace_estimator
        self.trace_k = trace_k
        self.robust = robust

        # Resolve probabilistic backend (PyMC, NumPyro, BlackJAX, nutpie).
        from .._backends import resolve_backend

        self.backend = resolve_backend(backend)
        self.backend_name = self.backend.name

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
            # Validate W and store as CSR sparse matrix.
            # Dense conversion is deferred to _W_dense (lazy property).
            self._W_sparse, self._is_row_std = _parse_W(W, len(self._y))
            # Eigenvalues are computed lazily via the _W_eigs cached property
            # to avoid the O(n³) eigendecomposition for large n where trace
            # or Chebyshev methods are used instead.
            # Resolve the logdet method up-front so the lazy property
            # accessors know whether eigenvalues are required.
            self._resolved_logdet_method = (
                self.logdet_method
                if self.logdet_method is not None
                else _auto_logdet_method(self._W_sparse.shape[0])
            )
            # Resolve rho/lambda bounds from method and priors.
            # For row-standardised W the spectral stability interval is
            # always approximately (-1, 1), so no eigenvalue computation
            # is needed here.
            self._logdet_bounds = resolve_logdet_bounds(
                self.logdet_method,
                n=len(self._y),
                priors=self.priors,
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
            # Logdet builders are constructed lazily on first access — see
            # the _logdet_numpy_fn, _logdet_numpy_vec_fn and
            # _logdet_pytensor_fn properties.  Caches are seeded as None so
            # that init never triggers the underlying eigendecomposition for
            # chebyshev / trace methods.
            self._logdet_numpy_fn_cache = None
            self._logdet_numpy_vec_fn_cache = None
            self._logdet_pytensor_fn_cache = None
            self._W_for_logdet_cache = None
            self._Wy: np.ndarray = np.asarray(
                self._W_sparse @ self._y, dtype=np.float64
            )
            if self._wx_column_indices:
                self._WX = np.asarray(
                    self._W_sparse @ self._X[:, self._wx_column_indices],
                    dtype=np.float64,
                )
            else:
                self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
        else:
            # W-free mode: no spatial structure; spec tests require W to be supplied.
            self._W_sparse = None
            self._is_row_std = False
            self._wx_column_indices: list[int] = []
            self._wx_feature_names: list[str] = []
            self._Wy = np.zeros(len(self._y), dtype=np.float64)
            self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
            if w_vars is not None:
                raise ValueError("w_vars requires a spatial weights matrix W.")

    @cached_property
    def _W_dense(self) -> np.ndarray:
        """Dense weight matrix, materialised lazily on first access."""
        return np.asarray(self._W_sparse.toarray(), dtype=np.float64)

    @cached_property
    def _W_pt_sparse(self):
        """PyTensor sparse variable wrapping :attr:`_W_sparse`.

        Cached so repeated PyMC model builds reuse the same symbolic sparse
        operator and avoid the ``O(n²)`` dense materialisation that
        ``pt.as_tensor_variable(self._W_dense)`` performs each time.

        Use with :func:`pytensor.sparse.structured_dot` (vector inputs must
        first be reshaped to ``(n, 1)`` because the vector overload's
        backward pass is broken in PyTensor).
        """
        import scipy.sparse as _sp
        from pytensor import sparse as _pts

        return _pts.as_sparse_variable(_sp.csc_matrix(self._W_sparse))

    @cached_property
    def _T_ww(self) -> float:
        """Trace of W'W + W², cached on first access.

        Computed as ``||W||_F² + sum(W * W')`` using sparse operations,
        which is O(nnz) rather than O(n²).
        """
        from ..graph import sparse_trace_WtW_plus_WW

        return sparse_trace_WtW_plus_WW(self._W_sparse)

    @cached_property
    def _W_eigs(self) -> np.ndarray | None:
        """Eigenvalues of W (complex), computed lazily on first access.

        For large n this is O(n³), so it is only computed when needed
        (e.g. by the eigenvalue logdet method).  Trace and Chebyshev
        methods never trigger this computation.
        """
        if self._W_sparse is None:
            return None
        return np.linalg.eigvals(self._W_sparse.toarray().astype(np.float64))

    @cached_property
    def _W_eigs_real(self) -> np.ndarray | None:
        """Real part of W eigenvalues as float64, computed lazily once."""
        eigs = self._W_eigs
        return None if eigs is None else eigs.real.astype(np.float64)

    @property
    def _W_for_logdet(self):
        """Argument passed to ``make_logdet_fn`` — eigenvalues or dense W.

        Computed lazily so that init never forces an eigendecomposition for
        chebyshev / trace methods.
        """
        if self._W_for_logdet_cache is None:
            if self._resolved_logdet_method == "eigenvalue":
                self._W_for_logdet_cache = self._W_eigs_real
            else:
                self._W_for_logdet_cache = self._W_sparse.toarray().astype(np.float64)
        return self._W_for_logdet_cache

    @property
    def _logdet_numpy_fn(self):
        """Pure-numpy ``(rho) -> float`` logdet evaluator (lazy)."""
        if self._logdet_numpy_fn_cache is None:
            eigs = (
                self._W_eigs_real
                if self._resolved_logdet_method == "eigenvalue"
                else None
            )
            self._logdet_numpy_fn_cache = make_logdet_numpy_fn(
                self._W_sparse,
                eigs,
                method=self.logdet_method,
                trace_estimator=self.trace_estimator,
                trace_k=self.trace_k,
            )
        return self._logdet_numpy_fn_cache

    @property
    def _logdet_numpy_vec_fn(self):
        """Vectorised pure-numpy logdet evaluator (lazy)."""
        if self._logdet_numpy_vec_fn_cache is None:
            eigs = (
                self._W_eigs_real
                if self._resolved_logdet_method == "eigenvalue"
                else None
            )
            self._logdet_numpy_vec_fn_cache = make_logdet_numpy_vec_fn(
                self._W_sparse,
                eigs,
                method=self.logdet_method,
                trace_estimator=self.trace_estimator,
                trace_k=self.trace_k,
            )
        return self._logdet_numpy_vec_fn_cache

    @property
    def _logdet_pytensor_fn(self):
        """PyTensor logdet evaluator used inside ``_build_pymc_model`` (lazy)."""
        if self._logdet_pytensor_fn_cache is None:
            self._logdet_pytensor_fn_cache = make_logdet_fn(
                self._W_for_logdet,
                method=self.logdet_method,
                rho_min=self._logdet_bounds.rho_min,
                rho_max=self._logdet_bounds.rho_max,
                trace_estimator=self.trace_estimator,
                trace_k=self.trace_k,
            )
        return self._logdet_pytensor_fn_cache

    @cached_property
    def _W_eigendecomposition(self):
        """Full eigendecomposition W = V diag(λ) V⁻¹ with complex128 arithmetic.

        Returns a 3-tuple ``(eigs, V, Vinv)`` of complex128 arrays, or
        ``None`` when no spatial weights matrix was supplied.

        Eigenvalues are sorted by real part (descending) for numerical
        stability.  Row-standardised W is generally non-symmetric, so V
        and Vinv are complex; taking ``.real`` prematurely drops imaginary
        parts and produces wrong results for spatial effects.
        """
        if self._W_sparse is None:
            return None
        W_dense = self._W_dense
        eigs, V = np.linalg.eig(W_dense)
        Vinv = np.linalg.inv(V)
        idx = np.argsort(eigs.real)[::-1]
        eigs_sorted = eigs[idx].astype(np.complex128)
        V_sorted = V[:, idx].astype(np.complex128)
        Vinv_sorted = Vinv[idx, :].astype(np.complex128)
        return (eigs_sorted, V_sorted, Vinv_sorted)

    @cached_property
    def _eig_inv_ones(self) -> Optional[np.ndarray]:
        """V⁻¹ @ 1ₙ (complex128), cached once.

        Used by :meth:`_batch_mean_row_sum` and
        :meth:`_batch_mean_row_sum_MW` for the eigenvalue-based
        computation of mean row sums.
        """
        decomp = self._W_eigendecomposition
        if decomp is None:
            return None
        return decomp[2] @ np.ones(decomp[0].shape[0], dtype=np.complex128)

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

        # Eigenvalue-based computation using shared eigendecomposition cache.
        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        c = self._eig_inv_ones  # complex128, (n,)
        eigs = decomp[0]  # complex128, (n,)
        V_col_sums = decomp[1].sum(axis=0)  # complex128, (n,)
        from ..diagnostics.spatial_effects import _chunked_eig_means

        return _chunked_eig_means(
            rho_draws,
            eigs.real.astype(np.float64),
            weights=(V_col_sums * c).real.astype(np.float64),
        )

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

        # Eigenvalue-based computation using shared eigendecomposition cache.
        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        c = self._eig_inv_ones  # complex128, (n,)
        eigs = decomp[0]  # complex128, (n,)
        V_col_sums = decomp[1].sum(axis=0)  # complex128, (n,)
        from ..diagnostics.spatial_effects import _chunked_eig_means

        return _chunked_eig_means(
            rho_draws,
            eigs.real.astype(np.float64),
            weights=(eigs * V_col_sums * c).real.astype(np.float64),
        )

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
    def _spatial_lag_column_indices(
        X: np.ndarray, feature_names: list[str]
    ) -> list[int]:
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

    def _gelman_default_beta_prior(
        self,
        design: np.ndarray,
        feature_names: list[str],
        scale: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Weakly-informative default prior on regression coefficients.

        Thin wrapper around :func:`gelman_default_beta_prior` that uses
        ``self._y`` as the response. See that function for details.
        """
        return gelman_default_beta_prior(self._y, design, feature_names, scale=scale)

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
        progressbar: bool = True,
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
        progressbar : bool, default True
            Show progress bar during sampling.
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
        idata_kwargs = sample_kwargs.pop("idata_kwargs", None)
        self._fit_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
            idata_kwargs=idata_kwargs,
            compute_log_likelihood=False,
            sample_kwargs=sample_kwargs,
        )
        return self._idata

    def _fit_gibbs_dispatch(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        thin: int,
        n_jobs: int,
        progressbar: bool,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        """Dispatch a ``fit(..., sampler='gibbs')`` call to :meth:`_fit_gibbs`.

        This keeps model ``fit`` methods thin by centralizing how Gibbs-specific
        kwargs are popped from ``sample_kwargs``.
        """
        sample_kwargs = dict(sample_kwargs or {})
        return self._fit_gibbs(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            n_jobs=n_jobs,
            progressbar=progressbar,
            gibbs_method=sample_kwargs.pop("gibbs_method", "numpy"),
            mala_step_size=sample_kwargs.pop("mala_step_size", 0.05),
            use_mala=sample_kwargs.pop("use_mala", True),
            use_slice=sample_kwargs.pop("use_slice", True),
            slice_width=sample_kwargs.pop("slice_width", None),
            chain_method=sample_kwargs.pop("chain_method", None),
        )

    def _fit_nuts(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        target_accept: float,
        random_seed: Optional[int],
        progressbar: bool,
        nuts_sampler: str = "pymc",
        idata_kwargs: dict[str, Any] | None = None,
        compute_log_likelihood: bool = False,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> tuple[az.InferenceData, bool]:
        """Shared NUTS sampling path used by model-specific ``fit`` methods.

        Returns
        -------
        tuple[arviz.InferenceData, bool]
            ``(idata, compute_log_likelihood)`` where the boolean reflects the
            post-policy value after :func:`prepare_idata_kwargs`.
        """
        sample_kwargs = dict(sample_kwargs or {})
        idata_kwargs = dict(idata_kwargs or {})

        build_kwargs: dict[str, Any] = {}
        build_sig = inspect.signature(self._build_pymc_model)
        if "compute_log_likelihood" in build_sig.parameters:
            build_kwargs["compute_log_likelihood"] = compute_log_likelihood
        if "nuts_sampler" in build_sig.parameters:
            build_kwargs["nuts_sampler"] = nuts_sampler

        model = self._build_pymc_model(**build_kwargs)
        self._pymc_model = model

        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
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
                progressbar=progressbar,
                **sample_kwargs,
            )
        return self._idata, compute_log_likelihood

    def _reconstruct_cross_sectional_log_likelihood(
        self,
        *,
        nuts_sampler: str,
    ) -> None:
        """Rebuild complete pointwise log-likelihood for cross-sectional models.

        Dispatches by spatial term type (``rho`` vs ``lam``) and whether the
        model's beta vector includes WX terms.
        """
        if not hasattr(self, "_idata"):
            return

        spatial_param = self._jacobian_param
        if spatial_param not in {"rho", "lam"}:
            return

        # SEM/SDEM on JAX backends build an observed CustomDist and already
        # have complete log_likelihood from PyMC.
        if spatial_param == "lam" and use_jax_likelihood(nuts_sampler):
            return

        idata = self._idata
        n = int(self._y.shape[0])
        Z = np.hstack([self._X, self._WX]) if self._has_wx_in_beta else self._X

        spatial_draws = idata.posterior[spatial_param].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(-1, Z.shape[1])
        sigma_draws = idata.posterior["sigma"].values.reshape(-1)
        nu_draws = idata.posterior["nu"].values.reshape(-1) if self.robust else None

        if spatial_param == "rho":
            mu = spatial_draws[:, None] * self._Wy[None, :] + (beta_draws @ Z.T)
            eps = self._y[None, :] - mu
        else:
            resid = self._y[None, :] - (beta_draws @ Z.T)
            W_resid = (self._W_sparse @ resid.T).T
            eps = resid - spatial_draws[:, None] * W_resid

        ll_data = _pointwise_gaussian_loglik(eps, sigma_draws, nu_draws)
        jacobian = self._logdet_numpy_vec_fn(spatial_draws)
        ll_total = ll_data + jacobian[:, None] / n

        n_chains = idata.posterior.sizes["chain"]
        n_draws_per_chain = idata.posterior.sizes["draw"]
        _write_log_likelihood_to_idata(
            idata,
            ll_total.reshape(n_chains, n_draws_per_chain, n),
        )

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        gibbs_method: str = "numpy",
        mala_step_size: float = 0.05,
        use_mala: bool = True,
        use_slice: bool = True,
        slice_width: float | None = None,
        chain_method: str | None = None,
    ) -> az.InferenceData:
        """Sample posterior via 3-block Gaussian Gibbs.

        Uses the model's :attr:`_gibbs_class` attribute to resolve the
        appropriate Gibbs sampler class at runtime.  Only models with
        ``_gibbs_class is not None`` support Gibbs sampling; calling this
        method on OLS or SLX raises :exc:`NotImplementedError`.

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup (burn-in) draws per chain.
        chains : int, default 4
            Number of independent chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup.
        n_jobs : int, default -1
            Number of parallel workers for the NumPy path.  ``-1`` uses
            all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "numpy"
            Execution backend: ``"numpy"`` for Python-loop Gibbs with
            adaptive slice sampling, or ``"jax"`` for full-JIT Gibbs
            with MALA for ρ/λ.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
        use_mala : bool, default True
            If True, use MALA for the ρ/λ update in the JAX path.
        use_slice : bool, default True
            If True, use slice sampling for the ρ/λ update.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.

        Returns
        -------
        arviz.InferenceData
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.

        Raises
        ------
        NotImplementedError
            If the model has no Gibbs sampler (``_gibbs_class is None``)
            or uses a robust (Student-t) likelihood.
        """
        if self._gibbs_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support Gibbs sampling. "
                f"Use sampler='nuts' (the default)."
            )
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        # --- Resolve Gibbs class (lazy import to avoid circular deps) ---
        import importlib

        from ..samplers.gaussian import GaussianGibbsPriors

        gibbs_module = importlib.import_module(
            "..samplers.gaussian", package=__package__
        )
        GibbsClass = getattr(gibbs_module, self._gibbs_class)

        # --- Build design matrix and feature names ---
        if self._has_wx_in_beta:
            Z = np.hstack([self._X, self._WX])  # (n, 2k)
            feature_names = list(self._feature_names) + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        else:
            Z = self._X
            feature_names = list(self._feature_names)

        # --- Build priors ---
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            Z, feature_names
        )
        priors = GaussianGibbsPriors(
            beta_mu=self.priors.get("beta_mu", default_beta_mu),
            beta_sigma=self.priors.get("beta_sigma", default_beta_sigma),
            sigma2_alpha=self.priors.get("sigma2_alpha", 2.0),
            sigma2_beta=self.priors.get("sigma2_beta", float(np.var(self._y))),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        # --- Build Gibbs sampler kwargs ---
        gibbs_kwargs: dict[str, Any] = dict(
            y=self._y,
            X=Z,
            W_sparse=self._W_sparse,
            priors=priors,
            logdet_fn=self._logdet_numpy_fn,
            logdet_vec_fn=self._logdet_numpy_vec_fn,
            feature_names=feature_names,
            model_type=self._model_type,
            W_eigs=self._W_eigs_real
            if self._resolved_logdet_method == "eigenvalue"
            else None,
            logdet_method=self.logdet_method,
        )
        # SAR/SDM need Wy; SEM/SDEM do not
        if self._jacobian_param == "rho":
            gibbs_kwargs["Wy"] = self._Wy

        gibbs = GibbsClass(**gibbs_kwargs)

        self._idata = gibbs.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            n_jobs=n_jobs,
            progressbar=progressbar,
            gibbs_method=gibbs_method,
            mala_step_size=mala_step_size,
            use_mala=use_mala,
            use_slice=use_slice,
            slice_width=slice_width,
            chain_method=chain_method,
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

        For Gibbs-fitted models the PyMC model is not constructed during
        sampling; it is built lazily on first access so that downstream
        consumers (e.g. bridge sampling for marginal likelihoods) can
        evaluate ``logp`` and the prior under the same model definition
        used by the NUTS path.

        Returns
        -------
        pymc.Model or None
            The model object used by :meth:`fit`, or ``None`` if the instance
            has not been fit yet.
        """
        if self._pymc_model is None and self._idata is not None:
            try:
                self._pymc_model = self._build_pymc_model()
            except TypeError:
                self._pymc_model = self._build_pymc_model(nuts_sampler="pymc")
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

    @staticmethod
    def _run_lm_diagnostics(model, tests: list[tuple]) -> pd.DataFrame:
        """Execute a registry of LM tests and return a tidy DataFrame.

        Shared helper used by :meth:`SpatialModel.spatial_diagnostics`,
        :meth:`SpatialPanelModel.spatial_diagnostics`,
        :meth:`FlowModel.spatial_diagnostics`, and
        :meth:`FlowPanelModel.spatial_diagnostics`.
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

        Looks up the diagnostic suite registered for this model class
        and calls each test function on this fitted model, collecting the
        results into a tidy DataFrame.  The set of tests depends on the
        model type — for example, an OLS model runs LM-Lag, LM-Error,
        LM-SDM-Joint, and LM-SLX-Error-Joint, while an SAR model runs
        LM-Error, LM-WX, and Robust-LM-WX.

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
        ValueError
            If no spatial weights matrix ``W`` was supplied.

        See Also
        --------
        spatial_diagnostics_decision : Model-selection decision based on
            the test results.
        spatial_effects : Posterior inference for direct/indirect/total
            impacts.

        Examples
        --------
        >>> ols = OLS(formula="price ~ income + crime", data=df, W=w)
        >>> ols.fit()
        >>> ols.spatial_diagnostics()
                         statistic  median  df  p_value  ci_lower  ci_upper
        LM-Lag                3.21    2.98   1    0.073      0.12      8.54
        LM-Error              5.67    5.34   1    0.017      0.34     12.10
        LM-SDM-Joint          7.89    7.12   4    0.096      1.23     18.32
        LM-SLX-Error-Joint    6.45    5.98   4    0.168      0.89     15.67
        """
        from ..diagnostics.lmtests import BayesianLMTestResult  # noqa: F401

        self._require_fit()
        self._require_W()

        from ..diagnostics.lmtests.registry import get_diagnostic_suite

        suite = get_diagnostic_suite(self)
        if suite is None:
            raise ValueError(
                f"No diagnostic suite registered for {type(self).__name__}. "
                f"Register one in bayespecon.diagnostics.lmtests.registry."
            )
        return self._run_lm_diagnostics(self, suite.tests)

    def spatial_diagnostics_decision(
        self, alpha: float = 0.05, format: str = "graphviz"
    ) -> Any:
        """Return a model-selection decision from Bayesian LM test results.

        Implements the decision tree from :cite:t:`koley2024UseNot`
        (the Bayesian analogue of the classical ``stge_kb`` procedure
        in :cite:t:`anselin1996SimpleDiagnostic`).  The decision logic
        depends on the current model type and the pattern of significant
        tests:

        **From OLS** (6-test decision tree):

        1. If only LM-Lag is significant → SAR.
        2. If only LM-Error is significant → SEM.
        3. If both are significant → use the Anselin–Florax / Koley–Bera
           robust pair: Robust-LM-Lag → SAR, Robust-LM-Error → SEM,
           both → SARAR. If neither robust test is significant, fall
           back to the lower raw p-value.
        4. If neither naive test is significant → OLS.

        **From SAR** (3-test decision tree):

        - LM-Error significant → SARAR; LM-WX significant → SDM;
          Robust-LM-WX significant → SDM.

        **From SEM** (2-test decision tree):

        - LM-Lag significant → SARAR; LM-WX significant → SDEM.

        **From SLX** (4-test decision tree):

        - Robust-LM-Lag-SDM significant → SDM;
          Robust-LM-Error-SDEM significant → SDEM;
          both → MANSAR; neither → SLX.

        **From SDM**: LM-Error-SDM significant → MANSAR; else SDM.

        **From SDEM**: LM-Lag-SDEM significant → MANSAR; else SDEM.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the Bayesian p-values.
        format : {"graphviz", "ascii", "model"}, default "graphviz"
            Output format. ``"model"`` returns the recommended-model name
            string. ``"ascii"`` returns an indented box-drawing rendering
            of the full decision tree with the chosen path highlighted.
            ``"graphviz"`` returns a :class:`graphviz.Digraph` object that
            renders inline in Jupyter; if the optional ``graphviz`` package
            is not installed a :class:`UserWarning` is issued and the
            ASCII rendering is returned instead.

        Returns
        -------
        str or graphviz.Digraph
            Recommended model name when ``format="model"``, an ASCII tree
            string when ``format="ascii"``, or a ``graphviz.Digraph`` when
            ``format="graphviz"`` (with ASCII fallback on missing dep).

        See Also
        --------
        spatial_diagnostics : Compute the Bayesian LM test statistics.

        References
        ----------
        :cite:t:`koley2024UseNot`, :cite:t:`anselin1996SimpleDiagnostic`
        """
        from ..diagnostics import _decision_trees as _dt

        diag = self.spatial_diagnostics()
        model_type = self.__class__.__name__

        def _sig(test_name: str) -> bool:
            if test_name not in diag.index:
                return False
            pval = diag.loc[test_name, "p_value"]
            return not np.isnan(pval) and pval < alpha

        def _lag_le_error() -> bool:
            return diag.loc["LM-Lag", "p_value"] <= diag.loc["LM-Error", "p_value"]

        def _robust_lag_le_error() -> bool:
            # OLS tree tie-break: when both naive AND both robust tests
            # fire, route to the dominant single-channel model based on
            # the smaller robust p-value.  We never escalate directly to
            # SARAR from OLS; the user must fit SAR (or SEM) and re-run
            # diagnostics from there.
            return (
                diag.loc["Robust-LM-Lag", "p_value"]
                <= diag.loc["Robust-LM-Error", "p_value"]
            )

        def _lag_sdm_le_error_sdem() -> bool:
            # Used by the SLX decision tree to break ties when both
            # ``Robust-LM-Lag-SDM`` and ``Robust-LM-Error-SDEM`` are
            # significant: the omitted channel with the smaller p-value
            # (i.e. the larger statistic) wins.
            return (
                diag.loc["Robust-LM-Lag-SDM", "p_value"]
                <= diag.loc["Robust-LM-Error-SDEM", "p_value"]
            )

        def _lag_sem_le_wx_sem() -> bool:
            # Used by the SEM decision tree to break ties when both
            # ``Robust-LM-Lag`` (SEM-null lag score, Schur-purged for the
            # WX block) and ``Robust-LM-WX`` (SEM-null WX score,
            # Schur-purged for the lag) survive.  Smaller p (larger stat)
            # wins: lag direction → SARAR, WX direction → SDEM.
            return (
                diag.loc["Robust-LM-Lag", "p_value"]
                <= diag.loc["Robust-LM-WX", "p_value"]
            )

        spec = _dt.get_spec(model_type)
        decision, path = _dt.evaluate(
            spec,
            sig_lookup=_sig,
            predicate_lookup={
                "lag_pval_le_error_pval": _lag_le_error,
                "robust_lag_pval_le_error_pval": _robust_lag_le_error,
                "lag_sdm_pval_le_error_sdem_pval": _lag_sdm_le_error_sdem,
                "lag_sem_pval_le_wx_sem_pval": _lag_sem_le_wx_sem,
            },
        )

        # Build p-value lookup for renderers (only test rows present).
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

    @abstractmethod
    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute model-specific impact measures at posterior mean.

        Returns
        -------
        dict
            Dictionary with direct, indirect, and total effects.
        """

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)`` where each
            array has shape ``(G, k)`` or ``(G, k_wx)``, with *G* being the
            total number of posterior draws and *k* / *k_wx* being the
            number of covariates for which effects are reported.
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
        return (
            f"{self.__class__.__name__}(n={n}, k={k}, features={self._feature_names})"
        )
