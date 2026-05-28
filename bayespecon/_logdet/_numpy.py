"""NumPy log-determinant factory functions and caching.

Contains ``make_logdet_fn`` (PyTensor), ``make_logdet_numpy_fn``,
``make_logdet_numpy_vec_fn``, ``get_cached_logdet_fn``, and supporting
helpers.
"""

import hashlib

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline

from ._config import (
    _LOGDET_FN_CACHE,
    _LOGDET_FN_CACHE_MAXSIZE,
    VALID_LOGDET_METHODS,
    TraceEstimatorName,
    _default_trace_k,
    _resolve_trace_estimator,
    resolve_logdet_method,
)
from ._grids import (
    _build_logdet_grid,
    chebyshev,
    ilu,
    mc,
    sparse_grid,
    spline,
)
from ._pytensor import (
    _make_pytensor_interp_fn,
    logdet_chebyshev,
    logdet_eigenvalue,
    logdet_exact,
)

_GRID_SPLINE_METHODS = (
    "grid_dense",
    "grid_sparse",
    "sparse_spline",
    "grid_mc",
    "grid_ilu",
)


def _build_grid_spline(
    W_dense: np.ndarray,
    method: str,
    rho_min: float,
    rho_max: float,
    *,
    clamp_nonnegative: bool,
) -> CubicSpline:
    """Build a CubicSpline approximation for grid/interpolation methods."""
    if method == "grid_dense":
        rho_grid, logdet_grid = _build_logdet_grid(W_dense, rho_min, rho_max)
        return CubicSpline(rho_grid, logdet_grid)
    if method == "grid_sparse":
        out = sparse_grid(W_dense, rho_min, rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    if method == "sparse_spline":
        rmin = max(rho_min, 0.0) if clamp_nonnegative else rho_min
        out = spline(W_dense, rmin, rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    if method == "grid_mc":
        rmin = max(rho_min, 1e-5) if clamp_nonnegative else rho_min
        out = mc(order=50, iter=30, W=W_dense, rmin=rmin, rmax=rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    if method == "grid_ilu":
        out = ilu(W_dense, rho_min, rho_max)
        return CubicSpline(out["rho"], out["lndet"])
    raise ValueError(
        f"Grid spline helper does not support method={method!r}. "
        f"Choose one of {_GRID_SPLINE_METHODS}."
    )


def make_logdet_numpy_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
    trace_estimator: TraceEstimatorName = "hutchpp",
    trace_k: int | None = None,
):
    """Return a **pure-numpy** ``(rho: float) -> float`` logdet evaluator.

    Used for post-sampling log-likelihood Jacobian computation (outside any
    PyMC/PyTensor graph context).  Mirrors :func:`make_logdet_fn` but returns
    a plain Python callable instead of a PyTensor expression.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        Row-standardised n×n spatial weights matrix.
    eigs : np.ndarray, optional
        Optional pre-computed eigenvalues of W (complex or real).
    method : str or None
        Same as :func:`make_logdet_fn`.  ``None`` auto-selects via
        :func:`_auto_logdet_method`.
    rho_min : float, default -1.0
        Lower bound (used for chebyshev/spline precomputation).
    rho_max : float, default 1.0
        Upper bound.
    T : int, default 1
        Panel time-period count.  The returned log-determinant is
        multiplied by *T*.
    trace_estimator : {"hutchinson", "hutchpp", "xtrace"}, default "hutchpp"
        Stochastic trace estimator used to build the Chebyshev
        coefficients when an eigendecomposition is unavailable.
    trace_k : int, optional
        Number of probe vectors for the trace estimator.  Defaults:
        ``30`` (hutchinson), ``50`` (hutchpp), ``25`` (xtrace).

    Returns
    -------
    callable
        Function ``(rho: float) -> float`` computing log|I - rho*W|
        (or T * log|I - rho*W| for panel models).
    """
    T = int(T)
    trace_estimator = _resolve_trace_estimator(trace_estimator)
    _k = trace_k if trace_k is not None else _default_trace_k(trace_estimator)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        # Keep complex eigenvalues — np.abs computes the complex modulus
        # correctly for non-symmetric W.  For symmetric W the eigenvalues
        # are real and this is equivalent to the old .real path.
        _eigs = np.asarray(eigs, dtype=np.complex128)
        if T == 1:
            return lambda r: float(np.sum(np.log(np.abs(1.0 - r * _eigs))))
        return lambda r: T * float(np.sum(np.log(np.abs(1.0 - r * _eigs))))

    elif method == "chebyshev":
        # Pass precomputed eigs to skip the redundant toarray + eigvals
        # that chebyshev() would otherwise perform internally.
        out = chebyshev(
            W_sparse,
            order=20,
            rmin=rho_min,
            rmax=rho_max,
            eigs=eigs,
            estimator=trace_estimator,
            n_mc_iter=_k,
        )
        coeffs = out["coeffs"]
        rmin_cb, rmax_cb = out["rmin"], out["rmax"]
        m = len(coeffs)

        def _cheb_numpy(r):
            r = float(r)
            x = (2.0 * r - rmax_cb - rmin_cb) / (rmax_cb - rmin_cb)
            if m == 0:
                return 0.0
            if m == 1:
                return float(coeffs[0])
            b_next = 0.0
            b_curr = float(coeffs[m - 1])
            for k in range(m - 2, 0, -1):
                b_new = 2.0 * x * b_curr - b_next + float(coeffs[k])
                b_next = b_curr
                b_curr = b_new
            val = float(coeffs[0]) + x * b_curr - b_next
            return val if T == 1 else T * val

        return _cheb_numpy

    elif method in _GRID_SPLINE_METHODS:
        # Grid/spline methods: precompute numpy spline, then evaluate directly.
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
        spl = _build_grid_spline(
            W_dense,
            method,
            rho_min,
            rho_max,
            clamp_nonnegative=True,
        )
        if T == 1:
            return lambda r: float(spl(float(r)))
        return lambda r: T * float(spl(float(r)))

    else:
        # Fallback: exact numpy slogdet (slow but always correct)
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
        n_mat = W_dense.shape[0]
        I = np.eye(n_mat)
        if T == 1:
            return lambda r: float(np.linalg.slogdet(I - r * W_dense)[1])
        return lambda r: T * float(np.linalg.slogdet(I - r * W_dense)[1])


def make_logdet_numpy_vec_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
    trace_estimator: TraceEstimatorName = "hutchpp",
    trace_k: int | None = None,
):
    """Return a **vectorized** numpy ``(rho_arr: np.ndarray) -> np.ndarray`` logdet evaluator.

    Companion to :func:`make_logdet_numpy_fn` for batch evaluation over an
    array of posterior draws without a Python loop. Each supported method uses
    an array-native implementation.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        Row-standardised n×n spatial weights matrix.
    eigs : np.ndarray, optional
        Optional pre-computed eigenvalues of W (complex or real).
    method : str or None
        Same as :func:`make_logdet_numpy_fn`.
    rho_min : float, default -1.0
    rho_max : float, default 1.0
    T : int, default 1
        Panel time-period count.  The returned log-determinant is
        multiplied by *T*.
    trace_estimator : {"hutchinson", "hutchpp", "xtrace"}, default "hutchpp"
        Stochastic trace estimator used to build the Chebyshev
        coefficients when an eigendecomposition is unavailable.
    trace_k : int, optional
        Number of probe vectors for the trace estimator.  Defaults:
        ``30`` (hutchinson), ``50`` (hutchpp), ``25`` (xtrace).

    Returns
    -------
    callable
        Function ``(rho_arr: np.ndarray) -> np.ndarray`` of shape ``(G,)``
        computing log|I - rho*W| (or T * log|I - rho*W| for panel models).
    """
    T = int(T)
    trace_estimator = _resolve_trace_estimator(trace_estimator)
    _k = trace_k if trace_k is not None else _default_trace_k(trace_estimator)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        # Keep complex eigenvalues — np.abs computes the complex modulus
        # correctly for non-symmetric W.
        _eigs = np.asarray(eigs, dtype=np.complex128)

        def _vec_eigenvalue(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            val = np.sum(
                np.log(np.abs(1.0 - rho_arr[:, None] * _eigs[None, :])), axis=1
            )
            return val if T == 1 else T * val

        return _vec_eigenvalue

    if method == "chebyshev":
        out = chebyshev(
            W_sparse,
            order=20,
            rmin=rho_min,
            rmax=rho_max,
            eigs=eigs,
            estimator=trace_estimator,
            n_mc_iter=_k,
        )
        coeffs = out["coeffs"].astype(np.float64)
        rmin_cb, rmax_cb = float(out["rmin"]), float(out["rmax"])
        m = len(coeffs)

        def _vec_chebyshev(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            x = (2.0 * rho_arr - rmax_cb - rmin_cb) / (rmax_cb - rmin_cb)

            if m == 0:
                return np.zeros_like(rho_arr, dtype=np.float64)
            if m == 1:
                return np.full_like(rho_arr, coeffs[0], dtype=np.float64)

            # Vectorized Clenshaw recurrence over all rho draws.
            b_next = np.zeros_like(x, dtype=np.float64)
            b_curr = np.full_like(x, coeffs[m - 1], dtype=np.float64)
            for k in range(m - 2, 0, -1):
                b_new = 2.0 * x * b_curr - b_next + coeffs[k]
                b_next = b_curr
                b_curr = b_new
            val = coeffs[0] + x * b_curr - b_next
            return val if T == 1 else T * val

        return _vec_chebyshev

    if method in _GRID_SPLINE_METHODS:
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
        spl = _build_grid_spline(
            W_dense,
            method,
            rho_min,
            rho_max,
            clamp_nonnegative=True,
        )

        def _vec_grid(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            val = np.asarray(spl(rho_arr), dtype=np.float64)
            return val if T == 1 else T * val

        return _vec_grid

    # Fallback: exact batched slogdet using stacked (G, n, n) matrices.
    W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
    n_mat = W_dense.shape[0]
    I = np.eye(n_mat, dtype=np.float64)

    def _vec_exact(rho_arr: np.ndarray) -> np.ndarray:
        rho_arr = np.asarray(rho_arr, dtype=np.float64)
        mats = I[None, :, :] - rho_arr[:, None, None] * W_dense[None, :, :]
        val = np.linalg.slogdet(mats)[1]
        return val if T == 1 else T * val

    return _vec_exact


def make_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
    trace_estimator: TraceEstimatorName = "hutchpp",
    trace_k: int | None = None,
):
    """Return a function (rho) -> pytensor log|I - rho*W| expression.

    Parameters
    ----------
    W : np.ndarray or scipy.sparse matrix
        Either a 2-D dense ``(n, n)`` spatial weights matrix **or** a 1-D
        array of pre-computed real eigenvalues.  Passing eigenvalues skips the
        O(n³) decomposition inside this function; the ``'grid_dense'`` and
        ``'exact'`` methods are not available in that case and fall back to
        ``'eigenvalue'``.
    method : str
        Auto-selected when ``None`` (``"eigenvalue"`` for ``n <= 500`` else
        ``"chebyshev"``). Supported values:

        ``"eigenvalue"`` — pre-compute W's eigenvalues once (O(n³)); every
        subsequent evaluation costs O(n) and is exact.
        ``"exact"`` — exact O(n³) symbolic det via pytensor (slow for n > 500).
        ``"grid_dense"`` — dense eigenvalue grid + cubic-spline interpolation.
        ``"grid_sparse"`` — sparse-LU grid + cubic-spline interpolation
        for large sparse W.
        ``"sparse_spline"`` — sparse-LU + cubic-spline interpolation on
        ``[max(rho_min, 0), rho_max]``.
        ``"grid_mc"`` — Monte Carlo trace approximation
        (:cite:p:`barry1999MonteCarlo`) + spline interpolation.
        ``"grid_ilu"`` — ILU-based approximation
        + spline interpolation.
        ``"chebyshev"`` — Chebyshev polynomial approximation
        (:cite:p:`pace2004ChebyshevApproximation`); near-minimax
        polynomial evaluated via Clenshaw's algorithm.  Coefficients are
        built from exact eigenvalues when ``n`` is small (or ``eigs`` is
        supplied); otherwise from a stochastic trace estimator selected
        by ``trace_estimator``.
    rho_min : float, default=-1.0
        Lower bound for the grid method.
    rho_max : float, default=1.0
        Upper bound for the grid method.
    T : int, default 1
        Panel time-period count.  The returned log-determinant is multiplied
        by *T*, exploiting
        ``log|I_{NT} - ρ(I_T⊗W_N)| = T · log|I_N - ρW_N|``.
        Leave at 1 for cross-sectional models.
    trace_estimator : {"hutchinson", "hutchpp", "xtrace"}, default "hutchpp"
        Stochastic trace estimator used to build the Chebyshev
        coefficients when an eigendecomposition is unavailable.  Ignored
        for non-Chebyshev methods and when eigenvalues are passed in.
        See ``docs/source/user-guide/logdet_profiling.ipynb`` for the
        cost/accuracy frontier.
    trace_k : int, optional
        Number of probe vectors for the trace estimator.  Defaults:
        ``30`` (hutchinson), ``50`` (hutchpp), ``25`` (xtrace).

    Returns
    -------
    callable
        Function mapping symbolic ``rho`` to symbolic log-determinant.
    """
    T = int(T)
    trace_estimator = _resolve_trace_estimator(trace_estimator)
    _k = trace_k if trace_k is not None else _default_trace_k(trace_estimator)

    if sp.issparse(W):
        W_sparse = W.tocsr().astype(np.float64)
        method = resolve_logdet_method(method, n=W_sparse.shape[0])

        if method == "chebyshev":
            out = chebyshev(
                W_sparse,
                order=20,
                rmin=rho_min,
                rmax=rho_max,
                estimator=trace_estimator,
                n_mc_iter=_k,
            )
            coeffs_np = out["coeffs"]
            rmin_cb = out["rmin"]
            rmax_cb = out["rmax"]

            def _chebyshev_sparse_interp(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _chebyshev_sparse_interp

        # Other methods expect dense/eigenvalue inputs below.
        W = np.asarray(W_sparse.toarray(), dtype=np.float64)
    else:
        W = np.asarray(W, dtype=np.float64)

    if W.ndim == 1:
        # 1-D eigenvalue array supplied — skip O(n³) decomposition.
        eigs = W
        method = resolve_logdet_method(method, n=eigs.shape[0])
        if method == "eigenvalue":
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        if method == "chebyshev":
            # Build Chebyshev approximation directly from cached eigenvalues
            # (no dense W needed).
            out = chebyshev(None, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
            coeffs_np = out["coeffs"]
            rmin_cb = out["rmin"]
            rmax_cb = out["rmax"]

            def _chebyshev_eig_interp(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _chebyshev_eig_interp
        # Other methods (grid_dense, exact, grid_sparse, sparse_spline, grid_mc, grid_ilu)
        # require the full matrix; fall back to eigenvalue with a note.
        if method in (
            "grid_dense",
            "exact",
            "grid_sparse",
            "sparse_spline",
            "grid_mc",
            "grid_ilu",
        ):
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        raise ValueError(f"Unsupported logdet method for eigenvalue input: {method!r}.")

    # 2-D dense matrix path.
    W_dense = W
    method = resolve_logdet_method(method, n=W_dense.shape[0])
    if method in ("sparse_spline", "grid_mc") and rho_min < 0.0:
        raise ValueError(
            f"method='{method}' is defined for nonnegative rho ranges; "
            "use rho_min >= 0 or choose 'eigenvalue'/'exact'/'grid_dense'/'grid_sparse'/'grid_ilu'/'chebyshev'."
        )
    if method == "eigenvalue":
        eigs = np.linalg.eigvals(W_dense)
        if T == 1:
            return lambda rho: logdet_eigenvalue(rho, eigs)
        return lambda rho: T * logdet_eigenvalue(rho, eigs)
    elif method == "exact":
        if T == 1:
            return lambda rho: logdet_exact(rho, W_dense)
        return lambda rho: T * logdet_exact(rho, W_dense)
    elif method in _GRID_SPLINE_METHODS:
        spline_obj = _build_grid_spline(
            W_dense,
            method,
            rho_min,
            rho_max,
            clamp_nonnegative=False,
        )
        return _make_pytensor_interp_fn(spline_obj, T)
    elif method == "chebyshev":
        out = chebyshev(
            W_dense,
            order=20,
            rmin=rho_min,
            rmax=rho_max,
            estimator=trace_estimator,
            n_mc_iter=_k,
        )
        coeffs_np = out["coeffs"]
        rmin_cb = out["rmin"]
        rmax_cb = out["rmax"]

        def _chebyshev_interp(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _chebyshev_interp
    else:
        valid = ", ".join(sorted(VALID_LOGDET_METHODS))
        raise ValueError(f"Unknown method: {method!r}. Choose one of: {valid}.")


def _hash_array(arr: np.ndarray) -> str:
    """Return a stable hash for numeric numpy arrays."""
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr_c.dtype).encode("ascii"))
    h.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _logdet_w_signature(W) -> tuple:
    """Build a stable cache signature for dense/sparse/eigenvalue W inputs."""
    if sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray(W_csr.shape, dtype=np.int64).tobytes())
        h.update(np.asarray([W_csr.nnz], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(W_csr.indptr).view(np.uint8))
        h.update(np.ascontiguousarray(W_csr.indices).view(np.uint8))
        h.update(np.ascontiguousarray(W_csr.data).view(np.uint8))
        return ("sparse", W_csr.shape, int(W_csr.nnz), h.hexdigest())

    W_arr = np.asarray(W, dtype=np.float64)
    if W_arr.ndim == 1:
        return ("eigs", W_arr.shape, _hash_array(W_arr))
    if W_arr.ndim == 2:
        return ("dense", W_arr.shape, _hash_array(W_arr))
    raise ValueError(f"Unsupported W with ndim={W_arr.ndim}; expected 1D or 2D.")


def clear_logdet_fn_cache() -> None:
    """Clear the shared cache of precomputed PyTensor logdet callables."""
    _LOGDET_FN_CACHE.clear()


def get_cached_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
    trace_estimator: TraceEstimatorName = "hutchpp",
    trace_k: int | None = None,
):
    """Return a shared cached ``make_logdet_fn`` callable.

    Cache key includes a stable signature of ``W`` plus ``method``, bounds,
    panel multiplier ``T``, and the Chebyshev trace estimator settings.
    This avoids repeatedly rebuilding equivalent logdet approximations
    across model instances.
    """
    T = int(T)
    trace_estimator = _resolve_trace_estimator(trace_estimator)
    if sp.issparse(W):
        n_w = int(W.shape[0])
    else:
        W_arr = np.asarray(W)
        n_w = int(W_arr.shape[0])
    if method is None and not sp.issparse(W) and W_arr.ndim == 1:
        resolved_method = "eigenvalue"
    else:
        resolved_method = resolve_logdet_method(method, n=n_w)

    key = (
        _logdet_w_signature(W),
        resolved_method,
        float(rho_min),
        float(rho_max),
        T,
        trace_estimator,
        trace_k,
    )
    fn = _LOGDET_FN_CACHE.get(key)
    if fn is not None:
        _LOGDET_FN_CACHE.move_to_end(key)
        return fn

    fn = make_logdet_fn(
        W,
        method=resolved_method,
        rho_min=rho_min,
        rho_max=rho_max,
        T=T,
        trace_estimator=trace_estimator,
        trace_k=trace_k,
    )
    _LOGDET_FN_CACHE[key] = fn
    if len(_LOGDET_FN_CACHE) > _LOGDET_FN_CACHE_MAXSIZE:
        _LOGDET_FN_CACHE.popitem(last=False)
    return fn
