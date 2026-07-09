"""Factory functions that build logdet evaluators.

Three backends:

* **PyTensor** (``make_logdet_fn``) — symbolic expression for ``pm.Potential``.
* **NumPy scalar** (``make_logdet_numpy_fn``) — ``(rho: float) -> float``.
* **NumPy vectorized** (``make_logdet_numpy_vec_fn``) — ``(rho_arr) -> np.ndarray``.
* **JAX** (``make_logdet_jax_fn`` in ``_jax.py``) — JIT-compatible.

All accept ``method="eigenvalue"`` or ``method="chebyshev"`` (or ``None``
for auto-select).  ``T`` multiplies the result for panel models.
"""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

from ._cheb_stochastic import (
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_precompute,
)
from ._chebyshev import chebyshev
from ._clenshaw import clenshaw_scalar as _clenshaw_scalar
from ._clenshaw import clenshaw_vec as _clenshaw_vec
from ._config import (
    _LOGDET_FN_CACHE,
    _LOGDET_FN_CACHE_MAXSIZE,
    resolve_logdet_method,
)
from ._pytensor import logdet_chebyshev, logdet_eigenvalue
from ._slq import (
    slq_logdet_precompute,
    slq_to_chebyshev_coeffs,
)

# ---------------------------------------------------------------------------
# Stochastic-Chebyshev coefficient helper
# ---------------------------------------------------------------------------


def _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max):
    """Stochastic-Chebyshev logdet → Chebyshev-in-ρ coefficients (order 20).

    Precomputes stochastic moments, evaluates the logdet at order-20
    Chebyshev nodes in ``[rho_min, rho_max]``, then fits a Chebyshev-in-ρ
    polynomial via DCT-I so it can be evaluated with an O(m) Clenshaw
    recurrence.
    """
    pre = cheb_stochastic_logdet_precompute(W_sparse)
    k_nodes = np.arange(1, 21)
    nodes_cos = np.cos((2 * k_nodes - 1) * np.pi / 40)
    rho_nodes = 0.5 * (rho_max - rho_min) * nodes_cos + 0.5 * (rho_max + rho_min)
    logdet_vals = np.array(
        [cheb_stochastic_logdet_eval(pre, float(r)) for r in rho_nodes]
    )
    coeffs = np.zeros(20, dtype=np.float64)
    for j in range(20):
        scale = 2.0 / 20 if j > 0 else 1.0 / 20
        coeffs[j] = scale * np.sum(
            logdet_vals * np.cos(j * (2 * k_nodes - 1) * np.pi / 40)
        )
    return coeffs, float(rho_min), float(rho_max)


# ---------------------------------------------------------------------------
# NumPy scalar factory
# ---------------------------------------------------------------------------


def make_logdet_numpy_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a pure-numpy ``(rho: float) -> float`` logdet evaluator."""
    T = int(T)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n, W=W_sparse)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        _eigs = np.asarray(eigs, dtype=np.complex128)
        if T == 1:
            return lambda r: float(np.sum(np.log(np.abs(1.0 - r * _eigs))))
        return lambda r: T * float(np.sum(np.log(np.abs(1.0 - r * _eigs))))

    if method == "chebyshev":
        out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
        coeffs = out["coeffs"]
        rmin_cb, rmax_cb = out["rmin"], out["rmax"]
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    if method == "cheb_stochastic":
        coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max)
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    if method == "cheb_cholesky":
        from ._chol_cheb import chol_cheb_logdet_precompute

        # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
        # No stochastic noise, no O(n³) eigendecomposition.  SPD for |ρ| < 1.
        pre = chol_cheb_logdet_precompute(
            W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = pre.coeffs
        rmin_cb, rmax_cb = pre.rho_min, pre.rho_max
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    if method == "aaa":
        from ._aaa import aaa_logdet_eval, aaa_logdet_precompute

        # AAA rational approximation: exact logdet via sparse LU at
        # adaptively-selected support points, then barycentric evaluation.
        # For non-symmetric W where Cholesky is unavailable.
        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _aaa_numpy(r):
            val = aaa_logdet_eval(pre, float(r))
            return val if T == 1 else T * val

        return _aaa_numpy

    if method == "slq":
        # SLQ precompute → Chebyshev coefficients → Clenshaw evaluation
        pre = slq_logdet_precompute(W_sparse)
        cheb = slq_to_chebyshev_coeffs(
            pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = cheb["coeffs"]
        rmin_cb, rmax_cb = cheb["rmin"], cheb["rmax"]
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    raise ValueError(f"Unsupported logdet method: {method!r}")


# ---------------------------------------------------------------------------
# NumPy vectorized factory
# ---------------------------------------------------------------------------


def make_logdet_numpy_vec_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a vectorized numpy ``(rho_arr: np.ndarray) -> np.ndarray`` logdet evaluator."""
    T = int(T)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n, W=W_sparse)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        _eigs = np.asarray(eigs, dtype=np.complex128)

        def _vec_eigenvalue(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            val = np.sum(
                np.log(np.abs(1.0 - rho_arr[:, None] * _eigs[None, :])), axis=1
            )
            return val if T == 1 else T * val

        return _vec_eigenvalue

    if method == "chebyshev":
        out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
        coeffs = out["coeffs"].astype(np.float64)
        rmin_cb, rmax_cb = float(out["rmin"]), float(out["rmax"])
        return lambda rho_arr: _clenshaw_vec(coeffs, rho_arr, rmin_cb, rmax_cb, T)

    if method == "cheb_stochastic":
        coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max)
        return lambda rho_arr: _clenshaw_vec(coeffs, rho_arr, rmin_cb, rmax_cb, T)

    if method == "cheb_cholesky":
        from ._chol_cheb import chol_cheb_logdet_eval_vec, chol_cheb_logdet_precompute

        # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
        pre = chol_cheb_logdet_precompute(
            W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
        )

        def _vec_cheb_chol(rho_arr: np.ndarray) -> np.ndarray:
            vals = chol_cheb_logdet_eval_vec(pre, np.asarray(rho_arr, dtype=np.float64))
            return vals if T == 1 else T * vals

        return _vec_cheb_chol

    if method == "aaa":
        from ._aaa import aaa_logdet_eval_vec, aaa_logdet_precompute

        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _vec_aaa(rho_arr: np.ndarray) -> np.ndarray:
            vals = aaa_logdet_eval_vec(pre, np.asarray(rho_arr, dtype=np.float64))
            return vals if T == 1 else T * vals

        return _vec_aaa

    if method == "slq":
        # SLQ precompute → Chebyshev coefficients → vectorized Clenshaw
        pre = slq_logdet_precompute(W_sparse)
        cheb = slq_to_chebyshev_coeffs(
            pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = cheb["coeffs"].astype(np.float64)
        rmin_cb, rmax_cb = float(cheb["rmin"]), float(cheb["rmax"])
        return lambda rho_arr: _clenshaw_vec(coeffs, rho_arr, rmin_cb, rmax_cb, T)

    raise ValueError(f"Unsupported logdet method: {method!r}")


# ---------------------------------------------------------------------------
# PyTensor factory
# ---------------------------------------------------------------------------


def make_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a function ``(rho) -> pytensor`` log|I - ρW| expression.

    Accepts a dense matrix, sparse matrix, or 1-D eigenvalue array.
    """
    T = int(T)

    if sp.issparse(W):
        W_sparse = W.tocsr().astype(np.float64)
        method = resolve_logdet_method(method, n=W_sparse.shape[0], W=W_sparse)
        if method == "cheb_stochastic":
            # Stochastic Chebyshev → Clenshaw-like eval (PyTensor-differentiable)
            # Build Cheb coeffs at Chebyshev nodes in ρ-space, then use logdet_chebyshev
            # for the differentiable Clenshaw evaluation.
            pre = cheb_stochastic_logdet_precompute(W_sparse)
            # Precompute Chebyshev-in-ρ coefficients from stochastic moments
            # Evaluate at order Chebyshev nodes in [rho_min, rho_max], fit polynomial
            k_nodes = np.arange(1, 21)
            nodes_cos = np.cos((2 * k_nodes - 1) * np.pi / 40)
            rho_nodes = 0.5 * (rho_max - rho_min) * nodes_cos + 0.5 * (
                rho_max + rho_min
            )
            logdet_vals = np.array(
                [cheb_stochastic_logdet_eval(pre, float(r)) for r in rho_nodes]
            )
            # DCT-I → Chebyshev coefficients in ρ
            coeffs_np = np.zeros(20, dtype=np.float64)
            for j in range(20):
                scale = 2.0 / 20 if j > 0 else 1.0 / 20
                coeffs_np[j] = scale * np.sum(
                    logdet_vals * np.cos(j * (2 * k_nodes - 1) * np.pi / 40)
                )
            rmin_cb, rmax_cb = rho_min, rho_max

            def _cheb_stoch_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_stoch_sparse
        if method == "cheb_cholesky":
            from ._chol_cheb import chol_cheb_logdet_precompute

            # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
            pre = chol_cheb_logdet_precompute(
                W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
            )
            coeffs_np = pre.coeffs.astype(np.float64)
            rmin_cb, rmax_cb = pre.rho_min, pre.rho_max

            def _cheb_chol_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_chol_sparse
        if method == "aaa":
            from ._aaa import aaa_logdet_precompute

            # AAA rational approximation for non-symmetric W.
            # PyTensor-differentiable via barycentric evaluation on
            # precomputed support points and weights.
            pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
            sp_z = pre.support_points.astype(np.float64)
            sp_f = pre.support_values.astype(np.float64)
            w = pre.weights.astype(np.float64)

            def _aaa_sparse(rho):
                # Barycentric formula using PyTensor operations
                import pytensor.tensor as pt

                diff = rho - sp_z  # (m,)
                n_val = pt.sum(w * sp_f / diff)
                d_val = pt.sum(w / diff)
                val = n_val / d_val
                return val if T == 1 else T * val

            return _aaa_sparse
        if method == "slq":
            # SLQ precompute → Chebyshev coefficients → differentiable Clenshaw
            pre = slq_logdet_precompute(W_sparse)
            cheb = slq_to_chebyshev_coeffs(
                pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
            )
            coeffs_np = cheb["coeffs"]
            rmin_cb, rmax_cb = cheb["rmin"], cheb["rmax"]

            def _slq_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _slq_sparse
        if method == "chebyshev":
            out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max)
            coeffs_np = out["coeffs"]
            rmin_cb, rmax_cb = out["rmin"], out["rmax"]

            def _cheb_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_sparse
        # eigenvalue path: materialize dense
        W = np.asarray(W_sparse.toarray(), dtype=np.float64)
    else:
        W = np.asarray(W, dtype=np.float64)

    if W.ndim == 1:
        # 1-D eigenvalue array
        eigs = W
        method = resolve_logdet_method(method, n=eigs.shape[0])
        if method == "eigenvalue":
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        if method == "chebyshev":
            out = chebyshev(None, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
            coeffs_np = out["coeffs"]
            rmin_cb, rmax_cb = out["rmin"], out["rmax"]

            def _cheb_eig(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_eig
        raise ValueError(f"Unsupported logdet method for eigenvalue input: {method!r}")

    # 2-D dense matrix
    W_dense = W
    method = resolve_logdet_method(method, n=W_dense.shape[0], W=W_dense)
    if method == "cheb_stochastic":
        # Stochastic Chebyshev → Clenshaw (PyTensor-differentiable via ρ-space coeffs)
        pre = cheb_stochastic_logdet_precompute(W_dense)
        k_nodes = np.arange(1, 21)
        nodes_cos = np.cos((2 * k_nodes - 1) * np.pi / 40)
        rho_nodes = 0.5 * (rho_max - rho_min) * nodes_cos + 0.5 * (rho_max + rho_min)
        logdet_vals = np.array(
            [cheb_stochastic_logdet_eval(pre, float(r)) for r in rho_nodes]
        )
        coeffs_np = np.zeros(20, dtype=np.float64)
        for j in range(20):
            scale = 2.0 / 20 if j > 0 else 1.0 / 20
            coeffs_np[j] = scale * np.sum(
                logdet_vals * np.cos(j * (2 * k_nodes - 1) * np.pi / 40)
            )
        rmin_cb, rmax_cb = rho_min, rho_max

        def _cheb_stoch_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _cheb_stoch_dense
    if method == "cheb_cholesky":
        from ._chol_cheb import chol_cheb_logdet_precompute

        # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
        pre = chol_cheb_logdet_precompute(
            sp.csr_matrix(W_dense), order=None, rho_min=rho_min, rho_max=rho_max
        )
        coeffs_np = pre.coeffs.astype(np.float64)
        rmin_cb, rmax_cb = pre.rho_min, pre.rho_max

        def _cheb_chol_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _cheb_chol_dense
    if method == "aaa":
        from ._aaa import aaa_logdet_precompute

        pre = aaa_logdet_precompute(
            sp.csc_matrix(W_dense), rho_min=rho_min, rho_max=rho_max
        )
        sp_z = pre.support_points.astype(np.float64)
        sp_f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        def _aaa_dense(rho):
            import pytensor.tensor as pt

            diff = rho - sp_z
            n_val = pt.sum(w * sp_f / diff)
            d_val = pt.sum(w / diff)
            val = n_val / d_val
            return val if T == 1 else T * val

        return _aaa_dense
    if method == "slq":
        # SLQ precompute → Chebyshev coefficients → differentiable Clenshaw
        pre = slq_logdet_precompute(W_dense)
        cheb = slq_to_chebyshev_coeffs(
            pre, W=sp.csr_matrix(W_dense), order=20, rho_min=rho_min, rho_max=rho_max
        )
        coeffs_np = cheb["coeffs"]
        rmin_cb, rmax_cb = cheb["rmin"], cheb["rmax"]

        def _slq_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _slq_dense
    if method == "eigenvalue":
        eigs = np.linalg.eigvals(W_dense)
        if T == 1:
            return lambda rho: logdet_eigenvalue(rho, eigs)
        return lambda rho: T * logdet_eigenvalue(rho, eigs)
    if method == "chebyshev":
        out = chebyshev(W_dense, order=20, rmin=rho_min, rmax=rho_max)
        coeffs_np = out["coeffs"]
        rmin_cb, rmax_cb = out["rmin"], out["rmax"]

        def _cheb_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _cheb_dense
    raise ValueError(f"Unsupported logdet method: {method!r}")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _hash_array(arr: np.ndarray) -> str:
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr_c.dtype).encode("ascii"))
    h.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _logdet_w_signature(W) -> tuple:
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
    raise ValueError(f"Unsupported W with ndim={W_arr.ndim}")


def clear_logdet_fn_cache() -> None:
    """Clear the shared cache of precomputed PyTensor logdet callables."""
    _LOGDET_FN_CACHE.clear()


def get_cached_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a shared cached ``make_logdet_fn`` callable."""
    T = int(T)
    if sp.issparse(W):
        n_w = int(W.shape[0])
    else:
        n_w = int(np.asarray(W).shape[0])
    resolved_method = resolve_logdet_method(method, n=n_w, W=W)

    key = (
        _logdet_w_signature(W),
        resolved_method,
        float(rho_min),
        float(rho_max),
        T,
    )
    fn = _LOGDET_FN_CACHE.get(key)
    if fn is not None:
        _LOGDET_FN_CACHE.move_to_end(key)
        return fn

    fn = make_logdet_fn(
        W, method=resolved_method, rho_min=rho_min, rho_max=rho_max, T=T
    )
    _LOGDET_FN_CACHE[key] = fn
    if len(_LOGDET_FN_CACHE) > _LOGDET_FN_CACHE_MAXSIZE:
        _LOGDET_FN_CACHE.popitem(last=False)
    return fn
