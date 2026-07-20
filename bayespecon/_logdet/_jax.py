"""JAX-native log-determinant evaluation.

Mirrors the PyTensor symbolic functions but uses ``jax.numpy`` so the
returned callables are compatible with ``jax.jit`` and ``jax.grad``.

Supports ``"eigenvalue"``, ``"chebyshev"``, ``"cheb_stochastic"``,
``"cheb_cholesky"``, ``"aaa"``, and ``"slq"``.  The stochastic Chebyshev
method precomputes moments in numpy and evaluates via JAX-native Clenshaw.
SLQ consumes the numpy :func:`slq_logdet_precompute` quadrature rules
(sparse batched D-symmetrised Lanczos, canonical ``n·v₁²`` weights; complex
bilinear ``γ`` for the directed-``W`` Arnoldi fallback) and evaluates the
ρ-dependent quadrature in JAX — differentiable and JIT-compatible, with no
dense ``W`` materialisation.  ``cheb_cholesky`` precomputes Chebyshev
coefficients via sparse Cholesky in numpy and evaluates via JAX-native
Clenshaw.  ``aaa`` precomputes support points and barycentric weights via
sparse LU in numpy and evaluates via JAX-native barycentric formula.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._chebyshev import chebyshev
from ._config import resolve_logdet_method
from ._slq import slq_logdet_precompute


def jax_logdet_chebyshev(
    rho,
    coeffs: np.ndarray,
    rmin: float = -1.0,
    rmax: float = 1.0,
):
    """Evaluate Chebyshev approximation of log|I - ρW| in JAX via Clenshaw."""
    import jax.numpy as jnp

    m = len(coeffs)
    if m == 0:
        return jnp.zeros_like(rho)

    x = (2.0 * rho - rmax - rmin) / (rmax - rmin)

    if m == 1:
        return jnp.full_like(rho, coeffs[0])

    c = jnp.asarray(coeffs, dtype=jnp.float64)
    b_next = jnp.zeros_like(x)
    b_curr = jnp.broadcast_to(c[m - 1], jnp.shape(x))

    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + c[k]
        b_next = b_curr
        b_curr = b_new

    return c[0] + x * b_curr - b_next


def make_logdet_jax_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a JAX-native ``(rho) -> log|I - ρW|`` callable.

    Supports ``"eigenvalue"``, ``"chebyshev"``, ``"cheb_stochastic"``,
    ``"cheb_cholesky"``, ``"aaa"``, ``"cholmod"``, and ``"slq"``.
    ``"cholmod"`` uses ``cholgraph`` for exact sparse CHOLMOD logdet
    (requires the ``cholgraph`` package; CPU-only).
    """
    T = int(T)

    eigs = None
    if sp.issparse(W):
        W_sparse = W.tocsr().astype(np.float64)
        n = W_sparse.shape[0]
    else:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim == 1:
            eigs = W_arr
            n = len(eigs)
        else:
            n = W_arr.shape[0]
            W_sparse = sp.csr_matrix(W_arr)

    method = resolve_logdet_method(
        method, n=n, W=W_sparse if "W_sparse" in dir() else W_arr
    )

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(W_sparse.toarray())
        _eigs = np.asarray(eigs, dtype=np.complex128)

        def _jax_eigenvalue(rho):
            import jax.numpy as jnp

            eigs_jax = jnp.asarray(_eigs)
            result = jnp.sum(jnp.log(jnp.abs(1.0 - rho * eigs_jax)))
            return result if T == 1 else T * result

        return _jax_eigenvalue

    if method == "chebyshev":
        out = chebyshev(
            W_sparse if eigs is None else None,
            order=20,
            rmin=rho_min,
            rmax=rho_max,
            eigs=eigs,
        )
        coeffs = out["coeffs"].astype(np.float64)
        rmin_cb = float(out["rmin"])
        rmax_cb = float(out["rmax"])

        def _jax_chebyshev(rho):
            val = jax_logdet_chebyshev(rho, coeffs, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _jax_chebyshev

    if method == "cheb_stochastic":
        # Precompute stochastic moments in numpy, then evaluate at Chebyshev
        # nodes in ρ-space and fit a Chebyshev-in-ρ polynomial for JAX
        # Clenshaw evaluation (differentiable, JIT-compatible).
        from ._factories import _cheb_stochastic_coeffs

        coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max)

        def _jax_cheb_stochastic(rho):
            val = jax_logdet_chebyshev(rho, coeffs, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _jax_cheb_stochastic

    if method == "slq":
        if eigs is not None:
            raise ValueError(
                "SLQ requires the weight matrix W, not a 1-D eigenvalue array."
            )
        # Consume the numpy sparse SLQ rules: batched D-symmetrised Lanczos
        # (real nodes θ and real n·v₁² weights) for undirected W, or Arnoldi
        # (complex Ritz values θ and complex bilinear weights γ) for directed
        # W.  Evaluate the ρ-dependent quadrature in JAX via the complex log;
        # the Lanczos case is the zero-imaginary special case of the same
        # formula.  No dense W is materialised — the precompute is matvec-only.
        pre = slq_logdet_precompute(W_sparse)
        nodes = np.asarray(pre.nodes)
        weights = np.asarray(pre.weights)
        n_probes = pre.n_probes

        nodes_real = np.ascontiguousarray(nodes.real.astype(np.float64))
        nodes_imag = np.ascontiguousarray(nodes.imag.astype(np.float64))
        w_real = np.ascontiguousarray(weights.real.astype(np.float64))
        w_imag = np.ascontiguousarray(weights.imag.astype(np.float64))

        def _jax_slq(rho):
            import jax.numpy as jnp

            nr = jnp.asarray(nodes_real)
            ni = jnp.asarray(nodes_imag)
            wr = jnp.asarray(w_real)
            wi = jnp.asarray(w_imag)
            # 1 - ρθ = (1 - ρ·Re θ) + i(-ρ·Im θ)
            # log(1 - ρθ) = 0.5·log|1-ρθ|² + i·atan2(Im, Re)
            re = 1.0 - rho * nr
            im = -rho * ni
            log_re = 0.5 * jnp.log(jnp.maximum(re**2 + im**2, 1e-300))
            log_im = jnp.arctan2(im, re)
            # Re(Σ γ·log(1-ρθ)) = Σ [Re(γ)·log_re - Im(γ)·log_im]; the second
            # term vanishes for real (Lanczos) weights but keeps the cross term
            # that a magnitude-only log would drop for the complex Arnoldi case.
            val = jnp.sum(wr * log_re - wi * log_im) / n_probes
            return val if T == 1 else T * val

        return _jax_slq

    if method == "cheb_cholesky":
        from ._chol_cheb import chol_cheb_logdet_precompute

        # Precompute Chebyshev coefficients via sparse Cholesky in numpy,
        # then evaluate via JAX-native Clenshaw (differentiable, JIT-compatible).
        pre = chol_cheb_logdet_precompute(
            W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = pre.coeffs.astype(np.float64)
        rmin_cb = float(pre.rho_min)
        rmax_cb = float(pre.rho_max)

        def _jax_cheb_chol(rho):
            val = jax_logdet_chebyshev(rho, coeffs, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _jax_cheb_chol

    if method == "aaa":
        from ._aaa import aaa_logdet_precompute

        # Precompute support points and barycentric weights via sparse LU
        # in numpy, then evaluate via JAX-native barycentric formula
        # (differentiable, JIT-compatible).
        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        sp_z = pre.support_points.astype(np.float64)
        sp_f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        def _jax_aaa(rho):
            import jax.numpy as jnp

            z_j = jnp.asarray(sp_z)
            f_j = jnp.asarray(sp_f)
            w_j = jnp.asarray(w)

            diff = rho - z_j
            n_val = jnp.sum(w_j * f_j / diff)
            d_val = jnp.sum(w_j / diff)
            val = n_val / d_val
            return val if T == 1 else T * val

        return _jax_aaa

    if method == "cholmod":
        # JAX-native exact logdet via cholgraph sparse CHOLMOD.
        # Requires W to be D-symmetrizable (row-standardised undirected
        # graph): W = D⁻¹A with symmetric A → W_sym = D^{1/2} W D^{-1/2}
        # is symmetric with the same eigenvalues, so I−ρW_sym is SPD
        # for |ρ| < 1 and cholgraph.logdet applies directly.
        # If W is not D-symmetrizable (directed graph), this raises
        # ValueError — use logdet_method="aaa" for such matrices.
        from .._jax_dispatch import _cholgraph_available

        if not _cholgraph_available():
            raise ImportError(
                "logdet method 'cholmod' requires the 'cholgraph' package. "
                "Install it with: pip install cholgraph"
            )

        from ._chol_cheb import _d_symmetrize

        # D-symmetrise: raises ValueError if W is not symmetrizable.
        W_sym_sp = _d_symmetrize(W_sparse)  # csc_matrix, symmetric

        # Build the COO pattern for I − ρW_sym.
        # cholgraph reads only Ai <= Aj entries (upper triangle),
        # so we include the upper triangle of W_sym plus all diagonal
        # entries (for the I in I − ρW_sym, since W_sym has zero diagonal
        # for graphs without self-loops).
        W_sym_coo = W_sym_sp.tocoo()
        mask_upper = W_sym_coo.row <= W_sym_coo.col
        upper_rows = W_sym_coo.row[mask_upper]
        upper_cols = W_sym_coo.col[mask_upper]
        upper_vals = W_sym_coo.data[mask_upper]

        # Add diagonal entries that are missing from W_sym's pattern
        existing_diag = set(zip(upper_rows.tolist(), upper_cols.tolist()))
        diag_rows = []
        diag_cols = []
        for i in range(n):
            if (i, i) not in existing_diag:
                diag_rows.append(i)
                diag_cols.append(i)

        _Ai_direct = np.concatenate(
            [
                upper_rows.astype(np.int32),
                np.array(diag_rows, dtype=np.int32),
            ]
        )
        _Aj_direct = np.concatenate(
            [
                upper_cols.astype(np.int32),
                np.array(diag_cols, dtype=np.int32),
            ]
        )
        # W_sym values at these positions (0 for added diagonal entries)
        _W_sym_vals = np.concatenate(
            [
                upper_vals.astype(np.float64),
                np.zeros(len(diag_rows), dtype=np.float64),
            ]
        )
        _nnz_direct = len(_Ai_direct)
        _n_static = n

        # Diagonal indices for I − ρW_sym
        _diag_idx_direct = np.full(n, -1, dtype=np.int32)
        for k_idx in range(_nnz_direct):
            if _Ai_direct[k_idx] == _Aj_direct[k_idx]:
                _diag_idx_direct[_Ai_direct[k_idx]] = k_idx

        def _jax_cholmod(rho):
            import cholgraph
            import jax.numpy as jnp

            Ai = jnp.asarray(_Ai_direct, dtype=jnp.int32)
            Aj = jnp.asarray(_Aj_direct, dtype=jnp.int32)
            W_vals = jnp.asarray(_W_sym_vals, dtype=jnp.float64)
            diag_idx = jnp.asarray(_diag_idx_direct, dtype=jnp.int32)

            # Ax = I − ρW_sym at pattern positions
            Ax = -rho * W_vals
            diag_vals = jnp.zeros(_nnz_direct, dtype=jnp.float64)
            diag_vals = diag_vals.at[diag_idx].set(1.0)
            Ax = Ax + diag_vals

            val = cholgraph.logdet(Ai, Aj, Ax, _n_static)
            return val if T == 1 else T * val

        return _jax_cholmod

    raise ValueError(
        f"Method '{method}' has no JAX implementation. "
        "Use 'eigenvalue', 'chebyshev', 'cheb_stochastic', "
        "'cheb_cholesky', 'aaa', 'cholmod', or 'slq'."
    )
