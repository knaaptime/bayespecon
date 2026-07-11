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

from ._cheb_stochastic import (
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_precompute,
)
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

    Supports ``"eigenvalue"``, ``"chebyshev"``, ``"cheb_stochastic"``, and
    ``"slq"``.  SLQ uses JAX-native Arnoldi iteration via ``jax.lax.scan``.
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
        pre = cheb_stochastic_logdet_precompute(W_sparse)
        # Evaluate at 20 Chebyshev nodes in [rho_min, rho_max]
        _k_nodes = np.arange(1, 21)
        _nodes_cos = np.cos((2 * _k_nodes - 1) * np.pi / 40)
        _rho_nodes = 0.5 * (rho_max - rho_min) * _nodes_cos + 0.5 * (rho_max + rho_min)
        _logdet_vals = np.array(
            [cheb_stochastic_logdet_eval(pre, float(r)) for r in _rho_nodes]
        )
        # DCT-I → Chebyshev coefficients in ρ
        coeffs = np.zeros(20, dtype=np.float64)
        for j in range(20):
            scale = 2.0 / 20 if j > 0 else 1.0 / 20
            coeffs[j] = scale * np.sum(
                _logdet_vals * np.cos(j * (2 * _k_nodes - 1) * np.pi / 40)
            )
        rmin_cb = float(rho_min)
        rmax_cb = float(rho_max)

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

    raise ValueError(
        f"Method '{method}' has no JAX implementation. "
        "Use 'eigenvalue', 'chebyshev', 'cheb_stochastic', "
        "'cheb_cholesky', 'aaa', or 'slq'."
    )
