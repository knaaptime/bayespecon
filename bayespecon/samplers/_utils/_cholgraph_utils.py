"""Shared helpers for cholgraph integration in JAX Gibbs samplers.

Provides the COO sparsity-pattern precomputation and value-assembly
utilities needed to use :mod:`cholgraph` (JAX-native sparse CHOLMOD)
inside JIT-compiled Gibbs steps.

The precision matrix

.. math::

    P = I + \\mathrm{diag}(\\omega) - \\rho (W + W^T) + \\rho^2 W^T W

is symmetric positive definite for any valid ``ρ`` and ``ω ≥ 0``.
Its sparsity pattern is **fixed** (independent of ``ρ`` and ``ω``),
so we precompute the COO indices ``(Ai, Aj)`` once on the host and
assemble only the values ``Ax(ρ, ω)`` inside the JIT boundary.

This mirrors the NumPy-CHOLMOD pattern in
:func:`bayespecon.samplers.negbin_reduced._core._make_cholmod_pattern`
but returns int32 COO arrays suitable for ``cholgraph``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def precompute_cholgraph_pattern(
    W_csc: sp.csc_matrix,
    n: int,
) -> dict:
    """Precompute the fixed COO sparsity pattern for the precision matrix.

    The pattern covers all fill-in positions of
    ``P = I + diag(ω) − ρ(W+Wᵀ) + ρ²WᵀW`` for any valid ``ρ`` and ``ω ≥ 0``.
    Built as ``I + 0.5*(W+Wᵀ) + 0.25*WᵀW`` so every possible nonzero is present.

    Parameters
    ----------
    W_csc : scipy.sparse.csc_matrix
        The **raw** (row-standardised) spatial weights matrix ``W`` in CSC
        format — *not* ``W+Wᵀ``.  This function derives ``W+Wᵀ`` and ``WᵀW``
        from it internally; passing an already-symmetrised matrix would double
        the symmetric part and corrupt ``WᵀW``.
    n : int
        Matrix dimension.

    Returns
    -------
    dict with keys:
        ``Ai`` : np.ndarray, shape (nnz,), dtype int32 — COO row indices.
        ``Aj`` : np.ndarray, shape (nnz,), dtype int32 — COO column indices.
        ``W_sym_vals`` : np.ndarray, shape (nnz,), dtype float64 —
            Values of ``W + Wᵀ`` at the pattern positions (0 where pattern
            has entries but W+Wᵀ does not).
        ``WtW_vals`` : np.ndarray, shape (nnz,), dtype float64 —
            Values of ``WᵀW`` at the pattern positions.
        ``is_diag`` : np.ndarray, shape (nnz,), dtype bool —
            Boolean mask for diagonal entries (``Ai == Aj``).
        ``diag_idx`` : np.ndarray, shape (n,), dtype int32 —
            Indices into the pattern arrays where the diagonal entries live.
            Used to scatter ``1 + ω`` into ``Ax``.
        ``n`` : int — Matrix dimension.
    """
    W_sym = (W_csc + W_csc.T).tocsc()
    WtW = (W_csc.T @ W_csc).tocsc()
    pattern = (sp.eye(n, format="csc") + 0.5 * W_sym + 0.25 * WtW).tocoo()

    Ai = pattern.row.astype(np.int32)
    Aj = pattern.col.astype(np.int32)
    nnz = len(Ai)

    # Extract W_sym and WtW values at the pattern positions.
    # We do this by converting to COO and aligning with the pattern.
    W_sym_coo = W_sym.tocoo()
    WtW_coo = WtW.tocoo()

    # Build a lookup from (row, col) → index in the pattern array.
    # The pattern is symmetric (upper triangle), so we only need to
    # match entries with Ai <= Aj (lower triangle entries are ignored
    # by cholgraph).
    pattern_lookup: dict[tuple[int, int], int] = {}
    for k in range(nnz):
        pattern_lookup[(int(Ai[k]), int(Aj[k]))] = k

    W_sym_vals = np.zeros(nnz, dtype=np.float64)
    WtW_vals = np.zeros(nnz, dtype=np.float64)

    for k in range(len(W_sym_coo.row)):
        i, j = int(W_sym_coo.row[k]), int(W_sym_coo.col[k])
        # cholgraph reads only upper triangle (Ai <= Aj)
        if i <= j:
            idx = pattern_lookup.get((i, j))
            if idx is not None:
                W_sym_vals[idx] = W_sym_coo.data[k]

    for k in range(len(WtW_coo.row)):
        i, j = int(WtW_coo.row[k]), int(WtW_coo.col[k])
        if i <= j:
            idx = pattern_lookup.get((i, j))
            if idx is not None:
                WtW_vals[idx] = WtW_coo.data[k]

    is_diag = Ai == Aj
    # For each diagonal position i, find its index in the pattern.
    diag_idx = np.full(n, -1, dtype=np.int32)
    for k in range(nnz):
        if is_diag[k]:
            diag_idx[Ai[k]] = k

    return {
        "Ai": Ai,
        "Aj": Aj,
        "W_sym_vals": W_sym_vals,
        "WtW_vals": WtW_vals,
        "is_diag": is_diag,
        "diag_idx": diag_idx,
        "n": n,
    }


def make_cholgraph_ops(Ai, Aj, n: int):
    """Return ``(eta_sample, solve_logdet)`` factor-once closures over a fixed pattern.

    Both do **one** numeric factorization per call (matching numpy's
    ``CholmodFactor`` reuse), using cholgraph 0.4's factor-once primitives when
    available and falling back to the 0.3 idiom otherwise:

    - ``eta_sample(Ax, mean_term, z) -> N(P⁻¹ mean_term, P⁻¹)`` draw — 0.4:
      :func:`cholgraph.sample_gaussian` (one factorization); 0.3: mean solve +
      ``MODE_LT`` + ``MODE_PT`` (three solves ≈ three factorizations under vmap).
    - ``solve_logdet(Ax, b) -> (P⁻¹ b, log|P|)`` — 0.4:
      :func:`cholgraph.factor_solve` with ``want_logdet=True`` (one factorization,
      no working-copy); 0.3: :func:`cholgraph.update_solve` with a zero update
      column and ``return_logdet=True``.

    ``Ai``/``Aj`` are the fixed COO indices (int32); ``b`` may be ``(n,)`` or
    ``(n, n_rhs)``.
    """
    import cholgraph as _chj
    import jax.numpy as jnp

    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)

    if hasattr(_chj, "sample_gaussian"):  # cholgraph >= 0.4
        _MODE_A = getattr(_chj, "MODE_A", 0)

        def eta_sample(Ax, mean_term, z):
            eta, _mean = _chj.sample_gaussian(Ai, Aj, Ax, mean_term, z)
            return eta

        def solve_logdet(Ax, b):
            sols, ld = _chj.factor_solve(Ai, Aj, Ax, [(b, _MODE_A)], want_logdet=True)
            return sols[0], ld

    else:  # cholgraph 0.3 fallback
        _Czero = jnp.zeros((n, 1), dtype=jnp.float64)
        _MODE_LT, _MODE_PT = _chj.MODE_LT, _chj.MODE_PT

        def eta_sample(Ax, mean_term, z):
            m = _chj.solve(Ai, Aj, Ax, mean_term)
            w = _chj.solve(Ai, Aj, Ax, z, mode=_MODE_LT)
            w = _chj.solve(Ai, Aj, Ax, w, mode=_MODE_PT)
            return m + w

        def solve_logdet(Ax, b):
            x, ld = _chj.update_solve(Ai, Aj, Ax, _Czero, b, return_logdet=True)
            return x, ld

    return eta_sample, solve_logdet


def resolve_pg_jax_backend(backend, *, W_sparse, W_sym, WtW, n, logdet_bounds):
    """Resolve the PG-Gibbs backend method and its JAX precomputes.

    Shared by the SAR-logit / SEM-logit / structural SAR-NB Gibbs fits, which
    previously each carried this ~40-line block verbatim.

    Parameters
    ----------
    backend : {"jax", "numpy"}
        Resolved execution backend.
    W_sparse, W_sym, WtW : scipy.sparse matrices
        Raw row-standardised ``W``, ``W + Wᵀ`` and ``WᵀW``.
    n : int
        Number of observations.
    logdet_bounds : LogdetBounds
        The model's resolved logdet bounds (method, rho_min, rho_max).

    Returns
    -------
    method : str
        One of ``"cholmod"`` (numpy), ``"jax_dense"``, ``"cholmod_jax"`` —
        used for all three of the cache's solve/logdet_P/sample methods.
    jax_parts : dict
        ``W_sym_dense``, ``WtW_dense``, ``logdet_jax``, ``cholgraph_pattern``
        (all ``None`` on the numpy path).
    """
    jax_parts = {
        "W_sym_dense": None,
        "WtW_dense": None,
        "logdet_jax": None,
        "cholgraph_pattern": None,
    }
    if backend != "jax":
        return "cholmod", jax_parts

    from bayespecon._jax_dispatch import (
        _cholgraph_available,
        _cholmod_jax_enabled,
        ensure_x64,
    )

    method = (
        "cholmod_jax"
        if _cholmod_jax_enabled() and _cholgraph_available()
        else "jax_dense"
    )

    import jax.numpy as jnp

    ensure_x64()

    # Only the dense-Cholesky fallback needs the dense (W+Wᵀ) and WᵀW; the
    # cholmod_jax path assembles P from the sparse COO pattern and does its
    # matvecs via BCOO, so we never densify W there.
    if method == "jax_dense":
        jax_parts["W_sym_dense"] = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        jax_parts["WtW_dense"] = jnp.asarray(WtW.toarray(), dtype=jnp.float64)

    from bayespecon._logdet import make_logdet_jax_fn

    jax_parts["logdet_jax"] = make_logdet_jax_fn(
        W_sparse,
        method=logdet_bounds.method,
        rho_min=logdet_bounds.rho_min,
        rho_max=logdet_bounds.rho_max,
    )

    if method == "cholmod_jax":
        # Pass the raw (row-standardised) W; the helper derives W+Wᵀ and WᵀW
        # internally.  Passing W_sym here would double the symmetric part and
        # corrupt WᵀW.
        jax_parts["cholgraph_pattern"] = precompute_cholgraph_pattern(
            W_sparse.tocsc(), n
        )

    return method, jax_parts
