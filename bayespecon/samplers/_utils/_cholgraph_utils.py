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


def assemble_Ax_logit(
    omega: "jnp.ndarray",  # noqa: F821
    rho: "jnp.ndarray",  # noqa: F821
    pattern: dict,
) -> "jnp.ndarray":  # noqa: F821
    """Assemble the COO values ``Ax(ρ, ω)`` for the logit precision matrix.

    For logit models (σ² = 1):

    .. math::

        P = I + \\mathrm{diag}(\\omega) - \\rho (W + W^T) + \\rho^2 W^T W

    Parameters
    ----------
    omega : jax.numpy.ndarray, shape (n,)
        PG auxiliary variables.
    rho : jax.numpy.ndarray (scalar)
        Spatial autoregressive parameter.
    pattern : dict
        Output of :func:`precompute_cholgraph_pattern`.

    Returns
    -------
    jax.numpy.ndarray, shape (nnz,)
        COO values for the precision matrix at ``(ρ, ω)``.
    """
    import jax.numpy as jnp

    W_sym_vals = jnp.asarray(pattern["W_sym_vals"], dtype=jnp.float64)
    WtW_vals = jnp.asarray(pattern["WtW_vals"], dtype=jnp.float64)
    diag_idx = jnp.asarray(pattern["diag_idx"], dtype=jnp.int32)
    nnz = len(pattern["Ai"])

    # Start with the ρ-dependent off-diagonal part.
    Ax = -rho * W_sym_vals + rho**2 * WtW_vals

    # Add the diagonal: 1 + ω_i at diagonal positions.
    diag_vals = jnp.zeros(nnz, dtype=jnp.float64)
    diag_vals = diag_vals.at[diag_idx].set(1.0 + omega)
    Ax = Ax + diag_vals

    return Ax


def assemble_Ax_negbin(
    omega: "jnp.ndarray",  # noqa: F821
    rho: "jnp.ndarray",  # noqa: F821
    sigma2: "jnp.ndarray",  # noqa: F821
    pattern: dict,
) -> "jnp.ndarray":  # noqa: F821
    """Assemble the COO values ``Ax(ρ, ω, σ²)`` for the negbin precision matrix.

    For negative-binomial models:

    .. math::

        P = I/\\sigma^2 + \\mathrm{diag}(\\omega)
            - (\\rho/\\sigma^2)(W + W^T) + (\\rho^2/\\sigma^2) W^T W

    Parameters
    ----------
    omega : jax.numpy.ndarray, shape (n,)
        PG auxiliary variables.
    rho : jax.numpy.ndarray (scalar)
        Spatial autoregressive parameter.
    sigma2 : jax.numpy.ndarray (scalar)
        Residual variance.
    pattern : dict
        Output of :func:`precompute_cholgraph_pattern`.

    Returns
    -------
    jax.numpy.ndarray, shape (nnz,)
        COO values for the precision matrix at ``(ρ, ω, σ²)``.
    """
    import jax.numpy as jnp

    W_sym_vals = jnp.asarray(pattern["W_sym_vals"], dtype=jnp.float64)
    WtW_vals = jnp.asarray(pattern["WtW_vals"], dtype=jnp.float64)
    diag_idx = jnp.asarray(pattern["diag_idx"], dtype=jnp.int32)
    nnz = len(pattern["Ai"])
    inv_s2 = 1.0 / sigma2

    # Off-diagonal: −(ρ/σ²)(W+Wᵀ) + (ρ²/σ²)WᵀW
    Ax = -rho * W_sym_vals * inv_s2 + rho**2 * WtW_vals * inv_s2

    # Diagonal: 1/σ² + ω_i
    diag_vals = jnp.zeros(nnz, dtype=jnp.float64)
    diag_vals = diag_vals.at[diag_idx].set(inv_s2 + omega)
    Ax = Ax + diag_vals

    return Ax


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


def cholgraph_mvn_sample(Ai, Aj, Ax, mean_term, key, n: int):
    """Draw from ``N(P⁻¹ mean_term, P⁻¹)`` (thin wrapper over :func:`make_cholgraph_ops`).

    Kept for callers/tests; delegates to the factor-once ``eta_sample`` closure so
    the draw costs a single factorization on cholgraph >= 0.4.
    """
    import jax
    import jax.numpy as jnp

    z = jax.random.normal(key, shape=(n,), dtype=jnp.float64)
    eta_sample, _ = make_cholgraph_ops(Ai, Aj, n)
    return eta_sample(Ax, mean_term, z)
