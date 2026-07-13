r"""Resolvent–Kronecker gradient of the unrestricted flow log-determinant.

For the unrestricted 3-parameter flow model the system matrix is

.. math::

    W_F = \rho_d\,(I_n \otimes W) + \rho_o\,(W \otimes I_n) + \rho_w\,(W \otimes W)

an :math:`N \times N` operator with :math:`N = n^2`.  Sampling the flow
:math:`\rho`'s with a gradient-based sampler (NUTS/MALA) needs

.. math::

    g_k \;=\; \frac{\partial}{\partial \rho_k}\,\log|I_N - W_F|
        \;=\; -\,\operatorname{tr}\!\big(W_k (I_N - W_F)^{-1}\big),
    \qquad W_k \in \{I\otimes W,\; W\otimes I,\; W\otimes W\}.

Unlike the log-determinant *value* — which the ``"traces"`` method estimates from
spectral moments and which is catastrophically noise-amplified at scale for large
directed ``W`` (the multinomial coefficients blow up Hutchinson error) — each
gradient component is a **single** resolvent trace.  Its Hutchinson relative error
scales like :math:`1/\sqrt{N P}` (``P`` probes), so it *improves* with the flow
sample size ``N``.  Everything here is **matvec-only** (no eigendecomposition,
directed-``W`` friendly): the Kronecker structure lets

.. math::

    W_F\,\operatorname{vec}(X)
      = \operatorname{vec}\!\big(\rho_d\,WX + \rho_o\,XW^\top + \rho_w\,WXW^\top\big)

be applied in :math:`O(n\cdot\mathrm{nnz})`, and the resolvent solve
:math:`(I_N - W_F)^{-\top} z` is done with GMRES on that operator.

This is the scalable, eigenvalue-free path for the unrestricted flow logdet; see
``resolvent_paper`` and the plan notes for the validation (relative error falling
as ``N`` grows).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "FlowKron",
    "flow_logdet_grad",
    "flow_logdet_grad_exact",
    "flow_logdet_value",
    "flow_logdet_value_and_grad",
]


def _as_csr(W) -> sp.csr_matrix:
    if sp.issparse(W):
        return W.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(W, dtype=np.float64))


class FlowKron:
    """Kronecker matvecs for the flow system built from an ``n x n`` weights ``W``.

    Holds ``W`` (CSR) and ``Wᵀ`` (CSR) once and applies the flow operators to a
    length-``N=n²`` vector by reshaping to ``n x n`` and using sparse–dense
    products.  All operators are :math:`O(n\\cdot\\mathrm{nnz}(W))`.
    """

    def __init__(self, W):
        self.W = _as_csr(W)
        self.Wt = self.W.T.tocsr()
        self.n = int(self.W.shape[0])
        self.N = self.n * self.n

    # -- W_F and its transpose --------------------------------------------
    # NumPy's ``reshape`` is row-major, for which ``(A ⊗ B) vec(X) = vec(A X Bᵀ)``.
    # Hence I⊗W ↦ X Wᵀ, W⊗I ↦ W X, W⊗W ↦ W X Wᵀ.
    def matvec_WF(self, x, rho_d, rho_o, rho_w):
        r"""``W_F x`` = vec(ρ_d X Wᵀ + ρ_o W X + ρ_w W X Wᵀ)."""
        X = x.reshape(self.n, self.n)
        WX = self.W @ X
        Y = rho_d * (X @ self.Wt) + rho_o * WX + rho_w * (WX @ self.Wt)
        return Y.ravel()

    def matvec_WF_T(self, x, rho_d, rho_o, rho_w):
        r"""``W_Fᵀ x`` = vec(ρ_d X W + ρ_o Wᵀ X + ρ_w Wᵀ X W)."""
        X = x.reshape(self.n, self.n)
        WtX = self.Wt @ X
        Y = rho_d * (X @ self.W) + rho_o * WtX + rho_w * (WtX @ self.W)
        return Y.ravel()

    # -- individual W_k (for the trace contractions) ----------------------
    def matvec_Wd(self, x):
        """``(I ⊗ W) x`` = vec(X Wᵀ)."""
        return (x.reshape(self.n, self.n) @ self.Wt).ravel()

    def matvec_Wo(self, x):
        """``(W ⊗ I) x`` = vec(W X)."""
        return (self.W @ x.reshape(self.n, self.n)).ravel()

    def matvec_Ww(self, x):
        """``(W ⊗ W) x`` = vec(W X Wᵀ)."""
        return (self.W @ (x.reshape(self.n, self.n) @ self.Wt)).ravel()

    def resolvent_T_operator(self, rho_d, rho_o, rho_w):
        """``LinearOperator`` for ``(I_N − W_Fᵀ)`` at the given ρ."""
        return spla.LinearOperator(
            (self.N, self.N),
            matvec=lambda x: x - self.matvec_WF_T(x, rho_d, rho_o, rho_w),
            dtype=np.float64,
        )

    def resolvent_T_sparse(self, rho_d, rho_o, rho_w):
        """Sparse CSC matrix for ``(I_N − W_Fᵀ)`` at the given ρ.

        ``W_Fᵀ = ρ_d(I⊗Wᵀ) + ρ_o(Wᵀ⊗I) + ρ_w(Wᵀ⊗Wᵀ)`` which, using
        ``kron(A, B)`` on CSR matrices, is a sparse N×N matrix with
        ``O(n²·nnz(W))`` nonzeros.  Used for KLU factorisation when
        the dense N×N system is too large but the sparse pattern is
        manageable.
        """
        I_n = sp.eye(self.n, format="csr")
        WF_T = (
            rho_d * sp.kron(I_n, self.Wt, format="csr")
            + rho_o * sp.kron(self.Wt, I_n, format="csr")
            + rho_w * sp.kron(self.Wt, self.Wt, format="csr")
        )
        return (sp.eye(self.N, format="csr") - WF_T).tocsc()

    def _resolvent_T_pattern(self):
        """Cached COO sparsity pattern + value decomposition for ``(I_N − W_Fᵀ)``.

        The pattern of ``I_N − W_Fᵀ`` is **fixed** regardless of ρ — only the
        numerical values change.  This decomposes the matrix values into:

        * ``rows``, ``cols`` — COO indices (int32, for klujax)
        * ``const_vals`` — values from the identity block (always 1.0)
        * ``coef_d``, ``coef_o``, ``coef_w`` — per-entry coefficients so that
          ``Ax = const_vals - ρ_d·coef_d - ρ_o·coef_o - ρ_w·coef_w``

        Computed once and cached so that klujax's symbolic analysis
        (``analyze``) can be reused across all ρ evaluations.
        """
        if hasattr(self, "_pattern_cache"):
            return self._pattern_cache
        I_n = sp.eye(self.n, format="csr")
        Wd_kron = sp.kron(I_n, self.Wt, format="coo")  # I⊗Wᵀ
        Wo_kron = sp.kron(self.Wt, I_n, format="coo")  # Wᵀ⊗I
        Ww_kron = sp.kron(self.Wt, self.Wt, format="coo")  # Wᵀ⊗Wᵀ
        I_coo = sp.eye(self.N, format="coo")

        # Build combined COO with all entries, then sum duplicates
        all_rows = np.concatenate(
            [I_coo.row, Wd_kron.row, Wo_kron.row, Ww_kron.row]
        )
        all_cols = np.concatenate(
            [I_coo.col, Wd_kron.col, Wo_kron.col, Ww_kron.col]
        )
        shape = (self.N, self.N)

        def _sum_dups(data):
            m = sp.coo_matrix((data, (all_rows, all_cols)), shape=shape)
            m.sum_duplicates()
            return m

        const_mat = _sum_dups(
            np.concatenate([
                np.ones(I_coo.nnz),
                np.zeros(Wd_kron.nnz),
                np.zeros(Wo_kron.nnz),
                np.zeros(Ww_kron.nnz),
            ])
        )
        d_mat = _sum_dups(
            np.concatenate([
                np.zeros(I_coo.nnz),
                Wd_kron.data,
                np.zeros(Wo_kron.nnz),
                np.zeros(Ww_kron.nnz),
            ])
        )
        o_mat = _sum_dups(
            np.concatenate([
                np.zeros(I_coo.nnz),
                np.zeros(Wd_kron.nnz),
                Wo_kron.data,
                np.zeros(Ww_kron.nnz),
            ])
        )
        w_mat = _sum_dups(
            np.concatenate([
                np.zeros(I_coo.nnz),
                np.zeros(Wd_kron.nnz),
                np.zeros(Wo_kron.nnz),
                Ww_kron.data,
            ])
        )

        rows = np.asarray(const_mat.row, dtype=np.int32)
        cols = np.asarray(const_mat.col, dtype=np.int32)
        const_vals = np.asarray(const_mat.data, dtype=np.float64)
        coef_d = np.asarray(d_mat.data, dtype=np.float64)
        coef_o = np.asarray(o_mat.data, dtype=np.float64)
        coef_w = np.asarray(w_mat.data, dtype=np.float64)
        self._pattern_cache = (rows, cols, const_vals, coef_d, coef_o, coef_w)
        return self._pattern_cache

    def klujax_symbolic(self):
        """One-time klujax symbolic analysis for ``(I_N − W_Fᵀ)``.

        Returns ``(Ai, Aj, symbolic)`` cached on the instance.  The sparsity
        pattern is fixed for all ρ, so ``analyze`` is called once and reused
        via ``klujax.solve_with_symbol``.
        """
        if hasattr(self, "_klujax_cache"):
            return self._klujax_cache
        import klujax

        rows, cols, *_ = self._resolvent_T_pattern()
        symbolic = klujax.analyze(rows, cols, self.N)
        self._klujax_cache = (rows, cols, symbolic)
        return self._klujax_cache


def flow_logdet_grad(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    n_probes: int = 32,
    rng=None,
    probes: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
    return_probes: bool = False,
):
    r"""Stochastic estimate of ``∇_ρ log|I_N − W_F|`` via Kronecker Krylov solves.

    Returns ``g = (g_d, g_o, g_w)`` where
    ``g_k = −tr(W_k (I_N − W_F)^{-1})``.  For each Rademacher probe ``z`` it solves
    the single system ``x̃ = (I_N − W_Fᵀ)^{-1} z`` (GMRES; directed-``W`` safe) and
    accumulates ``x̃ᵀ (W_k z)`` for all three ``k`` — so one solve per probe yields
    the whole gradient.

    Parameters
    ----------
    W : array or sparse
        The ``n x n`` (row-standardised, possibly directed) weights matrix.
    rho_d, rho_o, rho_w : float
        Flow spatial parameters.
    n_probes : int
        Number of Hutchinson probes (ignored if ``probes`` is given).
    rng : numpy Generator, optional
        Randomness source for fresh probes.
    probes : ndarray (N, P), optional
        **Frozen** probe matrix reused across ρ (columns are ±1 vectors).  Giving
        the same ``probes`` at every call yields a smooth, consistent gradient
        field suitable for MALA / pseudo-marginal correction.
    tol, maxiter : float, int
        GMRES ``rtol`` and iteration cap.
    return_probes : bool
        Also return the probe matrix used (handy for freezing on the first call).

    Returns
    -------
    grad : ndarray (3,)
        ``(g_d, g_o, g_w)``.
    probes : ndarray (N, P), optional
        Only if ``return_probes``.
    """
    kron = W if isinstance(W, FlowKron) else FlowKron(W)
    N = kron.N

    if probes is None:
        rng = np.random.default_rng() if rng is None else rng
        probes = rng.choice([-1.0, 1.0], size=(N, int(n_probes))).astype(np.float64)
    else:
        probes = np.asarray(probes, dtype=np.float64)
        if probes.ndim == 1:
            probes = probes[:, None]
        if probes.shape[0] != N:
            raise ValueError(f"probes must have {N} rows; got {probes.shape[0]}.")

    op = kron.resolvent_T_operator(rho_d, rho_o, rho_w)
    P = probes.shape[1]
    acc = np.zeros(3, dtype=np.float64)

    # Solver priority: klujax (JAX-native, batched) > KLU/UMFPACK > GMRES
    from bayespecon._jax_dispatch import _klujax_available
    from bayespecon._ops._backend import _select_sparse_backend, _sparse_factor

    if _klujax_available():
        import klujax

        rows, cols, const_vals, coef_d, coef_o, coef_w = kron._resolvent_T_pattern()
        symbolic = kron.klujax_symbolic()[2]
        # Assemble numeric values: Ax = const - ρ_d·coef_d - ρ_o·coef_o - ρ_w·coef_w
        Ax = const_vals - rho_d * coef_d - rho_o * coef_o - rho_w * coef_w
        # Batched solve: all P probes in one call (N×P RHS)
        Xt = np.asarray(
            klujax.solve_with_symbol(rows, cols, Ax, probes, symbolic),
            dtype=np.float64,
        )  # (N, P)
        for p in range(P):
            z = probes[:, p]
            xt = Xt[:, p]
            acc[0] += xt @ kron.matvec_Wd(z)
            acc[1] += xt @ kron.matvec_Wo(z)
            acc[2] += xt @ kron.matvec_Ww(z)
    else:
        backend = _select_sparse_backend()
        if backend in ("klu", "umfpack"):
            A_csc = kron.resolvent_T_sparse(rho_d, rho_o, rho_w)
            factor = _sparse_factor(A_csc, backend)
            for p in range(P):
                z = probes[:, p]
                xt = np.asarray(factor.solve(z), dtype=np.float64)
                acc[0] += xt @ kron.matvec_Wd(z)
                acc[1] += xt @ kron.matvec_Wo(z)
                acc[2] += xt @ kron.matvec_Ww(z)
        else:
            for p in range(P):
                z = probes[:, p]
                xt, _info = spla.lgmres(op, z, rtol=tol, atol=0.0, maxiter=maxiter)
                acc[0] += xt @ kron.matvec_Wd(z)
                acc[1] += xt @ kron.matvec_Wo(z)
                acc[2] += xt @ kron.matvec_Ww(z)
    grad = -acc / P

    if return_probes:
        return grad, probes
    return grad


def flow_logdet_value(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    n_probes: int = 32,
    n_quad: int = 8,
    rng=None,
    probes: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
):
    r"""Estimate ``log|I_N − W_F(ρ)|`` by integrating the gradient along ``0→ρ``.

    Since ``W_F(tρ) = t\,W_F(ρ)`` is linear in ``ρ``,

    .. math::

        \log|I_N - W_F(\rho)|
          = \int_0^1 \frac{d}{dt}\log|I_N - W_F(t\rho)|\,dt
          = \int_0^1 \sum_k \rho_k\,g_k(t\rho)\,dt ,

    a 1-D integral of the (scalable, resolvent-estimated) gradient.  With
    ``n_quad`` Gauss–Legendre nodes this is exact given exact gradients, and
    inherits the gradient's :math:`1/\sqrt{N P}` accuracy otherwise — so, unlike
    the moment-based ``"traces"`` value, it *improves* with the flow sample size.

    A single **frozen** probe matrix is shared across all quadrature nodes
    (generated once here if not supplied) so the value is a smooth, consistent
    function of ``ρ`` — suitable for a Metropolis correction.
    """
    kron = W if isinstance(W, FlowKron) else FlowKron(W)
    rho = np.array([rho_d, rho_o, rho_w], dtype=np.float64)

    if probes is None:
        rng = np.random.default_rng() if rng is None else rng
        probes = rng.choice([-1.0, 1.0], size=(kron.N, int(n_probes))).astype(
            np.float64
        )

    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    nodes = 0.5 * (nodes + 1.0)  # map [-1,1] -> [0,1]
    weights = 0.5 * weights

    val = 0.0
    for t, wq in zip(nodes, weights):
        g = flow_logdet_grad(
            kron,
            t * rho_d,
            t * rho_o,
            t * rho_w,
            probes=probes,
            tol=tol,
            maxiter=maxiter,
        )
        val += wq * float(rho @ g)
    return val


def flow_logdet_value_and_grad(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    n_probes: int = 32,
    n_quad: int = 8,
    rng=None,
    probes: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
):
    """Return ``(value, grad)`` = ``(log|I_N−W_F|, ∇_ρ log|I_N−W_F|)``.

    The gradient is evaluated at ``ρ`` (the ``t=1`` endpoint) and the value by
    ray integration; both share the same frozen probes for consistency.
    """
    kron = W if isinstance(W, FlowKron) else FlowKron(W)
    if probes is None:
        rng = np.random.default_rng() if rng is None else rng
        probes = rng.choice([-1.0, 1.0], size=(kron.N, int(n_probes))).astype(
            np.float64
        )
    grad = flow_logdet_grad(
        kron, rho_d, rho_o, rho_w, probes=probes, tol=tol, maxiter=maxiter
    )
    value = flow_logdet_value(
        kron,
        rho_d,
        rho_o,
        rho_w,
        n_quad=n_quad,
        probes=probes,
        tol=tol,
        maxiter=maxiter,
    )
    return value, grad


def flow_logdet_grad_exact(W, rho_d: float, rho_o: float, rho_w: float) -> np.ndarray:
    r"""Exact ``∇_ρ log|I_N − W_F|`` from the spectrum of ``W`` (reference; small ``n``).

    Uses the shared Kronecker eigenbasis: ``μ_ij = ρ_o λ_i + ρ_d λ_j + ρ_w λ_i λ_j``
    and ``g_d = −Σ_ij λ_j/(1−μ_ij)`` etc.  Only for testing / small ``n`` — forms
    the ``n`` eigenvalues of ``W`` (``O(n³)``), never the ``N×N`` matrix.
    """
    Wd = _as_csr(W).toarray()
    lam = np.linalg.eigvals(Wd)
    li = lam[:, None]
    lj = lam[None, :]
    mu = rho_o * li + rho_d * lj + rho_w * (li * lj)
    inv = 1.0 / (1.0 - mu)
    g_d = -np.sum(lj * inv).real
    g_o = -np.sum(li * inv).real
    g_w = -np.sum(li * lj * inv).real
    return np.array([g_d, g_o, g_w], dtype=np.float64)
