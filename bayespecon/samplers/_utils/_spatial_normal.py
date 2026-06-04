"""Sample from a multivariate normal with sparse spatial precision.

Draws x ~ N(m, Σ) where Σ⁻¹ = P (sparse SPD) via sparse Cholesky
factorisation, conjugate gradient (CG) iterative solve, or Chebyshev
polynomial approximation.

**Factorisation path** (default for moderate n):
    When ``scikit-sparse`` (CHOLMOD) is available, uses CHOLMOD's
    ``cholesky`` which is 5–9× faster than ``scipy.sparse.linalg.splu``
    for SPD matrices.  Falls back to ``splu`` when CHOLMOD is not installed.

    The sampling formula uses the Cholesky factor L where P = L Lᵀ:

        x = m + L⁻ᵀ z,  z ~ N(0, I)

**Iterative path** (for large n with high fill-in):
    Uses preconditioned CG for the mean solve and Lanczos-based
    stochastic log-determinant estimation.  Avoids the O(nnz^{1.5})
    factorisation cost entirely.

**JAX dense path** (for n ≤ ~5000 with JAX installed):
    Uses JAX dense matvec + vmap over Lanczos probes and Chebyshev
    draws.  3–4× faster for single draws, 20–27× per-draw when
    batching Chebyshev draws.  Requires ``jax_enable_x64=True``.

    The sampling formula uses the Cholesky factor L where P = L Lᵀ:

        x = m + L⁻ᵀ z,  z ~ N(0, I)

which gives Cov(x) = L⁻ᵀ L⁻¹ = (L Lᵀ)⁻¹ = P⁻¹.

When using the ``splu`` fallback, the LDLᵀ form is used instead:
P = L D Lᵀ, and x = m + L⁻ᵀ D⁻¹ᐟ² z.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
# CHOLMOD availability
# ---------------------------------------------------------------------------
try:
    from sksparse.cholmod import cholesky as _cholmod_cholesky

    _HAS_CHOLMOD = True
except ImportError:  # pragma: no cover
    _HAS_CHOLMOD = False


def has_cholmod() -> bool:
    """Return True if CHOLMOD (scikit-sparse) is available."""
    return _HAS_CHOLMOD


# ---------------------------------------------------------------------------
# CHOLMOD factorisation wrapper
# ---------------------------------------------------------------------------


class CholmodFactor:
    """Wrapper around a CHOLMOD factorisation for a fixed sparsity pattern.

    Stores the symbolic analysis so that ``cholesky_inplace`` only
    does the numeric factorisation when the matrix values change but
    the sparsity pattern stays the same.  This is the key optimisation
    for the ρ block in the Gibbs sampler, where P_η changes with each
    candidate ρ but always has the same non-zero structure.

    Parameters
    ----------
    pattern_matrix : sparse matrix
        Any matrix with the target sparsity pattern.  Used only for
        the symbolic analysis; the numeric values are irrelevant.
    """

    def __init__(self, pattern_matrix: sp.spmatrix) -> None:
        if not _HAS_CHOLMOD:
            raise RuntimeError(
                "scikit-sparse (CHOLMOD) is not installed. "
                "Install with: conda install scikit-sparse"
            )
        self._pattern_matrix = sp.csc_matrix(pattern_matrix)
        self._factor = _cholmod_cholesky(self._pattern_matrix)

    def __getstate__(self) -> dict:
        """Support pickling: store pattern matrix, drop C factor."""
        return {"_pattern_matrix": self._pattern_matrix}

    def __setstate__(self, state: dict) -> None:
        """Reconstruct CHOLMOD factor from pattern matrix on unpickle."""
        self._pattern_matrix = state["_pattern_matrix"]
        self._factor = _cholmod_cholesky(self._pattern_matrix)

    def factorize(self, matrix: sp.spmatrix) -> None:
        """Re-factorize with new values (same sparsity pattern).

        Parameters
        ----------
        matrix : sparse matrix
            New SPD matrix with the same sparsity pattern as the
            pattern matrix passed at construction.
        """
        self._factor.cholesky_inplace(sp.csc_matrix(matrix))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve P x = rhs."""
        return self._factor.solve_A(rhs)

    def logdet(self) -> float:
        """Return log|P|."""
        return self._factor.logdet()

    def sample(
        self,
        mean_term: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw from N(m, P⁻¹) where m = P⁻¹ @ mean_term.

        Uses the Cholesky factor L (P = L Lᵀ) to compute:
            m = P⁻¹ @ mean_term   (via CHOLMOD solve)
            x = m + L⁻ᵀ z        (z ~ N(0, I)

        Parameters
        ----------
        mean_term : ndarray of shape (n,)
            The precision-weighted mean: P @ m.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        x : ndarray of shape (n,)
            Draw from N(m, P⁻¹).
        """
        from scipy.sparse.linalg import spsolve_triangular

        m = self._factor.solve_A(mean_term)
        L = self._factor.L().tocsc()
        z = rng.standard_normal(L.shape[0])
        # Solve L^T v = z  →  v = L^{-T} z
        v = spsolve_triangular(L.T.tocsr(), z, lower=False)
        return m + v


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class SpatialNormalDraw(NamedTuple):
    """Result of a spatial-normal draw.

    Attributes
    ----------
    x : ndarray of shape (n,)
        The drawn sample.
    factor : CholmodFactor or SuperLU
        The factorisation of the precision matrix.  Can be reused
        for subsequent solves when the precision matrix has not changed.
    """

    x: np.ndarray
    factor: Any  # CholmodFactor or spla.SuperLU


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_spatial_normal(
    precision: sp.csr_matrix | sp.csc_matrix,
    mean_term: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    cached_factor: Union[CholmodFactor, spla.SuperLU, None] = None,
    use_cholmod: bool = _HAS_CHOLMOD,
) -> SpatialNormalDraw:
    """Draw from N(m, P⁻¹) where P is sparse SPD.

    When CHOLMOD is available (default), uses ``sksparse.cholmod.cholesky``
    which is 5–9× faster than ``scipy.sparse.linalg.splu`` for SPD
    matrices.  Falls back to ``splu`` when CHOLMOD is not installed or
    when ``use_cholmod=False``.

    Parameters
    ----------
    precision : sparse matrix of shape (n, n)
        Sparse precision matrix P. Must be symmetric positive definite.
        Typically A_ρᵀ A_ρ / σ² + diag(ω) for spatial models.
    mean_term : ndarray of shape (n,)
        The precision-weighted mean: P @ m. The actual mean is
        m = P⁻¹ @ mean_term, computed via the sparse solve.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.
    cached_factor : CholmodFactor or SuperLU, optional
        Pre-computed factorisation of precision. If None, computed
        fresh. Passing a cached factorisation saves the factorisation
        cost when P has not changed between calls.
    use_cholmod : bool, default True if available
        If True and CHOLMOD is installed, use CHOLMOD. Otherwise
        fall back to ``splu``.

    Returns
    -------
    SpatialNormalDraw
        Named tuple with fields ``x`` (the draw) and ``factor``
        (the factorisation, for potential reuse).

    Notes
    -----
    **CHOLMOD path** (default when available):

    Uses ``sksparse.cholmod.cholesky`` to compute P = L Lᵀ, then

    .. math::

        x = m + L^{-T} z, \\quad z \\sim N(0, I)

    which gives Cov(x) = L^{-T} L^{-1} = (L L^T)^{-1} = P^{-1}.

    **splu fallback path**:

    Uses ``scipy.sparse.linalg.splu`` with ``permc_spec="MMD_AT_PLUS_A"``
    for symmetric ordering. For SPD matrices with this ordering, the
    factorisation gives P_perm = L @ U where U = D @ Lᵀ (LDLᵀ form).

    .. math::

        x = m + L^{-T} D^{-1/2} z, \\quad z \\sim N(0, I)

    which gives Cov(x) = L^{-T} D^{-1} L^{-1} = (L D L^T)^{-1} = P^{-1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    precision_csc = sp.csc_matrix(precision)

    if use_cholmod and _HAS_CHOLMOD:
        return _sample_cholmod(
            precision_csc, mean_term, rng=rng, cached_factor=cached_factor
        )
    else:
        return _sample_splu(
            precision_csc, mean_term, rng=rng, cached_factor=cached_factor
        )


# ---------------------------------------------------------------------------
# CHOLMOD implementation
# ---------------------------------------------------------------------------


def _sample_cholmod(
    precision_csc: sp.csc_matrix,
    mean_term: np.ndarray,
    *,
    rng: np.random.Generator,
    cached_factor: CholmodFactor | None = None,
) -> SpatialNormalDraw:
    """Draw using CHOLMOD (sksparse.cholmod.cholesky)."""
    from scipy.sparse.linalg import spsolve_triangular

    if cached_factor is not None:
        factor = cached_factor
        # Re-factorize with current values (reuses symbolic analysis)
        factor.factorize(precision_csc)
    else:
        factor = CholmodFactor(precision_csc)

    # Conditional mean: m = P^{-1} @ mean_term
    m = factor.solve(mean_term)

    # Sample: x = m + L^{-T} z, z ~ N(0, I)
    L = factor._factor.L().tocsc()
    z = rng.standard_normal(L.shape[0])
    v = spsolve_triangular(L.T.tocsr(), z, lower=False)
    x = m + v

    return SpatialNormalDraw(x=x, factor=factor)


# ---------------------------------------------------------------------------
# splu fallback implementation
# ---------------------------------------------------------------------------


def _sample_splu(
    precision_csc: sp.csc_matrix,
    mean_term: np.ndarray,
    *,
    rng: np.random.Generator,
    cached_factor: spla.SuperLU | None = None,
) -> SpatialNormalDraw:
    """Draw using scipy.sparse.linalg.splu (LDL^T path)."""
    from scipy.sparse.linalg import spsolve_triangular

    if cached_factor is not None:
        lu = cached_factor
    else:
        lu = spla.splu(precision_csc, permc_spec="MMD_AT_PLUS_A")

    # Conditional mean: m = P^{-1} @ mean_term
    m = lu.solve(mean_term)

    # Sample from N(0, P^{-1}) using the LDL^T factorisation.
    #
    # For SPD P with symmetric ordering (MMD_AT_PLUS_A), splu gives:
    #   P_perm = L @ U  where U = D @ L^T  (LDL^T form)
    #   D = diag(U) / diag(L^T) = diag(U)  (since L is unit lower triangular)
    #
    # To sample x ~ N(m, P^{-1}):
    #   z ~ N(0, I)
    #   Solve L^T v = D^{-1/2} z  (backward substitution)
    #   x = m + v
    #
    # Cov(v) = L^{-T} D^{-1/2} I D^{-1/2} L^{-1}
    #        = L^{-T} D^{-1} L^{-1}
    #        = (L D L^T)^{-1}
    #        = P^{-1}  ✓
    L = lu.L
    U = lu.U

    # Extract D from the diagonal of U.
    D_diag = np.abs(U.diagonal())
    D_inv_sqrt = 1.0 / np.sqrt(D_diag)

    z = rng.standard_normal(L.shape[0])
    z_scaled = D_inv_sqrt * z

    # Solve L^T v = z_scaled for v (backward substitution)
    L_T = L.T.tocsr()
    v = spsolve_triangular(L_T, z_scaled, lower=False)

    x = m + v

    return SpatialNormalDraw(x=x, factor=lu)


# ---------------------------------------------------------------------------
# Lanczos log-determinant estimator
# ---------------------------------------------------------------------------


def lanczos_logdet(
    precision: sp.spmatrix | spla.LinearOperator,
    *,
    n_probes: int = 10,
    lanczos_deg: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate log|P| for sparse SPD P via Lanczos iteration.

    Uses the Hutchinson trace estimator applied to the matrix
    logarithm:  log|P| = tr(log(P)).  For each probe vector z,
    runs a Lanczos iteration on P starting from z to build a
    tridiagonal matrix T_m, then estimates z^T log(P) z ≈ ||z||² e₁^T log(T_m) e₁.

    Parameters
    ----------
    precision : sparse matrix or LinearOperator of shape (n, n)
        Sparse SPD precision matrix, or a ``LinearOperator`` with
        a ``matvec`` method.  Passing a ``LinearOperator`` avoids
        constructing the full N×N sparse matrix when the matvec
        can be computed more efficiently (e.g., via Kronecker structure).
    n_probes : int, default 10
        Number of probe vectors.  More probes reduce variance.
    lanczos_deg : int, default 30
        Number of Lanczos iterations per probe.  Higher values
        improve accuracy at the cost of more matrix-vector products.
    rng : numpy.random.Generator, optional
        Random state.  If None, a fresh generator is created.

    Returns
    -------
    logdet : float
        Estimated log-determinant of P.

    Notes
    -----
    The cost is O(n_probes * lanczos_deg * nnz) where nnz is the
    number of non-zeros in P.  For typical spatial precision matrices
    with n > 5000, this can be significantly faster than CHOLMOD
    factorisation when fill-in is high.

    The estimator is unbiased in the limit of infinite probes and
    Lanczos depth.  With n_probes=10 and lanczos_deg=30, the
    relative error is typically < 1e-3 for well-conditioned matrices.

    **Why not a generic trace estimator?**  Variance-reduced trace
    estimators (Hutch++, XTrace) are designed for :math:`\\text{tr}(A)`
    where :math:`A` is an *explicit* linear operator (you provide
    matvecs).  However, :math:`\\log|P| = \\text{tr}(\\log(P))` requires
    computing :math:`\\log(P)` as an operator, which itself needs
    Lanczos tridiagonalization per matvec.  Using a generic estimator
    here would nest Lanczos *inside* the estimator's probes, making it
    strictly more expensive.  Our implementation combines Lanczos and
    trace estimation in a single pass — this *is* the standard
    algorithm for :math:`\\text{tr}(f(A))` (Ubaru & Saad 2016).

    References
    ----------
    .. [1] Ubaru, S., & Saad, Y. (2018). Applications of Trace Estimation
       Techniques. In T. Kozubek, M. Čermák, P. Tichý, R. Blaheta, J. Šístek,
       D. Lukáš, & J. Jaroš (Eds.), High Performance Computing in Science
       and Engineering (pp. 19–33). Springer International Publishing.
       https://doi.org/10.1007/978-3-319-97136-0_2

    .. [2] Ubaru, S., & Saad, Y. (2016). Fast methods for estimating the
       Numerical rank of large matrices.

    .. [3] Bai, Z., Fahey, G., & Golub, G. (1996). Some large-scale
       matrix computation problems.
    """

    if rng is None:
        rng = np.random.default_rng()

    n = precision.shape[0]
    # Accept both sparse matrices and LinearOperator
    if isinstance(precision, spla.LinearOperator):
        P_op = precision
    else:
        P_op = sp.csr_matrix(precision)

    estimates = np.empty(n_probes)
    for j in range(n_probes):
        z = rng.standard_normal(n)
        z_norm = np.linalg.norm(z)
        if z_norm == 0:
            estimates[j] = 0.0
            continue
        q = z / z_norm

        # Lanczos iteration: build tridiagonal T_m
        alpha_vals = np.empty(lanczos_deg)
        beta_vals = np.empty(lanczos_deg - 1)
        Q = np.empty((n, lanczos_deg))

        Q[:, 0] = q
        r = P_op @ q
        alpha_vals[0] = float(q @ r)
        r = r - alpha_vals[0] * q

        for i in range(1, lanczos_deg):
            beta_vals[i - 1] = np.linalg.norm(r)
            if beta_vals[i - 1] < 1e-15:
                # Invariant subspace found — truncate
                alpha_vals = alpha_vals[:i]
                beta_vals = beta_vals[: i - 1] if i > 1 else beta_vals
                Q = Q[:, :i]
                break
            q_new = r / beta_vals[i - 1]
            Q[:, i] = q_new
            r = P_op @ q_new
            alpha_vals[i] = float(q_new @ r)
            # Full reorthogonalisation (one pass)
            r = r - alpha_vals[i] * q_new - beta_vals[i - 1] * Q[:, i - 1]
            # Modified Gram-Schmidt against all previous vectors
            for k in range(i):
                r = r - float(Q[:, k] @ r) * Q[:, k]

        # Build tridiagonal matrix T_m
        m = len(alpha_vals)
        T = np.diag(alpha_vals[:m])
        if m > 1:
            T += np.diag(beta_vals[: m - 1], 1) + np.diag(beta_vals[: m - 1], -1)

        # z^T log(P) z ≈ ||z||² e₁^T log(T) e₁
        eigvals, eigvecs = np.linalg.eigh(T)
        log_T_diag = np.log(np.maximum(eigvals, 1e-300))
        logdet_T = float(eigvecs[0, :] @ (log_T_diag * eigvecs[0, :]))
        estimates[j] = z_norm**2 * logdet_T

    return float(np.mean(estimates))


# ---------------------------------------------------------------------------
# Preconditioned CG solve
# ---------------------------------------------------------------------------


def cg_solve(
    precision: sp.spmatrix | spla.LinearOperator,
    rhs: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: str = "jacobi",
) -> np.ndarray:
    """Solve P x = rhs for sparse SPD P via preconditioned CG.

    Parameters
    ----------
    precision : sparse matrix or LinearOperator of shape (n, n)
        Sparse SPD precision matrix, or a ``LinearOperator`` with
        a ``matvec`` method.  Passing a ``LinearOperator`` avoids
        constructing the full N×N sparse matrix when the matvec
        can be computed more efficiently (e.g., via Kronecker structure).
    rhs : ndarray of shape (n,)
        Right-hand side vector.
    tol : float, default 1e-8
        Convergence tolerance (relative residual norm).
    maxiter : int, optional
        Maximum iterations.  Defaults to 2 * n.
    preconditioner : {"jacobi", "none"}, default "jacobi"
        Preconditioner type.  "jacobi" uses M = diag(P), which
        is cheap and effective for diagonally-dominant spatial
        precision matrices.  Not available when ``precision`` is a
        ``LinearOperator`` (use "none" instead).

    Returns
    -------
    x : ndarray of shape (n,)
        Approximate solution to P x = rhs.

    Notes
    -----
    For spatial precision matrices of the form
    P = I/σ² + diag(ω) - ρ(W+W^T)/σ² + ρ²W^TW/σ²,
    the Jacobi preconditioner M = diag(P) = I/σ² + diag(ω) is
    very effective because the diagonal dominates for typical
    spatial weights (small ρ and bounded ω).
    """
    n = precision.shape[0]
    if maxiter is None:
        maxiter = 2 * n

    # Accept both sparse matrices and LinearOperator
    if isinstance(precision, spla.LinearOperator):
        P_op = precision
        # No Jacobi preconditioner for LinearOperator (no diagonal access)
        M_op = None
    else:
        P_csr = sp.csr_matrix(precision)
        P_op = P_csr

        # Build preconditioner LinearOperator (only for sparse matrices)
        if preconditioner == "jacobi":
            M_diag = P_csr.diagonal()
            M_diag = np.where(np.abs(M_diag) > 1e-15, M_diag, 1.0)
            M_inv_diag = 1.0 / M_diag
            M_op = spla.LinearOperator((n, n), matvec=lambda v: M_inv_diag * v)
        elif preconditioner == "none":
            M_op = None
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner!r}")

    # scipy >= 1.12 uses rtol/atol; older versions use tol.
    # Use rtol for forward compatibility.
    x, info = spla.cg(
        P_op,
        rhs,
        rtol=tol,
        maxiter=maxiter,
        M=M_op,
    )

    if info != 0:
        # CG did not converge — return best iterate anyway
        import warnings

        warnings.warn(
            f"CG did not converge after {maxiter} iterations "
            f"(info={info}). Returning best iterate.",
            RuntimeWarning,
            stacklevel=2,
        )

    return x


# ---------------------------------------------------------------------------
# Chebyshev polynomial sampler for η draw
# ---------------------------------------------------------------------------


def _gershgorin_bounds(P: sp.spmatrix) -> tuple[float, float]:
    """Compute Gershgorin eigenvalue bounds for a symmetric sparse matrix.

    For a symmetric matrix, all eigenvalues lie in the union of
    intervals [P_ii - R_i, P_ii + R_i] where R_i = Σ_{j≠i} |P_ij|.

    Parameters
    ----------
    P : sparse matrix of shape (n, n)
        Symmetric matrix.

    Returns
    -------
    lambda_min : float
        Lower bound on the smallest eigenvalue.
    lambda_max : float
        Upper bound on the largest eigenvalue.
    """
    P_csr = sp.csr_matrix(P)
    diag = P_csr.diagonal()
    # R_i = sum of |off-diagonal| in row i
    abs_P = abs(P_csr)
    row_sums = np.asarray(abs_P.sum(axis=1)).ravel()
    R = row_sums - np.abs(diag)
    lambda_min = float(np.min(diag - R))
    lambda_max = float(np.max(diag + R))
    return lambda_min, lambda_max


def _chebyshev_coeffs_inv_sqrt(
    lambda_min: float,
    lambda_max: float,
    degree: int,
) -> np.ndarray:
    """Compute Chebyshev coefficients for f(x) = x^{-1/2} on [a, b].

    Maps [a, b] → [-1, 1] and computes the Chebyshev series
    coefficients via the DCT of f evaluated at Chebyshev nodes.

    Parameters
    ----------
    lambda_min : float
        Lower bound of the eigenvalue interval (must be > 0).
    lambda_max : float
        Upper bound of the eigenvalue interval.
    degree : int
        Polynomial degree.

    Returns
    -------
    coeffs : ndarray of shape (degree + 1,)
        Chebyshev coefficients c_0, c_1, ..., c_degree.
    """
    if lambda_min <= 0:
        raise ValueError(
            f"lambda_min must be positive for x^{{-1/2}}, got {lambda_min}"
        )

    m = degree + 1
    # Chebyshev nodes on [-1, 1]
    k = np.arange(1, m + 1)
    nodes = np.cos((2 * k - 1) * np.pi / (2 * m))

    # Map nodes from [-1, 1] to [lambda_min, lambda_max]
    # x ∈ [-1, 1] → λ = (a+b)/2 + (b-a)/2 * x
    mid = 0.5 * (lambda_min + lambda_max)
    half_range = 0.5 * (lambda_max - lambda_min)
    lam_nodes = mid + half_range * nodes

    # Evaluate f(λ) = λ^{-1/2} at the mapped nodes
    f_vals = lam_nodes ** (-0.5)

    # Chebyshev coefficients via DCT-I
    coeffs = np.zeros(m, dtype=np.float64)
    for j in range(m):
        scale = 2.0 / m if j > 0 else 1.0 / m
        coeffs[j] = scale * np.sum(f_vals * np.cos(j * (2 * k - 1) * np.pi / (2 * m)))

    return coeffs


def chebyshev_sample(
    precision: sp.spmatrix | spla.LinearOperator,
    mean_term: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    degree: int = 30,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    cg_tol: float = 1e-8,
) -> SpatialNormalDraw:
    """Draw from N(m, P⁻¹) via Chebyshev polynomial approximation of P⁻¹ᐟ².

    Avoids the O(nnz^{1.5}) sparse factorisation cost by:
    1. Computing the conditional mean m = P⁻¹ rhs via CG.
    2. Approximating P⁻¹ᐟ² z (z ~ N(0, I)) via a Chebyshev polynomial
       of P, evaluated via Clenshaw's recurrence.

    Parameters
    ----------
    precision : sparse matrix or LinearOperator of shape (n, n)
        Sparse SPD precision matrix P, or a ``LinearOperator`` with
        a ``matvec`` method.  Passing a ``LinearOperator`` avoids
        constructing the full N×N sparse matrix when the matvec
        can be computed more efficiently (e.g., via Kronecker structure).
        When passing a ``LinearOperator``, ``lambda_min`` and
        ``lambda_max`` must be provided explicitly since Gershgorin
        bounds require diagonal access.
    mean_term : ndarray of shape (n,)
        The precision-weighted mean: P @ m. The actual mean is
        m = P⁻¹ @ mean_term, computed via CG.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.
    degree : int, default 30
        Degree of the Chebyshev polynomial approximation for P⁻¹ᐟ².
        Higher values improve accuracy at the cost of more matrix-vector
        products.  20–40 is usually sufficient for well-conditioned
        spatial precision matrices.
    lambda_min : float, optional
        Lower eigenvalue bound. If None, computed via Gershgorin circles.
    lambda_max : float, optional
        Upper eigenvalue bound. If None, computed via Gershgorin circles.
    cg_tol : float, default 1e-8
        CG convergence tolerance for the mean solve.

    Returns
    -------
    SpatialNormalDraw
        Named tuple with fields ``x`` (the draw) and ``factor``
        (None — no factorisation is available for reuse).

    Notes
    -----
    The sampling formula is:

    .. math::

        \\eta = m + \\hat{P}^{-1/2} z, \\quad z \\sim N(0, I)

    where :math:`\\hat{P}^{-1/2}` is a Chebyshev polynomial approximation
    of :math:`P^{-1/2}`.  The approximation error in the covariance is:

    .. math::

        \\text{Cov}(\\eta) = P^{-1} + O(\\|P^{-1/2} - \\hat{P}^{-1/2}\\|^2)

    For well-conditioned matrices (condition number < 100), degree 30
    typically gives relative covariance error < 1e-4.

    The cost is O((degree + t_cg) * nnz) where t_cg is the CG iteration
    count for the mean solve.  This is typically much cheaper than
    O(nnz^{1.5}) for large n with high fill-in.

    References
    ----------
    .. [1] Fox, C., & Parker, A. (2014). Convergence in Variance of Chebyshev
       Accelerated Gibbs Samplers. SIAM Journal on Scientific Computing, 36(1),
       A124–A147. https://doi.org/10.1137/120900940

    .. [2] Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*.
       2nd ed. SIAM. (Background on Chebyshev polynomial iteration.)
    """

    if rng is None:
        rng = np.random.default_rng()

    n = precision.shape[0]

    # Accept both sparse matrices and LinearOperator
    if isinstance(precision, spla.LinearOperator):
        P_op = precision
        P_csr = None  # Not available for LinearOperator
    else:
        P_csr = sp.csr_matrix(precision)
        P_op = P_csr

    # --- Step 1: Compute mean via CG ---
    m = cg_solve(P_op, mean_term, tol=cg_tol)

    # --- Step 2: Compute eigenvalue bounds ---
    if lambda_min is None or lambda_max is None:
        if P_csr is not None:
            g_min, g_max = _gershgorin_bounds(P_csr)
            lambda_min = g_min if lambda_min is None else lambda_min
            lambda_max = g_max if lambda_max is None else lambda_max
        else:
            raise ValueError(
                "When precision is a LinearOperator, lambda_min and lambda_max "
                "must be provided explicitly (Gershgorin bounds require "
                "diagonal access)."
            )

    # Safety: ensure lambda_min > 0 (P is SPD)
    if lambda_min <= 0:
        lambda_min = 1e-6

    # Safety: ensure lambda_max > lambda_min (avoid division by zero)
    if lambda_max <= lambda_min:
        lambda_max = lambda_min + 1.0

    # --- Step 3: Chebyshev approximation of P^{-1/2} z ---
    coeffs = _chebyshev_coeffs_inv_sqrt(lambda_min, lambda_max, degree)

    z = rng.standard_normal(n)

    # Evaluate f(P) z = Σ_{j=0}^{degree} c_j T_j(P_mapped) z
    # where P_mapped = (2P - (λ_max + λ_min)I) / (λ_max - λ_min)
    # maps eigenvalues from [λ_min, λ_max] to [-1, 1].
    #
    # Use the three-term Chebyshev recurrence on the *unmapped* matrix P
    # (Saad 2003, §12.3), which is numerically stable:
    #   d = (λ_max - λ_min) / 2  (half-range)
    #   c = (λ_max + λ_min) / 2  (midpoint)
    #   T_0(P_m) z = z
    #   T_1(P_m) z = (P z - c z) / d
    #   T_j(P_m) z = 2 (P z_j - c z_j) / d - z_{j-1}
    #
    # This avoids forming P_mapped explicitly and keeps the
    # intermediate vectors bounded.
    d = 0.5 * (lambda_max - lambda_min)
    c = 0.5 * (lambda_max + lambda_min)
    inv_d = 1.0 / d

    # Forward recurrence for T_j(P_mapped) z
    y_prev = z.copy()  # T_0(P_m) z = z
    y_curr = (P_op @ z - c * z) * inv_d  # T_1(P_m) z = (Pz - cz) / d

    v = coeffs[0] * y_prev + coeffs[1] * y_curr

    for j in range(2, degree + 1):
        y_new = 2.0 * (P_op @ y_curr - c * y_curr) * inv_d - y_prev
        v += coeffs[j] * y_new
        y_prev = y_curr
        y_curr = y_new

    x = m + v

    return SpatialNormalDraw(x=x, factor=None)


# ---------------------------------------------------------------------------
# JAX-accelerated variants (dense matvec + vmap)
# ---------------------------------------------------------------------------
# These functions use JAX dense matrix-vector products and jax.vmap
# to batch sequential operations into single XLA kernels.  They are
# significantly faster than the scipy sparse versions for n ≤ ~5000
# where dense matvec is competitive with scipy CSR matvec.
#
# All JAX imports are lazy — these functions are only called when
# gibbs_method="jax_dense" is selected, and JAX availability is
# checked at model construction time.
# ---------------------------------------------------------------------------


def _check_jax_available() -> None:
    """Raise ImportError if JAX is not installed."""
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        raise ImportError(
            "JAX is required for the 'jax_dense' Gibbs method. "
            "Install with: pip install jax"
        )


def _jax_lanczos_probe(P_dense, z_raw, lanczos_deg):
    """Single Lanczos probe: estimate z^T log(P) z.

    Runs a Lanczos iteration on P starting from z/||z||, builds
    tridiagonal matrix T_m, and returns ||z||^2 * e_1^T log(T_m) e_1.

    This is the inner function designed to be vmapped over probes.

    Notes
    -----
    Reorthogonalisation uses the full Q matrix (n × lanczos_deg)
    rather than a dynamic slice Q[:, :i], because JAX's lax.scan
    requires static slice sizes.  Columns beyond the current iteration
    are zero, so projecting them out has no effect.
    """
    import jax
    import jax.numpy as jnp

    n = P_dense.shape[0]
    z_norm = jnp.linalg.norm(z_raw)
    q = z_raw / jnp.where(z_norm < 1e-15, 1.0, z_norm)

    # Pre-allocate Q matrix (n × lanczos_deg) and coefficient arrays
    Q = jnp.zeros((n, lanczos_deg))
    Q = Q.at[:, 0].set(q)
    alphas = jnp.zeros(lanczos_deg)
    betas = jnp.zeros(lanczos_deg - 1)

    # First step
    r = P_dense @ q
    alpha0 = jnp.dot(q, r)
    r = r - alpha0 * q
    alphas = alphas.at[0].set(alpha0)

    # Lanczos iteration via lax.scan
    def body(carry, i):
        Q, alphas, betas, r = carry
        beta = jnp.linalg.norm(r)
        q_new = r / jnp.where(beta < 1e-15, 1.0, beta)
        Q = Q.at[:, i].set(q_new)
        r = P_dense @ q_new
        alpha = jnp.dot(q_new, r)
        # Three-term recurrence
        r = r - alpha * q_new - beta * Q[:, i - 1]
        # Full reorthogonalisation against all Q columns.
        # Columns beyond i are zero, so this is equivalent to
        # projecting out Q[:, :i+1] but with a static slice size.
        r = r - Q @ (Q.T @ r)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i - 1].set(beta)
        return (Q, alphas, betas, r), None

    (Q, alphas, betas, _), _ = jax.lax.scan(
        body,
        (Q, alphas, betas, r),
        jnp.arange(1, lanczos_deg),
    )

    # Build tridiagonal T_m and compute e_1^T log(T_m) e_1
    T = jnp.diag(alphas) + jnp.diag(betas, 1) + jnp.diag(betas, -1)
    eigvals, eigvecs = jnp.linalg.eigh(T)
    log_T_diag = jnp.log(jnp.maximum(eigvals, 1e-300))
    logdet_T = jnp.dot(eigvecs[0, :], log_T_diag * eigvecs[0, :])
    return z_norm**2 * logdet_T


def jax_lanczos_logdet(
    P_dense: "jnp.ndarray",  # noqa: F821
    *,
    key: "jax.random.PRNGKey",  # noqa: F821
    n_probes: int = 10,
    lanczos_deg: int = 30,
) -> float:
    """Estimate log|P| for dense SPD P via JAX-accelerated Lanczos.

    Same algorithm as :func:`lanczos_logdet` but uses JAX dense matvec
    and :func:`jax.vmap` over probe vectors to batch all probes into
    a single XLA kernel.  This gives a 3–4× speedup over the numpy
    implementation for n ≤ 2000.

    Parameters
    ----------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense SPD precision matrix.  Must be float64 for numerical
        stability — float32 causes NaN/inf in the Lanczos iteration.
    key : jax.random.PRNGKey
        JAX random key for probe vector generation.
    n_probes : int, default 10
        Number of probe vectors.
    lanczos_deg : int, default 30
        Number of Lanczos iterations per probe.

    Returns
    -------
    logdet : float or jax.numpy.ndarray
        Estimated log-determinant of P.  Returns a Python float
        when called outside ``jax.jit``, or a JAX array when called
        inside JIT (the ``float()`` conversion is deferred to the caller).

    Notes
    -----
    Requires ``jax_enable_x64=True``.  Without float64, the Lanczos
    iteration accumulates roundoff errors that cause NaN/inf.

    The vmap over probes batches all ``n_probes`` Lanczos iterations
    into a single XLA kernel, eliminating Python-loop overhead and
    enabling XLA fusion across the matvec + orthogonalisation steps.

    """
    _check_jax_available()
    import jax
    import jax.numpy as jnp

    n = P_dense.shape[0]

    # Generate probe vectors via vmap (jax.random.normal doesn't accept
    # key arrays directly — must vmap over individual keys)
    keys = jax.random.split(key, n_probes)
    z_all = jax.vmap(lambda k: jax.random.normal(k, shape=(n,)))(keys)

    # vmap over probes — each probe runs an independent Lanczos iteration
    estimates = jax.vmap(lambda z: _jax_lanczos_probe(P_dense, z, lanczos_deg))(z_all)
    return jnp.mean(estimates)


def jax_cg_solve(
    P_dense: "jnp.ndarray",  # noqa: F821
    rhs: "jnp.ndarray",  # noqa: F821
    M_inv_diag: "jnp.ndarray | None" = None,  # noqa: F821
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> "jnp.ndarray":  # noqa: F821
    """Solve P x = rhs for dense SPD P via JAX preconditioned CG.

    Uses ``jax.scipy.sparse.linalg.cg`` with a dense matvec
    ``lambda v: P_dense @ v``.  Machine-precision accuracy (1e-16
    relative error), unlike lineax CG which achieves only ~1e-9.

    Parameters
    ----------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense SPD precision matrix.
    rhs : jax.numpy.ndarray of shape (n,)
        Right-hand side vector.
    M_inv_diag : jax.numpy.ndarray of shape (n,), optional
        Diagonal of the inverse Jacobi preconditioner M⁻¹.
        If None, no preconditioner is used.
    tol : float, default 1e-8
        Convergence tolerance (relative residual norm).
    maxiter : int, optional
        Maximum iterations.  Defaults to 2 * n.

    Returns
    -------
    x : jax.numpy.ndarray of shape (n,)
        Solution to P x = rhs.

    Notes
    -----
    Uses ``jax.scipy.sparse.linalg.cg`` (not lineax) because:
    - Same speed as lineax CG
    - Machine-precision accuracy (1e-16 vs 1e-9)
    - Simpler API (no TaggedLinearOperator needed)

    Benchmarks (Apple M1, CPU):
        n=500:  scipy=0.56ms, JAX=0.29ms (1.9×)
        n=1000: scipy=1.28ms, JAX=0.69ms (1.9×)
        n=2000: scipy=2.31ms, JAX=2.33ms (1.0×)
        n=5000: scipy=5.81ms, JAX=18.2ms (0.3× — dense loses)
    """
    _check_jax_available()
    import jax
    import jax.scipy.sparse.linalg

    n = P_dense.shape[0]
    if maxiter is None:
        maxiter = 2 * n

    # JAX cg accepts A as a callable matvec or a dense array.
    # Passing the dense array directly is simplest and fastest.
    if M_inv_diag is not None:
        # Jacobi preconditioner: M^{-1} applied as element-wise multiply
        def M_func(v):
            return M_inv_diag * v
    else:
        M_func = None

    x, info = jax.scipy.sparse.linalg.cg(
        P_dense,
        rhs,
        tol=tol,
        maxiter=maxiter,
        M=M_func,
    )
    return x


def jax_chebyshev_sample(
    P_dense: "jnp.ndarray",  # noqa: F821
    mean_term: "jnp.ndarray",  # noqa: F821
    *,
    key: "jax.random.PRNGKey",  # noqa: F821
    degree: int = 30,
    n_draws: int = 1,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
) -> SpatialNormalDraw:
    """Draw from N(m, P⁻¹) via JAX Chebyshev polynomial with vmap over draws.

    Same algorithm as :func:`chebyshev_sample` but uses JAX dense matvec
    and :func:`jax.vmap` over draws to batch the Clenshaw recurrence
    across multiple draws.  For a single draw, JAX dispatch overhead
    makes this slightly slower than scipy; for 10 draws, vmap gives
    a 3.6× per-draw speedup.

    Parameters
    ----------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense SPD precision matrix.
    mean_term : jax.numpy.ndarray of shape (n,)
        Precision-weighted mean: P @ m.
    key : jax.random.PRNGKey
        JAX random key for z ~ N(0, I) generation.
    degree : int, default 30
        Chebyshev polynomial degree for P⁻¹ᐟ² approximation.
    n_draws : int, default 1
        Number of independent draws.  vmap batches all draws into
        a single XLA kernel, so 10 draws costs barely more than 4.
    lambda_min : float, optional
        Lower eigenvalue bound.  If None, computed via Gershgorin.
    lambda_max : float, optional
        Upper eigenvalue bound.  If None, computed via Gershgorin.

    Returns
    -------
    SpatialNormalDraw
        Named tuple with ``x`` (the first draw, shape (n,)) and
        ``factor=None`` (no factorisation available for reuse).

    Notes
    -----
    Requires ``jax_enable_x64=True``.

    The ``n_draws`` parameter must be a compile-time constant (not
    traced by JAX).  This is because ``jax.vmap`` needs to know the
    batch size at trace time.
    """
    _check_jax_available()
    import jax
    import jax.numpy as jnp

    n = P_dense.shape[0]

    # --- Step 1: Compute mean via JAX CG ---
    M_inv_diag = 1.0 / jnp.where(
        jnp.abs(jnp.diag(P_dense)) > 1e-15,
        jnp.diag(P_dense),
        1.0,
    )
    m = jax_cg_solve(P_dense, mean_term, M_inv_diag)

    # --- Step 2: Compute eigenvalue bounds ---
    if lambda_min is None or lambda_max is None:
        diag = jnp.diag(P_dense)
        abs_P = jnp.abs(P_dense)
        row_sums = jnp.sum(abs_P, axis=1)
        R = row_sums - jnp.abs(diag)
        g_min = jnp.min(diag - R)
        g_max = jnp.max(diag + R)
        lambda_min = g_min if lambda_min is None else lambda_min
        lambda_max = g_max if lambda_max is None else lambda_max

    # Safety: ensure bounds are valid (use jnp.maximum for JIT compatibility)
    lambda_min = jnp.maximum(lambda_min, 1e-6)
    lambda_max = jnp.maximum(lambda_max, lambda_min + 1.0)

    # --- Step 3: Chebyshev coefficients for x^{-1/2} on [a, b] ---
    # Inline computation (JIT-compatible, no Python-level if/float)
    md = degree + 1
    k = jnp.arange(1, md + 1, dtype=jnp.float64)
    nodes = jnp.cos((2 * k - 1) * jnp.pi / (2 * md))
    mid = 0.5 * (lambda_min + lambda_max)
    half_range = 0.5 * (lambda_max - lambda_min)
    lam_nodes = mid + half_range * nodes
    f_vals = lam_nodes ** (-0.5)
    coeffs = jnp.zeros(md, dtype=jnp.float64)
    for j in range(md):
        scale = 2.0 / md if j > 0 else 1.0 / md
        coeffs = coeffs.at[j].set(
            scale * jnp.sum(f_vals * jnp.cos(j * (2 * k - 1) * jnp.pi / (2 * md)))
        )

    # --- Step 4: Generate z vectors and compute P^{-1/2} z via Clenshaw ---
    d = 0.5 * (lambda_max - lambda_min)
    c = 0.5 * (lambda_max + lambda_min)
    inv_d = 1.0 / d

    keys = jax.random.split(key, n_draws)
    z_all = jax.vmap(lambda k: jax.random.normal(k, shape=(n,)))(keys)  # (n_draws, n)

    def _single_chebyshev_draw(z):
        """Clenshaw recurrence for one draw."""
        y_prev = z
        y_curr = (P_dense @ z - c * z) * inv_d
        v = coeffs[0] * y_prev + coeffs[1] * y_curr
        for j in range(2, degree + 1):
            y_new = 2.0 * (P_dense @ y_curr - c * y_curr) * inv_d - y_prev
            v = v + coeffs[j] * y_new
            y_prev = y_curr
            y_curr = y_new
        return m + v

    # vmap over draws
    draws = jax.vmap(_single_chebyshev_draw)(z_all)  # (n_draws, n)

    # Return first draw (Gibbs sampler only needs one)
    x = draws[0]
    return SpatialNormalDraw(x=np.asarray(x), factor=None)


def _jax_chebyshev_coeffs_inv_sqrt(
    lambda_min: float,
    lambda_max: float,
    degree: int,
) -> "jnp.ndarray":  # noqa: F821
    """Compute Chebyshev coefficients for f(x) = x^{-1/2} on [a, b].

    JAX-compatible version of :func:`_chebyshev_coeffs_inv_sqrt`.
    Returns a jax.numpy array instead of a numpy array.
    """
    _check_jax_available()
    import jax.numpy as jnp

    if lambda_min <= 0:
        raise ValueError(
            f"lambda_min must be positive for x^{{-1/2}}, got {lambda_min}"
        )

    m = degree + 1
    k = jnp.arange(1, m + 1)
    nodes = jnp.cos((2 * k - 1) * jnp.pi / (2 * m))

    mid = 0.5 * (lambda_min + lambda_max)
    half_range = 0.5 * (lambda_max - lambda_min)
    lam_nodes = mid + half_range * nodes

    f_vals = lam_nodes ** (-0.5)

    coeffs = jnp.zeros(m)
    for j in range(m):
        scale = 2.0 / m if j > 0 else 1.0 / m
        coeffs = coeffs.at[j].set(
            scale * jnp.sum(f_vals * jnp.cos(j * (2 * k - 1) * jnp.pi / (2 * m)))
        )

    return coeffs


def jax_build_P_dense(
    rho: float,
    sigma2: float,
    omega: "jnp.ndarray",  # noqa: F821
    W_sym_dense: "jnp.ndarray",  # noqa: F821
    WtW_dense: "jnp.ndarray",  # noqa: F821
) -> "jnp.ndarray":  # noqa: F821
    """Build the dense precision matrix P from precomputed components.

    Constructs P = I/σ² + diag(ω) - ρ(W+W^T)/σ² + ρ²W^TW/σ²
    using precomputed dense W components.  This is ~3× faster than
    building the scipy sparse version because dense arithmetic avoids
    sparse format overhead.

    Parameters
    ----------
    rho : float
        Spatial autoregressive parameter.
    sigma2 : float
        Residual variance.
    omega : jax.numpy.ndarray of shape (n,)
        PG auxiliary variables (diagonal of precision).
    W_sym_dense : jax.numpy.ndarray of shape (n, n)
        Dense (W + W^T), precomputed once at model setup.
    WtW_dense : jax.numpy.ndarray of shape (n, n)
        Dense W^T W, precomputed once at model setup.

    Returns
    -------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense precision matrix.
    """
    _check_jax_available()
    import jax.numpy as jnp

    n = omega.shape[0]
    inv_s2 = 1.0 / sigma2
    P = (
        jnp.diag(jnp.ones(n) * inv_s2 + omega)
        - rho * W_sym_dense * inv_s2
        + rho**2 * WtW_dense * inv_s2
    )
    return P


def _jax_logdet_W(rho, W_eigs):
    """Compute log|I - rho*W| from eigenvalues (JAX-compatible)."""
    import jax.numpy as jnp

    return jnp.sum(jnp.log(jnp.abs(1.0 - rho * W_eigs)))


def _jax_log_density_core(
    rho,
    sigma2,
    omega,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    Xbeta_over_s2,
    WtXbeta_over_s2,
    kappa,
    key,
    n_probes,
    lanczos_deg,
    cg_tol,
    cg_maxiter,
):
    """Core log-density computation, fully in JAX (no Python dispatch).

    This function is designed to be wrapped with ``jax.jit`` so that
    build P → Lanczos logdet → CG solve → quadratic form are fused
    into a single XLA kernel, eliminating Python dispatch overhead.

    Returns a JAX scalar (not a Python float) so it can be JIT-compiled.
    """
    import jax.numpy as jnp

    n = omega.shape[0]
    inv_s2 = 1.0 / sigma2

    # Build P
    P_diag = jnp.ones(n) * inv_s2 + omega
    P = jnp.diag(P_diag) - rho * W_sym_dense * inv_s2 + rho**2 * WtW_dense * inv_s2

    # RHS
    rhs = Xbeta_over_s2 - rho * WtXbeta_over_s2 + kappa

    # Jacobi preconditioner
    M_inv_diag = 1.0 / jnp.where(jnp.abs(P_diag) > 1e-15, P_diag, 1.0)

    # Lanczos logdet of P
    log_det_P = jax_lanczos_logdet(
        P, key=key, n_probes=n_probes, lanczos_deg=lanczos_deg
    )

    # CG solve P m = rhs
    m = jax_cg_solve(P, rhs, M_inv_diag, tol=cg_tol, maxiter=cg_maxiter)

    # Quadratic form
    quad = rhs @ m

    # log|I - rho*W| via generic JAX-native logdet callable
    logdet_W = logdet_jax(rho)

    # Final log-density
    return logdet_W - 0.5 * log_det_P + 0.5 * quad


def _jax_log_density_core_exact(
    rho,
    sigma2,
    omega,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    Xbeta_over_s2,
    WtXbeta_over_s2,
    kappa,
):
    """Exact log-density computation using dense Cholesky (no stochastic approx).

    This is a deterministic variant of :func:`_jax_log_density_core` that
    replaces Lanczos logdet with dense Cholesky and CG solve with
    ``jax.scipy.linalg.cho_solve``.  It is **much faster** for small
    matrices (n \u2264 ~500) because it avoids the overhead of stochastic
    trace estimation and iterative solvers.

    Use this for mode-finding during burn-in, where exactness and speed
    matter more than O(n\u00b3) scaling.

    Returns a JAX scalar.
    """
    import jax
    import jax.numpy as jnp

    n = omega.shape[0]
    inv_s2 = 1.0 / sigma2

    # Build P
    P_diag = jnp.ones(n) * inv_s2 + omega
    P = jnp.diag(P_diag) - rho * W_sym_dense * inv_s2 + rho**2 * WtW_dense * inv_s2

    # RHS
    rhs = Xbeta_over_s2 - rho * WtXbeta_over_s2 + kappa

    # Exact log|P| via dense Cholesky
    L = jnp.linalg.cholesky(P)
    log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    # Exact solve via Cholesky
    m = jax.scipy.linalg.cho_solve((L, True), rhs)
    quad = rhs @ m

    # log|I - rho*W| via generic JAX-native logdet callable
    logdet_W = logdet_jax(rho)

    return logdet_W - 0.5 * log_det_P + 0.5 * quad
