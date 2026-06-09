"""JAX dispatch registrations for the custom Ops in :mod:`bayespecon.ops`.

This module enables JAX-backed NUTS samplers (``"blackjax"``, ``"numpyro"``)
for models that depend on :class:`~bayespecon.ops.SparseFlowSolveOp`,
:class:`~bayespecon.ops.SparseFlowSolveMatrixOp`,
:class:`~bayespecon.ops.SparseSARSolveOp`,
:class:`~bayespecon.ops.KroneckerFlowSolveOp`, and
:class:`~bayespecon.ops.KroneckerFlowSolveMatrixOp`.

The Kronecker Ops are translated into pure-JAX dense LU solves
(:math:`n\\times n`, jittable, vmappable).  The general sparse Ops are wrapped
in :func:`jax.pure_callback` because JAX has no CPU sparse direct solver;
their gradients are routed through the corresponding ``_*VJPOp`` whose JAX
dispatch is also a callback.  PyTensor inserts the VJP node into the symbolic
graph (via ``L_op``) *before* JAX transpilation, so JAX never has to
differentiate across the callback.

Availability is probed via :func:`importlib.util.find_spec`; the registration
function is a no-op when JAX or PyTensor's JAX dispatch module is missing,
so importing this module is always safe.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from functools import lru_cache


@lru_cache(maxsize=1)
def _eqx_available() -> bool:
    """Return ``True`` when optional ``equinox`` is importable."""
    return importlib.util.find_spec("equinox") is not None


@lru_cache(maxsize=1)
def _jax_available() -> bool:
    """Return ``True`` if JAX and PyTensor's JAX dispatch are importable."""
    return (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("pytensor.link.jax.dispatch") is not None
    )


@lru_cache(maxsize=1)
def _klujax_available() -> bool:
    """Return ``True`` when optional ``klujax`` is importable."""
    return importlib.util.find_spec("klujax") is not None


@lru_cache(maxsize=1)
def _lineax_available() -> bool:
    """Return ``True`` when optional ``lineax`` is importable."""
    return importlib.util.find_spec("lineax") is not None


@lru_cache(maxsize=1)
def _umfpack_available() -> bool:
    """Return ``True`` when optional ``scikits.umfpack`` is importable."""
    # ``find_spec`` raises ``ModuleNotFoundError`` in Python 3.14+ when the
    # parent package (here ``scikits``) is not installed at all, so guard.
    try:
        return importlib.util.find_spec("scikits.umfpack") is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _warn_jax_auto_fallback_once(missing: str, target: str) -> None:
    """Emit a one-time advisory warning for JAX sparse backend auto-fallbacks."""
    install_hint = ""
    if missing == "scikits.umfpack":
        install_hint = " Install 'scikit-umfpack' to enable the UMFPACK callback path."
    elif missing == "klujax":
        install_hint = " Install 'klujax' to enable the faster JAX-native sparse path."
    warnings.warn(
        "BAYESPECON_JAX_SPARSE_BACKEND=auto selected fallback backend "
        f"'{target}' because optional dependency '{missing}' is not installed. "
        f"Estimation is likely faster when the optional sparse backend is installed.{install_hint}",
        RuntimeWarning,
        stacklevel=3,
    )


@lru_cache(maxsize=1)
def _select_jax_sparse_backend() -> str:
    """Resolve JAX sparse backend from env vars with robust fallback.

    Environment
    -----------
    BAYESPECON_JAX_SPARSE_BACKEND : {"auto", "callback", "klujax"}
        Default ``auto``. ``auto`` prefers ``klujax`` when available.
    BAYESPECON_JAX_SPARSE_STRICT : {"0", "1", "false", "true"}
        If truthy, missing requested optional backends raise ImportError.
    """
    requested = os.environ.get("BAYESPECON_JAX_SPARSE_BACKEND", "auto").strip().lower()
    strict = os.environ.get("BAYESPECON_JAX_SPARSE_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if requested in {"", "auto"}:
        if _klujax_available():
            return "klujax"
        # JAX path fallback chain:
        #   1) klujax
        #   2) callback + umfpack
        #   3) callback + scipy
        # The callback solver selection is handled in ops._select_sparse_backend.
        if _umfpack_available():
            _warn_jax_auto_fallback_once("klujax", "callback+umfpack")
        else:
            _warn_jax_auto_fallback_once("klujax", "callback+scipy")
            _warn_jax_auto_fallback_once("scikits.umfpack", "callback+scipy")
        return "callback"

    if requested in {"callback", "scipy", "pure_callback"}:
        return "callback"

    if requested in {"klu", "klujax"}:
        if _klujax_available():
            return "klujax"
        msg = (
            "BAYESPECON_JAX_SPARSE_BACKEND=klujax requested, but optional "
            "dependency 'klujax' is not installed. Falling back to callback backend."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "callback"

    msg = (
        f"Unknown BAYESPECON_JAX_SPARSE_BACKEND='{requested}'. "
        "Valid values are: auto, callback, klujax. Falling back to auto."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "klujax" if _klujax_available() else "callback"


def _strict_env() -> bool:
    return os.environ.get("BAYESPECON_JAX_SPARSE_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Threshold above which the auto solver switches from eigen to lineax.
# Eigen uses O(N^2) memory (two N×N complex128 matrices) and O(N^3)
# eigendecomposition time, which becomes prohibitive for very large N.
# Lineax is matrix-free and uses O(nnz) memory with O(nnz) per iteration,
# but can fail for near-singular systems during NUTS warmup.
#
# Default is 0 (eigen path disabled). The eigen path materialises three
# N×N complex128 matrices (eigenvalues, eigenvectors, inverse eigenvectors)
# plus a dense N×N float64 W for the gradient — totalling ~24N² bytes.
# For n=2000 this is ~96 MB of GPU constants that XLA must trace through,
# causing multi-minute JIT compilation. The callback path (scipy sparse LU
# via host callback) avoids all of this and is faster for n > ~500.
#
# Set BAYESPECON_JAX_SAR_EIGEN_N_MAX to a positive value to re-enable
# the eigen path for small problems where it may be slightly faster per-step.
_JAX_SAR_EIGEN_N_MAX = int(os.environ.get("BAYESPECON_JAX_SAR_EIGEN_N_MAX", "0"))


def _resolve_auto_sar_solver(n: int) -> str:
    """Resolve ``"auto"`` to a concrete solver based on problem size *n*.

    Selection order:

    1. ``eigen`` when *n* is at or below ``BAYESPECON_JAX_SAR_EIGEN_N_MAX``
       (default 0, i.e. opt-in only). The eigen path materialises three
       N×N complex128 matrices plus a dense N×N float64 W and triggers
       multi-minute XLA compile times for n > ~500, so we keep it gated.
    2. ``klujax`` when installed. Per
       ``scripts/benchmarks/lineax_sar_benchmark.csv``, klujax is the
       fastest pure sparse path in JAX (0.54 ms at n=1024 vs 0.72 ms
       for lineax-bicgstab) and is robust at the boundary of the
       stationary region where the iterative Krylov paths can stall.
    3. ``lineax`` when installed but klujax is not. Matrix-free
       iterative solve; can return NaN near singular operators (NUTS
       rejects those steps).
    4. ``callback`` as the final fallback (scipy splu via host
       callback). Always available since scipy is a hard dependency.
    """
    if n <= _JAX_SAR_EIGEN_N_MAX:
        return "eigen"
    if _klujax_available():
        return "klujax"
    if _lineax_available():
        return "lineax"
    return "jax_gmres"


@lru_cache(maxsize=1)
def _select_jax_sar_solver() -> str:
    """Resolve the JAX SAR solver from env vars.

    Returns one of ``"auto"``, ``"eigen"``, ``"callback"``, ``"klujax"``, ``"lineax"``.

    ``"auto"`` is resolved to a concrete solver at Op registration time
    by :func:`_resolve_auto_sar_solver` based on the problem size *n*.

    Environment
    -----------
    BAYESPECON_JAX_SAR_SOLVER : {"auto", "eigen", "callback", "klujax", "lineax"}
        Default ``auto``. ``auto`` selects ``eigen`` when
        N ≤ ``BAYESPECON_JAX_SAR_EIGEN_N_MAX`` (default 0, i.e. opt-in),
        otherwise ``klujax`` when installed, else ``lineax`` when
        installed, else ``callback``. ``eigen`` is a pure-JAX
        eigendecomposition path that avoids sparse LU factorisation
        entirely. ``callback`` wraps scipy via host callback.
    BAYESPECON_JAX_SAR_EIGEN_N_MAX : int, default 0
        Maximum N for which ``auto`` selects the eigen path. Default
        0 disables eigen in ``auto`` because the dense materialisation
        triggers multi-minute XLA compile times for n > ~500.
    BAYESPECON_JAX_SPARSE_STRICT : truthy
        If set, missing requested optional dependencies raise ImportError
        instead of falling back.
    """
    requested = os.environ.get("BAYESPECON_JAX_SAR_SOLVER", "auto").strip().lower()
    strict = _strict_env()

    if requested in {"", "auto"}:
        return "auto"

    if requested == "eigen":
        return "eigen"

    if requested in {"callback", "scipy", "pure_callback"}:
        return "callback"

    if requested in {"klu", "klujax"}:
        if _klujax_available():
            return "klujax"
        msg = (
            "BAYESPECON_JAX_SAR_SOLVER=klujax requested, but optional "
            "dependency 'klujax' is not installed. Falling back to callback."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "callback"

    if requested == "lineax":
        if _lineax_available():
            return "lineax"
        msg = (
            "BAYESPECON_JAX_SAR_SOLVER=lineax requested, but optional "
            "dependency 'lineax' is not installed. Falling back to jax_gmres."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "jax_gmres"

    if requested in {"jax_gmres", "gmres", "jaxgmres"}:
        return "jax_gmres"

    msg = (
        f"Unknown BAYESPECON_JAX_SAR_SOLVER='{requested}'. "
        "Valid values are: auto, eigen, callback, klujax, lineax, jax_gmres. Falling back to auto."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "auto"


@lru_cache(maxsize=1)
def _select_jax_sar_lineax_solver() -> str:
    """Resolve the Lineax iterative solver (``bicgstab`` or ``gmres``)."""
    requested = (
        os.environ.get("BAYESPECON_JAX_SAR_LINEAX_SOLVER", "bicgstab").strip().lower()
    )
    if requested in {"", "bicgstab"}:
        return "bicgstab"
    if requested == "gmres":
        return "gmres"
    msg = (
        f"Unknown BAYESPECON_JAX_SAR_LINEAX_SOLVER='{requested}'. "
        "Valid values are: bicgstab, gmres. Falling back to bicgstab."
    )
    if _strict_env():
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "bicgstab"


@lru_cache(maxsize=1)
def _select_jax_sar_lineax_precond() -> str:
    """Resolve the Lineax SAR preconditioner kind.

    Environment variable
    --------------------
    BAYESPECON_JAX_SAR_LINEAX_PRECOND : {"neumann", "none"}
        Default ``"neumann"``. The Neumann-series left preconditioner
        :math:`M^{-1} \\approx \\sum_{j=0}^{k} \\rho^j W^j` is exact in the
        limit :math:`k \\to \\infty` when :math:`|\\rho| \\, \\sigma(W) < 1` and
        strongly accelerates BiCGStab/GMRES convergence on the SAR system
        :math:`(I - \\rho W) x = b` near the upper end of the stability
        region. Disable with ``"none"`` to recover the unpreconditioned
        behaviour (rarely useful — mostly for benchmarking).
    """
    requested = (
        os.environ.get("BAYESPECON_JAX_SAR_LINEAX_PRECOND", "neumann").strip().lower()
    )
    if requested in {"", "neumann"}:
        return "neumann"
    if requested == "none":
        return "none"
    msg = (
        f"Unknown BAYESPECON_JAX_SAR_LINEAX_PRECOND='{requested}'. "
        "Valid values are: neumann, none. Falling back to neumann."
    )
    if _strict_env():
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "neumann"


@lru_cache(maxsize=1)
def _select_jax_sar_lineax_neumann_k() -> int:
    """Resolve the Neumann-series truncation order ``k`` (default 3).

    Environment variable
    --------------------
    BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K : int, default 3
        Number of Neumann correction terms. ``k = 0`` is equivalent to
        ``BAYESPECON_JAX_SAR_LINEAX_PRECOND=none``. Each extra term costs
        one additional sparse mat-vec per Krylov iteration but reduces the
        effective spectral radius geometrically.
    """
    raw = os.environ.get("BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K", "3").strip()
    try:
        k = int(raw)
    except ValueError:
        msg = (
            f"BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K='{raw}' is not an integer. "
            "Falling back to 3."
        )
        if _strict_env():
            raise ValueError(msg) from None
        warnings.warn(msg, RuntimeWarning)
        return 3
    if k < 0:
        msg = (
            f"BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K={k} must be >= 0. Falling back to 3."
        )
        if _strict_env():
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)
        return 3
    return k


@lru_cache(maxsize=1)
def register_jax_dispatch() -> bool:
    """Register JAX dispatches for all Ops in :mod:`bayespecon.ops`.

    Idempotent (cached). Returns ``True`` if registration ran, ``False`` if
    JAX is not available.
    """
    if not _jax_available():
        return False

    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsla
    import jax.scipy.sparse.linalg as jssl
    import numpy as np
    import scipy.sparse as sp
    from pytensor.link.jax.dispatch import jax_funcify

    sparse_backend = _select_jax_sparse_backend()
    klujax = None
    if sparse_backend == "klujax":
        import klujax

    sar_solver = _select_jax_sar_solver()
    lineax_solver_name = _select_jax_sar_lineax_solver()
    lineax_precond_kind = _select_jax_sar_lineax_precond()
    lineax_neumann_k = _select_jax_sar_lineax_neumann_k()

    from ._ops import (
        KroneckerFlowSolveMatrixOp,
        KroneckerFlowSolveOp,
        SparseFlowSolveMatrixOp,
        SparseFlowSolveOp,
        SparseSARSolveOp,
        _KroneckerFlowVJPMatrixOp,
        _KroneckerFlowVJPOp,
        _SparseFlowVJPMatrixOp,
        _SparseFlowVJPOp,
        _SparseSARVJPOp,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dense(W):
        """scipy CSR -> dense float64 numpy array (closed over once)."""
        return np.asarray(W.toarray(), dtype=np.float64)

    def _reshape_F(arr, shape):
        """Equivalent to ``arr.reshape(shape, order='F')`` for 2D targets.

        For a 1D ``arr`` of length ``n*n`` reshaped to ``(n, n)`` Fortran-order,
        this is ``arr.reshape((n, n)).T``.
        """
        # Only used here for 1D -> (n, n)
        n = shape[0]
        return arr.reshape((n, n)).T

    def _ravel_F_2d(M):
        """Equivalent to ``M.ravel(order='F')`` for a 2D array."""
        return M.T.reshape(-1)

    # ------------------------------------------------------------------
    # Kronecker forward — pure JAX
    # ------------------------------------------------------------------

    @jax_funcify.register(KroneckerFlowSolveOp)
    def _funcify_kron_solve(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def kron_solve(rho_d, rho_o, b):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            Hb = _reshape_F(b, (n, n))  # (n, n)
            Hp = jsla.solve(Ld, Hb)  # Ld H' = Hb
            Z = jsla.solve(Lo, Hp.T)  # Lo Z = Hp^T
            # perform: Z.T.ravel(order='F') == Z.ravel()
            return Z.reshape(-1)

        return kron_solve

    # ------------------------------------------------------------------
    # Kronecker VJP — pure JAX
    # ------------------------------------------------------------------

    @jax_funcify.register(_KroneckerFlowVJPOp)
    def _funcify_kron_vjp(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def kron_vjp(rho_d, rho_o, eta, g):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d

            H_eta = _reshape_F(eta, (n, n))  # (n, n)
            Hg = _reshape_F(g, (n, n))  # (n, n)

            # Adjoint: (Lo^T ⊗ Ld^T) v = g  =>  Ld^T H_v Lo = Hg
            P = jsla.solve(Ld.T, Hg)  # Ld^T P = Hg
            Q = jsla.solve(Lo.T, P.T)  # Lo^T Q = P^T  (Q = H_v^T)
            H_v = Q.T  # (n, n)

            W_H = W_d @ H_eta  # (n, n)
            Ld_H = Ld @ H_eta  # (n, n)
            grad_rd = jnp.sum(H_v * (W_H @ Lo.T))
            grad_ro = jnp.sum(H_v * (Ld_H @ W_d.T))
            grad_b = _ravel_F_2d(H_v)
            return grad_rd, grad_ro, grad_b

        return kron_vjp

    # ------------------------------------------------------------------
    # Kronecker matrix forward / VJP — vmap over the single-vector path
    # ------------------------------------------------------------------

    @jax_funcify.register(KroneckerFlowSolveMatrixOp)
    def _funcify_kron_solve_matrix(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def _solve_one(rho_d, rho_o, b):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            Hb = _reshape_F(b, (n, n))
            Hp = jsla.solve(Ld, Hb)
            Z = jsla.solve(Lo, Hp.T)
            return Z.reshape(-1)

        def kron_solve_mat(rho_d, rho_o, B):
            # vmap over the column (T) axis: B is (N, T) -> output (T, N) -> (N, T)
            solver = jax.vmap(_solve_one, in_axes=(None, None, 1), out_axes=1)
            return solver(rho_d, rho_o, B)

        return kron_solve_mat

    @jax_funcify.register(_KroneckerFlowVJPMatrixOp)
    def _funcify_kron_vjp_matrix(op, **kwargs):
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def _vjp_one(rho_d, rho_o, eta_col, g_col):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            H_eta = _reshape_F(eta_col, (n, n))
            Hg = _reshape_F(g_col, (n, n))
            P = jsla.solve(Ld.T, Hg)
            Q = jsla.solve(Lo.T, P.T)
            H_v = Q.T
            W_H = W_d @ H_eta
            Ld_H = Ld @ H_eta
            grad_rd = jnp.sum(H_v * (W_H @ Lo.T))
            grad_ro = jnp.sum(H_v * (Ld_H @ W_d.T))
            grad_b = _ravel_F_2d(H_v)
            return grad_rd, grad_ro, grad_b

        def kron_vjp_mat(rho_d, rho_o, H_eta, G):
            # vmap over column axis; sum scalar grads, stack vector grad
            vjper = jax.vmap(_vjp_one, in_axes=(None, None, 1, 1), out_axes=(0, 0, 1))
            grad_rd_per_t, grad_ro_per_t, grad_B = vjper(rho_d, rho_o, H_eta, G)
            return jnp.sum(grad_rd_per_t), jnp.sum(grad_ro_per_t), grad_B

        return kron_vjp_mat

    # ------------------------------------------------------------------
    # Sparse Ops — wrap scipy splu via jax.pure_callback
    # ------------------------------------------------------------------
    #
    # JAX has no CPU sparse direct solver. We use a host callback that runs
    # the existing perform() logic. Two distinct gradient paths must work:
    #
    # 1. PyTensor's symbolic L_op path inserts the VJP node into the graph
    #    BEFORE JAX transpilation.  Each VJP node has its own callback
    #    dispatch, so JAX never differentiates across the callback here.
    # 2. PyMC's JAX samplers (blackjax, numpyro) compile only the forward
    #    log-density and then call ``jax.grad`` on it.  ``jax.grad`` traces
    #    through ``pure_callback`` and raises ``Pure callbacks do not
    #    support JVP``.  To make this path work we wrap the forward solve
    #    in ``jax.custom_vjp`` — the bwd rule calls the existing analytic
    #    adjoint via another ``pure_callback``.

    def _make_solve_with_custom_vjp(forward_op, vjp_op, *, matrix: bool):
        """Build a ``custom_vjp``-decorated solver that reuses the Op callbacks."""

        def _host_solve(rd, ro, rw, rhs):
            outputs = [[None]]
            forward_op.perform(
                None,
                [np.asarray(rd), np.asarray(ro), np.asarray(rw), np.asarray(rhs)],
                outputs,
            )
            return outputs[0][0]

        def _host_vjp(rd, ro, rw, sol, g):
            outputs = [[None], [None], [None], [None]]
            vjp_op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(sol),
                    np.asarray(g),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        @jax.custom_vjp
        def solve(rho_d, rho_o, rho_w, rhs):
            return jax.pure_callback(
                _host_solve,
                jax.ShapeDtypeStruct(rhs.shape, jnp.float64),
                rho_d,
                rho_o,
                rho_w,
                rhs,
                vmap_method="sequential",
            )

        def solve_fwd(rho_d, rho_o, rho_w, rhs):
            sol = solve(rho_d, rho_o, rho_w, rhs)
            return sol, (rho_d, rho_o, rho_w, sol)

        def solve_bwd(residuals, g):
            rho_d, rho_o, rho_w, sol = residuals
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(sol.shape, jnp.float64),
            )
            grad_rd, grad_ro, grad_rw, grad_rhs = jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                sol,
                g,
                vmap_method="sequential",
            )
            return grad_rd, grad_ro, grad_rw, grad_rhs

        solve.defvjp(solve_fwd, solve_bwd)
        return solve, _host_vjp

    @jax_funcify.register(SparseFlowSolveOp)
    def _funcify_sparse_solve(op, **kwargs):
        solve, _ = _make_solve_with_custom_vjp(op, op._vjp_op, matrix=False)

        def sparse_solve(rho_d, rho_o, rho_w, b):
            return solve(rho_d, rho_o, rho_w, b)

        return sparse_solve

    @jax_funcify.register(_SparseFlowVJPOp)
    def _funcify_sparse_vjp(op, **kwargs):
        # Used by PyTensor's symbolic L_op path. Pure callback is fine
        # here because PyTensor never differentiates through this node
        # (it IS the gradient).
        def _host_vjp(rd, ro, rw, eta, g):
            outputs = [[None], [None], [None], [None]]
            op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(eta),
                    np.asarray(g),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        def sparse_vjp(rho_d, rho_o, rho_w, eta, g):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(eta.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                eta,
                g,
                vmap_method="sequential",
            )

        return sparse_vjp

    @jax_funcify.register(SparseFlowSolveMatrixOp)
    def _funcify_sparse_solve_matrix(op, **kwargs):
        solve, _ = _make_solve_with_custom_vjp(op, op._vjp_op, matrix=True)

        def sparse_solve_mat(rho_d, rho_o, rho_w, B):
            return solve(rho_d, rho_o, rho_w, B)

        return sparse_solve_mat

    @jax_funcify.register(_SparseFlowVJPMatrixOp)
    def _funcify_sparse_vjp_matrix(op, **kwargs):
        def _host_vjp(rd, ro, rw, H, G):
            outputs = [[None], [None], [None], [None]]
            op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(H),
                    np.asarray(G),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        def sparse_vjp_mat(rho_d, rho_o, rho_w, H, G):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(H.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                H,
                G,
                vmap_method="sequential",
            )

        return sparse_vjp_mat

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — wrap scipy splu via jax.pure_callback
    # ------------------------------------------------------------------

    def _make_sar_solve_with_custom_vjp(forward_op, vjp_op):
        """Build a custom_vjp wrapper for SparseSARSolveOp."""

        def _host_solve(rho, rhs):
            outputs = [[None]]
            forward_op.perform(
                None,
                [np.asarray(rho), np.asarray(rhs)],
                outputs,
            )
            return outputs[0][0]

        def _host_vjp(rho, sol, g):
            outputs = [[None], [None]]
            vjp_op.perform(
                None,
                [np.asarray(rho), np.asarray(sol), np.asarray(g)],
                outputs,
            )
            return (outputs[0][0], outputs[1][0])

        @jax.custom_vjp
        def solve(rho, rhs):
            return jax.pure_callback(
                _host_solve,
                jax.ShapeDtypeStruct(rhs.shape, jnp.float64),
                rho,
                rhs,
                vmap_method="sequential",
            )

        def solve_fwd(rho, rhs):
            sol = solve(rho, rhs)
            return sol, (rho, sol)

        def solve_bwd(residuals, g):
            rho, sol = residuals
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                jax.ShapeDtypeStruct(sol.shape, jnp.float64),
            )
            grad_rho, grad_rhs = jax.pure_callback(
                _host_vjp,
                shapes,
                rho,
                sol,
                g,
                vmap_method="sequential",
            )
            return grad_rho, grad_rhs

        solve.defvjp(solve_fwd, solve_bwd)
        return solve

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — Lineax matrix-free iterative solve
    # ------------------------------------------------------------------

    def _build_lineax_sar_paths(op):
        """Return ``(forward_fn, vjp_fn)`` for the Lineax SAR path.

        Both functions are pure-JAX and tracer-compatible. The forward solves
        :math:`(I - \\rho W) \\eta = b` and the VJP solves the adjoint system
        :math:`(I - \\rho W^\\top) v = g` matrix-free over a BCOO ``W``.

        A truncated Neumann-series left preconditioner

        .. math::

            M^{-1} \\approx \\sum_{j=0}^{k} \\rho^{\\,j} W^{\\,j}

        is applied to both the operator and the right-hand side (and its
        transpose for the adjoint path) when
        ``BAYESPECON_JAX_SAR_LINEAX_PRECOND="neumann"`` (default). The
        truncation order ``k`` is set by
        ``BAYESPECON_JAX_SAR_LINEAX_NEUMANN_K`` (default 3). Each extra
        term costs one additional sparse mat-vec per Krylov iteration but
        clusters the spectrum of :math:`M^{-1} A` tightly around 1,
        which dramatically reduces BiCGStab breakdown and GMRES restart
        cost near :math:`|\\rho|\\,\\sigma(W) \\to 1`.
        """
        import lineax as lx
        from jax.experimental import sparse as jsparse

        n = op._n
        W_bcoo = jsparse.BCOO.from_scipy_sparse(op._W)
        W_T_bcoo = jsparse.BCOO.from_scipy_sparse(op._W.transpose().tocsr())
        max_steps = max(int(10 * n), 5000)
        rtol = 1e-8
        atol = 1e-8

        use_precond = lineax_precond_kind == "neumann" and lineax_neumann_k > 0
        neumann_k = lineax_neumann_k

        def _make_solver():
            if lineax_solver_name == "gmres":
                return lx.GMRES(rtol=rtol, atol=atol, max_steps=max_steps, restart=20)
            return lx.BiCGStab(rtol=rtol, atol=atol, max_steps=max_steps)

        def _apply_minv(rho, W_matvec, x):
            """Apply the truncated Neumann preconditioner :math:`M^{-1} x`.

            ``M^{-1} x = x + rho * W (x + rho * W (x + ...))`` evaluated
            via Horner-style nesting so each term reuses the previous
            mat-vec result (``k`` total mat-vecs of ``W_matvec``).
            """
            out = x
            term = x
            for _ in range(neumann_k):
                term = rho * W_matvec(term)
                out = out + term
            return out

        def _solve(W_matvec, rho, b):
            if use_precond:

                def matvec(x):
                    Ax = x - rho * W_matvec(x)
                    return _apply_minv(rho, W_matvec, Ax)

                rhs = _apply_minv(rho, W_matvec, b)
            else:

                def matvec(x):
                    return x - rho * W_matvec(x)

                rhs = b

            structure = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            operator = lx.FunctionLinearOperator(matvec, structure)
            # ``throw=False`` so that non-convergence / near-singular
            # systems return NaNs rather than raising Equinox runtime
            # errors. NUTS rejects steps with non-finite log-prob or
            # gradient, which is the correct behaviour at the edge of
            # the stationary region. Raising would flood stderr with
            # tracebacks on every rejected leapfrog proposal.
            solution = lx.linear_solve(operator, rhs, _make_solver(), throw=False)
            return solution.value

        def forward(rho, b):
            return _solve(lambda x: W_bcoo @ x, rho, b)

        def vjp(rho, eta, g):
            # Adjoint system: A^T v = g. Left-preconditioning with
            # M^{-T} = sum_j rho^j (W^T)^j leaves the solution unchanged
            # and accelerates convergence the same way as the forward path.
            v = _solve(lambda x: W_T_bcoo @ x, rho, g)
            grad_rho = jnp.vdot(v, W_bcoo @ eta)
            return grad_rho, v

        return forward, vjp

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — JAX native GMRES with BCOO
    # ------------------------------------------------------------------

    def _build_jax_gmres_sar_paths(op):
        """Return ``(forward_fn, vjp_fn)`` for the JAX-native GMRES path.

        Uses :class:`jax.experimental.sparse.BCOO` for the weight matrix
        and :func:`jax.scipy.sparse.linalg.gmres` with a diagonal
        (Jacobi) preconditioner.  This path is JIT-compilable, vmappable,
        and differentiable — no host callbacks required.

        The diagonal preconditioner is nearly optimal for 2-D lattice
        spatial weights (bounded degree ≈ 4) and keeps GMRES iteration
        counts low for typical ρ ∈ [0.3, 0.7].

        Parameters
        ----------
        op : SparseSARSolveOp
            The Op being dispatched; ``op._W`` and ``op._n`` are used.

        Returns
        -------
        forward : callable
            ``forward(rho, b) -> eta``.
        vjp : callable
            ``vjp(rho, eta, g) -> (grad_rho, grad_b)``.
        """
        from jax.experimental import sparse as jsparse

        W_bcoo = jsparse.BCOO.from_scipy_sparse(op._W)
        W_T_bcoo = jsparse.BCOO.from_scipy_sparse(op._W.transpose().tocsr())

        # Diagonal preconditioner: M^{-1} = diag(1 / (1 - rho * diag(W)))
        # For row-standardised W, diag(W) is the self-loop weight.
        W_diag_np = np.asarray(op._W.diagonal(), dtype=np.float64)
        W_diag_j = jnp.asarray(W_diag_np, dtype=jnp.float64)

        # GMRES settings — tuned for spatial SAR systems on lattice graphs.
        _GMRES_TOL = float(os.environ.get("BAYESPECON_JAX_GMRES_TOL", "1e-8"))
        _GMRES_MAXITER = int(os.environ.get("BAYESPECON_JAX_GMRES_MAXITER", "100"))
        _GMRES_RESTART = int(os.environ.get("BAYESPECON_JAX_GMRES_RESTART", "20"))

        def _solve(W_matvec, rho, b):
            # Build diagonal preconditioner for this rho
            diag_inv = 1.0 / (1.0 - rho * W_diag_j)

            def matvec(x):
                return x - rho * W_matvec(x)

            def precond(x):
                return diag_inv * x

            x, info = jssl.gmres(
                matvec,
                b,
                tol=_GMRES_TOL,
                atol=0.0,
                maxiter=_GMRES_MAXITER,
                restart=_GMRES_RESTART,
                M=precond,
            )
            # info == 0 means converged; non-zero means maxiter reached.
            # Return the iterate regardless — NUTS will reject non-finite
            # log-prob if the solve is poor.
            return x

        def forward(rho, b):
            return _solve(lambda x: W_bcoo @ x, rho, b)

        def vjp(rho, eta, g):
            # Adjoint system: (I - rho W^T) v = g
            v = _solve(lambda x: W_T_bcoo @ x, rho, g)
            grad_rho = jnp.vdot(v, W_bcoo @ eta)
            return grad_rho, v

        return forward, vjp

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — eigendecomposition path (default)
    # ------------------------------------------------------------------

    def _build_eigen_sar_paths(op):
        """Return ``(forward_fn, vjp_fn)`` for the eigen SAR path.

        Precomputes the eigendecomposition of W once, then solves
        ``(I - rho W)^{-1} b = V @ diag(1/(1 - rho*lambda)) @ V^{-1} @ b``
        using pure dense JAX operations.  This avoids sparse LU factorisation
        entirely and is robust to near-singular system matrices that cause
        ``klu_factor`` to segfault or raise ``INVALID_ARGUMENT``.

        Row-standardised spatial weight matrices are generally non-symmetric,
        so the eigendecomposition uses complex arithmetic.  The final result
        is real-valued (the imaginary parts cancel), and JAX's autodiff
        correctly propagates gradients through the complex→real conversion.

        The gradient w.r.t. ``rho`` is ``v^T W eta`` where
        ``v = (I - rho W^T)^{-1} g``.

        If the Op was constructed with a shared ``eigendecomposition``
        cache (from the model's ``_W_eigendecomposition`` property), it
        is reused here to avoid a redundant O(n³) decomposition.
        """

        # Consume shared eigendecomposition cache if available.
        if op._eigendecomposition is not None:
            eigs_np, V_np, Vinv_np = op._eigendecomposition
        else:
            W_dense = np.asarray(op._W.toarray(), dtype=np.float64)
            eigs_np, V_np = np.linalg.eig(W_dense)
            Vinv_np = np.linalg.inv(V_np)
            # Sort eigenvalues by real part (descending) for numerical stability.
            idx = np.argsort(eigs_np.real)[::-1]
            eigs_np = eigs_np[idx]
            V_np = V_np[:, idx]
            Vinv_np = Vinv_np[idx, :]

        # Use complex128 to handle non-symmetric W correctly.
        # Row-standardised W can have complex eigenvalues/eigenvectors.
        eigs_j = jnp.asarray(eigs_np.astype(np.complex128))
        V_j = jnp.asarray(V_np.astype(np.complex128))
        Vinv_j = jnp.asarray(Vinv_np.astype(np.complex128))
        # Materialize dense W for the gradient (W @ eta).
        # This is O(n²) — dominated by the O(n³) eigendecomposition
        # that we either reuse from cache or compute above.
        W_dense_for_grad = np.asarray(op._W.toarray(), dtype=np.float64)
        W_j = jnp.asarray(W_dense_for_grad, dtype=jnp.float64)

        def forward(rho, b):
            inv_eigs = 1.0 / (1.0 - rho * eigs_j)
            return (V_j @ (inv_eigs * (Vinv_j @ b.astype(jnp.complex128)))).real

        def vjp(rho, eta, g):
            # Adjoint: v = (I - rho W^T)^{-1} g
            # (I - rho W^T)^{-1} = V^{-T} @ diag(1/(1-rho*lambda)) @ V^T g
            # where V^{-T} = conj(Vinv) for the eigendecomposition W = V diag(lam) V^{-1}
            inv_eigs = 1.0 / (1.0 - rho * eigs_j)
            g_c = g.astype(jnp.complex128)
            v = (jnp.conj(Vinv_j).T @ (inv_eigs * (jnp.conj(V_j).T @ g_c))).real
            grad_rho = jnp.vdot(v, W_j @ eta)
            return grad_rho, v

        return forward, vjp

    @jax_funcify.register(SparseSARSolveOp)
    def _funcify_sparse_sar_solve(op, **kwargs):
        # Resolve "auto" to a concrete solver based on problem size.
        resolved = (
            _resolve_auto_sar_solver(op._n) if sar_solver == "auto" else sar_solver
        )

        if resolved == "eigen":
            forward, _ = _build_eigen_sar_paths(op)

            def sparse_sar_solve(rho, b):
                return forward(rho, b)

            return sparse_sar_solve

        if resolved == "lineax":
            forward, _ = _build_lineax_sar_paths(op)

            def sparse_sar_solve(rho, b):
                return forward(rho, b)

            return sparse_sar_solve

        if resolved == "jax_gmres":
            forward, _ = _build_jax_gmres_sar_paths(op)

            def sparse_sar_solve(rho, b):
                return forward(rho, b)

            return sparse_sar_solve

        if resolved == "klujax" or (
            sar_solver != "auto" and sparse_backend == "klujax"
        ):
            n = op._n
            I = np.eye(n, dtype=np.float64)
            W_dense = np.asarray(op._W.toarray(), dtype=np.float64)
            A_pat = (sp.eye(n, format="csr", dtype=np.float64) + op._W).tocoo()
            Ai = jnp.asarray(np.asarray(A_pat.row, dtype=np.int32))
            Aj = jnp.asarray(np.asarray(A_pat.col, dtype=np.int32))
            const_vals = jnp.asarray(I[A_pat.row, A_pat.col], dtype=jnp.float64)
            w_vals = jnp.asarray(W_dense[A_pat.row, A_pat.col], dtype=jnp.float64)
            # Sparsity pattern is fixed for A(rho)=I-rho*W, so analyze once and
            # reuse symbolic metadata across all solves.
            symbolic = klujax.analyze(Ai, Aj, n)

            def sparse_sar_solve(rho, b):
                Ax = const_vals - rho * w_vals
                return klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)

            return sparse_sar_solve

        solve = _make_sar_solve_with_custom_vjp(op, op._vjp_op)

        def sparse_sar_solve(rho, b):
            return solve(rho, b)

        return sparse_sar_solve

    @jax_funcify.register(_SparseSARVJPOp)
    def _funcify_sparse_sar_vjp(op, **kwargs):
        # Resolve "auto" to a concrete solver based on problem size.
        resolved = (
            _resolve_auto_sar_solver(op._n) if sar_solver == "auto" else sar_solver
        )

        if resolved == "eigen":
            _, vjp = _build_eigen_sar_paths(op)

            def sparse_sar_vjp(rho, eta, g):
                return vjp(rho, eta, g)

            return sparse_sar_vjp

        if resolved == "lineax":
            _, vjp = _build_lineax_sar_paths(op)

            def sparse_sar_vjp(rho, eta, g):
                return vjp(rho, eta, g)

            return sparse_sar_vjp

        if resolved == "jax_gmres":
            _, vjp = _build_jax_gmres_sar_paths(op)

            def sparse_sar_vjp(rho, eta, g):
                return vjp(rho, eta, g)

            return sparse_sar_vjp

        # Used by PyTensor's symbolic L_op path. This node is itself the
        # gradient, so pure_callback is sufficient.
        def _host_vjp(rho, eta, g):
            outputs = [[None], [None]]
            op.perform(
                None,
                [np.asarray(rho), np.asarray(eta), np.asarray(g)],
                outputs,
            )
            return (outputs[0][0], outputs[1][0])

        def sparse_sar_vjp(rho, eta, g):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                jax.ShapeDtypeStruct(eta.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho,
                eta,
                g,
                vmap_method="sequential",
            )

        return sparse_sar_vjp

    return True
