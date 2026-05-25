"""JAX-native log-determinant evaluation functions.

These mirror the pytensor symbolic functions (logdet_chebyshev,
logdet_mc_poly_pytensor) but use jax.numpy so they can be called
inside jax.jit and are autodiff-compatible via jax.grad.
"""

import numpy as np
import scipy.sparse as sp

from ._config import (
    TraceEstimatorName,
    _default_trace_k,
    _resolve_trace_estimator,
    resolve_logdet_method,
)
from ._grids import chebyshev


def jax_logdet_chebyshev(
    rho,
    coeffs: np.ndarray,
    rmin: float = -1.0,
    rmax: float = 1.0,
):
    """Evaluate Chebyshev approximation of log|I - rho*W| in JAX.

    JAX-native version of :func:`logdet_chebyshev` using Clenshaw's
    algorithm.  Fully compatible with ``jax.jit`` and ``jax.grad``.

    Parameters
    ----------
    rho : jax.numpy scalar or array
        Spatial autoregressive parameter.  Can be a scalar or an
        array of shape ``(G,)`` for vectorized evaluation over
        posterior draws.
    coeffs : np.ndarray, shape (m,)
        Chebyshev coefficients from :func:`chebyshev`.
    rmin : float, default=-1.0
        Lower bound of the rho interval (must match what was used to
        compute *coeffs*).
    rmax : float, default=1.0
        Upper bound of the rho interval (must match what was used to
        compute *coeffs*).

    Returns
    -------
    jax.numpy.ndarray
        Chebyshev approximation of the log-determinant.  Same shape
        as *rho*.

    Notes
    -----
    The mapped variable is

    .. math::

        x = \\frac{2\\rho - r_{\\max} - r_{\\min}}{r_{\\max} - r_{\\min}}

    and the approximation is evaluated via Clenshaw's recurrence:

    .. math::

        b_{m+1} = 0, \\quad b_m = c_m

        b_k = 2x \\, b_{k+1} - b_{k+2} + c_k

        f(x) = x \\, b_1 - b_2 + c_0

    This is the same algorithm as :func:`logdet_chebyshev` but uses
    ``jax.numpy`` instead of pytensor, making it compatible with
    ``jax.jit`` and ``jax.grad``.

    See Also
    --------
    chebyshev : Compute Chebyshev coefficients from W.
    logdet_chebyshev : PyTensor symbolic version (for NUTS).
    """
    import jax.numpy as jnp

    m = len(coeffs)
    if m == 0:
        return jnp.zeros_like(rho)

    # Map rho ∈ [rmin, rmax] → x ∈ [-1, 1]
    x = (2.0 * rho - rmax - rmin) / (rmax - rmin)

    if m == 1:
        # Only c_0 * T_0(x) = c_0
        return jnp.full_like(rho, coeffs[0])

    c = jnp.asarray(coeffs, dtype=jnp.float64)

    # Clenshaw's algorithm for Σ_{j=0}^{m-1} c_j T_j(x)
    # Iterate from k = m-1 down to k = 1:
    #   b_{m} = c_{m-1},  b_{m+1} = 0
    #   b_k = 2x b_{k+1} - b_{k+2} + c_k
    # Then: f(x) = c_0 + x*b_1 - b_2
    b_next = jnp.zeros_like(x)
    b_curr = jnp.broadcast_to(c[m - 1], jnp.shape(x))

    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + c[k]
        b_next = b_curr
        b_curr = b_new

    return c[0] + x * b_curr - b_next


def jax_logdet_trace_poly(
    rho,
    traces: np.ndarray,
):
    r"""Evaluate trace-polynomial approximation of log|I - rho*W| in JAX.

    JAX-native version of :func:`logdet_mc_poly_pytensor` using
    Horner's method.  Fully compatible with ``jax.jit`` and
    ``jax.grad``.

    Computes the truncated power-series approximation

    .. math::

        \log|I_n - \rho W| \approx -\sum_{k=1}^{m} \frac{\rho^k}{k}\,\hat{\tau}_k

    where :math:`\hat{\tau}_k \approx \text{tr}(W^k)` are stochastic
    trace estimates from :func:`~bayespecon._logdet._trace.traceax_traces`
    or :func:`~bayespecon._logdet._grids.compute_flow_traces`.

    Parameters
    ----------
    rho : jax.numpy scalar or array
        Spatial autoregressive parameter.  Can be a scalar or an
        array of shape ``(G,)`` for vectorized evaluation over
        posterior draws.
    traces : np.ndarray, shape (m,)
        Trace estimates ``traces[k-1] ≈ tr(W^k)`` for k=1..m.

    Returns
    -------
    jax.numpy.ndarray
        Polynomial approximation of the log-determinant.  Same shape
        as *rho*.

    Notes
    -----
    Horner evaluation of :math:`-\sum_{k=1}^m w_k \rho^k` where
    :math:`w_k = \hat{\tau}_k / k`:

    .. math::

        -\rho \bigl(w_1 + \rho(w_2 + \rho(\cdots + \rho\, w_m)\cdots)\bigr)

    This is the same algorithm as :func:`logdet_mc_poly_pytensor` but
    uses ``jax.numpy`` instead of pytensor, making it compatible with
    ``jax.jit`` and ``jax.grad``.

    See Also
    --------
    logdet_mc_poly_pytensor : PyTensor symbolic version (for NUTS).
    traceax_traces : Compute trace estimates via variance-reduced estimators.
    compute_flow_traces : Compute trace estimates via Barry-Pace Hutchinson.
    """
    import jax.numpy as jnp

    m = len(traces)
    if m == 0:
        return jnp.zeros_like(rho)

    k_arr = np.arange(1, m + 1, dtype=np.float64)
    w = jnp.asarray((traces / k_arr).astype(np.float64))

    # Horner's method, high-to-low coefficients
    result = jnp.broadcast_to(w[m - 1], jnp.shape(rho))
    for j in range(m - 2, -1, -1):
        result = result * rho + w[j]
    result = result * rho
    return -result


def make_logdet_jax_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
    trace_estimator: TraceEstimatorName = "hutchpp",
    trace_k: int | None = None,
):
    """Return a JAX-native function (rho) -> log|I - rho*W|.

    Companion to :func:`make_logdet_fn` (pytensor) and
    :func:`make_logdet_numpy_fn` (numpy) that returns a JAX-native
    callable suitable for use inside ``jax.jit`` and ``jax.grad``.

    Parameters
    ----------
    W : np.ndarray or scipy.sparse matrix
        Either a 2-D dense ``(n, n)`` spatial weights matrix **or** a 1-D
        array of pre-computed real eigenvalues.  Passing eigenvalues skips
        the O(n³) decomposition.
    method : str or None
        Auto-selected when ``None`` (``"eigenvalue"`` for ``n <= 500``
        else ``"chebyshev"``).  Supported values:

        ``"eigenvalue"`` — exact evaluation from eigenvalues, O(n) per call.
        ``"chebyshev"`` — Chebyshev polynomial via Clenshaw's algorithm,
        O(m) per call.  Coefficients are built from exact eigenvalues when
        ``n`` is small (or ``eigs`` is supplied); otherwise from a stochastic
        trace estimator selected by ``trace_estimator``.
    rho_min : float, default=-1.0
        Lower bound for the rho interval.
    rho_max : float, default=1.0
        Upper bound for the rho interval.
    T : int, default 1
        Panel time-period count.  The returned log-determinant is
        multiplied by *T*.
    trace_estimator : {"hutchinson", "hutchpp", "xtrace"}, default "hutchpp"
        Stochastic trace estimator used to build the Chebyshev
        coefficients when an eigendecomposition is unavailable.  Ignored
        when ``method="eigenvalue"`` or when eigenvalues are passed in.
    trace_k : int, optional
        Number of probe vectors for the trace estimator.  Defaults:
        ``30`` (hutchinson), ``50`` (hutchpp), ``25`` (xtrace).

    Returns
    -------
    callable
        Function ``(rho) -> jax.numpy.ndarray`` that computes
        log|I - rho*W| (or T * log|I - rho*W| for panel models).
        Fully compatible with ``jax.jit`` and ``jax.grad``.

    Raises
    ------
    ValueError
        If *method* is not one of the supported JAX-compatible methods.

    Notes
    -----
    Not all logdet methods have JAX-native implementations.  Grid/spline
    methods (``"grid_dense"``, ``"grid_sparse"``, ``"sparse_spline"``,
    ``"grid_mc"``, ``"grid_ilu"``) and ``"exact"`` are not supported
    because they rely on scipy or pytensor-specific operations that
    cannot be called inside ``jax.jit``.  Use ``"eigenvalue"`` or
    ``"chebyshev"`` instead.

    See Also
    --------
    make_logdet_fn : PyTensor symbolic version (for NUTS).
    make_logdet_numpy_fn : NumPy scalar version (for Python-loop Gibbs).
    make_logdet_numpy_vec_fn : NumPy vectorized version (for post-processing).
    """
    T = int(T)
    _JAX_METHODS = frozenset({"eigenvalue", "chebyshev"})

    trace_estimator = _resolve_trace_estimator(trace_estimator)
    _k = trace_k if trace_k is not None else _default_trace_k(trace_estimator)

    # Resolve W to eigenvalues or sparse matrix
    eigs = None
    W_arr = None
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

    method = resolve_logdet_method(method, n=n)

    if method not in _JAX_METHODS:
        raise ValueError(
            f"Method '{method}' does not have a JAX-native implementation. "
            f"JAX-compatible methods: {sorted(_JAX_METHODS)}. "
            f"Use 'eigenvalue' or 'chebyshev'."
        )

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(W_sparse.toarray()).real
        _eigs = eigs.real.astype(np.float64)

        def _jax_eigenvalue(rho):
            import jax.numpy as jnp

            eigs_jax = jnp.asarray(_eigs)
            result = jnp.sum(jnp.log(jnp.abs(1.0 - rho * eigs_jax)))
            return result if T == 1 else T * result

        return _jax_eigenvalue

    # method == "chebyshev"
    out = chebyshev(
        W_sparse if eigs is None else None,
        order=20,
        rmin=rho_min,
        rmax=rho_max,
        eigs=eigs,
        estimator=trace_estimator,
        n_mc_iter=_k,
    )
    coeffs = out["coeffs"].astype(np.float64)
    rmin_cb = float(out["rmin"])
    rmax_cb = float(out["rmax"])

    def _jax_chebyshev(rho):
        val = jax_logdet_chebyshev(rho, coeffs, rmin=rmin_cb, rmax=rmax_cb)
        return val if T == 1 else T * val

    return _jax_chebyshev
