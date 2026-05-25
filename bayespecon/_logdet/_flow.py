"""Flow-model log-determinant functions.

Contains the flow SAR log-determinant (PyTensor and NumPy), trace
computation helpers, and separable-flow logdet factories.  These are
only used by ``models/flow.py`` and ``models/flow_panel.py``.
"""

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ._config import (
    TraceEstimatorName,
    _auto_logdet_method,
    _default_trace_k,
    _resolve_trace_estimator,
)
from ._grids import _barry_pace_traces, chebyshev
from ._pytensor import logdet_chebyshev, logdet_eigenvalue


def compute_flow_traces(
    W_sparse,
    miter: int = 30,
    riter: int = 50,
    random_state: int | None = None,
) -> np.ndarray:
    """Estimate tr(W^k) for k=1..miter via Barry-Pace stochastic traces.

    Thin public wrapper around :func:`_barry_pace_traces`, mirroring
    ``ftrace1.m`` from the LeSage spatial flows toolbox.  Used by
    :func:`_flow_logdet_poly_coeffs` to pre-compute trace products for the
    flow log-determinant.

    Parameters
    ----------
    W_sparse : array-like or scipy.sparse matrix
        Row-standardised n×n spatial weights matrix.
    miter : int, default=30
        Number of trace orders to estimate (``traces[k-1] ≈ tr(W^k)`` for
        k=1..miter).  Higher values improve the polynomial approximation;
        30–50 is usually sufficient with ``titer=800`` for the geometric tail.
    riter : int, default=50
        Number of Monte Carlo probe vectors for trace estimation.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (miter,)
        Trace estimates: ``traces[k-1] ≈ tr(W^k)`` for k=1..miter.
    """
    if sp.issparse(W_sparse):
        W_sp = W_sparse.tocsr().astype(np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W_sparse, dtype=np.float64))
    rng = np.random.default_rng(random_state)
    raw = _barry_pace_traces(W_sp, miter, riter, rng)  # (miter, riter)
    return raw.mean(axis=1)  # (miter,)


def _flow_logdet_poly_coeffs(
    traces: np.ndarray,
    n: int,
    miter: int,
) -> tuple:
    """Pre-compute polynomial coefficients for the flow log-determinant.

    Ports the multinomial trace identity from ``fodet1.m`` (LeSage 2005).
    For the flow SAR weight matrix
    :math:`W_F = \\rho_d W_d + \\rho_o W_o + \\rho_w W_w` the log-determinant
    expands as:

    .. math::

        \\log|I_N - W_F| = -\\sum_{k=1}^{\\infty}
            \\frac{1}{k} \\text{tr}(W_F^k)

    where by Kronecker properties:

    .. math::

        \\text{tr}(W_F^k) = \\sum_{a+b+c=k} \\binom{k}{a,b,c}
            \\rho_d^a \\rho_o^b \\rho_w^c \\cdot
            \\text{tr}(W^{a+c})\\,\\text{tr}(W^{b+c})

    This function enumerates all (a, b, c) triples for k=1..miter and
    returns flat numpy arrays ready for vectorised evaluation inside
    :func:`flow_logdet_pytensor`.

    Parameters
    ----------
    traces : np.ndarray, shape (miter,)
        Trace estimates from :func:`compute_flow_traces`:
        ``traces[k-1] ≈ tr(W^k)`` for k=1..miter.
    n : int
        Spatial unit count (not flow count N=n²).  Used for ``tr(I_n) = n``
        (the k=0 trace).
    miter : int
        Highest polynomial order for the exact series.  Must equal
        ``len(traces)``.

    Returns
    -------
    tuple of 8 np.ndarray
        ``(poly_a, poly_b, poly_c, poly_coeffs,
           miter_a, miter_b, miter_c, miter_coeffs)``

        ``poly_*`` arrays cover all triples with :math:`a+b+c \\in [1, miter]`.
        ``poly_coeffs[i] = -C(k;a,b,c) \\cdot tw[a+c] \\cdot tw[b+c] / k``
        where tw[0]=n and tw[p]=tr(W^p) for p≥1.

        ``miter_*`` arrays cover only triples with :math:`a+b+c = miter`.
        ``miter_coeffs[i] = C(miter;a,b,c) \\cdot tw[a+c] \\cdot tw[b+c]``
        (positive, without the 1/k division — used for the geometric tail
        inside :func:`flow_logdet_pytensor`).
    """
    from math import factorial

    if len(traces) != miter:
        raise ValueError(f"len(traces)={len(traces)} must equal miter={miter}.")

    # tw[0] = n = tr(I_n),  tw[k] = tr(W^k) for k=1..miter
    tw = np.empty(miter + 1, dtype=np.float64)
    tw[0] = float(n)
    tw[1:] = traces

    poly_rows: list[tuple] = []
    miter_rows: list[tuple] = []

    for k in range(1, miter + 1):
        for a in range(k + 1):
            for b in range(k - a + 1):
                c = k - a - b
                multi = factorial(k) // (factorial(a) * factorial(b) * factorial(c))
                # Trace product: tr(W^{a+c}) * tr(W^{b+c}); indices a+c, b+c ∈ [0, k]
                trace_prod = float(tw[a + c] * tw[b + c])
                coeff = -float(multi) * trace_prod / k
                poly_rows.append((float(a), float(b), float(c), coeff))
                if k == miter:
                    miter_rows.append(
                        (float(a), float(b), float(c), float(multi) * trace_prod)
                    )

    poly_arr = np.array(poly_rows, dtype=np.float64)
    miter_arr = np.array(miter_rows, dtype=np.float64)

    return (
        poly_arr[:, 0],  # poly_a
        poly_arr[:, 1],  # poly_b
        poly_arr[:, 2],  # poly_c
        poly_arr[:, 3],  # poly_coeffs
        miter_arr[:, 0],  # miter_a
        miter_arr[:, 1],  # miter_b
        miter_arr[:, 2],  # miter_c
        miter_arr[:, 3],  # miter_coeffs
    )


def flow_logdet_pytensor(
    rho_d,
    rho_o,
    rho_w,
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    poly_c: np.ndarray,
    poly_coeffs: np.ndarray,
    miter_a: np.ndarray,
    miter_b: np.ndarray,
    miter_c: np.ndarray,
    miter_coeffs: np.ndarray,
    miter: int,
    titer: int = 800,
) -> "pt.TensorVariable":
    """Differentiable PyTensor log-determinant for the flow SAR model.

    Evaluates

    .. math::

        \\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|

    as a fully differentiable PyTensor expression suitable for use as
    ``pm.Potential("jacobian", flow_logdet_pytensor(...))``.

    The computation has two parts:

    1. **Polynomial part** (orders 1 to *miter*): vectorised sum over
       precomputed ``(a, b, c, coeff)`` triples — no Python loop at
       evaluation time.

    2. **Geometric tail** (orders *miter+1* to *titer*): closed-form sum
       using the upper-bound approximation
       :math:`\\text{tr}(W_F^k) \\approx s^{k-m} \\cdot \\text{tr}(W_F^m)`
       where :math:`s = \\rho_d + \\rho_o + \\rho_w` is the spectral-radius
       bound for row-stochastic W, following ``fodet1.m`` lines 60–70.

    Parameters
    ----------
    rho_d, rho_o, rho_w :
        PyTensor scalar variables for the three spatial parameters.
    poly_a, poly_b, poly_c, poly_coeffs :
        Precomputed exponent arrays and coefficients for the polynomial part,
        from :func:`_flow_logdet_poly_coeffs`.
    miter_a, miter_b, miter_c, miter_coeffs :
        Exponents and trace-product weights for the highest-order polynomial
        terms (k = miter), used to compute ``tr(W_F^miter)`` symbolically
        for the geometric tail.
    miter : int
        Highest polynomial order included in the exact series.
    titer : int, default=800
        Highest order included in the geometric tail approximation.

    Returns
    -------
    pytensor.tensor.TensorVariable
        Scalar log-determinant expression.
    """
    # --- Polynomial part: k = 1 .. miter ---
    pa = pt.as_tensor_variable(poly_a)
    pb = pt.as_tensor_variable(poly_b)
    pc = pt.as_tensor_variable(poly_c)
    pcoeffs = pt.as_tensor_variable(poly_coeffs)

    poly_part = pt.sum(
        pcoeffs * pt.power(rho_d, pa) * pt.power(rho_o, pb) * pt.power(rho_w, pc)
    )

    # --- Geometric tail: k = miter+1 .. titer ---
    # tr(W_F^miter) as a PyTensor expression
    ma = pt.as_tensor_variable(miter_a)
    mb = pt.as_tensor_variable(miter_b)
    mc_ = pt.as_tensor_variable(miter_c)
    mcoeffs = pt.as_tensor_variable(miter_coeffs)

    trace_last = pt.sum(
        mcoeffs * pt.power(rho_d, ma) * pt.power(rho_o, mb) * pt.power(rho_w, mc_)
    )

    # scalarparm = rho_d + rho_o + rho_w  (spectral radius bound for row-stochastic W)
    # The geometric tail series ``sum_j s^j / (miter+j)`` only converges
    # for ``|s| < 1``. We clip ``s`` strictly inside the unit interval to
    # avoid ``s^titer`` overflowing for j up to ~``titer-miter``; the
    # bias introduced very close to the boundary is preferable to NaN/Inf
    # gradients during NUTS adaptation.
    s_raw = rho_d + rho_o + rho_w
    scalarparm = pt.clip(s_raw, -1.0 + 1e-6, 1.0 - 1e-6)

    # tail_sum = sum_{j=1}^{titer-miter} scalarparm^j / (miter + j)
    j_arr = np.arange(1, titer - miter + 1, dtype=np.float64)
    recip_arr = pt.as_tensor_variable((1.0 / (miter + j_arr)).astype(np.float64))
    tail_sum = pt.dot(pt.power(scalarparm, j_arr), recip_arr)

    tail_part = -trace_last * tail_sum

    return poly_part + tail_part


def flow_logdet_numpy(
    rho_d,
    rho_o,
    rho_w,
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    poly_c: np.ndarray,
    poly_coeffs: np.ndarray,
    miter_a: np.ndarray,
    miter_b: np.ndarray,
    miter_c: np.ndarray,
    miter_coeffs: np.ndarray,
    miter: int,
    titer: int = 800,
) -> np.ndarray:
    """Vectorised numpy port of :func:`flow_logdet_pytensor`.

    Evaluates :math:`\\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|`
    for arrays of posterior draws ``(rho_d, rho_o, rho_w)``.  Used by
    :class:`~bayespecon.models.flow.FlowModel`-derived models to attach a
    complete pointwise log-likelihood to the InferenceData after sampling
    (the ``pm.Potential("jacobian", ...)`` Jacobian term is not captured by
    PyMC's automatic log-likelihood accounting).

    Parameters
    ----------
    rho_d, rho_o, rho_w : array-like, shape (G,) or scalar
        Posterior draws for the three spatial parameters.
    poly_a, poly_b, poly_c, poly_coeffs, miter_a, miter_b, miter_c, miter_coeffs :
        Precomputed exponent arrays and coefficients from
        :func:`_flow_logdet_poly_coeffs`.
    miter : int
        Highest polynomial order included in the exact series.
    titer : int, default 800
        Highest order included in the geometric tail approximation.

    Returns
    -------
    np.ndarray, shape (G,)
        Log-determinant evaluated at each draw.
    """
    rho_d = np.atleast_1d(np.asarray(rho_d, dtype=np.float64))
    rho_o = np.atleast_1d(np.asarray(rho_o, dtype=np.float64))
    rho_w = np.atleast_1d(np.asarray(rho_w, dtype=np.float64))

    # Polynomial part: vectorised over draws with in-place accumulation
    # to avoid materialising the full (G, P) 4-way broadcast temporary.
    rd_pow = np.power(rho_d[:, None], poly_a[None, :])  # (G, P)
    ro_pow = np.power(rho_o[:, None], poly_b[None, :])  # (G, P)
    rw_pow = np.power(rho_w[:, None], poly_c[None, :])  # (G, P)
    rd_pow *= ro_pow  # in-place: rd_pow now holds rd^a * ro^b
    rd_pow *= rw_pow  # in-place: rd_pow now holds rd^a * ro^b * rw^c
    rd_pow *= poly_coeffs[None, :]  # in-place: multiply by coefficients
    poly_part = rd_pow.sum(axis=1)

    # tr(W_F^miter) symbolically per draw — same in-place pattern.
    ml_rd = np.power(rho_d[:, None], miter_a[None, :])
    ml_ro = np.power(rho_o[:, None], miter_b[None, :])
    ml_rw = np.power(rho_w[:, None], miter_c[None, :])
    ml_rd *= ml_ro
    ml_rd *= ml_rw
    ml_rd *= miter_coeffs[None, :]
    trace_last = ml_rd.sum(axis=1)

    # Geometric tail (clip to avoid overflow at the boundary).
    s_raw = rho_d + rho_o + rho_w
    scalarparm = np.clip(s_raw, -1.0 + 1e-6, 1.0 - 1e-6)

    j_arr = np.arange(1, titer - miter + 1, dtype=np.float64)
    recip = 1.0 / (miter + j_arr)
    tail_sum = (np.power(scalarparm[:, None], j_arr[None, :]) * recip[None, :]).sum(
        axis=1
    )

    return poly_part - trace_last * tail_sum


def make_flow_separable_logdet(
    W_sparse,
    n: int,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    cheb_order: int = 20,
    trace_estimator: TraceEstimatorName = "hutchpp",
    trace_k: int | None = None,
):
    r"""Pre-compute logdet data for separable flow models and return a logdet callable.

    For the separable constraint :math:`\rho_w = -\rho_d \rho_o` the full
    system log-determinant factors exactly as

    .. math::

        \log|L_o \otimes L_d|
        = n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    This function pre-computes the required data once at model initialisation
    and returns a closure that evaluates the expression as a symbolic
    :mod:`pytensor` scalar, suitable for ``pm.Potential``.

    Parameters
    ----------
    W_sparse : array-like or scipy.sparse matrix
        Row-standardised :math:`n \times n` spatial weights matrix.
    n : int
        Number of spatial units.
    method : str, default ``"eigenvalue"``
        ``"eigenvalue"`` — exact O(n) per-step evaluation after O(n³)
        eigendecomposition.  Exact for any rho.
        ``"chebyshev"`` — near-minimax Chebyshev polynomial; O(m) per step
        after O(n³) or O(R·n·m) precomputation via :func:`chebyshev`.
        Coefficients use exact eigenvalues when ``n`` is small, otherwise
        a stochastic trace estimator selected by ``trace_estimator``.
    rho_min : float, default -1.0
        Lower bound of the rho domain (``"chebyshev"`` only).
    rho_max : float, default 1.0
        Upper bound of the rho domain (``"chebyshev"`` only).
    cheb_order : int, default 20
        Chebyshev polynomial order (``"chebyshev"`` only).
    trace_estimator : {"hutchinson", "hutchpp", "xtrace"}, default "hutchpp"
        Stochastic trace estimator used to build Chebyshev coefficients
        when an eigendecomposition is unavailable.
    trace_k : int, optional
        Number of probe vectors for the trace estimator.

    Returns
    -------
    callable
        Function ``fn(rho_d, rho_o) -> pt.TensorVariable`` evaluating
        :math:`n\,f(\rho_d) + n\,f(\rho_o)` where
        :math:`f(\rho) = \log|I_n - \rho W|`.
    """
    trace_estimator = _resolve_trace_estimator(trace_estimator)
    _k = trace_k if trace_k is not None else _default_trace_k(trace_estimator)

    if sp.issparse(W_sparse):
        W_dense = np.asarray(W_sparse.toarray(), dtype=np.float64)
    else:
        W_dense = np.asarray(W_sparse, dtype=np.float64)

    if method is None:
        method = _auto_logdet_method(n)

    if method == "eigenvalue":
        eigs = np.linalg.eigvals(W_dense).real.astype(np.float64)
        return lambda rho_d, rho_o: (
            n * logdet_eigenvalue(rho_d, eigs) + n * logdet_eigenvalue(rho_o, eigs)
        )
    elif method == "chebyshev":
        out = chebyshev(
            W_dense,
            order=cheb_order,
            rmin=rho_min,
            rmax=rho_max,
            estimator=trace_estimator,
            n_mc_iter=_k,
        )
        coeffs = out["coeffs"]
        rmin_cb = out["rmin"]
        rmax_cb = out["rmax"]
        return lambda rho_d, rho_o: (
            n * logdet_chebyshev(rho_d, coeffs, rmin=rmin_cb, rmax=rmax_cb)
            + n * logdet_chebyshev(rho_o, coeffs, rmin=rmin_cb, rmax=rmax_cb)
        )
    else:
        raise ValueError(
            f"make_flow_separable_logdet: method={method!r} not recognised. "
            "Choose one of: 'eigenvalue', 'chebyshev'."
        )


def make_flow_separable_logdet_numpy(
    W_sparse,
    n: int,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    cheb_order: int = 20,
    trace_estimator: TraceEstimatorName = "hutchpp",
    trace_k: int | None = None,
):
    r"""Pre-compute numeric logdet data for separable flow models.

    Returns a vectorized numpy closure for post-fit Jacobian reconstruction:

    .. math::

        n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    Parameters are aligned with :func:`make_flow_separable_logdet` for API
    symmetry.
    """
    from ._numpy import make_logdet_numpy_vec_fn

    trace_estimator = _resolve_trace_estimator(trace_estimator)

    if sp.issparse(W_sparse):
        W_sp = W_sparse.tocsr().astype(np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W_sparse, dtype=np.float64))

    if method is None:
        method = _auto_logdet_method(n)
    if method not in {"eigenvalue", "chebyshev"}:
        raise ValueError(
            f"make_flow_separable_logdet_numpy: method={method!r} not recognised. "
            "Choose one of: 'eigenvalue', 'chebyshev'."
        )

    eigs = None
    if method == "eigenvalue":
        eigs = np.linalg.eigvals(np.asarray(W_sp.toarray(), dtype=np.float64)).real

    logdet_vec = make_logdet_numpy_vec_fn(
        W_sp,
        eigs,
        method=method,
        rho_min=rho_min,
        rho_max=rho_max,
        trace_estimator=trace_estimator,
        trace_k=trace_k,
    )

    def _eval(rho_d, rho_o) -> np.ndarray:
        rd = np.asarray(rho_d, dtype=np.float64).reshape(-1)
        ro = np.asarray(rho_o, dtype=np.float64).reshape(-1)
        if rd.shape != ro.shape:
            raise ValueError(
                "rho_d and rho_o must have the same shape for separable logdet evaluation."
            )
        return n * (logdet_vec(rd) + logdet_vec(ro))

    return _eval
