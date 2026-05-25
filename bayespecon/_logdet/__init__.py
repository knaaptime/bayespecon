"""Log-determinant utilities for spatial models.

Provides log|I - rho*W| as a pytensor expression or a pre-computed
grid interpolation for large n, used as pm.Potential in spatial likelihoods.

This subpackage is private (``_logdet``) — it is only used internally by
model classes and should not be imported directly by users.
"""

# Configuration, enums, type aliases, resolution
from ._config import (
    LogDetMethod,
    LogdetBounds,
    LogDetMethodName,
    TraceEstimatorName,
    VALID_LOGDET_METHODS,
    VALID_TRACE_ESTIMATORS,
    _LOGDET_FN_CACHE,
    _LOGDET_FN_CACHE_MAXSIZE,
    _LOGDET_GRID_EIG_MAX,
    _auto_logdet_method,
    _default_trace_k,
    _resolve_trace_estimator,
    resolve_logdet_bounds,
    resolve_logdet_method,
)

# Grid builders and Chebyshev coefficient computation
from ._grids import (
    _barry_pace_traces,
    _build_logdet_grid,
    _stable_rho_grid,
    chebyshev,
    ilu,
    mc,
    sparse_grid,
    spline,
)

# PyTensor symbolic evaluation
from ._pytensor import (
    _make_pytensor_interp_fn,
    logdet_chebyshev,
    logdet_eigenvalue,
    logdet_exact,
    logdet_interpolated,
    logdet_mc_poly_pytensor,
)

# JAX-native evaluation
from ._jax import (
    jax_logdet_chebyshev,
    jax_logdet_trace_poly,
    make_logdet_jax_fn,
)

# NumPy factory functions and caching
from ._numpy import (
    _GRID_SPLINE_METHODS,
    _build_grid_spline,
    _hash_array,
    _logdet_w_signature,
    clear_logdet_fn_cache,
    get_cached_logdet_fn,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)

# Flow-model logdet
from ._flow import (
    _flow_logdet_poly_coeffs,
    compute_flow_traces,
    flow_logdet_numpy,
    flow_logdet_pytensor,
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)

# Trace estimation (moved from _trace_estimation.py)
from ._trace import (
    traceax_available,
    traceax_traces,
    traceax_traces_for_chebyshev,
)

__all__ = [
    # Config
    "LogDetMethod",
    "LogdetBounds",
    "LogDetMethodName",
    "TraceEstimatorName",
    "VALID_LOGDET_METHODS",
    "VALID_TRACE_ESTIMATORS",
    "resolve_logdet_method",
    "resolve_logdet_bounds",
    # Grid builders
    "chebyshev",
    "sparse_grid",
    "spline",
    "mc",
    "ilu",
    # PyTensor
    "logdet_eigenvalue",
    "logdet_exact",
    "logdet_chebyshev",
    "logdet_mc_poly_pytensor",
    "logdet_interpolated",
    # JAX
    "jax_logdet_chebyshev",
    "jax_logdet_trace_poly",
    "make_logdet_jax_fn",
    # NumPy factories
    "make_logdet_fn",
    "make_logdet_numpy_fn",
    "make_logdet_numpy_vec_fn",
    "get_cached_logdet_fn",
    "clear_logdet_fn_cache",
    # Flow
    "flow_logdet_pytensor",
    "flow_logdet_numpy",
    "compute_flow_traces",
    "make_flow_separable_logdet",
    "make_flow_separable_logdet_numpy",
    # Trace estimation
    "traceax_available",
    "traceax_traces",
    "traceax_traces_for_chebyshev",
]