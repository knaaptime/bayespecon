"""Log-determinant utilities for spatial models.

Provides log|I - rho*W| as a pytensor expression or a pre-computed
grid interpolation for large n, used as pm.Potential in spatial likelihoods.

This subpackage is private (``_logdet``) — it is only used internally by
model classes and should not be imported directly by users.
"""

# Configuration, enums, type aliases, resolution
from ._config import (
    VALID_LOGDET_METHODS,
    LogdetBounds,
    LogDetMethod,
    LogDetMethodName,
    _auto_logdet_method,
    resolve_logdet_bounds,
    resolve_logdet_method,
)

# Flow-model logdet
from ._flow import (
    compute_flow_traces,
    flow_logdet_numpy,
    flow_logdet_pytensor,
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)

# Grid / polynomial primitives (used by edge-case tests and benchmarks)
from ._grids import (
    _stable_rho_grid,
    chebyshev,
    mc,
    spline,
)

# JAX-native evaluation
from ._jax import (
    jax_logdet_chebyshev,
    jax_logdet_trace_poly,
    make_logdet_jax_fn,
)

# NumPy factory functions and caching
from ._numpy import (
    clear_logdet_fn_cache,
    get_cached_logdet_fn,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)

# PyTensor symbolic evaluation
from ._pytensor import (
    logdet_chebyshev,
    logdet_eigenvalue,
    logdet_exact,
    logdet_interpolated,
    logdet_mc_poly_pytensor,
)

__all__ = [
    # Config
    "LogDetMethod",
    "LogdetBounds",
    "LogDetMethodName",
    "VALID_LOGDET_METHODS",
    "resolve_logdet_method",
    "resolve_logdet_bounds",
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
    # Grid / polynomial primitives
    "_stable_rho_grid",
    "chebyshev",
    "mc",
    "spline",
]
