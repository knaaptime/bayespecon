"""Log-determinant computation for spatial econometric models.

Five methods:

* ``"eigenvalue"`` — exact, O(n) per call after O(n³) setup.
* ``"slq"`` — Stochastic Lanczos Quadrature; 300 matvecs, ρ-independent.
* ``"chebyshev"`` — Barry-Pace MC traces → Chebyshev; O(m) per call.
* ``"cheb_stochastic"`` — stochastic Chebyshev expansion (Han et al. 2015);
  geometric convergence, same matvec cost as ``chebyshev``.
* ``"traces"`` — multinomial trace expansion for unrestricted 3-parameter
  flow models (the only option when the system matrix doesn't factor).

Auto-selection: ``"eigenvalue"`` for n ≤ 500, ``"cheb_stochastic"`` otherwise.
"""

from ._cheb_stochastic import (
    ChebStochasticPrecompute,
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_eval_vec,
    cheb_stochastic_logdet_precompute,
)
from ._chebyshev import chebyshev, clear_chebyshev_cache
from ._config import (
    VALID_LOGDET_METHODS,
    LogdetBounds,
    LogDetMethod,
    LogDetMethodName,
    resolve_logdet_bounds,
    resolve_logdet_method,
)
from ._factories import (
    clear_logdet_fn_cache,
    get_cached_logdet_fn,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)
from ._flow import (
    compute_flow_traces,
    flow_logdet_numpy,
    flow_logdet_pytensor,
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)
from ._jax import jax_logdet_chebyshev, jax_slq_logdet_precompute, make_logdet_jax_fn
from ._pytensor import logdet_chebyshev, logdet_eigenvalue
from ._slq import (
    SLQPrecompute,
    slq_logdet_eval,
    slq_logdet_eval_vec,
    slq_logdet_precompute,
    slq_to_chebyshev_coeffs,
)

__all__ = [
    # Config
    "LogDetMethod",
    "LogDetMethodName",
    "LogdetBounds",
    "VALID_LOGDET_METHODS",
    "resolve_logdet_method",
    "resolve_logdet_bounds",
    # PyTensor primitives
    "logdet_eigenvalue",
    "logdet_chebyshev",
    # JAX
    "jax_logdet_chebyshev",
    "make_logdet_jax_fn",
    # Factories
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
    # Chebyshev
    "chebyshev",
    "clear_chebyshev_cache",
    # SLQ
    "SLQPrecompute",
    "slq_logdet_precompute",
    "slq_logdet_eval",
    "slq_logdet_eval_vec",
    "slq_to_chebyshev_coeffs",
    # Stochastic Chebyshev
    "ChebStochasticPrecompute",
    "cheb_stochastic_logdet_precompute",
    "cheb_stochastic_logdet_eval",
    "cheb_stochastic_logdet_eval_vec",
]
