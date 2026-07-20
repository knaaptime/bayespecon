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

from ._aaa import (
    AAAPrecompute,
    aaa_logdet_eval,
    aaa_logdet_eval_vec,
    aaa_logdet_precompute,
)
from ._cheb_stochastic import (
    ChebStochasticPrecompute,
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_eval_vec,
    cheb_stochastic_logdet_precompute,
)
from ._chebyshev import chebyshev
from ._chol_cheb import (
    CholChebPrecompute,
    chol_cheb_logdet_eval,
    chol_cheb_logdet_eval_vec,
    chol_cheb_logdet_precompute,
)
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
    make_logdet_grad_numpy_fn,
    make_logdet_grad_numpy_vec_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)
from ._flow import (
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)

# Flow-resolvent exports are resolved lazily (see __getattr__ below):
# _flow_resolvent pulls in the quadrature/JAX machinery, which the
# cross-sectional import path never needs.
_FLOW_RESOLVENT_EXPORTS = frozenset(
    {
        "FlowKron",
        "FlowKronJax",
        "flow_logdet_grad",
        "flow_logdet_grad_exact",
        "flow_logdet_grad_jax",
        "flow_logdet_value",
        "flow_logdet_value_and_grad",
        "flow_logdet_value_and_grad_jax",
    }
)


def __getattr__(name):
    if name in _FLOW_RESOLVENT_EXPORTS:
        from . import _flow_resolvent

        return getattr(_flow_resolvent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from ._jax import jax_logdet_chebyshev, make_logdet_jax_fn
from ._pytensor import logdet_chebyshev, logdet_eigenvalue
from ._resolvent import (
    logdet_grad_aaa,
    logdet_grad_chebyshev,
    logdet_grad_eigenvalue,
    logdet_grad_slq,
)
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
    "make_logdet_grad_numpy_fn",
    "make_logdet_grad_numpy_vec_fn",
    "get_cached_logdet_fn",
    "clear_logdet_fn_cache",
    # Resolvent trace / logdet gradient core
    "logdet_grad_eigenvalue",
    "logdet_grad_chebyshev",
    "logdet_grad_aaa",
    "logdet_grad_slq",
    # Flow (separable factorisation)
    "make_flow_separable_logdet",
    "make_flow_separable_logdet_numpy",
    # Flow resolvent-Kronecker gradient (scalable, eigenvalue-free)
    "FlowKron",
    "flow_logdet_grad",
    "flow_logdet_grad_exact",
    "flow_logdet_value",
    "flow_logdet_value_and_grad",
    # Chebyshev
    "chebyshev",
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
