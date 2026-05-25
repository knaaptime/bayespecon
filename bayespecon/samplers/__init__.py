"""Custom Gibbs samplers for spatial econometric models.

Subpackages
-----------
gaussian
    Gaussian (SAR/SEM/SDM/SDEM) Gibbs samplers.
panel
    Panel (FE/RE) Gibbs samplers.
negbin
    Negative Binomial (Pólya-Gamma) Gibbs samplers.
_utils
    Internal utilities (not public API).

Re-exports
----------
The most commonly-used symbols are re-exported here for backward
compatibility.  New code should import from subpackages directly::

    from bayespecon.samplers.gaussian import GaussianSARGibbs
    from bayespecon.samplers.panel import REGibbsEstimation
    from bayespecon.samplers.negbin import run_chain
"""

from .gaussian import (
    GaussianGibbsPriors,
    GaussianSARGibbs,
    GaussianSEMGibbs,
    GibbsEstimation,
)
from .negbin import GibbsCache, GibbsPriors, GibbsState, run_chain
from .panel import (
    GaussianSARREGibbs,
    GaussianSEMREGibbs,
    REGibbsEstimation,
    REGibbsPriors,
)

__all__ = [
    # Gaussian
    "GibbsEstimation",
    "GaussianSARGibbs",
    "GaussianSEMGibbs",
    "GaussianGibbsPriors",
    # Panel
    "REGibbsEstimation",
    "GaussianSARREGibbs",
    "GaussianSEMREGibbs",
    "REGibbsPriors",
    # NegBin
    "GibbsState",
    "GibbsPriors",
    "GibbsCache",
    "run_chain",
]
