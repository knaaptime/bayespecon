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

    from bayespecon._samplers.gaussian import GaussianSARGibbs
    from bayespecon._samplers.panel import REGibbsEstimation
    from bayespecon._samplers.negbin import run_chain
"""

from .gaussian import GibbsEstimation, GaussianSARGibbs, GaussianSEMGibbs, GaussianGibbsPriors
from .panel import REGibbsEstimation, GaussianSARREGibbs, GaussianSEMREGibbs, REGibbsPriors
from .negbin import GibbsState, GibbsPriors, GibbsCache, run_chain

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
