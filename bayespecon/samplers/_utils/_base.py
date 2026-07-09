"""Base dataclasses shared across Gibbs sampler families.

All spatial Gibbs samplers (Gaussian, NegBin, Logit, RE) share common
state fields (``beta``, ``rho``) and prior fields (``beta_mu``,
``beta_sigma``, ``rho_lower``, ``rho_upper``).  These base classes
factor out that shared structure to reduce duplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Prior structs live in models.priors (single home for all prior containers).
from ...models.priors import GibbsBasePriors  # noqa: F401


@dataclass
class GibbsBaseState:
    """Base state for all Gibbs samplers.

    Subclasses add model-specific fields (e.g. ``sigma2``, ``eta``,
    ``omega``, ``alpha``).
    """

    beta: np.ndarray
    rho: float
