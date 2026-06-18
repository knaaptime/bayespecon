"""Base dataclasses shared across Gibbs sampler families.

All spatial Gibbs samplers (Gaussian, NegBin, Logit, RE) share common
state fields (``beta``, ``rho``) and prior fields (``beta_mu``,
``beta_sigma``, ``rho_lower``, ``rho_upper``).  These base classes
factor out that shared structure to reduce duplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GibbsBaseState:
    """Base state for all Gibbs samplers.

    Subclasses add model-specific fields (e.g. ``sigma2``, ``eta``,
    ``omega``, ``alpha``).
    """

    beta: np.ndarray
    rho: float


@dataclass
class GibbsBasePriors:
    """Base priors for all Gibbs samplers.

    Subclasses add model-specific prior fields (e.g. ``sigma2_alpha``,
    ``alpha_sigma``).
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    rho_lower: float = -0.999
    rho_upper: float = 0.999
