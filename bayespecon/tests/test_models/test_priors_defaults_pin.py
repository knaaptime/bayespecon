"""Ground-truth pin for every model family's default prior hyperparameters.

Phase 5a prerequisite: before any restructuring of ``models/priors.py`` this
snapshot locks the resolved default hyperparameters (``priors_as_dict`` of the
zero-argument dataclass) for *every* exported priors class.  Any refactor must
reproduce this table exactly, so a changed default fails loudly here.

Note the deliberate entanglement this captures — it is why the priors do not
factor into three orthogonal axes:

* ``rho``/``lam`` bounds differ by likelihood and static/dynamic:
  SAR ``[-1, 1]``, SARProbit ``[-0.95, 0.95]``, SARLogit ``[-0.999, 0.999]``,
  PanelSARDynamic ``[-0.95, 0.95]``.
* ``beta`` defaults are data-scaled (omitted) for Gaussian/NB/Tobit but fixed
  ``(0.0, 10.0)`` for Probit/Logit.
* the noise block differs: Gaussian/NB/Tobit carry ``sigma2_alpha`` +
  ``sigma_sigma``; SARProbit carries ``sigma_a_sigma``; Logit carries neither.
"""

from __future__ import annotations

import pytest

from bayespecon.models import priors as P
from bayespecon.models.priors import priors_as_dict

NU = 1.0 / 30.0

EXPECTED: dict[str, dict[str, float]] = {
    "BasePriors": {"nu_lam": NU, "sigma2_alpha": 2.0, "sigma_sigma": 10.0},
    "OLSPriors": {"nu_lam": NU, "sigma2_alpha": 2.0, "sigma_sigma": 10.0},
    "SLXPriors": {"nu_lam": NU, "sigma2_alpha": 2.0, "sigma_sigma": 10.0},
    "SARPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SEMPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "SDMPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SDEMPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "NegBinPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "alpha_sigma": 2.5,
        "alpha_nu": 3.0,
    },
    "SARNegBinPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "alpha_sigma": 2.5,
        "alpha_nu": 3.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SARTobitPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
        "censor_sigma": 10.0,
    },
    "SEMTobitPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
        "censor_sigma": 10.0,
    },
    "SDMTobitPriors": {
        "nu_lam": NU,
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
        "censor_sigma": 10.0,
    },
    "SARProbitPriors": {
        "beta_mu": 0.0,
        "beta_sigma": 10.0,
        "rho_lower": -0.95,
        "rho_upper": 0.95,
        "sigma_a_sigma": 2.0,
    },
    "SARLogitPriors": {
        "beta_mu": 0.0,
        "beta_sigma": 10.0,
        "rho_lower": -0.999,
        "rho_upper": 0.999,
    },
    "SEMLogitPriors": {
        "beta_mu": 0.0,
        "beta_sigma": 10.0,
        "lam_lower": -0.999,
        "lam_upper": 0.999,
    },
    "PanelBasePriors": {"sigma2_alpha": 2.0, "sigma_sigma": 10.0},
    "PanelOLSPriors": {"sigma2_alpha": 2.0, "sigma_sigma": 10.0},
    "PanelSLXPriors": {"sigma2_alpha": 2.0, "sigma_sigma": 10.0},
    "PanelSARPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "PanelSEMPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelSDMPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "PanelSDEMPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelOLSREPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "sigma_alpha_sigma": 10.0,
    },
    "PanelSARREPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "sigma_alpha_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "PanelSEMREPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "sigma_alpha_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelSDEMREPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "sigma_alpha_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelSARTobitPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
        "censor_sigma": 10.0,
    },
    "PanelSEMTobitPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
        "censor_sigma": 10.0,
    },
    "PanelDynamicBasePriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
    },
    "PanelOLSDynamicPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
    },
    "PanelSLXDynamicPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
    },
    "PanelSARDynamicPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "rho_lower": -0.95,
        "rho_upper": 0.95,
    },
    "PanelSEMDynamicPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "lam_lower": -0.95,
        "lam_upper": 0.95,
    },
    "PanelSDMRDynamicPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "rho_lower": -0.95,
        "rho_upper": 0.95,
    },
    "PanelSDMUDynamicPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "rho_lower": -0.95,
        "rho_upper": 0.95,
        "theta_lower": -0.95,
        "theta_upper": 0.95,
    },
    "PanelSDEMDynamicPriors": {
        "sigma2_alpha": 2.0,
        "sigma_sigma": 10.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "lam_lower": -0.95,
        "lam_upper": 0.95,
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_default_prior_hyperparams_pinned(name):
    cls = getattr(P, name)
    assert priors_as_dict(cls()) == EXPECTED[name]


# Internal Gibbs-sampler prior structs (resolved-hyperparameter containers the
# numpy/JAX kernels consume) also live in ``priors.py``, but they are not
# user-facing validated priors and are exercised by the sampler recovery tests
# rather than this default-hyperparameter snapshot.
_GIBBS_PRIOR_STRUCTS = {
    "GibbsBasePriors",
    "GaussianGibbsPriors",
    "GibbsPriors",
    "ReducedGibbsPriors",
    "FlowReducedGibbsPriors",
    "ZINBGibbsPriors",
    "LogitGibbsPriors",
    "SEMLogitGibbsPriors",
    "REGibbsPriors",
    "PanelGaussianPriors",
}


def test_snapshot_covers_every_user_facing_priors_class():
    """Guard: every user-facing ``*Priors`` dataclass in ``__all__`` is pinned above."""
    exported = {
        n for n in P.__all__ if n.endswith("Priors") and n != "PriorsLike"
    } - _GIBBS_PRIOR_STRUCTS
    assert exported == set(EXPECTED), (
        "Priors classes not covered by the pin: "
        f"{sorted(exported - set(EXPECTED))}; stale entries: "
        f"{sorted(set(EXPECTED) - exported)}"
    )
