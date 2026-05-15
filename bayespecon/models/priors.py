"""Typed prior containers for :mod:`bayespecon.models`.

Each model family declares its supported priors via a frozen dataclass.  The
dataclass serves two purposes:

* **Validation.** Constructing :class:`SARPriors` with an unknown keyword
  raises :class:`TypeError`, so typos like ``rho_lo=-1`` fail fast at model
  construction instead of being silently ignored deep inside
  ``_build_pymc_model``.
* **Discoverability.** Users get IDE autocomplete + type checking on the
  supported keys and their defaults.

The user-facing ``priors=...`` argument on every model still accepts a plain
``dict``; the model's ``__init__`` calls :func:`resolve_priors` to coerce
``dict | Priors | None`` into the canonical dataclass and rejects unknown
keys with a helpful error.

Example
-------

>>> from bayespecon.models import SAR
>>> from bayespecon.models.priors import SARPriors
>>> SAR(y=y, X=X, W=W, priors=SARPriors(rho_lower=0.0))           # typed form
>>> SAR(y=y, X=X, W=W, priors={"rho_lower": 0.0})                  # dict form
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Type, Union

# ---------------------------------------------------------------------------
# Cross-sectional dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasePriors:
    """Priors shared by every cross-sectional model.

    Attributes
    ----------
    beta_mu, beta_sigma
        Normal prior on regression coefficients.
    sigma_sigma
        Half-normal scale on the observation-noise standard deviation.
    nu_lam
        Rate of the Exponential prior on Student-t degrees of freedom when
        ``robust=True``.  Mean ``1/nu_lam`` (default mean ≈ 30).
    """

    beta_mu: float = 0.0
    beta_sigma: float = 1e6
    sigma_sigma: float = 10.0
    nu_lam: float = 1.0 / 30.0


@dataclass(frozen=True)
class OLSPriors(BasePriors):
    """Priors for :class:`bayespecon.models.OLS`."""


@dataclass(frozen=True)
class SLXPriors(BasePriors):
    """Priors for :class:`bayespecon.models.SLX`."""


@dataclass(frozen=True)
class SARPriors(BasePriors):
    """Priors for :class:`bayespecon.models.SAR`.

    Adds bounds on the spatial autoregressive parameter ``rho``.
    """

    rho_lower: float = -1.0
    rho_upper: float = 1.0


@dataclass(frozen=True)
class SEMPriors(BasePriors):
    """Priors for :class:`bayespecon.models.SEM`.

    Adds bounds on the spatial error parameter ``lam``.
    """

    lam_lower: float = -1.0
    lam_upper: float = 1.0


@dataclass(frozen=True)
class SDMPriors(SARPriors):
    """Priors for :class:`bayespecon.models.SDM` (lag + WX)."""


@dataclass(frozen=True)
class SDEMPriors(SEMPriors):
    """Priors for :class:`bayespecon.models.SDEM` (SLX + spatial error)."""


# ---------------------------------------------------------------------------
# Tobit / probit dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SARTobitPriors(SARPriors):
    """Priors for :class:`bayespecon.models.SARTobit`.

    Adds ``censor_sigma``: scale of the half-normal prior on the censored
    latent-variable gap.
    """

    censor_sigma: float = 10.0


@dataclass(frozen=True)
class SEMTobitPriors(SEMPriors):
    """Priors for :class:`bayespecon.models.SEMTobit`."""

    censor_sigma: float = 10.0


@dataclass(frozen=True)
class SDMTobitPriors(SDMPriors):
    """Priors for :class:`bayespecon.models.SDMTobit`."""

    censor_sigma: float = 10.0


@dataclass(frozen=True)
class SpatialProbitPriors:
    """Priors for :class:`bayespecon.models.SpatialProbit`.

    Differs from :class:`SARPriors`: there is no ``sigma`` parameter (the
    probit link absorbs the noise scale) and the regional random-effect
    variance ``sigma_a`` has its own half-normal scale.
    """

    rho_lower: float = -0.95
    rho_upper: float = 0.95
    beta_mu: float = 0.0
    beta_sigma: float = 10.0
    sigma_a_sigma: float = 2.0


# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------


PriorsLike = Union[Mapping[str, Any], BasePriors, "SpatialProbitPriors", None]


def resolve_priors(
    priors: PriorsLike,
    priors_cls: Type[Any],
) -> Any:
    """Coerce a user-supplied ``priors`` argument to a dataclass instance.

    Parameters
    ----------
    priors
        Either ``None`` (use defaults), a :class:`dict` of overrides, or an
        already-constructed priors dataclass.
    priors_cls
        The dataclass type expected by the model.  Used to validate keys
        and to build the result when ``priors`` is ``None`` or a ``dict``.

    Returns
    -------
    An instance of ``priors_cls``.

    Raises
    ------
    TypeError
        If ``priors`` is a dict containing keys that are not defined on
        ``priors_cls``, or if ``priors`` is a dataclass of an incompatible
        type.
    """
    if priors is None:
        return priors_cls()
    if isinstance(priors, priors_cls):
        return priors
    if isinstance(priors, Mapping):
        allowed = {f.name for f in fields(priors_cls)}
        unknown = set(priors) - allowed
        if unknown:
            raise TypeError(
                f"Unknown prior key(s) for {priors_cls.__name__}: "
                f"{sorted(unknown)}. Allowed keys: {sorted(allowed)}."
            )
        return priors_cls(**dict(priors))
    raise TypeError(
        f"priors must be None, a dict, or a {priors_cls.__name__} instance; "
        f"got {type(priors).__name__}."
    )


def priors_as_dict(priors: Any) -> dict[str, Any]:
    """Return a plain ``dict`` view of a priors dataclass.

    Provided so that existing model code that relies on ``self.priors.get(...)``
    continues to function unchanged while migration to typed access proceeds.
    """
    return asdict(priors)


# ---------------------------------------------------------------------------
# Panel dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelBasePriors:
    """Priors shared by every panel model.

    Panel models do not currently expose Student-t robustification, so
    ``nu_lam`` from :class:`BasePriors` is omitted.

    Attributes
    ----------
    beta_mu, beta_sigma
        Normal prior on regression coefficients.
    sigma_sigma
        Half-normal scale on the observation-noise standard deviation.
    """

    beta_mu: float = 0.0
    beta_sigma: float = 1e6
    sigma_sigma: float = 10.0


@dataclass(frozen=True)
class PanelOLSPriors(PanelBasePriors):
    """Priors for :class:`bayespecon.models.OLSPanelFE`."""


@dataclass(frozen=True)
class PanelSLXPriors(PanelBasePriors):
    """Priors for :class:`bayespecon.models.SLXPanelFE`."""


@dataclass(frozen=True)
class PanelSARPriors(PanelBasePriors):
    """Priors for :class:`bayespecon.models.SARPanelFE` (adds rho bounds)."""

    rho_lower: float = -1.0
    rho_upper: float = 1.0


@dataclass(frozen=True)
class PanelSEMPriors(PanelBasePriors):
    """Priors for :class:`bayespecon.models.SEMPanelFE` (adds lam bounds)."""

    lam_lower: float = -1.0
    lam_upper: float = 1.0


@dataclass(frozen=True)
class PanelSDMPriors(PanelSARPriors):
    """Priors for :class:`bayespecon.models.SDMPanelFE`."""


@dataclass(frozen=True)
class PanelSDEMPriors(PanelSEMPriors):
    """Priors for :class:`bayespecon.models.SDEMPanelFE`."""


# -- Panel random-effects ----------------------------------------------------


@dataclass(frozen=True)
class PanelREMixinPriors:
    """Mixin providing the random-effects scale prior."""

    sigma_alpha_sigma: float = 10.0


@dataclass(frozen=True)
class PanelOLSREPriors(PanelOLSPriors, PanelREMixinPriors):
    """Priors for :class:`bayespecon.models.OLSPanelRE`."""


@dataclass(frozen=True)
class PanelSARREPriors(PanelSARPriors, PanelREMixinPriors):
    """Priors for :class:`bayespecon.models.SARPanelRE`."""


@dataclass(frozen=True)
class PanelSEMREPriors(PanelSEMPriors, PanelREMixinPriors):
    """Priors for :class:`bayespecon.models.SEMPanelRE`."""


@dataclass(frozen=True)
class PanelSDEMREPriors(PanelSDEMPriors, PanelREMixinPriors):
    """Priors for :class:`bayespecon.models.SDEMPanelRE`."""


# -- Panel Tobit -------------------------------------------------------------


@dataclass(frozen=True)
class PanelSARTobitPriors(PanelSARPriors):
    """Priors for :class:`bayespecon.models.SARPanelTobit`."""

    censor_sigma: float = 10.0


@dataclass(frozen=True)
class PanelSEMTobitPriors(PanelSEMPriors):
    """Priors for :class:`bayespecon.models.SEMPanelTobit`."""

    censor_sigma: float = 10.0


# -- Panel dynamic (tighter [-0.95, 0.95] bounds; adds phi) -----------------


@dataclass(frozen=True)
class PanelDynamicBasePriors(PanelBasePriors):
    """Priors shared by every dynamic panel model.

    Adds bounds on the AR(1) coefficient ``phi`` on the lagged dependent
    variable and tightens spatial-parameter bounds to ``[-0.95, 0.95]``.
    """

    phi_lower: float = -0.95
    phi_upper: float = 0.95


@dataclass(frozen=True)
class PanelOLSDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`bayespecon.models.OLSPanelDynamic`."""


@dataclass(frozen=True)
class PanelSLXDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`bayespecon.models.SLXPanelDynamic`."""


@dataclass(frozen=True)
class PanelSARDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`bayespecon.models.SARPanelDynamic`."""

    rho_lower: float = -0.95
    rho_upper: float = 0.95


@dataclass(frozen=True)
class PanelSEMDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`bayespecon.models.SEMPanelDynamic`."""

    lam_lower: float = -0.95
    lam_upper: float = 0.95


@dataclass(frozen=True)
class PanelSDMRDynamicPriors(PanelSARDynamicPriors):
    """Priors for :class:`bayespecon.models.SDMRPanelDynamic`."""


@dataclass(frozen=True)
class PanelSDMUDynamicPriors(PanelSARDynamicPriors):
    """Priors for :class:`bayespecon.models.SDMUPanelDynamic`.

    Adds bounds on the time-space cross-term coefficient ``theta``.
    """

    theta_lower: float = -0.95
    theta_upper: float = 0.95


@dataclass(frozen=True)
class PanelSDEMDynamicPriors(PanelSEMDynamicPriors):
    """Priors for :class:`bayespecon.models.SDEMPanelDynamic`."""


__all__ = [
    "BasePriors",
    "OLSPriors",
    "SLXPriors",
    "SARPriors",
    "SEMPriors",
    "SDMPriors",
    "SDEMPriors",
    "SARTobitPriors",
    "SEMTobitPriors",
    "SDMTobitPriors",
    "SpatialProbitPriors",
    "PanelBasePriors",
    "PanelOLSPriors",
    "PanelSLXPriors",
    "PanelSARPriors",
    "PanelSEMPriors",
    "PanelSDMPriors",
    "PanelSDEMPriors",
    "PanelOLSREPriors",
    "PanelSARREPriors",
    "PanelSEMREPriors",
    "PanelSDEMREPriors",
    "PanelSARTobitPriors",
    "PanelSEMTobitPriors",
    "PanelDynamicBasePriors",
    "PanelOLSDynamicPriors",
    "PanelSLXDynamicPriors",
    "PanelSARDynamicPriors",
    "PanelSEMDynamicPriors",
    "PanelSDMRDynamicPriors",
    "PanelSDMUDynamicPriors",
    "PanelSDEMDynamicPriors",
    "PriorsLike",
    "resolve_priors",
    "priors_as_dict",
]
