"""Registry mapping model classes to their diagnostic test suites.

This module decouples model classes from their diagnostic test assignments.
Instead of each model class storing ``_spatial_diagnostics_tests`` as a
class attribute (which creates circular-import risk between the models
and diagnostics packages), the mapping lives here as a standalone dict.

Model classes look up their suite via :func:`get_diagnostic_suite`, which
falls back to MRO traversal so that subclasses inherit their parent's
suite unless they override it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .suites import DiagnosticSuite

if TYPE_CHECKING:
    from ...models.base import SpatialModel

# ---------------------------------------------------------------------------
# Registry: model class → DiagnosticSuite
# ---------------------------------------------------------------------------
# Populated lazily to avoid circular imports.  The ``_register`` function
# is called at the bottom of this module after all suite definitions are
# available, and the model classes are referenced by *fully-qualified
# dotted path* so that importing this module does not trigger model imports.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, DiagnosticSuite] = {}


def _register(class_path: str, suite: DiagnosticSuite) -> None:
    """Register a model class (by dotted path) to a diagnostic suite."""
    _REGISTRY[class_path] = suite


def get_diagnostic_suite(model: "SpatialModel | type") -> DiagnosticSuite | None:
    """Return the :class:`DiagnosticSuite` for *model*, or ``None``.

    Looks up the model class in the registry.  If *model* is an instance,
    uses ``type(model)``; if it is already a class, uses it directly.
    If the exact class is not found, walks the MRO to find the first
    registered ancestor.  Returns ``None`` when no suite is registered
    for any class in the MRO.
    """
    cls = model if isinstance(model, type) else type(model)
    for klass in cls.__mro__:
        fqn = f"{klass.__module__}.{klass.__qualname__}"
        if fqn in _REGISTRY:
            return _REGISTRY[fqn]
    return None


# ---------------------------------------------------------------------------
# Cross-sectional models
# ---------------------------------------------------------------------------

from .suites import (  # noqa: E402
    FLOW_INTRA_SUITE,
    FLOW_PANEL_SUITE,
    FLOW_SUITE,
    OLS_SUITE,
    SAR_NEGBIN_SUITE,
    SAR_SUITE,
    SAR_TOBIT_SUITE,
    SDEM_SUITE,
    SDM_SUITE,
    SDM_TOBIT_SUITE,
    SEM_SUITE,
    SLX_SUITE,
)

# Cross-sectional
_register("bayespecon.models.ols.OLS", OLS_SUITE)
_register("bayespecon.models.sar.SAR", SAR_SUITE)
_register("bayespecon.models.sem.SEM", SEM_SUITE)
_register("bayespecon.models.slx.SLX", SLX_SUITE)
_register("bayespecon.models.sdm.SDM", SDM_SUITE)
_register("bayespecon.models.sdem.SDEM", SDEM_SUITE)

# Tobit
_register("bayespecon.models.tobit.SARTobit", SAR_TOBIT_SUITE)
_register("bayespecon.models.tobit.SEMTobit", SEM_SUITE)
_register("bayespecon.models.tobit.SDMTobit", SDM_TOBIT_SUITE)

# NegBin
_register("bayespecon.models.sar_negbin.SARNegativeBinomial", SAR_NEGBIN_SUITE)
_register("bayespecon.models.sar_negbin_latent.SARNegBinLatent", SAR_NEGBIN_SUITE)

# Flow (cross-sectional)
_register("bayespecon.models.flow.SARFlow", FLOW_SUITE)
_register("bayespecon.models.flow.SARFlowSeparable", FLOW_SUITE)
_register("bayespecon.models.flow.PoissonSARFlow", FLOW_SUITE)
_register("bayespecon.models.flow.PoissonSARFlowSeparable", FLOW_SUITE)
_register("bayespecon.models.flow.NegativeBinomialSARFlow", FLOW_SUITE)
_register("bayespecon.models.flow.NegativeBinomialSARFlowSeparable", FLOW_SUITE)
_register("bayespecon.models.flow.OLSFlow", FLOW_INTRA_SUITE)
_register("bayespecon.models.flow.PoissonFlow", FLOW_INTRA_SUITE)
_register("bayespecon.models.flow.NegativeBinomialFlow", FLOW_INTRA_SUITE)
_register("bayespecon.models.flow.SEMFlow", FLOW_SUITE)
_register("bayespecon.models.flow.SEMFlowSeparable", FLOW_SUITE)
_register("bayespecon.models.flow.SARNegBinFlowLatent", FLOW_SUITE)
_register("bayespecon.models.flow.SARNegBinFlowSeparableLatent", FLOW_SUITE)

# ---------------------------------------------------------------------------
# Panel models
# ---------------------------------------------------------------------------

from .suites import (  # noqa: E402
    OLS_PANEL_SUITE,
    SAR_PANEL_SUITE,
    SDEM_PANEL_SUITE,
    SDM_PANEL_SUITE,
    SEM_PANEL_DYNAMIC_SUITE,
    SEM_PANEL_SUITE,
    SLX_PANEL_DYNAMIC_SUITE,
    SLX_PANEL_SUITE,
)

# Panel FE
_register("bayespecon.models.panel.OLSPanelFE", OLS_PANEL_SUITE)
_register("bayespecon.models.panel.SARPanelFE", SAR_PANEL_SUITE)
_register("bayespecon.models.panel.SEMPanelFE", SEM_PANEL_SUITE)
_register("bayespecon.models.panel.SDMPanelFE", SDM_PANEL_SUITE)
_register("bayespecon.models.panel.SDEMPanelFE", SDEM_PANEL_SUITE)
_register("bayespecon.models.panel.SLXPanelFE", SLX_PANEL_SUITE)

# Panel RE
_register("bayespecon.models.panel_re.OLSPanelRE", OLS_PANEL_SUITE)
_register("bayespecon.models.panel_re.SARPanelRE", SAR_PANEL_SUITE)
_register("bayespecon.models.panel_re.SEMPanelRE", SEM_PANEL_SUITE)
_register("bayespecon.models.panel_re.SDEMPanelRE", SDEM_PANEL_SUITE)

# Panel Dynamic
_register("bayespecon.models.panel_dynamic.OLSPanelDynamic", OLS_PANEL_SUITE)
_register("bayespecon.models.panel_dynamic.SDMRPanelDynamic", SDM_PANEL_SUITE)
_register("bayespecon.models.panel_dynamic.SDMUPanelDynamic", SDM_PANEL_SUITE)
_register("bayespecon.models.panel_dynamic.SARPanelDynamic", SAR_PANEL_SUITE)
_register("bayespecon.models.panel_dynamic.SEMPanelDynamic", SEM_PANEL_DYNAMIC_SUITE)
_register("bayespecon.models.panel_dynamic.SDEMPanelDynamic", SDEM_PANEL_SUITE)
_register("bayespecon.models.panel_dynamic.SLXPanelDynamic", SLX_PANEL_DYNAMIC_SUITE)

# Panel Tobit
_register("bayespecon.models.panel_tobit.SARPanelTobit", SAR_PANEL_SUITE)
_register("bayespecon.models.panel_tobit.SEMPanelTobit", SEM_PANEL_DYNAMIC_SUITE)

# Flow Panel
_register("bayespecon.models.flow_panel.SARFlowPanel", FLOW_PANEL_SUITE)
_register("bayespecon.models.flow_panel.SARFlowSeparablePanel", FLOW_PANEL_SUITE)
_register("bayespecon.models.flow_panel.PoissonSARFlowPanel", FLOW_PANEL_SUITE)
_register("bayespecon.models.flow_panel.PoissonSARFlowSeparablePanel", FLOW_PANEL_SUITE)
_register("bayespecon.models.flow_panel.OLSFlowPanel", FLOW_PANEL_SUITE)
_register("bayespecon.models.flow_panel.SEMFlowPanel", FLOW_PANEL_SUITE)
_register("bayespecon.models.flow_panel.SEMFlowSeparablePanel", FLOW_PANEL_SUITE)
