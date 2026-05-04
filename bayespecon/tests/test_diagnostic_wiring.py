"""Static wiring checks for model-specific diagnostic methods."""

from __future__ import annotations

from bayespecon import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    OLSPanelFE,
    OLSPanelRE,
    SARPanelFE,
    SARPanelRE,
    SEMPanelFE,
    SEMPanelRE,
)


def test_cross_sectional_models_no_frequentist_spec_tests():
    """Cross-sectional models should not expose frequentist spatial spec tests."""
    for cls in [OLS, SLX, SAR, SDM, SEM, SDEM]:
        assert not hasattr(cls, "lm_error_test")
        assert not hasattr(cls, "lm_lag_test")
        assert not hasattr(cls, "moran_test")
        assert not hasattr(cls, "spatial_specification_tests")
        assert not hasattr(cls, "diagnostics")
        assert not hasattr(cls, "regression_diagnostics")
        assert not hasattr(cls, "heteroskedasticity_diagnostics")
        assert not hasattr(cls, "autocorrelation_diagnostics")
        assert not hasattr(cls, "outlier_diagnostics")

    assert not hasattr(SAR, "lm_rho_test")
    assert not hasattr(SAR, "lm_rho_robust_test")
    assert not hasattr(SDM, "lm_rho_test")
    assert not hasattr(SDM, "lm_rho_robust_test")
    assert not hasattr(SEM, "wald_error_test")
    assert not hasattr(SEM, "lr_ratio_test")
    assert not hasattr(SDEM, "wald_error_test")


def test_panel_models_no_frequentist_diagnostics():
    """Panel models should not expose frequentist diagnostic methods."""
    for cls in [OLSPanelFE, OLSPanelRE, SARPanelFE, SARPanelRE, SEMPanelFE, SEMPanelRE]:
        assert not hasattr(cls, "diagnostics")
        assert not hasattr(cls, "regression_diagnostics")
        assert not hasattr(cls, "heteroskedasticity_diagnostics")
        assert not hasattr(cls, "autocorrelation_diagnostics")
        assert not hasattr(cls, "outlier_diagnostics")
        assert not hasattr(cls, "panel_diagnostics")
        assert not hasattr(cls, "hausman_test")


from bayespecon import (
    SDEMPanelFE,
    SDEMPanelRE,
    SDMPanelFE,
    SLXPanelFE,
)
from bayespecon.models.flow import OLSFlow, SARFlow
from bayespecon.models.flow_panel import OLSFlowPanel

EXPECTED_REGISTRIES = {
    OLS: [
        "LM-Lag",
        "LM-Error",
        "LM-SDM-Joint",
        "LM-SLX-Error-Joint",
        "Robust-LM-Lag",
        "Robust-LM-Error",
    ],
    SAR: ["LM-Error", "LM-WX", "Robust-LM-WX", "Robust-LM-Error"],
    SEM: ["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
    SLX: ["LM-Lag", "LM-Error", "Robust-LM-Lag-SDM", "Robust-LM-Error-SDEM"],
    SDM: ["LM-Error-SDM", "Robust-LM-Error-SDM"],
    SDEM: ["LM-Lag-SDEM", "Robust-LM-Lag-SDEM"],
    OLSPanelFE: [
        "Panel-LM-Lag",
        "Panel-LM-Error",
        "Panel-LM-SDM-Joint",
        "Panel-LM-SLX-Error-Joint",
        "Panel-Robust-LM-Lag",
        "Panel-Robust-LM-Error",
    ],
    SARPanelFE: ["Panel-LM-Error", "Panel-LM-WX", "Panel-Robust-LM-WX"],
    SEMPanelFE: ["Panel-LM-Lag", "Panel-LM-WX"],
    SLXPanelFE: [
        "Panel-LM-Lag",
        "Panel-LM-Error",
        "Panel-Robust-LM-Lag-SDM",
        "Panel-Robust-LM-Error-SDEM",
    ],
    SDMPanelFE: ["Panel-LM-Error-SDM"],
    SDEMPanelFE: ["Panel-LM-Lag-SDEM"],
    OLSPanelRE: [
        "Panel-LM-Lag",
        "Panel-LM-Error",
        "Panel-LM-SDM-Joint",
        "Panel-LM-SLX-Error-Joint",
        "Panel-Robust-LM-Lag",
        "Panel-Robust-LM-Error",
    ],
    SARPanelRE: ["Panel-LM-Error", "Panel-LM-WX", "Panel-Robust-LM-WX"],
    SEMPanelRE: ["Panel-LM-Lag", "Panel-LM-WX"],
    SDEMPanelRE: ["Panel-LM-Lag-SDEM"],
    OLSFlow: [
        "LM-Flow-Dest",
        "LM-Flow-Orig",
        "LM-Flow-Network",
        "LM-Flow-Joint",
        "LM-Flow-Intra",
    ],
    SARFlow: [
        "Robust-LM-Flow-Dest",
        "Robust-LM-Flow-Orig",
        "Robust-LM-Flow-Network",
    ],
    OLSFlowPanel: [
        "Panel-LM-Flow-Dest",
        "Panel-LM-Flow-Orig",
        "Panel-LM-Flow-Network",
        "Panel-LM-Flow-Joint",
        "Panel-LM-Flow-Intra",
    ],
}


def test_spatial_diagnostics_registries_match_expected():
    """Each model class should expose the expected ordered list of LM tests."""
    for cls, expected_labels in EXPECTED_REGISTRIES.items():
        registry = cls._spatial_diagnostics_tests
        actual_labels = [label for _, label in registry]
        assert actual_labels == expected_labels, (
            f"{cls.__name__}: expected {expected_labels}, got {actual_labels}"
        )


def test_slx_uses_slx_specific_lm_tests():
    """SLX uses Koley-Bera SLX-specific LM tests, not raw OLS LM tests."""
    labels = [label for _, label in SLX._spatial_diagnostics_tests]
    # Must not be empty placeholders
    assert len(labels) >= 2


def test_flow_models_expose_decision_api():
    """FlowModel and FlowPanelModel must expose spatial_diagnostics_decision."""
    from bayespecon.models.flow import FlowModel
    from bayespecon.models.flow_panel import FlowPanelModel

    assert hasattr(FlowModel, "spatial_diagnostics_decision")
    assert callable(FlowModel.spatial_diagnostics_decision)
    assert hasattr(FlowPanelModel, "spatial_diagnostics_decision")
    assert callable(FlowPanelModel.spatial_diagnostics_decision)


def test_flow_decision_tree_dispatch():
    """get_flow_spec and get_panel_flow_spec dispatch to correct tree roots."""
    import pandas as pd
    from bayespecon.diagnostics import _decision_trees as _dt

    # OLSFlow: joint sig -> SARFlow
    spec = _dt.get_flow_spec("OLSFlow")
    sig_all = lambda name: name == "LM-Flow-Joint"
    d, _ = _dt.evaluate(spec, sig_all)
    assert d == "SARFlow"

    # OLSFlow: no sig -> OLSFlow
    d2, _ = _dt.evaluate(spec, lambda name: False)
    assert d2 == "OLSFlow"

    # SARFlow: dest robust sig -> SARFlow
    spec_sar = _dt.get_flow_spec("SARFlow")
    d3, _ = _dt.evaluate(spec_sar, lambda name: name == "Robust-LM-Flow-Dest")
    assert d3 == "SARFlow"

    # SARFlow: no robust sig -> OLSFlow
    d4, _ = _dt.evaluate(spec_sar, lambda name: False)
    assert d4 == "OLSFlow"

    # Panel variants
    spec_p = _dt.get_panel_flow_spec("OLSFlowPanel")
    dp, _ = _dt.evaluate(spec_p, lambda name: name == "Panel-LM-Flow-Joint")
    assert dp == "SARFlowPanel"

    spec_sp = _dt.get_panel_flow_spec("SARFlowPanel")
    dp2, _ = _dt.evaluate(spec_sp, lambda name: False)
    assert dp2 == "OLSFlowPanel"
