"""Static wiring checks for model-specific diagnostic methods."""

from __future__ import annotations

from bayespecon import (
    OLS,
    SAR,
    SDM,
    SDEM,
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
