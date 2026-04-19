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


def test_cross_sectional_spatial_tests_are_on_supported_models():
    """Cross-sectional models should expose their expected spatial tests."""
    assert hasattr(OLS, "lm_error_test")
    assert hasattr(OLS, "lm_lag_test")

    assert hasattr(SLX, "lm_error_test")
    assert hasattr(SLX, "lm_lag_test")

    assert hasattr(SAR, "lm_error_test")
    assert hasattr(SAR, "lm_rho_test")
    assert hasattr(SAR, "lm_rho_robust_test")

    assert hasattr(SDM, "lm_error_test")
    assert hasattr(SDM, "lm_rho_test")
    assert hasattr(SDM, "lm_rho_robust_test")

    assert hasattr(SEM, "lm_lag_test")
    assert hasattr(SEM, "wald_error_test")
    assert hasattr(SEM, "lr_ratio_test")

    assert hasattr(SDEM, "lm_lag_test")
    assert hasattr(SDEM, "wald_error_test")


def test_panel_hausman_is_only_on_ols_fe():
    """Hausman FE-vs-RE test should be exposed only by OLSPanelFE."""
    assert hasattr(OLSPanelFE, "hausman_test")

    assert not hasattr(OLSPanelRE, "hausman_test")
    assert not hasattr(SARPanelFE, "hausman_test")
    assert not hasattr(SARPanelRE, "hausman_test")
    assert not hasattr(SEMPanelFE, "hausman_test")
    assert not hasattr(SEMPanelRE, "hausman_test")
