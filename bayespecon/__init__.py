"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes and Bayesian specification tests.

Examples
--------
Import a model class directly from the package namespace::

        from bayespecon import SAR
"""

from . import dgp
from .diagnostics import (
    BayesianLMTestResult,
    bayesian_lm_lag_test,
    bayesian_lm_error_test,
    bayesian_lm_wx_test,
    bayesian_lm_sdm_joint_test,
    bayesian_lm_slx_error_joint_test,
    bayesian_robust_lm_lag_sdm_test,
    bayesian_robust_lm_wx_test,
    bayesian_robust_lm_error_sdem_test,
    summarize_bayesian_lm_test,
    bayes_factor_compare_models,
    bic_to_bf,
    compile_log_posterior,
    post_prob,
)
from .models import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    DLMPanelFE,
    OLSPanelFE,
    OLSPanelRE,
    SARPanelFE,
    SARPanelRE,
    SARPanelTobit,
    SARTobit,
    SDEMPanelFE,
    SDMPanelFE,
    SDMRPanelFE,
    SDMTobit,
    SDMUPanelFE,
    SEMPanelFE,
    SEMPanelRE,
    SEMPanelTobit,
    SEMTobit,
    SpatialProbit,
)

__all__ = [
    "SLX",
    "OLS",
    "SAR",
    "SEM",
    "SDM",
    "SDEM",
    "SARTobit",
    "SEMTobit",
    "SDMTobit",
    "SpatialProbit",
    "OLSPanelFE",
    "SARPanelFE",
    "SEMPanelFE",
    "SDMPanelFE",
    "SDEMPanelFE",
    "DLMPanelFE",
    "SDMRPanelFE",
    "SDMUPanelFE",
    "OLSPanelRE",
    "SARPanelRE",
    "SEMPanelRE",
    "SARPanelTobit",
    "SEMPanelTobit",
    "dgp",
    "bayes_factor_compare_models",
    "bic_to_bf",
    "BayesianLMTestResult",
    "bayesian_lm_lag_test",
    "bayesian_lm_error_test",
    "bayesian_lm_wx_test",
    "bayesian_lm_sdm_joint_test",
    "bayesian_lm_slx_error_joint_test",
    "bayesian_robust_lm_lag_sdm_test",
    "bayesian_robust_lm_wx_test",
    "bayesian_robust_lm_error_sdem_test",
    "summarize_bayesian_lm_test",
]


import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespecon")
