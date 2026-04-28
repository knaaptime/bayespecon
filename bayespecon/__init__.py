"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes and Bayesian specification tests.

Examples
--------
Import a model class directly from the package namespace::

        from bayespecon import SAR
"""

from . import dgp
from .dgp.flows import generate_flow_data, generate_poisson_flow_data
from .diagnostics import (
    BayesianLMTestResult,
    bayes_factor_compare_models,
    bayesian_lm_error_test,
    bayesian_lm_lag_test,
    bayesian_lm_sdm_joint_test,
    bayesian_lm_slx_error_joint_test,
    bayesian_lm_wx_test,
    bayesian_panel_lm_error_test,
    # Panel LM tests
    bayesian_panel_lm_lag_test,
    bayesian_panel_lm_sdm_joint_test,
    bayesian_panel_lm_slx_error_joint_test,
    bayesian_panel_lm_wx_test,
    bayesian_panel_robust_lm_error_sdem_test,
    bayesian_panel_robust_lm_error_test,
    bayesian_panel_robust_lm_lag_sdm_test,
    bayesian_panel_robust_lm_lag_test,
    bayesian_panel_robust_lm_wx_test,
    bayesian_robust_lm_error_sdem_test,
    bayesian_robust_lm_lag_sdm_test,
    bayesian_robust_lm_wx_test,
    bic_to_bf,
    compile_log_posterior,
    post_prob,
)
from .graph import (
    FlowDesignMatrix,
    destination_weights,
    flow_design_matrix,
    flow_design_matrix_with_orig,
    flow_weight_matrices,
    network_weights,
    origin_weights,
)
from .models import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    OLSPanelDynamic,
    OLSPanelFE,
    OLSPanelRE,
    PoissonSARFlowPanel,
    PoissonSARFlowSeparablePanel,
    SARFlowPanel,
    SARFlowSeparablePanel,
    SARPanelDynamic,
    SARPanelFE,
    SARPanelRE,
    SARPanelTobit,
    SARTobit,
    SDEMPanelDynamic,
    SDEMPanelFE,
    SDEMPanelRE,
    SDMPanelFE,
    SDMRPanelDynamic,
    SDMTobit,
    SDMUPanelDynamic,
    SEMPanelDynamic,
    SEMPanelFE,
    SEMPanelRE,
    SEMPanelTobit,
    SEMTobit,
    SLXPanelDynamic,
    SLXPanelFE,
    SpatialProbit,
)
from .models.flow import (
    PoissonSARFlow,
    PoissonSARFlowSeparable,
    SARFlow,
    SARFlowSeparable,
)
from .ops import SparseFlowSolveOp, kron_solve_matrix, kron_solve_vec

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
    "SLXPanelFE",
    "OLSPanelDynamic",
    "SDMRPanelDynamic",
    "SDMUPanelDynamic",
    "SARPanelDynamic",
    "SEMPanelDynamic",
    "SDEMPanelDynamic",
    "SLXPanelDynamic",
    "OLSPanelRE",
    "SARPanelRE",
    "SEMPanelRE",
    "SDEMPanelRE",
    "SARPanelTobit",
    "SEMPanelTobit",
    "dgp",
    "SARFlow",
    "SARFlowSeparable",
    "PoissonSARFlow",
    "PoissonSARFlowSeparable",
    "SARFlowPanel",
    "SARFlowSeparablePanel",
    "PoissonSARFlowPanel",
    "PoissonSARFlowSeparablePanel",
    "SparseFlowSolveOp",
    "destination_weights",
    "origin_weights",
    "network_weights",
    "flow_weight_matrices",
    "flow_design_matrix",
    "flow_design_matrix_with_orig",
    "FlowDesignMatrix",
    "generate_flow_data",
    "generate_poisson_flow_data",
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
    # Panel LM tests
    "bayesian_panel_lm_lag_test",
    "bayesian_panel_lm_error_test",
    "bayesian_panel_robust_lm_lag_test",
    "bayesian_panel_robust_lm_error_test",
    "bayesian_panel_lm_wx_test",
    "bayesian_panel_lm_sdm_joint_test",
    "bayesian_panel_lm_slx_error_joint_test",
    "bayesian_panel_robust_lm_lag_sdm_test",
    "bayesian_panel_robust_lm_wx_test",
    "bayesian_panel_robust_lm_error_sdem_test",
]


import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespecon")
