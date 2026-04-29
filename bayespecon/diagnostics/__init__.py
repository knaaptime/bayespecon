"""Bayesian diagnostics for spatial econometric models.

This sub-package provides:

- **Bayesian LM tests** — Lagrange-multiplier-style specification tests
  evaluated over posterior draws rather than point estimates.
- **Bayes factor comparison** — Bridge-sampling and BIC-based model comparison.

Both modules operate on fitted Bayesian model objects (i.e. models with
``inference_data`` attached) and are fully posterior-aware.
"""

from .bayesfactor import (
    bayes_factor_compare_models,
    bic_to_bf,
    compile_log_posterior,
    post_prob,
)
from .mcmc_efficiency import SpatialMCMCReport, spatial_mcmc_diagnostic
from .bayesian_lmtests import (
    BayesianLMTestResult,
    bayesian_lm_error_test,
    bayesian_lm_flow_dest_test,
    bayesian_lm_flow_intra_test,
    bayesian_lm_flow_joint_test,
    bayesian_lm_flow_network_test,
    bayesian_lm_flow_orig_test,
    bayesian_lm_lag_test,
    bayesian_lm_sdm_joint_test,
    bayesian_lm_slx_error_joint_test,
    bayesian_lm_wx_test,
    bayesian_panel_lm_error_test,
    bayesian_panel_lm_flow_dest_test,
    bayesian_panel_lm_flow_intra_test,
    bayesian_panel_lm_flow_joint_test,
    bayesian_panel_lm_flow_network_test,
    bayesian_panel_lm_flow_orig_test,
    # Panel LM tests
    bayesian_panel_lm_lag_test,
    bayesian_panel_lm_sdm_joint_test,
    bayesian_panel_lm_slx_error_joint_test,
    bayesian_panel_lm_wx_sem_test,
    bayesian_panel_lm_wx_test,
    bayesian_panel_robust_lm_error_sdem_test,
    bayesian_panel_robust_lm_error_test,
    bayesian_panel_robust_lm_lag_sdm_test,
    bayesian_panel_robust_lm_lag_test,
    bayesian_panel_robust_lm_wx_test,
    bayesian_robust_lm_error_sdem_test,
    bayesian_robust_lm_flow_dest_test,
    bayesian_robust_lm_flow_network_test,
    bayesian_robust_lm_flow_orig_test,
    bayesian_robust_lm_lag_sdm_test,
    bayesian_robust_lm_wx_test,
)

__all__ = [
    # Bayesian LM tests (cross-sectional)
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
    "bayesian_panel_lm_wx_sem_test",
    # Flow LM tests (cross-sectional)
    "bayesian_lm_flow_dest_test",
    "bayesian_lm_flow_orig_test",
    "bayesian_lm_flow_network_test",
    "bayesian_lm_flow_joint_test",
    "bayesian_lm_flow_intra_test",
    "bayesian_robust_lm_flow_dest_test",
    "bayesian_robust_lm_flow_orig_test",
    "bayesian_robust_lm_flow_network_test",
    # Flow LM tests (panel)
    "bayesian_panel_lm_flow_dest_test",
    "bayesian_panel_lm_flow_orig_test",
    "bayesian_panel_lm_flow_network_test",
    "bayesian_panel_lm_flow_joint_test",
    "bayesian_panel_lm_flow_intra_test",
    # Bayes factor comparison
    "bayes_factor_compare_models",
    "bic_to_bf",
    "compile_log_posterior",
    "post_prob",
    # MCMC sampling-efficiency diagnostic (Wolf et al. 2018)
    "SpatialMCMCReport",
    "spatial_mcmc_diagnostic",
]
