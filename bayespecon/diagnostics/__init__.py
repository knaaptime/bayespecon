"""Bayesian diagnostics for spatial econometric models.

This sub-package provides:

- **Bayesian LM tests** — Lagrange-multiplier-style specification tests
  evaluated over posterior draws rather than point estimates.
- **Bayes factor comparison** — Bridge-sampling and BIC-based model comparison.

Both modules operate on fitted Bayesian model objects (i.e. models with
``inference_data`` attached) and are fully posterior-aware.
"""

from .bayesian_lmtests import (
    BayesianLMTestResult,
    bayesian_lm_lag_test,
    bayesian_lm_error_test,
    bayesian_lm_wx_test,
    bayesian_lm_sdm_joint_test,
    bayesian_lm_slx_error_joint_test,
    bayesian_robust_lm_lag_sdm_test,
    bayesian_robust_lm_wx_test,
    bayesian_robust_lm_error_sdem_test,
    # Panel LM tests
    bayesian_panel_lm_lag_test,
    bayesian_panel_lm_error_test,
    bayesian_panel_robust_lm_lag_test,
    bayesian_panel_robust_lm_error_test,
    bayesian_panel_lm_wx_test,
    bayesian_panel_lm_sdm_joint_test,
    bayesian_panel_lm_slx_error_joint_test,
    bayesian_panel_robust_lm_lag_sdm_test,
    bayesian_panel_robust_lm_wx_test,
    bayesian_panel_robust_lm_error_sdem_test,
)
from .bayesfactor import (
    bayes_factor_compare_models,
    bic_to_bf,
    compile_log_posterior,
    post_prob,
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
    # Bayes factor comparison
    "bayes_factor_compare_models",
    "bic_to_bf",
    "compile_log_posterior",
    "post_prob",
]