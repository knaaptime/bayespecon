"""Unit tests for the FE-vs-RE Hausman diagnostic helper."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import hausman_fe_re_test


def test_hausman_fe_re_returns_valid_result():
    """Hausman helper should return finite statistic and bounded p-value."""
    beta_fe = np.array([1.10, 1.95])
    beta_re = np.array([1.00, 1.80])
    cov_fe = np.array([[0.060, 0.010], [0.010, 0.080]])
    cov_re = np.array([[0.020, 0.005], [0.005, 0.030]])

    out = hausman_fe_re_test(
        beta_fe=beta_fe,
        cov_fe=cov_fe,
        beta_re=beta_re,
        cov_re=cov_re,
        coef_names=["x1", "x2"],
    )

    assert out.name == "hausman_fe_re"
    assert np.isfinite(out.statistic)
    assert 0.0 <= out.pvalue <= 1.0
    assert out.extra["n_coef"] == 2
    assert out.extra["coefficients"] == ["x1", "x2"]


def test_hausman_fe_re_shape_mismatch_raises():
    """Mismatched FE/RE coefficient lengths should raise."""
    with pytest.raises(ValueError, match="same shape"):
        hausman_fe_re_test(
            beta_fe=np.array([1.0, 2.0]),
            cov_fe=np.eye(2),
            beta_re=np.array([1.0]),
            cov_re=np.eye(1),
        )


def test_hausman_fe_re_coef_names_mismatch_raises():
    """coef_names length must match the number of coefficients."""
    with pytest.raises(ValueError, match="coef_names length"):
        hausman_fe_re_test(
            beta_fe=np.array([1.0, 2.0]),
            cov_fe=np.eye(2),
            beta_re=np.array([1.1, 1.9]),
            cov_re=0.5 * np.eye(2),
            coef_names=["x1"],
        )


def test_hausman_fe_re_rank_zero_returns_nan_pvalue():
    """When covariance difference is rank-0, p-value should be NaN."""
    out = hausman_fe_re_test(
        beta_fe=np.array([1.0, 2.0]),
        cov_fe=np.eye(2),
        beta_re=np.array([1.0, 2.0]),
        cov_re=np.eye(2),
    )
    assert out.extra["dof"] == 0
    assert np.isnan(out.pvalue)
