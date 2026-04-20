"""Fast branch/error-path tests for diagnostics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import diagnostics


def test_lag_validation_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="positive"):
        diagnostics.arch_test(np.array([1.0, 2.0, 3.0]), lags=0)
    with pytest.raises(ValueError, match="positive"):
        diagnostics.ljung_box_q(np.array([1.0, 2.0, 3.0]), lags=[1, -2])


def test_panel_residual_structure_validates_length():
    with pytest.raises(ValueError, match=r"N\*T"):
        diagnostics.panel_residual_structure(np.arange(5.0), N=2, T=3)


def test_pesaran_cd_returns_nan_for_insufficient_dimensions():
    out = diagnostics.pesaran_cd_test(np.array([1.0, 2.0, 3.0]), N=1, T=3)
    assert np.isnan(out.statistic)
    assert np.isnan(out.pvalue)


def test_outlier_candidates_output_keys_present():
    d = {
        "hatdi": np.array([0.1, 0.8, 0.1]),
        "rstud": np.array([0.1, 2.5, -0.1]),
        "dffit": np.array([0.1, 1.6, 0.1]),
        "dfbeta": np.array([[0.1, 0.1], [2.0, 0.1], [0.1, 0.1]]),
    }
    out = diagnostics.outlier_candidates(d, n=3, k=2)
    assert set(out.keys()) == {"leverage_idx", "rstudent_idx", "dffit_idx", "dfbeta_idx"}
