import numpy as np
import arviz as az
import pytest
from unittest.mock import MagicMock

from bayespecon.diagnostics.bayesian_lmtests import (
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
)


def make_sar_sem_data(n=20, seed=42):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([1.0, 2.0])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)
    # Simple row-normalized spatial weights (ring)
    W_dense = np.eye(n, k=1) + np.eye(n, k=-1)
    W_dense[0, -1] = W_dense[-1, 0] = 1
    W_dense = W_dense / W_dense.sum(axis=1, keepdims=True)
    import scipy.sparse as sp
    W_sparse = sp.csr_matrix(W_dense)
    return y, X, W_dense, W_sparse


def _make_mock_model(y, X, W_sparse, beta_noise=0.1, draws=100):
    """Build a mock model object that exposes the attributes expected by the
    Bayesian LM test functions."""
    n, k = X.shape
    beta_samples = np.tile(np.linalg.lstsq(X, y, rcond=None)[0], (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    # Pre-compute Wy from sparse W (as SpatialModel does)
    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    # Build InferenceData using az.from_dict (works with all ArViZ versions)
    idata = az.from_dict(
        posterior={"beta": beta_samples[:, None, :], "sigma": np.ones(draws)[:, None]},
        observed_data={"y": y},
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = np.empty((n, 0), dtype=float)
    model._Wy = Wy
    model._W_sparse = W_sparse
    model.inference_data = idata
    return model


def test_bayesian_lm_lag_and_error_basic():
    y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
    model = _make_mock_model(y, X, W_sparse, beta_noise=0.1, draws=200)

    # LM-lag (SAR)
    result_lag = bayesian_lm_lag_test(model)
    assert isinstance(result_lag.mean, float)
    assert result_lag.lm_samples.shape[0] == 200
    assert 0.0 <= result_lag.bayes_pvalue <= 1.0
    summarize_bayesian_lm_test(result_lag)

    # LM-error (SEM)
    result_err = bayesian_lm_error_test(model)
    assert isinstance(result_err.mean, float)
    assert result_err.lm_samples.shape[0] == 200
    assert 0.0 <= result_err.bayes_pvalue <= 1.0
    summarize_bayesian_lm_test(result_err)


def test_bayesian_lm_lag_and_error_extreme():
    y, X, W_dense, W_sparse = make_sar_sem_data(n=10)
    model = _make_mock_model(y, X, W_sparse, beta_noise=0.0, draws=50)

    # All beta draws are the same (perfect fit), so LM should be very small
    result_lag = bayesian_lm_lag_test(model)
    assert np.all(result_lag.lm_samples >= 0)
    assert result_lag.mean < 10

    result_err = bayesian_lm_error_test(model)
    assert np.all(result_err.lm_samples >= 0)
    assert result_err.mean < 10


def test_bayesian_lm_result_df_field():
    """Test that BayesianLMTestResult has a df field with default 1."""
    y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
    model = _make_mock_model(y, X, W_sparse, beta_noise=0.1, draws=50)
    result = bayesian_lm_lag_test(model)
    assert hasattr(result, "df")
    assert result.df == 1


# ---------------------------------------------------------------------------
# Mock model builders for SDM/SDEM variant tests
# ---------------------------------------------------------------------------


def _make_mock_ols_model_with_wx(y, X, WX, W_sparse, beta_noise=0.1, draws=100):
    """Build a mock OLS model with WX columns (for joint tests)."""
    n, k = X.shape
    Z = np.hstack([X, WX])  # OLS uses only X for residuals, but has WX available
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]  # OLS on X only
    beta_samples = np.tile(beta_ols, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    idata = az.from_dict(
        posterior={"beta": beta_samples[:, None, :], "sigma": np.ones(draws)[:, None]},
        observed_data={"y": y},
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model.inference_data = idata
    return model


def _make_mock_sar_model(y, X, WX, W_sparse, beta_noise=0.1, rho_noise=0.05, draws=100):
    """Build a mock SAR model with rho in the posterior."""
    n, k = X.shape
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_samples = np.tile(beta_ols, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)
    rho_samples = rng.normal(scale=rho_noise, size=draws)  # near zero under H₀

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    idata = az.from_dict(
        posterior={
            "beta": beta_samples[:, None, :],
            "rho": rho_samples[:, None],
            "sigma": np.ones(draws)[:, None],
        },
        observed_data={"y": y},
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model.inference_data = idata
    return model


def _make_mock_slx_model(y, X, WX, W_sparse, beta_noise=0.1, draws=100):
    """Build a mock SLX model (beta covers [X, WX], no spatial params)."""
    n, k = X.shape
    k_wx = WX.shape[1]
    Z = np.hstack([X, WX])
    beta_slx = np.linalg.lstsq(Z, y, rcond=None)[0]
    beta_samples = np.tile(beta_slx, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    idata = az.from_dict(
        posterior={
            "beta": beta_samples[:, None, :],
            "sigma": np.ones(draws)[:, None],
        },
        observed_data={"y": y},
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model.inference_data = idata
    return model


def _make_data_with_wx(n=20, k_wx=2, seed=42):
    """Generate data with WX columns for multi-parameter tests."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    WX = rng.normal(size=(n, k_wx))
    beta_true = np.array([1.0, 2.0])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)

    W_dense = np.eye(n, k=1) + np.eye(n, k=-1)
    W_dense[0, -1] = W_dense[-1, 0] = 1
    W_dense = W_dense / W_dense.sum(axis=1, keepdims=True)
    import scipy.sparse as sp
    W_sparse = sp.csr_matrix(W_dense)
    return y, X, WX, W_dense, W_sparse


# ---------------------------------------------------------------------------
# Tests for bayesian_lm_wx_test
# ---------------------------------------------------------------------------


class TestBayesianLMWxTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_wx_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 2  # k_wx = 2
        assert result.test_type == "bayesian_lm_wx"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_single_wx_column(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=1)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=100)
        result = bayesian_lm_wx_test(model)

        assert result.df == 1
        assert result.lm_samples.shape[0] == 100

    def test_no_wx_raises(self):
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        with pytest.raises(ValueError, match="no WX columns"):
            bayesian_lm_wx_test(model)


# ---------------------------------------------------------------------------
# Tests for bayesian_lm_sdm_joint_test
# ---------------------------------------------------------------------------


class TestBayesianLMSDMJointTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_sdm_joint_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 3  # 1 + k_wx = 1 + 2
        assert result.test_type == "bayesian_lm_sdm_joint"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_single_wx_column(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=1)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=100)
        result = bayesian_lm_sdm_joint_test(model)

        assert result.df == 2  # 1 + 1

    def test_zero_wx_columns(self):
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        result = bayesian_lm_sdm_joint_test(model)

        assert result.df == 1  # 1 + 0
        assert result.test_type == "bayesian_lm_sdm_joint"


# ---------------------------------------------------------------------------
# Tests for bayesian_lm_slx_error_joint_test
# ---------------------------------------------------------------------------


class TestBayesianLMSLXErrorJointTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_slx_error_joint_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 3  # 1 + k_wx = 1 + 2
        assert result.test_type == "bayesian_lm_slx_error_joint"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_single_wx_column(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=1)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=100)
        result = bayesian_lm_slx_error_joint_test(model)

        assert result.df == 2  # 1 + 1


# ---------------------------------------------------------------------------
# Tests for bayesian_robust_lm_lag_sdm_test
# ---------------------------------------------------------------------------


class TestBayesianRobustLMLagSDMTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_slx_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_robust_lm_lag_sdm_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_robust_lm_lag_sdm"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_columns(self):
        """Robust test with no WX columns should still work (no adjustment)."""
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        result = bayesian_robust_lm_lag_sdm_test(model)

        assert result.df == 1
        assert result.lm_samples.shape[0] == 50


# ---------------------------------------------------------------------------
# Tests for bayesian_robust_lm_wx_test
# ---------------------------------------------------------------------------


class TestBayesianRobustLMWXTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_robust_lm_wx_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 2  # k_wx = 2
        assert result.test_type == "bayesian_robust_lm_wx"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_raises(self):
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        with pytest.raises(ValueError, match="no WX columns"):
            bayesian_robust_lm_wx_test(model)


# ---------------------------------------------------------------------------
# Tests for bayesian_robust_lm_error_sdem_test
# ---------------------------------------------------------------------------


class TestBayesianRobustLMErrorSDEMTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_slx_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_robust_lm_error_sdem_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_robust_lm_error_sdem"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_columns(self):
        """Robust test with no WX columns should still work (no adjustment)."""
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        result = bayesian_robust_lm_error_sdem_test(model)

        assert result.df == 1
        assert result.lm_samples.shape[0] == 50


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


class TestSummarizeAllNewTests:
    """Ensure summarize_bayesian_lm_test works for all new test types."""

    def test_summarize_wx(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=50)
        result = bayesian_lm_wx_test(model)
        # Should not raise
        summarize_bayesian_lm_test(result)

    def test_summarize_sdm_joint(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=50)
        result = bayesian_lm_sdm_joint_test(model)
        summarize_bayesian_lm_test(result)

    def test_summarize_slx_error_joint(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=50)
        result = bayesian_lm_slx_error_joint_test(model)
        summarize_bayesian_lm_test(result)

    def test_summarize_robust_lag_sdm(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_slx_model(y, X, WX, W_sparse, draws=50)
        result = bayesian_robust_lm_lag_sdm_test(model)
        summarize_bayesian_lm_test(result)

    def test_summarize_robust_wx(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=50)
        result = bayesian_robust_lm_wx_test(model)
        summarize_bayesian_lm_test(result)

    def test_summarize_robust_error_sdem(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_slx_model(y, X, WX, W_sparse, draws=50)
        result = bayesian_robust_lm_error_sdem_test(model)
        summarize_bayesian_lm_test(result)


# ---------------------------------------------------------------------------
# Tests that robust tests differ from non-robust tests (cross-term fix)
# ---------------------------------------------------------------------------


class TestRobustVsNonRobust:
    """Verify that robust tests produce different results from non-robust tests.

    After fixing J_{ργ·σ} to be non-zero (instead of zeros), the Neyman
    adjustment should have an effect, making robust tests differ from
    their non-robust counterparts.
    """

    def test_robust_wx_differs_from_nonrobust_wx(self):
        """Robust LM-WX (from SAR) should differ from LM-WX (from SAR)."""
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=200)

        result_wx = bayesian_lm_wx_test(model)
        result_robust_wx = bayesian_robust_lm_wx_test(model)

        # The means should differ because the Neyman adjustment removes
        # the correlation with ρ, changing the test statistic
        assert result_wx.mean != result_robust_wx.mean, (
            f"Robust LM-WX mean ({result_robust_wx.mean:.4f}) should differ "
            f"from LM-WX mean ({result_wx.mean:.4f})"
        )

    def test_robust_lag_sdm_differs_from_lag(self):
        """Robust LM-Lag-SDM (from SLX) should differ from LM-Lag (from OLS).

        Note: These use different null models (SLX vs OLS), so they naturally
        differ. But the Neyman adjustment should also contribute.
        """
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        ols_model = _make_mock_model(y, X, W_sparse, draws=200)
        slx_model = _make_mock_slx_model(y, X, WX, W_sparse, draws=200)

        result_lag = bayesian_lm_lag_test(ols_model)
        result_robust_lag = bayesian_robust_lm_lag_sdm_test(slx_model)

        # These should differ (different null models + Neyman adjustment)
        assert result_lag.mean != result_robust_lag.mean
