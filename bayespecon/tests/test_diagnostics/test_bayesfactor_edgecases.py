"""Fast unit tests for bayespecon.diagnostics.bayesfactor edge cases.

Tests bic_to_bf, post_prob, and compile_log_posterior transform
edge cases without running MCMC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayespecon.diagnostics.bayesfactor import bic_to_bf, post_prob

# ---------------------------------------------------------------------------
# bic_to_bf
# ---------------------------------------------------------------------------


class TestBicToBf:
    def test_identity(self):
        """BF of model against itself is 1."""
        result = bic_to_bf([100, 100])
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 1.0)

    def test_lower_bic_preferred(self):
        """Model with lower BIC should have higher BF."""
        result = bic_to_bf([100, 95])
        assert result[1] > result[0]

    def test_explicit_denominator(self):
        result = bic_to_bf([100, 95, 110], denominator=100)
        np.testing.assert_allclose(result[0], 1.0)
        assert result[1] > 1.0  # BIC=95 < 100, so BF > 1
        assert result[2] < 1.0  # BIC=110 > 100, so BF < 1

    def test_log_mode(self):
        result = bic_to_bf([100, 95], log=True)
        np.testing.assert_allclose(result[1], (100 - 95) / 2)

    def test_single_model(self):
        result = bic_to_bf([100])
        np.testing.assert_allclose(result[0], 1.0)


# ---------------------------------------------------------------------------
# post_prob
# ---------------------------------------------------------------------------


class TestPostProb:
    def test_uniform_prior(self):
        result = post_prob([-20.0, -18.0, -19.0])
        assert isinstance(result, pd.Series)
        np.testing.assert_allclose(result.sum(), 1.0)
        # Model with highest log-ML should have highest probability
        assert result.iloc[1] > result.iloc[0]
        assert result.iloc[1] > result.iloc[2]

    def test_custom_names(self):
        result = post_prob([-20.0, -18.0], model_names=["H0", "H1"])
        assert list(result.index) == ["H0", "H1"]

    def test_custom_prior(self):
        result = post_prob([-20.0, -18.0], prior_prob=[0.9, 0.1])
        # Even with strong prior on H0, H1 may still win if evidence is strong
        assert result.sum() == pytest.approx(1.0)

    def test_prior_mismatch_raises(self):
        with pytest.raises(ValueError, match="prior_prob must match"):
            post_prob([-20.0, -18.0], prior_prob=[0.5])

    def test_negative_prior_raises(self):
        with pytest.raises(ValueError, match="prior_prob must be non-negative"):
            post_prob([-20.0, -18.0], prior_prob=[-0.5, 1.5])

    def test_equal_logml(self):
        result = post_prob([-20.0, -20.0])
        np.testing.assert_allclose(result.iloc[0], 0.5)
        np.testing.assert_allclose(result.iloc[1], 0.5)

    def test_auto_names(self):
        result = post_prob([-20.0, -18.0])
        assert list(result.index) == ["model_0", "model_1"]
