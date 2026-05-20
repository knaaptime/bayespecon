"""Unit tests for the truncated-normal sampler."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon._samplers._truncnorm import sample_truncnorm


class TestSampleTruncnorm:
    """Tests for the truncated-normal draw function."""

    def test_right_truncation(self, rng):
        """TN(0, 1, lower=0, upper=inf): all draws > 0, mean ≈ sqrt(2/pi)."""
        lower = np.zeros(5000)
        upper = np.full(5000, np.inf)
        draws = sample_truncnorm(lower, upper, rng=rng)
        assert draws.shape == (5000,)
        assert np.all(draws > 0)
        expected_mean = np.sqrt(2.0 / np.pi)  # ≈ 0.798
        assert abs(np.mean(draws) - expected_mean) < 0.05

    def test_left_truncation(self, rng):
        """TN(0, 1, lower=-inf, upper=0): all draws < 0."""
        lower = np.full(1000, -np.inf)
        upper = np.zeros(1000)
        draws = sample_truncnorm(lower, upper, rng=rng)
        assert np.all(draws < 0)

    def test_two_sided_truncation(self, rng):
        """TN(0, 1, lower=-1, upper=1): all draws in [-1, 1]."""
        lower = np.full(2000, -1.0)
        upper = np.full(2000, 1.0)
        draws = sample_truncnorm(lower, upper, rng=rng)
        assert np.all(draws >= -1.0)
        assert np.all(draws <= 1.0)

    def test_far_tail(self, rng):
        """TN(10, 1, lower=10, upper=inf): no zeros or NaNs."""
        lower = np.full(500, 10.0)
        upper = np.full(500, np.inf)
        draws = sample_truncnorm(lower, upper, rng=rng)
        assert np.all(np.isfinite(draws))
        assert np.all(draws >= 10.0)

    def test_scalar_inputs(self, rng):
        """Scalar inputs work correctly."""
        draw = sample_truncnorm(0.0, np.inf, rng=rng)
        assert np.isscalar(draw) or draw.shape == ()
        assert draw > 0

    def test_bounds_violation_raises(self):
        """lower >= upper raises ValueError."""
        with pytest.raises(ValueError, match="lower"):
            sample_truncnorm(1.0, 0.0)

    def test_negative_sigma_raises(self):
        """sigma <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            sample_truncnorm(0.0, 1.0, sigma=-1.0)

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)
