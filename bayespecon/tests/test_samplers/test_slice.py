"""Unit tests for the slice sampler primitive."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon.samplers._utils._slice import slice_sample_1d


class TestSliceSample1d:
    """Tests for the univariate slice sampler."""

    def test_truncated_normal(self, rng):
        """Sample from N(0, 1) truncated to [-3, 3]."""

        def log_density(x):
            return -0.5 * x * x  # unnormalized N(0, 1)

        samples = []
        x = 0.0
        for _ in range(5000):
            x, _ = slice_sample_1d(log_density, x, -3.0, 3.0, w=1.0, rng=rng)
            samples.append(x)
        samples = np.array(samples)
        assert abs(np.mean(samples)) < 0.1, f"Mean {np.mean(samples):.3f} far from 0"
        assert abs(np.var(samples) - 1.0) < 0.2, f"Var {np.var(samples):.3f} far from 1"

    def test_gamma(self, rng):
        """Sample from Gamma(2, 1) via log-density log(x) - x."""

        def log_density(x):
            return np.log(x) - x if x > 0 else -np.inf

        samples = []
        x = 2.0
        for _ in range(5000):
            x, _ = slice_sample_1d(log_density, x, 1e-6, 20.0, w=1.0, rng=rng)
            samples.append(x)
        samples = np.array(samples)
        assert abs(np.mean(samples) - 2.0) < 0.5, (
            f"Mean {np.mean(samples):.3f} far from 2"
        )
        assert abs(np.var(samples) - 2.0) < 1.0, f"Var {np.var(samples):.3f} far from 2"

    def test_returns_log_density(self, rng):
        """slice_sample_1d returns the log-density at the new point."""

        def log_density(x):
            return -0.5 * x * x

        x_new, ld_new = slice_sample_1d(log_density, 0.0, -5.0, 5.0, rng=rng)
        assert np.isclose(ld_new, log_density(x_new), atol=1e-12)

    def test_bounds_violation_raises(self):
        """x0 outside [lower, upper] raises ValueError."""
        with pytest.raises(ValueError, match="x0"):
            slice_sample_1d(lambda x: 0.0, 5.0, 0.0, 1.0)

    def test_degenerate_bounds_raises(self):
        """lower >= upper raises ValueError."""
        with pytest.raises(ValueError, match="lower"):
            slice_sample_1d(lambda x: 0.0, 0.5, 1.0, 0.0)

    def test_small_w_converges(self, rng):
        """Very small w still produces valid samples (slower mixing)."""

        def log_density(x):
            return -0.5 * x * x

        x = 0.0
        for _ in range(200):
            x, _ = slice_sample_1d(log_density, x, -5.0, 5.0, w=0.01, rng=rng)
        assert np.isfinite(x)

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)
