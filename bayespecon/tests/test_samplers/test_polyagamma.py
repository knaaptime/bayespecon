"""Unit tests for the Pólya–Gamma sampler wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon._samplers._polyagamma import sample_polyagamma


class TestSamplePolyagamma:
    """Tests for the PG draw wrapper."""

    def test_pg1_mean(self, rng):
        """PG(1, 0) has known mean ≈ 0.25."""
        h = np.ones(5000)
        z = np.zeros(5000)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (5000,)
        assert np.all(omega > 0)
        assert abs(np.mean(omega) - 0.25) < 0.05

    def test_pg5_positive(self, rng):
        """PG(5, 2) draws are all positive."""
        h = np.full(100, 5.0)
        z = np.full(100, 2.0)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (100,)
        assert np.all(omega > 0)

    def test_shape_mismatch_raises(self):
        """h and z with different shapes raises ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            sample_polyagamma(np.ones(10), np.ones(5))

    def test_h_nonpositive_raises(self):
        """h <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            sample_polyagamma(np.array([0.0]), np.array([1.0]))

    def test_scalar_broadcast(self, rng):
        """Scalar h and z work correctly."""
        omega = sample_polyagamma(np.array([3.0]), np.array([1.0]), rng=rng)
        assert omega.shape == (1,)
        assert omega[0] > 0

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)
