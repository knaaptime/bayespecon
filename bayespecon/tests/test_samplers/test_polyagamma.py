"""Unit tests for the Pólya–Gamma sampler wrapper."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from bayespecon.samplers._utils._polyagamma import sample_polyagamma


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


class TestPolyagammaMethodDispatch:
    """Tests that sample_polyagamma dispatches on h integrality."""

    @patch("polyagamma.random_polyagamma")
    def test_integer_h_uses_hybrid(self, mock_pg, rng):
        """When all h are integer, method=None (hybrid) is used."""
        mock_pg.return_value = np.ones(10)

        h = np.ones(10)  # all integer (logit case)
        z = np.zeros(10)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] is None

    @patch("polyagamma.random_polyagamma")
    def test_noninteger_h_uses_alternate(self, mock_pg, rng):
        """When h has non-integer values, method='alternate' is forced."""
        mock_pg.return_value = np.ones(10)

        h = np.array([1.5, 2.3, 3.7, 4.1, 5.9, 6.2, 7.8, 8.0, 9.5, 10.1])
        z = np.zeros(10)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] == "alternate"

    @patch("polyagamma.random_polyagamma")
    def test_mixed_h_uses_alternate(self, mock_pg, rng):
        """When h has mixed integer and non-integer values, method='alternate'."""
        mock_pg.return_value = np.ones(5)

        h = np.array([1.0, 2.5, 3.0, 4.7, 5.0])  # some integer, some not
        z = np.zeros(5)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] == "alternate"

    @patch("polyagamma.random_polyagamma")
    def test_large_integer_h_uses_hybrid(self, mock_pg, rng):
        """Large integer h values (e.g. h=5) also use hybrid method."""
        mock_pg.return_value = np.ones(10)

        h = np.full(10, 5.0)  # all integer, h=5
        z = np.zeros(10)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] is None

    def test_logit_h_produces_valid_draws(self, rng):
        """End-to-end: logit-style h=1 draws are valid PG(1, z) samples."""
        n = 5000
        h = np.ones(n)  # logit case
        z = rng.normal(size=n)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (n,)
        assert np.all(omega > 0)
        # PG(1, 0) has mean 0.25; with z != 0 the mean shifts but stays positive
        assert np.mean(omega) > 0

    def test_negbin_h_produces_valid_draws(self, rng):
        """End-to-end: negbin-style non-integer h draws are valid."""
        n = 5000
        y = rng.integers(0, 10, size=n)
        alpha = 2.5
        h = y + alpha  # non-integer
        z = rng.normal(size=n)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (n,)
        assert np.all(omega > 0)
