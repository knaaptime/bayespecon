"""Unit tests for JAX-accelerated Pólya–Gamma samplers.

Tests the callback method (``jax_polyagamma`` with ``method='callback'``)
and the sum-of-exponentials method (``method='exp'``) against the reference
``polyagamma`` package for correctness, JIT compatibility, and edge cases.

All tests are skipped when JAX is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# Skip entire module if JAX is not available
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX not installed",
)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bayespecon.samplers._utils._jax_polyagamma import (
    jax_polyagamma,
)

# Try to import polyagamma for reference comparisons
try:
    from polyagamma import random_polyagamma as pg_ref_draw

    HAS_POLYAGAMMA = True
except ImportError:
    HAS_POLYAGAMMA = False

needs_polyagamma = pytest.mark.skipif(
    not HAS_POLYAGAMMA,
    reason="polyagamma package not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg1_mean(z: float, n_terms: int = 10000) -> float:
    """Analytical mean of PG(1, z) via truncated series."""
    k = np.arange(n_terms, dtype=np.float64)
    denom = (k + 0.5) ** 2 + (z / (2 * np.pi)) ** 2
    return np.sum(1.0 / denom) / (2 * np.pi**2)


def _pg1_var(z: float, n_terms: int = 10000) -> float:
    """Analytical variance of PG(1, z) via truncated series."""
    k = np.arange(n_terms, dtype=np.float64)
    denom = (k + 0.5) ** 2 + (z / (2 * np.pi)) ** 2
    return np.sum(1.0 / denom**2) / (2 * np.pi**2) ** 2


# ---------------------------------------------------------------------------
# jax_polyagamma tests — sum-of-exponentials method
# ---------------------------------------------------------------------------


class TestJaxPolyaGammaExp:
    """Tests for jax_polyagamma with method='exp' (sum-of-exponentials)."""

    def test_output_shape_vector(self):
        """Output shape should match input shape for vector inputs."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0, 10.0])
        z = jnp.array([0.5, 1.0, 2.0])
        result = jax_polyagamma(h, z, key=key, method="exp")
        assert result.shape == (3,)

    def test_output_shape_scalar(self):
        """Output should be scalar for scalar inputs."""
        key = jax.random.PRNGKey(42)
        result = jax_polyagamma(1.0, 0.5, key=key, method="exp")
        assert result.shape == ()

    def test_positivity(self):
        """All PG draws should be positive."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 2.0, 5.0, 10.0, 50.0])
        z = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
        result = jax_polyagamma(h, z, key=key, method="exp")
        assert jnp.all(result > 0), f"Non-positive draws: {result}"

    def test_mean_approximately_correct(self):
        """Mean of many draws should be within ~5% of analytical mean."""
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        h_val = 1.0
        z_val = 1.0

        # Draw many samples
        keys = jax.random.split(key, n_draws)
        draws = jax.vmap(lambda k: jax_polyagamma(h_val, z_val, key=k, method="exp"))(
            keys
        )

        empirical_mean = float(jnp.mean(draws))
        analytical_mean = _pg1_mean(z_val) * h_val
        rel_error = abs(empirical_mean - analytical_mean) / analytical_mean
        assert rel_error < 0.05, (
            f"Mean error too large: empirical={empirical_mean:.4f}, "
            f"analytical={analytical_mean:.4f}, rel_error={rel_error:.4f}"
        )

    def test_mean_multiple_h(self):
        """Mean should scale with h (PG(h,z) ≈ h * PG(1,z))."""
        key = jax.random.PRNGKey(42)
        n_draws = 3000
        z_val = 1.0

        means = {}
        for h_val in [1.0, 3.0, 10.0]:
            keys = jax.random.split(key, n_draws)
            draws = jax.vmap(
                lambda k: jax_polyagamma(h_val, z_val, key=k, method="exp")
            )(keys)
            means[h_val] = float(jnp.mean(draws))
            key = jax.random.split(key, 1)[0]

        # PG(3,z) mean should be ~3x PG(1,z) mean
        ratio_3 = means[3.0] / means[1.0]
        assert abs(ratio_3 - 3.0) < 0.5, f"PG(3,z)/PG(1,z) = {ratio_3:.2f}, expected ~3"

        # PG(10,z) mean should be ~10x PG(1,z) mean
        ratio_10 = means[10.0] / means[1.0]
        assert abs(ratio_10 - 10.0) < 2.0, (
            f"PG(10,z)/PG(1,z) = {ratio_10:.2f}, expected ~10"
        )

    def test_z_zero_larger_than_z_nonzero(self):
        """PG(h, 0) should be larger than PG(h, z) for z > 0."""
        key = jax.random.PRNGKey(42)
        n_draws = 3000
        h_val = 5.0

        keys = jax.random.split(key, n_draws)
        draws_z0 = jax.vmap(lambda k: jax_polyagamma(h_val, 0.0, key=k, method="exp"))(
            keys
        )

        key = jax.random.split(key, 1)[0]
        keys = jax.random.split(key, n_draws)
        draws_z2 = jax.vmap(lambda k: jax_polyagamma(h_val, 2.0, key=k, method="exp"))(
            keys
        )

        mean_z0 = float(jnp.mean(draws_z0))
        mean_z2 = float(jnp.mean(draws_z2))
        assert mean_z0 > mean_z2, (
            f"PG({h_val},0) mean={mean_z0:.4f} should be > PG({h_val},2) mean={mean_z2:.4f}"
        )

    def test_jit_compatible(self):
        """jax_polyagamma should be JIT-compilable."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0])
        z = jnp.array([0.5, 1.0])

        @jax.jit
        def sample(key, h, z):
            return jax_polyagamma(h, z, key=key, method="exp")

        result = sample(key, h, z)
        assert result.shape == (2,)
        assert jnp.all(result > 0)

    def test_jit_compatible_in_loop(self):
        """jax_polyagamma should work inside a lax.fori_loop (Gibbs use case)."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0])
        z = jnp.array([0.5, 1.0])

        @jax.jit
        def gibbs_step(key, h, z, omega_prev):
            key, subkey = jax.random.split(key)
            omega = jax_polyagamma(h, z, key=subkey, method="exp")
            # Simulate a simple Gibbs update: omega_new = 0.5 * (omega + omega_prev)
            omega_new = 0.5 * (omega + omega_prev)
            return key, omega_new

        omega = jnp.ones_like(h)
        key, omega = gibbs_step(key, h, z, omega)
        assert omega.shape == (2,)
        assert jnp.all(omega > 0)

    def test_batch_sampling(self):
        """Vectorized sampling over observations should work correctly."""
        key = jax.random.PRNGKey(42)
        n = 100
        h = jnp.ones(n)
        z = jnp.linspace(-2, 2, n)

        result = jax_polyagamma(h, z, key=key, method="exp")
        assert result.shape == (n,)
        assert jnp.all(result > 0)

    def test_negative_z(self):
        """PG(h, z) should be the same for z and -z (symmetric in z)."""
        key = jax.random.PRNGKey(42)
        n_draws = 2000
        h_val = 3.0
        z_val = 1.5

        keys = jax.random.split(key, n_draws)
        draws_pos = jax.vmap(
            lambda k: jax_polyagamma(h_val, z_val, key=k, method="exp")
        )(keys)

        key = jax.random.split(key, 1)[0]
        keys = jax.random.split(key, n_draws)
        draws_neg = jax.vmap(
            lambda k: jax_polyagamma(h_val, -z_val, key=k, method="exp")
        )(keys)

        # Means should be approximately equal (PG is symmetric in z)
        mean_pos = float(jnp.mean(draws_pos))
        mean_neg = float(jnp.mean(draws_neg))
        assert abs(mean_pos - mean_neg) / mean_pos < 0.1, (
            f"PG(h,z) != PG(h,-z): {mean_pos:.4f} vs {mean_neg:.4f}"
        )


@needs_polyagamma
class TestJaxPolyaGammaCallback:
    """Tests for jax_polyagamma with method='callback' (exact C extension)."""

    def test_callback_method_basic(self):
        """Callback method should produce positive draws."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0, 10.0])
        z = jnp.array([0.5, 1.0, 2.0])
        result = jax_polyagamma(h, z, key=key, method="callback")
        assert result.shape == (3,)
        assert jnp.all(result > 0)

    def test_callback_method_jit(self):
        """Callback method should work inside JIT."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0])
        z = jnp.array([0.5, 1.0])

        @jax.jit
        def sample(key, h, z):
            return jax_polyagamma(h, z, key=key, method="callback")

        result = sample(key, h, z)
        assert jnp.all(result > 0)

    def test_callback_method_mean_matches_reference(self):
        """Mean of callback method should match polyagamma reference."""
        key = jax.random.PRNGKey(42)
        n_draws = 2000
        h_val = 5.0
        z_val = 1.0

        # Draw many samples with different keys
        jax_means = []
        for i in range(n_draws):
            key, subkey = jax.random.split(key)
            result = jax_polyagamma(
                jnp.array([h_val]),
                jnp.array([z_val]),
                key=subkey,
                method="callback",
            )
            jax_means.append(float(result[0]))

        # Reference: E[PG(h, z)] = h / (2|z|) * tanh(|z|/2) for z ≠ 0
        # E[PG(5, 1)] = 5 / 2 * tanh(0.5) = 1.1553
        expected_mean = h_val / (2 * abs(z_val)) * np.tanh(abs(z_val) / 2.0)

        jax_mean = np.mean(jax_means)
        # Should be within 10% of the exact mean (generous due to MC noise)
        assert abs(jax_mean - expected_mean) / expected_mean < 0.10

    def test_callback_method_non_integer_h(self):
        """Callback method should work with non-integer h (NB case)."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([5.06, 10.5, 0.06])  # Non-integer h values
        z = jnp.array([0.5, 1.0, 1.5])
        result = jax_polyagamma(h, z, key=key, method="callback")
        assert result.shape == (3,)
        assert jnp.all(result > 0)

    def test_callback_default_method(self):
        """Callback should be the default method."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0])
        z = jnp.array([0.5, 1.0])
        # Default (no method arg) should work and produce positive draws
        result = jax_polyagamma(h, z, key=key)
        assert result.shape == (2,)
        assert jnp.all(result > 0)


# ---------------------------------------------------------------------------
# Comparison with polyagamma reference (if available)
# ---------------------------------------------------------------------------


@needs_polyagamma
class TestPolyaGammaReferenceComparison:
    """Compare JAX PG sampler against the polyagamma reference package."""

    def test_exp_method_mean_matches_reference(self):
        """Mean of exp method should be within 5% of polyagamma reference."""
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        h_val = 1.0
        z_val = 1.0

        # JAX draws
        keys = jax.random.split(key, n_draws)
        jax_draws = jax.vmap(
            lambda k: jax_polyagamma(h_val, z_val, key=k, method="exp")
        )(keys)

        # Reference draws (vectorized)
        ref_draws = pg_ref_draw(1, z_val, size=n_draws)

        jax_mean = float(jnp.mean(jax_draws))
        ref_mean = float(np.mean(ref_draws))
        rel_error = abs(jax_mean - ref_mean) / ref_mean
        assert rel_error < 0.05, (
            f"JAX mean={jax_mean:.4f}, ref mean={ref_mean:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    def test_callback_method_mean_matches_reference(self):
        """Mean of callback method should match polyagamma reference."""
        key = jax.random.PRNGKey(42)
        n_draws = 2000
        h_val = 5.0
        z_val = 1.0

        # JAX callback draws
        jax_means = []
        for i in range(n_draws):
            key, subkey = jax.random.split(key)
            result = jax_polyagamma(
                jnp.array([h_val]),
                jnp.array([z_val]),
                key=subkey,
                method="callback",
            )
            jax_means.append(float(result[0]))

        # Reference draws
        ref_draws = pg_ref_draw(h_val, z_val, size=n_draws)

        jax_mean = np.mean(jax_means)
        ref_mean = float(np.mean(ref_draws))
        rel_error = abs(jax_mean - ref_mean) / ref_mean
        assert rel_error < 0.10, (
            f"JAX mean={jax_mean:.4f}, ref mean={ref_mean:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    def test_exp_method_vectorized_matches_reference(self):
        """Vectorized JAX draws should match reference for multiple (h, z)."""
        key = jax.random.PRNGKey(42)
        n_draws = 3000
        h_vals = [1.0, 3.0, 10.0]
        z_vals = [0.5, 1.0, 2.0]

        for h_val, z_val in zip(h_vals, z_vals):
            keys = jax.random.split(key, n_draws)
            jax_draws = jax.vmap(
                lambda k: jax_polyagamma(h_val, z_val, key=k, method="exp")
            )(keys)

            # Reference draws (vectorized)
            ref_draws = pg_ref_draw(h_val, z_val, size=n_draws)

            jax_mean = float(jnp.mean(jax_draws))
            ref_mean = float(np.mean(ref_draws))
            rel_error = abs(jax_mean - ref_mean) / ref_mean
            assert rel_error < 0.10, (
                f"h={h_val}, z={z_val}: JAX mean={jax_mean:.4f}, "
                f"ref mean={ref_mean:.4f}, rel_error={rel_error:.4f}"
            )
            key = jax.random.split(key, 1)[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_z_zero(self):
        """PG(h, 0) should work (z=0 is a valid input)."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 5.0, 10.0])
        z = jnp.zeros(3)
        result = jax_polyagamma(h, z, key=key, method="exp")
        assert jnp.all(result > 0)
        assert jnp.all(jnp.isfinite(result))

    def test_large_z(self):
        """PG(h, z) should work for large |z| (heavy tilting)."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 5.0])
        z = jnp.array([10.0, -10.0])
        result = jax_polyagamma(h, z, key=key, method="exp")
        assert jnp.all(result > 0)
        assert jnp.all(jnp.isfinite(result))
        # PG(h, z) should be small for large |z|
        assert jnp.all(result < 1.0), f"PG should be small for large |z|: {result}"

    def test_large_h(self):
        """PG(h, z) should work for large h."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([50.0, 100.0])
        z = jnp.array([1.0, 2.0])
        result = jax_polyagamma(h, z, key=key, method="exp")
        assert jnp.all(result > 0)
        assert jnp.all(jnp.isfinite(result))

    def test_single_observation(self):
        """Should work for a single observation (n=1)."""
        key = jax.random.PRNGKey(42)
        result = jax_polyagamma(
            jnp.array([1.0]), jnp.array([0.5]), key=key, method="exp"
        )
        assert result.shape == (1,)
        assert result[0] > 0

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        key = jax.random.PRNGKey(42)
        with pytest.raises(ValueError, match="Unknown method"):
            jax_polyagamma(1.0, 0.5, key=key, method="invalid")

    def test_exp_vs_callback_consistency(self):
        """Exp and callback methods should give similar means for h=1."""
        key = jax.random.PRNGKey(42)
        n_draws = 3000
        h_val = 1.0
        z_val = 1.0

        # Exp method
        keys = jax.random.split(key, n_draws)
        exp_draws = jax.vmap(
            lambda k: jax_polyagamma(h_val, z_val, key=k, method="exp")
        )(keys)

        # Callback method
        cb_means = []
        key = jax.random.PRNGKey(123)
        for i in range(n_draws):
            key, subkey = jax.random.split(key)
            result = jax_polyagamma(
                jnp.array([h_val]),
                jnp.array([z_val]),
                key=subkey,
                method="callback",
            )
            cb_means.append(float(result[0]))

        # For h=1, both methods should give similar means
        exp_mean = float(jnp.mean(exp_draws))
        cb_mean = np.mean(cb_means)
        rel_diff = abs(exp_mean - cb_mean) / cb_mean
        assert rel_diff < 0.10, (
            f"Exp vs callback mean differ: {exp_mean:.4f} vs {cb_mean:.4f}, "
            f"rel_diff={rel_diff:.4f}"
        )
