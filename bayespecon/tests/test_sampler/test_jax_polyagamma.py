"""Unit tests for JAX-accelerated Pólya–Gamma samplers.

Tests the sum-of-exponentials method (``jax_polyagamma``) and the
Normal approximation (``jax_polyagamma_normal``) against the reference
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

from bayespecon._samplers._jax_polyagamma import (
    jax_polyagamma,
    jax_polyagamma_normal,
)

# Try to import polyagamma for reference comparisons
try:
    from polyagamma import polyagamma as pg_ref

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
    return np.sum(1.0 / denom) / (2 * np.pi ** 2)


def _pg1_var(z: float, n_terms: int = 10000) -> float:
    """Analytical variance of PG(1, z) via truncated series."""
    k = np.arange(n_terms, dtype=np.float64)
    denom = (k + 0.5) ** 2 + (z / (2 * np.pi)) ** 2
    return np.sum(1.0 / denom ** 2) / (2 * np.pi ** 2) ** 2


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
        result = jax_polyagamma(h, z, key=key, n_terms=20)
        assert result.shape == (3,)

    def test_output_shape_scalar(self):
        """Output should be scalar for scalar inputs."""
        key = jax.random.PRNGKey(42)
        result = jax_polyagamma(1.0, 0.5, key=key, n_terms=20)
        assert result.shape == ()

    def test_positivity(self):
        """All PG draws should be positive."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 2.0, 5.0, 10.0, 50.0])
        z = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
        result = jax_polyagamma(h, z, key=key, n_terms=20)
        assert jnp.all(result > 0), f"Non-positive draws: {result}"

    def test_mean_approximately_correct(self):
        """Mean of many draws should be within ~5% of analytical mean.

        Uses n_terms=20 which gives ~2% bias, so 5% tolerance is generous.
        """
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        h_val = 1.0
        z_val = 1.0

        # Draw many samples
        keys = jax.random.split(key, n_draws)
        h_arr = jnp.full(n_draws, h_val)
        z_arr = jnp.full(n_draws, z_val)
        draws = jax.vmap(lambda k: jax_polyagamma(h_val, z_val, key=k, n_terms=20))(
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
                lambda k: jax_polyagamma(h_val, z_val, key=k, n_terms=20)
            )(keys)
            means[h_val] = float(jnp.mean(draws))
            key = jax.random.split(key, 1)[0]

        # PG(3,z) mean should be ~3x PG(1,z) mean
        ratio_3 = means[3.0] / means[1.0]
        assert abs(ratio_3 - 3.0) < 0.5, f"PG(3,z)/PG(1,z) = {ratio_3:.2f}, expected ~3"

        # PG(10,z) mean should be ~10x PG(1,z) mean
        ratio_10 = means[10.0] / means[1.0]
        assert abs(ratio_10 - 10.0) < 2.0, f"PG(10,z)/PG(1,z) = {ratio_10:.2f}, expected ~10"

    def test_z_zero_larger_than_z_nonzero(self):
        """PG(h, 0) should be larger than PG(h, z) for z > 0."""
        key = jax.random.PRNGKey(42)
        n_draws = 3000
        h_val = 5.0

        keys = jax.random.split(key, n_draws)
        draws_z0 = jax.vmap(
            lambda k: jax_polyagamma(h_val, 0.0, key=k, n_terms=20)
        )(keys)

        key = jax.random.split(key, 1)[0]
        keys = jax.random.split(key, n_draws)
        draws_z2 = jax.vmap(
            lambda k: jax_polyagamma(h_val, 2.0, key=k, n_terms=20)
        )(keys)

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
            return jax_polyagamma(h, z, key=key, n_terms=20)

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
            omega = jax_polyagamma(h, z, key=subkey, n_terms=20)
            # Simulate a simple Gibbs update: omega_new = 0.5 * (omega + omega_prev)
            omega_new = 0.5 * (omega + omega_prev)
            return key, omega_new

        omega = jnp.ones_like(h)
        key, omega = gibbs_step(key, h, z, omega)
        assert omega.shape == (2,)
        assert jnp.all(omega > 0)

    def test_n_terms_affects_accuracy(self):
        """More terms should give more accurate means."""
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        z_val = 1.0
        analytical = _pg1_mean(z_val)

        errors = {}
        for n_terms in [5, 20, 200]:
            keys = jax.random.split(key, n_draws)
            draws = jax.vmap(
                lambda k: jax_polyagamma(1.0, z_val, key=k, n_terms=n_terms)
            )(keys)
            empirical = float(jnp.mean(draws))
            errors[n_terms] = abs(empirical - analytical) / analytical
            key = jax.random.split(key, 1)[0]

        # More terms should give smaller (or equal) bias
        # Note: stochastic, so we just check the general trend
        assert errors[200] < 0.01, f"n_terms=200 error {errors[200]:.4f} > 1%"
        assert errors[20] < 0.05, f"n_terms=20 error {errors[20]:.4f} > 5%"

    def test_batch_sampling(self):
        """Vectorized sampling over observations should work correctly."""
        key = jax.random.PRNGKey(42)
        n = 100
        h = jnp.ones(n)
        z = jnp.linspace(-2, 2, n)

        result = jax_polyagamma(h, z, key=key, n_terms=20)
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
            lambda k: jax_polyagamma(h_val, z_val, key=k, n_terms=20)
        )(keys)

        key = jax.random.split(key, 1)[0]
        keys = jax.random.split(key, n_draws)
        draws_neg = jax.vmap(
            lambda k: jax_polyagamma(h_val, -z_val, key=k, n_terms=20)
        )(keys)

        # Means should be approximately equal (PG is symmetric in z)
        mean_pos = float(jnp.mean(draws_pos))
        mean_neg = float(jnp.mean(draws_neg))
        assert abs(mean_pos - mean_neg) / mean_pos < 0.1, (
            f"PG(h,z) != PG(h,-z): {mean_pos:.4f} vs {mean_neg:.4f}"
        )


# ---------------------------------------------------------------------------
# jax_polyagamma tests — gamma method
# ---------------------------------------------------------------------------

class TestJaxPolyaGammaGamma:
    """Tests for jax_polyagamma with method='gamma' (sum-of-gammas)."""

    def test_gamma_method_basic(self):
        """Gamma method should produce positive draws."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0, 10.0])
        z = jnp.array([0.5, 1.0, 2.0])
        result = jax_polyagamma(h, z, key=key, n_terms=20, method="gamma")
        assert result.shape == (3,)
        assert jnp.all(result > 0)

    def test_gamma_method_jit(self):
        """Gamma method should be JIT-compatible."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 3.0])
        z = jnp.array([0.5, 1.0])

        @jax.jit
        def sample(key, h, z):
            return jax_polyagamma(h, z, key=key, n_terms=20, method="gamma")

        result = sample(key, h, z)
        assert jnp.all(result > 0)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        key = jax.random.PRNGKey(42)
        with pytest.raises(ValueError, match="Unknown method"):
            jax_polyagamma(1.0, 0.5, key=key, method="invalid")


# ---------------------------------------------------------------------------
# jax_polyagamma_normal tests
# ---------------------------------------------------------------------------

class TestJaxPolyaGammaNormal:
    """Tests for jax_polyagamma_normal (Normal approximation)."""

    def test_output_shape_vector(self):
        """Output shape should match input shape for vector inputs."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([5.0, 10.0, 50.0])
        z = jnp.array([0.5, 1.0, 2.0])
        result = jax_polyagamma_normal(h, z, key=key)
        assert result.shape == (3,)

    def test_output_shape_scalar(self):
        """Output should be scalar for scalar inputs."""
        key = jax.random.PRNGKey(42)
        result = jax_polyagamma_normal(5.0, 0.5, key=key)
        assert result.shape == ()

    def test_positivity(self):
        """Normal approximation draws should be positive (clamped)."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([5.0, 10.0, 50.0])
        z = jnp.array([0.0, 0.5, 2.0])
        result = jax_polyagamma_normal(h, z, key=key)
        assert jnp.all(result > 0), f"Non-positive draws: {result}"

    def test_mean_approximately_correct(self):
        """Mean of Normal draws should match analytical mean within 5%."""
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        h_val = 10.0
        z_val = 1.0

        keys = jax.random.split(key, n_draws)
        draws = jax.vmap(
            lambda k: jax_polyagamma_normal(h_val, z_val, key=k)
        )(keys)

        empirical_mean = float(jnp.mean(draws))
        analytical_mean = _pg1_mean(z_val) * h_val
        rel_error = abs(empirical_mean - analytical_mean) / analytical_mean
        assert rel_error < 0.05, (
            f"Mean error too large: empirical={empirical_mean:.4f}, "
            f"analytical={analytical_mean:.4f}, rel_error={rel_error:.4f}"
        )

    def test_variance_approximately_correct(self):
        """Variance of Normal draws should match analytical variance within 10%."""
        key = jax.random.PRNGKey(42)
        n_draws = 10000
        h_val = 10.0
        z_val = 1.0

        keys = jax.random.split(key, n_draws)
        draws = jax.vmap(
            lambda k: jax_polyagamma_normal(h_val, z_val, key=k)
        )(keys)

        empirical_var = float(jnp.var(draws))
        analytical_var = _pg1_var(z_val) * h_val
        rel_error = abs(empirical_var - analytical_var) / analytical_var
        assert rel_error < 0.10, (
            f"Variance error too large: empirical={empirical_var:.6f}, "
            f"analytical={analytical_var:.6f}, rel_error={rel_error:.4f}"
        )

    def test_scales_with_h(self):
        """Mean should scale linearly with h."""
        key = jax.random.PRNGKey(42)
        n_draws = 3000
        z_val = 1.0

        means = {}
        for h_val in [5.0, 10.0, 50.0]:
            keys = jax.random.split(key, n_draws)
            draws = jax.vmap(
                lambda k: jax_polyagamma_normal(h_val, z_val, key=k)
            )(keys)
            means[h_val] = float(jnp.mean(draws))
            key = jax.random.split(key, 1)[0]

        # PG(10,z) mean should be ~2x PG(5,z) mean
        ratio = means[10.0] / means[5.0]
        assert abs(ratio - 2.0) < 0.3, f"PG(10)/PG(5) = {ratio:.2f}, expected ~2"

    def test_jit_compatible(self):
        """jax_polyagamma_normal should be JIT-compilable."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([5.0, 10.0])
        z = jnp.array([0.5, 1.0])

        @jax.jit
        def sample(key, h, z):
            return jax_polyagamma_normal(h, z, key=key)

        result = sample(key, h, z)
        assert result.shape == (2,)
        assert jnp.all(result > 0)

    def test_z_zero_larger_than_z_nonzero(self):
        """PG(h, 0) should be larger than PG(h, z) for z > 0."""
        key = jax.random.PRNGKey(42)
        n_draws = 3000
        h_val = 10.0

        keys = jax.random.split(key, n_draws)
        draws_z0 = jax.vmap(
            lambda k: jax_polyagamma_normal(h_val, 0.0, key=k)
        )(keys)

        key = jax.random.split(key, 1)[0]
        keys = jax.random.split(key, n_draws)
        draws_z2 = jax.vmap(
            lambda k: jax_polyagamma_normal(h_val, 2.0, key=k)
        )(keys)

        mean_z0 = float(jnp.mean(draws_z0))
        mean_z2 = float(jnp.mean(draws_z2))
        assert mean_z0 > mean_z2, (
            f"PG({h_val},0) mean={mean_z0:.4f} should be > PG({h_val},2) mean={mean_z2:.4f}"
        )


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
            lambda k: jax_polyagamma(h_val, z_val, key=k, n_terms=20)
        )(keys)

        # Reference draws
        ref_draws = np.array([pg_ref(1, z_val) for _ in range(n_draws)])

        jax_mean = float(jnp.mean(jax_draws))
        ref_mean = float(np.mean(ref_draws))
        rel_error = abs(jax_mean - ref_mean) / ref_mean
        assert rel_error < 0.05, (
            f"JAX mean={jax_mean:.4f}, ref mean={ref_mean:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    def test_normal_approx_mean_matches_reference(self):
        """Mean of Normal approx should be within 5% of polyagamma reference."""
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        h_val = 10.0
        z_val = 1.0

        # JAX Normal draws
        keys = jax.random.split(key, n_draws)
        jax_draws = jax.vmap(
            lambda k: jax_polyagamma_normal(h_val, z_val, key=k)
        )(keys)

        # Reference draws
        ref_draws = np.array([pg_ref(h_val, z_val) for _ in range(n_draws)])

        jax_mean = float(jnp.mean(jax_draws))
        ref_mean = float(np.mean(ref_draws))
        rel_error = abs(jax_mean - ref_mean) / ref_mean
        assert rel_error < 0.05, (
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
                lambda k: jax_polyagamma(h_val, z_val, key=k, n_terms=20)
            )(keys)

            ref_draws = np.array([pg_ref(h_val, z_val) for _ in range(n_draws)])

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
        result = jax_polyagamma(h, z, key=key, n_terms=20)
        assert jnp.all(result > 0)
        assert jnp.all(jnp.isfinite(result))

    def test_large_z(self):
        """PG(h, z) should work for large |z| (heavy tilting)."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([1.0, 5.0])
        z = jnp.array([10.0, -10.0])
        result = jax_polyagamma(h, z, key=key, n_terms=20)
        assert jnp.all(result > 0)
        assert jnp.all(jnp.isfinite(result))
        # PG(h, z) should be small for large |z|
        assert jnp.all(result < 1.0), f"PG should be small for large |z|: {result}"

    def test_large_h(self):
        """PG(h, z) should work for large h."""
        key = jax.random.PRNGKey(42)
        h = jnp.array([50.0, 100.0])
        z = jnp.array([1.0, 2.0])
        result = jax_polyagamma(h, z, key=key, n_terms=20)
        assert jnp.all(result > 0)
        assert jnp.all(jnp.isfinite(result))

    def test_single_observation(self):
        """Should work for a single observation (n=1)."""
        key = jax.random.PRNGKey(42)
        result = jax_polyagamma(jnp.array([1.0]), jnp.array([0.5]), key=key, n_terms=20)
        assert result.shape == (1,)
        assert result[0] > 0

    def test_normal_approx_large_h(self):
        """Normal approximation should be accurate for large h (h >= 5)."""
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        h_val = 50.0
        z_val = 1.0

        keys = jax.random.split(key, n_draws)
        draws = jax.vmap(
            lambda k: jax_polyagamma_normal(h_val, z_val, key=k)
        )(keys)

        analytical_mean = _pg1_mean(z_val) * h_val
        empirical_mean = float(jnp.mean(draws))
        rel_error = abs(empirical_mean - analytical_mean) / analytical_mean
        assert rel_error < 0.03, f"Normal approx error for h=50: {rel_error:.4f}"

    def test_exp_vs_gamma_consistency(self):
        """Exp and gamma methods should give similar means for h=1."""
        key = jax.random.PRNGKey(42)
        n_draws = 5000
        h_val = 1.0
        z_val = 1.0

        # Exp method
        keys = jax.random.split(key, n_draws)
        exp_draws = jax.vmap(
            lambda k: jax_polyagamma(h_val, z_val, key=k, n_terms=200, method="exp")
        )(keys)

        # Gamma method
        key = jax.random.split(key, 1)[0]
        keys = jax.random.split(key, n_draws)
        gamma_draws = jax.vmap(
            lambda k: jax_polyagamma(h_val, z_val, key=k, n_terms=200, method="gamma")
        )(keys)

        # For h=1, both methods should give similar means (both are exact for PG(1,z))
        exp_mean = float(jnp.mean(exp_draws))
        gamma_mean = float(jnp.mean(gamma_draws))
        rel_diff = abs(exp_mean - gamma_mean) / gamma_mean
        assert rel_diff < 0.05, (
            f"Exp vs gamma mean differ: {exp_mean:.4f} vs {gamma_mean:.4f}, "
            f"rel_diff={rel_diff:.4f}"
        )