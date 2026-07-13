"""Bias acceptance test for the scan-compatible Gamma-series PG draw (jax_dense path).

The truncated series PG(h, z) ≈ (1/2π²) Σ_{k<K} g_k / ((k+0.5)² + (z/2π)²),
g_k ~ Gamma(h, 1), under-estimates the mean by the (deterministic) tail of the
series — measured at −0.8% to −1.7% for K=25 before the tail-mean correction.
E[PG(h, z)] = h/(2z)·tanh(z/2) is exact, so the corrected draw must match it
to Monte-Carlo precision (~0.1–0.2% at 2×10⁵ draws; tolerance 0.4%).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")


def _exact_pg_mean(h: float, z: float) -> float:
    if z == 0.0:
        return h / 4.0
    return h * np.tanh(z / 2.0) / (2.0 * z)


@pytest.mark.parametrize("z", [0.0, 0.5, 1.5, 3.0])
def test_pg_gamma_series_mean_is_exact(z):
    import jax.numpy as jnp

    from bayespecon.samplers.negbin._jax import _pg_gamma_series_draw

    jax.config.update("jax_enable_x64", True)

    h_val = 5.37  # non-integer, NB-like (h = y + alpha)
    m = 200_000
    h = jnp.full(m, h_val, dtype=jnp.float64)
    z_arr = jnp.full(m, z, dtype=jnp.float64)

    omega = _pg_gamma_series_draw(jax.random.PRNGKey(0), h, z_arr, n_terms=25)
    omega = np.asarray(omega)

    expected = _exact_pg_mean(h_val, z)
    rel_err = abs(omega.mean() - expected) / expected
    assert rel_err < 4e-3, (
        f"PG mean bias {rel_err:.2%} at h={h_val}, z={z} "
        f"(sample {omega.mean():.5f} vs exact {expected:.5f})"
    )
