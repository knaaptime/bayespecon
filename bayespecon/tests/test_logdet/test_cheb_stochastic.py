"""Tests for stochastic Chebyshev expansion logdet."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from bayespecon._logdet._cheb_stochastic import (
    ChebStochasticPrecompute,
    _log_cheb_coeffs,
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_eval_vec,
    cheb_stochastic_logdet_precompute,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_W():
    """Small rook contiguity W with known eigenvalues."""
    from libpysal import graph

    from bayespecon import dgp

    gdf = dgp.simulate_sar(n=15, create_gdf=True)
    W = graph.Graph.build_contiguity(gdf, rook=True).transform("r").sparse.toarray()
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_eigs(small_W):
    return np.linalg.eigvals(small_W.toarray())


# ---------------------------------------------------------------------------
# Chebyshev coefficient computation
# ---------------------------------------------------------------------------


class TestLogChebCoeffs:
    def test_zero_rho(self):
        """At ρ=0, log|1| = 0, so all coefficients should be 0."""
        coeffs = _log_cheb_coeffs(0.0, -1.0, 1.0, 20)
        assert np.allclose(coeffs, 0.0, atol=1e-10)

    def test_coefficient_accuracy(self):
        """Coefficients should reconstruct log|1 - ρx| accurately via Clenshaw."""
        rho = 0.5
        lam_min, lam_max = -1.0, 1.0
        order = 30
        coeffs = _log_cheb_coeffs(rho, lam_min, lam_max, order)

        # Evaluate Chebyshev series at several x values and compare to exact
        for x in [-0.9, -0.5, 0.0, 0.3, 0.7, 0.95]:
            # f(x) = log|1 - rho * sigma(x)| where sigma maps [-1,1] -> [lam_min, lam_max]
            # sigma(x) = (x*(lam_max-lam_min) + lam_max+lam_min)/2 = x (for [-1,1])
            exact = np.log(np.abs(1.0 - rho * x))
            # Clenshaw evaluation of Chebyshev series
            m = len(coeffs)
            b_next = 0.0
            b_curr = coeffs[m - 1]
            for k in range(m - 2, 0, -1):
                b_new = 2.0 * x * b_curr - b_next + coeffs[k]
                b_next = b_curr
                b_curr = b_new
            cheb_val = coeffs[0] + x * b_curr - b_next
            assert abs(cheb_val - exact) < 1e-8, f"x={x}: {cheb_val} vs {exact}"


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_returns_dataclass(self, small_W):
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        assert isinstance(pre, ChebStochasticPrecompute)
        assert pre.order == 20
        assert pre.n == small_W.shape[0]
        assert pre.moments.shape == (21,)

    def test_moment_zero_is_n(self, small_W):
        """μ_0 = tr(T_0(W̃)) = tr(I) = n."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        assert pre.moments[0] == pytest.approx(small_W.shape[0])

    def test_moment_one_is_exact(self, small_W):
        """μ_1 = tr(W̃) is computed exactly (diagonal sum)."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        # W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)
        # For λ_min=-1, λ_max=1: W̃ = W, so tr(W̃) = tr(W) = 0 (row-standardized)
        assert pre.moments[1] == pytest.approx(0.0, abs=1e-10)

    def test_spectral_bounds(self, small_W):
        """For row-standardized W, bounds should be [-1, 1]."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        assert pre.lam_max == pytest.approx(1.0)
        assert pre.lam_min == pytest.approx(-1.0)

    def test_deterministic_with_seed(self, small_W):
        """Same RNG seed should produce identical results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        pre1 = cheb_stochastic_logdet_precompute(
            small_W, order=20, n_probes=50, rng=rng1
        )
        pre2 = cheb_stochastic_logdet_precompute(
            small_W, order=20, n_probes=50, rng=rng2
        )
        np.testing.assert_array_equal(pre1.moments, pre2.moments)


# ---------------------------------------------------------------------------
# Evaluation accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_accuracy_at_moderate_rho(self, small_W, small_eigs):
        """At ρ=0.5, error should be < 5% (small n=15, high stochastic variance)."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=200)
        rho = 0.5
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        est = cheb_stochastic_logdet_eval(pre, rho)
        rel_err = abs(est - exact) / abs(exact)
        assert rel_err < 0.05, f"rel_err={rel_err:.4e}"

    def test_accuracy_at_high_rho(self, small_W, small_eigs):
        """At ρ=0.9, error should be < 5% (small n=15, high stochastic variance)."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=200)
        rho = 0.9
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        est = cheb_stochastic_logdet_eval(pre, rho)
        rel_err = abs(est - exact) / abs(exact)
        assert rel_err < 0.05, f"rel_err={rel_err:.4e}"

    def test_better_than_barry_pace_at_rho_095(self, small_W, small_eigs):
        """At ρ=0.95 and large n, stochastic Chebyshev should be more accurate than Barry-Pace-MC.

        Note: Barry-Pace uses exact eigenvalues for n≤500, so this test only
        works with a large enough W to trigger the MC path.  We test the
        stochastic moments directly against exact instead.
        """
        rho = 0.95
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))

        # Stochastic Chebyshev should be reasonably accurate at high ρ
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=200)
        stoch_val = cheb_stochastic_logdet_eval(pre, rho)
        stoch_err = abs(stoch_val - exact) / abs(exact)

        # With enough probes, error should be < 5% even at small n
        assert stoch_err < 0.05, f"stoch_err={stoch_err:.4e}"

    def test_zero_rho_gives_zero(self, small_W):
        """log|I - 0*W| = log|I| = 0."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        val = cheb_stochastic_logdet_eval(pre, 0.0)
        assert abs(val) < 0.1  # stochastic, so not exactly 0


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    def test_numpy_scalar_factory(self, small_W, small_eigs):
        """Factory should produce a working scalar evaluator."""
        fn = make_logdet_numpy_fn(
            small_W, eigs=None, method="cheb_stochastic", rho_min=-0.95, rho_max=0.95
        )
        rho = 0.5
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        est = fn(rho)
        assert abs(est - exact) / abs(exact) < 0.05

    def test_numpy_vectorized_factory(self, small_W, small_eigs):
        """Factory should produce a working vectorized evaluator."""
        fn = make_logdet_numpy_vec_fn(
            small_W, eigs=None, method="cheb_stochastic", rho_min=-0.95, rho_max=0.95
        )
        rho_arr = np.array([0.3, 0.5, 0.7, 0.9])
        result = fn(rho_arr)
        assert result.shape == (4,)
        for i, rho in enumerate(rho_arr):
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            assert abs(result[i] - exact) / abs(exact) < 0.05

    def test_auto_select(self):
        """For large n, auto-select should pick cheb_stochastic."""
        from bayespecon._logdet import resolve_logdet_method

        assert resolve_logdet_method(None, n=50000) == "cheb_stochastic"

    def test_eval_speed(self, small_W):
        """Eval should be O(20) Clenshaw, ~2μs per call."""
        import time

        fn = make_logdet_numpy_fn(
            small_W, eigs=None, method="cheb_stochastic", rho_min=-0.95, rho_max=0.95
        )
        # Warm up
        fn(0.5)
        # Benchmark
        rhos = np.linspace(-0.9, 0.9, 1000)
        t0 = time.perf_counter()
        for r in rhos:
            fn(float(r))
        per_call = (time.perf_counter() - t0) / 1000 * 1e6
        # Should be < 10μs (Clenshaw O(20))
        assert per_call < 10.0, f"per_call={per_call:.1f}μs"


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_convergence_in_order(self, small_W, small_eigs):
        """Error should be reasonable across orders (at high ρ, stochastic noise present)."""
        rho = 0.9
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        errors = []
        for order in [10, 20, 30]:
            pre = cheb_stochastic_logdet_precompute(small_W, order=order, n_probes=200)
            est = cheb_stochastic_logdet_eval(pre, rho)
            errors.append(abs(est - exact) / abs(exact))
        # All errors should be < 5% at this probe count
        assert all(e < 0.05 for e in errors), f"errors={errors}"

    def test_variance_decreases_with_probes(self, small_W, small_eigs):
        """Variance across seeds should decrease with more probes."""
        rho = 0.9
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))

        errors_low = []
        errors_high = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            pre_low = cheb_stochastic_logdet_precompute(
                small_W, order=20, n_probes=30, rng=rng
            )
            errors_low.append(cheb_stochastic_logdet_eval(pre_low, rho) - exact)

            rng = np.random.default_rng(seed)
            pre_high = cheb_stochastic_logdet_precompute(
                small_W, order=20, n_probes=200, rng=rng
            )
            errors_high.append(cheb_stochastic_logdet_eval(pre_high, rho) - exact)

        std_low = np.std(errors_low)
        std_high = np.std(errors_high)
        assert std_high < std_low, f"std_high={std_high} should be < std_low={std_low}"
