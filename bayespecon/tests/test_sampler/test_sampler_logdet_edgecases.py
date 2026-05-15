"""Fast unit tests for bayespecon.models._sampler and bayespecon.logdet edge cases.

Tests enforce_c_backend, prepare_compile_kwargs, prepare_idata_kwargs,
logdet_exact, _build_logdet_grid large-matrix path, and _stable_rho_grid
edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon.logdet import (
    _build_logdet_grid,
    _stable_rho_grid,
    logdet_exact,
)
from bayespecon.models._sampler import (
    enforce_c_backend,
    prepare_compile_kwargs,
    prepare_idata_kwargs,
)

# ---------------------------------------------------------------------------
# enforce_c_backend
# ---------------------------------------------------------------------------


class TestEnforceCBackend:
    """Tests for enforce_c_backend downgrade logic."""

    def test_no_c_backend_required_passthrough(self):
        assert (
            enforce_c_backend("blackjax", requires_c_backend=False, model_name="X")
            == "blackjax"
        )

    def test_pymc_passthrough(self):
        assert (
            enforce_c_backend("pymc", requires_c_backend=True, model_name="X") == "pymc"
        )

    def test_requires_c_backend_with_no_jax(self, monkeypatch):
        """When requires_c_backend=True and JAX dispatch unavailable, fall back to pymc."""
        from bayespecon.models import _sampler

        monkeypatch.setattr(_sampler, "_jax_dispatches_available", lambda: False)
        result = enforce_c_backend(
            "blackjax", requires_c_backend=True, model_name="TestModel"
        )
        assert result == "pymc"

    def test_requires_c_backend_with_jax_available(self, monkeypatch):
        """When JAX dispatch is available, the requested sampler is preserved."""
        from bayespecon.models import _sampler

        monkeypatch.setattr(_sampler, "_jax_dispatches_available", lambda: True)
        result = enforce_c_backend(
            "blackjax", requires_c_backend=True, model_name="TestModel"
        )
        assert result == "blackjax"


# ---------------------------------------------------------------------------
# prepare_idata_kwargs
# ---------------------------------------------------------------------------


class TestPrepareIdataKwargs:
    """Tests for prepare_idata_kwargs log_likelihood stripping."""

    def test_none_idata_kwargs(self):
        result = prepare_idata_kwargs(None, model=None, nuts_sampler="pymc")
        assert result == {}

    def test_no_log_likelihood(self):
        result = prepare_idata_kwargs({"chains": 2}, model=None, nuts_sampler="pymc")
        assert result == {"chains": 2}

    def test_pymc_sampler_keeps_log_likelihood(self):
        result = prepare_idata_kwargs(
            {"log_likelihood": True}, model=None, nuts_sampler="pymc"
        )
        assert "log_likelihood" in result

    def test_jax_sampler_with_observed_rvs_keeps_log_likelihood(self):
        """When model has observed_RVs, keep log_likelihood even for JAX."""

        class MockModel:
            observed_RVs = [1, 2, 3]

        result = prepare_idata_kwargs(
            {"log_likelihood": True}, model=MockModel(), nuts_sampler="blackjax"
        )
        assert "log_likelihood" in result

    def test_jax_sampler_potential_only_strips_log_likelihood(self):
        """When model has no observed_RVs (Potential-only), strip log_likelihood for JAX."""

        class MockModel:
            observed_RVs = []

        result = prepare_idata_kwargs(
            {"log_likelihood": True}, model=MockModel(), nuts_sampler="blackjax"
        )
        assert "log_likelihood" not in result

    def test_numpyro_potential_only_strips_log_likelihood(self):
        class MockModel:
            observed_RVs = []

        result = prepare_idata_kwargs(
            {"log_likelihood": True}, model=MockModel(), nuts_sampler="numpyro"
        )
        assert "log_likelihood" not in result


# ---------------------------------------------------------------------------
# prepare_compile_kwargs
# ---------------------------------------------------------------------------


class TestPrepareCompileKwargs:
    """Tests for prepare_compile_kwargs NUMBA injection."""

    def test_non_pymc_sampler_unchanged(self):
        result = prepare_compile_kwargs({"chains": 2}, nuts_sampler="blackjax")
        assert result == {"chains": 2}
        assert "compile_kwargs" not in result

    def test_existing_compile_kwargs_preserved(self):
        result = prepare_compile_kwargs(
            {"compile_kwargs": {"mode": "FAST_COMPILE"}},
            nuts_sampler="pymc",
        )
        assert result["compile_kwargs"] == {"mode": "FAST_COMPILE"}

    def test_empty_compile_kwargs_preserved(self):
        """Even empty dict is a caller override and should be preserved."""
        result = prepare_compile_kwargs(
            {"compile_kwargs": {}},
            nuts_sampler="pymc",
        )
        assert result["compile_kwargs"] == {}

    def test_none_sample_kwargs(self):
        result = prepare_compile_kwargs(None, nuts_sampler="pymc")
        # Either numba is available (compile_kwargs added) or not (unchanged)
        assert isinstance(result, dict)

    def test_numba_missing_warns(self, monkeypatch):
        from bayespecon.models import _sampler

        monkeypatch.setattr(_sampler, "_has_module", lambda name: False)
        with pytest.warns(UserWarning, match="numba is not installed"):
            prepare_compile_kwargs({}, nuts_sampler="pymc")


# ---------------------------------------------------------------------------
# _stable_rho_grid
# ---------------------------------------------------------------------------


class TestStableRhoGrid:
    """Tests for _stable_rho_grid edge cases."""

    def test_negative_grid_raises(self):
        with pytest.raises(ValueError, match="grid must be positive"):
            _stable_rho_grid(-0.5, 0.5, -0.1)

    def test_rmax_leq_rmin_raises(self):
        with pytest.raises(ValueError, match="rmax must be greater than rmin"):
            _stable_rho_grid(0.5, 0.5, 0.1)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            _stable_rho_grid(0.0, 1.0, 0.01, eps=-1)

    def test_narrow_interval_raises(self):
        """When eps is so large that hi <= lo, raise ValueError."""
        with pytest.raises(ValueError, match="rho interval too narrow"):
            _stable_rho_grid(0.0, 0.001, 0.01, eps=0.5)

    def test_normal_grid(self):
        result = _stable_rho_grid(-1.0, 1.0, 0.1)
        assert len(result) > 0
        assert result[0] > -1.0  # lo = rmin + eps
        # np.arange may overshoot hi slightly due to float arithmetic;
        # the grid spacing is 0.1 so overshoot is at most one grid step
        assert result[-1] <= 1.0 + 0.1


# ---------------------------------------------------------------------------
# logdet_exact
# ---------------------------------------------------------------------------


class TestLogdetExact:
    """Tests for logdet_exact (small matrix)."""

    def test_identity_matrix(self):
        """log|I - 0*I| = log|I| = 0."""
        import pytensor
        import pytensor.tensor as pt

        W = np.eye(3) * 0.5
        rho = pt.scalar("rho")
        expr = logdet_exact(rho, W)
        fn = pytensor.function([rho], expr)
        result = fn(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_zero_rho(self):
        """log|I - 0*W| = log|I| = 0 for any W."""
        import pytensor
        import pytensor.tensor as pt

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        rho = pt.scalar("rho")
        expr = logdet_exact(rho, W)
        fn = pytensor.function([rho], expr)
        result = fn(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _build_logdet_grid
# ---------------------------------------------------------------------------


class TestBuildLogdetGrid:
    """Tests for _build_logdet_grid."""

    def test_small_matrix_eigenvalue_path(self):
        """For small n, uses eigendecomposition path."""
        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        rho_grid, logdet_grid = _build_logdet_grid(W, 0.0, 0.99, n_grid=10)
        assert len(rho_grid) == 10
        assert len(logdet_grid) == 10
        # At rho≈0, log|I| ≈ 0
        assert logdet_grid[0] < 0.1  # Should be near 0

    def test_large_matrix_slogdet_path(self):
        """For n > _LOGDET_GRID_EIG_MAX, uses slogdet loop."""
        from bayespecon.logdet import _LOGDET_GRID_EIG_MAX

        n = _LOGDET_GRID_EIG_MAX + 1
        rng = np.random.default_rng(42)
        # Create a small random row-standardized W
        W_small = rng.random((n, n))
        W_small = W_small / W_small.sum(axis=1, keepdims=True)
        rho_grid, logdet_grid = _build_logdet_grid(W_small, 0.0, 0.99, n_grid=5)
        assert len(rho_grid) == 5
        assert len(logdet_grid) == 5
        # All values should be finite
        assert np.all(np.isfinite(logdet_grid))


# ---------------------------------------------------------------------------
# logdet.mc
# ---------------------------------------------------------------------------


class TestLogdetMC:
    """Tests for logdet.mc (Monte Carlo log-determinant)."""

    def test_basic(self):
        from bayespecon.logdet import mc

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        result = mc(
            order=5, iter=10, W=W, rmin=0.01, rmax=0.99, grid=0.1, random_state=42
        )
        assert "rho" in result
        assert "lndet" in result
        assert len(result["rho"]) == len(result["lndet"])

    def test_invalid_order_raises(self):
        from bayespecon.logdet import mc

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="order must be positive"):
            mc(order=0, iter=10, W=W)

    def test_invalid_iter_raises(self):
        from bayespecon.logdet import mc

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="iter must be positive"):
            mc(order=5, iter=0, W=W)

    def test_negative_rmin_raises(self):
        from bayespecon.logdet import mc

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="nonnegative"):
            mc(order=5, iter=10, W=W, rmin=-0.5)

    def test_rmax_leq_rmin_raises(self):
        from bayespecon.logdet import mc

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="rmax must be greater than rmin"):
            mc(order=5, iter=10, W=W, rmin=0.5, rmax=0.5)


# ---------------------------------------------------------------------------
# logdet.spline
# ---------------------------------------------------------------------------


class TestLogdetSpline:
    """Tests for logdet.spline (LU interpolation style)."""

    def test_basic(self):
        from bayespecon.logdet import spline

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        result = spline(W, rmin=0.01, rmax=0.99, n_grid=50)
        assert "rho" in result
        assert "lndet" in result
        assert len(result["rho"]) == len(result["lndet"])

    def test_small_n_grid_raises(self):
        from bayespecon.logdet import spline

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="n_grid must be at least 20"):
            spline(W, n_grid=10)

    def test_negative_rmin_raises(self):
        from bayespecon.logdet import spline

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="nonnegative"):
            spline(W, rmin=-0.5)

    def test_rmax_leq_rmin_raises(self):
        from bayespecon.logdet import spline

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="rmax must be greater than rmin"):
            spline(W, rmin=0.5, rmax=0.5)


# ---------------------------------------------------------------------------
# logdet.sparse_grid
# ---------------------------------------------------------------------------


class TestLogdetSparseGrid:
    """Tests for logdet.sparse_grid."""

    def test_basic(self):
        from bayespecon.logdet import sparse_grid

        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        result = sparse_grid(W, lmin=0.01, lmax=0.99, grid=0.1)
        assert "rho" in result
        assert "lndet" in result
