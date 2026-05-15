"""Tests for legacy-style log-determinant helper implementations."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon.logdet import (
    chebyshev,
    clear_logdet_fn_cache,
    get_cached_logdet_fn,
    ilu,
    make_logdet_fn,
    mc,
    sparse_grid,
    spline,
)


def _toy_w() -> np.ndarray:
    # Small row-stochastic matrix with spectral radius <= 1.
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def test_sparse_grid_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = sparse_grid(W, -0.5, 0.5, grid=0.25)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_spline_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = spline(W, 1e-5, 0.5, n_grid=30)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_mc_returns_confidence_bounds() -> None:
    W = _toy_w()
    out = mc(order=12, iter=32, W=W, rmin=1e-5, rmax=0.5, grid=0.05, random_state=7)

    assert set(out.keys()) == {"rho", "lndet", "up95", "lo95"}
    assert (
        out["rho"].shape == out["lndet"].shape == out["up95"].shape == out["lo95"].shape
    )
    assert np.all(out["up95"] >= out["lo95"])


def test_ilu_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = ilu(W, -0.5, 0.5, grid=0.25)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_make_logdet_fn_accepts_new_method_names() -> None:
    W = _toy_w()

    for method in ("sparse_grid", "spline", "mc", "ilu"):
        fn = make_logdet_fn(W, method=method, rho_min=1e-5, rho_max=0.5)
        assert callable(fn)


def test_logdet_grids_match_direct_slogdet_for_full_int_ichol() -> None:
    W = _toy_w()
    I = np.eye(W.shape[0])
    rmin, rmax = 1e-5, 0.5

    out_full = sparse_grid(W, rmin, rmax, grid=0.05)
    out_int = spline(W, rmin=rmin, rmax=rmax, n_grid=80)
    out_ichol = ilu(W, rmin, rmax, grid=0.05)

    exact_full = np.array([np.linalg.slogdet(I - r * W)[1] for r in out_full["rho"]])
    exact_int = np.array([np.linalg.slogdet(I - r * W)[1] for r in out_int["rho"]])
    exact_ichol = np.array([np.linalg.slogdet(I - r * W)[1] for r in out_ichol["rho"]])

    assert np.max(np.abs(out_full["lndet"] - exact_full)) < 1e-10
    assert np.max(np.abs(out_int["lndet"] - exact_int)) < 5e-2
    assert np.max(np.abs(out_ichol["lndet"] - exact_ichol)) < 1e-8


def test_mc_tracks_exact_trend_for_symmetric_row_standardized_w() -> None:
    n = 20
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, (i - 1) % n] = 0.5
        W[i, (i + 1) % n] = 0.5

    out = mc(order=30, iter=120, W=W, rmin=1e-5, rmax=0.8, grid=0.05, random_state=0)
    I = np.eye(n)
    exact = np.array([np.linalg.slogdet(I - r * W)[1] for r in out["rho"]])

    # MC approximation should stay close to exact values on this stable toy case.
    assert np.mean(np.abs(out["lndet"] - exact)) < 0.2


def test_chebyshev_returns_expected_keys_and_values() -> None:
    W = _toy_w()
    # Use order=2 (< n=3) so eigenvalue path is used
    out = chebyshev(W, order=2, rmin=-0.5, rmax=0.5)

    assert set(out.keys()) == {"coeffs", "rmin", "rmax", "order", "method"}
    assert out["order"] == 2
    assert out["rmin"] == -0.5
    assert out["rmax"] == 0.5
    assert out["method"] == "eigenvalue"
    assert out["coeffs"].shape == (2,)
    assert np.all(np.isfinite(out["coeffs"]))


def test_chebyshev_accuracy_against_exact() -> None:
    W = _toy_w()
    I = np.eye(W.shape[0])
    out = chebyshev(W, order=20, rmin=-0.5, rmax=0.5)
    coeffs = out["coeffs"]
    rmin, rmax = out["rmin"], out["rmax"]

    # Evaluate Chebyshev approximation at several rho values
    import pytensor
    import pytensor.tensor as pt

    from bayespecon.logdet import logdet_chebyshev

    rho_sym = pt.scalar("rho")
    expr = logdet_chebyshev(rho_sym, coeffs, rmin=rmin, rmax=rmax)
    fn = pytensor.function([rho_sym], expr)

    test_rhos = np.linspace(-0.4, 0.4, 9)
    for rho in test_rhos:
        approx = float(fn(rho))
        exact = np.linalg.slogdet(I - rho * W)[1]
        # Chebyshev with order=20 should be very accurate for this small matrix
        assert abs(approx - exact) < 0.05, f"rho={rho}: approx={approx}, exact={exact}"


def test_make_logdet_fn_chebyshev() -> None:
    W = _toy_w()
    fn = make_logdet_fn(W, method="chebyshev", rho_min=-0.5, rho_max=0.5)
    assert callable(fn)

    import pytensor
    import pytensor.tensor as pt

    rho_sym = pt.scalar("rho")
    expr = fn(rho_sym)
    compiled = pytensor.function([rho_sym], expr)

    I = np.eye(W.shape[0])
    for rho in [-0.3, 0.0, 0.3]:
        approx = float(compiled(rho))
        exact = np.linalg.slogdet(I - rho * W)[1]
        assert abs(approx - exact) < 0.05


def test_get_cached_logdet_fn_reuses_callable_for_same_key() -> None:
    W = _toy_w()
    clear_logdet_fn_cache()

    fn1 = get_cached_logdet_fn(W, method="chebyshev", rho_min=-0.5, rho_max=0.5)
    fn2 = get_cached_logdet_fn(W, method="chebyshev", rho_min=-0.5, rho_max=0.5)

    assert fn1 is fn2


def test_get_cached_logdet_fn_separates_different_bounds() -> None:
    W = _toy_w()
    clear_logdet_fn_cache()

    fn1 = get_cached_logdet_fn(W, method="chebyshev", rho_min=-0.5, rho_max=0.5)
    fn2 = get_cached_logdet_fn(W, method="chebyshev", rho_min=-0.3, rho_max=0.3)

    assert fn1 is not fn2


# ---------------------------------------------------------------------------
# logdet_mc_poly_pytensor
# ---------------------------------------------------------------------------


def test_logdet_mc_poly_pytensor_matches_eigenvalue() -> None:
    """mc_poly polynomial agrees with exact eigenvalue log-det within 2%."""
    import pytensor
    import pytensor.tensor as pt

    from bayespecon.logdet import (
        compute_flow_traces,
        logdet_eigenvalue,
        logdet_mc_poly_pytensor,
    )

    rng = np.random.default_rng(42)
    n = 10
    W_dense = rng.random((n, n))
    W_dense /= W_dense.sum(axis=1, keepdims=True)
    import scipy.sparse as sp

    W_sp = sp.csr_matrix(W_dense)

    traces = compute_flow_traces(W_sp, miter=50, riter=100, random_state=0)
    eigs = np.linalg.eigvals(W_dense).real

    rho_sym = pt.dscalar("rho")
    mc_fn = pytensor.function([rho_sym], logdet_mc_poly_pytensor(rho_sym, traces))
    eig_fn = pytensor.function([rho_sym], logdet_eigenvalue(rho_sym, eigs))

    for rho in [0.05, 0.2, 0.4, 0.6]:
        mc_val = float(mc_fn(rho))
        eig_val = float(eig_fn(rho))
        rel_err = abs(mc_val - eig_val) / (abs(eig_val) + 1e-12)
        assert rel_err < 0.02, (
            f"rho={rho}: mc_poly={mc_val:.6f}, exact={eig_val:.6f}, rel_err={rel_err:.4f}"
        )


def test_logdet_mc_poly_pytensor_empty_traces() -> None:
    """Empty trace array returns zero."""
    import pytensor
    import pytensor.tensor as pt

    from bayespecon.logdet import logdet_mc_poly_pytensor

    rho_sym = pt.dscalar("rho")
    expr = logdet_mc_poly_pytensor(rho_sym, np.array([]))
    val = pytensor.function([rho_sym], expr)(0.3)
    assert float(val) == 0.0


def test_make_logdet_fn_mc_poly() -> None:
    """make_logdet_fn with method='mc_poly' produces a valid callable."""
    import pytensor
    import pytensor.tensor as pt

    W = _toy_w()
    fn = make_logdet_fn(W, method="mc_poly")
    assert callable(fn)

    rho_sym = pt.dscalar("rho")
    compiled = pytensor.function([rho_sym], fn(rho_sym))

    I = np.eye(W.shape[0])
    for rho in [0.1, 0.3, 0.5]:
        approx = float(compiled(rho))
        exact = np.linalg.slogdet(I - rho * W)[1]
        assert abs(approx - exact) < 0.1, (
            f"rho={rho}: mc_poly={approx:.4f}, exact={exact:.4f}"
        )


# ---------------------------------------------------------------------------
# make_flow_separable_logdet
# ---------------------------------------------------------------------------


class TestMakeFlowSeparableLogdet:
    """make_flow_separable_logdet returns fn(rho_d, rho_o) = n*f(rho_d) + n*f(rho_o)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import pytensor
        import pytensor.tensor as pt
        import scipy.sparse as sp

        self.pytensor = pytensor
        self.pt = pt

        rng = np.random.default_rng(7)
        n = 8
        W_dense = rng.random((n, n))
        W_dense /= W_dense.sum(axis=1, keepdims=True)
        self.W = W_dense
        self.W_sp = sp.csr_matrix(W_dense)
        self.n = n
        eigs = np.linalg.eigvals(W_dense).real
        self.rho_d, self.rho_o = 0.3, 0.25

        # Reference: exact eigenvalue-based answer
        rho_d_t = pt.dscalar("rd")
        rho_o_t = pt.dscalar("ro")
        from bayespecon.logdet import logdet_eigenvalue

        ref_expr = n * logdet_eigenvalue(rho_d_t, eigs) + n * logdet_eigenvalue(
            rho_o_t, eigs
        )
        self.ref_fn = pytensor.function([rho_d_t, rho_o_t], ref_expr)
        self.ref_val = self.ref_fn(self.rho_d, self.rho_o)

    def _compile(self, fn):
        rho_d_t = self.pt.dscalar("rd")
        rho_o_t = self.pt.dscalar("ro")
        return self.pytensor.function([rho_d_t, rho_o_t], fn(rho_d_t, rho_o_t))

    def test_eigenvalue_method(self):
        from bayespecon.logdet import make_flow_separable_logdet

        fn = make_flow_separable_logdet(self.W_sp, self.n, method="eigenvalue")
        compiled = self._compile(fn)
        val = float(compiled(self.rho_d, self.rho_o))
        assert abs(val - self.ref_val) < 1e-8

    def test_chebyshev_method(self):
        from bayespecon.logdet import make_flow_separable_logdet

        fn = make_flow_separable_logdet(
            self.W_sp, self.n, method="chebyshev", cheb_order=25
        )
        compiled = self._compile(fn)
        val = float(compiled(self.rho_d, self.rho_o))
        assert abs(val - self.ref_val) / (abs(self.ref_val) + 1e-12) < 0.02

    def test_mc_poly_method(self):
        from bayespecon.logdet import make_flow_separable_logdet

        fn = make_flow_separable_logdet(
            self.W_sp, self.n, method="mc_poly", miter=50, riter=100, random_state=0
        )
        compiled = self._compile(fn)
        val = float(compiled(self.rho_d, self.rho_o))
        assert abs(val - self.ref_val) / (abs(self.ref_val) + 1e-12) < 0.02

    def test_invalid_method_raises(self):
        from bayespecon.logdet import make_flow_separable_logdet

        with pytest.raises(ValueError, match="not recognised"):
            make_flow_separable_logdet(self.W_sp, self.n, method="spline")
