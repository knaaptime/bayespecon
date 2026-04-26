"""Tests for MATLAB-style log-determinant helper implementations."""

from __future__ import annotations

import numpy as np

from bayespecon.logdet import sparse_grid, ilu, spline, mc, chebyshev, make_logdet_fn


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


def test_lndetfull_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = sparse_grid(W, -0.5, 0.5, grid=0.25)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_lndetint_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = spline(W, 1e-5, 0.5, n_grid=30)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_lndetmc_returns_confidence_bounds() -> None:
    W = _toy_w()
    out = mc(order=12, iter=32, W=W, rmin=1e-5, rmax=0.5, grid=0.05, random_state=7)

    assert set(out.keys()) == {"rho", "lndet", "up95", "lo95"}
    assert out["rho"].shape == out["lndet"].shape == out["up95"].shape == out["lo95"].shape
    assert np.all(out["up95"] >= out["lo95"])


def test_lndetichol_returns_expected_keys_and_lengths() -> None:
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


def test_lndetmc_tracks_exact_trend_for_symmetric_row_standardized_w() -> None:
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
    from bayespecon.logdet import logdet_chebyshev
    import pytensor
    import pytensor.tensor as pt
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
