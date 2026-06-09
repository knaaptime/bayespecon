"""Fast build/method tests for the reduced-form SARNegativeBinomial.

These exercise the reduced-form (PG-Gibbs, no σ) model.  A separate,
slower recovery-style test lives in ``test_sar_negbin_reduced_recovery.py``
(gated by a marker if added).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

import bayespecon as bp
from bayespecon.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _count_data(seed: int = 101, n: int = 10):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.6 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    W = W_to_graph(make_line_W(n))
    return y, X, W


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_reduced_inherits_from_spatial_model():
    """The reduced-form class inherits from SpatialModel."""
    from bayespecon.models.base import SpatialModel

    assert issubclass(bp.SARNegativeBinomial, SpatialModel)


def test_reduced_rejects_noninteger_or_negative_y():
    _, _, _ = _count_data()
    y_bad = np.array([0.0, 1.2, 2.0, 1.0])
    X_bad = np.column_stack([np.ones(4), np.arange(4)])
    with pytest.raises(ValueError, match="integer-valued"):
        bp.SARNegativeBinomial(y=y_bad, X=X_bad, W=W_to_graph(make_line_W(4)))

    y_neg = np.array([0.0, 1.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        bp.SARNegativeBinomial(y=y_neg, X=X_bad, W=W_to_graph(make_line_W(4)))


def test_reduced_robust_is_unsupported():
    y, X, W = _count_data()
    with pytest.raises(NotImplementedError):
        bp.SARNegativeBinomial(y=y, X=X, W=W, robust=True)


def test_reduced_build_pymc_model_returns_valid_model():
    """The reduced-form model builds a valid PyMC model with Jacobian."""
    import pymc as pm

    y, X, W = _count_data()
    model = bp.SARNegativeBinomial(y=y, X=X, W=W)
    pymc_model = model._build_pymc_model()
    assert isinstance(pymc_model, pm.Model)
    assert "rho" in pymc_model.named_vars
    assert "beta" in pymc_model.named_vars
    assert "alpha" in pymc_model.named_vars
    assert "jacobian" in pymc_model.named_vars
    # Reduced form must NOT have sigma, sigma2, or z
    assert "sigma" not in pymc_model.named_vars
    assert "sigma2" not in pymc_model.named_vars
    assert "z" not in pymc_model.named_vars


# ---------------------------------------------------------------------------
# Post-fit machinery (mock posterior — no σ / z draws)
# ---------------------------------------------------------------------------


def test_reduced_fitted_values_and_effects_with_mock_posterior():
    y, X, W = _count_data(seed=103)
    model = bp.SARNegativeBinomial(y=y, X=X, W=W)

    # Reduced form: only β, ρ, α — no σ, no z.
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    fitted = model.fitted_values()
    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(fitted > 0)

    effects = model.spatial_effects()
    assert "direct" in effects.columns
    assert np.all(np.isfinite(effects["direct"].values))


# ---------------------------------------------------------------------------
# End-to-end Gibbs fit (small, fast)
# ---------------------------------------------------------------------------


def test_reduced_fit_returns_inference_data():
    """A short Gibbs run should return a well-formed InferenceData."""
    rng = np.random.default_rng(0)
    n = 60
    W = W_to_graph(make_line_W(n))
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.5 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu)

    model = bp.SARNegativeBinomial(y=y, X=X, W=W)
    idata = model.fit(draws=30, tune=30, chains=2, random_seed=0)

    assert isinstance(idata, az.InferenceData)
    assert "posterior" in idata.groups()
    assert "log_likelihood" in idata.groups()
    assert "observed_data" in idata.groups()
    # Reduced-form posterior must NOT contain σ or z.
    assert "sigma" not in idata.posterior.data_vars
    assert "z" not in idata.posterior.data_vars
    # Required parameters present with correct shapes.
    assert idata.posterior["rho"].shape == (2, 30)
    assert idata.posterior["alpha"].shape == (2, 30)
    assert idata.posterior["beta"].shape == (2, 30, 2)
    assert idata.log_likelihood["obs"].shape == (2, 30, n)


def test_reduced_fit_default_is_gibbs():
    """Default sampler='gibbs' should produce the same result as explicit."""
    rng = np.random.default_rng(42)
    n = 30
    W = W_to_graph(make_line_W(n))
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.5 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu)

    model = bp.SARNegativeBinomial(y=y, X=X, W=W)
    # Default call (no sampler kwarg) should use Gibbs
    idata = model.fit(draws=10, tune=10, chains=1, random_seed=0)
    assert isinstance(idata, az.InferenceData)
    assert "rho" in idata.posterior.data_vars
    assert "alpha" in idata.posterior.data_vars


# ---------------------------------------------------------------------------
# Mixing: β-marginalised ρ slice should give healthy ESS
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reduced_rho_mixing_with_beta_marginalisation():
    """ρ ESS_bulk should be a non-trivial fraction of total draws.

    Regression test for the β-marginalised ρ slice sampler.  Without
    marginalisation, ρ ESS at this DGP is in single digits per chain;
    with marginalisation it is in the hundreds.  We set a deliberately
    loose floor so the test is robust to seed / library variation.
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    side = 10  # n = 100 keeps the test fast (~10s)
    n = side * side
    rows, cols = [], []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    rows.append(i)
                    cols.append(rr * side + cc)
    W_unweighted = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(W_unweighted.sum(axis=1)).ravel()
    W_sparse = sp.diags(1.0 / deg) @ W_unweighted

    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta_true = np.array([0.5, 0.4])
    rho_true = 0.5
    alpha_true = 2.0
    A = sp.eye(n) - rho_true * W_sparse
    eta = sp.linalg.spsolve(A.tocsc(), X @ beta_true)
    mu = np.exp(eta)
    p = alpha_true / (alpha_true + mu)
    y = rng.negative_binomial(alpha_true, p).astype(float)

    model = bp.SARNegativeBinomial(y=y, X=X, W=W_to_graph(W_sparse))
    model.fit(draws=400, tune=400, chains=2, random_seed=0)

    rho_ess = float(az.ess(model.inference_data, var_names=["rho"]).rho.values)
    # Conservative floor: at 800 total post-warmup draws, even a fraction
    # of 0.1 (ESS=80) signals that ρ is mixing, vs the pre-marginalised
    # ESS that was ~5–10 at this DGP.
    assert rho_ess >= 80.0, f"ρ ESS too low: {rho_ess}"
