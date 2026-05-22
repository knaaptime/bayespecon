"""Parameter recovery tests for SARNegBinLatent (Gibbs sampler).

These tests verify that the Pólya–Gamma Gibbs sampler recovers known
parameters from simulated SAR-NB data. They are marked ``slow`` and
``recovery`` because they run MCMC.

Run with::

    pytest -m recovery              # recovery tests only
    pytest -m slow                   # all slow tests
    pytest test_sar_negbin_latent.py # this file only
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import SARNegBinLatent, dgp
from bayespecon.tests.helpers import W_to_graph, make_rook_W

# ---------------------------------------------------------------------------
# Fast build / validation tests (not marked slow)
# ---------------------------------------------------------------------------


class TestSARNegBinLatentBuild:
    """Construction and validation tests (no sampling)."""

    def test_build_model(self):
        """SARNegBinLatent constructs without error."""
        n = 10
        W = W_to_graph(make_rook_W(3))  # 9 units
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.poisson(2, size=n).astype(float)
        model = SARNegBinLatent(y=y[:9], X=X[:9], W=W)
        assert model is not None

    def test_rejects_noninteger_y(self):
        """Non-integer y raises ValueError."""
        n = 10
        W = W_to_graph(make_rook_W(3))
        y = np.array([0.5, 1.0, 2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 0.0])
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        with pytest.raises(ValueError, match="integer-valued"):
            SARNegBinLatent(y=y, X=X, W=W)

    def test_rejects_negative_y(self):
        """Negative y raises ValueError."""
        W = W_to_graph(make_rook_W(3))
        y = np.array([0, 1, -1, 2, 0, 3, 1, 2, 0], dtype=float)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        with pytest.raises(ValueError, match="non-negative"):
            SARNegBinLatent(y=y, X=X, W=W)

    def test_rejects_robust(self):
        """robust=True raises NotImplementedError."""
        W = W_to_graph(make_rook_W(3))
        y = np.ones(9)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        with pytest.raises(NotImplementedError, match="robust"):
            SARNegBinLatent(y=y, X=X, W=W, robust=True)

    def test_rejects_nuts_kwargs(self):
        """NUTS-specific kwargs raise TypeError."""
        W = W_to_graph(make_rook_W(3))
        y = np.ones(9)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        model = SARNegBinLatent(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="nuts_sampler"):
            model.fit(draws=10, nuts_sampler="blackjax")

    def test_rejects_target_accept(self):
        """target_accept kwarg raises TypeError."""
        W = W_to_graph(make_rook_W(3))
        y = np.ones(9)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        model = SARNegBinLatent(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="target_accept"):
            model.fit(draws=10, target_accept=0.95)


# ---------------------------------------------------------------------------
# Parameter recovery tests (marked slow/recovery)
# ---------------------------------------------------------------------------

# Larger grid for parameter recovery — n=100 gives enough information
# for the Gibbs sampler to recover parameters with tight tolerances.
SIDE = 10  # 100 cross-sectional units
RHO_TRUE = 0.4
BETA_TRUE = np.array([0.5, 0.8])
ALPHA_TRUE = 2.0
SIGMA2_TRUE = 0.5  # structural-form residual variance
DRAWS = 1000
TUNE = 1000
CHAINS = 2


@pytest.fixture(scope="module")
def sar_nb_data():
    """Simulated SAR-NB data from the structural-form DGP (n=100)."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE)
    W_graph = W_to_graph(W_dense)
    return dgp.simulate_sar_negbin(
        W=W_graph,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        alpha=ALPHA_TRUE,
        sigma2=SIGMA2_TRUE,
        rng=rng,
    )


@pytest.mark.slow
@pytest.mark.recovery
class TestSARNegBinLatentRecovery:
    """Parameter recovery tests for the Gibbs sampler.

    Uses n=100 (10×10 rook grid) with 1000 draws × 2 chains so that
    posterior summaries are precise enough for meaningful tolerance checks.
    """

    def test_fit_returns_idata(self, sar_nb_data):
        """fit() returns InferenceData with expected groups."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinLatent(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        assert "posterior" in idata
        assert "log_likelihood" in idata
        assert "observed_data" in idata
        assert "rho" in idata.posterior
        assert "beta" in idata.posterior
        assert "sigma" in idata.posterior
        assert "alpha" in idata.posterior

    def test_posterior_shapes(self, sar_nb_data):
        """Posterior arrays have correct shapes."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]
        n_chains = CHAINS
        n_draws = DRAWS

        model = SARNegBinLatent(y=y, X=X, W=W)
        idata = model.fit(
            draws=n_draws,
            tune=TUNE,
            chains=n_chains,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        assert idata.posterior["rho"].shape == (n_chains, n_draws)
        assert idata.posterior["beta"].shape == (n_chains, n_draws, X.shape[1])
        assert idata.posterior["sigma"].shape == (n_chains, n_draws)
        assert idata.posterior["alpha"].shape == (n_chains, n_draws)

    def test_rho_recovery(self, sar_nb_data):
        """Posterior mean for rho is within 0.2 of the true value."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinLatent(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        rho_mean = float(idata.posterior["rho"].mean())
        # The collapsed ρ update integrates out η, giving much better
        # mixing than the conditional update.  With n=100 and
        # sigma2=0.5, the posterior mean should be within 0.2 of truth.
        assert abs(rho_mean - RHO_TRUE) < 0.2, (
            f"rho_mean={rho_mean:.3f} too far from rho_true={RHO_TRUE}"
        )

    def test_alpha_recovery(self, sar_nb_data):
        """Posterior mean for alpha is within 1.5 of the true value."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinLatent(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        alpha_mean = float(idata.posterior["alpha"].mean())
        assert alpha_mean > 0, f"alpha_mean={alpha_mean:.3f} should be positive"
        assert abs(alpha_mean - ALPHA_TRUE) < 1.5, (
            f"alpha_mean={alpha_mean:.3f} too far from alpha_true={ALPHA_TRUE}"
        )

    def test_beta_recovery(self, sar_nb_data):
        """Posterior mean for slope coefficient is within 0.4 of true value."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinLatent(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        beta_mean = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
        # Check slope (beta[1]) is within 0.4 of true value
        assert abs(beta_mean[1] - BETA_TRUE[1]) < 0.4, (
            f"slope={beta_mean[1]:.3f} too far from true={BETA_TRUE[1]}"
        )

    def test_spatial_effects(self, sar_nb_data):
        """spatial_effects() returns a DataFrame with expected columns."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinLatent(y=y, X=X, W=W)
        model.fit(
            draws=200, tune=200, chains=1, random_seed=42, n_jobs=1, progressbar=False
        )

        effects = model.spatial_effects()
        assert "direct" in effects.columns
        assert "indirect" in effects.columns
        assert "total" in effects.columns
        assert np.all(np.isfinite(effects["direct"].values))

    def test_return_eta(self, sar_nb_data):
        """return_eta=True stores the latent field in the posterior."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]
        n = len(y)

        model = SARNegBinLatent(y=y, X=X, W=W)
        idata = model.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            return_eta=True,
        )

        assert "eta" in idata.posterior
        assert idata.posterior["eta"].shape[-1] == n

    def test_thinning(self, sar_nb_data):
        """thin=2 halves the number of stored draws."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinLatent(y=y, X=X, W=W)
        idata = model.fit(
            draws=100,
            tune=100,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            thin=2,
        )

        # With thin=2, we keep 100/2 = 50 draws
        assert idata.posterior["rho"].shape[1] == 50
