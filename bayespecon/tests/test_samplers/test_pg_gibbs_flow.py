the """Tests for Kronecker matvec primitives and flow Gibbs samplers.

Smoke tests use small n=5 grids. Recovery tests use n=8.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bayespecon._samplers._kronecker import (
    kron_At_matvec,
    kron_eigenvalue_bounds,
    kron_logdet_A,
    kron_matvec,
    kron_P_matvec,
)


# ---------------------------------------------------------------------------
# Kronecker matvec primitives
# ---------------------------------------------------------------------------


class TestKronMatvec:
    """Tests for kron_matvec against brute-force Kronecker product."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n = 4
        self.N = self.n * self.n
        self.Ld = rng.standard_normal((self.n, self.n))
        self.Lo = rng.standard_normal((self.n, self.n))
        self.v = rng.standard_normal(self.N)

    def test_kron_matvec_matches_brute_force(self):
        """kron_matvec(v, Ld, Lo) == (Lo ⊗ Ld) @ v."""
        Lo_kron_Ld = np.kron(self.Lo, self.Ld)
        expected = Lo_kron_Ld @ self.v
        result = kron_matvec(self.v, self.Ld, self.Lo)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_kron_At_matvec_matches_brute_force(self):
        """kron_At_matvec(v, Ld.T, Lo.T) == (Lo^T ⊗ Ld^T) @ v."""
        LoT_kron_LdT = np.kron(self.Lo.T, self.Ld.T)
        expected = LoT_kron_LdT @ self.v
        result = kron_At_matvec(self.v, self.Ld.T, self.Lo.T)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_kron_P_matvec_matches_brute_force(self):
        """kron_P_matvec matches (Ld^T Ld ⊗ Lo^T Lo)/σ² + diag(ω)."""
        rng = np.random.default_rng(123)
        sigma2 = 2.0
        omega = rng.uniform(0.5, 2.0, size=self.N)

        LdtLd = self.Ld.T @ self.Ld
        LotLo = self.Lo.T @ self.Lo
        kron_part = np.kron(LotLo, LdtLd) / sigma2
        P_dense = kron_part + np.diag(omega)

        v = rng.standard_normal(self.N)
        expected = P_dense @ v
        result = kron_P_matvec(v, LdtLd, LotLo, omega, sigma2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_kron_matvec_with_identity(self):
        """kron_matvec with Lo=I gives Ld ⊗ I applied to v."""
        I_n = np.eye(self.n)
        result = kron_matvec(self.v, self.Ld, I_n)
        expected = np.kron(I_n, self.Ld) @ self.v
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestKronLogdetA:
    """Tests for kron_logdet_A."""

    def test_zero_rho_gives_zero(self):
        """log|A| = 0 when both rho_d and rho_o are zero."""
        n = 5
        logdet_fn = lambda rho: n * np.log(abs(1 - rho))  # trivial W with single eigenvalue
        result = kron_logdet_A(0.0, 0.0, n, logdet_fn)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_matches_eigenvalue(self):
        """kron_logdet_A matches n*log|I-rho_d*W| + n*log|I-rho_o*W|."""
        n = 5
        rng = np.random.default_rng(42)
        W = sp.random(n, n, density=0.3, format="csr", random_state=0, dtype=np.float64)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        eigs = np.linalg.eigvals(W.toarray().astype(np.float64)).real

        def logdet_fn(rho):
            return float(np.sum(np.log(np.maximum(1 - rho * eigs, 1e-300))))

        rho_d, rho_o = 0.3, 0.2
        expected = n * logdet_fn(rho_d) + n * logdet_fn(rho_o)
        result = kron_logdet_A(rho_d, rho_o, n, logdet_fn)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestKronEigenvalueBounds:
    """Tests for kron_eigenvalue_bounds."""

    def test_bounds_contain_true_eigenvalues(self):
        """Gershgorin bounds should contain all true eigenvalues of P."""
        n = 5
        rng = np.random.default_rng(42)
        W = sp.random(n, n, density=0.3, format="csr", random_state=0, dtype=np.float64)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        W_dense = W.toarray().astype(np.float64)

        rho_d, rho_o = 0.3, 0.2
        omega_val, sigma2 = 1.5, 2.0

        Ld = np.eye(n) - rho_d * W_dense
        Lo = np.eye(n) - rho_o * W_dense
        LdtLd = Ld.T @ Ld
        LotLo = Lo.T @ Lo
        omega = np.full(n * n, omega_val)

        lam_min, lam_max = kron_eigenvalue_bounds(LdtLd, LotLo, omega, sigma2)

        # Build full P matrix using Kronecker product (separable structure)
        N = n * n
        P_full = np.kron(LdtLd, LotLo) / sigma2 + omega_val * np.eye(N)
        true_eigs = np.linalg.eigvalsh(P_full)
        assert lam_min <= true_eigs.min() + 1e-10
        assert lam_max >= true_eigs.max() - 1e-10

    def test_zero_rho_bounds(self):
        """With zero rho, LdtLd = LotLo = I, so P = I/sigma2 + omega*I."""
        n = 4
        LdtLd = np.eye(n)
        LotLo = np.eye(n)
        omega_val, sigma2 = 1.0, 1.0
        omega = np.full(n * n, omega_val)
        lam_min, lam_max = kron_eigenvalue_bounds(LdtLd, LotLo, omega, sigma2)
        # P = I/sigma2 + omega*I = (1 + omega)*I when rho=0
        expected = 1.0 / sigma2 + omega_val
        assert lam_min <= expected + 1e-10
        assert lam_max >= expected - 1e-10


# ---------------------------------------------------------------------------
# Flow Gibbs sampler: construction and smoke tests
# ---------------------------------------------------------------------------


class TestFlowGibbsState:
    """Test FlowGibbsState construction."""

    def test_separable_state(self):
        from bayespecon._samplers.pg_gibbs_flow import FlowGibbsState

        N = 25
        k = 3
        state = FlowGibbsState(
            eta=np.zeros(N),
            beta=np.zeros(k),
            sigma2=1.0,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=None,
            alpha=1.0,
            omega=np.ones(N),
        )
        assert state.rho_w is None
        assert state.eta.shape == (N,)

    def test_nonseparable_state(self):
        from bayespecon._samplers.pg_gibbs_flow import FlowGibbsState

        N = 25
        k = 3
        state = FlowGibbsState(
            eta=np.zeros(N),
            beta=np.zeros(k),
            sigma2=1.0,
            rho_d=0.1,
            rho_o=0.2,
            rho_w=0.05,
            alpha=1.0,
            omega=np.ones(N),
        )
        assert state.rho_w == 0.05


class TestFlowGibbsPriors:
    """Test FlowGibbsPriors defaults."""

    def test_defaults(self):
        from bayespecon._samplers.pg_gibbs_flow import FlowGibbsPriors

        priors = FlowGibbsPriors()
        assert priors.rho_lower == -0.999
        assert priors.rho_upper == 0.999
        assert priors.alpha_sigma == 10.0


# ---------------------------------------------------------------------------
# Model class construction tests
# ---------------------------------------------------------------------------


class TestSARNegBinFlowLatentBuild:
    """Construction and validation tests for SARNegBinFlowLatent."""

    def setup_method(self):
        from bayespecon.dgp.flows import generate_flow_data

        self.n = 5
        self.N = self.n * self.n
        out = generate_flow_data(
            n=self.n,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=10,
        )
        self.G = out["G"]
        self.X = out["X"]
        self.col_names = out["col_names"]
        rng = np.random.default_rng(10)
        self.y = rng.poisson(2, size=self.N).astype(float)

    def test_nonseparable_builds(self):
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            self.y, self.G, self.X,
            col_names=self.col_names,
            miter=5, titer=50, trace_seed=0,
        )
        assert model._n == self.n
        assert model._N == self.N

    def test_nonseparable_rejects_noninteger_y(self):
        from bayespecon.models.flow import SARNegBinFlowLatent

        y_float = np.array([0.5, 1.0, 2.0] * 9)[:self.N]
        with pytest.raises(ValueError, match="integer-valued"):
            SARNegBinFlowLatent(
                y_float, self.G, self.X,
                col_names=self.col_names,
                miter=5, titer=50, trace_seed=0,
            )

    def test_nonseparable_rejects_nuts_kwargs(self):
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            self.y, self.G, self.X,
            col_names=self.col_names,
            miter=5, titer=50, trace_seed=0,
        )
        with pytest.raises(TypeError, match="nuts_sampler"):
            model.fit(draws=10, nuts_sampler="blackjax")

    def test_separable_builds(self):
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        model = SARNegBinFlowSeparableLatent(
            self.y, self.G, self.X,
            col_names=self.col_names,
            trace_seed=0,
        )
        assert model._n == self.n

    def test_separable_rejects_noninteger_y(self):
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        y_float = np.array([0.5, 1.0, 2.0] * 9)[:self.N]
        with pytest.raises(ValueError, match="integer-valued"):
            SARNegBinFlowSeparableLatent(
                y_float, self.G, self.X,
                col_names=self.col_names,
                trace_seed=0,
            )


# ---------------------------------------------------------------------------
# Parameter recovery tests (marked slow/recovery)
# ---------------------------------------------------------------------------

SIDE = 4  # 16 cross-sectional units → N = 256
RHO_D_TRUE = 0.3
RHO_O_TRUE = 0.2
ALPHA_TRUE = 2.0
SIGMA2_TRUE = 0.5
DRAWS = 100
TUNE = 100
CHAINS = 2


@pytest.fixture(scope="module")
def flow_nb_data():
    """Simulated flow NB data from the separable DGP."""
    from bayespecon.dgp.flows import generate_negbin_flow_data_separable

    out = generate_negbin_flow_data_separable(
        n=SIDE,
        rho_d=RHO_D_TRUE,
        rho_o=RHO_O_TRUE,
        alpha=ALPHA_TRUE,
        seed=42,
    )
    return {
        "y": out["y_vec"].astype(float),
        "G": out["G"],
        "X": out["X"],
        "col_names": out["col_names"],
        "n": SIDE,
    }


@pytest.mark.slow
@pytest.mark.recovery
class TestSARNegBinFlowSeparableLatentRecovery:
    """Parameter recovery tests for the separable flow Gibbs sampler."""

    def test_fit_returns_idata(self, flow_nb_data):
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        model = SARNegBinFlowSeparableLatent(
            flow_nb_data["y"],
            flow_nb_data["G"],
            flow_nb_data["X"],
            col_names=flow_nb_data["col_names"],
            trace_seed=42,
        )
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        assert "posterior" in idata
        assert "rho_d" in idata.posterior
        assert "rho_o" in idata.posterior
        assert "rho_w" in idata.posterior
        assert "beta" in idata.posterior
        assert "sigma" in idata.posterior
        assert "alpha" in idata.posterior

    def test_rho_d_recovery(self, flow_nb_data):
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        model = SARNegBinFlowSeparableLatent(
            flow_nb_data["y"],
            flow_nb_data["G"],
            flow_nb_data["X"],
            col_names=flow_nb_data["col_names"],
            trace_seed=42,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        rho_d_mean = float(idata.posterior["rho_d"].mean())
        assert abs(rho_d_mean - RHO_D_TRUE) < 0.3, (
            f"rho_d_mean={rho_d_mean:.3f} too far from rho_d_true={RHO_D_TRUE}"
        )

    def test_rho_o_recovery(self, flow_nb_data):
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        model = SARNegBinFlowSeparableLatent(
            flow_nb_data["y"],
            flow_nb_data["G"],
            flow_nb_data["X"],
            col_names=flow_nb_data["col_names"],
            trace_seed=42,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        rho_o_mean = float(idata.posterior["rho_o"].mean())
        # rho_o is hard to estimate with n=4 (16 units) and few draws;
        # use a wider tolerance than other parameters
        assert abs(rho_o_mean - RHO_O_TRUE) < 0.6, (
            f"rho_o_mean={rho_o_mean:.3f} too far from rho_o_true={RHO_O_TRUE}"
        )

    def test_alpha_recovery(self, flow_nb_data):
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        model = SARNegBinFlowSeparableLatent(
            flow_nb_data["y"],
            flow_nb_data["G"],
            flow_nb_data["X"],
            col_names=flow_nb_data["col_names"],
            trace_seed=42,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        alpha_mean = float(idata.posterior["alpha"].mean())
        assert alpha_mean > 0, f"alpha_mean={alpha_mean:.3f} should be positive"
        assert abs(alpha_mean - ALPHA_TRUE) < 2.0, (
            f"alpha_mean={alpha_mean:.3f} too far from alpha_true={ALPHA_TRUE}"
        )

    def test_posterior_shapes(self, flow_nb_data):
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        model = SARNegBinFlowSeparableLatent(
            flow_nb_data["y"],
            flow_nb_data["G"],
            flow_nb_data["X"],
            col_names=flow_nb_data["col_names"],
            trace_seed=42,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        n_chains = CHAINS
        n_draws = DRAWS
        k = flow_nb_data["X"].shape[1]

        assert idata.posterior["rho_d"].shape == (n_chains, n_draws)
        assert idata.posterior["rho_o"].shape == (n_chains, n_draws)
        assert idata.posterior["rho_w"].shape == (n_chains, n_draws)
        assert idata.posterior["beta"].shape == (n_chains, n_draws, k)
        assert idata.posterior["sigma"].shape == (n_chains, n_draws)
        assert idata.posterior["alpha"].shape == (n_chains, n_draws)

    def test_rho_w_deterministic(self, flow_nb_data):
        """rho_w = -rho_d * rho_o should hold in posterior samples."""
        from bayespecon.models.flow import SARNegBinFlowSeparableLatent

        model = SARNegBinFlowSeparableLatent(
            flow_nb_data["y"],
            flow_nb_data["G"],
            flow_nb_data["X"],
            col_names=flow_nb_data["col_names"],
            trace_seed=42,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        rho_d = idata.posterior["rho_d"].values
        rho_o = idata.posterior["rho_o"].values
        rho_w = idata.posterior["rho_w"].values
        np.testing.assert_allclose(rho_w, -rho_d * rho_o, atol=1e-10)


# ---------------------------------------------------------------------------
# Non-separable model recovery tests (marked slow/recovery)
# ---------------------------------------------------------------------------

RHO_W_TRUE = -RHO_D_TRUE * RHO_O_TRUE  # consistent with separable DGP


@pytest.fixture(scope="module")
def flow_nb_data_ns():
    """Simulated flow NB data for the non-separable model.

    Uses the same separable DGP (rho_w = -rho_d * rho_o) since the
    non-separable model should recover the same parameters.
    """
    from bayespecon.dgp.flows import generate_negbin_flow_data_separable

    out = generate_negbin_flow_data_separable(
        n=SIDE,
        rho_d=RHO_D_TRUE,
        rho_o=RHO_O_TRUE,
        alpha=ALPHA_TRUE,
        seed=42,
    )
    return {
        "y": out["y_vec"].astype(float),
        "G": out["G"],
        "X": out["X"],
        "col_names": out["col_names"],
        "n": SIDE,
    }


@pytest.mark.slow
@pytest.mark.recovery
class TestSARNegBinFlowLatentRecovery:
    """Parameter recovery tests for the non-separable flow Gibbs sampler."""

    def test_fit_returns_idata(self, flow_nb_data_ns):
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            flow_nb_data_ns["y"],
            flow_nb_data_ns["G"],
            flow_nb_data_ns["X"],
            col_names=flow_nb_data_ns["col_names"],
            miter=5, titer=50, trace_seed=0,
        )
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        assert "posterior" in idata
        assert "rho_d" in idata.posterior
        assert "rho_o" in idata.posterior
        assert "rho_w" in idata.posterior
        assert "beta" in idata.posterior
        assert "sigma" in idata.posterior
        assert "alpha" in idata.posterior

    def test_rho_d_recovery(self, flow_nb_data_ns):
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            flow_nb_data_ns["y"],
            flow_nb_data_ns["G"],
            flow_nb_data_ns["X"],
            col_names=flow_nb_data_ns["col_names"],
            miter=5, titer=50, trace_seed=0,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        rho_d_mean = float(idata.posterior["rho_d"].mean())
        assert abs(rho_d_mean - RHO_D_TRUE) < 0.3, (
            f"rho_d_mean={rho_d_mean:.3f} too far from rho_d_true={RHO_D_TRUE}"
        )

    def test_rho_o_recovery(self, flow_nb_data_ns):
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            flow_nb_data_ns["y"],
            flow_nb_data_ns["G"],
            flow_nb_data_ns["X"],
            col_names=flow_nb_data_ns["col_names"],
            miter=5, titer=50, trace_seed=0,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        rho_o_mean = float(idata.posterior["rho_o"].mean())
        # rho_o is hard to estimate with n=4 (16 units) and few draws;
        # use a wider tolerance than other parameters
        assert abs(rho_o_mean - RHO_O_TRUE) < 0.6, (
            f"rho_o_mean={rho_o_mean:.3f} too far from rho_o_true={RHO_O_TRUE}"
        )

    def test_alpha_recovery(self, flow_nb_data_ns):
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            flow_nb_data_ns["y"],
            flow_nb_data_ns["G"],
            flow_nb_data_ns["X"],
            col_names=flow_nb_data_ns["col_names"],
            miter=5, titer=50, trace_seed=0,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        alpha_mean = float(idata.posterior["alpha"].mean())
        assert alpha_mean > 0, f"alpha_mean={alpha_mean:.3f} should be positive"
        assert abs(alpha_mean - ALPHA_TRUE) < 2.0, (
            f"alpha_mean={alpha_mean:.3f} too far from alpha_true={ALPHA_TRUE}"
        )

    def test_posterior_shapes(self, flow_nb_data_ns):
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            flow_nb_data_ns["y"],
            flow_nb_data_ns["G"],
            flow_nb_data_ns["X"],
            col_names=flow_nb_data_ns["col_names"],
            miter=5, titer=50, trace_seed=0,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        n_chains = CHAINS
        n_draws = DRAWS
        k = flow_nb_data_ns["X"].shape[1]

        assert idata.posterior["rho_d"].shape == (n_chains, n_draws)
        assert idata.posterior["rho_o"].shape == (n_chains, n_draws)
        assert idata.posterior["rho_w"].shape == (n_chains, n_draws)
        assert idata.posterior["beta"].shape == (n_chains, n_draws, k)
        assert idata.posterior["sigma"].shape == (n_chains, n_draws)
        assert idata.posterior["alpha"].shape == (n_chains, n_draws)

    def test_rho_w_free(self, flow_nb_data_ns):
        """rho_w is a free parameter in the non-separable model."""
        from bayespecon.models.flow import SARNegBinFlowLatent

        model = SARNegBinFlowLatent(
            flow_nb_data_ns["y"],
            flow_nb_data_ns["G"],
            flow_nb_data_ns["X"],
            col_names=flow_nb_data_ns["col_names"],
            miter=5, titer=50, trace_seed=0,
        )
        idata = model.fit(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            random_seed=42, n_jobs=1, progressbar=False,
        )

        rho_w_mean = float(idata.posterior["rho_w"].mean())
        # rho_w should be near -rho_d * rho_o ≈ -0.06
        assert abs(rho_w_mean - RHO_W_TRUE) < 0.3, (
            f"rho_w_mean={rho_w_mean:.3f} too far from rho_w_true={RHO_W_TRUE}"
        )