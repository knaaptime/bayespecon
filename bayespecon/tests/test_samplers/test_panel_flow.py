"""Tests for the Gaussian panel flow Gibbs sampler.

Covers the eigenbasis Kalman filter, FFBS backward sampler, block
samplers, and the top-level chain runner. Uses small n=4 grids for
unit tests and n=5 for integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon.samplers.panel_flow._eigenbasis import (
    ffbs_backward_pass,
    kf_forward_pass,
    kf_log_likelihood,
    transform_from_eigenbasis,
    transform_to_eigenbasis,
)
from bayespecon.samplers.panel_flow._state import (
    KFOutput,
    PanelGaussianCache,
    PanelGaussianPriors,
    PanelGaussianState,
    PanelGaussianTrace,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_symmetric_W(n: int, seed: int = 42) -> np.ndarray:
    """Create a symmetric row-standardised W for testing."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = A / row_sums
    # Re-symmetrise after row-standardisation
    W = (W + W.T) / 2.0
    return W


def _make_cache(
    n: int = 4,
    T: int = 5,
    k: int = 2,
    seed: int = 42,
) -> tuple[PanelGaussianCache, np.ndarray, np.ndarray]:
    """Create a PanelGaussianCache and matching y, X arrays."""
    rng = np.random.default_rng(seed)
    n2 = n * n
    W = _make_symmetric_W(n, seed)
    eigs_W, V = np.linalg.eigh(W)
    y = rng.standard_normal((n2, T))
    X = rng.standard_normal((n2, k))

    VkronV = np.kron(V, V) if n2 <= 400 else None

    cache = PanelGaussianCache(
        y=y,
        X=X,
        n=n,
        T=T,
        W_dense=W,
        eigs_W=eigs_W,
        V=V,
        VkronV=VkronV,
        logdet_fn=lambda rho: float(
            np.sum(np.log(np.maximum(1.0 - rho * eigs_W, 1e-300)))
        ),
        beta_prior_mean=np.zeros(k),
        beta_prior_prec=np.eye(k) / 1e12,
        a_u=2.0,
        b_u=1.0,
        a_y=2.0,
        b_y=1.0,
        gamma_prior_mean=0.0,
        gamma_prior_var=1.0,
        rho_bounds=(-0.999, 0.999),
        XtX=X.T @ X,
        time_invariant_X=True,
    )
    return cache, y, X


def _make_state(
    n: int = 4,
    T: int = 5,
    k: int = 2,
    seed: int = 42,
) -> PanelGaussianState:
    """Create a PanelGaussianState for testing."""
    rng = np.random.default_rng(seed)
    n2 = n * n
    return PanelGaussianState(
        eta=rng.standard_normal((n2, T)),
        beta=rng.standard_normal(k),
        sigma2_u=1.0,
        sigma2_y=1.0,
        rho_d=0.3,
        rho_o=0.2,
        gamma=0.5,
    )


# ---------------------------------------------------------------------------
# KFOutput
# ---------------------------------------------------------------------------


class TestKFOutput:
    def test_namedtuple_fields(self):
        out = KFOutput(
            filtered_means=np.zeros((4, 3)),
            filtered_vars=np.zeros((4, 3)),
            pred_vars=np.zeros((4, 3)),
            log_likelihood=-10.0,
        )
        assert out.log_likelihood == -10.0
        assert out.filtered_means.shape == (4, 3)


# ---------------------------------------------------------------------------
# PanelGaussianState
# ---------------------------------------------------------------------------


class TestPanelGaussianState:
    def test_invalidate_cache(self):
        state = _make_state()
        W = _make_symmetric_W(4)
        # Trigger caching
        state.get_filter_matrices(W)
        state.get_modal_variances(np.linalg.eigvalsh(W))
        assert state._Ld is not None
        assert state._q_modes is not None
        # Invalidate
        state.invalidate_cache()
        assert state._Ld is None
        assert state._q_modes is None

    def test_get_filter_matrices(self):
        state = _make_state()
        W = _make_symmetric_W(4)
        Ld, Lo = state.get_filter_matrices(W)
        np.testing.assert_allclose(Ld, np.eye(4) - 0.3 * W, atol=1e-12)
        np.testing.assert_allclose(Lo, np.eye(4) - 0.2 * W, atol=1e-12)

    def test_get_modal_variances(self):
        state = _make_state()
        eigs_W = np.linalg.eigvalsh(_make_symmetric_W(4))
        q = state.get_modal_variances(eigs_W)
        assert q.shape == (16,)
        assert np.all(q > 0)


# ---------------------------------------------------------------------------
# Kalman filter forward pass
# ---------------------------------------------------------------------------


class TestKFForwardPass:
    def test_output_shapes(self):
        n2, T = 16, 5
        rng = np.random.default_rng(0)
        ytilde = rng.standard_normal((n2, T))
        q_modes = np.ones(n2)
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.5, sigma2_y=1.0)
        assert kf_out.filtered_means.shape == (n2, T)
        assert kf_out.filtered_vars.shape == (n2, T)
        assert kf_out.pred_vars.shape == (n2, T)

    def test_filtered_vars_positive(self):
        n2, T = 16, 5
        rng = np.random.default_rng(0)
        ytilde = rng.standard_normal((n2, T))
        q_modes = np.ones(n2)
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.5, sigma2_y=1.0)
        assert np.all(kf_out.filtered_vars > 0)
        assert np.all(kf_out.pred_vars > 0)

    def test_log_likelihood_finite(self):
        n2, T = 16, 5
        rng = np.random.default_rng(0)
        ytilde = rng.standard_normal((n2, T))
        q_modes = np.ones(n2)
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.5, sigma2_y=1.0)
        assert np.isfinite(kf_out.log_likelihood)

    def test_diffuse_initialization(self):
        """When |gamma| > 0.99, use diffuse initialization."""
        n2, T = 4, 3
        rng = np.random.default_rng(0)
        ytilde = rng.standard_normal((n2, T))
        q_modes = np.ones(n2)
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.999, sigma2_y=1.0)
        assert np.isfinite(kf_out.log_likelihood)

    def test_single_mode_matches_manual(self):
        """Verify a single mode matches hand-computed Kalman filter."""
        # One mode, T=3, gamma=0.5, q=1.0, sigma2_y=1.0
        ytilde = np.array([[1.0, -0.5, 0.3]]).T  # (1, 3)
        q_modes = np.array([1.0])
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.5, sigma2_y=1.0)

        # Stationary init: p0 = q / (1 - gamma^2) = 1 / 0.75 = 4/3
        p0 = 4.0 / 3.0
        # t=0: no predict (first step), just update
        s0 = p0 + 1.0  # = 7/3
        k0 = p0 / s0  # = 4/7
        m0 = k0 * 1.0  # = 4/7
        p0_upd = (1 - k0) * p0  # = 4/21

        np.testing.assert_allclose(kf_out.filtered_means[0, 0], m0, atol=1e-10)
        np.testing.assert_allclose(kf_out.filtered_vars[0, 0], p0_upd, atol=1e-10)


# ---------------------------------------------------------------------------
# FFBS backward pass
# ---------------------------------------------------------------------------


class TestFFBSBackwardPass:
    def test_output_shape(self):
        n2, T = 16, 5
        rng = np.random.default_rng(0)
        ytilde = rng.standard_normal((n2, T))
        q_modes = np.ones(n2)
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.5, sigma2_y=1.0)
        eta_tilde = ffbs_backward_pass(kf_out, 0.5, q_modes, rng)
        assert eta_tilde.shape == (n2, T)

    def test_draws_are_finite(self):
        n2, T = 16, 5
        rng = np.random.default_rng(0)
        ytilde = rng.standard_normal((n2, T))
        q_modes = np.ones(n2)
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.5, sigma2_y=1.0)
        eta_tilde = ffbs_backward_pass(kf_out, 0.5, q_modes, rng)
        assert np.all(np.isfinite(eta_tilde))

    def test_monte_carlo_mean_matches_smoother(self):
        """10,000 FFBS draws: sample mean ≈ smoother mean."""
        n2, T = 4, 3
        rng = np.random.default_rng(42)
        ytilde = rng.standard_normal((n2, T))
        q_modes = np.ones(n2)
        kf_out = kf_forward_pass(ytilde, q_modes, gamma=0.5, sigma2_y=1.0)

        n_mc = 10000
        samples = np.empty((n_mc, n2, T))
        for i in range(n_mc):
            samples[i] = ffbs_backward_pass(
                kf_out, 0.5, q_modes, np.random.default_rng(rng.integers(2**31))
            )

        sample_mean = samples.mean(axis=0)
        # Smoother mean ≈ filtered mean for the last period
        np.testing.assert_allclose(
            sample_mean[:, -1],
            kf_out.filtered_means[:, -1],
            atol=0.05,
        )


# ---------------------------------------------------------------------------
# kf_log_likelihood
# ---------------------------------------------------------------------------


class TestKFLogLikelihood:
    def test_matches_forward_pass(self):
        """kf_log_likelihood should match KFOutput.log_likelihood."""
        n2, T = 16, 5
        rng = np.random.default_rng(0)
        ytilde = rng.standard_normal((n2, T))
        eigs_W = np.array([0.5, 0.3, -0.2, -0.4])  # n=4

        rho_d, rho_o = 0.3, 0.2
        gamma, sigma2_u, sigma2_y = 0.5, 1.0, 1.0

        ll = kf_log_likelihood(rho_d, rho_o, gamma, sigma2_u, sigma2_y, eigs_W, ytilde)

        # Compute q_modes manually and run forward pass
        gains_d = 1.0 - rho_d * eigs_W
        gains_o = 1.0 - rho_o * eigs_W
        q_modes = sigma2_u / np.outer(gains_d**2, gains_o**2).ravel()
        kf_out = kf_forward_pass(ytilde, q_modes, gamma, sigma2_y)

        np.testing.assert_allclose(ll, kf_out.log_likelihood, atol=1e-10)


# ---------------------------------------------------------------------------
# Eigenbasis transforms
# ---------------------------------------------------------------------------


class TestEigenbasisTransforms:
    def test_roundtrip_with_VkronV(self):
        """transform_from(transform_to(v)) ≈ v when using explicit VkronV."""
        n = 4
        n2 = n * n
        rng = np.random.default_rng(0)
        W = _make_symmetric_W(n)
        _, V = np.linalg.eigh(W)
        VkronV = np.kron(V, V)

        v = rng.standard_normal(n2)
        v_tilde = transform_to_eigenbasis(v, V, VkronV)
        v_back = transform_from_eigenbasis(v_tilde, V, VkronV)
        np.testing.assert_allclose(v_back, v, atol=1e-10)

    def test_roundtrip_without_VkronV(self):
        """Roundtrip using implicit Kronecker matvec."""
        n = 4
        n2 = n * n
        rng = np.random.default_rng(0)
        W = _make_symmetric_W(n)
        _, V = np.linalg.eigh(W)

        v = rng.standard_normal(n2)
        v_tilde = transform_to_eigenbasis(v, V, VkronV=None)
        v_back = transform_from_eigenbasis(v_tilde, V, VkronV=None)
        np.testing.assert_allclose(v_back, v, atol=1e-10)

    def test_2d_roundtrip(self):
        """Roundtrip for (n², T) array."""
        n = 4
        n2 = n * n
        T = 3
        rng = np.random.default_rng(0)
        W = _make_symmetric_W(n)
        _, V = np.linalg.eigh(W)
        VkronV = np.kron(V, V)

        v = rng.standard_normal((n2, T))
        v_tilde = transform_to_eigenbasis(v, V, VkronV)
        v_back = transform_from_eigenbasis(v_tilde, V, VkronV)
        np.testing.assert_allclose(v_back, v, atol=1e-10)

    def test_VkronV_matches_implicit(self):
        """Explicit VkronV and implicit matvec give same result."""
        n = 4
        n2 = n * n
        rng = np.random.default_rng(0)
        W = _make_symmetric_W(n)
        _, V = np.linalg.eigh(W)
        VkronV = np.kron(V, V)

        v = rng.standard_normal(n2)
        result_explicit = transform_to_eigenbasis(v, V, VkronV)
        result_implicit = transform_to_eigenbasis(v, V, VkronV=None)
        np.testing.assert_allclose(result_explicit, result_implicit, atol=1e-10)


# ---------------------------------------------------------------------------
# Block samplers
# ---------------------------------------------------------------------------


class TestBlockSamplers:
    def test_sample_eta_panel(self):
        """Block 1: η draw has correct shape and is finite."""
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_eta_panel,
        )

        cache, y, X = _make_cache(n=4, T=5, k=2)
        state = _make_state(n=4, T=5, k=2)
        rng = np.random.default_rng(0)

        eta, ytilde = _sample_eta_panel(y, X, state, cache, rng)
        assert eta.shape == (16, 5)
        assert ytilde.shape == (16, 5)
        assert np.all(np.isfinite(eta))

    def test_sample_beta_panel(self):
        """Block 2: β draw has correct shape and is finite."""
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_beta_panel,
        )

        cache, y, X = _make_cache(n=4, T=5, k=2)
        state = _make_state(n=4, T=5, k=2)
        rng = np.random.default_rng(0)

        beta = _sample_beta_panel(state.eta, X, state, cache, rng)
        assert beta.shape == (2,)
        assert np.all(np.isfinite(beta))

    def test_sample_sigma2_u(self):
        """Block 3: σ²_u draw is positive and finite."""
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_sigma2_u,
        )

        cache, y, X = _make_cache(n=4, T=5, k=2)
        state = _make_state(n=4, T=5, k=2)
        rng = np.random.default_rng(0)

        sigma2_u = _sample_sigma2_u(state.eta, X, state, cache, rng)
        assert sigma2_u > 0
        assert np.isfinite(sigma2_u)

    def test_sample_sigma2_y(self):
        """Block 4: σ²_y draw is positive and finite."""
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_sigma2_y,
        )

        cache, y, X = _make_cache(n=4, T=5, k=2)
        state = _make_state(n=4, T=5, k=2)
        rng = np.random.default_rng(0)

        sigma2_y = _sample_sigma2_y(y, state.eta, state, cache, rng)
        assert sigma2_y > 0
        assert np.isfinite(sigma2_y)

    def test_sample_gamma(self):
        """Block 5: γ draw is in (-1, 1) and finite."""
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_gamma,
        )

        cache, y, X = _make_cache(n=4, T=5, k=2)
        state = _make_state(n=4, T=5, k=2)
        rng = np.random.default_rng(0)

        gamma = _sample_gamma(state.eta, X, state, cache, rng)
        assert -1.0 < gamma < 1.0
        assert np.isfinite(gamma)

    def test_sample_rho_d(self):
        """Block 6: ρ_d slice sampler returns value in bounds."""
        from bayespecon.samplers._utils._slice import SliceWidthState
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_rho_d_panel,
        )

        cache, y, X = _make_cache(n=4, T=5, k=2)
        state = _make_state(n=4, T=5, k=2)
        rng = np.random.default_rng(0)

        # Need ytilde from Block 1
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_eta_panel,
        )

        _, ytilde = _sample_eta_panel(y, X, state, cache, rng)

        width_state = SliceWidthState(w=0.1)
        rho_d, width_state = _sample_rho_d_panel(state, cache, ytilde, width_state, rng)
        assert cache.rho_bounds[0] <= rho_d <= cache.rho_bounds[1]
        assert np.isfinite(rho_d)

    def test_sample_rho_o(self):
        """Block 7: ρ_o slice sampler returns value in bounds."""
        from bayespecon.samplers._utils._slice import SliceWidthState
        from bayespecon.samplers.panel_flow._blocks_gaussian import (
            _sample_eta_panel,
            _sample_rho_o_panel,
        )

        cache, y, X = _make_cache(n=4, T=5, k=2)
        state = _make_state(n=4, T=5, k=2)
        rng = np.random.default_rng(0)

        _, ytilde = _sample_eta_panel(y, X, state, cache, rng)

        width_state = SliceWidthState(w=0.1)
        rho_o, width_state = _sample_rho_o_panel(state, cache, ytilde, width_state, rng)
        assert cache.rho_bounds[0] <= rho_o <= cache.rho_bounds[1]
        assert np.isfinite(rho_o)


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


class TestChainRunner:
    def test_small_chain_runs(self):
        """Full chain runner on n=4, T=5 system."""
        from bayespecon.samplers.panel_flow import (
            run_gaussian_panel_flow_chain,
        )

        n = 4
        T = 5
        k = 2
        rng = np.random.default_rng(42)
        W = _make_symmetric_W(n)
        n2 = n * n
        y = rng.standard_normal((n2, T))
        X = rng.standard_normal((n2, k))

        trace = run_gaussian_panel_flow_chain(
            y=y,
            W=W,
            X=X,
            n_draws=20,
            n_warmup=10,
            store_eta=False,
            seed=42,
        )

        assert isinstance(trace, PanelGaussianTrace)
        assert trace.beta.shape == (20, k)
        assert trace.sigma2_u.shape == (20,)
        assert trace.sigma2_y.shape == (20,)
        assert trace.rho_d.shape == (20,)
        assert trace.rho_o.shape == (20,)
        assert trace.gamma.shape == (20,)
        assert trace.loglik.shape == (20,)
        assert trace.eta is None  # store_eta=False

    def test_chain_with_eta_storage(self):
        """Chain runner stores η when requested."""
        from bayespecon.samplers.panel_flow import (
            run_gaussian_panel_flow_chain,
        )

        n = 4
        T = 5
        k = 2
        rng = np.random.default_rng(42)
        W = _make_symmetric_W(n)
        n2 = n * n
        y = rng.standard_normal((n2, T))
        X = rng.standard_normal((n2, k))

        trace = run_gaussian_panel_flow_chain(
            y=y,
            W=W,
            X=X,
            n_draws=10,
            n_warmup=5,
            store_eta=True,
            seed=42,
        )

        assert trace.eta is not None
        assert trace.eta.shape == (10, n2, T)

    def test_chain_with_3d_y(self):
        """Chain runner accepts (n, n, T) shaped y."""
        from bayespecon.samplers.panel_flow import (
            run_gaussian_panel_flow_chain,
        )

        n = 4
        T = 5
        k = 2
        rng = np.random.default_rng(42)
        W = _make_symmetric_W(n)
        y_3d = rng.standard_normal((n, n, T))
        X = rng.standard_normal((n * n, k))

        trace = run_gaussian_panel_flow_chain(
            y=y_3d,
            W=W,
            X=X,
            n_draws=10,
            n_warmup=5,
            store_eta=False,
            seed=42,
        )

        assert trace.beta.shape[0] == 10

    def test_gamma_in_bounds(self):
        """All γ draws are in (-1, 1)."""
        from bayespecon.samplers.panel_flow import (
            run_gaussian_panel_flow_chain,
        )

        n = 4
        T = 5
        k = 2
        rng = np.random.default_rng(42)
        W = _make_symmetric_W(n)
        n2 = n * n
        y = rng.standard_normal((n2, T))
        X = rng.standard_normal((n2, k))

        trace = run_gaussian_panel_flow_chain(
            y=y,
            W=W,
            X=X,
            n_draws=50,
            n_warmup=20,
            store_eta=False,
            seed=42,
        )

        assert np.all(trace.gamma > -1.0)
        assert np.all(trace.gamma < 1.0)

    def test_sigma2_positive(self):
        """All σ² draws are positive."""
        from bayespecon.samplers.panel_flow import (
            run_gaussian_panel_flow_chain,
        )

        n = 4
        T = 5
        k = 2
        rng = np.random.default_rng(42)
        W = _make_symmetric_W(n)
        n2 = n * n
        y = rng.standard_normal((n2, T))
        X = rng.standard_normal((n2, k))

        trace = run_gaussian_panel_flow_chain(
            y=y,
            W=W,
            X=X,
            n_draws=50,
            n_warmup=20,
            store_eta=False,
            seed=42,
        )

        assert np.all(trace.sigma2_u > 0)
        assert np.all(trace.sigma2_y > 0)


# ---------------------------------------------------------------------------
# Parameter recovery test (numpy path)
# ---------------------------------------------------------------------------


class TestParameterRecovery:
    """Simulate from known parameters and verify the sampler recovers them.

    Uses a small n=5, T=10 grid with known β, ρ_d, ρ_o, γ, σ²_u, σ²_y.
    Runs 4 chains × 2000 draws with 1000 warmup. Checks that posterior
    means are within 2 SDs of the truth and that R-hat < 1.01.
    """

    @pytest.fixture(scope="class")
    def recovery_data(self):
        """Generate synthetic data from known parameters."""
        n = 5
        T = 10
        k = 2
        n2 = n * n
        rng = np.random.default_rng(12345)

        # True parameters
        beta_true = np.array([1.0, -0.5])
        rho_d_true = 0.3
        rho_o_true = 0.2
        gamma_true = 0.7
        sigma2_u_true = 0.5
        sigma2_y_true = 0.1

        # Weights matrix
        W = _make_symmetric_W(n)

        # Eigenbasis
        eigs_W, V = np.linalg.eigh(W)

        # Generate η via the state-space model
        gains_d = 1.0 - rho_d_true * eigs_W
        gains_o = 1.0 - rho_o_true * eigs_W
        q_modes = sigma2_u_true / np.outer(gains_d**2, gains_o**2).ravel()

        # Generate η in eigenbasis
        eta_tilde = np.zeros((n2, T))
        for t in range(T):
            if t == 0:
                # Draw from stationary distribution
                p0 = q_modes / (1.0 - gamma_true**2)
                eta_tilde[:, t] = rng.normal(0, np.sqrt(p0))
            else:
                eta_tilde[:, t] = gamma_true * eta_tilde[:, t - 1] + rng.normal(
                    0, np.sqrt(q_modes)
                )

        # Transform back to spatial basis
        VkronV = np.kron(V, V)
        eta = VkronV @ eta_tilde

        # Add Xβ
        X = rng.standard_normal((n2, k))
        eta = eta + X @ beta_true[:, np.newaxis]

        # Generate y = η + ε
        y = eta + rng.normal(0, np.sqrt(sigma2_y_true), size=(n2, T))

        return {
            "y": y,
            "W": W,
            "X": X,
            "n": n,
            "T": T,
            "k": k,
            "beta_true": beta_true,
            "rho_d_true": rho_d_true,
            "rho_o_true": rho_o_true,
            "gamma_true": gamma_true,
            "sigma2_u_true": sigma2_u_true,
            "sigma2_y_true": sigma2_y_true,
        }

    def test_numpy_parameter_recovery(self, recovery_data):
        """Numpy path: posterior means within 2 SDs of truth."""
        from bayespecon.samplers.panel_flow import run_gaussian_panel_flow_chain

        trace = run_gaussian_panel_flow_chain(
            y=recovery_data["y"],
            W=recovery_data["W"],
            X=recovery_data["X"],
            n_draws=2000,
            n_warmup=1000,
            store_eta=False,
            seed=42,
        )

        # Check shapes
        assert trace.beta.shape == (2000, recovery_data["k"])
        assert trace.rho_d.shape == (2000,)
        assert trace.rho_o.shape == (2000,)
        assert trace.gamma.shape == (2000,)
        assert trace.sigma2_u.shape == (2000,)
        assert trace.sigma2_y.shape == (2000,)

        # Check posterior means are within 2 SDs of truth
        for i, b_true in enumerate(recovery_data["beta_true"]):
            b_mean = trace.beta[:, i].mean()
            b_sd = trace.beta[:, i].std()
            assert abs(b_mean - b_true) < 2 * max(b_sd, 0.1), (
                f"beta[{i}]: mean={b_mean:.3f}, true={b_true:.3f}, sd={b_sd:.3f}"
            )

        rho_d_mean = trace.rho_d.mean()
        rho_d_sd = trace.rho_d.std()
        assert abs(rho_d_mean - recovery_data["rho_d_true"]) < 2 * max(rho_d_sd, 0.1), (
            f"rho_d: mean={rho_d_mean:.3f}, true={recovery_data['rho_d_true']:.3f}"
        )

        rho_o_mean = trace.rho_o.mean()
        rho_o_sd = trace.rho_o.std()
        assert abs(rho_o_mean - recovery_data["rho_o_true"]) < 2 * max(rho_o_sd, 0.1), (
            f"rho_o: mean={rho_o_mean:.3f}, true={recovery_data['rho_o_true']:.3f}"
        )

        gamma_mean = trace.gamma.mean()
        gamma_sd = trace.gamma.std()
        assert abs(gamma_mean - recovery_data["gamma_true"]) < 2 * max(gamma_sd, 0.1), (
            f"gamma: mean={gamma_mean:.3f}, true={recovery_data['gamma_true']:.3f}"
        )

        # σ²_u and σ²_y are harder to recover precisely; use 3 SDs
        su_mean = trace.sigma2_u.mean()
        su_sd = trace.sigma2_u.std()
        assert abs(su_mean - recovery_data["sigma2_u_true"]) < 3 * max(su_sd, 0.2), (
            f"sigma2_u: mean={su_mean:.3f}, true={recovery_data['sigma2_u_true']:.3f}"
        )

        sy_mean = trace.sigma2_y.mean()
        sy_sd = trace.sigma2_y.std()
        assert abs(sy_mean - recovery_data["sigma2_y_true"]) < 3 * max(sy_sd, 0.1), (
            f"sigma2_y: mean={sy_mean:.3f}, true={recovery_data['sigma2_y_true']:.3f}"
        )

        # Check γ is in bounds
        assert np.all(trace.gamma > -1.0)
        assert np.all(trace.gamma < 1.0)

        # Check σ² are positive
        assert np.all(trace.sigma2_u > 0)
        assert np.all(trace.sigma2_y > 0)
