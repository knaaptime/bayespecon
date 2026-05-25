"""Unit tests for the spatial-normal sampler primitive."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._samplers._utils._spatial_normal import (
    CholmodFactor,
    has_cholmod,
    sample_spatial_normal,
)


class TestSampleSpatialNormal:
    """Tests for sparse-precision Gaussian sampling."""

    def test_identity_precision(self, rng):
        """P = I gives N(mean_term, I)."""
        n = 50
        P = sp.eye(n, format="csc")
        mean_term = rng.standard_normal(n)

        draw = sample_spatial_normal(P, mean_term, rng=rng)
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))
        # The mean should be mean_term since P = I → m = P^{-1} @ mean_term = mean_term
        # Check empirical mean over many draws
        draws = np.array(
            [sample_spatial_normal(P, mean_term, rng=rng).x for _ in range(2000)]
        )
        empirical_mean = draws.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, mean_term, atol=0.15)

    def test_diagonal_precision(self, rng):
        """P = diag(v) gives N(P^{-1} mean_term, P^{-1})."""
        n = 20
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        mean_term = rng.standard_normal(n)

        draw = sample_spatial_normal(P, mean_term, rng=rng)
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))

        # Expected mean: P^{-1} @ mean_term = mean_term / v
        expected_mean = mean_term / v
        draws = np.array(
            [sample_spatial_normal(P, mean_term, rng=rng).x for _ in range(3000)]
        )
        empirical_mean = draws.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.1)

    def test_tridiagonal_precision(self, rng):
        """Tridiagonal precision: empirical covariance matches P^{-1}."""
        n = 30
        # Build a tridiagonal SPD precision matrix
        diag_val = 2.0
        offdiag_val = -0.5
        P = sp.diags(
            [
                offdiag_val * np.ones(n - 1),
                diag_val * np.ones(n),
                offdiag_val * np.ones(n - 1),
            ],
            offsets=[-1, 0, 1],
            format="csc",
        )
        mean_term = rng.standard_normal(n)

        # Draw many samples
        draws = np.array(
            [sample_spatial_normal(P, mean_term, rng=rng).x for _ in range(5000)]
        )

        # Check mean
        P_dense = P.toarray()
        expected_mean = np.linalg.solve(P_dense, mean_term)
        empirical_mean = draws.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.1)

        # Check diagonal of covariance (most reliably estimated)
        P_inv = np.linalg.inv(P_dense)
        empirical_cov = np.cov(draws.T)
        np.testing.assert_allclose(np.diag(empirical_cov), np.diag(P_inv), atol=0.08)
        # Check off-diagonal elements with looser tolerance (MC noise)
        np.testing.assert_allclose(empirical_cov, P_inv, atol=0.25)

    def test_cached_factor_reuse(self, rng):
        """Cached factor gives same mean but different draws."""
        n = 20
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        mean_term = rng.standard_normal(n)

        # First draw to get the factor
        draw1 = sample_spatial_normal(P, mean_term, rng=rng)
        factor = draw1.factor

        # Second draw reusing the factor
        draw2 = sample_spatial_normal(P, mean_term, rng=rng, cached_factor=factor)

        # Both should have the same mean (P^{-1} @ mean_term)
        expected_mean = mean_term / v
        assert np.allclose(draw1.x, draw2.x) is False  # different random draws
        # But the means should be close (both centered on the same point)
        draws = np.array(
            [
                sample_spatial_normal(P, mean_term, rng=rng, cached_factor=factor).x
                for _ in range(2000)
            ]
        )
        np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.1)

    @pytest.mark.skipif(not has_cholmod(), reason="CHOLMOD not available")
    def test_cholmod_factor(self, rng):
        """CholmodFactor produces correct samples and log-determinant."""
        n = 30
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        mean_term = rng.standard_normal(n)

        factor = CholmodFactor(P)
        x = factor.sample(mean_term, rng=rng)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

        # Check logdet matches scipy
        import scipy.sparse.linalg as spla

        lu = spla.splu(sp.csc_matrix(P), permc_spec="MMD_AT_PLUS_A")
        logdet_splu = np.sum(np.log(np.abs(lu.U.diagonal())))
        np.testing.assert_allclose(factor.logdet(), logdet_splu, rtol=1e-10)

        # Check solve matches scipy
        x_cholmod = factor.solve(mean_term)
        x_splu = lu.solve(mean_term)
        np.testing.assert_allclose(x_cholmod, x_splu, atol=1e-12)

    @pytest.mark.skipif(not has_cholmod(), reason="CHOLMOD not available")
    def test_cholmod_refactorize(self, rng):
        """CholmodFactor can re-factorize with same sparsity pattern."""
        n = 30
        # Two different diagonal precision matrices (same sparsity)
        v1 = rng.uniform(0.5, 2.0, size=n)
        v2 = rng.uniform(0.5, 2.0, size=n)
        P1 = sp.diags(v1, format="csc")
        P2 = sp.diags(v2, format="csc")
        mean_term = rng.standard_normal(n)

        factor = CholmodFactor(P1)
        # Re-factorize with P2
        factor.factorize(P2)
        x = factor.sample(mean_term, rng=rng)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

        # Verify solve matches fresh factorization
        from sksparse.cholmod import cholesky

        f2 = cholesky(P2)
        np.testing.assert_allclose(
            factor.solve(mean_term), f2.solve_A(mean_term), atol=1e-12
        )

    def test_splu_fallback(self, rng):
        """splu fallback produces correct samples."""
        n = 20
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        mean_term = rng.standard_normal(n)

        draw = sample_spatial_normal(P, mean_term, rng=rng, use_cholmod=False)
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))

        # Check empirical mean
        expected_mean = mean_term / v
        draws = np.array(
            [
                sample_spatial_normal(P, mean_term, rng=rng, use_cholmod=False).x
                for _ in range(2000)
            ]
        )
        np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.1)

    @pytest.mark.skipif(not has_cholmod(), reason="scikit-sparse not installed")
    def test_cholmod_pickle_roundtrip(self, rng):
        """CholmodFactor survives pickle round-trip."""
        import pickle

        n = 40
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        cf = CholmodFactor(P)

        # Pickle and unpickle
        data = pickle.dumps(cf)
        cf2 = pickle.loads(data)

        # Should produce same results
        rhs = rng.standard_normal(n)
        cf.factorize(P)
        cf2.factorize(P)
        np.testing.assert_allclose(cf.solve(rhs), cf2.solve(rhs), atol=1e-12)
        assert cf.logdet() == cf2.logdet()

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)
