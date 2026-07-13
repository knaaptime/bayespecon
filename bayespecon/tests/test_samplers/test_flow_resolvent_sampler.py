"""Correctness of the resolvent-gradient flow sampler's ρ-conditional target.

The novel coupling is the ρ log-posterior and its gradient (exact O(N) data term +
resolvent log-determinant).  We validate the gradient against a finite difference of
the target value using the *exact* log-determinant — deterministic and fast, with no
GMRES or MCMC loop.  (End-to-end posterior recovery is a GPU benchmark.)
"""

from __future__ import annotations

import numpy as np

from bayespecon._logdet._flow_resolvent import flow_logdet_grad_exact
from bayespecon.samplers.gaussian._flow_resolvent import FlowResolventTarget


def _directed_flow_data(n, seed):
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0.0)
    A[A.sum(1) == 0, 0] = 1.0
    W = A / A.sum(1, keepdims=True)
    N = n * n
    Ide = np.eye(n)
    WF = 0.3 * np.kron(Ide, W) + 0.2 * np.kron(W, Ide) - 0.05 * np.kron(W, W)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    beta = np.array([1.0, -0.5])
    y = np.linalg.solve(np.eye(N) - WF, X @ beta + 0.4 * rng.standard_normal(N))
    return W, y, X


def _exact_value_and_grad(W):
    lam = np.linalg.eigvals(W)
    li, lj = lam[:, None], lam[None, :]

    def _vg(rd, ro, rw):
        mu = ro * li + rd * lj + rw * (li * lj)
        val = float(np.sum(np.log(np.abs(1.0 - mu))))
        return val, flow_logdet_grad_exact(W, rd, ro, rw)

    return _vg


def test_rho_gradient_matches_finite_difference():
    """logpost_and_grad's gradient equals a finite difference of its value (exact ld)."""
    W, y, X = _directed_flow_data(n=15, seed=0)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))

    beta = np.array([0.8, -0.4])
    sigma2 = 0.3
    h = 1e-6
    for rho in ([0.3, 0.2, -0.05], [0.5, 0.1, 0.1], [-0.2, 0.4, -0.1]):
        rho = np.array(rho)
        _, grad = tgt.logpost_and_grad(rho, beta, sigma2)
        fd = np.zeros(3)
        for k in range(3):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += h
            rm[k] -= h
            vp, _ = tgt.logpost_and_grad(rp, beta, sigma2)
            vm, _ = tgt.logpost_and_grad(rm, beta, sigma2)
            fd[k] = (vp - vm) / (2 * h)
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-3)


def test_Ay_is_linear_in_rho():
    """A(ρ)y = y - Lρ uses the precomputed lags and is exactly linear in ρ."""
    W, y, X = _directed_flow_data(n=12, seed=1)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))
    r1 = tgt.Ay(np.array([0.3, 0.2, -0.05]))
    r2 = tgt.Ay(np.array([0.1, 0.0, 0.2]))
    mid = tgt.Ay(np.array([0.2, 0.1, 0.075]))
    np.testing.assert_allclose(mid, 0.5 * (r1 + r2), atol=1e-12)


def test_beta_sigma2_gibbs_runs():
    """The conjugate β, σ² draw returns finite values of the right shape."""
    W, y, X = _directed_flow_data(n=12, seed=2)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))
    beta, sigma2 = tgt.draw_beta_sigma2(
        np.array([0.3, 0.2, -0.05]), np.random.default_rng(0)
    )
    assert beta.shape == (2,)
    assert np.isfinite(beta).all() and np.isfinite(sigma2) and sigma2 > 0


def test_short_chain_with_exact_logdet_recovers_signs():
    """A short MALA-within-Gibbs run (exact ld, no GMRES) moves ρ toward the truth."""
    from bayespecon.samplers.gaussian._flow_resolvent import run_flow_resolvent_gibbs

    W, y, X = _directed_flow_data(n=15, seed=3)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))
    post = run_flow_resolvent_gibbs(tgt, draws=400, tune=400, step_size=6e-4, seed=1)
    # True ρ = (0.3, 0.2, -0.05): dominant components positive, small and finite.
    assert np.isfinite(post["rho_d"]).all()
    assert post["rho_d"].mean() > 0.0
    assert post["rho_o"].mean() > 0.0
    assert abs(post["rho_d"].mean()) < 0.9
