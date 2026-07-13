r"""Resolvent-gradient sampler for the unrestricted Gaussian flow model.

Model: :math:`A(\rho)\,y = X\beta + \varepsilon`, ``A = I_N - W_F``,
:math:`W_F = \rho_d(I\otimes W) + \rho_o(W\otimes I) + \rho_w(W\otimes W)`,
:math:`\varepsilon \sim N(0, \sigma^2 I_N)`, ``N = n²``.

The conditional log-posterior for the flow parameters ``ρ`` is

.. math::

    \log p(\rho \mid \beta, \sigma^2)
      = \log|A(\rho)| - \frac{1}{2\sigma^2}\lVert A(\rho)y - X\beta\rVert^2
        + \log\pi(\rho),

where — crucially — ``A(ρ)y = y - ρ_d\,W_d y - ρ_o\,W_o y - ρ_w\,W_w y`` is *linear*
in ``ρ`` using the precomputed spatial lags ``W_k y``, so the data term and its
gradient are **exact and O(N)**.  Only the ``log|A|`` Jacobian and its gradient are
non-trivial; these come from the scalable resolvent-Kronecker estimator
(:mod:`bayespecon._logdet._flow_resolvent`), whose relative error *improves* with
``N`` — replacing the noise-amplified ``"traces"`` value method.

``β`` and ``σ²`` have the usual conjugate Gibbs updates given ``ρ`` (``A(ρ)y`` is a
cheap linear transform of the data), so the sampler is MALA-on-``ρ`` within Gibbs.

.. note::
   Per-step this needs a handful of Kronecker GMRES solves; it is intended for the
   GPU path (batched ``n x n`` matmuls) where it becomes competitive.  The exactness
   of the ``ρ`` gradient/target coupling is unit-tested against finite differences of
   the exact log-determinant; end-to-end ρ-ESS/sec is a GPU benchmark (see the plan).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._logdet._flow_resolvent import FlowKron, flow_logdet_value_and_grad


def _default_logdet_value_and_grad(kron, probes):
    """Resolvent value+grad closure sharing frozen probes across a chain."""

    def _fn(rho_d, rho_o, rho_w):
        return flow_logdet_value_and_grad(kron, rho_d, rho_o, rho_w, probes=probes)

    return _fn


@dataclass
class FlowResolventTarget:
    """ρ-conditional log-posterior and gradient for the unrestricted flow model.

    Parameters
    ----------
    W : array or sparse
        The ``n x n`` (directed, row-standardised) weights matrix.
    y : ndarray (N,)
        Flow observations (``N = n²``).
    X : ndarray (N, k)
        Design matrix.
    logdet_value_and_grad : callable, optional
        ``(rho_d, rho_o, rho_w) -> (value, grad3)`` for ``log|A|``.  Defaults to the
        resolvent estimator with a fixed frozen-probe set (deterministic field).
        Inject an exact implementation for testing.
    rho_bound : float
        Reject ρ with ``|ρ_d|+|ρ_o|+|ρ_w| >= rho_bound`` or ``sum(ρ) >= rho_bound``
        (a conservative row-stochastic stability wall).
    """

    W: object
    y: np.ndarray
    X: np.ndarray
    logdet_value_and_grad: object = None
    rho_bound: float = 0.999
    n_probes: int = 48
    seed: int = 0

    def __post_init__(self):
        self.kron = FlowKron(self.W)
        self.n = self.kron.n
        self.N = self.kron.N
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        self.X = np.asarray(self.X, dtype=np.float64)
        # Precompute the three spatial lags W_k y (exact, O(N)); these make the
        # data term linear in rho.
        Y = self.y.reshape(self.n, self.n)
        Wd_y = (Y @ self.kron.Wt).ravel()  # (I⊗W) y
        Wo_y = (self.kron.W @ Y).ravel()  # (W⊗I) y
        Ww_y = (self.kron.W @ (Y @ self.kron.Wt)).ravel()  # (W⊗W) y
        self.L = np.column_stack([Wd_y, Wo_y, Ww_y])  # (N, 3)
        self.XtX = self.X.T @ self.X
        self.XtX_inv = np.linalg.inv(self.XtX)
        if self.logdet_value_and_grad is None:
            rng = np.random.default_rng(self.seed)
            probes = rng.choice([-1.0, 1.0], size=(self.N, self.n_probes)).astype(
                np.float64
            )
            self.logdet_value_and_grad = _default_logdet_value_and_grad(
                self.kron, probes
            )

    # -- data helpers ------------------------------------------------------
    def Ay(self, rho):
        """``A(ρ) y = y - L ρ`` (linear in ρ)."""
        return self.y - self.L @ np.asarray(rho, dtype=np.float64)

    def in_bounds(self, rho) -> bool:
        rho = np.asarray(rho, dtype=np.float64)
        return bool(np.abs(rho).sum() < self.rho_bound and rho.sum() < self.rho_bound)

    # -- ρ-conditional target ---------------------------------------------
    def logpost_and_grad(self, rho, beta, sigma2):
        """Return ``(logp, grad3)`` of ``log p(ρ | β, σ²)`` (flat prior in-bounds)."""
        rho = np.asarray(rho, dtype=np.float64)
        if not self.in_bounds(rho):
            return -np.inf, np.zeros(3)
        r = self.Ay(rho) - self.X @ beta
        ld_val, ld_grad = self.logdet_value_and_grad(rho[0], rho[1], rho[2])
        logp = ld_val - 0.5 * float(r @ r) / sigma2
        # d/dρ_k [-||r||²/2σ²] = (rᵀ L_k)/σ²  since ∂r/∂ρ_k = -L_k
        grad = np.asarray(ld_grad, dtype=np.float64) + (self.L.T @ r) / sigma2
        return logp, grad

    # -- conjugate Gibbs updates for β, σ² given ρ ------------------------
    def draw_beta_sigma2(self, rho, rng, a0=1e-3, b0=1e-3):
        """Draw ``(β, σ²)`` from their conjugate conditionals given ρ."""
        e = self.Ay(rho)
        bhat = self.XtX_inv @ (self.X.T @ e)
        resid = e - self.X @ bhat
        sse = float(resid @ resid)
        # σ² | ρ  (inverse-gamma with weak prior)
        a_n = a0 + self.N / 2.0
        b_n = b0 + 0.5 * sse
        sigma2 = 1.0 / rng.gamma(a_n, 1.0 / b_n)
        # β | ρ, σ²  (Normal)
        chol = np.linalg.cholesky(sigma2 * self.XtX_inv)
        beta = bhat + chol @ rng.standard_normal(self.X.shape[1])
        return beta, sigma2


def run_flow_resolvent_gibbs(
    target: FlowResolventTarget,
    *,
    draws: int = 1000,
    tune: int = 500,
    step_size: float = 5e-4,
    rho_init=None,
    seed: int = 0,
):
    """MALA-on-ρ within Gibbs for the unrestricted Gaussian flow model.

    Returns a dict of stacked posterior draws for ``rho_d, rho_o, rho_w, beta, sigma``.
    This is the reference driver; for large ``N`` run the GMRES solves on GPU.
    """
    rng = np.random.default_rng(seed)
    rho = np.zeros(3) if rho_init is None else np.asarray(rho_init, dtype=np.float64)
    beta, sigma2 = target.draw_beta_sigma2(rho, rng)
    logp, grad = target.logpost_and_grad(rho, beta, sigma2)

    out = {k: [] for k in ("rho_d", "rho_o", "rho_w", "beta", "sigma")}
    total = tune + draws
    eps = step_size
    for it in range(total):
        # --- MALA proposal on ρ ---
        prop = rho + eps * grad + np.sqrt(2 * eps) * rng.standard_normal(3)
        logp_p, grad_p = target.logpost_and_grad(prop, beta, sigma2)
        if np.isfinite(logp_p):
            q_fwd = -np.sum((prop - rho - eps * grad) ** 2) / (4 * eps)
            q_bwd = -np.sum((rho - prop - eps * grad_p) ** 2) / (4 * eps)
            if np.log(rng.uniform()) < (logp_p - logp) + (q_bwd - q_fwd):
                rho, logp, grad = prop, logp_p, grad_p
        # --- Gibbs β, σ² ---
        beta, sigma2 = target.draw_beta_sigma2(rho, rng)
        logp, grad = target.logpost_and_grad(rho, beta, sigma2)
        if it >= tune:
            out["rho_d"].append(rho[0])
            out["rho_o"].append(rho[1])
            out["rho_w"].append(rho[2])
            out["beta"].append(beta.copy())
            out["sigma"].append(np.sqrt(sigma2))
    return {k: np.asarray(v) for k, v in out.items()}
