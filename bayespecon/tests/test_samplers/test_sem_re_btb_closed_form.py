"""The SEM-RE α-block closed form must equal the explicit BᵀB it replaces.

``_sample_alpha_re`` (and ``_sem_re_marginalized_log_density``) previously
rebuilt a dense ``NT × N`` matrix ``B = A D`` (``A = I - λW``) every Gibbs
sweep and formed ``BᵀB``.  That is now computed as

    BᵀB(λ) = T·I_N − λ(M1 + M1ᵀ) + λ² M2,   M1 = DᵀWD, M2 = DᵀWᵀWD

from λ-independent terms precomputed once.  This pins the algebraic identity:
the closed form must reproduce the explicit ``B.T @ B`` to machine precision
across the λ grid (so the posterior is provably unchanged).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon.samplers.panel._re_core import (
    _sem_re_BtB,
    _sem_re_unit_aggregated_terms,
)

LAM_GRID = [-0.5, -0.2, 0.0, 0.3, 0.8]


def _row_std_ring(n: int) -> sp.csr_matrix:
    row = np.concatenate([np.arange(n), np.arange(n)])
    col = np.concatenate([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n])
    W = sp.csr_matrix((np.ones(2 * n), (row, col)), shape=(n, n))
    d = np.asarray(W.sum(axis=1)).ravel()
    return (sp.diags(1.0 / d) @ W).tocsr()


def _block_diag_W(W: sp.csr_matrix, T: int) -> sp.csr_matrix:
    """NT × NT block-diagonal spatial weights (same W each period)."""
    return sp.block_diag([W] * T, format="csr")


def _explicit_BtB(lam: float, W_nt: sp.csr_matrix, unit_idx: np.ndarray, N: int):
    """Reference: build dense B = (I - λW) D explicitly and form BᵀB."""
    n = len(unit_idx)
    B = np.zeros((n, N))
    for i in range(N):
        e_i = (unit_idx == i).astype(np.float64)
        B[:, i] = e_i - lam * (W_nt @ e_i)
    return B.T @ B


@pytest.mark.parametrize("N,T", [(6, 4), (10, 3)])
def test_closed_form_matches_explicit_BtB(N, T):
    W = _row_std_ring(N)
    W_nt = _block_diag_W(W, T)
    # unit_idx for a balanced panel stacked period-by-period: obs -> unit
    unit_idx = np.tile(np.arange(N), T)

    M1, M2 = _sem_re_unit_aggregated_terms(W_nt, unit_idx, N)
    for lam in LAM_GRID:
        closed = _sem_re_BtB(lam, M1, M2, T, N)
        explicit = _explicit_BtB(lam, W_nt, unit_idx, N)
        np.testing.assert_allclose(closed, explicit, atol=1e-10, rtol=0)
