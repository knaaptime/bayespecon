"""Regression tests: chol_cheb must be exact for row-standardized *weighted* undirected graphs.

``_d_symmetrize`` recovers the symmetrizing degree; using the neighbor count
(``getnnz``) is only correct for binary adjacency. These tests pin the weighted
case (kernel-like weights on a symmetric pattern), where the wrong degree makes
``W_sym`` non-symmetric and CHOLMOD silently returns a wrong log-determinant.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet._chol_cheb import (
    _d_symmetrize,
    chol_cheb_logdet_eval_vec,
    chol_cheb_logdet_precompute,
)

RHO_GRID = np.array([-0.7, -0.3, 0.0, 0.3, 0.6, 0.85])


def _ring_lattice_adjacency(n: int, weighted: bool, seed: int = 42) -> sp.csr_matrix:
    """Symmetric ring-lattice adjacency (links to i±1, i±2), optionally weighted."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        for off in (1, 2):
            j = (i + off) % n
            w = rng.uniform(0.5, 2.0) if weighted else 1.0
            rows += [i, j]
            cols += [j, i]
            vals += [w, w]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    assert abs(A - A.T).max() < 1e-14
    return A


def _row_standardize(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / d) @ A)


@pytest.mark.parametrize("weighted", [False, True], ids=["binary", "weighted"])
def test_d_symmetrize_output_is_symmetric(weighted):
    W = _row_standardize(_ring_lattice_adjacency(300, weighted=weighted))
    W_sym = _d_symmetrize(W)
    assert abs(W_sym - W_sym.T).max() < 1e-10


@pytest.mark.parametrize("weighted", [False, True], ids=["binary", "weighted"])
def test_chol_cheb_matches_dense_slogdet(weighted):
    n = 300
    W = _row_standardize(_ring_lattice_adjacency(n, weighted=weighted))
    pre = chol_cheb_logdet_precompute(W, rho_min=-0.9, rho_max=0.9)
    approx = chol_cheb_logdet_eval_vec(pre, RHO_GRID)

    W_dense = W.toarray()
    eye = np.eye(n)
    exact = np.array([np.linalg.slogdet(eye - rho * W_dense)[1] for rho in RHO_GRID])
    np.testing.assert_allclose(approx, exact, atol=1e-5, rtol=0)
