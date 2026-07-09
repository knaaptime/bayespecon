"""Stochastic logdet precomputations must be reproducible run-to-run.

The SLQ and stochastic-Chebyshev precomputes draw random probe vectors;
with no user-supplied Generator they must default to a *seeded* RNG so two
identical builds produce identical logdet approximations.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet._cheb_stochastic import (
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_precompute,
)
from bayespecon._logdet._slq import slq_logdet_precompute

RHO_GRID = np.array([-0.6, -0.2, 0.3, 0.7])


@pytest.fixture(scope="module")
def W():
    n = 400
    rows, cols = [], []
    for i in range(n):
        for off in (1, 2):
            j = (i + off) % n
            rows += [i, j]
            cols += [j, i]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    d = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / d) @ A)


def test_slq_default_rng_reproducible(W):
    pre1 = slq_logdet_precompute(W)
    pre2 = slq_logdet_precompute(W)
    np.testing.assert_array_equal(pre1.nodes, pre2.nodes)
    np.testing.assert_array_equal(pre1.weights, pre2.weights)


def test_slq_custom_seeds_differ(W):
    pre1 = slq_logdet_precompute(W, rng=np.random.default_rng(1))
    pre2 = slq_logdet_precompute(W, rng=np.random.default_rng(2))
    assert not np.array_equal(pre1.nodes, pre2.nodes)


def test_cheb_stochastic_default_rng_reproducible(W):
    pre1 = cheb_stochastic_logdet_precompute(W)
    pre2 = cheb_stochastic_logdet_precompute(W)
    vals1 = np.array([cheb_stochastic_logdet_eval(pre1, float(r)) for r in RHO_GRID])
    vals2 = np.array([cheb_stochastic_logdet_eval(pre2, float(r)) for r in RHO_GRID])
    np.testing.assert_array_equal(vals1, vals2)


def test_cheb_stochastic_custom_seeds_differ(W):
    pre1 = cheb_stochastic_logdet_precompute(W, rng=np.random.default_rng(1))
    pre2 = cheb_stochastic_logdet_precompute(W, rng=np.random.default_rng(2))
    vals1 = np.array([cheb_stochastic_logdet_eval(pre1, float(r)) for r in RHO_GRID])
    vals2 = np.array([cheb_stochastic_logdet_eval(pre2, float(r)) for r in RHO_GRID])
    assert not np.array_equal(vals1, vals2)
