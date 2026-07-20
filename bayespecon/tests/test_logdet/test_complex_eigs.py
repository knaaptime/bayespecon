"""Regression tests: complex eigenvalues must survive the logdet factories.

Row-standardised directed weights (KNN, travel time, migration flows) have a
complex spectrum.  ``make_logdet_fn`` used to narrow a 1-D eigenvalue array to
``float64``, silently discarding the imaginary parts and biasing
``log|I - rho W|`` — while ``make_logdet_numpy_fn`` kept them, so the PyTensor
(NUTS) and NumPy (Gibbs) paths disagreed on the same model.
"""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.sparse as sp

from bayespecon._logdet import make_logdet_fn, make_logdet_numpy_fn


def _directed_ring(n: int) -> sp.csr_matrix:
    """Row-standardised directed ring: a permutation matrix, spectrum on |z|=1."""
    W = sp.lil_matrix((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = 1.0
    return sp.csr_matrix(W)


def _knn_like(n: int, k: int, seed: int = 3) -> sp.csr_matrix:
    """Row-standardised k-nearest-neighbour graph — non-symmetric, complex spectrum."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    W = sp.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
    return sp.csr_matrix(sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W)


def _exact_logdet(W: sp.csr_matrix, rho: float) -> float:
    n = W.shape[0]
    return float(np.linalg.slogdet(np.eye(n) - rho * W.toarray())[1])


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
def test_pytensor_eigenvalue_path_keeps_imaginary_parts(rho: float) -> None:
    """The PyTensor evaluator must match the exact logdet for a complex spectrum."""
    W = _directed_ring(60)
    eigs = np.linalg.eigvals(W.toarray())
    assert np.abs(eigs.imag).max() > 0.5, "fixture should have a complex spectrum"

    rho_s = pt.dscalar("rho")
    fn = pytensor.function([rho_s], make_logdet_fn(eigs, method="eigenvalue")(rho_s))
    assert float(fn(rho)) == pytest.approx(_exact_logdet(W, rho), abs=1e-8)


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_pytensor_and_numpy_paths_agree_on_directed_graph(rho: float) -> None:
    """Gibbs (NumPy) and NUTS (PyTensor) must not disagree on the same weights."""
    W = _knn_like(200, k=6)
    eigs = np.linalg.eigvals(W.toarray())
    assert np.abs(eigs.imag).max() > 0.0

    rho_s = pt.dscalar("rho")
    pt_fn = pytensor.function([rho_s], make_logdet_fn(eigs, method="eigenvalue")(rho_s))
    np_fn = make_logdet_numpy_fn(W, eigs, method="eigenvalue")

    exact = _exact_logdet(W, rho)
    assert float(pt_fn(rho)) == pytest.approx(exact, abs=1e-8)
    assert np_fn(rho) == pytest.approx(exact, abs=1e-8)


def test_eigenvalue_array_dtype_is_not_narrowed() -> None:
    """A complex eigenvalue array must not be cast to float on the way in."""
    W = _directed_ring(40)
    eigs = np.linalg.eigvals(W.toarray())
    assert np.iscomplexobj(eigs)

    # Real-cast eigenvalues give a materially different answer; if the factory
    # narrowed its input the two would coincide.
    rho = 0.7
    rho_s = pt.dscalar("rho")
    fn = pytensor.function([rho_s], make_logdet_fn(eigs, method="eigenvalue")(rho_s))
    discarded = float(np.sum(np.log(np.abs(1.0 - rho * eigs.real))))
    assert not np.isclose(float(fn(rho)), discarded)
