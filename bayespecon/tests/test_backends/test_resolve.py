"""Tests for the probabilistic-backend protocol layer (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest
from libpysal.graph import Graph
from scipy import sparse as sp

from bayespecon._backends import (
    BlackjaxBackend,
    NumPyroBackend,
    ProbabilisticBackend,
    PyMCBackend,
    available_backends,
    resolve_backend,
)
from bayespecon.models import OLS, SAR


def _make_w(n: int = 5) -> Graph:
    rng = np.random.default_rng(0)
    A = (rng.random((n, n)) < 0.5).astype(float)
    np.fill_diagonal(A, 0)
    rs = A.sum(axis=1)
    rs[rs == 0] = 1
    A = A / rs[:, None]
    return Graph.from_sparse(sp.csr_matrix(A))


# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------


def test_resolve_backend_defaults_to_pymc():
    backend = resolve_backend(None)
    assert isinstance(backend, PyMCBackend)
    assert backend.name == "pymc"


@pytest.mark.parametrize("name", ["pymc", "PyMC", "PYMC"])
def test_resolve_backend_pymc_case_insensitive(name):
    assert isinstance(resolve_backend(name), PyMCBackend)


def test_resolve_backend_numpyro_returns_stub():
    backend = resolve_backend("numpyro")
    assert isinstance(backend, NumPyroBackend)


def test_resolve_backend_blackjax_returns_stub():
    backend = resolve_backend("blackjax")
    assert isinstance(backend, BlackjaxBackend)


def test_resolve_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_backend("stan")


def test_resolve_backend_bad_type_raises():
    with pytest.raises(TypeError):
        resolve_backend(42)


def test_resolve_backend_passthrough_for_instance():
    inst = PyMCBackend()
    assert resolve_backend(inst) is inst


def test_available_backends_lists_all():
    assert set(available_backends()) == {"pymc", "numpyro", "blackjax"}


def test_backend_implements_protocol():
    assert isinstance(PyMCBackend(), ProbabilisticBackend)
    assert isinstance(NumPyroBackend(), ProbabilisticBackend)
    assert isinstance(BlackjaxBackend(), ProbabilisticBackend)


# ---------------------------------------------------------------------------
# Stubs raise on use
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [NumPyroBackend, BlackjaxBackend])
def test_stub_backend_methods_raise(cls):
    b = cls()
    with pytest.raises(NotImplementedError):
        b.use_jax_likelihood("pymc")
    with pytest.raises(NotImplementedError):
        b.prepare_sample_kwargs(None, "pymc")
    with pytest.raises(NotImplementedError):
        b.prepare_idata_kwargs(None, None, "pymc")


# ---------------------------------------------------------------------------
# PyMCBackend façade matches _sampler helpers
# ---------------------------------------------------------------------------


def test_pymc_backend_use_jax_likelihood():
    b = PyMCBackend()
    assert b.use_jax_likelihood("numpyro") is True
    assert b.use_jax_likelihood("blackjax") is True
    assert b.use_jax_likelihood("pymc") is False


def test_pymc_backend_prepare_sample_kwargs_non_pymc_unchanged():
    b = PyMCBackend()
    out = b.prepare_sample_kwargs({"draws": 100}, "numpyro")
    assert out == {"draws": 100}


# ---------------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def xy():
    rng = np.random.default_rng(1)
    n = 5
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    return y, X, _make_w(n)


def test_model_default_backend_is_pymc(xy):
    y, X, W = xy
    m = OLS(y=y, X=X, W=W)
    assert m.backend_name == "pymc"
    assert isinstance(m.backend, PyMCBackend)


def test_model_explicit_backend_string(xy):
    y, X, W = xy
    m = SAR(y=y, X=X, W=W, backend="pymc")
    assert isinstance(m.backend, PyMCBackend)


def test_model_unknown_backend_raises(xy):
    y, X, W = xy
    with pytest.raises(ValueError, match="Unknown backend"):
        SAR(y=y, X=X, W=W, backend="stan")
