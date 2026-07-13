"""Tests for the LogDetMethod enum and resolve_logdet_method validator."""

import pytest

from bayespecon._logdet import (
    VALID_LOGDET_METHODS,
    LogDetMethod,
    resolve_logdet_method,
)

CANONICAL_METHODS = {
    "eigenvalue",
    "slq",
    "chebyshev",
    "cheb_stochastic",
    "cheb_cholesky",
    "aaa",
    "traces",
}


def test_enum_members_match_canonical_names():
    assert {m.value for m in LogDetMethod} == CANONICAL_METHODS


def test_valid_logdet_methods_constant():
    assert VALID_LOGDET_METHODS == CANONICAL_METHODS


def test_enum_str_equality():
    assert LogDetMethod.EIGENVALUE == "eigenvalue"
    assert LogDetMethod.SLQ == "slq"
    assert LogDetMethod.CHEBYSHEV == "chebyshev"
    assert LogDetMethod.TRACES == "traces"


@pytest.mark.parametrize("name", sorted(CANONICAL_METHODS))
def test_resolve_accepts_canonical_names(name):
    assert resolve_logdet_method(name, n=100) == name


def test_resolve_none_auto_selects():
    assert resolve_logdet_method(None, n=100) == "eigenvalue"
    assert resolve_logdet_method(None, n=1000) == "cheb_cholesky"
    assert resolve_logdet_method(None, n=10000) == "cheb_cholesky"
    assert resolve_logdet_method(None, n=50000) == "cheb_stochastic"


def test_resolve_none_auto_selects_nonsymmetric_W():
    """Non-symmetric W in medium range should auto-select aaa."""
    import numpy as np
    import scipy.sparse as sp

    # Build a non-symmetric sparse W (directed graph)
    n = 1000
    np.random.seed(42)
    rows = np.array([0, 0, 1, 1, 2, 3, 3, 4])
    cols = np.array([1, 2, 0, 3, 4, 1, 4, 2])
    vals = np.ones(len(rows))
    W_nonsym = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    assert resolve_logdet_method(None, n=n, W=W_nonsym) == "aaa"

    # Symmetric W should still select cheb_cholesky
    W_sym = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W_sym = W_sym + W_sym.T
    assert resolve_logdet_method(None, n=n, W=W_sym) == "cheb_cholesky"

    # No W provided defaults to cheb_cholesky
    assert resolve_logdet_method(None, n=n) == "cheb_cholesky"


def test_resolve_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown logdet method"):
        resolve_logdet_method("bogus", n=100)
