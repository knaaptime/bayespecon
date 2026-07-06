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
    assert resolve_logdet_method(None, n=1000) == "chebyshev"
    assert resolve_logdet_method(None, n=10000) == "cheb_stochastic"


def test_resolve_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown logdet method"):
        resolve_logdet_method("bogus", n=100)
