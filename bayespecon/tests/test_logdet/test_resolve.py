"""Tests for the LogDetMethod enum and resolve_logdet_method validator."""

import pytest

from bayespecon.logdet import (
    VALID_LOGDET_METHODS,
    LogDetMethod,
    resolve_logdet_method,
)

CANONICAL_METHODS = {
    "exact",
    "eigenvalue",
    "grid_dense",
    "grid_sparse",
    "sparse_spline",
    "grid_mc",
    "trace_mc",
    "grid_ilu",
    "chebyshev",
}


def test_enum_members_match_canonical_names():
    assert {m.value for m in LogDetMethod} == CANONICAL_METHODS


def test_valid_logdet_methods_constant():
    assert VALID_LOGDET_METHODS == CANONICAL_METHODS


def test_enum_str_equality():
    # str-Enum mixin: members are equal to their string values.
    assert LogDetMethod.EIGENVALUE == "eigenvalue"
    assert LogDetMethod.CHEBYSHEV == "chebyshev"


@pytest.mark.parametrize("name", sorted(CANONICAL_METHODS))
def test_resolve_accepts_canonical_names(name):
    assert resolve_logdet_method(name, n=100) == name


def test_resolve_unknown_method_raises_with_list():
    with pytest.raises(ValueError, match="Unknown logdet method"):
        resolve_logdet_method("bogus", n=100)


def test_resolve_unknown_method_error_lists_valid_options():
    try:
        resolve_logdet_method("bogus", n=100)
    except ValueError as exc:
        msg = str(exc)
        for name in CANONICAL_METHODS:
            assert name in msg
    else:
        pytest.fail("Expected ValueError")


def test_resolve_none_small_n_auto_selects_eigenvalue():
    # Default cutoff is 500 — small n picks eigenvalue.
    assert resolve_logdet_method(None, n=100) == "eigenvalue"


def test_resolve_none_large_n_auto_selects_chebyshev():
    # Above the default cutoff, auto-selection switches to chebyshev.
    assert resolve_logdet_method(None, n=10_000) == "chebyshev"


def test_resolve_none_cutoff_env(monkeypatch):
    monkeypatch.setenv("BAYESPECON_LOGDET_EIGEN_MAX_N", "50")
    assert resolve_logdet_method(None, n=100) == "chebyshev"
    assert resolve_logdet_method(None, n=10) == "eigenvalue"
