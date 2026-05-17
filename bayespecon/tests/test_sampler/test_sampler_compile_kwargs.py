"""Unit tests for ``prepare_compile_kwargs``."""

from __future__ import annotations

import warnings

import pytest

from bayespecon.models import _sampler
from bayespecon.models._sampler import prepare_compile_kwargs


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset module-level caches between tests."""
    _sampler._has_module.cache_clear()
    _sampler._warn_numba_missing_once.cache_clear()
    yield
    _sampler._has_module.cache_clear()
    _sampler._warn_numba_missing_once.cache_clear()


def test_non_pymc_sampler_passthrough():
    # JAX-backed samplers ("blackjax", "numpyro") get a default
    # chain_method="vectorized" so multiple chains run in one
    # XLA-compiled batched call. They still must not receive
    # compile_kwargs (only the "pymc" sampler uses it).
    out = prepare_compile_kwargs({"random_seed": 0}, "blackjax")
    assert out == {
        "random_seed": 0,
        "nuts_sampler_kwargs": {"chain_method": "vectorized"},
    }
    assert "compile_kwargs" not in out


def test_jax_sampler_chain_method_user_override():
    user = {"nuts_sampler_kwargs": {"chain_method": "parallel"}}
    out = prepare_compile_kwargs(user, "numpyro")
    assert out["nuts_sampler_kwargs"] == {"chain_method": "parallel"}


def test_nutpie_sampler_untouched():
    out = prepare_compile_kwargs({"random_seed": 0}, "nutpie")
    assert out == {"random_seed": 0}


def test_pymc_with_numba_injects_numba_mode(monkeypatch):
    monkeypatch.setattr(
        _sampler, "_has_module", lambda name: True if name == "numba" else False
    )
    out = prepare_compile_kwargs({}, "pymc")
    assert out == {"compile_kwargs": {"mode": "NUMBA"}}


def test_pymc_user_override_wins(monkeypatch):
    monkeypatch.setattr(_sampler, "_has_module", lambda name: True)
    user = {"compile_kwargs": {"mode": "FAST_RUN"}}
    out = prepare_compile_kwargs(user, "pymc")
    assert out == {"compile_kwargs": {"mode": "FAST_RUN"}}


def test_pymc_user_empty_compile_kwargs_respected(monkeypatch):
    monkeypatch.setattr(_sampler, "_has_module", lambda name: True)
    out = prepare_compile_kwargs({"compile_kwargs": {}}, "pymc")
    assert out == {"compile_kwargs": {}}


def test_pymc_without_numba_warns_and_passthrough(monkeypatch):
    monkeypatch.setattr(_sampler, "_has_module", lambda name: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prepare_compile_kwargs({"random_seed": 1}, "pymc")
    assert out == {"random_seed": 1}
    assert any("numba is not installed" in str(w.message) for w in caught)


def test_pymc_warning_emitted_once(monkeypatch):
    monkeypatch.setattr(_sampler, "_has_module", lambda name: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_compile_kwargs({}, "pymc")
        prepare_compile_kwargs({}, "pymc")
    matches = [w for w in caught if "numba is not installed" in str(w.message)]
    assert len(matches) == 1


def test_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(_sampler, "_has_module", lambda name: True)
    original = {"random_seed": 7}
    out = prepare_compile_kwargs(original, "pymc")
    assert original == {"random_seed": 7}
    assert out is not original


def test_none_sample_kwargs(monkeypatch):
    monkeypatch.setattr(_sampler, "_has_module", lambda name: True)
    out = prepare_compile_kwargs(None, "pymc")
    assert out == {"compile_kwargs": {"mode": "NUMBA"}}
