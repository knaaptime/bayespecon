"""Unit tests for the nutpie entry in the backend registry.

These tests exercise the backend dispatch surface without sampling.
They run whenever ``nutpie`` is importable; otherwise they are skipped
via :data:`bayespecon.tests.helpers.requires_nutpie`.
"""

from __future__ import annotations

import pytest

from bayespecon._backends import (
    NutpieBackend,
    PyMCBackend,
    available_backends,
    resolve_backend,
)
from bayespecon.tests.helpers import requires_nutpie

pytestmark = requires_nutpie


def test_nutpie_listed_in_available_backends() -> None:
    assert "nutpie" in available_backends()


def test_resolve_backend_returns_nutpie_instance() -> None:
    backend = resolve_backend("nutpie")
    assert isinstance(backend, NutpieBackend)
    assert isinstance(backend, PyMCBackend)
    assert backend.name == "nutpie"


def test_nutpie_resolves_nuts_sampler_to_nutpie() -> None:
    backend = resolve_backend("nutpie")
    assert backend.resolve_nuts_sampler(None) == "nutpie"
    assert backend.resolve_nuts_sampler("nutpie") == "nutpie"


@pytest.mark.parametrize("bad", ["pymc", "numpyro", "blackjax"])
def test_nutpie_rejects_mismatched_nuts_sampler(bad: str) -> None:
    backend = resolve_backend("nutpie")
    with pytest.raises(ValueError, match="nutpie"):
        backend.resolve_nuts_sampler(bad)


def test_nutpie_does_not_use_jax_likelihood() -> None:
    backend = resolve_backend("nutpie")
    assert backend.use_jax_likelihood("nutpie") is False
