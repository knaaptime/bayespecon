"""Tests for the Phase 5 ``recommend_backend`` helper.

The recommendation table is grounded in
``scripts/benchmarks/results/nutpie_decision.csv``; these tests lock in the
public surface and the conservative fallback behavior so future benchmark
updates require an explicit, reviewable change to the policy table.
"""

from __future__ import annotations

import pytest

from bayespecon._backends import (
    _RECOMMENDATIONS,
    available_backends,
    recommend_backend,
)


class TestRecommendBackend:
    def test_known_families_resolve_to_available_backends(self):
        for family, (backend, _) in _RECOMMENDATIONS.items():
            assert backend in available_backends(), (
                f"{family} recommends unknown backend {backend!r}"
            )

    def test_sar_recommends_nutpie(self):
        # Benchmark evidence: nutpie wins ESS/sec on cross-sectional SAR.
        assert recommend_backend("SAR") == "nutpie"

    @pytest.mark.parametrize("family", ["SEM", "SDM", "PoissonSARFlow"])
    def test_conservative_families_stay_on_pymc(self, family):
        assert recommend_backend(family) == "pymc"

    def test_unknown_family_falls_back_to_pymc(self):
        assert recommend_backend("DoesNotExist") == "pymc"

    def test_case_insensitive_lookup(self):
        assert recommend_backend("sar") == recommend_backend("SAR")
        assert recommend_backend("PoissonSarFlow") == recommend_backend(
            "PoissonSARFlow"
        )

    def test_with_rationale_returns_tuple(self):
        backend, rationale = recommend_backend("SAR", with_rationale=True)
        assert backend == "nutpie"
        assert isinstance(rationale, str)
        assert rationale  # non-empty

    def test_unknown_family_rationale_is_informative(self):
        backend, rationale = recommend_backend(
            "MysteryModel", with_rationale=True
        )
        assert backend == "pymc"
        assert "MysteryModel" in rationale

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="model_family must be a string"):
            recommend_backend(42)  # type: ignore[arg-type]
