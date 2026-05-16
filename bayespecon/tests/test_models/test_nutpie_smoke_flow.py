"""Smoke tests: Poisson flow models sample under the nutpie backend.

The Phase 0 spike confirmed that ``PoissonSARFlow`` (the highest-risk model
because of its custom Kronecker / sparse-flow Ops) samples successfully via
nutpie, even though those Ops fall back to Numba object mode. We therefore
treat all three flow models as expected-pass smoke tests.
"""

from __future__ import annotations

import pytest

from bayespecon import dgp
from bayespecon.models.flow import (
    PoissonFlow,
    PoissonSARFlow,
    PoissonSARFlowSeparable,
)
from bayespecon.tests.helpers import requires_nutpie

pytestmark = [requires_nutpie, pytest.mark.slow]

_NUTPIE_SAMPLE_KWARGS = dict(
    draws=100,
    tune=100,
    chains=1,
    random_seed=2026,
    nuts_sampler="nutpie",
    progressbar=False,
    compute_convergence_checks=False,
)


@pytest.fixture(scope="module")
def _flow_data() -> dict:
    return dgp.generate_poisson_flow_data(n=8, seed=2026)


def _assert_posterior_ok(idata) -> None:
    assert "posterior" in idata.groups()
    assert idata.posterior.sizes.get("draw", 0) > 0


def test_poisson_flow_samples(_flow_data: dict) -> None:
    model = PoissonFlow(
        y=_flow_data["y_vec"],
        G=_flow_data["G"],
        X=_flow_data["X"],
        col_names=_flow_data["col_names"],
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_poisson_sar_flow_separable_samples(_flow_data: dict) -> None:
    model = PoissonSARFlowSeparable(
        y=_flow_data["y_vec"],
        G=_flow_data["G"],
        X=_flow_data["X"],
        col_names=_flow_data["col_names"],
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_poisson_sar_flow_samples(_flow_data: dict) -> None:
    # Phase 0 spike: this model samples cleanly under nutpie even though the
    # KroneckerFlowSolveOp / SparseFlowSolveOp dispatches fall back to Numba
    # object mode. If that ever regresses, mark xfail with a clear reason.
    model = PoissonSARFlow(
        y=_flow_data["y_vec"],
        G=_flow_data["G"],
        X=_flow_data["X"],
        col_names=_flow_data["col_names"],
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)
