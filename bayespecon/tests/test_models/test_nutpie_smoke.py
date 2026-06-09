"""Smoke tests: models sample successfully under the nutpie backend.

These tests do not assert posterior correctness — recovery is covered by the
existing PyMC-backend recovery suite. We only check that:

* the model fits without raising;
* the returned :class:`arviz.InferenceData` contains a non-empty
  ``posterior`` group.

Note: nutpie currently ignores ``idata_kwargs``, so we do not request or
assert the ``log_likelihood`` group here. Downstream callers who need it
can run :func:`pymc.compute_log_likelihood` post-fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    OLSPanelRE,
    SARPanelRE,
    SEMPanelRE,
    dgp,
)
from bayespecon.models.flow._flow import (
    NegativeBinomialFlow,
    NegativeBinomialSARFlow,
    NegativeBinomialSARFlowSeparable,
)
from bayespecon.tests.helpers import (
    PANEL_N,
    PANEL_T,
    make_panel_ols_data,
    make_panel_sar_data,
    make_panel_sem_data,
    requires_nutpie,
)

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


def _assert_posterior_ok(idata) -> None:
    assert "posterior" in idata.groups()
    assert idata.posterior.sizes.get("draw", 0) > 0


# ---------------------------------------------------------------------------
# Cross-sectional spatial models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _sar_data() -> dict:
    return dgp.simulate_sar(n=5, seed=2026, rho=0.35)


def _make_cross(model_cls, data):
    kwargs = dict(y=data["y"], X=data["X"], W=data["W_graph"])
    if model_cls in (SAR, SEM, SDM, SDEM):
        kwargs["logdet_method"] = "eigenvalue"
    return model_cls(**kwargs)


@pytest.mark.parametrize("model_cls", [OLS, SLX, SAR, SEM, SDM, SDEM])
def test_crosssectional_samples_with_nutpie(model_cls, _sar_data: dict) -> None:
    model = _make_cross(model_cls, _sar_data)
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


# ---------------------------------------------------------------------------
# Panel RE spatial models
# ---------------------------------------------------------------------------


_PANEL_CASES = [
    (OLSPanelRE, make_panel_ols_data),
    (SARPanelRE, make_panel_sar_data),
    (SEMPanelRE, make_panel_sem_data),
]


@pytest.mark.parametrize(
    "model_cls,make_data", _PANEL_CASES, ids=[c[0].__name__ for c in _PANEL_CASES]
)
def test_panel_re_samples_with_nutpie(
    model_cls, make_data, rng: np.random.Generator, W_panel_dense, W_panel_graph
) -> None:
    y, X, _ = make_data(rng, W_panel_dense, PANEL_N, PANEL_T)
    model = model_cls(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


# ---------------------------------------------------------------------------
# Flow models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _flow_data() -> dict:
    return dgp.generate_negbin_flow_data(n=8, seed=2026)


@pytest.mark.parametrize(
    "model_cls",
    [NegativeBinomialFlow, NegativeBinomialSARFlowSeparable, NegativeBinomialSARFlow],
)
def test_negbin_flow_samples_with_nutpie(model_cls, _flow_data: dict) -> None:
    model = model_cls(
        y=_flow_data["y_vec"],
        G=_flow_data["G"],
        X=_flow_data["X"],
        col_names=_flow_data["col_names"],
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)
