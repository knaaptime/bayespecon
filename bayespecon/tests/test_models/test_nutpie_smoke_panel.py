"""Smoke tests: panel RE spatial models sample under the nutpie backend."""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import OLSPanelRE, SARPanelRE, SEMPanelRE
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


def test_ols_panel_re_samples_with_nutpie(
    rng: np.random.Generator, W_panel_dense, W_panel_graph
) -> None:
    y, X, _ = make_panel_ols_data(rng, W_panel_dense, PANEL_N, PANEL_T)
    model = OLSPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_sar_panel_re_samples_with_nutpie(
    rng: np.random.Generator, W_panel_dense, W_panel_graph
) -> None:
    y, X, _ = make_panel_sar_data(rng, W_panel_dense, PANEL_N, PANEL_T)
    model = SARPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_sem_panel_re_samples_with_nutpie(
    rng: np.random.Generator, W_panel_dense, W_panel_graph
) -> None:
    y, X, _ = make_panel_sem_data(rng, W_panel_dense, PANEL_N, PANEL_T)
    model = SEMPanelRE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)
