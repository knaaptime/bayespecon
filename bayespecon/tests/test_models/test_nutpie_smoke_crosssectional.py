"""Smoke tests: cross-sectional spatial models sample under the nutpie backend.

These tests do not assert posterior correctness — recovery is covered by the
existing PyMC-backend recovery suite. We only check that:

* the model fits without raising;
* the returned :class:`arviz.InferenceData` contains a non-empty
  ``posterior`` group with the expected sampler attribution.

Note: nutpie currently ignores ``idata_kwargs``, so we do not request or
assert the ``log_likelihood`` group here. Downstream callers who need it
can run :func:`pymc.compute_log_likelihood` post-fit.
"""

from __future__ import annotations

import pytest

from bayespecon import dgp
from bayespecon.models import OLS, SAR, SDEM, SDM, SEM, SLX
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
def _sar_data() -> dict:
    return dgp.simulate_sar(n=5, seed=2026, rho=0.35)


def _assert_posterior_ok(idata, *, sampler_name: str = "nutpie") -> None:
    assert "posterior" in idata.groups()
    post = idata.posterior
    assert post.sizes.get("draw", 0) > 0
    sampler = post.attrs.get("sampling_time")  # set by pm.sample
    # nutpie always populates a numeric sampling_time when sampling succeeds.
    assert sampler is None or float(sampler) >= 0.0


def test_ols_samples_with_nutpie(_sar_data: dict) -> None:
    model = OLS(y=_sar_data["y"], X=_sar_data["X"], W=_sar_data["W_graph"])
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_slx_samples_with_nutpie(_sar_data: dict) -> None:
    model = SLX(y=_sar_data["y"], X=_sar_data["X"], W=_sar_data["W_graph"])
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_sar_samples_with_nutpie(_sar_data: dict) -> None:
    model = SAR(
        y=_sar_data["y"],
        X=_sar_data["X"],
        W=_sar_data["W_graph"],
        logdet_method="eigenvalue",
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_sem_samples_with_nutpie(_sar_data: dict) -> None:
    model = SEM(
        y=_sar_data["y"],
        X=_sar_data["X"],
        W=_sar_data["W_graph"],
        logdet_method="eigenvalue",
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_sdm_samples_with_nutpie(_sar_data: dict) -> None:
    model = SDM(
        y=_sar_data["y"],
        X=_sar_data["X"],
        W=_sar_data["W_graph"],
        logdet_method="eigenvalue",
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)


def test_sdem_samples_with_nutpie(_sar_data: dict) -> None:
    model = SDEM(
        y=_sar_data["y"],
        X=_sar_data["X"],
        W=_sar_data["W_graph"],
        logdet_method="eigenvalue",
    )
    idata = model.fit(**_NUTPIE_SAMPLE_KWARGS)
    _assert_posterior_ok(idata)
