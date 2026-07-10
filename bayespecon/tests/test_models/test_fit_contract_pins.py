"""Phase 5b migration contract pins: posterior var-names per model × sampler.

These lock the *structural* ``fit()`` contract — the exact set of posterior
variables each model produces under each sampler, and that Gibbs attaches a
``log_likelihood`` group — on the **current** code, so the registry / one-``fit()``
migration must reproduce it exactly.  They assert structure, not draws, so they
are insensitive to RNG-order changes (the roadmap's rule).

Families are added to the tables as each migration wave begins; the existing
recovery/model suites remain the correctness (parameter-recovery) pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon.models import OLS, SAR, SDEM, SDM, SEM, SLX
from bayespecon.tests.helpers import (
    W_to_graph,
    make_rook_W,
    make_sar_data,
    make_sdem_data,
    make_sdm_data,
    make_sem_data,
    make_slx_data,
)

_W_DENSE = make_rook_W(6)  # n = 36


def _graph():
    return W_to_graph(_W_DENSE)


def _fit_varnames(model, sampler):
    idata = model.fit(
        sampler=sampler, draws=6, tune=6, chains=1, progressbar=False, random_seed=1
    )
    return set(idata.posterior.data_vars), ("log_likelihood" in idata.groups())


# name -> (ctor, data_fn, expected_varnames, samplers, gibbs_has_ll)
GAUSSIAN_XS: dict[str, tuple] = {
    "SAR": (SAR, make_sar_data, {"beta", "rho", "sigma", "sigma2"}, ("gibbs", "nuts")),
    "SEM": (SEM, make_sem_data, {"beta", "lam", "sigma", "sigma2"}, ("gibbs", "nuts")),
    "SDM": (SDM, make_sdm_data, {"beta", "rho", "sigma", "sigma2"}, ("gibbs", "nuts")),
    "SDEM": (
        SDEM,
        make_sdem_data,
        {"beta", "lam", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "OLS": (OLS, make_sar_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
    "SLX": (SLX, make_slx_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
}


def _gaussian_cases():
    for name, (ctor, data_fn, expected, samplers) in GAUSSIAN_XS.items():
        for sampler in samplers:
            yield pytest.param(
                name, ctor, data_fn, expected, sampler, id=f"{name}-{sampler}"
            )


@pytest.mark.parametrize("name,ctor,data_fn,expected,sampler", list(_gaussian_cases()))
def test_gaussian_xs_fit_contract(name, ctor, data_fn, expected, sampler):
    rng = np.random.default_rng(0)
    y, X = data_fn(rng, _W_DENSE)
    model = ctor(y=y, X=X, W=_graph())
    varnames, has_ll = _fit_varnames(model, sampler)
    assert varnames == expected, f"{name} [{sampler}]: {sorted(varnames)}"
    if sampler == "gibbs":
        assert has_ll, f"{name} gibbs should attach a log_likelihood group"
