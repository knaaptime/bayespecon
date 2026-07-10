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
from bayespecon.models.panel._fe import (
    OLSPanelFE,
    SARPanelFE,
    SDEMPanelFE,
    SDMPanelFE,
    SEMPanelFE,
    SLXPanelFE,
)
from bayespecon.tests.helpers import (
    PANEL_N,
    PANEL_T,
    W_to_graph,
    make_line_W,
    make_panel_ols_data,
    make_panel_sar_data,
    make_panel_sdem_fe_data,
    make_panel_sdm_fe_data,
    make_panel_sem_data,
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


# ---------------------------------------------------------------------------
# Gaussian panel fixed-effects family
# ---------------------------------------------------------------------------

_W_PANEL = make_line_W(PANEL_N)


def _panel_graph():
    return W_to_graph(_W_PANEL)


# name -> (ctor, data_fn, expected_varnames, samplers)
GAUSSIAN_PANEL_FE: dict[str, tuple] = {
    "SAR": (
        SARPanelFE,
        make_panel_sar_data,
        {"beta", "rho", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "SEM": (
        SEMPanelFE,
        make_panel_sem_data,
        {"beta", "lam", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "SDM": (
        SDMPanelFE,
        make_panel_sdm_fe_data,
        {"beta", "rho", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "SDEM": (
        SDEMPanelFE,
        make_panel_sdem_fe_data,
        {"beta", "lam", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "OLS": (OLSPanelFE, make_panel_ols_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
    # SLX has no closed-form panel DGP helper; the SDM generator supplies WX
    # signal and SLX ignores the ρ lag.
    "SLX": (SLXPanelFE, make_panel_sdm_fe_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
}


def _panel_fe_cases():
    for name, (ctor, data_fn, expected, samplers) in GAUSSIAN_PANEL_FE.items():
        for sampler in samplers:
            yield pytest.param(
                name, ctor, data_fn, expected, sampler, id=f"{name}-{sampler}"
            )


@pytest.mark.parametrize("name,ctor,data_fn,expected,sampler", list(_panel_fe_cases()))
def test_gaussian_panel_fe_fit_contract(name, ctor, data_fn, expected, sampler):
    y, X, _ = data_fn(np.random.default_rng(1), _W_PANEL, PANEL_N, PANEL_T)
    model = ctor(y=y, X=X, W=_panel_graph(), N=PANEL_N, T=PANEL_T, effects=1)
    kwargs = dict(
        sampler=sampler, draws=6, tune=6, chains=1, progressbar=False, random_seed=1
    )
    if sampler == "gibbs":
        kwargs["n_jobs"] = 1
    idata = model.fit(**kwargs)
    varnames = set(idata.posterior.data_vars)
    has_ll = "log_likelihood" in idata.groups()
    assert varnames == expected, f"{name} [{sampler}]: {sorted(varnames)}"
    if sampler == "gibbs":
        assert has_ll, f"{name} gibbs should attach a log_likelihood group"
