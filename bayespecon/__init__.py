"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes, residual diagnostics, and MATLAB-style test/statistics helpers.

Examples
--------
Import a model class directly from the package namespace::

	from bayespecon import SAR

Import MATLAB-style statistics helpers from the stats submodule::

	from bayespecon import stats
"""

from . import dgp, stats
from .diagnostics import (
    DiagnosticResult,
    arch_test,
    bpagan_test,
	hausman_fe_re_test,
    ljung_box_q,
    panel_residual_structure,
    pesaran_cd_test,
    rdiagnose_like,
)
from .models import (
    SAR,
		OLS,
    SDEM,
    SDM,
    SEM,
	SARTobit,
	SDMTobit,
	SEMTobit,
    SLX,
    OLSPanelFE,
    SARPanelFE,
    SDEMPanelFE,
    SDMPanelFE,
    SEMPanelFE,
	DLMPanelFE,
	SDMRPanelFE,
	SDMUPanelFE,
    OLSPanelRE,
    SARPanelRE,
	SARPanelTobit,
    SEMPanelRE,
	SEMPanelTobit,
    SpatialProbit,
)

__all__ = [
	"SLX",
		"OLS",
	"SAR",
	"SEM",
	"SDM",
	"SDEM",
	"SARTobit",
	"SEMTobit",
	"SDMTobit",
	"SpatialProbit",
	"OLSPanelFE",
	"SARPanelFE",
	"SEMPanelFE",
	"SDMPanelFE",
	"SDEMPanelFE",
	"DLMPanelFE",
	"SDMRPanelFE",
	"SDMUPanelFE",
	"OLSPanelRE",
	"SARPanelRE",
	"SEMPanelRE",
	"SARPanelTobit",
	"SEMPanelTobit",
	"rdiagnose_like",
	"bpagan_test",
	"arch_test",
	"hausman_fe_re_test",
	"ljung_box_q",
	"panel_residual_structure",
	"pesaran_cd_test",
	"stats",
	"dgp",
]


import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespecon")
