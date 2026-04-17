"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes, residual diagnostics, and MATLAB-style test/statistics helpers.

Examples
--------
Import a model class directly from the package namespace::

	from bayespreg import SAR

Import MATLAB-style statistics helpers from the stats submodule::

	from bayespreg import stats
"""

from . import stats
from .diagnostics import (
    arch_test,
    bpagan_test,
    ljung_box_q,
    panel_residual_structure,
    pesaran_cd_test,
    rdiagnose_like,
)
from .models import (
    SAR,
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
    OLSPanelRE,
    SARPanelRE,
	SARPanelTobit,
    SEMPanelRE,
	SEMPanelTobit,
    SpatialProbit,
)

__all__ = [
	"SLX",
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
	"OLSPanelRE",
	"SARPanelRE",
	"SEMPanelRE",
	"SARPanelTobit",
	"SEMPanelTobit",
	"rdiagnose_like",
	"bpagan_test",
	"arch_test",
	"ljung_box_q",
	"panel_residual_structure",
	"pesaran_cd_test",
	"stats",
]


import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespreg")
