"""Data-generating processes (DGPs) for BayesPecon models.

This package centralizes simulation routines used in tests, examples, and
benchmarks. All simulators support either a prebuilt spatial weights object
(``libpysal.graph.Graph`` or dense matrix) or a ``geopandas.GeoDataFrame``
from which weights are constructed.
"""

from .cross_sectional import simulate_sar, simulate_sdem, simulate_sdm, simulate_sem, simulate_slx
from .cross_sectional import simulate_ols
from .nonlinear import simulate_sar_tobit, simulate_sdm_tobit, simulate_sem_tobit, simulate_spatial_probit
from .panel_dynamic import simulate_panel_dlm_fe, simulate_panel_sdmr_fe, simulate_panel_sdmu_fe
from .panel_fe import (
    simulate_panel_ols_fe,
    simulate_panel_sar_fe,
    simulate_panel_sdem_fe,
    simulate_panel_sdm_fe,
    simulate_panel_sem_fe,
)
from .panel_re import simulate_panel_ols_re, simulate_panel_sar_re, simulate_panel_sem_re
from .panel_tobit import simulate_panel_sar_tobit_fe, simulate_panel_sem_tobit_fe

__all__ = [
    "simulate_ols",
    "simulate_sar",
    "simulate_sem",
    "simulate_slx",
    "simulate_sdm",
    "simulate_sdem",
    "simulate_sar_tobit",
    "simulate_sem_tobit",
    "simulate_sdm_tobit",
    "simulate_spatial_probit",
    "simulate_panel_ols_fe",
    "simulate_panel_sar_fe",
    "simulate_panel_sem_fe",
    "simulate_panel_sdm_fe",
    "simulate_panel_sdem_fe",
    "simulate_panel_ols_re",
    "simulate_panel_sar_re",
    "simulate_panel_sem_re",
    "simulate_panel_dlm_fe",
    "simulate_panel_sdmr_fe",
    "simulate_panel_sdmu_fe",
    "simulate_panel_sar_tobit_fe",
    "simulate_panel_sem_tobit_fe",
]
