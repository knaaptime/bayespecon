"""Panel censored (Tobit-style) DGP functions.

Simulates left-censored panel data from SAR and SEM spatial Tobit models
with fixed-effects transformations.
"""

from __future__ import annotations

import numpy as np

from .panel_fe import simulate_panel_sar_fe, simulate_panel_sem_fe
from .utils import make_panel_output_geodataframe


def _left_censor(y_latent: np.ndarray, censoring: float) -> tuple[np.ndarray, np.ndarray]:
    mask = y_latent <= censoring
    y_obs = y_latent.copy()
    y_obs[mask] = censoring
    return y_obs, mask


def simulate_panel_sar_tobit_fe(*, censoring: float = 0.0, **kwargs) -> dict:
    """Simulate left-censored panel SAR FE data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c`` where observed ``y = max(c, y*)``.
    **kwargs
        Forwarded to :func:`bayespecon.dgp.panel_fe.simulate_panel_sar_fe`.

    Returns
    -------
    dict
        Adds ``y_latent`` and ``censored_mask`` to panel SAR FE simulation output.
    """
    create_gdf = kwargs.pop("create_gdf", False)
    wide = kwargs.pop("wide", False)
    geometry_type = kwargs.pop("geometry_type", "polygon")
    gdf = kwargs.get("gdf", None)
    out = simulate_panel_sar_fe(**kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    if create_gdf or gdf is not None or wide:
        N = kwargs.get("N")
        T = kwargs.get("T")
        return make_panel_output_geodataframe(out["y"], out["X"], out["unit"], out["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sem_tobit_fe(*, censoring: float = 0.0, **kwargs) -> dict:
    """Simulate left-censored panel SEM FE data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c`` where observed ``y = max(c, y*)``.
    **kwargs
        Forwarded to :func:`bayespecon.dgp.panel_fe.simulate_panel_sem_fe`.

    Returns
    -------
    dict
        Adds ``y_latent`` and ``censored_mask`` to panel SEM FE simulation output.
    """
    create_gdf = kwargs.pop("create_gdf", False)
    wide = kwargs.pop("wide", False)
    geometry_type = kwargs.pop("geometry_type", "polygon")
    gdf = kwargs.get("gdf", None)
    out = simulate_panel_sem_fe(**kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    if create_gdf or gdf is not None or wide:
        N = kwargs.get("N")
        T = kwargs.get("T")
        return make_panel_output_geodataframe(out["y"], out["X"], out["unit"], out["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out
