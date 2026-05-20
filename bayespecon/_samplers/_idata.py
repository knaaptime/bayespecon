"""Assemble arviz.InferenceData from Gibbs sampler output.

Model-agnostic: takes a dict of arrays and metadata, returns InferenceData
with proper warmup/posterior split, log-likelihood, and observed data.
"""

from __future__ import annotations

import arviz as az
import numpy as np
from typing import Sequence


def gibbs_to_inference_data(
    *,
    posterior_samples: dict[str, np.ndarray],
    warmup_samples: dict[str, np.ndarray] | None = None,
    log_likelihood: dict[str, np.ndarray] | None = None,
    observed_data: dict[str, np.ndarray] | None = None,
    coords: dict[str, Sequence] | None = None,
    dims: dict[str, list[str]] | None = None,
    sample_stats: dict[str, np.ndarray] | None = None,
) -> az.InferenceData:
    """Build InferenceData from Gibbs sampler chain output.

    Parameters
    ----------
    posterior_samples : dict
        ``{var_name: array of shape (chains, draws, ...)}``.
        The post-warmup draws for each parameter.
    warmup_samples : dict, optional
        Same structure as ``posterior_samples`` for the warmup phase.
        Stored in the ``warmup_posterior`` group.
    log_likelihood : dict, optional
        ``{obs_name: array of shape (chains, draws, n)}`` for
        LOO/WAIC computation.
    observed_data : dict, optional
        ``{obs_name: array of shape (n,)}`` observed data.
    coords : dict, optional
        ArviZ coordinate mappings, e.g. ``{"coefficient": ["x1", "x2"]}``.
    dims : dict, optional
        ArviZ dimension mappings, e.g. ``{"beta": ["coefficient"]}``.
    sample_stats : dict, optional
        Additional per-draw statistics (e.g., acceptance rates).

    Returns
    -------
    az.InferenceData
        With ``posterior``, ``warmup_posterior`` (if provided),
        ``log_likelihood`` (if provided), ``observed_data`` (if provided),
        and ``sample_stats`` (if provided) groups.
    """
    idata_kwargs: dict = {}
    if coords is not None:
        idata_kwargs["coords"] = coords
    if dims is not None:
        idata_kwargs["dims"] = dims

    # Build posterior group
    idata = az.from_dict(
        posterior=posterior_samples,
        **idata_kwargs,
    )

    # Add warmup group if provided
    if warmup_samples is not None:
        warmup_idata = az.from_dict(
            posterior=warmup_samples,
            **idata_kwargs,
        )
        idata.add_groups({"warmup_posterior": warmup_idata.posterior})

    # Add log_likelihood group
    if log_likelihood is not None:
        ll_idata = az.from_dict(
            log_likelihood=log_likelihood,
            **idata_kwargs,
        )
        idata.add_groups({"log_likelihood": ll_idata.log_likelihood})

    # Add observed_data group
    if observed_data is not None:
        import xarray as xr

        obs_dict = {}
        for name, arr in observed_data.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                obs_dict[name] = xr.DataArray(arr, dims=["obs_dim"])
            else:
                obs_dict[name] = xr.DataArray(arr)
        idata.add_groups(
            {"observed_data": xr.Dataset(obs_dict)}
        )

    # Add sample_stats group
    if sample_stats is not None:
        stats_idata = az.from_dict(
            sample_stats=sample_stats,
        )
        idata.add_groups({"sample_stats": stats_idata.sample_stats})

    return idata