"""Nonlinear and censored DGP functions.

Simulates data from spatial Tobit and probit models, including left-censored
SAR, SEM, and SDM Tobit variants, as well as spatial probit with regional
random effects.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .cross_sectional import simulate_sar, simulate_sdm, simulate_sem
from .utils import _hetero_scale, ensure_rng, make_design_matrix, resolve_weights


def _left_censor(y_latent: np.ndarray, censoring: float) -> tuple[np.ndarray, np.ndarray]:
    mask = y_latent <= censoring
    y_obs = y_latent.copy()
    y_obs[mask] = censoring
    return y_obs, mask


def simulate_sar_tobit(*, censoring: float = 0.0, err_hetero: bool = False, **kwargs) -> dict:
    """Simulate left-censored SAR Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c`` where observed ``y = max(c, y*)``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sar`.
    **kwargs
        Forwarded to :func:`simulate_sar`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    out = simulate_sar(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return out


def simulate_sem_tobit(*, censoring: float = 0.0, err_hetero: bool = False, **kwargs) -> dict:
    """Simulate left-censored SEM Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sem`.
    **kwargs
        Forwarded to :func:`simulate_sem`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    out = simulate_sem(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return out


def simulate_sdm_tobit(*, censoring: float = 0.0, err_hetero: bool = False, **kwargs) -> dict:
    """Simulate left-censored SDM Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sdm`.
    **kwargs
        Forwarded to :func:`simulate_sdm`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    out = simulate_sdm(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return out


def simulate_spatial_probit(
    W=None,
    gdf=None,
    n: int | None = None,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    sigma_a: float = 0.8,
    n_per_region: int = 25,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    """Simulate SpatialProbit-style binary outcome data.

    DGP
    ---
    ``a = (I-rho W)^(-1) sigma_a z`` (region effects),
    ``eta = X beta + a[region]``,
    ``y ~ Bernoulli(Phi(eta))``.

    Parameters
    ----------
    W, gdf
        Spatial unit structure. If ``W`` is provided it takes precedence;
        otherwise ``gdf`` is used with ``contiguity``.
    rho : float, default=0.35
        Spatial dependence in regional effects.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    sigma_a : float, default=0.8
        Regional effect innovation scale.
    n_per_region : int, default=25
        Number of observations per region.
    err_hetero : bool, default=False
        If True, generate heteroskedastic region effects with
        region-specific standard deviations
        :math:`\\sigma_{a,j} = \\sigma_a \\sqrt{1 + \\|\\bar{x}_j\\|^2}`
        where :math:`\\bar{x}_j` is the mean regressor vector for region
        ``j``.
    rng, seed
        Random state controls.
    contiguity : str, default="queen"
        GeoDataFrame neighbor rule when ``W`` is omitted.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``region_ids``, ``W_dense``, ``W_graph``,
        ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    m = Wd.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    nobs = int(m * n_per_region)
    region_ids = np.repeat(np.arange(m), n_per_region)

    if err_hetero:
        # Generate X first so we can compute region-level means for
        # heteroskedastic scaling.  This changes the RNG draw order
        # relative to the homoskedastic path.
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
        X_region_mean = X.reshape(m, n_per_region, -1).mean(axis=1)
        a_scale = _hetero_scale(X_region_mean, sigma_a)
    else:
        # Preserve the original RNG draw order: a is drawn before X.
        a_scale = sigma_a

    a = np.linalg.solve(np.eye(m) - rho * Wd, a_scale * rng.standard_normal(m))

    if not err_hetero:
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)

    eta = X @ beta + a[region_ids]
    p = 0.5 * (1.0 + erf(eta / np.sqrt(2.0)))
    y = rng.binomial(1, p).astype(float)

    return {
        "y": y,
        "X": X,
        "region_ids": region_ids,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {
            "rho": rho,
            "beta": beta,
            "sigma_a": sigma_a,
            "n_per_region": n_per_region,
        },
    }
