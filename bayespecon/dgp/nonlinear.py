"""Nonlinear and censored DGP functions.

Simulates data from spatial Tobit and probit models, including left-censored
SAR, SEM, and SDM Tobit variants, as well as spatial probit with regional
random effects.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .cross_sectional import (
    _attach_optional_gdf,
    simulate_sar,
    simulate_sdm,
    simulate_sem,
)
from .utils import _hetero_scale, ensure_rng, make_design_matrix, resolve_weights


def _left_censor(
    y_latent: np.ndarray, censoring: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = y_latent <= censoring
    y_obs = y_latent.copy()
    y_obs[mask] = censoring
    return y_obs, mask


def _shift_intercept_for_rate(
    eta: np.ndarray,
    beta: np.ndarray,
    target_rate: float | None,
    link: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the intercept so E[link(η)] ≈ ``target_rate``.

    Solves for the scalar ``c`` such that
    ``mean(link(eta + c)) == target_rate`` via bisection, then returns
    ``(eta + c, beta_shifted)`` where ``beta_shifted[0] = beta[0] + c``.
    This assumes the design matrix's first column is the intercept
    (which ``make_design_matrix(add_intercept=True)`` guarantees), so
    the model is fit on the same DGP that produced ``y``.

    Parameters
    ----------
    eta : ndarray of shape (n,)
        Latent linear predictor before any shift.
    beta : ndarray of shape (k,)
        Coefficient vector; ``beta[0]`` is the intercept.
    target_rate : float or None
        Desired marginal ``P(y=1)``. If ``None``, ``eta`` and ``beta``
        are returned unchanged.
    link : {"logit", "probit"}
        Inverse link to use when computing the marginal probability.

    Returns
    -------
    eta_new, beta_new : ndarray
        Shifted latent predictor and updated coefficient vector.
    """
    if target_rate is None:
        return eta, beta
    if not 0.0 < float(target_rate) < 1.0:
        raise ValueError(f"target_rate must lie in (0, 1); got {target_rate!r}")

    if link == "logit":

        def _mean_prob(c: float) -> float:
            return float(np.mean(1.0 / (1.0 + np.exp(-(eta + c)))))
    elif link == "probit":

        def _mean_prob(c: float) -> float:
            return float(np.mean(0.5 * (1.0 + erf((eta + c) / np.sqrt(2.0)))))
    else:
        raise ValueError(f"link must be 'logit' or 'probit'; got {link!r}")

    # Mean prob is monotonically increasing in c; bisect on a wide interval.
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _mean_prob(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    beta_new = np.asarray(beta, dtype=float).copy()
    beta_new[0] = beta_new[0] + c
    return eta + c, beta_new


def simulate_sar_tobit(
    *,
    censoring: float = 0.0,
    err_hetero: bool = False,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    **kwargs,
) -> dict:
    """Simulate left-censored SAR Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c`` where observed ``y = max(c, y*)``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sar`.
    create_gdf : bool, default=False
        If True, return a GeoDataFrame with the observed ``y`` and
        ``X_*`` columns attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type when ``create_gdf=True`` and no ``gdf`` is supplied.
    **kwargs
        Forwarded to :func:`simulate_sar`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    source_gdf = kwargs.get("gdf")
    out = simulate_sar(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return _attach_optional_gdf(
        out,
        source_gdf=source_gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sem_tobit(
    *,
    censoring: float = 0.0,
    err_hetero: bool = False,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    **kwargs,
) -> dict:
    """Simulate left-censored SEM Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sem`.
    create_gdf : bool, default=False
        If True, return a GeoDataFrame with the observed ``y`` and
        ``X_*`` columns attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type when ``create_gdf=True`` and no ``gdf`` is supplied.
    **kwargs
        Forwarded to :func:`simulate_sem`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    source_gdf = kwargs.get("gdf")
    out = simulate_sem(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return _attach_optional_gdf(
        out,
        source_gdf=source_gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sdm_tobit(
    *,
    censoring: float = 0.0,
    err_hetero: bool = False,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    **kwargs,
) -> dict:
    """Simulate left-censored SDM Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sdm`.
    create_gdf : bool, default=False
        If True, return a GeoDataFrame with the observed ``y`` and
        ``X_*`` columns attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type when ``create_gdf=True`` and no ``gdf`` is supplied.
    **kwargs
        Forwarded to :func:`simulate_sdm`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    source_gdf = kwargs.get("gdf")
    out = simulate_sdm(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return _attach_optional_gdf(
        out,
        source_gdf=source_gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


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
    target_rate: float | None = None,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
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
    target_rate : float, optional
        If given, an intercept shift ``c`` is solved so that
        ``mean(Phi(eta)) == target_rate``.  ``c`` is added to ``eta``
        and to ``beta[0]``, so ``params_true['beta']`` reflects the
        actual coefficients used to generate ``y``.
    create_gdf : bool, default=False
        If True, return a GeoDataFrame with ``y`` and ``X_*`` columns
        attached to geometry (one row per observation, not per region).
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type when ``create_gdf=True`` and no ``gdf`` is supplied.

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
    eta, beta = _shift_intercept_for_rate(eta, beta, target_rate, link="probit")
    p = 0.5 * (1.0 + erf(eta / np.sqrt(2.0)))
    y = rng.binomial(1, p).astype(float)

    out = {
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
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sar_logit(
    W=None,
    gdf=None,
    n: int | None = None,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    target_rate: float | None = None,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    **kwargs,
) -> dict:
    """Simulate SAR-logit binary outcome data.

    DGP
    ---
    ``eta = (I - rho W)^{-1} (X beta + nu)``, ``nu ~ N(0, I)``,
    ``y ~ Bernoulli(logit^{-1}(eta))``.

    Parameters
    ----------
    W, gdf
        Spatial unit structure. If ``W`` is provided it takes precedence;
        otherwise ``gdf`` is used with ``contiguity``.
    rho : float, default=0.35
        Spatial autoregressive parameter.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    rng, seed
        Random state controls.
    contiguity : str, default="queen"
        GeoDataFrame neighbor rule when ``W`` is omitted.
    target_rate : float, optional
        If given, an intercept shift ``c`` is solved so that
        ``mean(logit^{-1}(eta)) == target_rate``.  ``c`` is added to
        ``eta`` and to ``beta[0]``, so ``params_true['beta']`` reflects
        the actual coefficients used to generate ``y``.
    create_gdf : bool, default=False
        If True, return a GeoDataFrame with ``y`` and ``X_*`` columns
        attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type when ``create_gdf=True`` and no ``gdf`` is supplied.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_sparse``, ``W_graph``, ``eta_true``,
        ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    n_obs = Wd.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    import scipy.sparse as sp

    W_sparse = sp.csr_matrix(Wd)

    X = make_design_matrix(rng, n_obs, k=max(len(beta) - 1, 0), add_intercept=True)

    # Generate latent field: eta = (I - rho W)^{-1} (X beta + nu)
    nu = rng.standard_normal(n_obs)
    Xbeta = X @ beta
    A_rho_inv = sp.linalg.spsolve(
        sp.eye(n_obs, format="csr") - rho * W_sparse, Xbeta + nu
    )
    eta = A_rho_inv
    eta, beta = _shift_intercept_for_rate(eta, beta, target_rate, link="logit")

    # Binary response: y ~ Bernoulli(logit^{-1}(eta))
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)

    out = {
        "y": y,
        "X": X,
        "W_sparse": W_sparse,
        "W_graph": Wg,
        "eta_true": eta,
        "params_true": {
            "rho": rho,
            "beta": beta,
        },
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sem_logit(
    W=None,
    gdf=None,
    n: int | None = None,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    target_rate: float | None = None,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    **kwargs,
) -> dict:
    """Simulate SEM-logit binary outcome data.

    DGP
    ---
    ``eta = X beta + (I - lam W)^{-1} nu``, ``nu ~ N(0, I)``,
    ``y ~ Bernoulli(logit^{-1}(eta))``.

    Parameters
    ----------
    W, gdf
        Spatial unit structure. If ``W`` is provided it takes precedence;
        otherwise ``gdf`` is used with ``contiguity``.
    lam : float, default 0.35
        Spatial error parameter.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    rng, seed
        Random state controls.
    contiguity : str, default "queen"
        GeoDataFrame neighbor rule when ``W`` is omitted.
    target_rate : float, optional
        If given, an intercept shift ``c`` is solved so that
        ``mean(logit^{-1}(eta)) == target_rate``.  ``c`` is added to
        ``eta`` and to ``beta[0]``, so ``params_true['beta']`` reflects
        the actual coefficients used to generate ``y``.
    create_gdf : bool, default=False
        If True, return a GeoDataFrame with ``y`` and ``X_*`` columns
        attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type when ``create_gdf=True`` and no ``gdf`` is supplied.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_sparse``, ``W_graph``, ``eta_true``,
        ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    n_obs = Wd.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    import scipy.sparse as sp

    W_sparse = sp.csr_matrix(Wd)

    X = make_design_matrix(rng, n_obs, k=max(len(beta) - 1, 0), add_intercept=True)

    # Generate latent field: eta = X beta + (I - lam W)^{-1} nu
    nu = rng.standard_normal(n_obs)
    Xbeta = X @ beta
    A_lam_inv = sp.linalg.spsolve(sp.eye(n_obs, format="csr") - lam * W_sparse, nu)
    eta = Xbeta + A_lam_inv
    eta, beta = _shift_intercept_for_rate(eta, beta, target_rate, link="logit")

    # Binary response: y ~ Bernoulli(logit^{-1}(eta))
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)

    out = {
        "y": y,
        "X": X,
        "W_sparse": W_sparse,
        "W_graph": Wg,
        "eta_true": eta,
        "params_true": {
            "lam": lam,
            "beta": beta,
        },
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )
