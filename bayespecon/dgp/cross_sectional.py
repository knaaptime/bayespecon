"""Cross-sectional linear spatial DGP functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import _hetero_scale, ensure_rng, make_design_matrix, make_output_geodataframe, resolve_weights


def _attach_optional_gdf(
    out: dict,
    *,
    source_gdf: Any | None,
    create_gdf: bool,
    geometry_type: str,
):
    if not create_gdf and source_gdf is None:
        return out
    return make_output_geodataframe(
        y=out["y"],
        X=out["X"],
        gdf=source_gdf,
        geometry_type=geometry_type,
    )


def simulate_sar(
    n: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SAR DGP ``y = (I-rho W)^(-1)(X beta + eps)``.

    Parameters
    ----------
    n : int, optional
        Square-grid side length used when only ``n`` is supplied. This
        generates ``n * n`` observations on an ``n x n`` rook grid.
        When ``W`` or ``gdf`` is provided, ``n`` (if provided) must match
        the implied number of observations.
    W : Graph or sparse/dense matrix, optional
        Spatial weights. If supplied, takes precedence over ``gdf``.
    gdf : geopandas.GeoDataFrame, optional
        Spatial units source used when ``W`` is not provided.
    rho : float, default=0.5
        Spatial autoregressive coefficient.
    beta : np.ndarray, optional
        Regression coefficients including intercept.
    sigma : float, default=1.0
        Innovation standard deviation.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_i\\|^2}`.
    rng : np.random.Generator, optional
        Random generator.
    seed : int, optional
        Seed used when ``rng`` is not supplied.
    contiguity : str, default="queen"
        Neighbor rule for ``gdf`` mode.
    create_gdf : bool, default=False
        If True, include a ``gdf`` key in the returned dict with ``y`` and
        ``X_*`` columns attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``create_gdf=True`` and ``gdf`` is not
        provided.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    y = np.linalg.solve(np.eye(nobs) - rho * Wd, X @ beta + eps)
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_ols(
    n: int | None = None,
    W=None,
    gdf=None,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from a non-spatial OLS DGP ``y = X beta + eps``.

    Generates a random design matrix with an intercept and ``len(beta) - 1``
    continuous regressors, and draws the response from a homoskedastic
    Normal error model. No spatial weights matrix is required or produced;
    this function is the natural complement to the spatial DGPs for use as
    a non-spatial baseline.

    Parameters
    ----------
    n : int, optional
        Number of observations when neither ``W`` nor ``gdf`` is provided.
    W : Graph or sparse/dense matrix, optional
        Spatial weights input used only to infer ``n`` and validate dimensions.
        Not used in the OLS data-generating mechanism.
    gdf : geopandas.GeoDataFrame, optional
        Spatial units source used only to infer ``n`` when ``W`` is not provided.
    beta : array-like, optional
        Coefficient vector including intercept. Defaults to
        ``[1.0, 2.0]`` (intercept = 1, one regressor with slope = 2).
    sigma : float, default=1.0
        Innovation standard deviation :math:`\\sigma`.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_i\\|^2}`.
    rng : numpy.random.Generator, optional
        Random generator instance for reproducibility.
    seed : int, optional
        Integer seed used when ``rng`` is not supplied.
    contiguity : str, default="queen"
        Neighbor rule used when inferring ``n`` from ``gdf``.
    create_gdf : bool, default=False
        If ``True``, attaches a GeoDataFrame with ``y`` and ``X_*`` columns
        to geometry generated on an ``n``-unit grid.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``create_gdf=True``.

    Returns
    -------
    dict
        Keys:

        - ``y`` : np.ndarray of shape ``(n,)`` — response variable.
        - ``X`` : np.ndarray of shape ``(n, k)`` — design matrix with
          intercept in the first column.
        - ``params_true`` : dict with ``beta`` and ``sigma``.
        - ``gdf`` : GeoDataFrame (only present when ``create_gdf=True``).
    """
    rng = ensure_rng(rng, seed)

    if n is None and W is None and gdf is None:
        raise ValueError("Provide one of n, W, or gdf.")

    if W is not None or gdf is not None:
        Wd, _ = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
        nobs = Wd.shape[0]
    else:
        nobs = int(n)

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    y = X @ beta + eps

    out: dict = {
        "y": y,
        "X": X,
        "params_true": {"beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sem(
    n: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SEM DGP ``y = X beta + (I-lam W)^(-1) eps``.

    Parameters are analogous to :func:`simulate_sar` with ``lam`` replacing
    ``rho``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    u = np.linalg.solve(np.eye(nobs) - lam * Wd, eps)
    y = X @ beta + u
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_slx(
    n: int | None = None,
    W=None,
    gdf=None,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SLX DGP ``y = X beta1 + W X_no_intercept beta2 + eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    Wx = Wd @ X[:, 1:]
    if Wx.shape[1] != len(beta2):
        raise ValueError("len(beta2) must match number of non-intercept regressors.")

    y = X @ beta1 + Wx @ beta2 + (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sdm(
    n: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SDM DGP ``y = (I-rho W)^(-1)(Xb1 + WXb2 + eps)``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    Wx = Wd @ X[:, 1:]
    if Wx.shape[1] != len(beta2):
        raise ValueError("len(beta2) must match number of non-intercept regressors.")

    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    y = np.linalg.solve(np.eye(nobs) - rho * Wd, X @ beta1 + Wx @ beta2 + eps)
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sdem(
    n: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SDEM DGP ``y = Xb1 + WXb2 + (I-lam W)^(-1)eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    Wx = Wd @ X[:, 1:]
    if Wx.shape[1] != len(beta2):
        raise ValueError("len(beta2) must match number of non-intercept regressors.")

    u = np.linalg.solve(np.eye(nobs) - lam * Wd, (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs))
    y = X @ beta1 + Wx @ beta2 + u
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )
