"""Dynamic panel DGP functions.

Simulates data from dynamic spatial panel models with lagged dependent
variables, including DLM, SDMR, and SDMU fixed-effects specifications.
"""

from __future__ import annotations

import numpy as np

from .panel_fe import _panel_finalize
from .utils import _hetero_scale, ensure_rng, make_design_matrix, make_panel_output_geodataframe, resolve_weights


def simulate_panel_dlm_fe(
    N: int,
    T: int,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate dynamic non-spatial panel FE data.

    DGP
    ---
    ``y_t = phi y_{t-1} + X_t beta + alpha + eps_t``.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    phi : float, default=0.4
        Dynamic autoregressive coefficient on lagged dependent variable.
    beta : np.ndarray, optional
        Coefficients including intercept.
    sigma : float, default=1.0
        Idiosyncratic noise scale.
    sigma_alpha : float, default=0.5
        Unit effect scale.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_{it}\\|^2}` per period.
    rng, seed
        Random state controls.
    W, gdf, contiguity
        Optional spatial structure inputs retained for API consistency.

    Returns
    -------
    dict
        Includes time-first stacked arrays and panel index columns.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        yt = phi * y_prev + Xt @ beta + alpha + eps
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"phi": phi, "beta": beta, "sigma": sigma, "sigma_alpha": sigma_alpha},
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sdmr_fe(
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate dynamic restricted SDM panel FE data.

    DGP
    ---
    ``y_t = rho W y_t + phi y_{t-1} - rho phi W y_{t-1} + X_t beta + alpha + eps_t``.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    rho : float, default=0.3
        Spatial lag coefficient.
    phi : float, default=0.4
        Temporal lag coefficient.
    beta : np.ndarray, optional
        Coefficients including intercept.
    sigma : float, default=1.0
        Idiosyncratic noise scale.
    sigma_alpha : float, default=0.5
        Unit effect scale.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_{it}\\|^2}` per period.
    rng, seed
        Random state controls.
    W, gdf, contiguity
        Spatial structure input and GeoDataFrame neighbor rule.

    Returns
    -------
    dict
        Includes time-first stacked arrays and panel index columns.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - rho * Wd)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        rhs = phi * y_prev - rho * phi * (Wd @ y_prev) + Xt @ beta + alpha + eps
        yt = A_inv @ rhs
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {
            "rho": rho,
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sdmu_fe(
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    theta: float = -0.1,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate dynamic unrestricted SDM panel FE data.

    DGP
    ---
    ``y_t = rho W y_t + phi y_{t-1} + theta W y_{t-1} + X_t beta + alpha + eps_t``.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    rho : float, default=0.3
        Spatial lag coefficient.
    phi : float, default=0.4
        Temporal lag coefficient.
    theta : float, default=-0.1
        Spatial lag coefficient on lagged dependent variable.
    beta : np.ndarray, optional
        Coefficients including intercept.
    sigma : float, default=1.0
        Idiosyncratic noise scale.
    sigma_alpha : float, default=0.5
        Unit effect scale.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_{it}\\|^2}` per period.
    rng, seed
        Random state controls.
    W, gdf, contiguity
        Spatial structure input and GeoDataFrame neighbor rule.

    Returns
    -------
    dict
        Includes time-first stacked arrays and panel index columns.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - rho * Wd)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        rhs = phi * y_prev + theta * (Wd @ y_prev) + Xt @ beta + alpha + eps
        yt = A_inv @ rhs
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {
            "rho": rho,
            "phi": phi,
            "theta": theta,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sar_dynamic_fe(
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate dynamic SAR panel FE data.

    DGP
    ---
    ``y_t = rho W y_t + phi y_{t-1} + X_t beta + alpha + eps_t``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - rho * Wd)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        rhs = phi * y_prev + Xt @ beta + alpha + eps
        yt = A_inv @ rhs
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {
            "rho": rho,
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sem_dynamic_fe(
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate dynamic SEM panel FE data.

    DGP
    ---
    ``y_t = phi y_{t-1} + X_t beta + alpha + u_t,  u_t = lam W u_t + eps_t``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    B_inv = np.linalg.inv(np.eye(N) - lam * Wd)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        u = B_inv @ eps
        yt = phi * y_prev + Xt @ beta + alpha + u
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {
            "lam": lam,
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sdem_dynamic_fe(
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate dynamic SDEM panel FE data.

    DGP
    ---
    ``y_t = phi y_{t-1} + X_t beta_1 + W X_t beta_2 + alpha + u_t,  u_t = lam W u_t + eps_t``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    B_inv = np.linalg.inv(np.eye(N) - lam * Wd)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        u = B_inv @ eps
        yt = phi * y_prev + Xt @ beta + alpha + u
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {
            "lam": lam,
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_slx_dynamic_fe(
    N: int,
    T: int,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate dynamic SLX panel FE data.

    DGP
    ---
    ``y_t = phi y_{t-1} + X_t beta_1 + W X_t beta_2 + alpha + eps_t``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        yt = phi * y_prev + Xt @ beta + alpha + eps
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out
