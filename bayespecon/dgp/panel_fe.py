"""Panel fixed-effects style DGP functions."""

from __future__ import annotations

import numpy as np

from .utils import _hetero_scale, ensure_rng, make_design_matrix, make_panel_output_geodataframe, panel_index, resolve_weights


def _panel_finalize(y_list: list[np.ndarray], X_list: list[np.ndarray], N: int, T: int) -> tuple[np.ndarray, np.ndarray, dict]:
    y = np.concatenate(y_list)
    X = np.vstack(X_list)
    idx_df = panel_index(N, T)
    return y, X, {"unit": idx_df["unit"].to_numpy(), "time": idx_df["time"].to_numpy()}


def simulate_panel_ols_fe(
    N: int,
    T: int,
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
    """Simulate pooled data compatible with OLSPanelFE model assumptions.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    beta : np.ndarray, optional
        Coefficients including intercept.
    sigma : float, default=1.0
        Idiosyncratic noise scale.
    sigma_alpha : float, default=0.5
        Unit effect scale used in simulation.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_{it}\\|^2}` per period.
    rng, seed
        Random state controls.
    W, gdf, contiguity
        Optional spatial structure accepted for API consistency. For OLS this is
        returned but not used in the data equation.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``unit``, ``time``, ``W_dense``, ``W_graph``,
        ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        yt = Xt @ beta + alpha + eps
        y_list.append(yt)
        X_list.append(Xt)

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"beta": beta, "sigma": sigma, "sigma_alpha": sigma_alpha},
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sar_fe(
    N: int,
    T: int,
    rho: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    n: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate SAR panel data in time-first stacking order.

    DGP
    ---
    For each period ``t``:
    ``y_t = (I-rho W)^(-1) (X_t beta + alpha + eps_t)``.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    rho : float, default=0.4
        Spatial lag coefficient.
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
        Keys: ``y``, ``X``, ``unit``, ``time``, ``W_dense``, ``W_graph``,
        ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - rho * Wd)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        yt = A_inv @ (Xt @ beta + alpha + eps)
        y_list.append(yt)
        X_list.append(Xt)

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta": beta, "sigma": sigma, "sigma_alpha": sigma_alpha},
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sem_fe(
    N: int,
    T: int,
    lam: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    n: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate SEM panel data in time-first stacking order.

    DGP
    ---
    ``u_t = (I-lam W)^(-1) eps_t`` and ``y_t = X_t beta + alpha + u_t``.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    lam : float, default=0.4
        Spatial error coefficient.
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
        Keys: ``y``, ``X``, ``unit``, ``time``, ``W_dense``, ``W_graph``,
        ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    if Wd.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - lam * Wd)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        u = A_inv @ ((_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N))
        yt = Xt @ beta + alpha + u
        y_list.append(yt)
        X_list.append(Xt)

    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta": beta, "sigma": sigma, "sigma_alpha": sigma_alpha},
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sdm_fe(
    N: int,
    T: int,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
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
    """Simulate SDM panel FE data.

    DGP
    ---
    ``y_t = (I-rho W)^(-1) (X_t beta1 + W X_t[:,1:] beta2 + alpha + eps_t)``.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    rho : float, default=0.4
        Spatial lag coefficient.
    beta1 : np.ndarray, optional
        Coefficients on ``X`` including intercept.
    beta2 : np.ndarray, optional
        Coefficients on spatially lagged non-intercept regressors.
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

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - rho * Wd)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta1) - 1, 0), add_intercept=True)
        Wx = Wd @ Xt[:, 1:]
        if Wx.shape[1] != len(beta2):
            raise ValueError("len(beta2) must match number of non-intercept regressors.")
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N)
        yt = A_inv @ (Xt @ beta1 + Wx @ beta2 + alpha + eps)
        y_list.append(yt)
        X_list.append(Xt)

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
            "beta1": beta1,
            "beta2": beta2,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out


def simulate_panel_sdem_fe(
    N: int,
    T: int,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
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
    """Simulate SDEM panel FE data.

    DGP
    ---
    ``u_t = (I-lam W)^(-1) eps_t`` and
    ``y_t = X_t beta1 + W X_t[:,1:] beta2 + alpha + u_t``.

    Parameters
    ----------
    N, T : int
        Number of units and time periods.
    lam : float, default=0.4
        Spatial error coefficient.
    beta1 : np.ndarray, optional
        Coefficients on ``X`` including intercept.
    beta2 : np.ndarray, optional
        Coefficients on spatially lagged non-intercept regressors.
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

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    alpha = rng.normal(0.0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - lam * Wd)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta1) - 1, 0), add_intercept=True)
        Wx = Wd @ Xt[:, 1:]
        if Wx.shape[1] != len(beta2):
            raise ValueError("len(beta2) must match number of non-intercept regressors.")
        u = A_inv @ ((_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(N))
        yt = Xt @ beta1 + Wx @ beta2 + alpha + u
        y_list.append(yt)
        X_list.append(Xt)

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
            "beta1": beta1,
            "beta2": beta2,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(y, X, idx["unit"], idx["time"], N, T, gdf=gdf, geometry_type=geometry_type, wide=wide)
    return out
