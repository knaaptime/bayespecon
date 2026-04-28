"""Data-generating process for Bayesian spatial flow (O-D) models.

Simulates origin-destination flow data from the SAR flow model:

.. math::

    y = (I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w)^{-1}
        (X\\beta + \\varepsilon), \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from libpysal.graph import Graph

from ..graph import (
    _validate_graph,
    flow_design_matrix,
    flow_design_matrix_with_orig,
    flow_weight_matrices,
)


def generate_flow_data(
    n: int,
    G: Graph,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    beta_d: Union[np.ndarray, list],
    beta_o: Union[np.ndarray, list],
    sigma: float = 1.0,
    X: Optional[np.ndarray] = None,
    col_names: Optional[list] = None,
    dist: Optional[np.ndarray] = None,
    alpha: float = 0.0,
    seed: Optional[int] = None,
    gdf=None,
    err_hetero: bool = False,
) -> dict:
    """Simulate flow data from a SAR flow model.

    Generates :math:`N = n^2` flow observations from:

    .. math::

        y = A^{-1}(X\\beta + \\varepsilon), \\quad
        A = I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w, \\quad
        \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

    Parameters
    ----------
    n : int
        Number of spatial units.  Must match the size of *G*.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    rho_w : float
        Network (origin-destination) spatial autoregressive parameter.
    beta_d : array-like, shape (k,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k,)
        Origin-side regression coefficients (must match length of *beta_d*).
    sigma : float, default 1.0
        Standard deviation of the error term.
    X : np.ndarray, shape (n, k), optional
        Regional attribute matrix.  If None, draws X ~ N(0, 1).
    col_names : list[str], optional
        Names for the *k* columns of *X*.
    dist : np.ndarray, shape (n, n), optional
        Distance / cost matrix.  If provided, its vector form is included in
        the design matrix with coefficient ``gamma=0`` (i.e., distance has no
        direct effect in the DGP unless *alpha* for distance is added to
        *beta_d*/*beta_o*).
    alpha : float, default 0.0
        Intercept term added uniformly to all flow cells.
    seed : int, optional
        Random seed for reproducibility.
    gdf : geopandas.GeoDataFrame, optional
        Accepted for API consistency with other DGP functions; not used by
        the flow simulator (which takes *G* directly instead).
    err_hetero : bool, default False
        If True, generate heteroskedastic innovations: each flow cell
        :math:`(i,j)` has standard deviation
        :math:`\\sigma \\sqrt{1 + \\|x_i\\|^2 + \\|x_j\\|^2}` where
        :math:`x_i`, :math:`x_j` are the destination and origin attribute
        vectors for that cell.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"y_vec"`` (N,): vectorised flows.
        - ``"y_mat"`` (n, n): flow matrix form.
        - ``"X"`` (N, p): full O-D design matrix (for model fitting).
        - ``"X_regional"`` (n, k): regional attribute matrix.
        - ``"design"`` :class:`~bayespecon.graph.FlowDesignMatrix`: full design.
        - ``"W"`` scipy.sparse.csr_matrix: n×n weight matrix.
        - ``"G"`` libpysal.graph.Graph: spatial graph.
        - ``"rho_d"``, ``"rho_o"``, ``"rho_w"``, ``"sigma"``: true parameters.
        - ``"beta_d"``, ``"beta_o"``: true coefficient vectors.

    Raises
    ------
    ValueError
        If the A matrix is singular (invalid parameter combination).
    """
    rng = np.random.default_rng(seed)

    W = _validate_graph(G)
    if W.shape[0] != n:
        raise ValueError(f"G has {W.shape[0]} units but n={n} was specified.")
    N = n * n

    beta_d_arr = np.asarray(beta_d, dtype=np.float64).ravel()
    beta_o_arr = np.asarray(beta_o, dtype=np.float64).ravel()
    k = len(beta_d_arr)
    if len(beta_o_arr) != k:
        raise ValueError("beta_d and beta_o must have the same length.")

    # Generate or validate regional attributes
    if X is None:
        X_arr = rng.standard_normal((n, k))
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape != (n, k):
            raise ValueError(f"X must have shape ({n}, {k}), got {X_arr.shape}.")

    # Build design matrix
    design = flow_design_matrix(X_arr, col_names=col_names, dist=dist)

    # Assemble A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww
    wms = flow_weight_matrices(G)
    Wd = wms["destination"]
    Wo = wms["origin"]
    Ww = wms["network"]
    I_N = sp.eye(N, format="csr", dtype=np.float64)
    A = I_N - rho_d * Wd - rho_o * Wo - rho_w * Ww

    # Build deterministic component: X_design @ beta_extended
    # beta_extended layout: [alpha, 0, beta_d..., beta_o..., 0...(intra), (0 for dist)]
    p = design.combined.shape[1]
    beta_full = np.zeros(p, dtype=np.float64)
    beta_full[0] = alpha  # intercept
    # intra_indicator coefficient stays 0 in DGP
    beta_full[2 : 2 + k] = beta_d_arr  # destination block
    beta_full[2 + k : 2 + 2 * k] = beta_o_arr  # origin block
    # intra and dist coefficients remain 0

    Xbeta = design.combined @ beta_full  # (N,)

    # Solve y = A^{-1}(Xbeta + epsilon)
    if err_hetero:
        # Cell-level standard deviations: sigma * sqrt(1 + ||x_dest||^2 + ||x_orig||^2)
        dest_idx = np.repeat(np.arange(n), n)  # row = destination unit
        orig_idx = np.tile(np.arange(n), n)  # col = origin unit
        scale_vec = sigma * np.sqrt(
            1.0
            + np.sum(X_arr[dest_idx] ** 2, axis=1)
            + np.sum(X_arr[orig_idx] ** 2, axis=1)
        )
        eps = rng.standard_normal(N) * scale_vec
    else:
        eps = rng.normal(scale=sigma, size=N)
    rhs = Xbeta + eps

    try:
        y_vec = sp.linalg.spsolve(A, rhs)
    except sp.linalg.MatrixRankWarning as exc:
        raise ValueError(
            "A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww is singular. "
            "Check that rho_d + rho_o + rho_w < 1 for row-stochastic W."
        ) from exc

    y_mat = y_vec.reshape(n, n)

    return {
        "y_vec": y_vec,
        "y_mat": y_mat,
        "X": design.combined,
        "X_regional": X_arr,
        "col_names": design.feature_names,
        "design": design,
        "W": W,
        "G": G,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "sigma": sigma,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
    }


def generate_poisson_flow_data(
    n: int = 10,
    k: int = 2,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: float | list[float] | None = None,
    beta_o: float | list[float] | None = None,
    seed: int = 42,
    G: Graph | None = None,
    # Accepted for API parity with other DGP functions but unused:
    err_hetero: bool = False,
    gdf: object = None,
) -> dict:
    """Generate synthetic origin-destination flow **count** data for a Poisson
    spatial autoregressive flow model.

    The data-generating process follows:

    .. math::

        \\eta = A(\\rho_d, \\rho_o, \\rho_w)^{-1} X\\beta, \\qquad
        y_{ij} \\sim \\operatorname{Poisson}(\\exp(\\eta_{ij}))

    where the system matrix is

    .. math::

        A = I_N - \\rho_d (I_n \\otimes W) - \\rho_o (W \\otimes I_n)
              - \\rho_w (W \\otimes W), \\quad N = n^2

    and :math:`W` is the row-standardised spatial weight matrix.

    Parameters
    ----------
    n : int, default 10
        Approximate number of spatial units.  When neither *G* nor *gdf*
        is provided, a rook-contiguity grid with
        round(sqrt(n)) units per side is created, yielding
        approximately *n* units.  Total number of flows is
        N = n_actual^2.  When *G* is provided, *n* must
        match the number of units in *G*.
    k : int, default 2
        Number of destination/origin attribute columns (excluding intercepts
        added internally).
    rho_d : float, default 0.3
        Destination autocorrelation parameter.
    rho_o : float, default 0.2
        Origin autocorrelation parameter.
    rho_w : float, default 0.1
        Network autocorrelation parameter.
    beta_d : float or list of float or None, default None
        Destination-side coefficients for the *k* attributes.  A scalar
        broadcasts to all columns.  Defaults to ``1.0`` for all columns.
    beta_o : float or list of float or None, default None
        Origin-side coefficients.  Defaults to ``1.0`` for all columns.
    seed : int, default 42
        Seed for :class:`numpy.random.default_rng`.
    G : libpysal.graph.Graph or None, default None
        Row-standardised spatial graph on *n* units.  If ``None``, a
        rook-contiguity graph on a regular grid is constructed automatically
        via :func:`~bayespecon.dgp.utils.resolve_weights`.
    err_hetero : bool, default False
        Accepted for API parity with other DGP functions; ignored for the
        Poisson model (the variance is determined by the mean).
    gdf : GeoDataFrame or None, default None
        Accepted for API parity; ignored (use *G* instead).

    Returns
    -------
    dict with keys:
        ``y_vec`` : np.ndarray, shape (N,), dtype int64
            Flattened count observations.
        ``y_mat`` : np.ndarray, shape (n, n), dtype int64
            Count observations reshaped as an O×D matrix.
        ``eta_vec`` : np.ndarray, shape (N,)
            Log-mean (spatially filtered linear predictor).
        ``lambda_vec`` : np.ndarray, shape (N,)
            Poisson means (:math:`\\exp(\\eta_{ij})`).
        ``Xd`` : np.ndarray, shape (n, k)
            Destination-side regional attribute matrix.
        ``Xd`` : np.ndarray, shape (n, k)
            Destination-side regional attribute matrix.
        ``Xo`` : np.ndarray, shape (n, k)
            Origin-side regional attribute matrix.
        ``X`` : np.ndarray, shape (N, p)
            Full O-D design matrix (for model fitting).
        ``design`` : FlowDesignMatrix
            Full O-D design matrix (for downstream inspection).
        ``W`` : np.ndarray, shape (n, n)
            Dense row-standardised weight matrix.
        ``G`` : libpysal.graph.Graph
            Spatial graph.
        ``rho_d``, ``rho_o``, ``rho_w``
            True autocorrelation parameters.
        ``beta_d``, ``beta_o``
            True coefficient vectors.

    Raises
    ------
    np.linalg.LinAlgError
        If the system matrix :math:`A` is singular (usually because
        ``rho_d + rho_o + rho_w >= 1``).

    Examples
    --------
    >>> from bayespecon.dgp import generate_poisson_flow_data
    >>> data = generate_poisson_flow_data(n=9, seed=0)
    >>> data["y_mat"].dtype
    dtype('int64')
    >>> data["lambda_vec"].shape
    (81,)
    >>> data["Xd"].shape
    (9, 2)
    >>> data["Xo"].shape
    (9, 2)
    """
    from .utils import resolve_weights, rook_grid_weights

    rng = np.random.default_rng(seed)

    # --- Resolve spatial weights ---
    if G is not None:
        W, G = resolve_weights(W=G, gdf=None, n=n)
    elif gdf is not None and hasattr(gdf, "__geo_interface__"):
        W, G = resolve_weights(W=None, gdf=gdf, n=n)
    else:
        # Default: create a rook-contiguity grid with ~n units.
        # n_side is chosen so that n_side² ≈ n.
        import math

        n_side = max(2, int(round(math.sqrt(n))))
        W, G = rook_grid_weights(n_side)
    n = W.shape[0]  # actual number of spatial units
    N = n * n

    # --- Coefficient vectors ---
    beta_d_arr = (
        np.ones(k, dtype=float)
        if beta_d is None
        else np.broadcast_to(np.asarray(beta_d, dtype=float), (k,)).copy()
    )
    beta_o_arr = (
        np.ones(k, dtype=float)
        if beta_o is None
        else np.broadcast_to(np.asarray(beta_o, dtype=float), (k,)).copy()
    )

    # --- Regional attributes ---
    Xd_raw = rng.standard_normal((n, k))
    Xo_raw = rng.standard_normal((n, k))

    # --- Build design matrix with separate Xd and Xo blocks ---
    design = flow_design_matrix_with_orig(
        Xd_raw, Xo_raw, col_names=[f"x{i}" for i in range(k)]
    )

    # --- Build deterministic component: X_design @ beta_extended ---
    # beta layout: [alpha=0, intra=0, beta_d..., beta_o..., intra_x=0..., (dist=0)]
    p = design.combined.shape[1]
    beta_full = np.zeros(p, dtype=np.float64)
    beta_full[2 : 2 + k] = beta_d_arr  # destination block
    beta_full[2 + k : 2 + 2 * k] = beta_o_arr  # origin block

    Xbeta = design.combined @ beta_full  # (N,)

    # --- Assemble A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww ---
    wms = flow_weight_matrices(G)
    Wd = wms["destination"]
    Wo = wms["origin"]
    Ww = wms["network"]
    I_N = sp.eye(N, format="csr", dtype=np.float64)
    A = I_N - rho_d * Wd - rho_o * Wo - rho_w * Ww

    # --- Solve A eta = Xbeta  (Poisson: log-mean scale) ---
    try:
        eta_vec = sp.linalg.spsolve(A, Xbeta)
    except sp.linalg.MatrixRankWarning as exc:
        raise np.linalg.LinAlgError(
            "A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww is singular. "
            "Check that rho_d + rho_o + rho_w < 1 for row-stochastic W."
        ) from exc

    lambda_vec = np.exp(eta_vec)
    y_vec = rng.poisson(lambda_vec).astype(np.int64)
    y_mat = y_vec.reshape(n, n)

    return {
        "y_vec": y_vec,
        "y_mat": y_mat,
        "eta_vec": eta_vec,
        "lambda_vec": lambda_vec,
        "Xd": Xd_raw,
        "Xo": Xo_raw,
        "X": design.combined,
        "col_names": design.feature_names,
        "design": design,
        "W": W,
        "G": G,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
    }


def generate_panel_flow_data(
    n: int,
    T: int,
    G: Graph,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    beta_d: Union[np.ndarray, list],
    beta_o: Union[np.ndarray, list],
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    seed: Optional[int] = None,
    k: Optional[int] = None,
    err_hetero: bool = False,
    gdf: object = None,
) -> dict:
    r"""Simulate panel flow data from a SAR flow model with unit effects.

    For each period :math:`t = 1, \dots, T`, generates :math:`N = n^2`
    flow observations from:

    .. math::

        y_t = A^{-1}(X_t \beta + \alpha + \varepsilon_t), \quad
        A = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w, \quad
        \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)

    where :math:`\alpha \sim \mathcal{N}(0, \sigma_\alpha^2 I_N)` are
    O-D-pair random effects drawn once and held fixed across periods.
    Observations are stacked in **time-first** order so that the
    observation at index :math:`t \cdot n^2 + k` is O-D pair :math:`k`
    at time :math:`t`.

    Parameters
    ----------
    n : int
        Number of spatial units.  Must match the size of *G*.
    T : int
        Number of time periods.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    rho_w : float
        Network (origin-destination) spatial autoregressive parameter.
    beta_d : array-like, shape (k,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k,)
        Origin-side regression coefficients (must match length of *beta_d*).
    sigma : float, default 1.0
        Standard deviation of the idiosyncratic error term.
    sigma_alpha : float, default 0.5
        Standard deviation of the O-D-pair random effect.
        Set to 0 for pooled (no unit effects).
    seed : int, optional
        Random seed for reproducibility.
    k : int, optional
        Number of regional attribute columns.  If ``None``, inferred
        from the length of *beta_d*.
    err_hetero : bool, default False
        Accepted for API parity with other DGP functions; currently
        ignored (homoskedastic errors are always generated).
    gdf : object, optional
        Accepted for API parity with other DGP functions; not used
        (pass *G* directly instead).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"y"`` (n²T,): time-first stacked flow vector.
        - ``"X"`` (n²T, p): time-first stacked O-D design matrix.
        - ``"col_names"`` list[str]: feature names.
        - ``"G"`` libpysal.graph.Graph: spatial graph.
        - ``"rho_d"``, ``"rho_o"``, ``"rho_w"``, ``"sigma"``,
          ``"sigma_alpha"``: true parameters.
        - ``"beta_d"``, ``"beta_o"``: true coefficient vectors.
        - ``"params_true"`` dict: nested dict of all true parameters.

    Raises
    ------
    ValueError
        If the A matrix is singular (invalid parameter combination).
    """
    rng = np.random.default_rng(seed)

    W = _validate_graph(G)
    if W.shape[0] != n:
        raise ValueError(f"G has {W.shape[0]} units but n={n} was specified.")
    N = n * n  # O-D pairs per period

    beta_d_arr = np.asarray(beta_d, dtype=np.float64).ravel()
    beta_o_arr = np.asarray(beta_o, dtype=np.float64).ravel()
    k_val = k if k is not None else len(beta_d_arr)
    if len(beta_o_arr) != k_val:
        raise ValueError("beta_d and beta_o must have the same length.")

    # Build Kronecker weight matrices (same for every period)
    wms = flow_weight_matrices(G)
    Wd = wms["destination"]
    Wo = wms["origin"]
    Ww = wms["network"]
    I_N = sp.eye(N, format="csr", dtype=np.float64)
    A = I_N - rho_d * Wd - rho_o * Wo - rho_w * Ww

    # Factorize A once (same A for every period)
    try:
        solve_A = sp.linalg.factorized(A.tocsc())
    except RuntimeError as exc:
        raise ValueError(
            "A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww is singular. "
            "Check that rho_d + rho_o + rho_w < 1 for row-stochastic W."
        ) from exc

    # Draw O-D-pair random effects once
    alpha = rng.normal(0.0, sigma_alpha, N) if sigma_alpha > 0 else np.zeros(N)

    y_list, X_list = [], []
    col_names = None

    for _ in range(T):
        # Generate fresh regional attributes each period
        X_reg = rng.standard_normal((n, k_val))
        design = flow_design_matrix(X_reg)
        if col_names is None:
            col_names = design.feature_names

        # Build beta_full: [alpha=0, intra=0, beta_d..., beta_o..., 0...]
        p = design.combined.shape[1]
        beta_full = np.zeros(p, dtype=np.float64)
        beta_full[2 : 2 + k_val] = beta_d_arr
        beta_full[2 + k_val : 2 + 2 * k_val] = beta_o_arr

        Xbeta = design.combined @ beta_full  # (N,)
        eps = rng.normal(scale=sigma, size=N)
        rhs = Xbeta + alpha + eps

        y_t = solve_A(rhs)
        y_list.append(y_t)
        X_list.append(design.combined)

    y = np.concatenate(y_list)  # (N*T,)
    X = np.vstack(X_list)  # (N*T, p)

    return {
        "y": y,
        "X": X,
        "col_names": col_names,
        "G": G,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "sigma": sigma,
        "sigma_alpha": sigma_alpha,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "params_true": {
            "rho_d": rho_d,
            "rho_o": rho_o,
            "rho_w": rho_w,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
            "beta_d": beta_d_arr,
            "beta_o": beta_o_arr,
        },
    }


def generate_panel_poisson_flow_data(
    n: int,
    T: int,
    G: Graph,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: float | list[float] | None = None,
    beta_o: float | list[float] | None = None,
    seed: int = 42,
    k: int = 2,
    err_hetero: bool = False,
    gdf: object = None,
) -> dict:
    r"""Simulate panel Poisson flow data from a spatial autoregressive model.

    For each period :math:`t = 1, \dots, T`, generates :math:`N = n^2`
    flow counts from:

    .. math::

        \eta_t = A^{-1} X_t \beta, \quad
        y_{ij,t} \sim \operatorname{Poisson}(\exp(\eta_{ij,t}))

    where the system matrix is

    .. math::

        A = I_N - \rho_d (I_n \otimes W) - \rho_o (W \otimes I_n)
              - \rho_w (W \otimes W), \quad N = n^2

    Observations are stacked in **time-first** order.  There is no
    :math:`\sigma` or :math:`\alpha` parameter (the Poisson variance
    equals the mean, and the panel is pooled-only).

    Parameters
    ----------
    n : int
        Number of spatial units.  Must match the size of *G*.
    T : int
        Number of time periods.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    rho_d : float, default 0.3
        Destination autocorrelation parameter.
    rho_o : float, default 0.2
        Origin autocorrelation parameter.
    rho_w : float, default 0.1
        Network autocorrelation parameter.
    beta_d : float or list of float or None, default None
        Destination-side coefficients for the *k* attributes.  A scalar
        broadcasts to all columns.  Defaults to ``1.0`` for all columns.
    beta_o : float or list of float or None, default None
        Origin-side coefficients.  Defaults to ``1.0`` for all columns.
    seed : int, default 42
        Seed for :class:`numpy.random.default_rng`.
    k : int, default 2
        Number of destination/origin attribute columns.
    err_hetero : bool, default False
        Accepted for API parity with other DGP functions; ignored for
        the Poisson model (the variance is determined by the mean).
    gdf : object, optional
        Accepted for API parity with other DGP functions; not used
        (pass *G* directly instead).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"y"`` (n²T,): time-first stacked count vector (int64).
        - ``"X"`` (n²T, p): time-first stacked O-D design matrix.
        - ``"col_names"`` list[str]: feature names.
        - ``"G"`` libpysal.graph.Graph: spatial graph.
        - ``"rho_d"``, ``"rho_o"``, ``"rho_w"``: true parameters.
        - ``"beta_d"``, ``"beta_o"``: true coefficient vectors.
        - ``"params_true"`` dict: nested dict of all true parameters.

    Raises
    ------
    ValueError
        If the A matrix is singular (invalid parameter combination).
    """
    rng = np.random.default_rng(seed)

    W = _validate_graph(G)
    if W.shape[0] != n:
        raise ValueError(f"G has {W.shape[0]} units but n={n} was specified.")
    N = n * n

    # Coefficient vectors
    beta_d_arr = (
        np.ones(k, dtype=float)
        if beta_d is None
        else np.broadcast_to(np.asarray(beta_d, dtype=float), (k,)).copy()
    )
    beta_o_arr = (
        np.ones(k, dtype=float)
        if beta_o is None
        else np.broadcast_to(np.asarray(beta_o, dtype=float), (k,)).copy()
    )

    # Build Kronecker weight matrices (same for every period)
    wms = flow_weight_matrices(G)
    Wd = wms["destination"]
    Wo = wms["origin"]
    Ww = wms["network"]
    I_N = sp.eye(N, format="csr", dtype=np.float64)
    A = I_N - rho_d * Wd - rho_o * Wo - rho_w * Ww

    # Factorize A once
    try:
        solve_A = sp.linalg.factorized(A.tocsc())
    except RuntimeError as exc:
        raise ValueError(
            "A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww is singular. "
            "Check that rho_d + rho_o + rho_w < 1 for row-stochastic W."
        ) from exc

    y_list, X_list = [], []
    col_names = None

    for _ in range(T):
        # Generate fresh regional attributes each period
        Xd_raw = rng.standard_normal((n, k))
        Xo_raw = rng.standard_normal((n, k))

        design = flow_design_matrix_with_orig(
            Xd_raw, Xo_raw, col_names=[f"x{i}" for i in range(k)]
        )
        if col_names is None:
            col_names = design.feature_names

        # Build beta_full: [alpha=0, intra=0, beta_d..., beta_o..., 0...]
        p = design.combined.shape[1]
        beta_full = np.zeros(p, dtype=np.float64)
        beta_full[2 : 2 + k] = beta_d_arr
        beta_full[2 + k : 2 + 2 * k] = beta_o_arr

        Xbeta = design.combined @ beta_full  # (N,)

        # Solve A eta = Xbeta
        eta_t = solve_A(Xbeta)
        lambda_t = np.exp(eta_t)
        y_t = rng.poisson(lambda_t).astype(np.int64)

        y_list.append(y_t)
        X_list.append(design.combined)

    y = np.concatenate(y_list)  # (N*T,)
    X = np.vstack(X_list)  # (N*T, p)

    return {
        "y": y,
        "X": X,
        "col_names": col_names,
        "G": G,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "params_true": {
            "rho_d": rho_d,
            "rho_o": rho_o,
            "rho_w": rho_w,
            "beta_d": beta_d_arr,
            "beta_o": beta_o_arr,
        },
    }


def generate_flow_data_separable(
    n: int,
    G: Graph,
    rho_d: float,
    rho_o: float,
    beta_d: Union[np.ndarray, list],
    beta_o: Union[np.ndarray, list],
    **kwargs,
) -> dict:
    """Simulate flow data from a *separable* SAR flow model.

    Identical to :func:`generate_flow_data` except the network parameter is
    derived from the separability constraint
    :math:`\\rho_w = -\\rho_d \\rho_o`, so it is **not** a free argument.
    Use this function to generate training data for
    :class:`~bayespecon.models.flow.SARFlowSeparable`.

    Parameters
    ----------
    n : int
        Number of spatial units.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    beta_d : array-like, shape (k,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k,)
        Origin-side regression coefficients.
    **kwargs
        Forwarded to :func:`generate_flow_data` (e.g. ``sigma``, ``seed``,
        ``X``, ``col_names``).

    Returns
    -------
    dict
        Same as :func:`generate_flow_data`.  The ``"rho_w"`` entry will equal
        ``-rho_d * rho_o``.
    """
    rho_w = -rho_d * rho_o
    return generate_flow_data(n, G, rho_d, rho_o, rho_w, beta_d, beta_o, **kwargs)


def generate_poisson_flow_data_separable(
    n: int = 10,
    k: int = 2,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    **kwargs,
) -> dict:
    """Simulate Poisson flow count data from a *separable* SAR flow model.

    Identical to :func:`generate_poisson_flow_data` except the network
    parameter is derived as :math:`\\rho_w = -\\rho_d \\rho_o`, so it is
    **not** a free argument.  Use this function to generate training data for
    :class:`~bayespecon.models.flow.PoissonSARFlowSeparable`.

    Parameters
    ----------
    n : int, default 10
        Number of spatial units.
    k : int, default 2
        Number of destination/origin attribute columns.
    rho_d : float, default 0.3
        Destination autocorrelation parameter.
    rho_o : float, default 0.2
        Origin autocorrelation parameter.
    **kwargs
        Forwarded to :func:`generate_poisson_flow_data` (e.g. ``beta_d``,
        ``beta_o``, ``seed``, ``G``).

    Returns
    -------
    dict
        Same as :func:`generate_poisson_flow_data`.  The ``"rho_w"`` entry
        will equal ``-rho_d * rho_o``.
    """
    rho_w = -rho_d * rho_o
    return generate_poisson_flow_data(
        n=n, k=k, rho_d=rho_d, rho_o=rho_o, rho_w=rho_w, **kwargs
    )


def generate_panel_flow_data_separable(
    n: int,
    T: int,
    G: Graph,
    rho_d: float,
    rho_o: float,
    beta_d: Union[np.ndarray, list],
    beta_o: Union[np.ndarray, list],
    **kwargs,
) -> dict:
    """Simulate panel flow data from a *separable* SAR flow model.

    Identical to :func:`generate_panel_flow_data` except the network parameter
    is derived as :math:`\\rho_w = -\\rho_d \\rho_o`.  Use this function to
    generate training data for
    :class:`~bayespecon.models.flow_panel.SARFlowSeparablePanel`.

    Parameters
    ----------
    n : int
        Number of spatial units.
    T : int
        Number of time periods.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    beta_d : array-like, shape (k,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k,)
        Origin-side regression coefficients.
    **kwargs
        Forwarded to :func:`generate_panel_flow_data` (e.g. ``sigma``,
        ``sigma_alpha``, ``seed``).

    Returns
    -------
    dict
        Same as :func:`generate_panel_flow_data`.  The ``"rho_w"`` entry will
        equal ``-rho_d * rho_o``.
    """
    rho_w = -rho_d * rho_o
    return generate_panel_flow_data(
        n, T, G, rho_d, rho_o, rho_w, beta_d, beta_o, **kwargs
    )


def generate_panel_poisson_flow_data_separable(
    n: int,
    T: int,
    G: Graph,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    **kwargs,
) -> dict:
    """Simulate panel Poisson flow data from a *separable* SAR flow model.

    Identical to :func:`generate_panel_poisson_flow_data` except the network
    parameter is derived as :math:`\\rho_w = -\\rho_d \\rho_o`.  Use this
    function to generate training data for
    :class:`~bayespecon.models.flow_panel.PoissonSARFlowSeparablePanel`.

    Parameters
    ----------
    n : int
        Number of spatial units.
    T : int
        Number of time periods.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    rho_d : float, default 0.3
        Destination autocorrelation parameter.
    rho_o : float, default 0.2
        Origin autocorrelation parameter.
    **kwargs
        Forwarded to :func:`generate_panel_poisson_flow_data` (e.g.
        ``beta_d``, ``beta_o``, ``seed``).

    Returns
    -------
    dict
        Same as :func:`generate_panel_poisson_flow_data`.  The ``"rho_w"``
        entry will equal ``-rho_d * rho_o``.
    """
    rho_w = -rho_d * rho_o
    return generate_panel_poisson_flow_data(
        n=n, T=T, G=G, rho_d=rho_d, rho_o=rho_o, rho_w=rho_w, **kwargs
    )
