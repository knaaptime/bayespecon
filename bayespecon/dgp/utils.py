"""Shared utilities for DGP simulation in bayespecon."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from libpysal.graph import Graph


def ensure_rng(
    rng: np.random.Generator | None = None, seed: int | None = None
) -> np.random.Generator:
    """Return a reproducible NumPy random generator.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Existing generator. If provided it takes precedence over ``seed``.
    seed : int, optional
        Seed for constructing a new generator when ``rng`` is not supplied.

    Returns
    -------
    np.random.Generator
        Random number generator used by simulation functions.
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def row_standardize(W: np.ndarray) -> np.ndarray:
    """Row-standardize a dense weights matrix.

    Each row of *W* is divided by its row sum so that
    ``W[i, :].sum() == 1`` for every isolated-free unit *i*.

    Parameters
    ----------
    W : np.ndarray
        Dense square matrix.

    Returns
    -------
    np.ndarray
        Row-standardized matrix, with zero-sum rows left unchanged.

    Notes
    -----
    Rows whose sum is exactly ``0`` (typically *isolates* — units with
    no neighbours) are returned untouched: dividing by zero would
    introduce ``NaN`` entries that propagate through every subsequent
    spatial product.  This means the returned matrix is row-stochastic
    on the **non-isolated** rows only; the isolate rows remain rows of
    zeros.  Downstream consumers that rely on every eigenvalue of the
    row-standardised weight matrix being ``≤ 1`` (e.g. the SAR stability
    domain ``ρ ∈ (-1, 1)``) should drop or reconnect isolates beforehand
    rather than rely on this routine to do so.
    """
    W = np.asarray(W, dtype=float)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return W / rs


def dense_to_graph(W_dense: np.ndarray, row_standardize_weights: bool = False) -> Graph:
    """Convert dense weights matrix to a libpysal Graph.

    Parameters
    ----------
    W_dense : np.ndarray
        Dense square weights matrix.
    row_standardize_weights : bool, default=False
        Whether to row-standardize before conversion.

    Returns
    -------
    Graph
        Graph representation of the same sparse structure.
    """
    W_arr = np.asarray(W_dense, dtype=float)
    if row_standardize_weights:
        W_arr = row_standardize(W_arr)

    n = W_arr.shape[0]
    focal, neighbor, weight = [], [], []
    for i in range(n):
        for j in range(n):
            wij = W_arr[i, j]
            if wij != 0.0:
                focal.append(i)
                neighbor.append(j)
                weight.append(wij)

    g = Graph.from_arrays(
        np.asarray(focal, dtype=int),
        np.asarray(neighbor, dtype=int),
        np.asarray(weight, dtype=float),
    )
    return g.transform("r")


def rook_grid_weights(n_side: int) -> tuple[np.ndarray, Graph]:
    """Build row-standardized rook-contiguity weights on an ``n_side x n_side`` grid.

    Parameters
    ----------
    n_side : int
        Number of rows and columns in the square grid.

    Returns
    -------
    tuple[np.ndarray, Graph]
        Dense and Graph forms of the same row-standardized weights.
    """
    n_side = int(n_side)
    if n_side <= 0:
        raise ValueError("n must be a positive integer when generating a default grid.")

    n_obs = n_side * n_side
    W = np.zeros((n_obs, n_obs), dtype=float)

    for r in range(n_side):
        for c in range(n_side):
            i = r * n_side + c
            if r > 0:
                W[i, (r - 1) * n_side + c] = 1.0
            if r < n_side - 1:
                W[i, (r + 1) * n_side + c] = 1.0
            if c > 0:
                W[i, r * n_side + (c - 1)] = 1.0
            if c < n_side - 1:
                W[i, r * n_side + (c + 1)] = 1.0

    W = row_standardize(W)
    return W, dense_to_graph(W)


def weights_from_geodataframe(
    gdf: Any,
    contiguity: str = "queen",
    k: int = 4,
    distance_threshold: float | None = None,
) -> Graph:
    """Build a row-standardized Graph from GeoDataFrame geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geodataframe with geometry column.
    contiguity : {"queen", "rook", "knn", "distance"}, default="queen"
        Rule to build neighbor structure.
    k : int, default=4
        Number of neighbors for ``contiguity='knn'``.
    distance_threshold : float, optional
        Distance cutoff for ``contiguity='distance'``.

    Returns
    -------
    Graph
        Row-standardized graph built from geometry.
    """
    if gdf is None:
        raise ValueError("gdf must be provided when building weights from geometry.")
    if not hasattr(gdf, "geometry"):
        raise TypeError(
            "gdf must be a GeoDataFrame-like object with a geometry column."
        )

    mode = contiguity.lower()
    if mode == "queen":
        g = Graph.build_contiguity(gdf, rook=False)
    elif mode == "rook":
        g = Graph.build_contiguity(gdf, rook=True)
    elif mode == "knn":
        g = Graph.build_knn(gdf, k=int(k))
    elif mode == "distance":
        if distance_threshold is None:
            raise ValueError(
                "distance_threshold must be supplied when contiguity='distance'."
            )
        g = Graph.build_distance_band(
            gdf, threshold=float(distance_threshold), binary=True
        )
    else:
        raise ValueError(
            "contiguity must be one of {'queen', 'rook', 'knn', 'distance'}."
        )

    return g.transform("r")


def resolve_weights(
    W: Graph | sp.spmatrix | np.ndarray | None = None,
    gdf: Any | None = None,
    n: int | None = None,
    contiguity: str = "queen",
    k: int = 4,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray, Graph]:
    """Resolve user-supplied spatial structure to dense matrix and Graph.

    Parameters
    ----------
    W : Graph or sparse/dense matrix, optional
        Explicit spatial structure. If provided together with ``gdf``, the
        matrix/graph is used and checked for dimensional compatibility with
        the GeoDataFrame.
    gdf : geopandas.GeoDataFrame, optional
        Used only when ``W`` is not supplied.
    n : int, optional
        If provided without ``W`` and ``gdf``, generate a default
        rook-contiguity square grid with side length ``n``.
    contiguity : str, default="queen"
        GeoDataFrame neighbor construction mode.
    k : int, default=4
        KNN neighbor count when ``contiguity='knn'``.
    distance_threshold : float, optional
        Distance threshold when ``contiguity='distance'``.

    Returns
    -------
    tuple[np.ndarray, Graph]
        ``(W_dense, W_graph)`` both row-standardized.
    """
    if W is not None:
        gdf_n = len(gdf) if gdf is not None else None
        n_obs = int(n) if n is not None else None
        if isinstance(W, Graph):
            g = W.transform("r")
            Wd = g.sparse.toarray().astype(float)
            if gdf_n is not None and Wd.shape[0] != gdf_n:
                raise ValueError(
                    "W and gdf must describe the same number of spatial units."
                )
            if n_obs is not None and Wd.shape[0] != n_obs:
                raise ValueError("n must match the size implied by W/gdf.")
            return Wd, g
        if sp.issparse(W):
            Wd = row_standardize(W.toarray().astype(float))
            if gdf_n is not None and Wd.shape[0] != gdf_n:
                raise ValueError(
                    "W and gdf must describe the same number of spatial units."
                )
            if n_obs is not None and Wd.shape[0] != n_obs:
                raise ValueError("n must match the size implied by W/gdf.")
            return Wd, dense_to_graph(Wd)
        Wd = row_standardize(np.asarray(W, dtype=float))
        if gdf_n is not None and Wd.shape[0] != gdf_n:
            raise ValueError(
                "W and gdf must describe the same number of spatial units."
            )
        if n_obs is not None and Wd.shape[0] != n_obs:
            raise ValueError("n must match the size implied by W/gdf.")
        return Wd, dense_to_graph(Wd)

    if gdf is None:
        if n is None:
            raise ValueError("Provide either W, gdf, or n.")
        return rook_grid_weights(int(n))

    g = weights_from_geodataframe(
        gdf, contiguity=contiguity, k=k, distance_threshold=distance_threshold
    )
    if n is not None and g.sparse.shape[0] != int(n):
        raise ValueError("n must match the size implied by W/gdf.")
    return g.sparse.toarray().astype(float), g


def _hetero_scale(X: np.ndarray, sigma: float) -> np.ndarray:
    """Compute observation-specific standard deviations for heteroskedastic errors.

    When ``err_hetero=True`` in a DGP simulator, each observation's error
    standard deviation is scaled by the norm of its regressor row so that
    units with larger covariate values receive noisier shocks.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` (including intercept column if
        applicable).
    sigma : float
        Base innovation standard deviation.

    Returns
    -------
    np.ndarray
        Array of shape ``(n,)`` with element-wise standard deviations
        ``sigma * sqrt(1 + ||x_i||^2)``.
    """
    return sigma * np.sqrt(1.0 + np.sum(X**2, axis=1))


def make_design_matrix(
    rng: np.random.Generator, n: int, k: int = 1, add_intercept: bool = True
) -> np.ndarray:
    """Generate synthetic design matrix.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator.
    n : int
        Number of observations.
    k : int, default=1
        Number of non-intercept regressors.
    add_intercept : bool, default=True
        Whether to prepend a constant column.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(n, k + int(add_intercept))``.
    """
    Z = rng.standard_normal((n, k))
    if add_intercept:
        return np.column_stack([np.ones(n), Z])
    return Z


def panel_index(N: int, T: int) -> pd.DataFrame:
    """Create time-first panel index DataFrame.

    Parameters
    ----------
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    pd.DataFrame
        Columns ``unit`` and ``time`` matching stacked time-first ordering.
    """
    units = np.tile(np.arange(N), T)
    times = np.repeat(np.arange(T), N)
    return pd.DataFrame({"unit": units, "time": times})


def make_output_geodataframe(
    y: np.ndarray,
    X: np.ndarray,
    gdf: Any | None = None,
    geometry_type: str = "polygon",
) -> Any:
    """Create a GeoDataFrame carrying simulated ``y`` and ``X`` columns.

    Parameters
    ----------
    y : np.ndarray
        Simulated dependent variable of shape ``(n_obs,)``.
    X : np.ndarray
        Simulated design matrix of shape ``(n_obs, k)``.
    gdf : geopandas.GeoDataFrame, optional
        Existing geometry source. If provided, its geometry is reused.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry to generate when ``gdf`` is not provided.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with columns ``y``, ``X_0``, ``X_1``, ... and geometry.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point, box
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "create_gdf=True requires optional dependencies geopandas and shapely."
        ) from exc

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array when building output GeoDataFrame.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array when building output GeoDataFrame.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of observations.")

    n_obs = y.shape[0]
    out = {"y": y}
    for j in range(X.shape[1]):
        out[f"X_{j}"] = X[:, j]

    if gdf is not None:
        if len(gdf) != n_obs:
            raise ValueError(
                "Provided gdf must have the same number of rows as y and X."
            )
        out_gdf = gdf.copy()
        for col, values in out.items():
            out_gdf[col] = values
        return out_gdf

    mode = str(geometry_type).lower()
    if mode not in {"point", "polygon"}:
        raise ValueError("geometry_type must be one of {'point', 'polygon'}.")

    n_cols = int(np.ceil(np.sqrt(n_obs)))
    int(np.ceil(n_obs / n_cols))

    geoms = []
    for idx in range(n_obs):
        r, c = divmod(idx, n_cols)
        if mode == "point":
            geoms.append(Point(c + 0.5, r + 0.5))
        else:
            geoms.append(box(c, r, c + 1.0, r + 1.0))

    return gpd.GeoDataFrame(out, geometry=geoms)


def make_panel_output_geodataframe(
    y: np.ndarray,
    X: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    N: int,
    T: int,
    *,
    gdf: Any | None = None,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> Any:
    """Create panel GeoDataFrame output from simulated arrays.

    Parameters
    ----------
    y : np.ndarray
        Stacked dependent variable of shape ``(N*T,)``.
    X : np.ndarray
        Stacked design matrix of shape ``(N*T, k)``.
    unit : np.ndarray
        Unit index of shape ``(N*T,)``.
    time : np.ndarray
        Time index of shape ``(N*T,)``.
    N : int
        Number of spatial units.
    T : int
        Number of time periods.
    gdf : geopandas.GeoDataFrame, optional
        N-row geometry source. If provided its geometry is reused; any
        non-geometry columns are dropped before merging.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``gdf`` is not provided.
    wide : bool, default=False
        If True return a single N-row wide GeoDataFrame with columns
        ``y_t0``, ``y_t1``, ..., ``X_0_t0``, ``X_0_t1``, ...
        If False return ``(unit_gdf, long_panel_df)`` where ``unit_gdf``
        carries geometry only and ``long_panel_df`` carries ``unit``,
        ``time``, ``y``, ``X_0``, ...

    Returns
    -------
    GeoDataFrame or tuple[GeoDataFrame, DataFrame]
        Wide GeoDataFrame when ``wide=True``; otherwise
        ``(unit_gdf, long_panel_df)`` 2-tuple.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point, box
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "create_gdf=True requires optional dependencies geopandas and shapely."
        ) from exc

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if y.shape[0] != N * T:
        raise ValueError(f"y length {y.shape[0]} does not match N*T={N * T}.")

    # Build N-row unit GeoDataFrame (geometry only).
    if gdf is not None:
        if len(gdf) != N:
            raise ValueError("Provided gdf must have N rows.")
        unit_gdf = gdf[[gdf.geometry.name]].copy()
    else:
        mode = str(geometry_type).lower()
        if mode not in {"point", "polygon"}:
            raise ValueError("geometry_type must be one of {'point', 'polygon'}.")
        n_cols = int(np.ceil(np.sqrt(N)))
        geoms = []
        for idx_i in range(N):
            r, c = divmod(idx_i, n_cols)
            if mode == "point":
                geoms.append(Point(c + 0.5, r + 0.5))
            else:
                geoms.append(box(c, r, c + 1.0, r + 1.0))
        unit_gdf = gpd.GeoDataFrame({"unit_id": np.arange(N)}, geometry=geoms)
        unit_gdf = unit_gdf[[unit_gdf.geometry.name]]

    unit_gdf = unit_gdf.reset_index(drop=True)

    # Build long panel DataFrame.
    long_data: dict[str, Any] = {"unit": unit, "time": time, "y": y}
    for j in range(X.shape[1]):
        long_data[f"X_{j}"] = X[:, j]
    long_df = pd.DataFrame(long_data)

    if not wide:
        return unit_gdf, long_df

    # Pivot to wide format: one row per unit.
    x_cols = [c for c in long_df.columns if c.startswith("X_")]
    value_cols = ["y"] + x_cols
    wide_df = long_df.pivot(index="unit", columns="time", values=value_cols)
    wide_df.columns = [f"{col}_t{t}" for col, t in wide_df.columns]
    wide_df = wide_df.reset_index(drop=True)

    combined = pd.concat([unit_gdf.reset_index(drop=True), wide_df], axis=1)
    return gpd.GeoDataFrame(combined, geometry=unit_gdf.geometry.name)


def synth_point_geodataframe(n: int) -> Any:
    """Synthesize a point GeoDataFrame on a √n × √n grid.

    Used by flow DGPs when the user does not supply a ``gdf`` so that a
    consistent set of centroid coordinates is available for distance
    computation.

    Parameters
    ----------
    n : int
        Number of points (rows) to generate.

    Returns
    -------
    geopandas.GeoDataFrame
        n-row GeoDataFrame whose ``geometry`` column contains
        ``Point(c + 0.5, r + 0.5)`` cells laid out row-major on a
        ``ceil(sqrt(n)) × ceil(sqrt(n))`` grid.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "synth_point_geodataframe requires optional dependencies "
            "geopandas and shapely."
        ) from exc

    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    n_cols = int(np.ceil(np.sqrt(n)))
    geoms = []
    for idx in range(n):
        r, c = divmod(idx, n_cols)
        geoms.append(Point(c + 0.5, r + 0.5))

    return gpd.GeoDataFrame({"unit_id": np.arange(n)}, geometry=geoms)


def pairwise_distance_matrix(gdf: Any) -> np.ndarray:
    """Compute the dense ``n × n`` pairwise Euclidean distance matrix from a
    GeoDataFrame.

    Polygons are downcast to centroids first; points are used directly.
    The resulting matrix has zero diagonal by construction and is symmetric.

    This is the dense-matrix complement to
    :meth:`libpysal.graph.Graph.build_distance_band`, which returns a
    thresholded sparse graph; flow regressions need the full pairwise
    distances to populate an O-D design column.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geometry source.  Polygons are reduced to centroids automatically.

    Returns
    -------
    np.ndarray
        ``(n, n)`` Euclidean distance matrix in the same units / CRS as
        ``gdf.geometry``.
    """
    from scipy.spatial.distance import cdist

    if gdf is None or not hasattr(gdf, "geometry"):
        raise TypeError("pairwise_distance_matrix requires a GeoDataFrame-like object.")

    geom = gdf.geometry
    # geopandas' .centroid is a no-op for points and yields polygon centroids
    # otherwise; explicitly access .x / .y to obtain coordinates.
    cents = geom.centroid
    coords = np.column_stack([cents.x.to_numpy(), cents.y.to_numpy()])
    return cdist(coords, coords)


def _resolve_flow_geometry(
    n: int | None = None,
    G: Graph | None = None,
    gdf: Any | None = None,
    knn_k: int = 4,
    default_n: int = 25,
) -> tuple[int, Graph, Any]:
    """Resolve ``(n, G, gdf)`` for flow DGPs from any combination of inputs.

    The flow DGPs need a row-standardised spatial graph *G* (size ``n``)
    and a point GeoDataFrame *gdf* (length ``n``) to derive a pairwise
    distance matrix.  Either, both, or neither of *G* and *gdf* may be
    supplied.  When *G* is missing it is constructed from *gdf* using
    KNN contiguity (``k = knn_k``) and row-standardised.  When *gdf* is
    missing a synthetic point grid is generated via
    :func:`synth_point_geodataframe`.

    Parameters
    ----------
    n : int, optional
        Desired number of spatial units.  Ignored when *G* or *gdf* is
        provided (the size is taken from those objects).  When neither
        is supplied, falls back to *default_n*.
    G : libpysal.graph.Graph, optional
        Pre-built spatial graph.  If supplied it is row-standardised on
        the way out (idempotent for already row-standard graphs).
    gdf : geopandas.GeoDataFrame, optional
        Point or polygon geometry.  Used both to build *G* (when
        missing) and to compute distances downstream.
    knn_k : int, default 4
        Number of nearest neighbours used when building *G* from *gdf*.
    default_n : int, default 25
        Number of units used when neither *G* nor *gdf* is provided.

    Returns
    -------
    n_actual : int
        Resolved number of spatial units.
    G : libpysal.graph.Graph
        Row-standardised graph on *n_actual* units.
    gdf : geopandas.GeoDataFrame
        Geometry source on *n_actual* units (synthesized when not
        supplied).

    Raises
    ------
    ValueError
        If both *G* and *gdf* are supplied but their sizes disagree, or
        if *n* is supplied alongside *G* / *gdf* with an inconsistent
        size.
    """
    if G is not None and gdf is not None:
        if len(gdf) != G.n_nodes:
            raise ValueError(f"gdf has {len(gdf)} rows but G has {G.n_nodes} nodes.")
        n_actual = G.n_nodes
    elif G is not None:
        n_actual = G.n_nodes
        gdf = synth_point_geodataframe(n_actual)
    elif gdf is not None:
        n_actual = len(gdf)
        k_eff = max(1, min(int(knn_k), n_actual - 1))
        G = Graph.build_knn(gdf, k=k_eff).transform("r")
    else:
        n_actual = int(n) if n is not None else int(default_n)
        gdf = synth_point_geodataframe(n_actual)
        k_eff = max(1, min(int(knn_k), n_actual - 1))
        G = Graph.build_knn(gdf, k=k_eff).transform("r")

    if n is not None and int(n) != n_actual:
        raise ValueError(f"n={n} disagrees with resolved geometry size {n_actual}.")

    # Ensure G is row-standardised
    G = G.transform("r")
    return n_actual, G, gdf
