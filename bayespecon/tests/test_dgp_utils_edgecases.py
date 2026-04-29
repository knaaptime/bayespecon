"""Fast unit tests for bayespecon.dgp.utils edge cases.

Tests weights_from_geodataframe, resolve_weights, and other utility
functions that have low coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from bayespecon.dgp.utils import (
    _hetero_scale,
    dense_to_graph,
    ensure_rng,
    make_design_matrix,
    resolve_weights,
    rook_grid_weights,
    row_standardize,
    weights_from_geodataframe,
)

# ---------------------------------------------------------------------------
# ensure_rng
# ---------------------------------------------------------------------------


class TestEnsureRng:
    def test_returns_provided_rng(self):
        rng = np.random.default_rng(42)
        result = ensure_rng(rng)
        assert result is rng

    def test_creates_rng_from_seed(self):
        result = ensure_rng(seed=123)
        assert isinstance(result, np.random.Generator)

    def test_creates_default_rng(self):
        result = ensure_rng()
        assert isinstance(result, np.random.Generator)


# ---------------------------------------------------------------------------
# row_standardize
# ---------------------------------------------------------------------------


class TestRowStandardize:
    def test_already_row_standardized(self):
        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        result = row_standardize(W)
        np.testing.assert_allclose(result, W)

    def test_unnormalized(self):
        W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        result = row_standardize(W)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_isolate_row(self):
        """Row of zeros should remain zeros (no NaN)."""
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=float)
        result = row_standardize(W)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result[2], 0.0)


# ---------------------------------------------------------------------------
# dense_to_graph
# ---------------------------------------------------------------------------


class TestDenseToGraph:
    def test_round_trip(self):
        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        g = dense_to_graph(W)
        assert isinstance(g, Graph)
        W_back = g.sparse.toarray().astype(float)
        np.testing.assert_allclose(W_back, W, atol=1e-10)

    def test_with_row_standardize(self):
        W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        g = dense_to_graph(W, row_standardize_weights=True)
        W_back = g.sparse.toarray().astype(float)
        np.testing.assert_allclose(W_back.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# rook_grid_weights
# ---------------------------------------------------------------------------


class TestRookGridWeights:
    def test_basic_grid(self):
        W, g = rook_grid_weights(3)
        assert W.shape == (9, 9)
        # Corner units have 2 neighbors, edge units have 3, center has 4
        assert W[0].sum() == pytest.approx(1.0)  # row-standardized

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            rook_grid_weights(0)

    def test_n1_grid(self):
        W, g = rook_grid_weights(1)
        assert W.shape == (1, 1)
        assert W[0, 0] == 0.0  # isolated unit


# ---------------------------------------------------------------------------
# weights_from_geodataframe
# ---------------------------------------------------------------------------


class TestWeightsFromGeodataframe:
    def test_gdf_none_raises(self):
        with pytest.raises(ValueError, match="gdf must be provided"):
            weights_from_geodataframe(None)

    def test_no_geometry_raises(self):
        with pytest.raises(TypeError, match="geometry column"):
            weights_from_geodataframe({"a": [1, 2, 3]})

    def test_invalid_contiguity_raises(self):
        """Invalid contiguity mode should raise ValueError."""
        # Create a minimal GeoDataFrame-like object
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0), Point(0, 1)])
        with pytest.raises(ValueError, match="contiguity must be one of"):
            weights_from_geodataframe(gdf, contiguity="invalid")

    def test_distance_without_threshold_raises(self):
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0), Point(0, 1)])
        with pytest.raises(ValueError, match="distance_threshold must be supplied"):
            weights_from_geodataframe(gdf, contiguity="distance")

    def test_knn_contiguity(self):
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
        )
        g = weights_from_geodataframe(gdf, contiguity="knn", k=2)
        assert isinstance(g, Graph)

    def test_distance_contiguity(self):
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
        )
        g = weights_from_geodataframe(
            gdf, contiguity="distance", distance_threshold=2.0
        )
        assert isinstance(g, Graph)

    def test_rook_contiguity(self):
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)]
        )
        g = weights_from_geodataframe(gdf, contiguity="rook")
        assert isinstance(g, Graph)

    def test_queen_contiguity(self):
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)]
        )
        g = weights_from_geodataframe(gdf, contiguity="queen")
        assert isinstance(g, Graph)


# ---------------------------------------------------------------------------
# resolve_weights
# ---------------------------------------------------------------------------


class TestResolveWeights:
    def test_with_graph(self):
        W_dense, g = resolve_weights(W=_W_to_graph(_rook_W(4)))
        assert W_dense.shape == (4, 4)
        assert isinstance(g, Graph)

    def test_with_sparse(self):
        W_sp = sp.csr_matrix(_rook_W(4))
        W_dense, g = resolve_weights(W=W_sp)
        assert W_dense.shape == (4, 4)

    def test_with_ndarray(self):
        W_arr = _rook_W(4)
        W_dense, g = resolve_weights(W=W_arr)
        assert W_dense.shape == (4, 4)

    def test_with_n_only(self):
        W_dense, g = resolve_weights(n=4)
        assert W_dense.shape == (16, 16)  # 4x4 grid = 16 units

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide either W, gdf, or n"):
            resolve_weights()

    def test_graph_with_gdf_mismatch_raises(self):
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)])
        W_graph = _W_to_graph(_rook_W(4))  # 4 units, gdf has 2
        with pytest.raises(ValueError, match="same number of spatial units"):
            resolve_weights(W=W_graph, gdf=gdf)

    def test_graph_with_n_mismatch_raises(self):
        W_graph = _W_to_graph(_rook_W(4))
        with pytest.raises(ValueError, match="n must match"):
            resolve_weights(W=W_graph, n=5)


# ---------------------------------------------------------------------------
# _hetero_scale
# ---------------------------------------------------------------------------


class TestHeteroScale:
    def test_basic(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _hetero_scale(X, sigma=1.0)
        assert result.shape == (2,)
        # sigma * sqrt(1 + ||x_i||^2)
        expected = 1.0 * np.sqrt(1.0 + np.array([1 + 4, 9 + 16]))
        np.testing.assert_allclose(result, expected)

    def test_sigma_scaling(self):
        X = np.array([[1.0, 0.0]])
        result = _hetero_scale(X, sigma=2.0)
        np.testing.assert_allclose(result, 2.0 * np.sqrt(2.0))


# ---------------------------------------------------------------------------
# make_design_matrix
# ---------------------------------------------------------------------------


class TestMakeDesignMatrix:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = make_design_matrix(rng, n=10, k=2)
        assert X.shape == (10, 3)  # k + intercept

    def test_no_intercept(self):
        rng = np.random.default_rng(42)
        X = make_design_matrix(rng, n=10, k=2, add_intercept=False)
        assert X.shape == (10, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rook_W(n: int = 4) -> np.ndarray:
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def _W_to_graph(W_dense: np.ndarray) -> Graph:
    n = W_dense.shape[0]
    focal, neighbor, weight = [], [], []
    for i in range(n):
        for j in range(n):
            if W_dense[i, j] != 0:
                focal.append(i)
                neighbor.append(j)
                weight.append(W_dense[i, j])
    return Graph.from_arrays(
        np.array(focal, int), np.array(neighbor, int), np.array(weight, float)
    ).transform("r")
