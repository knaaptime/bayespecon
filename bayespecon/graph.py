"""Kronecker-product spatial weight utilities for flow (O-D) models.

Provides helpers to build destination, origin, and network weight matrices
from a single n×n :class:`libpysal.graph.Graph` using Kronecker products,
following :cite:t:`lesage2008SpatialEconometric`.

The N×N flow weight matrices are:

.. math::

    W_d = I_n \\otimes W \\quad (\\text{destination effects})

    W_o = W \\otimes I_n \\quad (\\text{origin effects})

    W_w = W \\otimes W \\quad (\\text{network / origin-destination effects})

where :math:`N = n^2` is the total number of O-D pairs and :math:`n` is the
number of spatial units.

Examples
--------
>>> import libpysal
>>> G = libpysal.graph.Graph.build_contiguity(gdf).transform("r")
>>> W_d, W_o, W_w = flow_weight_matrices(G).values()
>>> dm = flow_design_matrix(X, dist=dist_matrix)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from libpysal.graph import Graph


def sparse_trace_WtW_plus_WW(W: sp.spmatrix) -> float:
    r"""Return :math:`\operatorname{tr}(W^\top W) + \operatorname{tr}(W^2)`.

    Computed via sparse element-wise operations
    (:math:`\|W\|_F^2 + \sum_{ij} W_{ij} W_{ji}`), which is :math:`O(\mathrm{nnz})`
    rather than :math:`O(n^2)`.  This quantity appears in many spatial
    Lagrange-multiplier information-matrix formulas.
    """
    return float(W.power(2).sum() + W.multiply(W.T).sum())


def _validate_graph(G: Graph) -> sp.csr_matrix:
    """Validate a :class:`libpysal.graph.Graph` and return its CSR sparse matrix.

    Parameters
    ----------
    G :
        A :class:`libpysal.graph.Graph`. Must be row-standardised.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-compressed representation of *G*'s adjacency matrix.

    Raises
    ------
    TypeError
        If *G* is not a :class:`libpysal.graph.Graph`.

    Warns
    -----
    UserWarning
        If *G* does not appear to be row-standardised.
    """
    if not isinstance(G, Graph):
        raise TypeError(
            f"G must be a libpysal.graph.Graph, got {type(G).__name__}. "
            "Convert a legacy libpysal.weights.W with Graph.from_W(w)."
        )
    W_csr = G.sparse.tocsr().astype(np.float64)
    transform = getattr(G, "transformation", None)
    row_std = transform in ("r", "R")
    if not row_std:
        row_sums = np.asarray(W_csr.sum(axis=1)).ravel()
        row_std = bool(np.allclose(row_sums, 1.0, atol=1e-6))
    if not row_std:
        warnings.warn(
            "G does not appear to be row-standardised (row sums ≠ 1). "
            "Flow weight matrices assume W is row-standardised. "
            "Apply graph.transform('r') before calling flow utilities.",
            UserWarning,
            stacklevel=3,
        )
    return W_csr


def destination_weights(G: Graph) -> sp.csr_matrix:
    """Build the N×N destination weight matrix :math:`W_d = I_n \\otimes W`.

    A non-zero entry at position :math:`(i \\cdot n + j,\\; i \\cdot n + k)`
    indicates that destination *k* is a spatial neighbour of destination *j*
    (for flows originating from *i*).

    Parameters
    ----------
    G :
        Row-standardised :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    scipy.sparse.csr_matrix
        :math:`N \\times N` destination weight matrix (:math:`N = n^2`).
    """
    W = _validate_graph(G)
    n = W.shape[0]
    return sp.kron(sp.eye(n, format="csr"), W, format="csr")


def origin_weights(G: Graph) -> sp.csr_matrix:
    """Build the N×N origin weight matrix :math:`W_o = W \\otimes I_n`.

    A non-zero entry at position :math:`(i \\cdot n + j,\\; k \\cdot n + j)`
    indicates that origin *k* is a spatial neighbour of origin *i*
    (for flows terminating at *j*).

    Parameters
    ----------
    G :
        Row-standardised :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    scipy.sparse.csr_matrix
        :math:`N \\times N` origin weight matrix (:math:`N = n^2`).
    """
    W = _validate_graph(G)
    n = W.shape[0]
    return sp.kron(W, sp.eye(n, format="csr"), format="csr")


def network_weights(G: Graph) -> sp.csr_matrix:
    """Build the N×N network weight matrix :math:`W_w = W \\otimes W`.

    Captures joint origin-destination neighbourhood structure: a non-zero
    entry at :math:`(i \\cdot n + j,\\; k \\cdot n + l)` indicates that
    origin *k* neighbours *i* AND destination *l* neighbours *j*.

    Parameters
    ----------
    G :
        Row-standardised :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    scipy.sparse.csr_matrix
        :math:`N \\times N` network weight matrix (:math:`N = n^2`).
    """
    W = _validate_graph(G)
    return sp.kron(W, W, format="csr")


def flow_weight_matrices(G: Graph) -> dict[str, sp.csr_matrix]:
    """Build all three N×N flow weight matrices from a single Graph.

    Parameters
    ----------
    G :
        Row-standardised :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    dict[str, scipy.sparse.csr_matrix]
        Dictionary with keys ``"destination"``, ``"origin"``, ``"network"``.
    """
    W = _validate_graph(G)
    n = W.shape[0]
    eye = sp.eye(n, format="csr")
    return {
        "destination": sp.kron(eye, W, format="csr"),
        "origin": sp.kron(W, eye, format="csr"),
        "network": sp.kron(W, W, format="csr"),
    }


@dataclass
class FlowDesignMatrix:
    """Combined design matrix for an O-D flow regression.

    Attributes
    ----------
    X_dest : np.ndarray, shape (N, k)
        Destination-side characteristics: :math:`\\iota_n \\otimes X`.
    X_orig : np.ndarray, shape (N, k)
        Origin-side characteristics: :math:`X \\otimes \\iota_n`.
    X_intra : np.ndarray, shape (N, k)
        Intra-zonal characteristics (non-zero only on the diagonal of the
        flow matrix): rows of *X* selected by ``vec(I_n)``.
    intra_indicator : np.ndarray, shape (N,)
        Binary indicator for diagonal (intra-zonal) O-D pairs.
    dist_vec : np.ndarray or None, shape (N,)
        Vectorised distance/cost matrix ``vec(dist)`` if provided.
    combined : np.ndarray, shape (N, p)
        Full design matrix ready for regression.  Column order:
        intercept | intra_indicator | X_dest (k cols) | X_orig (k cols) |
        X_intra (k cols) [| dist (1 col if provided)].
    feature_names : list[str]
        Column labels for *combined* aligned with ``beta`` in the flow model.
    n : int
        Number of spatial units (*n*).  Flow count is :math:`N = n^2`.
    k : int
        Number of regional attribute columns in *X*.
    """

    X_dest: np.ndarray
    X_orig: np.ndarray
    X_intra: np.ndarray
    intra_indicator: np.ndarray
    dist_vec: np.ndarray | None
    combined: np.ndarray
    feature_names: list
    n: int
    k: int


def flow_design_matrix(
    X: np.ndarray,
    col_names: list[str] | None = None,
    dist: np.ndarray | None = None,
) -> FlowDesignMatrix:
    """Build a flow regression design matrix from regional attribute data.

    Constructs the standard LeSage-Fischer O-D design matrix with separate
    destination, origin, and intra-zonal blocks, plus an optional distance
    column, following :cite:t:`lesage2008SpatialEconometric` (Section 4.2).

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Array of *k* regional attributes for *n* spatial units.
        Should **not** include an intercept column.
    col_names : list[str], optional
        Names for the *k* columns of *X*.  Defaults to ``["x0", "x1", ...]``.
    dist : np.ndarray, shape (n, n), optional
        Distance or cost matrix.  If provided, ``vec(dist)`` is appended as
        the last column of *combined*.

    Returns
    -------
    FlowDesignMatrix
        Dataclass with all sub-matrices and a combined design matrix.

    Notes
    -----
    The full beta vector is structured as:

    .. math::

        \\beta = [\\alpha,\\; \\alpha_i,\\; \\beta_d^1 \\ldots \\beta_d^k,\\;
                 \\beta_o^1 \\ldots \\beta_o^k,\\; \\beta_i^1 \\ldots \\beta_i^k
                 [,\\; \\gamma]]

    matching ``f2_sarfm.m`` from the LeSage spatial flows toolbox.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    n, k = X_arr.shape
    N = n * n
    ones_n = np.ones((n, 1), dtype=np.float64)

    # Intra-zonal selector: vec(I_n), length N
    intra_indicator = np.eye(n, dtype=np.float64).ravel()  # 1 on diagonal cells

    # Destination characteristics: kron(ones_n, X) — each row repeats its destination X
    X_dest = np.kron(ones_n, X_arr)  # (N, k)

    # Origin characteristics: kron(X, ones_n) — each row repeats its origin X
    X_orig = np.kron(X_arr, ones_n)  # (N, k)

    # Intra-zonal characteristics: non-zero only on diagonal cells
    X_intra = intra_indicator[:, None] * X_dest  # (N, k); zeros off-diagonal

    # Intercept column
    intercept = np.ones((N, 1), dtype=np.float64)

    # Assemble combined design matrix: [1, ia, X_dest, X_orig, X_intra, (dist)]
    parts = [intercept, intra_indicator[:, None], X_dest, X_orig, X_intra]
    names = ["intercept", "intra_indicator"]

    if col_names is None:
        col_names = [f"x{i}" for i in range(k)]
    names += [f"dest_{c}" for c in col_names]
    names += [f"orig_{c}" for c in col_names]
    names += [f"intra_{c}" for c in col_names]

    dist_vec: np.ndarray | None = None
    if dist is not None:
        dist_arr = np.asarray(dist, dtype=np.float64)
        if dist_arr.shape != (n, n):
            raise ValueError(
                f"dist must have shape ({n}, {n}) to match X, got {dist_arr.shape}."
            )
        dist_vec = dist_arr.ravel()
        parts.append(dist_vec[:, None])
        names.append("dist")

    combined = np.concatenate(parts, axis=1)

    return FlowDesignMatrix(
        X_dest=X_dest,
        X_orig=X_orig,
        X_intra=X_intra,
        intra_indicator=intra_indicator,
        dist_vec=dist_vec,
        combined=combined,
        feature_names=names,
        n=n,
        k=k,
    )


def flow_design_matrix_with_orig(
    Xd: np.ndarray,
    Xo: np.ndarray,
    col_names: list[str] | None = None,
    dist: np.ndarray | None = None,
) -> FlowDesignMatrix:
    """Build a flow design matrix with separate destination and origin data.

    This is a convenience wrapper around :func:`flow_design_matrix` that
    handles the common pattern of building a design matrix from destination
    attributes *Xd* and then splicing in separate origin attributes *Xo*.

    :func:`flow_design_matrix` uses a single attribute matrix *X* for both
    the destination and origin blocks (``X_dest = kron(ones_n, X)`` and
    ``X_orig = kron(X, ones_n)``).  When destination and origin attributes
    differ, call this function instead — it replaces the origin block with
    ``kron(Xo, ones_n)`` and updates ``X_intra`` accordingly.

    Parameters
    ----------
    Xd : np.ndarray, shape (n, k)
        Destination-side regional attribute matrix (no intercept).
    Xo : np.ndarray, shape (n, k)
        Origin-side regional attribute matrix (same columns as *Xd*).
    col_names : list[str], optional
        Names for the *k* columns.  Defaults to ``["x0", "x1", ...]``.
    dist : np.ndarray, shape (n, n), optional
        Distance / cost matrix appended as the last predictor.

    Returns
    -------
    FlowDesignMatrix
        Dataclass with ``X_orig`` and ``combined`` updated to reflect *Xo*.

    Raises
    ------
    ValueError
        If *Xd* and *Xo* have incompatible shapes.

    Examples
    --------
    >>> import numpy as np
    >>> n, k = 4, 2
    >>> Xd = np.ones((n, k))
    >>> Xo = 2 * np.ones((n, k))
    >>> dm = flow_design_matrix_with_orig(Xd, Xo)
    >>> dm.X_orig.shape
    (16, 2)
    >>> np.allclose(dm.X_orig[:, 0], 2.0)
    True
    """
    Xd_arr = np.asarray(Xd, dtype=np.float64)
    Xo_arr = np.asarray(Xo, dtype=np.float64)
    if Xd_arr.ndim == 1:
        Xd_arr = Xd_arr[:, None]
    if Xo_arr.ndim == 1:
        Xo_arr = Xo_arr[:, None]
    if Xd_arr.shape[0] != Xo_arr.shape[0]:
        raise ValueError(
            f"Xd and Xo must have the same number of rows, "
            f"got {Xd_arr.shape[0]} and {Xo_arr.shape[0]}."
        )
    if Xd_arr.shape[1] != Xo_arr.shape[1]:
        raise ValueError(
            f"Xd and Xo must have the same number of columns, "
            f"got {Xd_arr.shape[1]} and {Xo_arr.shape[1]}."
        )

    # Build base design using Xd for both dest and orig blocks
    design = flow_design_matrix(Xd_arr, col_names=col_names, dist=dist)

    # Override the X_orig block with actual Xo: kron(Xo, ones_n)
    k = Xd_arr.shape[1]
    n = Xd_arr.shape[0]
    ones_n = np.ones((n, 1), dtype=np.float64)
    X_orig_actual = np.kron(Xo_arr, ones_n)  # (N, k)

    # Splice into combined: intercept | ia | Xd_block | Xo_block | Xi_block [| dist]
    combined = design.combined.copy()
    combined[:, 2 + k : 2 + 2 * k] = X_orig_actual

    return FlowDesignMatrix(
        X_dest=design.X_dest,
        X_orig=X_orig_actual,
        X_intra=design.X_intra,
        intra_indicator=design.intra_indicator,
        dist_vec=design.dist_vec,
        combined=combined,
        feature_names=design.feature_names,
        n=design.n,
        k=design.k,
    )
