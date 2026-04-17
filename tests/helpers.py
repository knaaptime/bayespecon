"""Synthetic data generators and constants shared across the test suite.

Import from here rather than from conftest.py to avoid sys.path issues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from libpysal.graph import Graph
from scipy.special import erf


# ---------------------------------------------------------------------------
# Sampling settings (moderate draws for reliable recovery, not too slow)
# ---------------------------------------------------------------------------

SAMPLE_KWARGS: dict = dict(
    tune=1000, draws=1500, chains=4, random_seed=42, progressbar=False
)

# Panel dimensions
PANEL_N = 5   # cross-sectional units
PANEL_T = 8   # time periods


# ---------------------------------------------------------------------------
# Spatial weight helpers
# ---------------------------------------------------------------------------

def make_rook_W(side: int) -> np.ndarray:
    """Row-standardized rook-contiguity weights on a ``side x side`` grid."""
    n = side * side
    W = np.zeros((n, n))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if r > 0:
                W[i, (r - 1) * side + c] = 1
            if r < side - 1:
                W[i, (r + 1) * side + c] = 1
            if c > 0:
                W[i, r * side + (c - 1)] = 1
            if c < side - 1:
                W[i, r * side + (c + 1)] = 1
    row_sums = W.sum(axis=1, keepdims=True)
    return W / np.where(row_sums == 0, 1, row_sums)


def make_line_W(n: int) -> np.ndarray:
    """Row-standardized line-lattice weights for ``n`` units.

    Unit ``i`` is connected to immediate neighbors ``i-1`` and ``i+1``.
    """
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    return W / np.where(row_sums == 0, 1, row_sums)


def W_to_graph(W_dense: np.ndarray) -> Graph:
    """Convert a dense weight matrix to a libpysal Graph."""
    n = W_dense.shape[0]
    focal, neighbor, weight = [], [], []
    for i in range(n):
        for j in range(n):
            if W_dense[i, j] != 0:
                focal.append(i)
                neighbor.append(j)
                weight.append(W_dense[i, j])
    return Graph.from_arrays(
        np.array(focal),
        np.array(neighbor),
        np.array(weight, dtype=float),
    )


# ---------------------------------------------------------------------------
# Cross-sectional data generators
# ---------------------------------------------------------------------------

def make_sar_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SAR data: y = (I - rho*W)^{-1}(X@beta + eps)."""
    n = W.shape[0]
    if beta is None:
        beta = np.array([1.0, 2.0])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, len(beta) - 1))])
    eps = sigma * rng.standard_normal(n)
    y = np.linalg.solve(np.eye(n) - rho * W, X @ beta + eps)
    return y, X


def make_sem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SEM data: u = (I - lam*W)^{-1}*eps; y = X@beta + u."""
    n = W.shape[0]
    if beta is None:
        beta = np.array([1.0, 2.0])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, len(beta) - 1))])
    eps = sigma * rng.standard_normal(n)
    u = np.linalg.solve(np.eye(n) - lam * W, eps)
    y = X @ beta + u
    return y, X


def make_slx_data(
    rng: np.random.Generator,
    W: np.ndarray,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SLX data: y = X@beta1 + W@X_noint@beta2 + eps."""
    n = W.shape[0]
    if beta1 is None:
        beta1 = np.array([1.0, 2.0])
    if beta2 is None:
        beta2 = np.array([0.8])
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Wx = W @ X[:, 1:]
    y = X @ beta1 + Wx @ beta2 + sigma * rng.standard_normal(n)
    return y, X


def make_sdm_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SDM data: y = (I-rho*W)^{-1}(X@beta1 + WX_noint@beta2 + eps)."""
    n = W.shape[0]
    if beta1 is None:
        beta1 = np.array([1.0, 2.0])
    if beta2 is None:
        beta2 = np.array([0.8])
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Wx = W @ X[:, 1:]
    eps = sigma * rng.standard_normal(n)
    y = np.linalg.solve(np.eye(n) - rho * W, X @ beta1 + Wx @ beta2 + eps)
    return y, X


def make_sdem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SDEM data: y = X@beta1 + WX_noint@beta2 + (I-lam*W)^{-1}eps."""
    n = W.shape[0]
    if beta1 is None:
        beta1 = np.array([1.0, 2.0])
    if beta2 is None:
        beta2 = np.array([0.8])
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Wx = W @ X[:, 1:]
    u = np.linalg.solve(np.eye(n) - lam * W, sigma * rng.standard_normal(n))
    y = X @ beta1 + Wx @ beta2 + u
    return y, X


# ---------------------------------------------------------------------------
# Panel data generators  (time-first stacking: obs t*N+i → unit i)
# ---------------------------------------------------------------------------

def make_panel_ols_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate panel OLS data with unit random effects."""
    if beta is None:
        beta = np.array([1.0, 2.0])
    alpha = rng.normal(0, sigma_alpha, N)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = np.column_stack([np.ones(N), rng.standard_normal(N)])
        yt = Xt @ beta + alpha + sigma * rng.standard_normal(N)
        y_list.append(yt)
        X_list.append(Xt)
    y = np.concatenate(y_list)
    X = np.vstack(X_list)
    units = np.tile(np.arange(N), T)
    times = np.repeat(np.arange(T), N)
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sar_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SAR panel data with unit random effects."""
    if beta is None:
        beta = np.array([1.0, 2.0])
    alpha = rng.normal(0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - rho * W)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = np.column_stack([np.ones(N), rng.standard_normal(N)])
        eps = sigma * rng.standard_normal(N)
        yt = A_inv @ (Xt @ beta + alpha + eps)
        y_list.append(yt)
        X_list.append(Xt)
    y = np.concatenate(y_list)
    X = np.vstack(X_list)
    units = np.tile(np.arange(N), T)
    times = np.repeat(np.arange(T), N)
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SEM panel data with unit random effects."""
    if beta is None:
        beta = np.array([1.0, 2.0])
    alpha = rng.normal(0, sigma_alpha, N)
    A_inv = np.linalg.inv(np.eye(N) - lam * W)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = np.column_stack([np.ones(N), rng.standard_normal(N)])
        u = A_inv @ (sigma * rng.standard_normal(N))
        yt = Xt @ beta + alpha + u
        y_list.append(yt)
        X_list.append(Xt)
    y = np.concatenate(y_list)
    X = np.vstack(X_list)
    units = np.tile(np.arange(N), T)
    times = np.repeat(np.arange(T), N)
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


# ---------------------------------------------------------------------------
# Spatial probit data generator
# ---------------------------------------------------------------------------

def make_spatial_probit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    sigma_a: float = 0.8,
    n_per_region: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate SpatialProbit data in matrix mode.

    Returns
    -------
    y : np.ndarray
        Binary outcomes, shape ``(nobs,)``.
    X : np.ndarray
        Covariates including intercept, shape ``(nobs, k)``.
    region_ids : np.ndarray
        Region code for each observation, shape ``(nobs,)``.
    """
    m = W.shape[0]
    if beta is None:
        beta = np.array([0.3, 1.0])

    # Spatially correlated regional effects: a = (I - rho W)^(-1) sigma_a z
    a = np.linalg.solve(np.eye(m) - rho * W, sigma_a * rng.standard_normal(m))

    nobs = m * n_per_region
    region_ids = np.repeat(np.arange(m), n_per_region)
    x1 = rng.standard_normal(nobs)
    X = np.column_stack([np.ones(nobs), x1])

    eta = X @ beta + a[region_ids]
    p = 0.5 * (1.0 + erf(eta / np.sqrt(2.0)))
    y = rng.binomial(1, p).astype(float)
    return y, X, region_ids
