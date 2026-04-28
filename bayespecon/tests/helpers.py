"""Synthetic data generators and constants shared across the test suite.

Import from here rather than from conftest.py to avoid sys.path issues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from libpysal.graph import Graph

from bayespecon import dgp

# ---------------------------------------------------------------------------
# Sampling settings (moderate draws for reliable recovery, not too slow)
# ---------------------------------------------------------------------------

SAMPLE_KWARGS: dict = dict(
    tune=1000, draws=1500, chains=4, random_seed=42, progressbar=False
)

# Panel dimensions
PANEL_N = 10  # cross-sectional units (larger N for reliable recovery)
PANEL_T = 10  # time periods


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
    ).transform("r")


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
    out = dgp.simulate_sar(W=W, rho=rho, beta=beta, sigma=sigma, rng=rng)
    return out["y"], out["X"]


def make_sem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SEM data: u = (I - lam*W)^{-1}*eps; y = X@beta + u."""
    out = dgp.simulate_sem(W=W, lam=lam, beta=beta, sigma=sigma, rng=rng)
    return out["y"], out["X"]


def make_slx_data(
    rng: np.random.Generator,
    W: np.ndarray,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SLX data: y = X@beta1 + W@X_noint@beta2 + eps."""
    out = dgp.simulate_slx(W=W, beta1=beta1, beta2=beta2, sigma=sigma, rng=rng)
    return out["y"], out["X"]


def make_sdm_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SDM data: y = (I-rho*W)^{-1}(X@beta1 + WX_noint@beta2 + eps)."""
    out = dgp.simulate_sdm(
        W=W,
        rho=rho,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        rng=rng,
    )
    return out["y"], out["X"]


def make_sdem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SDEM data: y = X@beta1 + WX_noint@beta2 + (I-lam*W)^{-1}eps."""
    out = dgp.simulate_sdem(
        W=W,
        lam=lam,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        rng=rng,
    )
    return out["y"], out["X"]


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
    out = dgp.simulate_panel_ols_fe(
        N=N,
        T=T,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=W,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
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
    out = dgp.simulate_panel_sar_fe(
        N=N,
        T=T,
        rho=rho,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=W,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
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
    out = dgp.simulate_panel_sem_fe(
        N=N,
        T=T,
        lam=lam,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=W,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_dlm_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic non-spatial panel data with unit effects."""
    out = dgp.simulate_panel_dlm_fe(
        N=N,
        T=T,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=W,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdmr_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic restricted SDM panel data with unit effects."""
    out = dgp.simulate_panel_sdmr_fe(
        N=N,
        T=T,
        rho=rho,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=W,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdmu_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    theta: float = -0.1,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic unrestricted SDM panel data with unit effects."""
    out = dgp.simulate_panel_sdmu_fe(
        N=N,
        T=T,
        rho=rho,
        phi=phi,
        theta=theta,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=W,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
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
    out = dgp.simulate_spatial_probit(
        W=W,
        rho=rho,
        beta=beta,
        sigma_a=sigma_a,
        n_per_region=n_per_region,
        rng=rng,
    )
    return out["y"], out["X"], out["region_ids"]


# ---------------------------------------------------------------------------
# Tobit data generators
# ---------------------------------------------------------------------------


def make_sar_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored SAR Tobit data using the dgp module."""
    out = dgp.simulate_sar_tobit(
        W=W,
        rho=rho,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_sem_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored SEM Tobit data using the dgp module."""
    out = dgp.simulate_sem_tobit(
        W=W,
        lam=lam,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_sdm_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored SDM Tobit data using the dgp module."""
    out = dgp.simulate_sdm_tobit(
        W=W,
        rho=rho,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_panel_sar_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored panel SAR FE Tobit data using the dgp module."""
    out = dgp.simulate_panel_sar_tobit_fe(
        N=N,
        T=T,
        W=W,
        rho=rho,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_panel_sem_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored panel SEM FE Tobit data using the dgp module."""
    out = dgp.simulate_panel_sem_tobit_fe(
        N=N,
        T=T,
        W=W,
        lam=lam,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


# ---------------------------------------------------------------------------
# Dynamic DE (direct-estimation) panel data generators
# ---------------------------------------------------------------------------


def make_panel_sar_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SAR panel FE data."""
    out = dgp.simulate_panel_sar_dynamic_fe(
        N=N,
        T=T,
        W=W,
        rho=rho,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sem_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SEM panel FE data."""
    out = dgp.simulate_panel_sem_dynamic_fe(
        N=N,
        T=T,
        W=W,
        lam=lam,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdem_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SDEM panel FE data."""
    out = dgp.simulate_panel_sdem_dynamic_fe(
        N=N,
        T=T,
        W=W,
        lam=lam,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_slx_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SLX panel FE data."""
    out = dgp.simulate_panel_slx_dynamic_fe(
        N=N,
        T=T,
        W=W,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


# ---------------------------------------------------------------------------
# Static SDM / SDEM panel data generators (for FE recovery tests)
# ---------------------------------------------------------------------------


def make_panel_sdm_fe_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SDM panel FE data with WX terms."""
    out = dgp.simulate_panel_sdm_fe(
        N=N,
        T=T,
        W=W,
        rho=rho,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdem_fe_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SDEM panel FE data with WX terms."""
    out = dgp.simulate_panel_sdem_fe(
        N=N,
        T=T,
        W=W,
        lam=lam,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


# ---------------------------------------------------------------------------
# Flow (O-D) data generators
# ---------------------------------------------------------------------------


def make_flow_data(
    rng: np.random.Generator,
    G: Graph,
    n: int,
    rho_d: float = 0.25,
    rho_o: float = 0.25,
    rho_w: float = 0.10,
    beta_d: np.ndarray | list | None = None,
    beta_o: np.ndarray | list | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate cross-sectional SAR flow data.

    Returns (y_vec, X, col_names).
    """
    if beta_d is None:
        beta_d = [1.0, -0.5]
    if beta_o is None:
        beta_o = [0.5, 0.3]
    out = dgp.generate_flow_data(
        n=n,
        G=G,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        beta_d=beta_d,
        beta_o=beta_o,
        sigma=sigma,
        seed=int(rng.integers(0, 2**31)),
    )
    return out["y_vec"], out["X"], out["col_names"]


def make_poisson_flow_data(
    rng: np.random.Generator,
    G: Graph,
    n: int,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: float | list[float] | None = None,
    beta_o: float | list[float] | None = None,
    k: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate cross-sectional Poisson flow data.

    Returns (y_vec, X, col_names).
    """
    out = dgp.generate_poisson_flow_data(
        n=n,
        G=G,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        beta_d=beta_d,
        beta_o=beta_o,
        k=k,
        seed=int(rng.integers(0, 2**31)),
    )
    return out["y_vec"], out["X"], out["col_names"]


def make_panel_flow_data(
    rng: np.random.Generator,
    G: Graph,
    n: int,
    T: int,
    rho_d: float = 0.25,
    rho_o: float = 0.25,
    rho_w: float = 0.10,
    beta_d: np.ndarray | list | None = None,
    beta_o: np.ndarray | list | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate panel SAR flow data with O-D-pair random effects.

    Returns (y, X, col_names) in time-first stacking order.
    """
    if beta_d is None:
        beta_d = [1.0, -0.5]
    if beta_o is None:
        beta_o = [0.5, 0.3]
    out = dgp.generate_panel_flow_data(
        n=n,
        T=T,
        G=G,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        beta_d=beta_d,
        beta_o=beta_o,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        seed=int(rng.integers(0, 2**31)),
    )
    return out["y"], out["X"], out["col_names"]


def make_panel_poisson_flow_data(
    rng: np.random.Generator,
    G: Graph,
    n: int,
    T: int,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: float | list[float] | None = None,
    beta_o: float | list[float] | None = None,
    k: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate panel Poisson flow data (pooled, no unit effects).

    Returns (y, X, col_names) in time-first stacking order.
    """
    out = dgp.generate_panel_poisson_flow_data(
        n=n,
        T=T,
        G=G,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        beta_d=beta_d,
        beta_o=beta_o,
        k=k,
        seed=int(rng.integers(0, 2**31)),
    )
    return out["y"], out["X"], out["col_names"]
