"""Regression diagnostics inspired by MATLAB toolbox_may2021/diagn.

Implemented counterparts:
- rdiagnose  -> rdiagnose_like
- bpagan     -> bpagan_test
- arch       -> arch_test
- qstat2     -> ljung_box_q
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import chi2


def _as_lag_array(lags: int | Iterable[int]) -> np.ndarray:
    """Normalize lag specification into validated 1D integer array.

    Parameters
    ----------
    lags : int or Iterable[int]
        Lag order specification.

    Returns
    -------
    np.ndarray
        Positive lag orders.
    """
    if np.isscalar(lags):
        lags = [int(lags)]
    arr = np.asarray(list(lags), dtype=int)
    if arr.ndim != 1 or np.any(arr <= 0):
        raise ValueError("lags must be a positive int or iterable of positive ints")
    return arr


def _safe_inv(a: np.ndarray) -> np.ndarray:
    """Return pseudo-inverse used by diagnostics regressions.

    Parameters
    ----------
    a : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Moore-Penrose pseudo-inverse.
    """
    return np.linalg.pinv(a)


def rdiagnose_like(y: np.ndarray, X: np.ndarray, resid: np.ndarray) -> dict[str, np.ndarray | float | str]:
    """OLS-style influence diagnostics (MATLAB rdiagnose analogue).

    Parameters
    ----------
    y : np.ndarray
        Dependent variable, shape (n,).
    X : np.ndarray
        Design matrix, shape (n, k).
    resid : np.ndarray
        Residual vector, shape (n,).

    Returns
    -------
    dict
        Influence and outlier diagnostics, including leverage, studentized
        residuals, DFBETA, DFFIT, and Cook's distance.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float).reshape(-1)

    n, k = X.shape
    p = np.linalg.matrix_rank(X)

    xtxi = _safe_inv(X.T @ X)
    h = np.diag(X @ xtxi @ X.T)

    dof = max(n - p, 1)
    sige = float((resid @ resid) / dof)
    serr = np.sqrt(max(sige, 1e-12))

    iota = np.ones(n)
    one_minus_h = np.clip(iota - h, 1e-10, None)

    stdr = resid / serr
    # Deleted-variance style term used in rdiagnose.m
    tmp1 = (n - p) * serr**2 / max(n - p - 1, 1)
    tmp2 = (resid**2) / max(n - p - 1, 1)
    sis = (tmp1 - tmp2) / one_minus_h
    sis = np.clip(sis, 1e-12, None)

    rstud = resid / np.sqrt(sis * one_minus_h)
    press = resid / one_minus_h
    pstat = float(press @ press)

    stud = resid / np.sqrt((serr**2) * one_minus_h)

    c = xtxi @ X.T
    diag_xtxi = np.diag(xtxi)
    denom = np.sqrt(np.outer(one_minus_h, diag_xtxi))
    denom = np.clip(denom, 1e-12, None)
    dfbeta = (c.T * rstud[:, None]) / denom

    dffit = np.sqrt(h / one_minus_h) * rstud
    cookd = ((stud**2) / max(p, 1)) * h / one_minus_h

    return {
        "meth": "rdiagnose_like",
        "hatdi": h,
        "stdr": stdr,
        "press": press,
        "pstat": pstat,
        "stud": stud,
        "rstud": rstud,
        "dffit": dffit,
        "cookd": cookd,
        "resid": resid,
        "dfbeta": dfbeta,
    }


def bpagan_test(resid: np.ndarray, X: np.ndarray) -> dict[str, float | int | str]:
    """Breusch-Pagan LM test.

    Parameters
    ----------
    resid : np.ndarray
        Residual vector.
    X : np.ndarray
        Design matrix.

    Returns
    -------
    dict
        Test statistic, degrees of freedom, and p-value.
    """
    resid = np.asarray(resid, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)

    n, k = X.shape
    u2 = resid * resid
    sig = (u2.sum() / n) * np.ones(n)
    f = u2 / np.clip(sig, 1e-12, None) - 1.0
    z = X * X

    ztz_inv = _safe_inv(z.T @ z)
    lm = float(0.5 * (f.T @ z @ ztz_inv @ z.T @ f))
    dof = max(k - 1, 1)
    prob = float(1.0 - chi2.cdf(lm, dof))

    return {
        "meth": "bpagan",
        "lm": lm,
        "dof": dof,
        "prob": prob,
    }


def arch_test(resid: np.ndarray, lags: int | Iterable[int] = 1) -> dict[str, np.ndarray | str]:
    """Engle ARCH test for one or multiple lag orders.

    Parameters
    ----------
    resid : np.ndarray
        Residual vector.
    lags : int or Iterable[int], default=1
        Lag order(s) for ARCH regression.

    Returns
    -------
    dict
        ARCH statistics and p-values for each lag order.
    """
    resid = np.asarray(resid, dtype=float).reshape(-1)
    p_arr = _as_lag_array(lags)
    n = resid.shape[0]
    x2 = resid * resid

    stats = []
    pvals = []
    for p in p_arr:
        y = x2[p:]
        if y.size <= 1:
            stats.append(np.nan)
            pvals.append(np.nan)
            continue

        lagged = np.column_stack([x2[p - j: n - j] for j in range(1, p + 1)])
        X = np.column_stack([np.ones(y.size), lagged])

        beta = _safe_inv(X.T @ X) @ X.T @ y
        fit = X @ beta
        sst = float(((y - y.mean()) ** 2).sum())
        ssr = float(((y - fit) ** 2).sum())
        rsq = 0.0 if sst <= 1e-12 else (1.0 - ssr / sst)

        stat = float((n - p) * rsq)
        pval = float(1.0 - chi2.cdf(stat, p))
        stats.append(stat)
        pvals.append(pval)

    return {
        "meth": "arch",
        "lags": p_arr,
        "archstat": np.asarray(stats),
        "pval": np.asarray(pvals),
    }


def ljung_box_q(resid: np.ndarray, lags: int | Iterable[int] = 1) -> dict[str, np.ndarray | str]:
    """Ljung-Box Q test for residual autocorrelation.

    Parameters
    ----------
    resid : np.ndarray
        Residual vector.
    lags : int or Iterable[int], default=1
        Lag order(s) to test.

    Returns
    -------
    dict
        Q statistics and p-values for each lag order.
    """
    resid = np.asarray(resid, dtype=float).reshape(-1)
    p_arr = _as_lag_array(lags)
    n = resid.shape[0]

    x = resid - resid.mean()
    denom = float(np.dot(x, x))
    denom = max(denom, 1e-12)

    def acf_at_lag(k: int) -> float:
        return float(np.dot(x[k:], x[: n - k]) / denom)

    stats = []
    pvals = []
    for p in p_arr:
        rho = np.array([acf_at_lag(k) for k in range(1, p + 1)])
        scale = np.array([n - k for k in range(1, p + 1)], dtype=float)
        q = float(n * (n + 2.0) * np.sum((rho * rho) / np.clip(scale, 1.0, None)))
        pval = float(1.0 - chi2.cdf(q, p))
        stats.append(q)
        pvals.append(pval)

    return {
        "meth": "ljung_box_q",
        "lags": p_arr,
        "qstat": np.asarray(stats),
        "pval": np.asarray(pvals),
    }


def outlier_candidates(
    diagnostics: dict[str, np.ndarray | float | str],
    n: int,
    k: int,
) -> dict[str, np.ndarray]:
    """Return candidate outlier/influence indices using common thresholds.

    Parameters
    ----------
    diagnostics : dict
        Output from :func:`rdiagnose_like`.
    n : int
        Number of observations.
    k : int
        Number of regressors.

    Returns
    -------
    dict
        Index arrays for leverage, r-student, DFFIT, and DFBETA flags.
    """
    hatdi = np.asarray(diagnostics["hatdi"])
    rstud = np.asarray(diagnostics["rstud"])
    dffit = np.asarray(diagnostics["dffit"])
    dfbeta = np.asarray(diagnostics["dfbeta"])

    return {
        "leverage_idx": np.where(hatdi > 2.0 * k / n)[0],
        "rstudent_idx": np.where(np.abs(rstud) > 2.0)[0],
        "dffit_idx": np.where(np.abs(dffit) > 2.0 * np.sqrt(k / n))[0],
        "dfbeta_idx": np.argwhere(np.abs(dfbeta) > 2.0 / np.sqrt(n)),
    }


def panel_residual_structure(resid: np.ndarray, N: int, T: int) -> dict[str, np.ndarray | float | str]:
    """Summarize residual structure in stacked panel data.

    Residuals must be sorted by time then unit, matching panel_g conventions.

    Parameters
    ----------
    resid : np.ndarray
        Stacked residual vector of length ``N*T``.
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    dict
        Variance decomposition and mean profiles by unit/time.
    """
    r = np.asarray(resid, dtype=float).reshape(-1)
    if r.size != N * T:
        raise ValueError("resid length must be N*T")

    r2 = r.reshape(T, N)
    unit_means = r2.mean(axis=0)
    time_means = r2.mean(axis=1)

    grand = float(r2.mean())
    total_var = float(r2.var(ddof=1)) if r2.size > 1 else 0.0
    between_unit_var = float(unit_means.var(ddof=1)) if N > 1 else 0.0
    between_time_var = float(time_means.var(ddof=1)) if T > 1 else 0.0
    within_var = float((r2 - unit_means[None, :] - time_means[:, None] + grand).var(ddof=1)) if r2.size > 1 else 0.0

    return {
        "meth": "panel_residual_structure",
        "grand_mean": grand,
        "total_var": total_var,
        "between_unit_var": between_unit_var,
        "between_time_var": between_time_var,
        "within_var": within_var,
        "unit_means": unit_means,
        "time_means": time_means,
    }


def pesaran_cd_test(resid: np.ndarray, N: int, T: int) -> dict[str, float | str]:
    """Pesaran CD test for cross-sectional dependence in panel residuals.

    Parameters
    ----------
    resid : np.ndarray
        Stacked residual vector of length ``N*T``.
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    dict
        CD test statistic and asymptotic normal p-value.
    """
    r = np.asarray(resid, dtype=float).reshape(-1)
    if r.size != N * T:
        raise ValueError("resid length must be N*T")
    if N < 2 or T < 2:
        return {"meth": "pesaran_cd", "cd": np.nan, "pval": np.nan}

    r2 = r.reshape(T, N)
    corr = np.corrcoef(r2, rowvar=False)
    iu = np.triu_indices(N, k=1)
    rho_ij = corr[iu]
    rho_ij = rho_ij[np.isfinite(rho_ij)]
    if rho_ij.size == 0:
        return {"meth": "pesaran_cd", "cd": np.nan, "pval": np.nan}

    cd = float(np.sqrt(2.0 * T / (N * (N - 1))) * rho_ij.sum())
    # Asymptotically N(0,1)
    from scipy.stats import norm

    pval = float(2.0 * (1.0 - norm.cdf(abs(cd))))
    return {"meth": "pesaran_cd", "cd": cd, "pval": pval}
