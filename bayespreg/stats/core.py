"""Spatial statistics functions mirroring MATLAB spatial/stats directory.

This module implements the computational functionality from:
- lmerror, lmlag, lmrho, lmrhorob, lmsar
- lratios, moran, walds
- model_probs
- prt_spat, prt_spat2, prt_old, plt_spat
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _as_array(x) -> np.ndarray:
    """Convert input to a floating NumPy array.

    Parameters
    ----------
    x : array-like
        Input object.

    Returns
    -------
    np.ndarray
        Floating-point array view/copy of ``x``.
    """
    return np.asarray(x, dtype=float)


def _normalize_w(W: np.ndarray) -> np.ndarray:
    """Row-standardize a spatial weights matrix.

    Parameters
    ----------
    W : np.ndarray
        Spatial weights matrix.

    Returns
    -------
    np.ndarray
        Row-standardized matrix. Zero-sum rows are left unchanged.
    """
    W = _as_array(W)
    rs = W.sum(axis=1)
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return W / rs[:, None]


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Compute basic OLS quantities used by spatial tests.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable vector of shape ``(n,)``.
    X : np.ndarray
        Regressor matrix of shape ``(n, k)``.

    Returns
    -------
    dict
        Dictionary with keys ``beta``, ``resid``, ``sige``, ``n``, ``k``.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sige = float((resid @ resid) / max(n - k, 1))
    return {"beta": beta, "resid": resid, "sige": sige, "n": n, "k": k}


def _logdet(I_minus_rW: np.ndarray) -> float:
    """Compute ``log|I - rW|`` from a matrix argument.

    Parameters
    ----------
    I_minus_rW : np.ndarray
        Matrix whose log-determinant is required.

    Returns
    -------
    float
        Log-determinant, or ``-np.inf`` when determinant sign is nonpositive.
    """
    sign, ld = np.linalg.slogdet(I_minus_rW)
    if sign <= 0:
        return -np.inf
    return float(ld)


def _sem_mle(y: np.ndarray, X: np.ndarray, W: np.ndarray, bounds=(-0.95, 0.95)) -> dict:
    """Estimate a cross-sectional SEM model by concentrated ML.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable of shape ``(n,)``.
    X : np.ndarray
        Regressor matrix of shape ``(n, k)``.
    W : np.ndarray
        Spatial weights matrix of shape ``(n, n)``.
    bounds : tuple[float, float], optional
        Search interval for the SEM spatial parameter.

    Returns
    -------
    dict
        SEM estimate dictionary with keys including ``rho``, ``beta``,
        ``resid``, ``sige``, and ``lik``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(X)
    W = _as_array(W)
    n = y.shape[0]
    eye_n = np.eye(n)

    def nll(lam: float) -> float:
        B = eye_n - lam * W
        ld = _logdet(B)
        if not np.isfinite(ld):
            return np.inf
        yb = B @ y
        Xb = B @ X
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        e = yb - Xb @ beta
        sig2 = float((e @ e) / n)
        if sig2 <= 1e-14:
            return np.inf
        ll = -(n / 2) * (np.log(2 * np.pi) + np.log(sig2) + 1.0) + ld
        return -ll

    opt = optimize.minimize_scalar(nll, bounds=bounds, method="bounded")
    lam = float(opt.x)
    B = eye_n - lam * W
    yb = B @ y
    Xb = B @ X
    beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    e = yb - Xb @ beta
    sig2 = float((e @ e) / n)
    ll = -float(opt.fun)

    return {
        "rho": lam,
        "beta": beta,
        "resid": y - X @ beta,
        "sige": sig2,
        "lik": ll,
        "meth": "sem",
        "nobs": n,
        "nvar": X.shape[1],
    }


def _sar_mle(y: np.ndarray, X: np.ndarray, W: np.ndarray, bounds=(-0.95, 0.95)) -> dict:
    """Estimate a cross-sectional SAR model by concentrated ML.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable of shape ``(n,)``.
    X : np.ndarray
        Regressor matrix of shape ``(n, k)``.
    W : np.ndarray
        Spatial weights matrix of shape ``(n, n)``.
    bounds : tuple[float, float], optional
        Search interval for the SAR spatial parameter.

    Returns
    -------
    dict
        SAR estimate dictionary with keys including ``rho``, ``beta``,
        ``resid``, ``sige``, ``lik``, and approximate ``tstat`` for ``rho``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(X)
    W = _as_array(W)
    n = y.shape[0]
    eye_n = np.eye(n)

    def nll(rho: float) -> float:
        A = eye_n - rho * W
        ld = _logdet(A)
        if not np.isfinite(ld):
            return np.inf
        ys = A @ y
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        e = ys - X @ beta
        sig2 = float((e @ e) / n)
        if sig2 <= 1e-14:
            return np.inf
        ll = -(n / 2) * (np.log(2 * np.pi) + np.log(sig2) + 1.0) + ld
        return -ll

    opt = optimize.minimize_scalar(nll, bounds=bounds, method="bounded")
    rho = float(opt.x)
    A = eye_n - rho * W
    ys = A @ y
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    e = ys - X @ beta
    sig2 = float((e @ e) / n)
    ll = -float(opt.fun)

    # Approximate rho variance from Hessian of nll.
    h = 1e-4
    f0 = nll(rho)
    fp = nll(rho + h)
    fm = nll(rho - h)
    d2 = (fp - 2 * f0 + fm) / (h * h)
    var_rho = float(1.0 / max(d2, 1e-8))
    se_rho = np.sqrt(var_rho)
    t_rho = rho / se_rho if se_rho > 0 else np.nan

    return {
        "rho": rho,
        "beta": beta,
        "resid": y - rho * (W @ y) - X @ beta,
        "sige": sig2,
        "lik": ll,
        "meth": "sar",
        "nobs": n,
        "nvar": X.shape[1],
        "tstat": np.concatenate([np.full(X.shape[1], np.nan), [t_rho]])[:, None],
    }


def lmerror(y, x, W) -> dict:
    """Compute the LM-error test for residual spatial autocorrelation.

    Parameters
    ----------
    y : array-like
        Dependent variable of shape ``(n,)``.
    x : array-like
        Regressor matrix of shape ``(n, k)``.
    W : array-like
        Spatial weights matrix of shape ``(n, n)``.

    Returns
    -------
    dict
        Result dictionary with ``lm``, ``prob``, ``chi1``, ``nobs``, ``nvar``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    n, k = X.shape

    ols = _ols(y, X)
    e = ols["resid"]
    epe = float((e @ e) / n)
    t1 = float(np.trace((W + W.T) @ W))
    lm1 = float((e @ W @ e) / max(epe, 1e-12))
    lm = (lm1 * lm1) / max(t1, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 1))

    return {
        "meth": "lmerror",
        "lm": lm,
        "prob": prob,
        "chi1": 6.635,
        "nobs": n,
        "nvar": k,
    }


def lmlag(y, x, W) -> dict:
    """Compute the LM-lag test for a missing spatially lagged dependent term.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W : array-like
        Spatial weights matrix.

    Returns
    -------
    dict
        Result dictionary with ``lm`` statistic and p-value ``prob``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    n, k = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    sigma = float((e @ e) / max(n - k, 1))
    epe = float((e @ e) / n)
    lm1 = float((e @ W @ y) / max(epe, 1e-12))

    t1 = float(np.trace((W + W.T) @ W))
    d1 = W @ (X @ beta)
    M = np.eye(n) - X @ np.linalg.pinv(X.T @ X) @ X.T
    D = float((d1.T @ M @ d1) / max(sigma, 1e-12) + t1)
    lm = (lm1 * lm1) / max(D, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 1))

    return {
        "meth": "lmlag",
        "lm": lm,
        "prob": prob,
        "chi1": 6.635,
        "nobs": n,
        "nvar": k,
    }


def lmrho(y, x, W) -> dict:
    """Compute LM statistic for SAR parameter ``rho``.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W : array-like
        Spatial weights matrix.

    Returns
    -------
    dict
        Result dictionary containing ``lmrho``, residual quantities, and p-value.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    N, k = X.shape

    bhat, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ bhat
    sige = float((e @ e) / max(N - k, 1))

    M1 = np.eye(N) - X @ np.linalg.pinv(X.T @ X) @ X.T
    M2 = W @ (X @ bhat)
    M3 = float(M2.T @ M1 @ M2)
    T = float(np.trace(W.T @ W + W @ W))
    J = (M3 + T * sige) / max(N * sige, 1e-12)
    M6 = float((e.T @ W @ y) / max(sige, 1e-12))
    lm = (M6 * M6) / max(N * J, 1e-12)

    return {
        "meth": "lmrho",
        "nobs": N,
        "beta": bhat,
        "resid": e,
        "sige": sige,
        "lmrho": lm,
        "prob": float(1.0 - stats.chi2.cdf(lm, 1)),
    }


def lmrhorob(y, x, W) -> dict:
    """Compute robust LM statistic for SAR parameter ``rho``.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W : array-like
        Spatial weights matrix.

    Returns
    -------
    dict
        Result dictionary containing ``lmrhorob`` and related diagnostics.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    N, k = X.shape

    bhat, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ bhat
    sige = float((e @ e) / max(N - k, 1))

    M1 = np.eye(N) - X @ np.linalg.pinv(X.T @ X) @ X.T
    M2 = W @ (X @ bhat)
    M3 = float(M2.T @ M1 @ M2)
    T = float(np.trace(W.T @ W + W @ W))
    J = (M3 + T * sige) / max(N * sige, 1e-12)
    M6 = float((e.T @ W @ y - e.T @ W @ e / max(sige, 1e-12)))
    lm = (M6 * M6) / max(N * J, 1e-12) - T

    return {
        "meth": "lmrhorob",
        "nobs": N,
        "beta": bhat,
        "resid": e,
        "sige": sige,
        "lmrhorob": lm,
        "prob": float(1.0 - stats.chi2.cdf(lm, 1)),
    }


def moran(y, x, W) -> dict:
    """Compute Moran's I for regression residuals.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W : array-like
        Spatial weights matrix (standardized internally).

    Returns
    -------
    dict
        Dictionary with Moran's ``morani``, standardized ``istat``, moments,
        and marginal probability ``prob``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _normalize_w(_as_array(W))
    n, k = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    epe = float(e.T @ e)
    mi = float((e.T @ W @ e) / max(epe, 1e-12))

    M = np.eye(n) - X @ np.linalg.pinv(X.T @ X) @ X.T
    tmw = float(np.trace(M @ W))
    meani = tmw / max(n - k, 1)
    vari = float(np.trace((M @ W) @ (M @ W).T) + np.trace((M @ W) @ (M @ W)) + tmw * tmw)
    vari = vari / max((n - k) * (n - k + 2), 1)
    vari = vari - meani * meani
    vari = max(vari, 1e-12)
    istat = (mi - meani) / np.sqrt(vari)
    prob = float(2 * (1 - stats.norm.cdf(abs(istat))))

    return {
        "meth": "moran",
        "nobs": n,
        "nvar": k,
        "morani": mi,
        "istat": istat,
        "imean": meani,
        "ivar": vari,
        "prob": prob,
    }


def walds(y, x, W) -> dict:
    """Compute Wald test statistic for spatial error autocorrelation.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W : array-like
        Spatial weights matrix.

    Returns
    -------
    dict
        Result dictionary with ``wald`` and chi-squared p-value ``prob``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    n, k = X.shape

    sem_res = _sem_mle(y, X, W)
    lam = float(sem_res["rho"])

    z = np.eye(n) - lam * W
    t1 = float(np.trace(W * z))
    t2 = float(np.trace(W @ z) ** 2)
    t3 = float(np.trace((W @ z).T @ (W @ z)))
    wald = (lam * lam) * (t2 + t3 - (1.0 / n) * (t1 * t1))
    prob = float(1.0 - stats.chi2.cdf(wald, 1))

    return {
        "meth": "walds",
        "wald": wald,
        "prob": prob,
        "chi1": 6.635,
        "nobs": n,
        "nvar": k,
    }


def lratios(y, x, W, sem_result: dict | None = None) -> dict:
    """Compute LR test for SEM versus OLS.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W : array-like
        Spatial weights matrix.
    sem_result : dict, optional
        Precomputed SEM result dictionary. If omitted, SEM is estimated
        internally.

    Returns
    -------
    dict
        Result dictionary with ``lratio`` and p-value ``prob``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    n, k = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e0 = y - X @ beta
    sig0 = float((e0 @ e0) / n)

    if sem_result is None:
        sem_result = _sem_mle(y, X, W)

    sig1 = float(sem_result["sige"])
    lam = float(sem_result["rho"])
    detval = _logdet(np.eye(n) - lam * W)

    lratio = n * (np.log(max(sig0, 1e-12)) - np.log(max(sig1, 1e-12))) + 2 * detval
    prob = float(1.0 - stats.chi2.cdf(lratio, 1))

    return {
        "meth": "lratios",
        "nobs": n,
        "nvar": k,
        "lratio": lratio,
        "chi1": 6.635,
        "prob": prob,
    }


def lmsar(y, x, W1, W2, lmin: float = -0.95, lmax: float = 0.95) -> dict:
    """Compute LM-error test conditional on a SAR specification.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W1 : array-like
        Spatial weights matrix used for SAR lag term.
    W2 : array-like
        Spatial weights matrix used for spatial error term.
    lmin, lmax : float, optional
        SAR parameter search bounds.

    Returns
    -------
    dict
        Result dictionary with ``lm`` statistic and p-value ``prob``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W1 = _as_array(W1)
    W2 = _as_array(W2)
    n, k = X.shape

    res = _sar_mle(y, X, W1, bounds=(lmin, lmax))
    e = res["resid"]
    rho = float(res["rho"])
    sige = float(res["sige"])

    # Approximate var(rho) from t-stat if available.
    t_rho = float(res["tstat"][k, 0]) if "tstat" in res else np.nan
    if np.isfinite(t_rho) and abs(t_rho) > 1e-8:
        stdt = t_rho / max(abs(rho), 1e-8)
        varr = float(1.0 / max(stdt * stdt, 1e-8))
    else:
        varr = 1.0

    A = np.eye(n) - rho * W1
    AI = np.linalg.inv(A)
    T22 = float(np.trace(W2 @ W2 + W2.T @ W2))
    T21 = float(np.trace(W2 @ W1 @ AI + W2.T @ W1 @ AI))
    lm1 = float((e.T @ W2 @ e) / max(sige, 1e-12))
    Tterm = T22 - T21 * T21 * varr
    lm = (lm1 * lm1) / max(Tterm, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 1))

    return {
        "meth": "lmsar",
        "lm": lm,
        "prob": prob,
        "chi1": 6.635,
        "nobs": n,
        "nvar": k,
    }


def model_probs(*results: dict) -> np.ndarray:
    """Compute posterior model probabilities from log-marginal vectors.

    Parameters
    ----------
    *results : dict
        Variable-length sequence of result dictionaries containing ``mlike``
        vectors. If available, ``lndet`` in any dictionary supplies the
        integration grid.

    Returns
    -------
    np.ndarray
        Column vector of posterior model probabilities summing to one.
    """
    if len(results) == 0:
        raise ValueError("Provide at least one results dictionary.")

    lmarginal = []
    detval = None
    for r in results:
        if "mlike" not in r:
            raise ValueError("Each result must contain 'mlike'.")
        if "lndet" in r:
            detval = np.asarray(r["lndet"])
        lmarginal.append(np.asarray(r["mlike"]).reshape(-1))

    L = np.column_stack(lmarginal)
    nrho = L.shape[0]
    if detval is None:
        xgrid = np.arange(nrho)
    else:
        xgrid = np.asarray(detval)[:, 0]

    adj = np.max(L)
    xx = np.exp(L - adj)

    isum = np.trapezoid(xx, x=xgrid, axis=0)
    probs = isum / np.sum(isum)
    return probs.reshape(-1, 1)


def prt_spat(result: dict) -> str:
    """Format a spatial-test result dictionary as printable text.

    Parameters
    ----------
    result : dict
        Result dictionary from a spatial test function.

    Returns
    -------
    str
        Human-readable multiline summary.
    """
    lines = [f"Method: {result.get('meth', 'unknown')}"]
    for key in ["lm", "lratio", "wald", "prob", "chi1", "nobs", "nvar"]:
        if key in result:
            lines.append(f"{key}: {result[key]}")
    return "\n".join(lines)


def prt_spat2(result: dict) -> str:
    """Alias of :func:`prt_spat` for MATLAB compatibility.

    Parameters
    ----------
    result : dict
        Result dictionary from a spatial test function.

    Returns
    -------
    str
        Human-readable multiline summary.
    """
    return prt_spat(result)


def prt_old(result: dict) -> str:
    """Legacy alias of :func:`prt_spat`.

    Parameters
    ----------
    result : dict
        Result dictionary from a spatial test function.

    Returns
    -------
    str
        Human-readable multiline summary.
    """
    return prt_spat(result)


def plt_spat(y, x, W):
    """Create a Moran-style residual scatter plot.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Regressor matrix.
    W : array-like
        Spatial weights matrix.

    Returns
    -------
    tuple
        Matplotlib ``(fig, ax)`` tuple.
    """
    import matplotlib.pyplot as plt

    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    Wn = _normalize_w(_as_array(W))
    e = _ols(y, X)["resid"]
    We = Wn @ e

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(e, We, alpha=0.7)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.axvline(0.0, color="k", linewidth=1)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Spatial lag of residuals")
    ax.set_title("Moran Scatter Plot")
    return fig, ax
