"""Panel spatial tests/functions mirroring FESPD_Est_Tests MATLAB directory."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _as_array(x):
    """Convert input to a floating NumPy array.

    Parameters
    ----------
    x : array-like
        Input object.

    Returns
    -------
    np.ndarray
        Floating-point array.
    """
    return np.asarray(x, dtype=float)


def trans_tslow(x, n):
    """Transform panel ordering from i-slow/t-fast to i-fast/t-slow.

    Parameters
    ----------
    x : array-like
        Stacked panel array.
    n : int
        Number of cross-sectional units.

    Returns
    -------
    np.ndarray
        Reordered panel array.
    """
    x = _as_array(x)
    nt = x.shape[0]
    t = nt // n
    res = []
    for j in range(t):
        for i in range(n):
            j2 = t * i + j
            res.append(x[j2, ...])
    return np.asarray(res)


def _lee_yu_transform(y, x, N):
    """Apply Lee-Yu within transformation for balanced panel FE models.

    Parameters
    ----------
    y : array-like
        Stacked dependent variable (time-slow ordering).
    x : array-like
        Stacked regressors.
    N : int
        Number of cross-sectional units.

    Returns
    -------
    tuple
        ``(y_transformed, X_transformed, T_minus_1)``.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    nt = X.shape[0]
    T = nt // N

    Q = np.eye(T) - np.ones((T, T)) / T
    evals, evecs = np.linalg.eigh(Q)
    F = evecs[:, evals > 1e-8]  # T x (T-1)
    TR = np.kron(F.T, np.eye(N))

    ytr = TR @ y
    Xtr = TR @ X
    return ytr, Xtr, T - 1


def _ols(y, X):
    """Compute OLS estimates and residual variance.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable.
    X : np.ndarray
        Regressor matrix.

    Returns
    -------
    dict
        Dictionary with ``beta``, ``resid``, ``sige``, ``n``, and ``k``.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sige = float((resid @ resid) / max(n - k, 1))
    return {"beta": beta, "resid": resid, "sige": sige, "n": n, "k": k}


def _logdet(I_minus_rW):
    """Compute matrix log-determinant with sign check.

    Parameters
    ----------
    I_minus_rW : np.ndarray
        Matrix argument.

    Returns
    -------
    float
        Log-determinant or ``-np.inf`` if determinant sign is nonpositive.
    """
    sign, ld = np.linalg.slogdet(I_minus_rW)
    return float(ld) if sign > 0 else -np.inf


def _panel_W(W, Tm1):
    """Build block-diagonal panel weights matrix ``I_(T-1) ⊗ W``.

    Parameters
    ----------
    W : array-like
        Cross-sectional weights matrix.
    Tm1 : int
        Number of transformed time periods.

    Returns
    -------
    np.ndarray
        Block-diagonal panel weights matrix.
    """
    W = _as_array(W)
    return np.kron(np.eye(Tm1), W)


def _panel_sem_mle(ytr, Xtr, W, Tm1, bounds=(-0.95, 0.95)):
    """Estimate transformed SEM panel model by concentrated ML.

    Parameters
    ----------
    ytr : np.ndarray
        Transformed dependent variable.
    Xtr : np.ndarray
        Transformed regressors.
    W : np.ndarray
        Cross-sectional weights matrix.
    Tm1 : int
        Number of transformed time periods.
    bounds : tuple[float, float], optional
        Search interval for spatial error parameter.

    Returns
    -------
    dict
        SEM parameter dictionary.
    """
    n = ytr.size
    N = W.shape[0]
    Wk = _panel_W(W, Tm1)
    eye_n = np.eye(n)

    def nll(lam):
        B = eye_n - lam * Wk
        ld = _logdet(np.eye(N) - lam * W)
        if not np.isfinite(ld):
            return np.inf
        yb = B @ ytr
        Xb = B @ Xtr
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        e = yb - Xb @ beta
        sig2 = float((e @ e) / n)
        if sig2 <= 1e-14:
            return np.inf
        ll = -(n / 2) * (np.log(2 * np.pi) + np.log(sig2) + 1.0) + Tm1 * ld
        return -ll

    opt = optimize.minimize_scalar(nll, bounds=bounds, method="bounded")
    lam = float(opt.x)
    B = np.eye(n) - lam * Wk
    yb = B @ ytr
    Xb = B @ Xtr
    beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    e = yb - Xb @ beta
    sig2 = float((e @ e) / n)
    return {
        "rho": lam,
        "beta": beta,
        "resid": e,
        "sige": sig2,
        "lik": -float(opt.fun),
        "cov2": np.eye(len(beta) + 2),
    }


def _panel_sar_mle(ytr, Xtr, W, Tm1, bounds=(-0.95, 0.95)):
    """Estimate transformed SAR panel model by concentrated ML.

    Parameters
    ----------
    ytr : np.ndarray
        Transformed dependent variable.
    Xtr : np.ndarray
        Transformed regressors.
    W : np.ndarray
        Cross-sectional weights matrix.
    Tm1 : int
        Number of transformed time periods.
    bounds : tuple[float, float], optional
        Search interval for spatial lag parameter.

    Returns
    -------
    dict
        SAR parameter dictionary.
    """
    n = ytr.size
    N = W.shape[0]
    Wk = _panel_W(W, Tm1)
    eye_n = np.eye(n)

    def nll(rho):
        A = eye_n - rho * Wk
        ld = _logdet(np.eye(N) - rho * W)
        if not np.isfinite(ld):
            return np.inf
        ys = A @ ytr
        beta, *_ = np.linalg.lstsq(Xtr, ys, rcond=None)
        e = ys - Xtr @ beta
        sig2 = float((e @ e) / n)
        if sig2 <= 1e-14:
            return np.inf
        ll = -(n / 2) * (np.log(2 * np.pi) + np.log(sig2) + 1.0) + Tm1 * ld
        return -ll

    opt = optimize.minimize_scalar(nll, bounds=bounds, method="bounded")
    rho = float(opt.x)
    A = np.eye(n) - rho * Wk
    ys = A @ ytr
    beta, *_ = np.linalg.lstsq(Xtr, ys, rcond=None)
    e = ys - Xtr @ beta
    sig2 = float((e @ e) / n)

    h = 1e-4
    f0 = nll(rho)
    fp = nll(rho + h)
    fm = nll(rho - h)
    d2 = (fp - 2 * f0 + fm) / (h * h)
    var_rho = float(1.0 / max(d2, 1e-8))
    se = np.sqrt(var_rho)
    t_rho = rho / se if se > 0 else np.nan

    k = Xtr.shape[1]
    tstat = np.concatenate([np.full(k, np.nan), [t_rho]])[:, None]

    return {
        "rho": rho,
        "beta": beta,
        "resid": e,
        "sige": sig2,
        "lik": -float(opt.fun),
        "tstat": tstat,
    }


def _panel_sarar_mle(ytr, Xtr, W1, W2, Tm1, bounds=(-0.9, 0.9)):
    """Estimate transformed SARAR panel model by concentrated ML.

    Parameters
    ----------
    ytr : np.ndarray
        Transformed dependent variable.
    Xtr : np.ndarray
        Transformed regressors.
    W1 : np.ndarray
        SAR weights matrix.
    W2 : np.ndarray
        SEM weights matrix.
    Tm1 : int
        Number of transformed time periods.
    bounds : tuple[float, float], optional
        Box constraints for ``(rho, lambda)``.

    Returns
    -------
    dict
        SARAR parameter dictionary.
    """
    n = ytr.size
    N = W1.shape[0]
    Wk1 = _panel_W(W1, Tm1)
    Wk2 = _panel_W(W2, Tm1)
    eye_n = np.eye(n)

    def nll(par):
        rho, lam = par
        A = eye_n - rho * Wk1
        B = eye_n - lam * Wk2
        ld1 = _logdet(np.eye(N) - rho * W1)
        ld2 = _logdet(np.eye(N) - lam * W2)
        if not np.isfinite(ld1) or not np.isfinite(ld2):
            return np.inf
        ys = A @ ytr
        Xs = B @ Xtr
        ys2 = B @ ys
        beta, *_ = np.linalg.lstsq(Xs, ys2, rcond=None)
        e = ys2 - Xs @ beta
        sig2 = float((e @ e) / n)
        if sig2 <= 1e-14:
            return np.inf
        ll = -(n / 2) * (np.log(2 * np.pi) + np.log(sig2) + 1.0) + Tm1 * (ld1 + ld2)
        return -ll

    opt = optimize.minimize(
        nll,
        x0=np.array([0.1, 0.1]),
        bounds=[bounds, bounds],
        method="L-BFGS-B",
    )
    rho, lam = opt.x
    return {"rho": float(rho), "lam": float(lam), "lik": -float(opt.fun)}


def f_sarpanel(rho, detval, epe0, eped, epe0d, N, T):
    """Evaluate concentrated SAR panel objective at ``rho``.

    Parameters
    ----------
    rho : float
        Spatial lag parameter.
    detval : array-like
        Two-column grid ``[rho, logdet]``.
    epe0, eped, epe0d : float
        Concentrated likelihood scalar terms.
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    float
        Concentrated negative log-likelihood objective value.
    """
    detval = _as_array(detval)
    rho_grid = detval[:, 0]
    ld_grid = detval[:, 1]
    detm = float(np.interp(rho, rho_grid, ld_grid))
    z = epe0 - 2 * rho * epe0d + rho * rho * eped
    return (N * T / 2) * np.log(max(z, 1e-12)) - T * detm


def f2_sarpanel(parm, y, x, W, detval, T):
    """Evaluate full SAR panel log-likelihood at parameter vector.

    Parameters
    ----------
    parm : array-like
        Parameter vector ``[beta..., rho, sigma2]``.
    y, x : array-like
        Stacked panel dependent variable and regressors.
    W : array-like
        Cross-sectional weights matrix.
    detval : array-like
        Two-column grid ``[rho, logdet]``.
    T : int
        Number of periods.

    Returns
    -------
    float
        Log-likelihood value.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    parm = _as_array(parm).reshape(-1)
    N = W.shape[0]

    b = parm[:-2]
    rho = float(parm[-2])
    sige = float(parm[-1])

    detm = float(np.interp(rho, _as_array(detval)[:, 0], _as_array(detval)[:, 1]))
    e = y - X @ b
    for t in range(T):
        t1, t2 = t * N, (t + 1) * N
        e[t1:t2] = e[t1:t2] - rho * (W @ y[t1:t2])
    epe = float(e @ e)
    return -(N * T / 2) * np.log(2 * np.pi * max(sige, 1e-12)) + T * detm - epe / (2 * max(sige, 1e-12))


def f_sempanel(rho, eD, W, detval, T):
    """Evaluate concentrated SEM panel objective at ``rho``.

    Parameters
    ----------
    rho : float
        Spatial error parameter.
    eD : array-like
        Residual vector from transformed model.
    W : array-like
        Cross-sectional weights matrix.
    detval : array-like
        Two-column grid ``[rho, logdet]``.
    T : int
        Number of periods.

    Returns
    -------
    float
        Concentrated negative log-likelihood objective value.
    """
    eD = _as_array(eD).reshape(-1)
    W = _as_array(W)
    N = W.shape[0]
    detm = float(np.interp(rho, _as_array(detval)[:, 0], _as_array(detval)[:, 1]))
    B = np.eye(N) - rho * W
    Be = np.zeros_like(eD)
    for t in range(T):
        t1, t2 = t * N, (t + 1) * N
        Be[t1:t2] = B @ eD[t1:t2]
    epe = float(Be @ Be)
    return (N * T / 2) * np.log(max(epe, 1e-12)) - T * detm


def f2_sempanel(parm, y, x, W, detval, T):
    """Evaluate full SEM panel log-likelihood at parameter vector.

    Parameters
    ----------
    parm : array-like
        Parameter vector ``[beta..., rho, sigma2]``.
    y, x : array-like
        Stacked panel dependent variable and regressors.
    W : array-like
        Cross-sectional weights matrix.
    detval : array-like
        Two-column grid ``[rho, logdet]``.
    T : int
        Number of periods.

    Returns
    -------
    float
        Log-likelihood value.
    """
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W = _as_array(W)
    parm = _as_array(parm).reshape(-1)
    N = W.shape[0]

    b = parm[:-2]
    rho = float(parm[-2])
    sige = float(parm[-1])

    detm = float(np.interp(rho, _as_array(detval)[:, 0], _as_array(detval)[:, 1]))
    B = np.eye(N) - rho * W
    Be = np.zeros_like(y)
    for t in range(T):
        t1, t2 = t * N, (t + 1) * N
        Be[t1:t2] = B @ (y[t1:t2] - X[t1:t2, :] @ b)
    epe = float(Be @ Be)
    return -(N * T / 2) * np.log(2 * np.pi * max(sige, 1e-12)) + T * detm - epe / (2 * max(sige, 1e-12))


def f_sarar_panel(parm, y, x, W1, W2, det1, det2, T):
    """Evaluate concentrated SARAR panel objective at ``(rho, lambda)``.

    Parameters
    ----------
    parm : array-like
        Two-parameter vector ``[rho, lambda]``.
    y, x : array-like
        Stacked panel dependent variable and regressors.
    W1, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    det1, det2 : array-like
        Two-column grids ``[rho, logdet]`` and ``[lambda, logdet]``.
    T : int
        Number of periods.

    Returns
    -------
    float
        Concentrated negative log-likelihood objective value.
    """
    parm = _as_array(parm).reshape(-1)
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W1 = _as_array(W1)
    W2 = _as_array(W2)

    rho, lam = float(parm[0]), float(parm[1])
    NT = y.size
    N = NT // T
    d1 = float(np.interp(rho, _as_array(det1)[:, 0], _as_array(det1)[:, 1]))
    d2 = float(np.interp(lam, _as_array(det2)[:, 0], _as_array(det2)[:, 1]))

    In = np.eye(N)
    It = np.eye(T)
    Ay = np.kron(It, In - rho * W1) @ y
    B = np.kron(It, In - lam * W2)
    b, *_ = np.linalg.lstsq(B @ X, B @ Ay, rcond=None)
    e = B @ (Ay - X @ b)
    epe = float(e @ e)
    return (NT / 2) * np.log(max(epe / NT, 1e-12)) - T * d1 - T * d2


def f2_sarar_panel(parm, y, x, W1, W2, det1, det2, T):
    """Evaluate full SARAR panel log-likelihood at parameter vector.

    Parameters
    ----------
    parm : array-like
        Parameter vector ``[beta..., rho, lambda, sigma2]``.
    y, x : array-like
        Stacked panel dependent variable and regressors.
    W1, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    det1, det2 : array-like
        Two-column grids for SAR and SEM log-determinants.
    T : int
        Number of periods.

    Returns
    -------
    float
        Log-likelihood value.
    """
    parm = _as_array(parm).reshape(-1)
    y = _as_array(y).reshape(-1)
    X = _as_array(x)
    W1 = _as_array(W1)
    W2 = _as_array(W2)

    NT = y.size
    N = NT // T
    b = parm[:-3]
    rho = float(parm[-3])
    lam = float(parm[-2])
    sige = float(parm[-1])

    d1 = float(np.interp(rho, _as_array(det1)[:, 0], _as_array(det1)[:, 1]))
    d2 = float(np.interp(lam, _as_array(det2)[:, 0], _as_array(det2)[:, 1]))

    In = np.eye(N)
    It = np.eye(T)
    Ay = np.kron(It, In - rho * W1) @ y
    BAy = np.kron(It, In - lam * W2) @ Ay
    Bx = np.kron(It, In - lam * W2) @ X

    e = BAy - Bx @ b
    epe = float(e @ e)
    return -(NT / 2) * np.log(2 * np.pi) - (NT / 2) * np.log(max(sige, 1e-12)) + T * d1 + T * d2 - epe / (2 * max(sige, 1e-12))


def sem_panel_FE_LY(y, x, W, N, info: dict | None = None):
    """Estimate SEM panel model with Lee-Yu transformation.

    Parameters
    ----------
    y : array-like
        Stacked dependent variable.
    x : array-like
        Stacked regressors.
    W : array-like
        Cross-sectional weights matrix.
    N : int
        Number of units.
    info : dict, optional
        Placeholder compatibility argument from MATLAB API.

    Returns
    -------
    dict
        SEM estimation dictionary including transformed data arrays.
    """
    ytr, Xtr, Tm1 = _lee_yu_transform(y, x, N)
    res = _panel_sem_mle(ytr, Xtr, _as_array(W), Tm1)
    res["meth"] = "sem_panel_FE_LY"
    res["ytrans"] = ytr
    res["xtrans"] = Xtr
    return res


def sar_panel_FE_LY(y, x, W, N, info: dict | None = None):
    """Estimate SAR panel model with Lee-Yu transformation.

    Parameters
    ----------
    y : array-like
        Stacked dependent variable.
    x : array-like
        Stacked regressors.
    W : array-like
        Cross-sectional weights matrix.
    N : int
        Number of units.
    info : dict, optional
        Placeholder compatibility argument from MATLAB API.

    Returns
    -------
    dict
        SAR estimation dictionary including transformed data arrays.
    """
    ytr, Xtr, Tm1 = _lee_yu_transform(y, x, N)
    res = _panel_sar_mle(ytr, Xtr, _as_array(W), Tm1)
    res["meth"] = "sar_panel_FE_LY"
    res["ytrans"] = ytr
    res["xtrans"] = Xtr
    return res


def sarar_panel_FE_LY(y, x, W1, W2, N, info: dict | None = None):
    """Estimate SARAR panel model with Lee-Yu transformation.

    Parameters
    ----------
    y : array-like
        Stacked dependent variable.
    x : array-like
        Stacked regressors.
    W1 : array-like
        SAR weights matrix.
    W2 : array-like
        SEM weights matrix.
    N : int
        Number of units.
    info : dict, optional
        Placeholder compatibility argument from MATLAB API.

    Returns
    -------
    dict
        SARAR estimation dictionary including transformed data arrays.
    """
    ytr, Xtr, Tm1 = _lee_yu_transform(y, x, N)
    res = _panel_sarar_mle(ytr, Xtr, _as_array(W1), _as_array(W2), Tm1)
    res["meth"] = "sarar_panel_FE_LY"
    res["ytrans"] = ytr
    res["xtrans"] = Xtr
    return res


def lm_f_err(y, x, W, N):
    """Compute LM test for spatial error dependence in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W : array-like
        Cross-sectional weights matrix.
    N : int
        Number of units.

    Returns
    -------
    dict
        Test result dictionary with ``lm`` and ``prob``.
    """
    ytr, Xtr, Tm1 = _lee_yu_transform(y, x, N)
    fe = _ols(ytr, Xtr)
    resid = fe["resid"]
    vew = float((resid @ resid) / max(N * Tm1, 1))

    Ws = _as_array(W)
    Wk = _panel_W(Ws, Tm1)

    lme = float(resid.T @ Wk @ resid / max(vew, 1e-12))
    tr1 = float(Tm1 * np.sum(Ws * Ws.T))
    tr2 = float(Tm1 * np.sum(Ws * Ws))
    T22 = tr1 + tr2

    lm = (lme * lme) / max(T22, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 1))

    return {
        "meth": "lm_f_err",
        "lm": lm,
        "prob": prob,
        "nvar": Xtr.shape[1],
        "chi_1": 6.635,
    }


def lm_f_sar(y, x, W, N):
    """Compute LM test for SAR dependence in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W : array-like
        Cross-sectional weights matrix.
    N : int
        Number of units.

    Returns
    -------
    dict
        Test result dictionary with ``lm`` and ``prob``.
    """
    ytr, Xtr, Tm1 = _lee_yu_transform(y, x, N)
    fe = _ols(ytr, Xtr)
    resid = fe["resid"]
    b = fe["beta"]
    vew = float((resid @ resid) / max(N * Tm1, 1))

    Wk = _panel_W(_as_array(W), Tm1)
    M = np.eye(N * Tm1) - Xtr @ np.linalg.pinv(Xtr.T @ Xtr) @ Xtr.T

    Ws = _as_array(W)
    tr1 = float(Tm1 * np.sum(Ws * Ws.T))
    tr2 = float(Tm1 * np.sum(Ws * Ws))
    T11 = tr1 + tr2

    D = float((Wk @ Xtr @ b).T @ M @ (Wk @ Xtr @ b) / max(vew, 1e-12))
    lml = float(resid.T @ Wk @ ytr / max(vew, 1e-12))
    lm = (lml * lml) / max(D + T11, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 1))

    return {
        "meth": "lm_f_sar",
        "lm": lm,
        "prob": prob,
        "nvar": Xtr.shape[1],
        "chi_1": 6.635,
    }


def lm_f_joint(y, x, W, W2, N):
    """Compute joint LM test for SAR and SEM effects in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    N : int
        Number of units.

    Returns
    -------
    dict
        Joint LM result dictionary with chi-squared(2) p-value.
    """
    ytr, Xtr, Tm1 = _lee_yu_transform(y, x, N)
    fe = _ols(ytr, Xtr)
    resid = fe["resid"]
    bw = fe["beta"]
    vew = float((resid @ resid) / max(N * Tm1, 1))

    Wk = _panel_W(_as_array(W), Tm1)
    W2k = _panel_W(_as_array(W2), Tm1)

    Re = float(resid.T @ W2k @ resid / max(vew, 1e-12))
    Ry = float(resid.T @ Wk @ ytr / max(vew, 1e-12))

    Ws = _as_array(W)
    W2s = _as_array(W2)

    T11 = float(Tm1 * (np.sum(Ws * Ws.T) + np.sum(Ws * Ws)))
    T22 = float(Tm1 * (np.sum(W2s * W2s.T) + np.sum(W2s * W2s)))
    T12 = float(Tm1 * (np.sum(Ws * W2s.T) + np.sum(Ws * W2s)))

    M = np.eye(N * Tm1) - Xtr @ np.linalg.pinv(Xtr.T @ Xtr) @ Xtr.T
    D = float((Wk @ Xtr @ bw).T @ M @ (Wk @ Xtr @ bw) / max(vew, 1e-12))

    E = (D + T11) * T22 - (T12 * T12)
    lm = (T22 * (Ry * Ry) - 2 * T12 * Re * Ry + (Re * Re) * (D + T11)) / max(E, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 2))

    return {
        "meth": "lm_f_joint",
        "lm": lm,
        "prob": prob,
        "nvar": Xtr.shape[1],
        "chi_1": 9.210,
    }


def lm_f_err_c(y, x, W1, W2, N):
    """Compute conditional LM test for SAR given SEM in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W1, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    N : int
        Number of units.

    Returns
    -------
    dict
        Conditional LM result dictionary.
    """
    nt, k = _as_array(x).shape
    T = nt // N

    res = sem_panel_FE_LY(y, x, W2, N)
    ytr = res["ytrans"]
    Xtr = res["xtrans"]
    resid = res["resid"]
    rho = float(res["rho"])
    sige = float(res["sige"])
    beta = res["beta"]

    It = np.eye(T - 1)
    W1s = _as_array(W1)
    W2s = _as_array(W2)
    Bn = np.eye(N) - rho * W2s
    Bt = np.kron(It, Bn)
    W1star = np.kron(It, W1s)

    epB = resid.T @ Bt
    Wsytil = W1star @ ytr
    num = float((epB @ Wsytil / max(sige, 1e-12)) ** 2)

    W1sp = W1s.T
    tr1 = float((T - 1) * np.sum(W1s * W1sp))
    xbet = Xtr @ beta
    Wxbet = W1star @ xbet
    BWxbet = Bt @ Wxbet
    tr2 = float((BWxbet.T @ BWxbet) / max(sige, 1e-12))
    Bi = np.linalg.inv(Bn)
    A = Bn @ W1s @ Bi
    tr3 = float((T - 1) * np.sum(A * A))
    I11 = tr1 + tr2 + tr3

    WBnpBn = W1s.T @ Bn.T @ Bn
    BpB = np.kron(It, WBnpBn)
    term1 = float((beta.T @ Xtr.T @ BpB @ Xtr @ beta) / max(sige, 1e-12))
    WBi = W2s @ Bi
    term2 = float((T - 1) * (np.trace(WBi.T @ A) + np.trace(W2s @ W1s @ Bi)))
    term3 = float((T - 1) * np.trace(A) / max(sige, 1e-12))
    I12 = np.array([term1, term2, term3], dtype=float)

    I22 = np.eye(3)
    den = I11 - I12 @ I22 @ I12.T
    lm = num / max(den, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 1))

    return {
        "meth": "lm_f_err_c",
        "lm": lm,
        "prob": prob,
        "nvar": k,
        "chi_1": 6.635,
    }


def lm_f_sar_c(y, x, W1, W2, N):
    """Compute conditional LM test for SEM given SAR in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W1, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    N : int
        Number of units.

    Returns
    -------
    dict
        Conditional LM result dictionary.
    """
    nt, k = _as_array(x).shape
    T = nt // N

    res = sar_panel_FE_LY(y, x, W1, N)
    e = res["resid"]
    sige = float(res["sige"])
    rho = float(res["rho"])

    it = np.eye(T - 1)
    W1s = _as_array(W1)
    W2s = _as_array(W2)
    iw2 = np.kron(it, W2s)

    t_rho = float(res["tstat"][k, 0])
    if np.isfinite(t_rho) and abs(t_rho) > 1e-8:
        tmps = t_rho / max(abs(rho), 1e-8)
        varp = float(1.0 / max(tmps * tmps, 1e-8))
    else:
        varp = 1.0

    A = np.eye(N) - rho * W1s
    AI = np.linalg.inv(A)

    T22 = float((T - 1) * (np.sum(W2s * W2s.T) + np.sum(W2s * W2s)))
    Tlr = float((T - 1) * (np.trace(W2s.T @ W1s @ AI) + np.trace(W2s @ W1s @ AI)))

    lm1 = float(e.T @ iw2 @ e / max(sige, 1e-12))
    lm = (lm1 * lm1) / max(T22 - (Tlr * Tlr) * varp, 1e-12)
    prob = float(1.0 - stats.chi2.cdf(lm, 1))

    return {
        "meth": "lm_f_sar_c",
        "lm": lm,
        "prob": prob,
        "nvar": k,
        "chi_1": 6.635,
    }


def _ols_loglik_transformed(y, X):
    """Compute Gaussian OLS log-likelihood for transformed data.

    Parameters
    ----------
    y : np.ndarray
        Transformed dependent variable.
    X : np.ndarray
        Transformed regressors.

    Returns
    -------
    float
        Gaussian log-likelihood under OLS residual variance estimate.
    """
    res = _ols(y, X)
    n = len(y)
    e = res["resid"]
    sig2 = float((e @ e) / n)
    return -(n / 2) * np.log(2 * np.pi) - (n / 2) * np.log(max(sig2, 1e-12)) - n / 2


def lr_f_err(y, x, W, N):
    """Compute LR test for SEM effect versus OLS in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W : array-like
        Cross-sectional SEM weights matrix.
    N : int
        Number of units.

    Returns
    -------
    dict
        LR result dictionary with ``lr`` and ``prob``.
    """
    _, X = _as_array(y), _as_array(x)
    ytr, Xtr, _ = _lee_yu_transform(y, x, N)
    liku = sem_panel_FE_LY(y, x, W, N)["lik"]
    likr = _ols_loglik_transformed(ytr, Xtr)
    lr = max(0.0, 2 * (liku - likr))
    prob = float(1.0 - stats.chi2.cdf(lr, 1))
    return {"meth": "lr_f_err", "lr": lr, "prob": prob, "nvar": X.shape[1], "chi_1": 6.635}


def lr_f_sar(y, x, W, N):
    """Compute LR test for SAR effect versus OLS in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W : array-like
        Cross-sectional SAR weights matrix.
    N : int
        Number of units.

    Returns
    -------
    dict
        LR result dictionary with ``lr`` and ``prob``.
    """
    X = _as_array(x)
    ytr, Xtr, _ = _lee_yu_transform(y, x, N)
    liku = sar_panel_FE_LY(y, x, W, N)["lik"]
    likr = _ols_loglik_transformed(ytr, Xtr)
    lr = max(0.0, 2 * (liku - likr))
    prob = float(1.0 - stats.chi2.cdf(lr, 1))
    return {"meth": "lr_f_sar", "lr": lr, "prob": prob, "nvar": X.shape[1], "chi_1": 6.635}


def lr_f_err_c(y, x, W, W2, N):
    """Compute conditional LR test for SAR given SEM in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    N : int
        Number of units.

    Returns
    -------
    dict
        Conditional LR result dictionary.
    """
    X = _as_array(x)
    liku = sarar_panel_FE_LY(y, x, W, W2, N)["lik"]
    likr = sem_panel_FE_LY(y, x, W2, N)["lik"]
    lr = max(0.0, 2 * (liku - likr))
    prob = float(1.0 - stats.chi2.cdf(lr, 1))
    return {"meth": "lr_f_err_c", "lr": lr, "prob": prob, "nvar": X.shape[1], "chi_1": 6.635}


def lr_f_sar_c(y, x, W, W2, N):
    """Compute conditional LR test for SEM given SAR in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    N : int
        Number of units.

    Returns
    -------
    dict
        Conditional LR result dictionary.
    """
    X = _as_array(x)
    liku = sarar_panel_FE_LY(y, x, W, W2, N)["lik"]
    likr = sar_panel_FE_LY(y, x, W, N)["lik"]
    lr = max(0.0, 2 * (liku - likr))
    prob = float(1.0 - stats.chi2.cdf(lr, 1))
    return {"meth": "lr_f_sar_c", "lr": lr, "prob": prob, "nvar": X.shape[1], "chi_1": 6.635}


def lr_f_joint(y, x, W, W2, N):
    """Compute joint LR test for SAR and SEM effects in FE panel model.

    Parameters
    ----------
    y, x : array-like
        Stacked dependent variable and regressors.
    W, W2 : array-like
        Cross-sectional SAR and SEM weights matrices.
    N : int
        Number of units.

    Returns
    -------
    dict
        Joint LR result dictionary with chi-squared(2) p-value.
    """
    X = _as_array(x)
    ytr, Xtr, _ = _lee_yu_transform(y, x, N)
    liku = sarar_panel_FE_LY(y, x, W, W2, N)["lik"]
    likr = _ols_loglik_transformed(ytr, Xtr)
    lr = max(0.0, 2 * (liku - likr))
    prob = float(1.0 - stats.chi2.cdf(lr, 2))
    return {"meth": "lr_f_joint", "lr": lr, "prob": prob, "nvar": X.shape[1], "chi_1": 9.210}


def prt_fe(result: dict) -> str:
    """Format panel FE test dictionary as printable text.

    Parameters
    ----------
    result : dict
        Result dictionary from panel FE test functions.

    Returns
    -------
    str
        Human-readable multiline summary.
    """
    lines = [f"Method: {result.get('meth', 'unknown')}"]
    for key in ["lm", "lr", "prob", "chi_1", "nvar"]:
        if key in result:
            lines.append(f"{key}: {result[key]}")
    return "\n".join(lines)


def prt_back(result: dict) -> str:
    """Alias of :func:`prt_fe` for compatibility.

    Parameters
    ----------
    result : dict
        Result dictionary from panel FE test functions.

    Returns
    -------
    str
        Human-readable multiline summary.
    """
    return prt_fe(result)


def prt_test_fe(result: dict) -> str:
    """Alias of :func:`prt_fe` for compatibility.

    Parameters
    ----------
    result : dict
        Result dictionary from panel FE test functions.

    Returns
    -------
    str
        Human-readable multiline summary.
    """
    return prt_fe(result)
