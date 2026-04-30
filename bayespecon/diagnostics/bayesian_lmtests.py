def bayesian_panel_lm_wx_sem_test(
    model,
) -> "BayesianLMTestResult":
    r"""Bayesian panel LM test for WX coefficients in SEM (H₀: γ = 0 | SEM).

    Tests whether spatially lagged covariates (WX) should be added to a
    panel SEM model, i.e., whether the SEM panel model should be extended to an SDEM panel.
    Follows the multi-parameter Bayesian LM test framework of
    :cite:t:`dogan2021BayesianRobust`. The classical WX-in-SEM construction is
    discussed in :cite:t:`koley2024UseNot`; panel-data extensions are covered
    in :cite:t:`elhorst2014SpatialEconometrics` and :cite:t:`anselin2008SpatialPanel`.

    The null model is panel SEM (includes λ but not γ). For each posterior draw
    from the SEM model, residuals are:

    .. math::
        \mathbf{e} = \mathbf{y} - X \beta

    The score vector for the WX coefficients is:

    .. math::
        \mathbf{g}_\gamma = (WX)^\top \mathbf{e}

    The concentration matrix is the information matrix evaluated at the posterior mean:

    .. math::
        J_{\gamma\gamma} = \frac{1}{\bar{\sigma}^2} (WX)^\top (WX)

    Parameters
    ----------
    model : SEMPanelFE or SEMPanelRE
        Fitted SEM panel model instance with ``inference_data`` attribute
        containing posterior draws for ``beta``, ``lambda``, and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`k_{wx}` (number of WX columns).
    """
    y = model._y
    X = model._X
    WX = model._WX
    k_wx = WX.shape[1]
    N = getattr(model, "_N", None)
    T = getattr(model, "_T", None)

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The panel WX test requires at least one "
            "spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Residuals: e = y - X@beta
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # For RE models, also subtract alpha
    resid = _maybe_subtract_alpha(model, idata, resid)

    # Score: g_gamma = WX' @ e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # M_X-projected raw-score variance (Koley-Bera 2024).
    sigma2_mean = float(np.mean(sigma_draws**2))
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)

    V_inv = _safe_inv(V_gamma_gamma, "V_gamma_gamma (panel WX-SEM)")
    LM = np.einsum("di,ij,dj->d", g_gamma, V_inv, g_gamma)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_wx_sem",
        df=k_wx,
        details={"k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_lm_wx_sem_test(
    model,
) -> "BayesianLMTestResult":
    r"""Bayesian LM test for WX coefficients in SEM (H₀: γ = 0 | SEM).

    Tests whether spatially lagged covariates (WX) should be added to a
    SEM model — i.e. whether SEM should be extended to SDEM.  Bayesian
    extension of the classical LM-WX test
    (:cite:p:`koley2024UseNot`) using the Doğan, Taşpınar & Bera (2021)
    framework (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is SEM (includes :math:`\lambda` but not
    :math:`\gamma`).  For each posterior draw of
    :math:`(\beta, \lambda, \sigma^2)` the raw score is

    .. math::
        \mathbf{g}_\gamma^{(d)} = (WX)^\top \mathbf{e}^{(d)},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Under :math:`H_0` the variance of the raw score is the same Schur-
    complemented quantity used by spreg's ``lm_wx``
    (:cite:p:`koley2024UseNot`):

    .. math::
        V_{\gamma\gamma} = \bar{\sigma}^2 \, (WX)^\top M_X (WX),

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` and
    :math:`\bar{\sigma}^2` is the posterior mean of :math:`\sigma^2`.
    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}_\gamma^{(d)\,\top}
            V_{\gamma\gamma}^{-1} \mathbf{g}_\gamma^{(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SEM
        Fitted SEM model with ``inference_data`` containing posterior
        draws of ``beta``, ``lambda``, ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = k_{wx}`` and
        metadata.
    """
    y = model._y
    X = model._X
    WX = model._WX
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The WX test requires at least one "
            "spatially lagged covariate. Ensure the model was constructed "
            "with a W matrix and w_vars."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Per-draw raw score
    g_gamma = resid @ WX  # (draws, k_wx)

    # M_X-projected raw-score variance (Koley-Bera 2024).
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)

    V_inv = _safe_inv(V_gamma_gamma, "V_gamma_gamma (LM-WX-SEM)")
    LM = np.einsum("di,ij,dj->d", g_gamma, V_inv, g_gamma)

    return _finalize_lm(
        LM,
        test_type="bayesian_lm_wx_sem",
        df=k_wx,
        details={"k_wx": k_wx},
    )


"""
Bayesian LM-type diagnostics for spatial models :cite:p:`dogan2021BayesianRobust`.

Implements Bayesian LM tests for omitted spatial lag (SAR) and error (SEM)
models, as well as SDM/SDEM variant tests (WX, joint, and robust),
following the formulas in :cite:t:`dogan2021BayesianRobust` and :cite:t:`koley2024UseNot`.

Panel variants follow :cite:t:`anselin2008SpatialPanel` and :cite:t:`elhorst2014SpatialEconometrics` for the
information-matrix adjustment (T multiplier, Wb'MWb term).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import arviz as az
import numpy as np
from scipy import stats as sp_stats


def _finalize_lm(
    LM: np.ndarray,
    *,
    test_type: str,
    df: int,
    details: Optional[Dict[str, Any]] = None,
) -> "BayesianLMTestResult":
    """Build a :class:`BayesianLMTestResult` from per-draw LM samples.

    Centralises the boilerplate (mean / median / 95% CI / Bayesian p-value
    against ``chi2(df)``) shared by every Bayesian LM test in this module.
    The ``n_draws`` entry is added to ``details`` automatically.
    """
    LM = np.asarray(LM)
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))
    merged: Dict[str, Any] = {"n_draws": int(LM.shape[0])}
    if details:
        merged.update(details)
    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type=test_type,
        df=df,
        details=merged,
    )


def _safe_inv(M: np.ndarray, label: str = "information matrix") -> np.ndarray:
    """Robust matrix inverse for LM-test information / cross-product matrices.

    Adds a tiny ridge (``1e-12 * I``) to prevent exact singularity, checks the
    condition number of the regularised matrix, and emits a ``RuntimeWarning``
    plus falls back to ``np.linalg.pinv`` when ``cond > 1e12``. This replaces
    the previous silent ``np.linalg.inv(M + 1e-12 * np.eye(...))`` pattern,
    which masked rank-deficient information matrices without notice.

    Parameters
    ----------
    M : np.ndarray
        Square matrix to invert (typically a Fisher information block or
        :math:`X^\\top X`).
    label : str
        Human-readable label used in the warning message.

    Returns
    -------
    np.ndarray
        Either ``inv(M + ε I)`` (well-conditioned) or ``pinv(M)``
        (ill-conditioned).
    """
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    M_reg = M + 1e-12 * np.eye(n)
    cond = np.linalg.cond(M_reg)
    if not np.isfinite(cond) or cond > 1e12:
        import warnings

        warnings.warn(
            f"{label} is ill-conditioned (cond={cond:.2e}); falling back to pseudo-inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.pinv(M)
    return np.linalg.inv(M_reg)


def _mx_quadratic(X: np.ndarray, v: np.ndarray) -> float:
    r"""Compute :math:`v^\top M_X v = v^\top v - v^\top X (X^\top X)^{-1} X^\top v`.

    Here :math:`M_X = I - X(X^\top X)^{-1} X^\top` is the OLS annihilator
    matrix.  Used in the concentrated information-matrix blocks of the
    Bayesian LM-Lag / LM-WX / SDM-joint tests
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13;
    :cite:p:`koley2024UseNot`, Section 3) where the OLS coefficient
    :math:`\beta` has been concentrated out via Schur complement.
    """
    Xv = X.T @ v
    XtX = X.T @ X
    sol, *_ = np.linalg.lstsq(XtX, Xv, rcond=None)
    proj = float(Xv @ sol)
    return float(v @ v) - proj


def _mx_cross(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    r"""Compute :math:`U^\top M_X V = U^\top V - U^\top X (X^\top X)^{-1} X^\top V`.

    Generalises :func:`_mx_quadratic` to arbitrary left/right factors.
    Returns shape inferred from ``U`` and ``V`` (matches ``U.T @ V``):
    scalar / vector / matrix as appropriate.
    """
    XtX = X.T @ X
    XU = X.T @ U  # (k,) or (k, m1)
    XV = X.T @ V  # (k,) or (k, m2)
    UtV = U.T @ V
    sol, *_ = np.linalg.lstsq(XtX, XV, rcond=None)
    return UtV - XU.T @ sol


def _resolve_X_for_beta(model, beta_draws: np.ndarray) -> np.ndarray:
    """Return ``model._X`` augmented with ``model._WX`` iff ``beta`` covers both blocks.

    Centralises the auto-detect-WX hstack pattern used by the cross-sectional
    LM-lag/LM-error tests where the same code path must accept residuals from
    OLS, SLX, SDM and SDEM posteriors.
    """
    k_beta = beta_draws.shape[1]
    if (
        hasattr(model, "_WX")
        and model._WX.shape[1] > 0
        and k_beta == model._X.shape[1] + model._WX.shape[1]
    ):
        return np.hstack([model._X, model._WX])
    return model._X


def _maybe_subtract_alpha(model, idata, resid: np.ndarray) -> np.ndarray:
    """Subtract per-unit random-effect ``alpha[unit_idx]`` from ``resid`` if present.

    Used by panel LM tests so that the cross-sectional residual formulas
    transparently work for both pooled and random-effects fits.
    """
    if hasattr(model, "_unit_idx") and "alpha" in idata.posterior:
        alpha_draws = _get_posterior_draws(idata, "alpha")
        resid = resid - alpha_draws[:, model._unit_idx]
    return resid


def _neyman_adjust_scalar(
    g_t: np.ndarray,
    g_n: np.ndarray,
    J_tt: float,
    J_tn: np.ndarray,
    J_nn: np.ndarray,
    *,
    label: str,
):
    """Apply the Neyman-orthogonal score correction for a scalar target parameter.

    Computes ``g_t* = g_t - J_{tn} J_{nn}^{-1} g_n`` and the adjusted variance
    ``V* = J_{tt} - J_{tn} J_{nn}^{-1} J_{nt}``. If the nuisance block is empty
    (``g_n.shape[1] == 0``) the unadjusted score and variance are returned.

    Parameters
    ----------
    g_t : np.ndarray, shape (draws,)
        Score samples for the scalar target parameter.
    g_n : np.ndarray, shape (draws, m)
        Score samples for the nuisance block.
    J_tt : float
        Information for the target parameter.
    J_tn : np.ndarray, shape (m,)
        Cross-information between target and nuisance.
    J_nn : np.ndarray, shape (m, m)
        Information for the nuisance block.
    label : str
        Label passed to :func:`_safe_inv` for diagnostic warnings.

    Returns
    -------
    g_t_star : np.ndarray, shape (draws,)
    V_star : float
    """
    g_n = np.atleast_2d(g_n)
    if g_n.shape[1] == 0:
        return g_t, float(J_tt)
    J_nn_inv = _safe_inv(np.atleast_2d(J_nn), label)
    J_tn_arr = np.atleast_1d(J_tn)
    coef = J_tn_arr @ J_nn_inv  # (m,)
    g_t_star = g_t - g_n @ coef
    V_star = float(J_tt) - float(coef @ J_tn_arr)
    return g_t_star, V_star


@dataclass
class BayesianLMTestResult:
    """Container for Bayesian LM test results.

    Attributes
    ----------
    lm_samples : np.ndarray
        Posterior samples of the LM statistic.
    mean : float
        Posterior mean of the LM statistic.
    median : float
        Posterior median of the LM statistic.
    credible_interval : Tuple[float, float]
        95% credible interval for the LM statistic.
    bayes_pvalue : float
        P-value computed at the posterior mean of the LM statistic,
        ``1 - chi2.cdf(mean, df)``. Under H₀ the LM statistic follows
        a :math:`\\chi^2` distribution (:cite:p:`dogan2021BayesianRobust`, Proposition 1).
    test_type : str
        Label identifying the test (e.g. ``"bayesian_lm_lag"``).
    df : int
        Degrees of freedom for the reference :math:`\\chi^2` distribution.
        Default is 1 for single-parameter tests.
    details : Dict[str, Any]
        Additional metadata (e.g. number of posterior draws).
    """

    lm_samples: np.ndarray
    mean: float
    median: float
    credible_interval: Tuple[float, float]
    bayes_pvalue: float
    test_type: str
    df: int = 1
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self):
        """Convert the test result to a dictionary."""
        d = {
            "lm_samples": self.lm_samples,
            "mean": self.mean,
            "median": self.median,
            "credible_interval": self.credible_interval,
            "bayes_pvalue": self.bayes_pvalue,
            "test_type": self.test_type,
            "df": self.df,
        }
        if self.details:
            for k in self.details:
                d[k] = self.details[k]
        return d

    def to_series(self):
        """Convert the test result to a pandas Series."""
        import pandas as pd

        return pd.Series(self.to_dict())


def _get_posterior_draws(idata: az.InferenceData, param: str) -> np.ndarray:
    """Extract posterior draws for a parameter from an ArviZ InferenceData object.

    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData object containing posterior samples.
    param : str
        Name of the parameter to extract.

    Returns
    -------
    np.ndarray
        Posterior draws, reshaped to (draws, ...) where draws is the total
        number of samples.
    """
    arr = idata.posterior[param]
    arr = arr.values if hasattr(arr, "values") else arr
    return arr.reshape(-1, *arr.shape[2:])


def bayesian_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM test for omitted spatial lag (SAR) model.

    Bayesian extension of the classical LM-Lag test
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13) using the Doğan,
    Taşpınar & Bera (2021) quadratic-net-loss framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1):

    1. Score :math:`s_\rho(\theta)` evaluated at every posterior draw of
       :math:`(\beta, \sigma^2)` from the OLS null fit.
    2. Concentration matrix :math:`C_{\rho\rho \cdot \beta}(\theta^\star)`
       evaluated at the posterior mean :math:`\theta^\star` (Doğan eq. 3.6).
    3. Posterior averaging of the per-draw quadratic form yields the
       Bayesian LM statistic with asymptotic :math:`\chi^2_1` reference.

    For each posterior draw the raw score is

    .. math::
        S^{(d)} = \mathbf{e}^{(d)\,\top} W \mathbf{y},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Concentrating :math:`\beta` out of the SAR Fisher information
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13) gives the variance of
    the raw score under :math:`H_0`:

    .. math::
        V = \bar{\sigma}^4 \, T_{WW}
            + \bar{\sigma}^2 \, \| M_X \, W X \bar{\beta} \|^2,

    where :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`,
    :math:`M_X = I - X(X^\top X)^{-1} X^\top` is the OLS annihilator,
    :math:`\bar{\sigma}^2` is the posterior mean of :math:`\sigma^2`, and
    :math:`\bar{\beta}` is the posterior mean of :math:`\beta`.  The
    projected term :math:`\| M_X W X \bar{\beta} \|^2` is the same
    Schur-complement quantity that appears in spreg's ``lmLag`` denominator
    (Anselin 1996 derivation), evaluated at the posterior-mean
    :math:`\beta` rather than the OLS estimate.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \frac{\bigl(S^{(d)}\bigr)^2}{V}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    and the Bayesian p-value is computed at the posterior-mean LM
    (:cite:p:`dogan2021BayesianRobust`, eq. 3.7).

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``, plus the cached
        ``_y``, ``_X``, ``_Wy``, ``_T_ww`` attributes.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df=1`` and metadata.

    """
    y = model._y
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    # When the model is SLX/SDM/SDEM the ``beta`` vector covers ``[X, WX]``
    X = _resolve_X_for_beta(model, beta_draws)
    Wy = model._Wy
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    T_ww = model._T_ww

    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)
    # Per-draw raw score s_rho^(d) = e^(d)' W y
    S = np.dot(resid, Wy)  # (draws,)

    # Concentration matrix evaluated at theta*: V = sigma^4 * T_ww +
    # sigma^2 * ||M_X (W X beta_bar)||^2  (Anselin 1996, eq. 13).
    beta_mean = np.mean(beta_draws, axis=0)
    Wy_hat = np.asarray(model._W_sparse @ (X @ beta_mean)).ravel()
    proj_norm_sq = _mx_quadratic(X, Wy_hat)
    V = sigma2_mean**2 * T_ww + sigma2_mean * proj_norm_sq

    LM = S**2 / (V + 1e-12)
    return _finalize_lm(LM, test_type="bayesian_lm_lag", df=1)


def bayesian_lm_error_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM test for omitted spatial error (SEM) model.

    Bayesian extension of the classical LM-Error test
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 9) using the Doğan,
    Taşpınar & Bera (2021) quadratic-net-loss framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).  The score and
    concentration matrix come from the OLS log-likelihood; the spatial
    error parameter :math:`\lambda` is information-orthogonal to
    :math:`\beta` under :math:`H_0` so no Schur projection is needed.

    For each posterior draw the raw score is

    .. math::
        S^{(d)} = \mathbf{e}^{(d)\,\top} W \mathbf{e}^{(d)},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Under :math:`H_0` with spherical errors, the variance of the raw
    score (negative-Hessian block at :math:`\theta^\star`) is

    .. math::
        V = \bar{\sigma}^4 \, T_{WW},
        \qquad T_{WW} = \mathrm{tr}(W^\top W + W^2),

    where :math:`\bar{\sigma}^2` is the posterior mean of :math:`\sigma^2`.
    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \frac{\bigl(S^{(d)}\bigr)^2}{V}
        \;\xrightarrow{d}\; \chi^2_1 \quad \text{under } H_0,

    and the Bayesian p-value is computed at the posterior-mean LM
    (:cite:p:`dogan2021BayesianRobust`, eq. 3.7).

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``, plus the cached
        ``_y``, ``_X``, ``_W_sparse``, ``_T_ww`` attributes.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df=1`` and metadata.

    """
    y = model._y
    W_sp = model._W_sparse
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    X = _resolve_X_for_beta(model, beta_draws)
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)
    # We = W @ resid via sparse matmul (avoids dense W)
    We = (W_sp @ resid.T).T  # (draws, n)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    T_ww = model._T_ww
    # Per-draw raw score s_lambda^(d) = e^(d)' W e^(d)
    S = np.sum(resid * We, axis=1)  # (draws,)
    # Variance of raw score at theta*: sigma^4 * T_ww (Anselin 1996, eq. 9).
    V = sigma2_mean**2 * T_ww
    LM = S**2 / (V + 1e-12)
    return _finalize_lm(LM, test_type="bayesian_lm_error", df=1)


def bayesian_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM test for WX coefficients (H₀: γ = 0 | SAR).

    Tests whether spatially lagged covariates (WX) should be added to a
    SAR model — i.e. whether SAR should be extended to SDM.  Bayesian
    extension of the classical LM-WX test
    (:cite:p:`koley2024UseNot`, eq. for ``RS_gamma``) using the Doğan,
    Taşpınar & Bera (2021) framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is SAR (includes :math:`\rho` but not :math:`\gamma`).
    For each posterior draw of :math:`(\beta, \rho, \sigma^2)` the raw
    score is

    .. math::
        \mathbf{g}_\gamma^{(d)} = (WX)^\top \mathbf{e}^{(d)},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - \rho^{(d)} W\mathbf{y}
                                  - X \beta^{(d)}.

    Concentrating :math:`\beta` out of the SDM information matrix gives
    the variance of the raw score under :math:`H_0`:

    .. math::
        V_{\gamma\gamma} = \bar{\sigma}^2 \,
            (WX)^\top M_X (WX), \qquad M_X = I - X(X^\top X)^{-1} X^\top.

    This is the same Schur-complement quantity used by spreg's ``lm_wx``
    (:cite:p:`koley2024UseNot`), evaluated at the posterior mean
    :math:`\bar{\sigma}^2`.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}_\gamma^{(d)\,\top}
            V_{\gamma\gamma}^{-1} \mathbf{g}_\gamma^{(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SAR
        Fitted SAR model with ``inference_data`` attribute providing
        posterior draws of ``beta``, ``rho``, ``sigma`` and the cached
        ``_y``, ``_X``, ``_WX``, ``_Wy`` attributes.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, and metadata.
        ``df = k_{wx}`` (the number of WX columns).
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The WX test requires at least one "
            "spatially lagged covariate. Ensure the model was constructed "
            "with a W matrix and w_vars."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)

    # Per-draw SAR residuals: e = y - rho W y - X beta
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Per-draw raw score g_gamma^(d) = (WX)' e^(d)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Variance of raw score at theta*: sigma^2 * (WX)' M_X (WX)
    # (Koley-Bera 2024 Schur complement; spreg's lm_wx uses this matrix.)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)  # (k_wx, k_wx)

    V_inv = _safe_inv(V_gamma_gamma, "V_gamma_gamma (LM-WX)")
    LM = np.einsum("di,ij,dj->d", g_gamma, V_inv, g_gamma)

    return _finalize_lm(
        LM,
        test_type="bayesian_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx},
    )


def bayesian_lm_sdm_joint_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian joint LM test for SDM (H₀: ρ = 0 AND γ = 0 | OLS).

    Bayesian extension of the joint LM-SDM test
    (:cite:p:`koley2024UseNot`, ``lm_spdurbin``) using the Doğan,
    Taşpınar & Bera (2021) framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is OLS.  For each posterior draw of
    :math:`(\beta, \sigma^2)` the joint raw score is

    .. math::
        \mathbf{g}^{(d)} = \begin{pmatrix}
            \mathbf{e}^{(d)\,\top} W \mathbf{y} \\
            (WX)^\top \mathbf{e}^{(d)}
        \end{pmatrix},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Concentrating :math:`\beta` out of the SDM information matrix
    (:cite:p:`koley2024UseNot`) gives the
    :math:`(1 + k_{wx}) \times (1 + k_{wx})` variance matrix of the raw
    score:

    .. math::
        V = \begin{pmatrix}
            \bar{\sigma}^4\, T_{WW}
              + \bar{\sigma}^2\, \| M_X W X \bar{\beta} \|^2
            & \bar{\sigma}^2\, (W X \bar{\beta})^\top M_X (WX) \\
            \bar{\sigma}^2\, (WX)^\top M_X (W X \bar{\beta})
            & \bar{\sigma}^2\, (WX)^\top M_X (WX)
        \end{pmatrix},

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` is the OLS annihilator
    and :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  This matches the
    Schur-complemented information matrix in spreg's ``lm_spdurbin``,
    evaluated at the posterior-mean :math:`(\bar{\beta}, \bar{\sigma}^2)`
    rather than the OLS estimate.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}^{(d)\,\top} V^{-1} \mathbf{g}^{(d)}
        \;\xrightarrow{d}\; \chi^2_{1 + k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1 + k_{wx}`` and
        metadata.

    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    # Per-draw OLS residuals: e = y - X beta
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Raw score components
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)
    g = np.column_stack([g_rho, g_gamma])  # (draws, 1+k_wx)

    # Concentration matrix at theta* with M_X Schur projection
    # (Koley-Bera 2024 / spreg lm_spdurbin algebra)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    beta_mean = np.mean(beta_draws, axis=0)
    Wy_hat = np.asarray(W_sp @ (X @ beta_mean)).ravel()
    T_ww = model._T_ww

    p = 1 + k_wx
    V = np.zeros((p, p))
    V[0, 0] = sigma2_mean**2 * T_ww + sigma2_mean * _mx_quadratic(X, Wy_hat)
    if k_wx > 0:
        cross = sigma2_mean * np.asarray(_mx_cross(X, Wy_hat, WX)).ravel()
        V[0, 1:] = cross
        V[1:, 0] = cross
        V[1:, 1:] = sigma2_mean * _mx_cross(X, WX, WX)

    V_inv = _safe_inv(V, "V (SDM joint)")
    LM = np.einsum("di,ij,dj->d", g, V_inv, g)

    return _finalize_lm(
        LM,
        test_type="bayesian_lm_sdm_joint",
        df=p,
        details={"k_wx": k_wx},
    )


def bayesian_lm_slx_error_joint_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian joint LM test for SDEM (H₀: λ = 0 AND γ = 0 | OLS).

    Bayesian extension of the joint LM-SLX-Error test
    (:cite:p:`koley2024UseNot`, ``lm_slxerr``) using the Doğan,
    Taşpınar & Bera (2021) framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is OLS.  For each posterior draw of
    :math:`(\beta, \sigma^2)` the joint raw score is

    .. math::
        \mathbf{g}^{(d)} = \begin{pmatrix}
            \mathbf{e}^{(d)\,\top} W \mathbf{e}^{(d)} \\
            (WX)^\top \mathbf{e}^{(d)}
        \end{pmatrix},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Under :math:`H_0` with spherical errors,
    :math:`\mathrm{Cov}(\mathbf{e}^\top W \mathbf{e},\ (WX)^\top
    \mathbf{e}) = 0` (third moments of normal errors vanish), so the
    information matrix is block-diagonal — matching spreg's
    ``lm_slxerr`` which simply adds ``LM_Error + LM_WX``
    (:cite:p:`koley2024UseNot`).  The variance of the raw score is

    .. math::
        V = \begin{pmatrix}
            \bar{\sigma}^4\, T_{WW} & 0 \\
            0 & \bar{\sigma}^2\, (WX)^\top M_X (WX)
        \end{pmatrix},

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` and
    :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  The per-draw LM
    statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}^{(d)\,\top} V^{-1} \mathbf{g}^{(d)}
        \;\xrightarrow{d}\; \chi^2_{1 + k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1 + k_{wx}`` and
        metadata.

    """
    y = model._y
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    # Per-draw OLS residuals
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Raw score components
    We = (W_sp @ resid.T).T  # (draws, n)
    g_lambda = np.sum(resid * We, axis=1)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)
    g = np.column_stack([g_lambda, g_gamma])  # (draws, 1+k_wx)

    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    T_ww = model._T_ww

    # Block-diagonal raw-score variance (Koley-Bera 2024).
    p = 1 + k_wx
    V = np.zeros((p, p))
    V[0, 0] = sigma2_mean**2 * T_ww
    if k_wx > 0:
        V[1:, 1:] = sigma2_mean * _mx_cross(X, WX, WX)

    V_inv = _safe_inv(V, "V (SLX-error joint)")
    LM = np.einsum("di,ij,dj->d", g, V_inv, g)

    return _finalize_lm(
        LM,
        test_type="bayesian_lm_slx_error_joint",
        df=p,
        details={"k_wx": k_wx},
    )


# ---------------------------------------------------------------------------
# Information matrix blocks for Neyman orthogonal score adjustment
# ---------------------------------------------------------------------------


def _info_matrix_blocks_sdm(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    sigma2: float,
    Wy_hat: np.ndarray | None = None,
    T_ww: float | None = None,
) -> dict:
    r"""Compute raw-score variance blocks for SDM Neyman-orthogonal adjustment.

    Returns the variance blocks of the **raw** scores
    :math:`g_\rho = \mathbf{e}^\top W \mathbf{y}` and
    :math:`\mathbf{g}_\gamma = (WX)^\top \mathbf{e}` evaluated at
    :math:`\theta^\star = (\bar{\beta}, \bar{\sigma}^2)`, with the
    nuisance :math:`\beta` concentrated out via the OLS annihilator
    :math:`M_X = I - X(X^\top X)^{-1} X^\top`
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13;
    :cite:p:`koley2024UseNot`, Section 3).

    .. math::
        V_{\rho\rho} &= \bar{\sigma}^4 \, T_{WW}
            + \bar{\sigma}^2 \, \| M_X W X \bar{\beta} \|^2 \\
        V_{\rho\gamma} &= \bar{\sigma}^2 \, (W X \bar{\beta})^\top M_X (WX) \\
        V_{\gamma\gamma} &= \bar{\sigma}^2 \, (WX)^\top M_X (WX)

    where :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)` and
    :math:`W X \bar{\beta}` are the spatially lagged fitted values under
    :math:`H_0`.  The Neyman-orthogonal adjustment used by
    :func:`bayesian_robust_lm_lag_sdm_test` and
    :func:`bayesian_robust_lm_wx_test`
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3) only depends on
    the **ratios** :math:`V_{\rho\gamma} V_{\gamma\gamma}^{-1}`, which are
    invariant to overall :math:`\sigma^2`-scaling, but the residual
    variance :math:`V_{\rho\rho \cdot \gamma}` must be on the same scale
    as the raw scores — hence the explicit :math:`\sigma^2` factors.

    The dict keys are kept as ``J_rho_rho`` / ``J_rho_gamma`` /
    ``J_gamma_gamma`` for backwards-compatibility.  Their *numerical
    semantics* are the raw-score variance blocks defined above.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` including intercept.
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``.
    sigma2 : float
        Posterior mean of :math:`\sigma^2`.
    Wy_hat : np.ndarray or None, optional
        Spatially lagged fitted values :math:`W X \bar{\beta}` (or
        :math:`W (\rho \mathbf{y} + X\bar{\beta})` for SAR-null contexts).
        If ``None``, the cross-term is set to zero (Neyman adjustment is a
        no-op).
    T_ww : float or None, optional
        Pre-computed :math:`\mathrm{tr}(W^\top W + W^2)`; computed from
        ``W_sparse`` if not supplied.

    Returns
    -------
    dict
        Keys ``J_rho_rho``, ``J_rho_gamma`` (vector of length ``k_wx``),
        ``J_gamma_gamma`` (``k_wx`` x ``k_wx``), ``T_ww``.
    """
    k_wx = WX.shape[1]

    # T_WW = tr(W'W + W²) = ||W||_F^2 + sum(W ⊙ W')
    if T_ww is None:
        T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())

    # V_{γγ} = σ² · (WX)' M_X (WX)  -- M_X-projected, raw-score scale.
    V_gamma_gamma = sigma2 * _mx_cross(X, WX, WX)

    # V_{ρρ} = σ⁴·T_WW + σ²·||M_X W X β̄||²; cross-term = σ²·(W X β̄)'M_X(WX)
    if Wy_hat is not None:
        V_rho_rho = float(sigma2**2 * T_ww + sigma2 * _mx_quadratic(X, Wy_hat))
        V_rho_gamma = sigma2 * np.asarray(_mx_cross(X, Wy_hat, WX)).ravel()
    else:
        V_rho_rho = float(sigma2**2 * T_ww)
        V_rho_gamma = np.zeros(k_wx)

    return {
        "J_rho_rho": V_rho_rho,
        "J_rho_gamma": V_rho_gamma,
        "J_gamma_gamma": V_gamma_gamma,
        "T_ww": T_ww,
    }


def _info_matrix_blocks_sdem(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    sigma2: float,
    T_ww: float | None = None,
) -> dict:
    r"""Compute raw-score variance blocks for SDEM Neyman-orthogonal adjustment.

    Returns the variance blocks of the **raw** scores
    :math:`g_\lambda = \mathbf{e}^\top W \mathbf{e}` and
    :math:`\mathbf{g}_\gamma = (WX)^\top \mathbf{e}` under
    :math:`H_0: \lambda = 0` with spherical errors
    (:cite:p:`koley2024UseNot`, Section 3):

    .. math::
        V_{\lambda\lambda} &= \bar{\sigma}^4 \, T_{WW} \\
        V_{\lambda\gamma}  &= 0 \quad \text{(odd normal moments vanish)} \\
        V_{\gamma\gamma}   &= \bar{\sigma}^2 \, (WX)^\top M_X (WX)

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` and
    :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  The block-diagonal
    structure mirrors spreg's ``lm_slxerr`` decomposition
    (:cite:p:`koley2024UseNot`).  As with
    :func:`_info_matrix_blocks_sdm`, the dict keys retain the historical
    ``J_*`` names but their numerical semantics are raw-score variance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` including intercept.
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``.
    sigma2 : float
        Posterior mean of :math:`\sigma^2`.
    T_ww : float or None, optional
        Pre-computed :math:`\mathrm{tr}(W^\top W + W^2)`; computed from
        ``W_sparse`` if not supplied.

    Returns
    -------
    dict
        Keys ``J_lam_lam``, ``J_lam_gamma`` (zero vector of length
        ``k_wx``), ``J_gamma_gamma`` (``k_wx`` x ``k_wx``), ``T_ww``.
    """
    if T_ww is None:
        T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())

    k_wx = WX.shape[1]

    V_lam_lam = float(sigma2**2 * T_ww)
    V_lam_gamma = np.zeros(k_wx)
    V_gamma_gamma = sigma2 * _mx_cross(X, WX, WX)

    return {
        "J_lam_lam": V_lam_lam,
        "J_lam_gamma": V_lam_gamma,
        "J_gamma_gamma": V_gamma_gamma,
        "T_ww": T_ww,
    }


# ---------------------------------------------------------------------------
# Robust Bayesian LM tests (Neyman orthogonal score, Dogan et al. 2021)
# ---------------------------------------------------------------------------


def bayesian_robust_lm_lag_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Lag test in SDM context (H₀: ρ = 0, robust to γ).

    Tests the null hypothesis that the spatial lag coefficient is zero,
    robust to the local presence of WX effects (γ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3,
    which is the Bayesian analogue of the robust LM-Lag-SDM test in
    :cite:t:`koley2024UseNot`.

    The alternative model is SAR (the SDM relaxation that adds
    :math:`\rho`); the null model used to draw posteriors is **SLX**, in
    which :math:`\gamma` is a free parameter and has already been
    absorbed into the residuals.  For each posterior draw of
    :math:`(\beta, \gamma, \sigma^2)` from the SLX fit, residuals are

    .. math::
        \mathbf{e} = \mathbf{y} - X\beta - WX\gamma,

    and the raw score for :math:`\rho` is
    :math:`g_\rho = \mathbf{e}^\top W \mathbf{y}`.  The companion score
    for :math:`\gamma`, :math:`\mathbf{g}_\gamma = (WX)^\top \mathbf{e}`,
    is identically zero by the OLS normal equations of the SLX fit, so
    the Doğan Neyman-orthogonal adjustment
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3) reduces to a
    no-op and the test simplifies to a Schur-concentrated lag LM.

    Concentrating both :math:`\beta` and :math:`\gamma` out of the SDM
    information matrix at :math:`\theta^\star = (\bar{\beta}, \bar{\gamma},
    \bar{\sigma}^2)` (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13;
    :cite:p:`koley2024UseNot`, Section 3) gives the variance of the raw
    score under :math:`H_0`:

    .. math::
        V_{\rho \cdot \beta, \gamma} = \bar{\sigma}^4 \, T_{WW}
            + \bar{\sigma}^2 \,
              \| M_Z\, W (X \bar{\beta} + W X \bar{\gamma}) \|^2,

    where :math:`Z = [X, WX]` is the SLX design and
    :math:`M_Z = I - Z(Z^\top Z)^{-1} Z^\top` is the SLX OLS annihilator.
    The per-draw robust LM statistic is

    .. math::
        \mathrm{LM}_R^{(d)} = \frac{\bigl(g_\rho^{(d)}\bigr)^2}
                                   {V_{\rho \cdot \beta, \gamma}}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    independent of local misspecification in :math:`\gamma`.

    Parameters
    ----------
    model : SLX
        Fitted SLX model with ``inference_data`` containing posterior
        draws of ``beta`` (covering the stacked ``[X, WX]`` design) and
        ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1`` and metadata.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals (gamma already absorbed)
    Z = np.hstack([X, WX])
    fitted = beta_draws @ Z.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Per-draw raw lag score
    g_rho = np.dot(resid, Wy)  # (draws,)

    # Schur-concentrated variance under H_0 with Z = [X, WX] annihilator
    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = Z @ beta_mean
    Wy_hat = np.asarray(W_sp @ y_hat).ravel()
    V_rho = sigma2_mean**2 * model._T_ww + sigma2_mean * _mx_quadratic(Z, Wy_hat)

    LM = g_rho**2 / (V_rho + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_lag_sdm",
        df=1,
        details={"k_wx": k_wx},
    )


def bayesian_robust_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-WX test (H₀: γ = 0, robust to ρ).

    Bayesian extension of the robust LM-WX test
    (:cite:p:`koley2024UseNot`, ``rlm_wx``) using the Doğan, Taşpınar &
    Bera (2021) Neyman-orthogonal score adjustment
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3).

    The alternative model is SAR (includes :math:`\rho` but not
    :math:`\gamma`).  For each posterior draw of
    :math:`(\beta, \rho, \sigma^2)` from the SAR fit, residuals are
    :math:`\mathbf{e} = \mathbf{y} - \rho W\mathbf{y} - X\beta` and the
    raw scores are

    .. math::
        g_\rho = \mathbf{e}^\top W \mathbf{y}, \qquad
        \mathbf{g}_\gamma = (WX)^\top \mathbf{e}.

    The Neyman-orthogonal adjusted score for :math:`\gamma` is

    .. math::
        \mathbf{g}_\gamma^* = \mathbf{g}_\gamma
            - \frac{V_{\gamma\rho}}{V_{\rho\rho}}\, g_\rho ,

    with raw-score variance blocks supplied by
    :func:`_info_matrix_blocks_sdm`.  By the standard Schur-complement
    identity (:cite:p:`anselin1996SimpleDiagnostic`, Appendix), the
    variance of :math:`\mathbf{g}_\gamma^*` under :math:`H_0` is

    .. math::
        V_{\gamma \cdot \rho} = V_{\gamma\gamma}
            - \frac{V_{\gamma\rho} V_{\rho\gamma}^\top}{V_{\rho\rho}}.

    The robust LM statistic is therefore

    .. math::
        \mathrm{LM}_R^{(d)} = \mathbf{g}_\gamma^{*\,(d)\,\top}
            V_{\gamma \cdot \rho}^{-1}\, \mathbf{g}_\gamma^{*\,(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0,

    independent of local misspecification in :math:`\rho`.

    Parameters
    ----------
    model : SAR
        Fitted SAR model with ``inference_data`` containing posterior
        draws of ``beta``, ``rho``, ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = k_{wx}`` and
        metadata.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The robust LM-WX test requires "
            "at least one spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Per-draw SAR residuals
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T
    resid = y[None, :] - fitted

    # Raw scores
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Posterior-mean fitted values for the M_X projection in the info blocks
    beta_mean = np.mean(beta_draws, axis=0)
    rho_mean = float(np.mean(rho_draws))
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = rho_mean * Wy + X @ beta_mean
    Wy_hat = np.asarray(W_sp @ y_hat).ravel()

    info = _info_matrix_blocks_sdm(
        X, WX, W_sp, sigma2_mean, Wy_hat=Wy_hat, T_ww=model._T_ww
    )
    V_rho_rho = info["J_rho_rho"]
    V_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    V_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # Neyman adjustment: g_gamma* = g_gamma - (V_gamma_rho / V_rho_rho) g_rho
    coef = V_rho_gamma / (V_rho_rho + 1e-12)  # (k_wx,)
    g_gamma_star = g_gamma - np.outer(g_rho, coef)  # (draws, k_wx)

    # Schur complement: Var(g_gamma*) = V_gamma_gamma - V_rho_gamma V_rho_gamma' / V_rho_rho
    V_gamma_given_rho = V_gamma_gamma - np.outer(V_rho_gamma, V_rho_gamma) / (
        V_rho_rho + 1e-12
    )

    V_inv = _safe_inv(V_gamma_given_rho, "V_gamma_given_rho (robust LM-WX)")
    LM = np.einsum("di,ij,dj->d", g_gamma_star, V_inv, g_gamma_star)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx},
    )


def bayesian_robust_lm_error_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Error test in SDEM context (H₀: λ = 0, robust to γ).

    Bayesian extension of the robust LM-Error test in the SDEM context
    (:cite:p:`koley2024UseNot`) using the Doğan, Taşpınar & Bera (2021)
    framework (:cite:p:`dogan2021BayesianRobust`, Proposition 3).

    The alternative model is SDEM (adds :math:`\lambda`); the null model
    is **SLX**, in which :math:`\gamma` is a free parameter and has
    already been absorbed into the residuals.  For each posterior draw of
    :math:`(\beta, \gamma, \sigma^2)` from the SLX fit, residuals are
    :math:`\mathbf{e} = \mathbf{y} - X\beta - WX\gamma` and the raw error
    score is :math:`g_\lambda = \mathbf{e}^\top W \mathbf{e}`.

    Under :math:`H_0` with spherical errors, the variance block
    :math:`V_{\lambda\gamma} = 0` because odd moments of normal errors
    vanish (:cite:p:`koley2024UseNot`).  Hence the Doğan
    Neyman-orthogonal adjustment for :math:`\gamma`-nuisance is exactly
    a no-op, and the test simplifies to

    .. math::
        \mathrm{LM}_R^{(d)} = \frac{\bigl(g_\lambda^{(d)}\bigr)^2}
                                   {\bar{\sigma}^4 \, T_{WW}}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    where :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  Note that this
    is structurally identical to :func:`bayesian_lm_error_test` but with
    residuals drawn from the SLX (γ-augmented) posterior rather than the
    OLS posterior — making the test robust to local misspecification in
    :math:`\gamma`.

    Parameters
    ----------
    model : SLX
        Fitted SLX model with ``inference_data`` containing posterior
        draws of ``beta`` (covering the stacked ``[X, WX]`` design) and
        ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1`` and metadata.
    """
    y = model._y
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals (gamma already absorbed)
    Z = np.hstack([X, WX])
    fitted = beta_draws @ Z.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Per-draw raw error score
    We = (W_sp @ resid.T).T  # (draws, n)
    g_lambda = np.sum(resid * We, axis=1)  # (draws,)

    # Variance under H_0 (J_{lam,gamma} = 0 → no Schur correction needed)
    sigma2_mean = float(np.mean(sigma_draws**2))
    V_lambda = sigma2_mean**2 * model._T_ww

    LM = g_lambda**2 / (V_lambda + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_error_sdem",
        df=1,
        details={"k_wx": k_wx},
    )


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _panel_residuals(model, beta_draws: np.ndarray) -> np.ndarray:
    """Compute panel residuals from posterior beta draws.

    Dispatches on FE vs RE model structure:

    - **FE models** (``_y`` and ``_X`` already demeaned):
      ``e = y - X @ beta``

    - **RE models** (raw scale, with ``alpha`` in posterior):
      ``e = y - X @ beta - alpha[unit_idx]``

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted panel model with ``inference_data`` attribute.
    beta_draws : np.ndarray
        Posterior draws of beta, shape ``(draws, k)``.

    Returns
    -------
    np.ndarray
        Residual matrix of shape ``(draws, n)`` where ``n = N*T``.
    """
    y = model._y
    X = model._X
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # RE models: subtract random effects alpha[unit_idx]
    resid = _maybe_subtract_alpha(model, model.inference_data, resid)

    return resid


def _panel_spatial_lag(W_sparse, v: np.ndarray, N: int, T: int) -> np.ndarray:
    """Apply panel spatial lag W⊗I_T to a vector or matrix of draws.

    For each of the T time periods, the N-length slice is multiplied by
    the N×N sparse weight matrix.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        N×N spatial weights matrix.
    v : np.ndarray
        Either a 1-D vector of length N*T, or a 2-D array of shape
        ``(draws, N*T)``.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.

    Returns
    -------
    np.ndarray
        Spatially lagged array with the same shape as *v*.
    """
    if v.ndim == 1:
        out = np.zeros_like(v)
        for t in range(T):
            s, e = t * N, (t + 1) * N
            out[s:e] = np.asarray(W_sparse @ v[s:e]).ravel()
        return out
    else:
        # v is (draws, N*T)
        out = np.zeros_like(v)
        for t in range(T):
            s, e = t * N, (t + 1) * N
            out[:, s:e] = np.asarray(W_sparse @ v[:, s:e].T).T
        return out


def _panel_trace_WtW_WW(W_sparse) -> float:
    """Compute tr(W'W + W²) from an N×N sparse weights matrix.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        N×N spatial weights matrix.

    Returns
    -------
    float
        Trace of W'W + W².
    """
    # tr(W'W) = ||W||_F^2 = sum(W_ij^2)  [O(nnz)]
    # tr(W^2) = sum_ij W_ij * W_ji = sum(W * W.T)  [O(nnz)]
    return float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())


def _panel_info_matrix_blocks(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    W_eigs: np.ndarray,
    sigma2: float,
    N: int,
    T: int,
    y_hat: np.ndarray | None = None,
    Wy_hat: np.ndarray | None = None,
    T_ww: float | None = None,
) -> dict:
    r"""Raw-score variance blocks for panel LM tests (SDM/SDEM).

    Returns the blocks of the **raw-score variance matrix**
    :math:`V = \sigma^2 J` (i.e. variance of unscaled scores
    :math:`\mathbf{e}^\top W \mathbf{y}`, :math:`(WX)^\top \mathbf{e}`)
    with the appropriate :math:`M_X = I - X(X^\top X)^{-1} X^\top`
    projection on the WX-direction blocks.  Panel-data structure
    contributes the :math:`T` multiplier on the trace term
    (:cite:p:`anselin2008SpatialPanel`, :cite:p:`elhorst2014SpatialEconometrics`):

    .. math::
        V_{\rho\rho}    &= \sigma^4 \, T \cdot \mathrm{tr}(W'W + W^2)
                         + \sigma^2 \, (W\hat{y})^\top M_X (W\hat{y}) \\
        V_{\rho\gamma}  &= \sigma^2 \, (W\hat{y})^\top M_X (WX) \\
        V_{\gamma\gamma}&= \sigma^2 \, (WX)^\top M_X (WX) \\
        V_{\lambda\lambda} &= \sigma^4 \, T \cdot \mathrm{tr}(W'W + W^2) \\
        V_{\rho\lambda} &= \sigma^4 \, T \cdot \mathrm{tr}(W'W + W^2) \\
        V_{\lambda\gamma} &= \mathbf{0} \quad \text{under spherical errors.}

    Dictionary keys are kept as ``J_*`` for backward compatibility with
    earlier API users; the returned values are the raw-score variance
    blocks (``V_*``) in the new convention.

    Parameters
    ----------
    X, WX : np.ndarray
        Design matrix ``(n, k)`` (FE-demeaned for FE models) and
        spatially-lagged covariates ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        N×N spatial weights matrix.
    W_eigs : np.ndarray
        Eigenvalues of W (currently unused; kept for signature stability).
    sigma2 : float
        Posterior-mean :math:`\bar{\sigma}^2`.
    N, T : int
        Cross-sectional units and time periods.
    y_hat, Wy_hat : np.ndarray, optional
        Fitted values and panel-spatial-lag of fitted values, shape
        ``(n,)``.  At least one must be supplied for the
        :math:`V_{\rho\rho}` and :math:`V_{\rho\gamma}` projection terms.
    T_ww : float, optional
        Pre-computed :math:`\mathrm{tr}(W'W + W^2)`; recomputed if None.

    Returns
    -------
    dict
        Keys ``J_rho_rho``, ``J_lam_lam``, ``J_rho_lam``,
        ``J_rho_gamma``, ``J_gamma_gamma``, ``T_ww``, ``T_mult``.  All
        ``J_*`` entries are raw-score variance blocks (with proper
        :math:`M_X` projection on WX directions).
    """
    n = X.shape[0]
    k_wx = WX.shape[1]

    if T_ww is None:
        T_ww = _panel_trace_WtW_WW(W_sparse)
    T_mult = T

    # Variance blocks (raw-score scale, panel)
    J_lam_lam = sigma2 * sigma2 * T_mult * T_ww
    J_rho_lam = sigma2 * sigma2 * T_mult * T_ww  # under H0 (Elhorst 2014)

    if y_hat is not None or Wy_hat is not None:
        if Wy_hat is None:
            Wy_hat = _panel_spatial_lag(W_sparse, y_hat, N, T_mult)
        J_rho_rho = sigma2 * sigma2 * T_mult * T_ww + sigma2 * _mx_quadratic(X, Wy_hat)
    else:
        # Fallback (pre-fit point): drop the M_X term.
        J_rho_rho = sigma2 * sigma2 * T_mult * T_ww + sigma2 * float(n)

    # WX-direction blocks: same M_X-projected form as cross-section.
    if k_wx > 0:
        J_gamma_gamma = sigma2 * _mx_cross(X, WX, WX)
    else:
        J_gamma_gamma = np.zeros((0, 0))

    if Wy_hat is not None and k_wx > 0:
        J_rho_gamma = sigma2 * np.asarray(_mx_cross(X, Wy_hat, WX)).ravel()
    else:
        J_rho_gamma = np.zeros(k_wx)

    return {
        "J_rho_rho": J_rho_rho,
        "J_lam_lam": J_lam_lam,
        "J_rho_lam": J_rho_lam,
        "J_rho_gamma": J_rho_gamma,
        "J_gamma_gamma": J_gamma_gamma,
        "T_ww": T_ww,
        "T_mult": T_mult,
    }


# ---------------------------------------------------------------------------
# Panel Bayesian LM tests — lag and error (Anselin et al. 2008, Elhorst 2014)
# ---------------------------------------------------------------------------


def bayesian_panel_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel LM test for omitted spatial lag (H₀: ρ = 0).

    Follows :cite:t:`anselin2008SpatialPanel` and the Bayesian framework of :cite:t:`dogan2021BayesianRobust`.
    (2021). Tests whether a spatial lag term should be added to a panel
    regression.

    The null model is a pooled/FE panel OLS. For each posterior draw,
    residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X \\beta

    (demeaned for FE models; with alpha subtracted for RE models).

    The score for each draw is:

    .. math::
        S = \\mathbf{e}^\\top W_{NT} \\mathbf{y}

    where :math:`W_{NT} = W \\otimes I_T` is the block-diagonal panel
    weights matrix.

    The information matrix (:cite:t:`anselin2008SpatialPanel`) is:

    .. math::
        J = \\frac{1}{\\sigma^2} \\left(
        (W\\hat{y})^\\top M (W\\hat{y}) + T \\cdot \\mathrm{tr}(W'W + W^2)
        \\right)

    where :math:`M = I - X(X^\\top X)^{-1} X^\\top` and
    :math:`\\hat{y} = X\\bar{\\beta}`.

    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\frac{S^2}{\\sigma^2 \\cdot J}

    which is distributed as :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted panel model (e.g. ``OLSPanelFE``, ``OLSPanelRE``) with
        ``inference_data`` attribute containing posterior draws for
        ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals (handles FE vs RE)
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score: S = e'Wy for each draw
    S = np.dot(resid, Wy)  # (draws,)

    # Compute information matrix for panel LM-lag (Anselin et al. 2008)
    # J = (Wb'MWb + T*tr(W'W+W²)*σ²) / σ²
    # LM = (e'Wy)² / (σ² * J) = (e'Wy)² / (Wb'MWb + T*tr*σ²)
    beta_mean = np.mean(beta_draws, axis=0)  # (k,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = X @ beta_mean  # (n,)

    # Panel spatial lag of y_hat
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)  # (n,)

    # Annihilator matrix: M = I - X(X'X)^{-1}X'
    XtX_inv = _safe_inv(X.T @ X, "X'X (panel LM-lag)")
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = model._T_ww
    J_val = WbMWb + T * T_ww * sigma2_mean

    # LM = S² / (sigma2 * J_val) for each draw
    # But sigma2 varies across draws. Use posterior mean sigma2 for J,
    # and per-draw sigma2 for the score variance.
    # Following the existing cross-sectional pattern: use information matrix
    # evaluated at posterior mean, compute LM per draw.
    sigma2_draws = sigma_draws**2  # (draws,)
    LM = S**2 / (sigma2_draws * J_val + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_lag",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_lm_error_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel LM test for omitted spatial error (H₀: λ = 0).

    Follows :cite:t:`anselin2008SpatialPanel` and the Bayesian framework of :cite:t:`dogan2021BayesianRobust`.
    (2021). Tests whether a spatial error term should be added to a panel
    regression.

    The null model is a pooled/FE panel OLS. For each posterior draw,
    residuals are computed (demeaned for FE, alpha-adjusted for RE).

    The score for each draw is:

    .. math::
        S = \\mathbf{e}^\\top W_{NT} \\mathbf{e}

    The variance is:

    .. math::
        V = \\sigma^4 \\cdot T \\cdot \\mathrm{tr}(W'W + W^2)

    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\frac{S^2}{V}

    which is distributed as :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted panel model (e.g. ``OLSPanelFE``, ``OLSPanelRE``) with
        ``inference_data`` attribute containing posterior draws for
        ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score: S = e'W_nt e for each draw
    # Apply N×N W period-by-period to the N*T stacked residuals
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)  # (draws, n)

    S = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Variance: V = sigma^4 * T * tr(W'W + W²)
    T_ww = model._T_ww
    sigma2_draws = sigma_draws**2  # (draws,)
    V = sigma2_draws**2 * T * T_ww  # (draws,)

    # LM = S² / V
    LM = S**2 / (V + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_error",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_robust_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel robust LM-Lag test (H₀: ρ = 0, robust to λ).

    Follows :cite:t:`elhorst2014SpatialEconometrics`. Tests the null hypothesis that the spatial
    lag coefficient is zero, robust to the local presence of spatial
    error autocorrelation.

    The null model is a pooled/FE panel OLS. The robust LM statistic is:

    .. math::
        \\mathrm{LM}_R = \\frac{
        \\left( \\frac{\\mathbf{e}^\\top W \\mathbf{y}}{\\sigma^2}
        - \\frac{\\mathbf{e}^\\top W \\mathbf{e}}{\\sigma^2} \\right)^2
        }{J - T \\cdot \\mathrm{tr}(W'W + W^2)}

    where :math:`J` is the information matrix from the panel LM-lag test
    and :math:`\\mathrm{tr}` denotes :math:`\\mathrm{tr}(W'W + W^2)`.

    This is distributed as :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted panel model (e.g. ``OLSPanelFE``) with ``inference_data``
        attribute containing posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score for lag: S_lag = e'Wy
    S_lag = np.dot(resid, Wy)  # (draws,)

    # Score for error: S_err = e'W_nt e
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    S_err = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Information matrix for lag test
    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = X @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)
    XtX_inv = _safe_inv(X.T @ X, "X'X (panel robust LM-lag)")
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = model._T_ww
    J_val = WbMWb + T * T_ww * sigma2_mean

    # Robust LM = (S_lag/σ² - S_err/σ²)² / (J - T*tr)
    # where J is in σ² units and tr = tr(W'W+W²)
    sigma2_draws = sigma_draws**2
    robust_score = S_lag / sigma2_draws - S_err / sigma2_draws  # (draws,)
    denom = J_val / sigma2_mean - T * T_ww  # J/σ² - T*tr (scalar)

    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_lag",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_robust_lm_error_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel robust LM-Error test (H₀: λ = 0, robust to ρ).

    Follows :cite:t:`elhorst2014SpatialEconometrics`. Tests the null hypothesis that the spatial
    error coefficient is zero, robust to the local presence of a spatial
    lag.

    The null model is a pooled/FE panel OLS. The robust LM statistic is:

    .. math::
        \\mathrm{LM}_R = \\frac{
        \\left( \\frac{\\mathbf{e}^\\top W \\mathbf{e}}{\\sigma^2}
        - \\frac{T \\cdot \\mathrm{tr}}{J} \\cdot
        \\frac{\\mathbf{e}^\\top W \\mathbf{y}}{\\sigma^2} \\right)^2
        }{
        T \\cdot \\mathrm{tr} \\cdot \\left(1 - \\frac{T \\cdot \\mathrm{tr}}{J}\\right)
        }

    where :math:`J` is the information matrix from the panel LM-lag test
    and :math:`\\mathrm{tr} = \\mathrm{tr}(W'W + W^2)`.

    This is distributed as :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted panel model (e.g. ``OLSPanelFE``) with ``inference_data``
        attribute containing posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score for lag: S_lag = e'Wy
    S_lag = np.dot(resid, Wy)  # (draws,)

    # Score for error: S_err = e'W_nt e
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    S_err = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Information matrix for lag test
    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = X @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)
    XtX_inv = _safe_inv(X.T @ X, "X'X (panel robust LM-error)")
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = model._T_ww
    J_val = WbMWb + T * T_ww * sigma2_mean

    # Robust LM = (S_err/σ² - T*tr/J * S_lag/σ²)² / (T*tr*(1 - T*tr/J))
    # where J is in raw units (not divided by σ²)
    # J/σ² is the scaled version
    J_scaled = J_val / sigma2_mean  # J/σ²
    Ttr = T * T_ww

    sigma2_draws = sigma_draws**2
    robust_score = S_err / sigma2_draws - (Ttr / J_scaled) * S_lag / sigma2_draws
    denom = Ttr * (1 - Ttr / J_scaled)

    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_error",
        df=1,
        details={"N": N, "T": T},
    )


# ---------------------------------------------------------------------------
# Panel Bayesian LM tests — SDM/SDEM variants (Koley & Bera 2024)
# ---------------------------------------------------------------------------


def bayesian_panel_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel LM test for WX coefficients (H₀: γ = 0).

    Tests whether spatially lagged covariates (WX) should be added to a
    SAR panel model, i.e., whether SAR should be extended to SDM.
    Follows the multi-parameter Bayesian LM test framework of
    :cite:t:`dogan2021BayesianRobust` and :cite:t:`koley2024UseNot`.

    The null model is a SAR panel (includes ρ but not γ). For each
    posterior draw from the SAR model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - \\rho W \\mathbf{y} - X \\beta

    The score vector for the WX coefficients is:

    .. math::
        \\mathbf{g}_\\gamma = (WX)^\\top \\mathbf{e}

    The concentration matrix is:

    .. math::
        J_{\\gamma\\gamma} = \\frac{1}{\\bar{\\sigma}^2} (WX)^\\top (WX)

    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\mathbf{g}_\\gamma^\\top J_{\\gamma\\gamma}^{-1}
        \\mathbf{g}_\\gamma

    distributed as :math:`\\chi^2_{k_{wx}}` under H₀.

    Parameters
    ----------
    model : SARPanelFE or SARPanelRE
        Fitted SAR panel model with ``inference_data`` containing
        posterior draws for ``beta``, ``rho``, and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`k_{wx}`.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The panel WX test requires at least one "
            "spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SAR residuals: e = y - rho*Wy - X@beta
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # For RE models, also subtract alpha
    resid = _maybe_subtract_alpha(model, idata, resid)

    # Score: g_gamma = WX' @ e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # M_X-projected raw-score variance (Koley-Bera 2024).
    sigma2_mean = float(np.mean(sigma_draws**2))
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)

    V_inv = _safe_inv(V_gamma_gamma, "V_gamma_gamma (panel LM-WX)")
    LM = np.einsum("di,ij,dj->d", g_gamma, V_inv, g_gamma)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_lm_sdm_joint_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel joint LM test for SDM (H₀: ρ = 0 AND γ = 0).

    Tests the joint null hypothesis that both the spatial lag coefficient
    and the WX coefficients are zero, i.e., whether the OLS panel model
    should be extended to an SDM specification.

    The null model is OLS panel. The joint score vector is:

    .. math::
        \\mathbf{g} = \\begin{pmatrix} \\mathbf{e}^\\top W \\mathbf{y} \\\\
        (WX)^\\top \\mathbf{e} \\end{pmatrix}

    a :math:`(1 + k_{wx}) \\times 1` vector for each draw. The
    concentration matrix uses panel-adjusted J_{ρρ}:

    .. math::
        J = \\begin{pmatrix}
        J_{\\rho\\rho} & J_{\\rho\\gamma} \\\\
        J_{\\gamma\\rho} & J_{\\gamma\\gamma}
        \\end{pmatrix}

    The LM statistic is :math:`\\chi^2_{1 + k_{wx}}` under H₀. The classical
    joint LM construction follows :cite:t:`koley2024UseNot`; panel-data
    extensions are discussed in :cite:t:`anselin2008SpatialPanel` and
    :cite:t:`elhorst2014SpatialEconometrics`. The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted OLS panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.
    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score components
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Joint score: g = [g_rho, g_gamma']' → (draws, 1+k_wx)
    g = np.column_stack([g_rho, g_gamma])  # (draws, 1+k_wx)

    # Information matrix (panel-adjusted)
    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = X @ beta_mean

    info = _panel_info_matrix_blocks(
        X,
        WX,
        W_sp,
        model._W_eigs,
        sigma2_mean,
        N,
        T,
        y_hat=y_hat,
        T_ww=model._T_ww,
    )

    p = 1 + k_wx
    J = np.zeros((p, p))
    J[0, 0] = info["J_rho_rho"]
    if k_wx > 0:
        J[0, 1:] = info["J_rho_gamma"]
        J[1:, 0] = info["J_rho_gamma"]
        J[1:, 1:] = info["J_gamma_gamma"]

    # LM = g' J^{-1} g for each draw
    J_inv = _safe_inv(J, "J (panel SDM joint)")
    LM = np.einsum("di,ij,dj->d", g, J_inv, g)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_sdm_joint",
        df=p,
        details={"k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_lm_slx_error_joint_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel joint LM test for SDEM (H₀: λ = 0 AND γ = 0).

    Tests the joint null hypothesis that both the spatial error coefficient
    and the WX coefficients are zero, i.e., whether the OLS panel model
    should be extended to an SDEM specification.

    The null model is OLS panel. The joint score vector is:

    .. math::
        \\mathbf{g} = \\begin{pmatrix} \\mathbf{e}^\\top W \\mathbf{e} \\\\
        (WX)^\\top \\mathbf{e} \\end{pmatrix}

    The concentration matrix has zero off-diagonal blocks under H₀
    (spherical errors):

    .. math::
        J = \\begin{pmatrix}
        T \\cdot \\mathrm{tr}(W'W + W^2) & 0 \\\\
        0 & \\frac{1}{\\bar{\\sigma}^2} (WX)^\\top (WX)
        \\end{pmatrix}

    The LM statistic is :math:`\\chi^2_{1 + k_{wx}}` under H₀. The classical
    joint LM construction follows :cite:t:`koley2024UseNot`; panel-data
    extensions are discussed in :cite:t:`anselin2008SpatialPanel` and
    :cite:t:`elhorst2014SpatialEconometrics`. The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted OLS panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.
    """
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score for error: g_lambda = e'W_nt e
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    g_lambda = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Score for WX: g_gamma = WX'e
    g_gamma = resid @ WX  # (draws, k_wx)

    # Joint score
    g = np.column_stack([g_lambda, g_gamma])  # (draws, 1+k_wx)

    # Raw-score variance matrix (block-diagonal under H0).
    sigma2_mean = float(np.mean(sigma_draws**2))
    T_ww = model._T_ww

    p = 1 + k_wx
    V = np.zeros((p, p))
    V[0, 0] = sigma2_mean * sigma2_mean * T * T_ww  # V_{lambda,lambda}
    if k_wx > 0:
        V[1:, 1:] = sigma2_mean * _mx_cross(X, WX, WX)

    # LM = g' V^{-1} g for each draw
    V_inv = _safe_inv(V, "V (panel SLX-error joint)")
    LM = np.einsum("di,ij,dj->d", g, V_inv, g)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_slx_error_joint",
        df=p,
        details={"k_wx": k_wx, "N": N, "T": T},
    )


# ---------------------------------------------------------------------------
# Panel robust Bayesian LM tests — SDM/SDEM (Neyman orthogonal score)
# ---------------------------------------------------------------------------


def bayesian_panel_robust_lm_lag_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-Lag in SDM context (H₀: ρ = 0 | SLX panel).

    Tests :math:`H_0: \rho = 0` using the **SLX panel as the restricted
    null model**.  Because the SLX OLS normal equations imply
    :math:`(WX)^\top \mathbf{e}_{slx} = \mathbf{0}` exactly, the Neyman
    orthogonal score adjustment for the WX nuisance direction is a no-op
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3) and the LM
    statistic reduces to the direct quadratic form

    .. math::
        \mathrm{LM}^{(d)} =
        \frac{(\mathbf{e}^{(d)\,\top} W_{NT} \mathbf{y})^2}{
            \bar{\sigma}^4\, T \cdot \mathrm{tr}(W'W + W^2)
          + \bar{\sigma}^2\, \| M_Z W_{NT} \hat{\mathbf{y}}_{slx} \|^2},

    where :math:`Z = [X,\;WX]` is the full SLX design matrix,
    :math:`M_Z = I - Z(Z^\top Z)^{-1} Z^\top`, and
    :math:`\hat{\mathbf{y}}_{slx} = Z \bar{\beta}_{slx}` uses the
    posterior mean.  Under :math:`H_0` the statistic is asymptotically
    :math:`\chi^2_1`.

    Note: this null model differs from the spreg-style algebraic
    subtraction (joint SDM minus LM-WX), which uses an OLS null and
    enforces orthogonality through information-matrix algebra.  The two
    statistics target the same hypothesis but are not numerically equal.

    Parameters
    ----------
    model : SLXPanelFE or SLX-like panel model
        Fitted SLX panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma`` (``beta`` covers the
        full SLX design ``[X, WX]``).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k_total)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals: e = y - [X, WX] @ beta
    Z = np.hstack([X, WX])  # (n, k+k_wx)
    fitted = beta_draws @ Z.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)
    resid = _maybe_subtract_alpha(model, idata, resid)

    # Per-draw raw score for rho
    g_rho = np.dot(resid, Wy)  # (draws,)

    # M_Z-projected variance (Z absorbs gamma => Neyman adjustment is no-op).
    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat_slx = Z @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat_slx, N, T)

    mz_quad = _mx_quadratic(Z, Wy_hat)
    T_ww = model._T_ww
    V_rho = sigma2_mean * sigma2_mean * T * T_ww + sigma2_mean * mz_quad

    LM = g_rho**2 / (V_rho + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_lag_sdm",
        df=1,
        details={"k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_robust_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-WX (H₀: γ = 0 | SAR panel, robust to ρ).

    Bayesian extension of the classical robust LM-WX test
    (:cite:p:`koley2024UseNot`) using the Dogǎn, Tas̛pınar & Bera (2021)
    framework (:cite:p:`dogan2021BayesianRobust`, Proposition 3) with
    panel-data adjustments (:cite:p:`anselin2008SpatialPanel`,
    :cite:p:`elhorst2014SpatialEconometrics`).

    The alternative model is SAR panel (includes :math:`\rho` but not
    :math:`\gamma`).  For each posterior draw of
    :math:`(\beta, \rho, \sigma^2)` the SAR residual is
    :math:`\mathbf{e}^{(d)} = \mathbf{y} - \rho^{(d)} W_{NT} \mathbf{y}
    - X \beta^{(d)}`.  The raw scores are

    .. math::
        g_\rho^{(d)} = \mathbf{e}^{(d)\,\top} W_{NT} \mathbf{y},
        \qquad
        \mathbf{g}_\gamma^{(d)} = (WX)^\top \mathbf{e}^{(d)}.

    The Neyman-orthogonal adjusted score uses the canonical Schur
    complement of the raw-score variance matrix:

    .. math::
        \mathbf{g}_\gamma^{*\,(d)} &= \mathbf{g}_\gamma^{(d)}
            - V_{\gamma\rho} V_{\rho\rho}^{-1} g_\rho^{(d)}, \\
        V_{\gamma\gamma\,|\,\rho} &= V_{\gamma\gamma}
            - V_{\gamma\rho} V_{\rho\rho}^{-1} V_{\rho\gamma}.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}_\gamma^{*\,(d)\,\top}
            V_{\gamma\gamma\,|\,\rho}^{-1} \mathbf{g}_\gamma^{*\,(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SARPanelFE or SARPanelRE
        Fitted SAR panel model with ``inference_data`` containing
        posterior draws for ``beta``, ``rho``, ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_{wx}``.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The robust panel LM-WX test requires "
            "at least one spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SAR residuals: e = y - rho*Wy - X@beta
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T
    resid = y[None, :] - fitted
    resid = _maybe_subtract_alpha(model, idata, resid)

    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Raw-score variance blocks at posterior-mean theta*.
    beta_mean = np.mean(beta_draws, axis=0)
    rho_mean = float(np.mean(rho_draws))
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = rho_mean * Wy + X @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)

    info = _panel_info_matrix_blocks(
        X,
        WX,
        W_sp,
        model._W_eigs,
        sigma2_mean,
        N,
        T,
        Wy_hat=Wy_hat,
        T_ww=model._T_ww,
    )

    V_rho_rho = info["J_rho_rho"]
    V_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    V_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # Canonical Schur complement: V_{gamma | rho}.
    coef = V_rho_gamma / (V_rho_rho + 1e-12)  # (k_wx,)
    V_gamma_given_rho = V_gamma_gamma - np.outer(V_rho_gamma, coef)

    # Neyman orthogonal score: g_gamma* = g_gamma - V_{gamma,rho} V_{rho,rho}^{-1} g_rho
    g_gamma_star = g_gamma - np.outer(g_rho, coef)  # (draws, k_wx)

    V_inv = _safe_inv(V_gamma_given_rho, "V_gamma_given_rho (panel robust LM-WX)")
    LM = np.einsum("di,ij,dj->d", g_gamma_star, V_inv, g_gamma_star)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_robust_lm_error_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-Error in SDEM context (H₀: λ = 0 | SLX panel).

    Tests :math:`H_0: \lambda = 0` using the **SLX panel as the
    restricted null**.  Under spherical errors
    :math:`V_{\lambda\gamma} = \mathbf{0}` and the SLX OLS normal
    equations imply :math:`(WX)^\top \mathbf{e}_{slx} = \mathbf{0}`, so
    the Neyman orthogonal score adjustment for the WX nuisance
    direction is a no-op (:cite:p:`dogan2021BayesianRobust`,
    Proposition 3) and the LM statistic reduces to

    .. math::
        \mathrm{LM}^{(d)} =
        \frac{(\mathbf{e}^{(d)\,\top} W_{NT} \mathbf{e}^{(d)})^2}{
            \bar{\sigma}^4 \, T \cdot \mathrm{tr}(W'W + W^2)},

    asymptotically :math:`\chi^2_1` under :math:`H_0`
    (:cite:p:`anselin2008SpatialPanel`,
    :cite:p:`elhorst2014SpatialEconometrics`).

    Parameters
    ----------
    model : SLXPanelFE or SLX-like panel model
        Fitted SLX panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    y = model._y
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k_total)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals: e = y - [X, WX] @ beta
    Z = np.hstack([X, WX])
    fitted = beta_draws @ Z.T
    resid = y[None, :] - fitted
    resid = _maybe_subtract_alpha(model, idata, resid)

    # Score for error: g_lambda = e'W_nt e
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    g_lambda = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Direct raw-score variance V_lambda_lambda = sigma^4 * T * T_ww
    sigma2_mean = float(np.mean(sigma_draws**2))
    T_ww = model._T_ww
    V_lambda = sigma2_mean * sigma2_mean * T * T_ww

    LM = g_lambda**2 / (V_lambda + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_error_sdem",
        df=1,
        details={"k_wx": k_wx, "N": N, "T": T},
    )


# =====================================================================
# Bayesian LM diagnostics for spatial flow (origin-destination) models
# =====================================================================
#
# These tests extend the Dogan et al. (2021) Bayesian LM framework to the
# OD-flow filter A = I_N - rho_d W_d - rho_o W_o - rho_w W_w used by the
# SARFlow family.  The score blocks come from the conditional Gaussian
# log-likelihood of an OLSFlow null model; the information matrix uses the
# cached Kronecker-trace block ``model._T_flow_traces`` (see
# :func:`bayespecon.graph.flow_trace_blocks`).  Robust variants pull a
# fitted SARFlow as the unrestricted alternative and apply the
# Neyman-orthogonal score adjustment.
#
# Conventions:
#   - Score for spatial-lag direction i evaluated at OLSFlow draws:
#         g_i^{(g)} = (W_i y)^T e^{(g)} ,    e^{(g)} = y - X beta^{(g)}
#   - Information matrix J = T_flow_traces * sigma2_bar + Q,
#     with Q[i,j] = (W_i y)^T (W_j y).
#   - chi^2 reference at the posterior-mean LM, df given per test.


def _flow_score_info(model, *, restrict_keys=("d", "o", "w")):
    """Return per-draw score matrix and the (3,3) information matrix.

    Parameters
    ----------
    model : OLSFlow
        Fitted flow model providing ``inference_data`` (with ``beta`` and
        ``sigma`` posterior draws), the cached spatial lags
        ``_Wd_y``/``_Wo_y``/``_Ww_y``, the cached trace matrix
        ``_T_flow_traces``, and the design matrix ``_X_design``.
    restrict_keys : tuple of {"d","o","w"}
        Subset of the three spatial-lag directions to keep (in order).
        Mainly for test readability — the joint test uses the full triple.

    Returns
    -------
    G : np.ndarray, shape (n_draws, k)
        Score draws, ``G[g, i] = (W_i y)^T e_g``.
    J : np.ndarray, shape (k, k)
        Information matrix
        ``J = T_flow_traces[K, K] * sigma2_bar + Q[K, K]``
        with ``Q[i, j] = (W_i y)^T (W_j y)``.
    """
    y = np.asarray(model._y_vec, dtype=np.float64)
    X = np.asarray(model._X_design, dtype=np.float64)
    Wy_all = np.column_stack(
        [
            np.asarray(model._Wd_y, dtype=np.float64),
            np.asarray(model._Wo_y, dtype=np.float64),
            np.asarray(model._Ww_y, dtype=np.float64),
        ]
    )  # (N, 3)
    idx = {"d": 0, "o": 1, "w": 2}
    cols = np.array([idx[k] for k in restrict_keys], dtype=int)
    Wy = Wy_all[:, cols]  # (N, k)

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (n_draws, p)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (n_draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))

    fitted = beta_draws @ X.T  # (n_draws, N)
    resid = y[None, :] - fitted  # (n_draws, N)

    G = resid @ Wy  # (n_draws, k)

    Q = Wy.T @ Wy  # (k, k)
    T_blk = model._T_flow_traces[np.ix_(cols, cols)]
    J = T_blk * sigma2_mean + Q  # (k, k)

    return G, J


def _flow_marginal_lm(model, key: str, test_type: str) -> BayesianLMTestResult:
    """Helper for the three marginal LM tests (dest / orig / network)."""
    G, J = _flow_score_info(model, restrict_keys=(key,))
    s = G[:, 0]
    v = float(J[0, 0])
    LM = s**2 / (v + 1e-12)
    return _finalize_lm(
        LM,
        test_type=test_type,
        df=1,
        details={"direction": key},
    )


def bayesian_lm_flow_dest_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for an omitted destination-side spatial lag.

    Tests :math:`H_0\colon \rho_d = 0` against the SARFlow alternative,
    using a fitted :class:`~bayespecon.models.flow.OLSFlow` as the null
    model.  For each posterior draw :math:`g` the residual is
    :math:`e_g = y - X \beta_g`; the score is
    :math:`s_g = (W_d y)^\top e_g` and the information-matrix denominator
    is

    .. math::

        V = \operatorname{tr}(W_d^\top W_d + W_d^2)\,\bar\sigma^{2}
            + \|W_d y\|^{2},

    with :math:`\bar\sigma^{2}` the posterior mean of :math:`\sigma^{2}`.
    The Bayesian p-value is :math:`1 - F_{\chi^{2}_{1}}(\overline{LM})`.

    The score / information construction follows the gravity-flow
    spatial-econometrics framework of :cite:t:`lesage2008SpatialEconometric`
    and :cite:t:`lesage2009IntroductionSpatial`. The Bayesian LM statistic
    is computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.
    """
    return _flow_marginal_lm(model, "d", "bayesian_lm_flow_dest")


def bayesian_lm_flow_orig_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for an omitted origin-side spatial lag
    (:math:`H_0\colon \rho_o = 0`).  See :func:`bayesian_lm_flow_dest_test`
    for details and references (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_marginal_lm(model, "o", "bayesian_lm_flow_orig")


def bayesian_lm_flow_network_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for an omitted network spatial lag
    (:math:`H_0\colon \rho_w = 0`).  See :func:`bayesian_lm_flow_dest_test`
    for details and references (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_marginal_lm(model, "w", "bayesian_lm_flow_network")


def bayesian_lm_flow_joint_test(model) -> BayesianLMTestResult:
    r"""Joint Bayesian LM test for the SARFlow filter
    (:math:`H_0\colon \rho_d = \rho_o = \rho_w = 0`).

    For each posterior draw :math:`g` of the :class:`OLSFlow` null,
    builds the score vector
    :math:`g_g = ((W_d y)^\top e_g, (W_o y)^\top e_g, (W_w y)^\top e_g)^\top`
    and the information matrix

    .. math::

        J = T_{\text{flow}}\,\bar\sigma^{2} + Q, \qquad
        Q_{ij} = (W_i y)^\top (W_j y),

    where :math:`T_{\text{flow}}` is the cached :math:`3\times 3` Kronecker
    trace matrix from :func:`bayespecon.graph.flow_trace_blocks`.  The
    statistic is :math:`LM_g = g_g^\top J^{-1} g_g`, distributed
    :math:`\chi^{2}_3` under :math:`H_0`. The construction follows the
    gravity-flow spatial-econometrics framework of
    :cite:t:`lesage2008SpatialEconometric` and
    :cite:t:`lesage2009IntroductionSpatial`; the Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.
    """
    G, J = _flow_score_info(model, restrict_keys=("d", "o", "w"))
    J_inv = _safe_inv(J, "J (flow joint LM)")
    LM = np.einsum("di,ij,dj->d", G, J_inv, G)
    return _finalize_lm(LM, test_type="bayesian_lm_flow_joint", df=3)


# ---------------------------------------------------------------------
# Robust (Neyman-orthogonal) variants — alternative model is a fitted
# SARFlow.  For each posterior draw g of (rho_d, rho_o, rho_w, beta,
# sigma) we build the spatially-filtered residual
#     e_g = y - rho_d^g W_d y - rho_o^g W_o y - rho_w^g W_w y - X beta_g
# and form the full 3-vector score g_g = (W_d y, W_o y, W_w y)^T e_g.
# The information matrix uses the SARFlow posterior mean of sigma^2.
# Marginal robust tests then apply the Neyman adjustment with respect to
# the two nuisance directions.
# ---------------------------------------------------------------------


def _flow_robust_score_info(sarflow_model):
    """Return per-draw 3-vector score and the (3,3) information matrix
    for a fitted :class:`SARFlow`.

    The score is evaluated at the SARFlow posterior draws of
    ``(rho_d, rho_o, rho_w, beta)``; the information matrix uses the
    SARFlow posterior mean of :math:`\\sigma^{2}`.
    """
    y = np.asarray(sarflow_model._y_vec, dtype=np.float64)
    X = np.asarray(sarflow_model._X_design, dtype=np.float64)
    Wd_y = np.asarray(sarflow_model._Wd_y, dtype=np.float64)
    Wo_y = np.asarray(sarflow_model._Wo_y, dtype=np.float64)
    Ww_y = np.asarray(sarflow_model._Ww_y, dtype=np.float64)
    Wy = np.column_stack([Wd_y, Wo_y, Ww_y])  # (N, 3)

    idata = sarflow_model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    rho_d_draws = _get_posterior_draws(idata, "rho_d").reshape(-1)
    rho_o_draws = _get_posterior_draws(idata, "rho_o").reshape(-1)
    rho_w_draws = _get_posterior_draws(idata, "rho_w").reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma").reshape(-1)
    sigma2_mean = float(np.mean(sigma_draws**2))

    fitted = beta_draws @ X.T  # (n_draws, N)
    rho_stack = np.column_stack([rho_d_draws, rho_o_draws, rho_w_draws])  # (n,3)
    spatial = rho_stack @ Wy.T  # (n_draws, N)
    resid = y[None, :] - spatial - fitted

    G = resid @ Wy  # (n_draws, 3)

    Q = Wy.T @ Wy
    J = sarflow_model._T_flow_traces * sigma2_mean + Q

    return G, J


def _flow_robust_marginal_lm(
    sarflow_model, target: int, test_type: str
) -> BayesianLMTestResult:
    """Neyman-orthogonal LM for a single direction with two nuisances."""
    G, J = _flow_robust_score_info(sarflow_model)
    nuisance = [i for i in (0, 1, 2) if i != target]
    g_t = G[:, target]
    g_n = G[:, nuisance]  # (n_draws, 2)
    J_tt = float(J[target, target])
    J_tn = J[target, nuisance]  # (2,)
    J_nn = J[np.ix_(nuisance, nuisance)]  # (2, 2)

    # adjusted score: g_t* = g_t - J_{t,n} J_{n,n}^{-1} g_n
    g_star, V_star = _neyman_adjust_scalar(
        g_t,
        g_n,
        J_tt,
        J_tn,
        J_nn,
        label=f"J_nuisance (flow robust {test_type})",
    )
    LM = g_star**2 / (V_star + 1e-12)

    direction = {0: "d", 1: "o", 2: "w"}[target]
    return _finalize_lm(
        LM,
        test_type=test_type,
        df=1,
        details={"direction": direction},
    )


def bayesian_robust_lm_flow_dest_test(sarflow_model) -> BayesianLMTestResult:
    r"""Robust Bayesian LM test for :math:`\rho_d` adjusting for
    :math:`(\rho_o, \rho_w)` nuisance via the Neyman-orthogonal score
    (:cite:t:`bera1993SpecificationTesting`, :cite:t:`anselin1996SimpleDiagnostic`,
    :cite:t:`dogan2021BayesianRobust`).  Requires a fitted
    :class:`~bayespecon.models.flow.SARFlow`."""
    return _flow_robust_marginal_lm(sarflow_model, 0, "bayesian_robust_lm_flow_dest")


def bayesian_robust_lm_flow_orig_test(sarflow_model) -> BayesianLMTestResult:
    r"""Robust Bayesian LM test for :math:`\rho_o` adjusting for
    :math:`(\rho_d, \rho_w)` nuisance.  See
    :func:`bayesian_robust_lm_flow_dest_test` for details and references
    (:cite:p:`bera1993SpecificationTesting`,
    :cite:p:`anselin1996SimpleDiagnostic`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_robust_marginal_lm(sarflow_model, 1, "bayesian_robust_lm_flow_orig")


def bayesian_robust_lm_flow_network_test(sarflow_model) -> BayesianLMTestResult:
    r"""Robust Bayesian LM test for :math:`\rho_w` adjusting for
    :math:`(\rho_d, \rho_o)` nuisance.  See
    :func:`bayesian_robust_lm_flow_dest_test` for details and references
    (:cite:p:`bera1993SpecificationTesting`,
    :cite:p:`anselin1996SimpleDiagnostic`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_robust_marginal_lm(sarflow_model, 2, "bayesian_robust_lm_flow_network")


# ---------------------------------------------------------------------
# Phase 5 — WX-style test for the intra-block in OLSFlow.
# ---------------------------------------------------------------------


def bayesian_lm_flow_intra_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM (WX-style) test for the intra block in an OLSFlow null.

    Tests :math:`H_0\colon \beta_{\text{intra}} = 0` in the gravity model

    .. math::

        y = \alpha + g \gamma + X_d \beta_d + X_o \beta_o
            + X_{\text{intra}} \beta_{\text{intra}} + \varepsilon,

    using a fitted :class:`OLSFlow` whose design already contains the
    intra block (the standard layout from
    :func:`bayespecon.graph.flow_design_matrix`).  Per posterior draw the
    intra-block score is :math:`g_g = X_{\text{intra}}^\top e_g` with
    :math:`e_g = y - X \beta_g` (full-design residual) and the variance
    is :math:`V = X_{\text{intra}}^\top X_{\text{intra}} / \bar\sigma^{2}`,
    matching the SDM-style WX information block of
    :cite:t:`dogan2021BayesianRobust`.  The statistic is
    :math:`LM_g = g_g^\top V^{-1} g_g`, df :math:`= k_{\text{intra}}`.

    Notes
    -----
    The test treats the intra block as the "WX" addition relative to a
    null with the same dest/orig coefficients but no intra columns.  The
    score is evaluated at the OLSFlow draws — this is asymptotically
    equivalent to refitting an "OLSFlow without intra" null when the
    intra block is orthogonal to ``X_dest``/``X_orig`` (typically the
    case for the diagonal-mask layout used by ``flow_design_matrix``).
    """
    intra_idx = getattr(model, "_intra_idx", None)
    if intra_idx is None or len(intra_idx) == 0:
        raise ValueError(
            "Model has no intra columns to test.  Build the design with "
            "flow_design_matrix(..., include_intra=True) before fitting."
        )
    intra_idx = np.asarray(intra_idx, dtype=int)

    y = np.asarray(model._y_vec, dtype=np.float64)
    X = np.asarray(model._X_design, dtype=np.float64)
    Xi = X[:, intra_idx]  # (N, k_intra)
    k_intra = Xi.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (n_draws, p)
    sigma_draws = _get_posterior_draws(idata, "sigma").reshape(-1)
    sigma2_mean = float(np.mean(sigma_draws**2))

    fitted = beta_draws @ X.T
    resid = y[None, :] - fitted  # (n_draws, N)
    G = resid @ Xi  # (n_draws, k_intra)

    V = (Xi.T @ Xi) / sigma2_mean
    V_inv = _safe_inv(V, "V_intra (flow intra LM)")
    LM = np.einsum("di,ij,dj->d", G, V_inv, G)

    return _finalize_lm(
        LM,
        test_type="bayesian_lm_flow_intra",
        df=k_intra,
        details={"k_intra": k_intra},
    )


# ---------------------------------------------------------------------
# Phase 6 — Panel analogues.  Operate on demeaned (_y, _X) panel stacks
# of length N_flow * T from FlowPanelModel.  The information matrix
# stays additive in T because Q[i,j] = (W_i y)^T (W_j y) is computed on
# the full stacked panel lags (already cached as _Wd_y/_Wo_y/_Ww_y on
# the demeaned panel).
# ---------------------------------------------------------------------


def _flow_panel_score_info(model, *, restrict_keys=("d", "o", "w")):
    """Score / information for a fitted :class:`OLSFlowPanel` null."""
    y = np.asarray(model._y, dtype=np.float64)  # demeaned panel stack
    X = np.asarray(model._X, dtype=np.float64)
    Wy_all = np.column_stack(
        [
            np.asarray(model._Wd_y, dtype=np.float64),
            np.asarray(model._Wo_y, dtype=np.float64),
            np.asarray(model._Ww_y, dtype=np.float64),
        ]
    )
    idx = {"d": 0, "o": 1, "w": 2}
    cols = np.array([idx[k] for k in restrict_keys], dtype=int)
    Wy = Wy_all[:, cols]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    sigma_draws = _get_posterior_draws(idata, "sigma").reshape(-1)
    sigma2_mean = float(np.mean(sigma_draws**2))

    fitted = beta_draws @ X.T
    resid = y[None, :] - fitted
    G = resid @ Wy

    # Per-period traces accumulated over T periods (independence across t
    # under H_0 with the demeaning transform).
    T = int(getattr(model, "_T", 1))
    Q = Wy.T @ Wy
    T_blk = model._T_flow_traces[np.ix_(cols, cols)]
    J = T * T_blk * sigma2_mean + Q
    return G, J


def _flow_panel_marginal_lm(model, key: str, test_type: str) -> BayesianLMTestResult:
    G, J = _flow_panel_score_info(model, restrict_keys=(key,))
    s = G[:, 0]
    v = float(J[0, 0])
    LM = s**2 / (v + 1e-12)
    return _finalize_lm(
        LM,
        test_type=test_type,
        df=1,
        details={"direction": key, "T": int(getattr(model, "_T", 1))},
    )


def bayesian_panel_lm_flow_dest_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_dest_test` operating on a
    fitted :class:`~bayespecon.models.flow_panel.OLSFlowPanel`.

    The score is summed over the demeaned panel stack
    (length :math:`n^{2}\cdot T`); the information matrix scales the
    Kronecker trace block by :math:`T` to reflect i.i.d. within-period
    contributions under :math:`H_0`. Spatial-flow gravity construction
    follows :cite:t:`lesage2008SpatialEconometric`; panel-data treatment
    follows :cite:t:`elhorst2014SpatialEconometrics` and
    :cite:t:`anselin2008SpatialPanel`. The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`."""
    return _flow_panel_marginal_lm(model, "d", "bayesian_panel_lm_flow_dest")


def bayesian_panel_lm_flow_orig_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_orig_test`
    (:math:`H_0\colon \rho_o = 0`). See
    :func:`bayesian_panel_lm_flow_dest_test` for details and references
    (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_panel_marginal_lm(model, "o", "bayesian_panel_lm_flow_orig")


def bayesian_panel_lm_flow_network_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_network_test`
    (:math:`H_0\colon \rho_w = 0`). See
    :func:`bayesian_panel_lm_flow_dest_test` for details and references
    (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_panel_marginal_lm(model, "w", "bayesian_panel_lm_flow_network")


def bayesian_panel_lm_flow_joint_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_joint_test` on
    :class:`OLSFlowPanel`. The joint score over
    :math:`(\rho_d, \rho_o, \rho_w)` is :math:`\chi^2_3` under :math:`H_0`.
    See :func:`bayesian_panel_lm_flow_dest_test` for additional details
    and references (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    G, J = _flow_panel_score_info(model, restrict_keys=("d", "o", "w"))
    J_inv = _safe_inv(J, "J (panel flow joint LM)")
    LM = np.einsum("di,ij,dj->d", G, J_inv, G)
    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_flow_joint",
        df=3,
        details={"T": int(getattr(model, "_T", 1))},
    )


def bayesian_panel_lm_flow_intra_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_intra_test` on
    :class:`OLSFlowPanel`. Operates on the demeaned panel design.
    Score :math:`g = X_{\text{intra}}^\top e` and information
    :math:`V = X_{\text{intra}}^\top X_{\text{intra}} / \bar\sigma^2`,
    distributed :math:`\chi^2_{k_{\text{intra}}}` under :math:`H_0`.
    See :func:`bayesian_lm_flow_intra_test` for the cross-sectional
    construction (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    intra_idx = getattr(model, "_intra_idx", None)
    if intra_idx is None or len(intra_idx) == 0:
        raise ValueError(
            "Panel model has no intra columns to test.  Build the design "
            "with flow_design_matrix(..., include_intra=True)."
        )
    intra_idx = np.asarray(intra_idx, dtype=int)

    y = np.asarray(model._y, dtype=np.float64)
    X = np.asarray(model._X, dtype=np.float64)
    Xi = X[:, intra_idx]
    k_intra = Xi.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    sigma_draws = _get_posterior_draws(idata, "sigma").reshape(-1)
    sigma2_mean = float(np.mean(sigma_draws**2))

    fitted = beta_draws @ X.T
    resid = y[None, :] - fitted
    G = resid @ Xi

    V = (Xi.T @ Xi) / sigma2_mean
    V_inv = _safe_inv(V, "V_intra (panel flow intra LM)")
    LM = np.einsum("di,ij,dj->d", G, V_inv, G)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_flow_intra",
        df=k_intra,
        details={"k_intra": k_intra, "T": int(getattr(model, "_T", 1))},
    )


# ---------------------------------------------------------------------------
# Cross-sectional robust LM tests (OLS-null, Anselin et al. 1996)
# ---------------------------------------------------------------------------


def _ols_lag_information(
    model,
    beta_mean: np.ndarray,
    sigma2_mean: float,
) -> float:
    r"""Anselin (1996) cross-sectional information for ρ in SAR | OLS.

    Computes :math:`J_{\rho\rho} = (W X \hat\beta)^\top M (W X \hat\beta)
    + T_{WW}\,\sigma^2`, where :math:`M = I - X(X^\top X)^{-1}X^\top`
    and :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.
    """
    X = model._X
    W_sp = model._W_sparse
    T_ww = model._T_ww
    y_hat = X @ beta_mean
    Wy_hat = np.asarray(W_sp @ y_hat).ravel()
    XtX_inv = _safe_inv(X.T @ X, "X'X (cross-sectional robust LM-Lag)")
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)
    return WbMWb + T_ww * sigma2_mean


def bayesian_robust_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Lag test (H₀: ρ = 0, robust to local λ).

    Cross-sectional analogue of :func:`bayesian_panel_robust_lm_lag_test`.
    Tests whether the spatial-lag coefficient is zero, robust to the local
    presence of spatial-error autocorrelation. The null model is OLS.

    For each posterior draw,

    .. math::
        \mathrm{LM}_R = \frac{\bigl(g_\rho/\sigma^2 - g_\lambda/\sigma^2\bigr)^2}
                              {J_{\rho\rho}/\sigma^2 - T_{WW}}

    where :math:`g_\rho = \mathbf{e}^\top W\mathbf{y}`,
    :math:`g_\lambda = \mathbf{e}^\top W\mathbf{e}`,
    :math:`J_{\rho\rho} = (WX\hat\beta)^\top M (WX\hat\beta) + T_{WW}\bar\sigma^2`,
    and :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`. Distributed as
    :math:`\chi^2_1` under H₀. The robust adjustment follows the
    Anselin–Bera–Florax–Yoon construction
    (:cite:p:`anselin1996SimpleDiagnostic`) derived from the locally-misspecified
    Lagrange-multiplier framework of :cite:t:`bera1993SpecificationTesting`.
    The Bayesian LM statistic is computed per posterior draw following
    :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-style model with ``inference_data`` containing posterior
        draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    X_full = _resolve_X_for_beta(model, beta_draws)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    fitted = beta_draws @ X_full.T  # (draws, n)
    resid = y[None, :] - fitted

    S_lag = np.dot(resid, Wy)  # (draws,)
    We = (W_sp @ resid.T).T  # (draws, n)
    S_err = np.sum(resid * We, axis=1)  # (draws,)

    sigma2_draws = sigma_draws**2
    sigma2_mean = float(np.mean(sigma2_draws))

    beta_mean = np.mean(beta_draws, axis=0)
    # Use the X-only design when computing the analytical J (matches the
    # OLS-null Anselin formulation; if beta covers WX, we still use X here).
    beta_mean_x = beta_mean[: X.shape[1]]
    J_val = _ols_lag_information(model, beta_mean_x, sigma2_mean)
    denom = J_val / sigma2_mean - T_ww

    robust_score = S_lag / sigma2_draws - S_err / sigma2_draws
    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(LM, test_type="bayesian_robust_lm_lag", df=1)


def bayesian_robust_lm_error_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Error test (H₀: λ = 0, robust to local ρ).

    Cross-sectional analogue of :func:`bayesian_panel_robust_lm_error_test`.
    Tests whether the spatial-error coefficient is zero, robust to the
    local presence of a spatial lag. The null model is OLS.

    For each posterior draw,

    .. math::
        \mathrm{LM}_R = \frac{\bigl(g_\lambda/\sigma^2 -
            (T_{WW}/J^*_{\rho\rho})\,g_\rho/\sigma^2\bigr)^2}
                              {T_{WW}\bigl(1 - T_{WW}/J^*_{\rho\rho}\bigr)}

    where :math:`J^*_{\rho\rho} = J_{\rho\rho}/\sigma^2` and the remaining
    quantities are as in :func:`bayesian_robust_lm_lag_test`. Distributed
    as :math:`\chi^2_1` under H₀ following the Anselin–Bera–Florax–Yoon
    locally-robust construction (:cite:p:`anselin1996SimpleDiagnostic`,
    :cite:p:`bera1993SpecificationTesting`). The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-style model with ``inference_data`` containing posterior
        draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    X_full = _resolve_X_for_beta(model, beta_draws)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    fitted = beta_draws @ X_full.T
    resid = y[None, :] - fitted

    S_lag = np.dot(resid, Wy)
    We = (W_sp @ resid.T).T
    S_err = np.sum(resid * We, axis=1)

    sigma2_draws = sigma_draws**2
    sigma2_mean = float(np.mean(sigma2_draws))

    beta_mean = np.mean(beta_draws, axis=0)
    beta_mean_x = beta_mean[: X.shape[1]]
    J_val = _ols_lag_information(model, beta_mean_x, sigma2_mean)
    J_scaled = J_val / sigma2_mean

    robust_score = S_err / sigma2_draws - (T_ww / J_scaled) * S_lag / sigma2_draws
    denom = T_ww * (1.0 - T_ww / J_scaled)
    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(LM, test_type="bayesian_robust_lm_error", df=1)


# ---------------------------------------------------------------------------
# SDM/SDEM-aware LM tests (correct residuals from the super-model posterior)
# ---------------------------------------------------------------------------


def bayesian_lm_error_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM-Error test from an SDM posterior (H₀: λ = 0 | SDM).

    Tests whether an SDM model should be extended to a MANSAR (Manski)
    model that adds a spatially correlated error term. Residuals are
    computed using the SDM mean structure including the spatial lag of
    ``y``:

    .. math::
        \mathbf{e} = \mathbf{y} - \rho W\mathbf{y} - X\beta - WX\theta

    so that the LM-Error score and variance are evaluated at posterior
    draws from the *correct* fitted model. The score and variance follow
    the standard LM-Error formulation
    (:cite:t:`anselin1996SimpleDiagnostic`):

    .. math::
        S = \mathbf{e}^\top W\mathbf{e}, \qquad
        V = T_{WW}\,\bar\sigma^2

    Returns ``LM = S^2 / V`` per draw, distributed as :math:`\chi^2_1`
    under H₀.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    rho_draws = _get_posterior_draws(idata, "rho").reshape(-1)  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    Z = np.hstack([X, WX])
    fitted = beta_draws @ Z.T  # (draws, n)
    # subtract rho * Wy per draw
    resid = y[None, :] - fitted - rho_draws[:, None] * Wy[None, :]

    We = (W_sp @ resid.T).T  # (draws, n)
    S = np.sum(resid * We, axis=1)  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    V = T_ww * sigma2_mean
    LM = S**2 / (V + 1e-12)

    return _finalize_lm(LM, test_type="bayesian_lm_error_sdm", df=1)


def bayesian_lm_lag_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM-Lag test from an SDEM posterior (H₀: ρ = 0 | SDEM).

    Tests whether an SDEM model should be extended to MANSAR by adding a
    spatial lag of ``y``. Residuals are the spatially filtered SDEM
    residuals:

    .. math::
        \boldsymbol{\varepsilon} = (I - \lambda W)
            \bigl(\mathbf{y} - X\beta - WX\theta\bigr)

    The score and variance follow the standard LM-Lag formulation
    (:cite:t:`anselin1996SimpleDiagnostic`):

    .. math::
        S = \boldsymbol{\varepsilon}^\top W\mathbf{y}, \qquad
        V = T_{WW}\,\bar\sigma^2 + \|W\mathbf{y}\|^2

    Returns ``LM = S^2 / V`` per draw, distributed as :math:`\chi^2_1`
    under H₀.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    lam_draws = _get_posterior_draws(idata, "lam").reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    Z = np.hstack([X, WX])
    u = y[None, :] - beta_draws @ Z.T  # (draws, n) — pre-filter residuals
    Wu = (W_sp @ u.T).T  # (draws, n)
    # eps = u - lam * W u
    eps = u - lam_draws[:, None] * Wu

    S = np.dot(eps, Wy)  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    V = T_ww * sigma2_mean + float(np.dot(Wy, Wy))
    LM = S**2 / (V + 1e-12)

    return _finalize_lm(LM, test_type="bayesian_lm_lag_sdem", df=1)


# ---------------------------------------------------------------------------
# Panel analogues of the SDM/SDEM-aware LM tests
# ---------------------------------------------------------------------------


def bayesian_panel_lm_error_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Panel LM-Error test from an SDM panel posterior (H₀: λ = 0 | SDM).

    Panel analogue of :func:`bayesian_lm_error_sdm_test`. Residuals are
    computed from the SDM panel mean structure, including
    :math:`\rho\,(I_T \otimes W)\mathbf{y}` and (for RE) the unit-level
    random effect:

    .. math::
        \mathbf{e} = \mathbf{y} - \rho\,(I_T\otimes W)\mathbf{y}
                     - X\beta - WX\gamma - (\iota_T \otimes \alpha).

    The score and variance follow the panel LM-Error construction
    (cf. :func:`bayesian_panel_lm_error_test`):

    .. math::
        S = \mathbf{e}^\top (I_T \otimes W)\mathbf{e},
        \qquad V = T \cdot T_{WW}\,\bar{\sigma}^2,

    with :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`. The LM statistic is
    :math:`\chi^2_1` under H₀. Tests whether an SDM panel should be
    extended to a SDARAR panel (SDM with spatial-error autocorrelation);
    this is the panel analogue of the SDM-aware diagnostic discussed in
    :cite:t:`koley2024UseNot`. Panel-data extensions follow
    :cite:t:`anselin2008SpatialPanel` and :cite:t:`elhorst2014SpatialEconometrics`.
    The Bayesian LM statistic is computed per posterior draw following
    :cite:t:`dogan2021BayesianRobust`.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    rho_draws = _get_posterior_draws(idata, "rho").reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    Z = np.hstack([X, WX])
    fitted = beta_draws @ Z.T
    resid = y[None, :] - fitted - rho_draws[:, None] * Wy[None, :]
    resid = _maybe_subtract_alpha(model, idata, resid)

    We = _panel_spatial_lag(W_sp, resid, N, T)
    S = np.sum(resid * We, axis=1)

    T_ww = model._T_ww
    sigma2_mean = float(np.mean(sigma_draws**2))
    # Panel LM-Error denominator scales with T (cf. bayesian_panel_lm_error_test)
    V = T * T_ww * sigma2_mean
    LM = S**2 / (V + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_error_sdm",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_lm_lag_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Panel LM-Lag test from an SDEM panel posterior (H₀: ρ = 0 | SDEM).

    Panel analogue of :func:`bayesian_lm_lag_sdem_test`. Residuals are the
    spatially-filtered SDEM panel residuals (with RE alpha removed if
    present):

    .. math::
        \mathbf{u} = \mathbf{y} - X\beta - WX\gamma - (\iota_T\otimes\alpha),
        \qquad \boldsymbol{\varepsilon} = \mathbf{u} - \lambda\,(I_T\otimes W)\mathbf{u}.

    The score and variance follow the panel LM-Lag construction
    (cf. :func:`bayesian_panel_lm_lag_test`):

    .. math::
        S = \boldsymbol{\varepsilon}^\top (I_T\otimes W)\mathbf{y},
        \qquad V = T_{WW}\,\bar{\sigma}^2 + \|(I_T\otimes W)\mathbf{y}\|^2.

    The LM statistic is :math:`\chi^2_1` under H₀. Tests whether an SDEM
    panel should be extended to a SDARAR panel (SDEM with a spatial lag);
    this is the panel analogue of the SDEM-aware diagnostic discussed in
    :cite:t:`koley2024UseNot`. Panel-data extensions follow
    :cite:t:`anselin2008SpatialPanel` and :cite:t:`elhorst2014SpatialEconometrics`.
    The Bayesian LM statistic is computed per posterior draw following
    :cite:t:`dogan2021BayesianRobust`.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    lam_draws = _get_posterior_draws(idata, "lam").reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    Z = np.hstack([X, WX])
    u = y[None, :] - beta_draws @ Z.T
    u = _maybe_subtract_alpha(model, idata, u)
    Wu = _panel_spatial_lag(W_sp, u, N, T)
    eps = u - lam_draws[:, None] * Wu

    S = np.dot(eps, Wy)

    T_ww = model._T_ww
    sigma2_mean = float(np.mean(sigma_draws**2))
    V = T * T_ww * sigma2_mean + float(np.dot(Wy, Wy))
    LM = S**2 / (V + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_lag_sdem",
        df=1,
        details={"N": N, "T": T},
    )
