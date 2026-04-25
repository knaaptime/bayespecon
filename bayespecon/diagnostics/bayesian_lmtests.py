"""
Bayesian LM-type diagnostics for spatial models :cite:p:`dogan2021BayesianRobust`.

Implements Bayesian LM tests for omitted spatial lag (SAR) and error (SEM)
models, as well as SDM/SDEM variant tests (WX, joint, and robust),
following the formulas in :cite:t:`dogan2021BayesianRobust` and :cite:t:`koley2024UseNot`.

Panel variants follow :cite:t:`anselin2008SpatialPanel` and :cite:t:`elhorst2014SpatialEconometrics` for the
information-matrix adjustment (T multiplier, Wb'MWb term).
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import arviz as az
import numpy as np
from scipy import stats as sp_stats


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
    beta_name: str = "beta",
    test_type: str = "lag",
) -> BayesianLMTestResult:
    """Bayesian LM test for omitted spatial lag (SAR) model.

    Follows :cite:t:`dogan2021BayesianRobust`. Implements the Bayesian LM test for
    omitted spatial lag (SAR) effects.

    The test statistic for each posterior draw is:

    .. math::
        S = \\mathbf{e}^\\top W \\mathbf{y}

    where :math:`\\mathbf{e}` are residuals, :math:`W` is the spatial
    weights matrix, and :math:`\\mathbf{y}` is the outcome vector.

    The variance is:

    .. math::
        V = \\mathbb{E}\\left[ \\sum_{i} (e_i W y_i)^2 \\right]

    The LM statistic is:

    .. math::
        \\mathrm{LM} = \\frac{S^2}{V}

    The Bayesian p-value is computed at the posterior mean of the LM
    statistic: ``1 - chi2.cdf(mean, df)`` (:cite:p:`dogan2021BayesianRobust`).

    Parameters
    ----------
    model : SpatialModel
        Fitted spatial model instance with ``inference_data`` attribute.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.
    test_type : str, default "lag"
        Test type label (for result metadata).

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
    # Only include WX in X if the model's beta covers WX columns
    # (e.g., SLX, SDM). For OLS, beta only covers X.
    k_beta = beta_draws.shape[1]
    if hasattr(model, "_WX") and model._WX.shape[1] > 0 and k_beta == model._X.shape[1] + model._WX.shape[1]:
        X = np.hstack([model._X, model._WX])
    else:
        X = model._X
    # Wy is pre-computed and stored as a dense array — no need to materialize W
    Wy = model._Wy
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)
    # Score for each draw: S = resid @ Wy
    S = np.dot(resid, Wy)  # (draws,)
    # Variance: V = mean(resid^2 @ Wy^2)
    V = np.mean(np.sum(resid**2 * Wy**2, axis=1))
    LM = S**2 / (V + 1e-12)
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, 1))
    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_lm_lag",
        df=1,
        details={"n_draws": LM.shape[0]},
    )


def bayesian_lm_error_test(
    model,
    beta_name: str = "beta",
    test_type: str = "error",
) -> BayesianLMTestResult:
    """Bayesian LM test for omitted spatial error (SEM) model.

    Follows :cite:t:`dogan2021BayesianRobust`. Implements the Bayesian LM test for
    omitted spatial error (SEM) effects.

    For each posterior draw, the test statistic is:

    .. math::
        S = \\sum_{i} e_i (W \\mathbf{e})_i

    where :math:`\\mathbf{e}` are residuals and :math:`W` is the spatial
    weights matrix.

    The variance is:

    .. math::
        V = \\mathbb{E}\\left[ \\sum_{i} (e_i (W \\mathbf{e})_i)^2 \\right]

    The LM statistic is:

    .. math::
        \\mathrm{LM} = \\frac{S^2}{V}

    The Bayesian p-value is computed at the posterior mean of the LM
    statistic: ``1 - chi2.cdf(mean, df)`` (:cite:p:`dogan2021BayesianRobust`).

    Parameters
    ----------
    model : SpatialModel
        Fitted spatial model instance with ``inference_data`` attribute.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.
    test_type : str, default "error"
        Test type label (for result metadata).

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    # Use sparse matrix for We = W @ resid (avoids dense materialisation)
    W_sp = model._W_sparse
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
    # Only include WX in X if the model's beta covers WX columns
    k_beta = beta_draws.shape[1]
    if hasattr(model, "_WX") and model._WX.shape[1] > 0 and k_beta == model._X.shape[1] + model._WX.shape[1]:
        X = np.hstack([model._X, model._WX])
    else:
        X = model._X
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)
    # sparse matmul: (draws, n) @ (n, n)^T = (draws, n)
    # We = W @ resid.T = (n, draws) then transpose to (draws, n)
    We = (W_sp @ resid.T).T  # (draws, n)
    # Score for each draw: S = resid * We (element-wise, then sum)
    S = np.sum(resid * We, axis=1)  # (draws,)
    # Variance: V = mean(sum((resid * We)^2))
    V = np.mean(np.sum((resid * We) ** 2, axis=1))
    LM = S**2 / (V + 1e-12)
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, 1))
    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_lm_error",
        df=1,
        details={"n_draws": LM.shape[0]},
    )


def bayesian_lm_wx_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian LM test for WX coefficients (H₀: γ = 0).

    Tests whether spatially lagged covariates (WX) should be added to a
    SAR model, i.e., whether the SAR model should be extended to an SDM
    specification. Follows the multi-parameter Bayesian LM test framework
    of :cite:t:`dogan2021BayesianRobust` and the classical LM-WX test of :cite:t:`koley2024UseNot`.

    The null model is SAR (includes ρ but not γ). For each posterior draw
    from the SAR model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - \\rho W \\mathbf{y} - X \\beta

    The score vector for the WX coefficients is:

    .. math::
        \\mathbf{g}_\\gamma = (WX)^\\top \\mathbf{e}

    a :math:`k_{wx} \\times 1` vector for each draw. The concentration
    matrix is the information matrix evaluated at the posterior mean:

    .. math::
        J_{\\gamma\\gamma} = \\frac{1}{\\bar{\\sigma}^2} (WX)^\\top (WX)

    where :math:`\\bar{\\sigma}^2` is the posterior mean of :math:`\\sigma^2`.
    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\mathbf{g}_\\gamma^\\top J_{\\gamma\\gamma}^{-1} \\mathbf{g}_\\gamma

    which is distributed as :math:`\\chi^2_{k_{wx}}` under H₀.

    Parameters
    ----------
    model : SAR
        Fitted SAR model instance with ``inference_data`` attribute
        containing posterior draws for ``beta``, ``rho``, and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`k_{wx}` (number of WX columns).

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
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)

    # Residuals: e = y - rho*Wy - X@beta
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Score: g_gamma = WX' @ e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Information matrix: J = (1/sigma2_mean) * WX'WX
    # This is the expected outer product of the score under H0,
    # evaluated at the posterior mean of sigma2.
    # Using E[gg'] would make E[LM] = df by construction (tr(V^{-1}V) = p),
    # so we use the information matrix instead (Dogan et al. 2021, Proposition 1).
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    J_gamma_gamma = (WX.T @ WX) / sigma2_mean  # (k_wx, k_wx)

    # LM = g' J^{-1} g for each draw
    J_inv = np.linalg.inv(J_gamma_gamma + 1e-12 * np.eye(k_wx))
    LM = np.array([g_gamma[g] @ J_inv @ g_gamma[g] for g in range(g_gamma.shape[0])])

    df = k_wx
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_lm_wx",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx},
    )


def bayesian_lm_sdm_joint_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian joint LM test for SDM (H₀: ρ = 0 AND γ = 0).

    Tests the joint null hypothesis that both the spatial lag coefficient
    and the WX coefficients are zero, i.e., whether the OLS model should
    be extended to an SDM specification. Follows the multi-parameter
    Bayesian LM test framework of :cite:t:`dogan2021BayesianRobust` and the classical
    joint LM-SDM test of :cite:t:`koley2024UseNot`.

    The null model is OLS. For each posterior draw from the OLS model,
    residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X \\beta

    The joint score vector is:

    .. math::
        \\mathbf{g} = \\begin{pmatrix} \\mathbf{e}^\\top W \\mathbf{y} \\\\
        (WX)^\\top \\mathbf{e} \\end{pmatrix}

    a :math:`(1 + k_{wx}) \\times 1` vector for each draw. The concentration
    matrix is the information matrix evaluated at the posterior mean:

    .. math::
        J = \\frac{1}{\\bar{\\sigma}^2}
        \\begin{pmatrix}
        (W\\mathbf{y})^\\top (W\\mathbf{y}) & (W\\mathbf{y})^\\top (WX) \\\\
        (WX)^\\top (W\\mathbf{y}) & (WX)^\\top (WX)
        \\end{pmatrix}

    where :math:`\\bar{\\sigma}^2` is the posterior mean of :math:`\\sigma^2`.
    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\mathbf{g}^\\top J^{-1} \\mathbf{g}

    which is distributed as :math:`\\chi^2_{1 + k_{wx}}` under H₀.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS model instance with ``inference_data`` attribute
        containing posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.

    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)

    # Residuals: e = y - X@beta
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Score components
    # g_rho = e'Wy for each draw → (draws,)
    g_rho = np.dot(resid, Wy)  # (draws,)
    # g_gamma = WX'e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Joint score: g = [g_rho, g_gamma']' → (draws, 1+k_wx)
    g = np.column_stack([g_rho, g_gamma])  # (draws, 1+k_wx)

    # Information matrix: J = (1/sigma2_mean) * [Wy'Wy, Wy'WX; WX'Wy, WX'WX]
    # This is the expected outer product of the score under H0,
    # evaluated at the posterior mean of sigma2.
    # Using E[gg'] would make E[LM] = df by construction (tr(V^{-1}V) = p),
    # so we use the information matrix instead (Dogan et al. 2021, Proposition 1).
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    p = 1 + k_wx
    J = np.zeros((p, p))
    J[0, 0] = Wy @ Wy / sigma2_mean
    J[0, 1:] = (Wy @ WX) / sigma2_mean
    J[1:, 0] = (WX.T @ Wy) / sigma2_mean
    J[1:, 1:] = (WX.T @ WX) / sigma2_mean

    # LM = g' J^{-1} g for each draw
    J_inv = np.linalg.inv(J + 1e-12 * np.eye(p))
    LM = np.array([g[d] @ J_inv @ g[d] for d in range(g.shape[0])])

    df = p
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_lm_sdm_joint",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx},
    )


def bayesian_lm_slx_error_joint_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian joint LM test for SDEM (H₀: λ = 0 AND γ = 0).

    Tests the joint null hypothesis that both the spatial error coefficient
    and the WX coefficients are zero, i.e., whether the OLS model should
    be extended to an SDEM specification. Follows the multi-parameter
    Bayesian LM test framework of :cite:t:`dogan2021BayesianRobust` and the classical
    joint LM-SLX-Error test of :cite:t:`koley2024UseNot`.

    The null model is OLS. For each posterior draw from the OLS model,
    residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X \\beta

    The joint score vector is:

    .. math::
        \\mathbf{g} = \\begin{pmatrix} \\mathbf{e}^\\top W \\mathbf{e} \\\\
        (WX)^\\top \\mathbf{e} \\end{pmatrix}

    a :math:`(1 + k_{wx}) \\times 1` vector for each draw. The concentration
    matrix is the information matrix evaluated at the posterior mean:

    .. math::
        J = \\begin{pmatrix}
        \\mathrm{tr}(W'W + W^2) & 0 \\\\
        0 & \\frac{1}{\\bar{\\sigma}^2} (WX)^\\top (WX)
        \\end{pmatrix}

    where :math:`\\bar{\\sigma}^2` is the posterior mean of :math:`\\sigma^2`.
    The off-diagonal blocks are zero under H₀ (spherical errors).
    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\mathbf{g}^\\top J^{-1} \\mathbf{g}

    which is distributed as :math:`\\chi^2_{1 + k_{wx}}` under H₀.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS model instance with ``inference_data`` attribute
        containing posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.

    """
    y = model._y
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)

    # Residuals: e = y - X@beta
    fitted = beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Score components
    # g_lambda = e'We for each draw → (draws,)
    We = (W_sp @ resid.T).T  # (draws, n)
    g_lambda = np.sum(resid * We, axis=1)  # (draws,)
    # g_gamma = WX'e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Joint score: g = [g_lambda, g_gamma']' → (draws, 1+k_wx)
    g = np.column_stack([g_lambda, g_gamma])  # (draws, 1+k_wx)

    # Information matrix under H0, evaluated at posterior mean of sigma2.
    # J = (1/sigma2_mean) * [tr(W'W+W^2)*sigma2,  0;  0,  WX'WX]
    # The off-diagonal blocks are zero under H0 (spherical errors).
    # Using E[gg'] would make E[LM] = df by construction (tr(V^{-1}V) = p),
    # so we use the information matrix instead (Dogan et al. 2021, Proposition 1).
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)
    sigma2_mean = float(np.mean(sigma_draws**2))
    WtW = W_sp.T @ W_sp
    WW = W_sp @ W_sp
    T_ww = float(WtW.diagonal().sum() + WW.diagonal().sum())
    p = 1 + k_wx
    J = np.zeros((p, p))
    J[0, 0] = T_ww  # J_{lambda,lambda} = tr(W'W + W^2)
    # Off-diagonal blocks are zero under H0 (spherical errors)
    J[1:, 1:] = (WX.T @ WX) / sigma2_mean

    # LM = g' J^{-1} g for each draw
    J_inv = np.linalg.inv(J + 1e-12 * np.eye(p))
    LM = np.array([g[d] @ J_inv @ g[d] for d in range(g.shape[0])])

    df = p
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_lm_slx_error_joint",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx},
    )


def summarize_bayesian_lm_test(result: BayesianLMTestResult) -> None:
    """Print a summary of the Bayesian LM test result.

    Parameters
    ----------
    result : BayesianLMTestResult
        Result object from a Bayesian LM test function.
    """
    print(f"Bayesian LM Test ({result.test_type}):")
    print(f"  Degrees of freedom: {result.df}")
    print(f"  Mean: {result.mean:.3f}")
    print(f"  Median: {result.median:.3f}")
    print(
        f"  95% Credible Interval: [{result.credible_interval[0]:.3f}, {result.credible_interval[1]:.3f}]"
    )
    print(f"  Bayesian p-value: {result.bayes_pvalue:.3f}")


# ---------------------------------------------------------------------------
# Information matrix blocks for Neyman orthogonal score adjustment
# ---------------------------------------------------------------------------


def _info_matrix_blocks_sdm(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    sigma2: float,
    Wy_hat: np.ndarray | None = None,
) -> dict:
    """Compute partitioned information matrix blocks for SDM specification.

    Computes the blocks of the information matrix :math:`J` needed for the
    Neyman orthogonal score adjustment in the SDM context, following
    :cite:t:`koley2024UseNot` and :cite:t:`anselin1996SimpleDiagnostic`.

    The parameter vector is :math:`\\theta = (\\beta', \\sigma^2, \\rho, \\gamma)'`
    where :math:`\\gamma` are the WX coefficients. The relevant blocks
    (partitioned on :math:`\\sigma^2`) are:

    .. math::
        J_{\\rho\\rho \\cdot \\sigma} &= n + \\mathrm{tr}(W'W + W^2) \\\\
        J_{\\rho\\gamma \\cdot \\sigma} &= (W\\hat{y})' (WX) / \\sigma^2 \\\\
        J_{\\gamma\\gamma \\cdot \\sigma} &= (WX)' (WX) / \\sigma^2

    where :math:`\\hat{y} = X\\bar{\\beta}` is the fitted values under H₀
    (posterior mean), and :math:`W\\hat{y}` is the spatially lagged fitted values.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` including intercept.
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``.
    sigma2 : float
        Error variance estimate (posterior mean of sigma^2).
    Wy_hat : np.ndarray or None, optional
        Spatially lagged fitted values under H₀, shape ``(n,)``.
        Computed as ``W @ X @ beta_mean`` for OLS null or
        ``W @ y_hat`` for SAR/SLX null models.
        If None, the cross-term ``J_rho_gamma`` is set to zero
        (which disables the Neyman adjustment).

    Returns
    -------
    dict
        Dictionary with keys ``J_rho_rho``, ``J_rho_gamma``, ``J_gamma_gamma``,
        ``T_ww`` (trace of W'W + W²).
    """
    n = X.shape[0]
    k_wx = WX.shape[1]

    # T = tr(W'W + W²) = tr(W'W) + tr(W²)
    WtW = W_sparse.T @ W_sparse
    WW = W_sparse @ W_sparse
    T_ww = float(WtW.diagonal().sum() + WW.diagonal().sum())

    # J_{ρρ·σ} = n + T_ww
    # Under H₀ from OLS: this matches spreg's J_{11} after conditioning on σ²
    J_rho_rho = float(n + T_ww)

    # J_{γγ·σ} = (WX)'(WX) / σ²
    J_gamma_gamma = (WX.T @ WX) / sigma2  # (k_wx, k_wx)

    # J_{ργ·σ}: cross term between ρ and γ scores
    # Following spreg's lm_spdurbin: J_{12} includes (WXb)'(WX) where b = β̂
    # For the Bayesian version: J_{ργ·σ} = (Wŷ)'(WX) / σ²
    # where ŷ = Xβ̄ (posterior mean fitted values under H₀)
    # This is non-zero in general and is essential for the Neyman adjustment
    # in the robust LM-Lag-SDM and robust LM-WX tests.
    if Wy_hat is not None:
        J_rho_gamma = (Wy_hat @ WX) / sigma2  # (k_wx,)
    else:
        J_rho_gamma = np.zeros(k_wx)  # (k_wx,) — no adjustment

    return {
        "J_rho_rho": J_rho_rho,
        "J_rho_gamma": J_rho_gamma,
        "J_gamma_gamma": J_gamma_gamma,
        "T_ww": T_ww,
    }


def _info_matrix_blocks_sdem(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    sigma2: float,
) -> dict:
    """Compute partitioned information matrix blocks for SDEM specification.

    Computes the blocks of the information matrix needed for the Neyman
    orthogonal score adjustment in the SDEM context, following
    :cite:t:`koley2024UseNot`.

    For the SDEM model, the testing parameter is :math:`\\lambda` (spatial
    error) and the nuisance parameter is :math:`\\gamma` (WX coefficients).

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` including intercept.
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``.
    sigma2 : float
        Error variance estimate (posterior mean of sigma^2).

    Returns
    -------
    dict
        Dictionary with keys ``J_lam_lam``, ``J_lam_gamma``, ``J_gamma_gamma``,
        ``T_ww``.
    """
    # T = tr(W'W + W²)
    WtW = W_sparse.T @ W_sparse
    WW = W_sparse @ W_sparse
    T_ww = float(WtW.diagonal().sum() + WW.diagonal().sum())

    # J_{λλ·σ} = T_ww  (for error model under H₀: λ=0)
    J_lam_lam = T_ww

    # J_{λγ·σ}: cross term between λ and γ
    # Under H₀: g_λ = e'We/σ², g_γ = e'WX/σ²
    # For mean-zero normal errors, E[e'We · e'WX] = 0 (odd moments vanish),
    # so J_{λγ·σ} = 0. This matches spreg's lm_slxerr which simply adds
    # LM_Error + LM_WX (assuming independence of λ and γ scores).
    k_wx = WX.shape[1]
    J_lam_gamma = np.zeros(k_wx)

    # J_{γγ·σ} = (WX)'(WX) / σ²
    J_gamma_gamma = (WX.T @ WX) / sigma2

    return {
        "J_lam_lam": J_lam_lam,
        "J_lam_gamma": J_lam_gamma,
        "J_gamma_gamma": J_gamma_gamma,
        "T_ww": T_ww,
    }


# ---------------------------------------------------------------------------
# Robust Bayesian LM tests (Neyman orthogonal score, Dogan et al. 2021)
# ---------------------------------------------------------------------------


def bayesian_robust_lm_lag_sdm_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian robust LM-Lag test in SDM context (H₀: ρ = 0, robust to γ).

    Tests the null hypothesis that the spatial lag coefficient is zero,
    robust to the local presence of WX effects (γ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3,
    which is the Bayesian analogue of the robust LM-Lag-SDM test in
    :cite:t:`koley2024UseNot`.

    The alternative model is SLX (includes γ but not ρ). For each
    posterior draw from the SLX model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X\\beta_1 - WX\\beta_2

    The unadjusted scores are:

    .. math::
        g_\\rho &= \\mathbf{e}^\\top W \\mathbf{y} \\\\
        \\boldsymbol{g}_\\gamma &= (WX)^\\top \\mathbf{e}

    The Neyman-adjusted score for ρ is:

    .. math::
        g_\\rho^* = g_\\rho - J_{\\rho\\gamma \\cdot \\sigma}
        J_{\\gamma\\gamma \\cdot \\sigma}^{-1} \\boldsymbol{g}_\\gamma

    The adjusted variance is:

    .. math::
        V^* = J_{\\rho\\rho \\cdot \\sigma} - J_{\\rho\\gamma \\cdot \\sigma}
        J_{\\gamma\\gamma \\cdot \\sigma}^{-1} J_{\\gamma\\rho \\cdot \\sigma}

    The robust LM statistic for each draw is:

    .. math::
        \\mathrm{LM}_R = \\frac{(g_\\rho^*)^2}{V^*}

    which is distributed as :math:`\\chi^2_1` under H₀, irrespective of
    local misspecification in γ (:cite:p:`dogan2021BayesianRobust`, Proposition 3).

    Parameters
    ----------
    model : SLX
        Fitted SLX model instance with ``inference_data`` attribute
        containing posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to 1.

    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k_total)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals: e = y - [X, WX] @ beta
    Z = np.hstack([X, WX])  # (n, k+k_wx)
    fitted = beta_draws @ Z.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Unadjusted scores
    # g_rho = e'Wy for each draw → (draws,)
    g_rho = np.dot(resid, Wy)  # (draws,)
    # g_gamma = WX'e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Compute Wy_hat = W @ y_hat for the information matrix cross-term
    # Under H₀ (ρ=0) from SLX model: y_hat = X @ beta1_mean + WX @ beta2_mean
    # Wy_hat = W @ y_hat is needed for J_{ργ·σ} = (Wy_hat)'(WX) / σ²
    beta_mean = np.mean(beta_draws, axis=0)  # (k_total,)
    y_hat = Z @ beta_mean  # (n,) — SLX fitted values at posterior mean
    Wy_hat = np.asarray(W_sp @ y_hat).ravel()  # (n,)

    # Information matrix blocks (evaluated at posterior mean of sigma²)
    sigma2_mean = float(np.mean(sigma_draws**2))
    info = _info_matrix_blocks_sdm(X, WX, W_sp, sigma2_mean, Wy_hat=Wy_hat)

    J_rho_rho = info["J_rho_rho"]
    J_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    J_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # Neyman adjustment: g_rho* = g_rho - J_{ργ·σ} J_{γγ·σ}^{-1} g_gamma
    if k_wx > 0:
        J_gamma_gamma_inv = np.linalg.inv(
            J_gamma_gamma + 1e-12 * np.eye(k_wx)
        )
        neyman_coef = J_rho_gamma @ J_gamma_gamma_inv  # (k_wx,)
        adjustment = g_gamma @ neyman_coef  # (draws,)
        g_rho_star = g_rho - adjustment

        # Adjusted variance: V* = J_{ρρ·σ} - J_{ργ·σ} J_{γγ·σ}^{-1} J_{γρ·σ}
        V_star = J_rho_rho - neyman_coef @ J_rho_gamma
    else:
        # No WX columns: no adjustment needed
        g_rho_star = g_rho
        V_star = J_rho_rho

    # Robust LM = (g_rho*)² / V*
    LM = g_rho_star**2 / (V_star + 1e-12)

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_robust_lm_lag_sdm",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx},
    )


def bayesian_robust_lm_wx_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian robust LM-WX test (H₀: γ = 0, robust to ρ).

    Tests the null hypothesis that the WX coefficients are zero,
    robust to the local presence of a spatial lag (ρ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3,
    which is the Bayesian analogue of the robust LM-WX test in
    :cite:t:`koley2024UseNot`.

    The alternative model is SAR (includes ρ but not γ). For each
    posterior draw from the SAR model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - \\rho W \\mathbf{y} - X\\beta

    The unadjusted scores are:

    .. math::
        g_\\rho &= \\mathbf{e}^\\top W \\mathbf{y} \\\\
        \\boldsymbol{g}_\\gamma &= (WX)^\\top \\mathbf{e}

    The Neyman-adjusted score for γ is:

    .. math::
        \\boldsymbol{g}_\\gamma^* = \\boldsymbol{g}_\\gamma -
        J_{\\gamma\\rho \\cdot \\sigma} J_{\\rho\\rho \\cdot \\sigma}^{-1} g_\\rho

    The adjusted weight matrix is:

    .. math::
        C_{\\gamma\\gamma}^* = P_{\\gamma\\gamma} J_{\\gamma \\cdot \\rho}

    where :math:`P_{\\gamma\\gamma} = I - J_{\\gamma\\rho \\cdot \\sigma}
    J_{\\rho\\rho \\cdot \\sigma}^{-1} J_{\\rho\\gamma \\cdot \\sigma}
    J_{\\gamma \\cdot \\rho}^{-1}` and :math:`J_{\\gamma \\cdot \\rho} =
    J_{\\gamma\\gamma \\cdot \\sigma} - J_{\\gamma\\rho \\cdot \\sigma}
    J_{\\rho\\rho \\cdot \\sigma}^{-1} J_{\\rho\\gamma \\cdot \\sigma}`.

    The robust LM statistic for each draw is:

    .. math::
        \\mathrm{LM}_R = (\\boldsymbol{g}_\\gamma^*)^\\top
        (C_{\\gamma\\gamma}^*)^{-1} \\boldsymbol{g}_\\gamma^*

    which is distributed as :math:`\\chi^2_{k_{wx}}` under H₀.

    Parameters
    ----------
    model : SAR
        Fitted SAR model instance with ``inference_data`` attribute
        containing posterior draws for ``beta``, ``rho``, and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

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
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The robust LM-WX test requires "
            "at least one spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SAR residuals: e = y - rho*Wy - X@beta
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Unadjusted scores
    # g_rho = e'Wy for each draw → (draws,)
    g_rho = np.dot(resid, Wy)  # (draws,)
    # g_gamma = WX'e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Compute Wy_hat = W @ y_hat for the information matrix cross-term
    # Under H₀ (γ=0) from SAR model: y_hat = rho_mean * Wy + X @ beta_mean
    # Wy_hat = W @ y_hat is needed for J_{ργ·σ} = (Wy_hat)'(WX) / σ²
    beta_mean = np.mean(beta_draws, axis=0)  # (k,)
    rho_mean = float(np.mean(rho_draws))
    y_hat = rho_mean * Wy + X @ beta_mean  # (n,) — SAR fitted values at posterior mean
    Wy_hat = np.asarray(W_sp @ y_hat).ravel()  # (n,)

    # Information matrix blocks (evaluated at posterior mean of sigma²)
    sigma2_mean = float(np.mean(sigma_draws**2))
    info = _info_matrix_blocks_sdm(X, WX, W_sp, sigma2_mean, Wy_hat=Wy_hat)

    J_rho_rho = info["J_rho_rho"]
    J_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    J_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # J_{γ·ρ} = J_{γγ·σ} - J_{γρ·σ} J_{ρρ·σ}^{-1} J_{ργ·σ}
    # J_rho_gamma is (k_wx,), so outer product gives (k_wx, k_wx)
    J_gamma_given_rho = J_gamma_gamma - np.outer(J_rho_gamma, J_rho_gamma) / (J_rho_rho + 1e-12)

    # Neyman adjustment: g_gamma* = g_gamma - J_{γρ·σ} J_{ρρ·σ}^{-1} g_rho
    neyman_coef = J_rho_gamma / (J_rho_rho + 1e-12)  # (k_wx,)
    g_gamma_star = g_gamma - np.outer(g_rho, neyman_coef)  # (draws, k_wx)

    # Adjusted weight matrix: C*_{γγ} = P_{γγ} J_{γ·ρ}
    # P_{γγ} = I - J_{γρ·σ} J_{ρρ·σ}^{-1} J_{ργ·σ} J_{γ·ρ}^{-1}
    J_gamma_given_rho_inv = np.linalg.inv(
        J_gamma_given_rho + 1e-12 * np.eye(k_wx)
    )
    P_gamma = np.eye(k_wx) - np.outer(J_rho_gamma, J_rho_gamma) / (
        (J_rho_rho + 1e-12) * (J_gamma_given_rho + 1e-12 * np.eye(k_wx))
    ).trace()  # Simplified: use the full formula
    # More precisely: P = I - (J_{γρ}/J_{ρρ}) (J_{ργ}/J_{γ·ρ})
    # which is a (k_wx, k_wx) matrix
    P_gamma = np.eye(k_wx) - (
        np.outer(J_rho_gamma, J_rho_gamma) / (J_rho_rho + 1e-12)
    ) @ J_gamma_given_rho_inv

    C_star = P_gamma @ J_gamma_given_rho  # (k_wx, k_wx)

    # Robust LM = g_gamma*' C*^{-1} g_gamma* for each draw
    C_star_inv = np.linalg.inv(C_star + 1e-12 * np.eye(k_wx))
    LM = np.array([
        g_gamma_star[d] @ C_star_inv @ g_gamma_star[d]
        for d in range(g_gamma_star.shape[0])
    ])

    df = k_wx
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_robust_lm_wx",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx},
    )


def bayesian_robust_lm_error_sdem_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian robust LM-Error test in SDEM context (H₀: λ = 0, robust to γ).

    Tests the null hypothesis that the spatial error coefficient is zero,
    robust to the local presence of WX effects (γ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3,
    which is the Bayesian analogue of the robust LM-Error test in the
    SDEM context.

    The alternative model is SLX (includes γ but not λ). For each
    posterior draw from the SLX model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X\\beta_1 - WX\\beta_2

    The unadjusted scores are:

    .. math::
        g_\\lambda &= \\mathbf{e}^\\top W \\mathbf{e} \\\\
        \\boldsymbol{g}_\\gamma &= (WX)^\\top \\mathbf{e}

    The Neyman-adjusted score for λ is:

    .. math::
        g_\\lambda^* = g_\\lambda - J_{\\lambda\\gamma \\cdot \\sigma}
        J_{\\gamma\\gamma \\cdot \\sigma}^{-1} \\boldsymbol{g}_\\gamma

    The adjusted variance is:

    .. math::
        V^* = J_{\\lambda\\lambda \\cdot \\sigma} - J_{\\lambda\\gamma \\cdot \\sigma}
        J_{\\gamma\\gamma \\cdot \\sigma}^{-1} J_{\\gamma\\lambda \\cdot \\sigma}

    The robust LM statistic for each draw is:

    .. math::
        \\mathrm{LM}_R = \\frac{(g_\\lambda^*)^2}{V^*}

    which is distributed as :math:`\\chi^2_1` under H₀, irrespective of
    local misspecification in γ.

    Parameters
    ----------
    model : SLX
        Fitted SLX model instance with ``inference_data`` attribute
        containing posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to 1.

    """
    y = model._y
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k_total)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals: e = y - [X, WX] @ beta
    Z = np.hstack([X, WX])  # (n, k+k_wx)
    fitted = beta_draws @ Z.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Unadjusted scores
    # g_lambda = e'We for each draw → (draws,)
    We = (W_sp @ resid.T).T  # (draws, n)
    g_lambda = np.sum(resid * We, axis=1)  # (draws,)
    # g_gamma = WX'e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Information matrix blocks (evaluated at posterior mean of sigma²)
    sigma2_mean = float(np.mean(sigma_draws**2))
    info = _info_matrix_blocks_sdem(X, WX, W_sp, sigma2_mean)

    J_lam_lam = info["J_lam_lam"]
    J_lam_gamma = info["J_lam_gamma"]  # (k_wx,)
    J_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # Neyman adjustment: g_lambda* = g_lambda - J_{λγ·σ} J_{γγ·σ}^{-1} g_gamma
    if k_wx > 0:
        J_gamma_gamma_inv = np.linalg.inv(
            J_gamma_gamma + 1e-12 * np.eye(k_wx)
        )
        neyman_coef = J_lam_gamma @ J_gamma_gamma_inv  # (k_wx,)
        adjustment = g_gamma @ neyman_coef  # (draws,)
        g_lambda_star = g_lambda - adjustment

        # Adjusted variance: V* = J_{λλ·σ} - J_{λγ·σ} J_{γγ·σ}^{-1} J_{γλ·σ}
        V_star = J_lam_lam - neyman_coef @ J_lam_gamma
    else:
        g_lambda_star = g_lambda
        V_star = J_lam_lam

    # Robust LM = (g_lambda*)² / V*
    LM = g_lambda_star**2 / (V_star + 1e-12)

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_robust_lm_error_sdem",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx},
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
    if hasattr(model, "_unit_idx") and "alpha" in model.inference_data.posterior:
        alpha_draws = _get_posterior_draws(model.inference_data, "alpha")  # (draws, N)
        unit_idx = model._unit_idx
        # alpha_draws[:, unit_idx] → (draws, n)
        resid = resid - alpha_draws[:, unit_idx]

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
    WtW = W_sparse.T @ W_sparse
    WW = W_sparse @ W_sparse
    return float(WtW.diagonal().sum() + WW.diagonal().sum())


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
) -> dict:
    """Compute partitioned information matrix blocks for panel models.

    Computes the blocks of the information matrix :math:`J` needed for
    panel LM tests, following :cite:t:`anselin2008SpatialPanel` and :cite:t:`elhorst2014SpatialEconometrics`.

    The key difference from cross-sectional is the T multiplier on the
    trace term and the Wb'MWb term in J_{ρρ}:

    .. math::
        J_{\\rho\\rho \\cdot \\sigma} &= \\frac{1}{\\sigma^2}
        \\left( (W\\hat{y})^\\top M (W\\hat{y}) + T \\cdot \\mathrm{tr}(W'W + W^2)
        \\right) \\\\
        J_{\\lambda\\lambda \\cdot \\sigma} &= T \\cdot \\mathrm{tr}(W'W + W^2) \\\\
        J_{\\rho\\lambda \\cdot \\sigma} &= T \\cdot \\mathrm{tr}(W'W + W^2)

    where :math:`M = I - X(X^\\top X)^{-1} X^\\top` is the annihilator
    matrix and :math:`\\hat{y} = X\\bar{\\beta}`.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` (demeaned for FE models).
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        N×N spatial weights matrix.
    W_eigs : np.ndarray
        Eigenvalues of the N×N W matrix.
    sigma2 : float
        Error variance estimate (posterior mean of sigma^2).
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    y_hat : np.ndarray or None, optional
        Fitted values under H₀, shape ``(n,)``.
        If provided, used to compute Wb'MWb term.
    Wy_hat : np.ndarray or None, optional
        Spatially lagged fitted values, shape ``(n,)``.
        If provided, used directly for Wb'MWb term.
        If None but y_hat is provided, computed as W_nt @ y_hat.

    Returns
    -------
    dict
        Dictionary with keys ``J_rho_rho``, ``J_lam_lam``, ``J_rho_lam``,
        ``J_rho_gamma``, ``J_gamma_gamma``, ``T_ww``, ``T_mult``.
    """
    n = X.shape[0]
    k_wx = WX.shape[1]

    # tr(W'W + W²) from N×N matrix
    T_ww = _panel_trace_WtW_WW(W_sparse)
    T_mult = T  # multiplier from Kronecker structure

    # J_{λλ·σ} = T * tr(W'W + W²)
    J_lam_lam = T_mult * T_ww

    # J_{ρλ·σ} = T * tr(W'W + W²) under H₀ (Elhorst 2014)
    J_rho_lam = T_mult * T_ww

    # J_{ρρ·σ} = (Wb'MWb + T*tr) / σ²
    # where M = I - X(X'X)^{-1}X' is the annihilator matrix
    if y_hat is not None or Wy_hat is not None:
        if Wy_hat is None:
            Wy_hat = _panel_spatial_lag(W_sparse, y_hat, N, T_mult)

        # M = I - X(X'X)^{-1}X' (annihilator matrix)
        XtX_inv = np.linalg.inv(X.T @ X + 1e-12 * np.eye(X.shape[1]))
        M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
        WbMWb = float(Wy_hat @ M_Wy)
        J_rho_rho = (WbMWb + T_mult * T_ww) / sigma2
    else:
        # Fallback: use n + T*tr (simplified, no Wb'MWb term)
        J_rho_rho = (n + T_mult * T_ww) / sigma2

    # J_{γγ·σ} = (WX)'(WX) / σ² (same as cross-sectional)
    J_gamma_gamma = (WX.T @ WX) / sigma2  # (k_wx, k_wx)

    # J_{ργ·σ}: cross term between ρ and γ
    # = (Wy_hat)'(WX) / σ² when Wy_hat is available
    if Wy_hat is not None and k_wx > 0:
        J_rho_gamma = (Wy_hat @ WX) / sigma2  # (k_wx,)
    else:
        J_rho_gamma = np.zeros(k_wx)  # (k_wx,)

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
    beta_name: str = "beta",
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
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
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
    XtX_inv = np.linalg.inv(X.T @ X + 1e-12 * np.eye(X.shape[1]))
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = _panel_trace_WtW_WW(W_sp)
    J_val = WbMWb + T * T_ww * sigma2_mean

    # LM = S² / (sigma2 * J_val) for each draw
    # But sigma2 varies across draws. Use posterior mean sigma2 for J,
    # and per-draw sigma2 for the score variance.
    # Following the existing cross-sectional pattern: use information matrix
    # evaluated at posterior mean, compute LM per draw.
    sigma2_draws = sigma_draws**2  # (draws,)
    LM = S**2 / (sigma2_draws * J_val + 1e-12)

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_lm_lag",
        df=df,
        details={"n_draws": LM.shape[0], "N": N, "T": T},
    )


def bayesian_panel_lm_error_test(
    model,
    beta_name: str = "beta",
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
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    X = model._X
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score: S = e'W_nt e for each draw
    # Apply N×N W period-by-period to the N*T stacked residuals
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)  # (draws, n)

    S = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Variance: V = sigma^4 * T * tr(W'W + W²)
    T_ww = _panel_trace_WtW_WW(W_sp)
    sigma2_draws = sigma_draws**2  # (draws,)
    V = sigma2_draws**2 * T * T_ww  # (draws,)

    # LM = S² / V
    LM = S**2 / (V + 1e-12)

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_lm_error",
        df=df,
        details={"n_draws": LM.shape[0], "N": N, "T": T},
    )


def bayesian_panel_robust_lm_lag_test(
    model,
    beta_name: str = "beta",
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
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
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
    XtX_inv = np.linalg.inv(X.T @ X + 1e-12 * np.eye(X.shape[1]))
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = _panel_trace_WtW_WW(W_sp)
    J_val = WbMWb + T * T_ww * sigma2_mean

    # Robust LM = (S_lag/σ² - S_err/σ²)² / (J - T*tr)
    # where J is in σ² units and tr = tr(W'W+W²)
    sigma2_draws = sigma_draws**2
    robust_score = (S_lag / sigma2_draws - S_err / sigma2_draws)  # (draws,)
    denom = J_val / sigma2_mean - T * T_ww  # J/σ² - T*tr (scalar)

    LM = robust_score**2 / (abs(denom) + 1e-12)

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_robust_lm_lag",
        df=df,
        details={"n_draws": LM.shape[0], "N": N, "T": T},
    )


def bayesian_panel_robust_lm_error_test(
    model,
    beta_name: str = "beta",
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
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
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
    XtX_inv = np.linalg.inv(X.T @ X + 1e-12 * np.eye(X.shape[1]))
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = _panel_trace_WtW_WW(W_sp)
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

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_robust_lm_error",
        df=df,
        details={"n_draws": LM.shape[0], "N": N, "T": T},
    )


# ---------------------------------------------------------------------------
# Panel Bayesian LM tests — SDM/SDEM variants (Koley & Bera 2024)
# ---------------------------------------------------------------------------


def bayesian_panel_lm_wx_test(
    model,
    beta_name: str = "beta",
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
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

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
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The panel WX test requires at least one "
            "spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SAR residuals: e = y - rho*Wy - X@beta
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # For RE models, also subtract alpha
    if hasattr(model, "_unit_idx") and "alpha" in idata.posterior:
        alpha_draws = _get_posterior_draws(idata, "alpha")  # (draws, N)
        resid = resid - alpha_draws[:, model._unit_idx]

    # Score: g_gamma = WX' @ e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Information matrix: J = (1/sigma2_mean) * WX'WX
    sigma2_mean = float(np.mean(sigma_draws**2))
    J_gamma_gamma = (WX.T @ WX) / sigma2_mean  # (k_wx, k_wx)

    # LM = g' J^{-1} g for each draw
    J_inv = np.linalg.inv(J_gamma_gamma + 1e-12 * np.eye(k_wx))
    LM = np.array([g_gamma[g] @ J_inv @ g_gamma[g] for g in range(g_gamma.shape[0])])

    df = k_wx
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_lm_wx",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_lm_sdm_joint_test(
    model,
    beta_name: str = "beta",
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

    The LM statistic is :math:`\\chi^2_{1 + k_{wx}}` under H₀.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted OLS panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.
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
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
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
        X, WX, W_sp, model._W_eigs, sigma2_mean, N, T, y_hat=y_hat,
    )

    p = 1 + k_wx
    J = np.zeros((p, p))
    J[0, 0] = info["J_rho_rho"]
    if k_wx > 0:
        J[0, 1:] = info["J_rho_gamma"]
        J[1:, 0] = info["J_rho_gamma"]
        J[1:, 1:] = info["J_gamma_gamma"]

    # LM = g' J^{-1} g for each draw
    J_inv = np.linalg.inv(J + 1e-12 * np.eye(p))
    LM = np.array([g[d] @ J_inv @ g[d] for d in range(g.shape[0])])

    df = p
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_lm_sdm_joint",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_lm_slx_error_joint_test(
    model,
    beta_name: str = "beta",
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

    The LM statistic is :math:`\\chi^2_{1 + k_{wx}}` under H₀.

    Parameters
    ----------
    model : SpatialPanelModel
        Fitted OLS panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.
    """
    y = model._y
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
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

    # Information matrix
    sigma2_mean = float(np.mean(sigma_draws**2))
    T_ww = _panel_trace_WtW_WW(W_sp)

    p = 1 + k_wx
    J = np.zeros((p, p))
    J[0, 0] = T * T_ww  # J_{λλ} = T * tr(W'W + W²)
    # Off-diagonal blocks are zero under H₀
    if k_wx > 0:
        J[1:, 1:] = (WX.T @ WX) / sigma2_mean

    # LM = g' J^{-1} g for each draw
    J_inv = np.linalg.inv(J + 1e-12 * np.eye(p))
    LM = np.array([g[d] @ J_inv @ g[d] for d in range(g.shape[0])])

    df = p
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_lm_slx_error_joint",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx, "N": N, "T": T},
    )


# ---------------------------------------------------------------------------
# Panel robust Bayesian LM tests — SDM/SDEM (Neyman orthogonal score)
# ---------------------------------------------------------------------------


def bayesian_panel_robust_lm_lag_sdm_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian panel robust LM-Lag test in SDM context (H₀: ρ = 0, robust to γ).

    Tests the null hypothesis that the spatial lag coefficient is zero,
    robust to the local presence of WX effects (γ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3.

    The alternative model is SLX panel (includes γ but not ρ). For each
    posterior draw from the SLX model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X\\beta_1 - WX\\beta_2

    The Neyman-adjusted score for ρ is:

    .. math::
        g_\\rho^* = g_\\rho - J_{\\rho\\gamma \\cdot \\sigma}
        J_{\\gamma\\gamma \\cdot \\sigma}^{-1} \\boldsymbol{g}_\\gamma

    The robust LM statistic is :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SLXPanelFE or SLX-like panel model
        Fitted SLX panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
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
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k_total)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals: e = y - [X, WX] @ beta
    Z = np.hstack([X, WX])  # (n, k+k_wx)
    fitted = beta_draws @ Z.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    # Unadjusted scores
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Compute Wy_hat for information matrix cross-term
    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = Z @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)

    info = _panel_info_matrix_blocks(
        X, WX, W_sp, model._W_eigs, sigma2_mean, N, T, Wy_hat=Wy_hat,
    )

    J_rho_rho = info["J_rho_rho"]
    J_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    J_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # Neyman adjustment
    if k_wx > 0:
        J_gamma_gamma_inv = np.linalg.inv(
            J_gamma_gamma + 1e-12 * np.eye(k_wx)
        )
        neyman_coef = J_rho_gamma @ J_gamma_gamma_inv  # (k_wx,)
        adjustment = g_gamma @ neyman_coef  # (draws,)
        g_rho_star = g_rho - adjustment

        V_star = J_rho_rho - neyman_coef @ J_rho_gamma
    else:
        g_rho_star = g_rho
        V_star = J_rho_rho

    LM = g_rho_star**2 / (V_star + 1e-12)

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_robust_lm_lag_sdm",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_robust_lm_wx_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian panel robust LM-WX test (H₀: γ = 0, robust to ρ).

    Tests the null hypothesis that the WX coefficients are zero,
    robust to the local presence of a spatial lag (ρ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3.

    The alternative model is SAR panel (includes ρ but not γ). For each
    posterior draw from the SAR model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - \\rho W \\mathbf{y} - X\\beta

    The Neyman-adjusted score for γ is:

    .. math::
        \\boldsymbol{g}_\\gamma^* = \\boldsymbol{g}_\\gamma -
        J_{\\gamma\\rho \\cdot \\sigma} J_{\\rho\\rho \\cdot \\sigma}^{-1} g_\\rho

    The robust LM statistic is :math:`\\chi^2_{k_{wx}}` under H₀.

    Parameters
    ----------
    model : SARPanelFE or SARPanelRE
        Fitted SAR panel model with ``inference_data`` containing
        posterior draws for ``beta``, ``rho``, and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
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
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SAR residuals: e = y - rho*Wy - X@beta
    fitted = rho_draws[:, None] * Wy[None, :] + beta_draws @ X.T
    resid = y[None, :] - fitted

    # For RE models, also subtract alpha
    if hasattr(model, "_unit_idx") and "alpha" in idata.posterior:
        alpha_draws = _get_posterior_draws(idata, "alpha")
        resid = resid - alpha_draws[:, model._unit_idx]

    # Unadjusted scores
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Compute Wy_hat for information matrix
    beta_mean = np.mean(beta_draws, axis=0)
    rho_mean = float(np.mean(rho_draws))
    sigma2_mean = float(np.mean(sigma_draws**2))
    y_hat = rho_mean * Wy + X @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)

    info = _panel_info_matrix_blocks(
        X, WX, W_sp, model._W_eigs, sigma2_mean, N, T, Wy_hat=Wy_hat,
    )

    J_rho_rho = info["J_rho_rho"]
    J_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    J_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # J_{γ·ρ} = J_{γγ·σ} - J_{γρ·σ} J_{ρρ·σ}^{-1} J_{ργ·σ}
    J_gamma_given_rho = J_gamma_gamma - np.outer(J_rho_gamma, J_rho_gamma) / (J_rho_rho + 1e-12)

    # Neyman adjustment: g_gamma* = g_gamma - J_{γρ·σ} J_{ρρ·σ}^{-1} g_rho
    neyman_coef = J_rho_gamma / (J_rho_rho + 1e-12)  # (k_wx,)
    g_gamma_star = g_gamma - np.outer(g_rho, neyman_coef)  # (draws, k_wx)

    # Adjusted weight matrix: C*_{γγ} = P_{γγ} J_{γ·ρ}
    J_gamma_given_rho_inv = np.linalg.inv(
        J_gamma_given_rho + 1e-12 * np.eye(k_wx)
    )
    P_gamma = np.eye(k_wx) - (
        np.outer(J_rho_gamma, J_rho_gamma) / (J_rho_rho + 1e-12)
    ) @ J_gamma_given_rho_inv

    C_star = P_gamma @ J_gamma_given_rho  # (k_wx, k_wx)

    # Robust LM = g_gamma*' C*^{-1} g_gamma* for each draw
    C_star_inv = np.linalg.inv(C_star + 1e-12 * np.eye(k_wx))
    LM = np.array([
        g_gamma_star[d] @ C_star_inv @ g_gamma_star[d]
        for d in range(g_gamma_star.shape[0])
    ])

    df = k_wx
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_robust_lm_wx",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx, "N": N, "T": T},
    )


def bayesian_panel_robust_lm_error_sdem_test(
    model,
    beta_name: str = "beta",
) -> BayesianLMTestResult:
    """Bayesian panel robust LM-Error test in SDEM context (H₀: λ = 0, robust to γ).

    Tests the null hypothesis that the spatial error coefficient is zero,
    robust to the local presence of WX effects (γ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3.

    The alternative model is SLX panel (includes γ but not λ). For each
    posterior draw from the SLX model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X\\beta_1 - WX\\beta_2

    The Neyman-adjusted score for λ is:

    .. math::
        g_\\lambda^* = g_\\lambda - J_{\\lambda\\gamma \\cdot \\sigma}
        J_{\\gamma\\gamma \\cdot \\sigma}^{-1} \\boldsymbol{g}_\\gamma

    Under H₀, :math:`J_{\\lambda\\gamma \\cdot \\sigma} = 0` (odd moments
    vanish for normal errors), so the adjustment is a no-op and the
    robust test equals the non-robust test.

    The robust LM statistic is :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SLXPanelFE or SLX-like panel model
        Fitted SLX panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.
    beta_name : str, default "beta"
        Name of the regression coefficient parameter in the posterior.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
    """
    y = model._y
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, beta_name)  # (draws, k_total)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # SLX residuals: e = y - [X, WX] @ beta
    Z = np.hstack([X, WX])
    fitted = beta_draws @ Z.T
    resid = y[None, :] - fitted

    # Score for error: g_lambda = e'W_nt e
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    g_lambda = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Score for WX: g_gamma = WX'e
    g_gamma = resid @ WX  # (draws, k_wx)

    # Information matrix blocks
    sigma2_mean = float(np.mean(sigma_draws**2))
    T_ww = _panel_trace_WtW_WW(W_sp)

    J_lam_lam = T * T_ww
    J_lam_gamma = np.zeros(k_wx)  # zero under H₀
    J_gamma_gamma = (WX.T @ WX) / sigma2_mean

    # Neyman adjustment (no-op since J_lam_gamma = 0)
    if k_wx > 0:
        J_gamma_gamma_inv = np.linalg.inv(
            J_gamma_gamma + 1e-12 * np.eye(k_wx)
        )
        neyman_coef = J_lam_gamma @ J_gamma_gamma_inv  # zeros
        adjustment = g_gamma @ neyman_coef  # zeros
        g_lambda_star = g_lambda - adjustment

        V_star = J_lam_lam - neyman_coef @ J_lam_gamma
    else:
        g_lambda_star = g_lambda
        V_star = J_lam_lam

    LM = g_lambda_star**2 / (V_star + 1e-12)

    df = 1
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))

    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type="bayesian_panel_robust_lm_error_sdem",
        df=df,
        details={"n_draws": LM.shape[0], "k_wx": k_wx, "N": N, "T": T},
    )
