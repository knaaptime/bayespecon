"""Cross-sectional linear spatial DGP functions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .utils import (
    _hetero_scale,
    ensure_rng,
    make_design_matrix,
    make_output_geodataframe,
    resolve_weights,
)


def _check_rho_stability(rho: float, W: np.ndarray, name: str = "rho") -> None:
    """Warn when ``|rho|`` exceeds the spectral stability bound of W.

    The DGP map :math:`y = (I - \\rho W)^{-1} u` is well-defined iff
    :math:`(I - \\rho W)` is invertible, which requires
    :math:`|\\rho| < 1 / \\max_i |\\omega_i|` where :math:`\\omega_i`
    are the (real-part) eigenvalues of :math:`W`.  For row-standardised
    W this bound is ``1``.  We emit a UserWarning rather than raising
    so that callers running deliberate boundary tests can proceed.
    """
    try:
        eig_max = float(np.max(np.abs(np.linalg.eigvals(W).real)))
    except np.linalg.LinAlgError:
        return
    if eig_max <= 0.0:
        return
    bound = 1.0 / eig_max
    if abs(rho) >= bound:
        warnings.warn(
            f"{name}={rho:g} is outside the stability domain "
            f"|{name}| < {bound:g} (1/max|eig(W)|); the simulated draw "
            "may be numerically singular or unbounded.",
            stacklevel=3,
        )


def _attach_optional_gdf(
    out: dict,
    *,
    source_gdf: Any | None,
    create_gdf: bool,
    geometry_type: str,
):
    if not create_gdf and source_gdf is None:
        return out
    return make_output_geodataframe(
        y=out["y"],
        X=out["X"],
        gdf=source_gdf,
        geometry_type=geometry_type,
    )


def _simulate_sdm_core(
    *,
    nobs: int,
    Wd: np.ndarray | None,
    X: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    rho: float,
    sigma: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """SDM kernel: ``y = (I - rho W)^-1 (X beta1 + WX beta2 + eps)``.

    Allows degenerate cases that nest the simpler models:

    * ``len(beta2) == 0`` skips the WX term (collapses SDM -> SAR / OLS).
    * ``rho == 0`` skips the spatial solve (collapses SDM -> SLX / OLS).
    * ``Wd is None`` is permitted only when both restrictions above hold,
      enabling the OLS path with no weights matrix.
    """
    has_wx = len(beta2) > 0
    if has_wx:
        if Wd is None:
            raise ValueError("W must be supplied when beta2 is non-empty.")
        Wx = Wd @ X[:, 1:]
        if Wx.shape[1] != len(beta2):
            raise ValueError(
                "len(beta2) must match number of non-intercept regressors."
            )
        wx_beta = Wx @ beta2
    else:
        wx_beta = 0.0

    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    rhs = X @ beta1 + wx_beta + eps
    if Wd is None or rho == 0.0:
        return rhs
    _check_rho_stability(rho, Wd, name="rho")
    return np.linalg.solve(np.eye(nobs) - rho * Wd, rhs)


def _simulate_sdem_core(
    *,
    nobs: int,
    Wd: np.ndarray | None,
    X: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    lam: float,
    sigma: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """SDEM kernel: ``y = X beta1 + WX beta2 + (I - lam W)^-1 eps``.

    Same nesting semantics as :func:`_simulate_sdm_core`: ``beta2=[]``
    drops the WX term, ``lam=0`` drops the error solve, and ``Wd=None``
    is allowed only when both restrictions hold.
    """
    has_wx = len(beta2) > 0
    if has_wx:
        if Wd is None:
            raise ValueError("W must be supplied when beta2 is non-empty.")
        Wx = Wd @ X[:, 1:]
        if Wx.shape[1] != len(beta2):
            raise ValueError(
                "len(beta2) must match number of non-intercept regressors."
            )
        wx_beta = Wx @ beta2
    else:
        wx_beta = 0.0

    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    if Wd is None or lam == 0.0:
        u = eps
    else:
        _check_rho_stability(lam, Wd, name="lam")
        u = np.linalg.solve(np.eye(nobs) - lam * Wd, eps)
    return X @ beta1 + wx_beta + u


def simulate_sar(
    n: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SAR DGP ``y = (I-rho W)^(-1)(X beta + eps)``.

    Parameters
    ----------
    n : int, optional
        Square-grid side length used when only ``n`` is supplied. This
        generates ``n * n`` observations on an ``n x n`` rook grid.
        When ``W`` or ``gdf`` is provided, ``n`` (if provided) must match
        the implied number of observations.
    W : Graph or sparse/dense matrix, optional
        Spatial weights. If supplied, takes precedence over ``gdf``.
    gdf : geopandas.GeoDataFrame, optional
        Spatial units source used when ``W`` is not provided.
    rho : float, default=0.5
        Spatial autoregressive coefficient.
    beta : np.ndarray, optional
        Regression coefficients including intercept.
    sigma : float, default=1.0
        Innovation standard deviation.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_i\\|^2}`.
    rng : np.random.Generator, optional
        Random generator.
    seed : int, optional
        Seed used when ``rng`` is not supplied.
    contiguity : str, default="queen"
        Neighbor rule for ``gdf`` mode.
    create_gdf : bool, default=False
        If True, include a ``gdf`` key in the returned dict with ``y`` and
        ``X_*`` columns attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``create_gdf=True`` and ``gdf`` is not
        provided.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.

    Notes
    -----
    Equivalent to ``simulate_sdm`` with ``beta2=[]`` (no WX terms); see
    :func:`simulate_sdm` for the unified Spatial Durbin form.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Wd=Wd,
        X=X,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        rho=rho,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sar_negbin(
    n: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    alpha: float = 2.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    r"""Simulate data from a SAR-NB2 DGP.

    The latent log-mean follows the SAR reduced form, matching the
    :class:`SARNegativeBinomial` model specification:

    .. math::

        \eta = (I - \rho W)^{-1} X\beta, \quad \mu = \exp(\eta)

    and counts are sampled as NB2:

    .. math::

        y_i \sim \mathrm{NegBin}(\mu_i, \alpha),
        \;\;\mathrm{Var}(y_i)=\mu_i+\mu_i^2/\alpha.
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if err_hetero:
        warnings.warn(
            "err_hetero is not implemented for simulate_sar_negbin and is ignored.",
            stacklevel=2,
        )

    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta is None:
        beta = np.array([1.0, 0.6], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    _check_rho_stability(rho, Wd, name="rho")
    eta = np.linalg.solve(np.eye(nobs) - rho * Wd, X @ beta)
    mu = np.exp(np.clip(eta, -30.0, 30.0))

    p = alpha / (alpha + mu)
    y = rng.negative_binomial(alpha, p).astype(np.float64)

    out = {
        "y": y,
        "X": X,
        "mu": mu,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta": beta, "alpha": alpha},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_ols(
    n: int | None = None,
    W=None,
    gdf=None,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from a non-spatial OLS DGP ``y = X beta + eps``.

    Generates a random design matrix with an intercept and ``len(beta) - 1``
    continuous regressors, and draws the response from a homoskedastic
    Normal error model. No spatial weights matrix is required or produced;
    this function is the natural complement to the spatial DGPs for use as
    a non-spatial baseline.

    Parameters
    ----------
    n : int, optional
        Number of observations when neither ``W`` nor ``gdf`` is provided.
    W : Graph or sparse/dense matrix, optional
        Spatial weights input used only to infer ``n`` and validate dimensions.
        Not used in the OLS data-generating mechanism.
    gdf : geopandas.GeoDataFrame, optional
        Spatial units source used only to infer ``n`` when ``W`` is not provided.
    beta : array-like, optional
        Coefficient vector including intercept. Defaults to
        ``[1.0, 2.0]`` (intercept = 1, one regressor with slope = 2).
    sigma : float, default=1.0
        Innovation standard deviation :math:`\\sigma`.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_i\\|^2}`.
    rng : numpy.random.Generator, optional
        Random generator instance for reproducibility.
    seed : int, optional
        Integer seed used when ``rng`` is not supplied.
    contiguity : str, default="queen"
        Neighbor rule used when inferring ``n`` from ``gdf``.
    create_gdf : bool, default=False
        If ``True``, attaches a GeoDataFrame with ``y`` and ``X_*`` columns
        to geometry generated on an ``n``-unit grid.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``create_gdf=True``.

    Returns
    -------
    dict
        Keys:

        - ``y`` : np.ndarray of shape ``(n,)`` — response variable.
        - ``X`` : np.ndarray of shape ``(n, k)`` — design matrix with
          intercept in the first column.
        - ``params_true`` : dict with ``beta`` and ``sigma``.
        - ``gdf`` : GeoDataFrame (only present when ``create_gdf=True``).

    Notes
    -----
    Equivalent to ``simulate_sdm`` with ``rho=0`` and ``beta2=[]``; the
    spatial weights matrix is ignored even when supplied. See
    :func:`simulate_sdm` for the unified Spatial Durbin form.
    """
    rng = ensure_rng(rng, seed)

    if n is None and W is None and gdf is None:
        raise ValueError("Provide one of n, W, or gdf.")

    if W is not None or gdf is not None:
        Wd, _ = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
        nobs = Wd.shape[0]
    else:
        nobs = int(n)

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Wd=None,
        X=X,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        rho=0.0,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )

    out: dict = {
        "y": y,
        "X": X,
        "params_true": {"beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sem(
    n: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SEM DGP ``y = X beta + (I-lam W)^(-1) eps``.

    Parameters are analogous to :func:`simulate_sar` with ``lam`` replacing
    ``rho``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.

    Notes
    -----
    Equivalent to ``simulate_sdem`` with ``beta2=[]`` (no WX terms); see
    :func:`simulate_sdem` for the unified Spatial Durbin Error form.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    y = _simulate_sdem_core(
        nobs=nobs,
        Wd=Wd,
        X=X,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        lam=lam,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_slx(
    n: int | None = None,
    W=None,
    gdf=None,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SLX DGP ``y = X beta1 + W X_no_intercept beta2 + eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Wd=Wd,
        X=X,
        beta1=beta1,
        beta2=beta2,
        rho=0.0,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sdm(
    n: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SDM DGP ``y = (I-rho W)^(-1)(Xb1 + WXb2 + eps)``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Wd=Wd,
        X=X,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sdem(
    n: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SDEM DGP ``y = Xb1 + WXb2 + (I-lam W)^(-1)eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, n=n, contiguity=contiguity)
    nobs = Wd.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    y = _simulate_sdem_core(
        nobs=nobs,
        Wd=Wd,
        X=X,
        beta1=beta1,
        beta2=beta2,
        lam=lam,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )
