"""Spatial block cross-validation for fitted Bayesian spatial models.

Implements the refit-based spatial k-fold predictive evaluation of
:cite:t:`roberts2017CrossValidationStrategies` for the models in
:mod:`bayespecon.models`.

The estimator avoids the well-known failures of PSIS-LOO on spatially
dependent data (importance ratios assume the per-observation likelihoods
factorise across observations, which spatial models violate by
construction).  Each fold refits the model on the training subset and
evaluates ``log p(y_test | y_train, theta)`` under the *full-data*
joint Gaussian induced by the model:

.. math::

    \\log p(y_{\\text{test}} \\mid y_{\\text{train}}, \\theta) =
    \\tfrac{1}{2} \\log |\\Lambda_{tt}|
    - \\tfrac{n_{\\text{test}}}{2} \\log(2\\pi)
    - \\tfrac{1}{2} z_{\\text{test}}^{\\top} \\Lambda_{tt}^{-1} z_{\\text{test}},

where :math:`\\Lambda` is the full :math:`n\\times n` precision matrix at
draw :math:`\\theta`, :math:`\\Lambda_{tt}` is its test-block, and
:math:`z = \\Lambda(y - \\mu)` with :math:`\\mu` the implied marginal
mean (:math:`A^{-1}X\\beta` for SAR/SDM, :math:`X\\beta` for SEM/SDEM,
:math:`X\\beta` for OLS/SLX).  Per-fold elpd is obtained by
``logsumexp`` over posterior draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve
from scipy.special import logsumexp

__all__ = ["SpatialCVResult", "spatial_kfold"]


@dataclass
class SpatialCVResult:
    """Result of :func:`spatial_kfold`.

    Attributes
    ----------
    elpd : float
        Expected log pointwise predictive density summed over folds.
    se : float
        Standard error of ``elpd``, estimated as
        ``sqrt(n * var(per_obs_elpd))`` where ``per_obs_elpd`` spreads each
        fold's elpd uniformly over its observations.
    elpd_per_fold : np.ndarray
        Per-fold elpd of shape ``(n_folds,)``.
    n_per_fold : np.ndarray
        Number of observations in each fold, shape ``(n_folds,)``.
    fold_ids : np.ndarray
        Integer fold assignment for each observation, shape ``(n,)``.
    n_folds : int
        Number of folds actually used.
    method : str
        ``"explicit"`` if ``fold_ids`` was supplied, ``"kmeans"`` if
        folds were derived from geometry.
    """

    elpd: float
    se: float
    elpd_per_fold: np.ndarray
    n_per_fold: np.ndarray
    fold_ids: np.ndarray
    n_folds: int
    method: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_kind(model: Any) -> str:
    """Classify a fitted model as ``"lag"``, ``"error"`` or ``"iid"``."""
    name = type(model).__name__.upper()
    if name in ("SAR", "SDM"):
        return "lag"
    if name in ("SEM", "SDEM"):
        return "error"
    if name in ("OLS", "SLX"):
        return "iid"
    raise NotImplementedError(
        f"spatial_kfold is not implemented for model class {type(model).__name__}."
    )


_WX_MODELS = {"SLX", "SDM", "SDEM"}


def _full_design(model: Any) -> np.ndarray:
    """Effective design matrix matching the model's ``beta`` posterior.

    SLX / SDM / SDEM concatenate ``[X, WX]`` into ``beta``; the other
    model families fit ``beta`` to ``X`` alone even though ``_WX`` is
    cached on the model.
    """
    uses_wx = type(model).__name__.upper() in _WX_MODELS
    if not uses_wx or model._WX.shape[1] == 0:
        return np.asarray(model._X, dtype=np.float64)
    return np.hstack([np.asarray(model._X), np.asarray(model._WX)]).astype(np.float64)


def _stack_draws(idata, name: str) -> np.ndarray:
    """Stack chain × draw → sample for posterior variable ``name``."""
    arr = idata.posterior[name]
    return arr.stack(sample=("chain", "draw")).transpose("sample", ...).values


def _kmeans_fold_ids(geometry, n_blocks: int, seed: int) -> np.ndarray:
    """Cluster centroid coordinates into ``n_blocks`` spatial blocks."""
    from sklearn.cluster import KMeans

    cx = np.asarray(geometry.centroid.x, dtype=np.float64)
    cy = np.asarray(geometry.centroid.y, dtype=np.float64)
    coords = np.column_stack([cx, cy])
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    return km.fit_predict(coords).astype(np.int64)


def _refit_on_train(
    model: Any,
    train_idx: np.ndarray,
    fit_kwargs: dict,
) -> Any:
    """Construct a fresh model instance on the training subset and fit it."""
    W_train: Optional[sp.spmatrix]
    if model._W_sparse is not None:
        W_train = model._W_sparse[train_idx, :][:, train_idx].tocsr()
    else:
        W_train = None
    new = model.__class__(
        y=model._y[train_idx],
        X=model._X[train_idx, :],
        W=W_train,
        priors=model.priors_obj,
        logdet_method=model.logdet_method,
        robust=model.robust,
        backend=model.backend_name,
    )
    new.fit(**fit_kwargs)
    return new


def _fold_elpd(
    refit: Any,
    *,
    y_full: np.ndarray,
    design_full: np.ndarray,
    W_full: Optional[sp.csr_matrix],
    test_idx: np.ndarray,
    kind: str,
) -> float:
    """log E_posterior[ p(y_test | y_train, theta) ] from refit's draws."""
    idata = refit.inference_data
    beta = _stack_draws(idata, "beta")  # (G, k_design)
    sigma = _stack_draws(idata, "sigma").reshape(-1)  # (G,)
    if beta.shape[1] != design_full.shape[1]:
        raise ValueError(
            "Posterior beta dimension does not match the full-data design "
            f"matrix ({beta.shape[1]} vs {design_full.shape[1]}). This can "
            "happen if a column became constant on the training subset."
        )

    G = beta.shape[0]
    n = y_full.shape[0]
    n_test = test_idx.shape[0]

    if kind == "iid":
        # Closed form: independent normal predictive.
        X_test = design_full[test_idx]
        y_test = y_full[test_idx]
        mu_g = X_test @ beta.T  # (n_test, G)
        r = y_test[:, None] - mu_g
        s = sigma[None, :]
        log_p = (
            -0.5 * np.sum((r / s) ** 2, axis=0)
            - n_test * np.log(sigma)
            - 0.5 * n_test * np.log(2.0 * np.pi)
        )
        return float(logsumexp(log_p) - np.log(G))

    if W_full is None:
        raise ValueError(f"Model kind {kind!r} requires W_full but it is None.")

    spatial = _stack_draws(idata, "rho" if kind == "lag" else "lam").reshape(-1)
    eye_n = sp.eye(n, format="csr")
    log_p = np.empty(G, dtype=np.float64)

    for g in range(G):
        theta = float(spatial[g])
        s2 = float(sigma[g]) ** 2
        A = eye_n - theta * W_full  # I - rho*W or I - lambda*W
        Xb = design_full @ beta[g]
        mu = spsolve(A.tocsc(), Xb) if kind == "lag" else Xb
        Lam = (A.T @ A).tocsc() / s2  # full precision
        r = y_full - mu
        z = Lam @ r
        z_test = z[test_idx]
        Lam_tt = Lam[test_idx, :][:, test_idx].tocsc()
        try:
            lu = splu(Lam_tt)
            v = lu.solve(z_test)
            diagU = lu.U.diagonal()
            logdet = float(np.sum(np.log(np.abs(diagU))))
        except Exception:
            Lam_tt_dense = Lam_tt.toarray()
            v = np.linalg.solve(Lam_tt_dense, z_test)
            _, logdet = np.linalg.slogdet(Lam_tt_dense)
            logdet = float(logdet)
        quad = float(z_test @ v)
        log_p[g] = 0.5 * logdet - 0.5 * n_test * np.log(2.0 * np.pi) - 0.5 * quad

    return float(logsumexp(log_p) - np.log(G))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spatial_kfold(
    model: Any,
    *,
    fold_ids: Optional[np.ndarray] = None,
    n_blocks: int = 10,
    geometry: Optional[Any] = None,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    random_seed: int = 0,
    progressbar: bool = False,
    **fit_kwargs: Any,
) -> SpatialCVResult:
    """Spatial block cross-validation for a fitted Bayesian spatial model.

    Refits the model on each training fold and evaluates the conditional
    Gaussian predictive density of the held-out fold under the full-data
    joint implied by the model.

    Parameters
    ----------
    model : SpatialModel
        A model from :mod:`bayespecon.models`.  The model must have been
        constructed (its ``_X``, ``_y`` and ``_W_sparse`` will be used for
        prediction); it does **not** need to be fit, since fold-specific
        refits are performed internally.
    fold_ids : np.ndarray, optional
        Integer fold assignment per observation, shape ``(n,)``.  When
        supplied, ``n_blocks`` and ``geometry`` are ignored.
    n_blocks : int, default 10
        Number of spatial blocks for the KMeans fallback when
        ``fold_ids`` is not provided.
    geometry : geopandas.GeoSeries, optional
        Geometry used by the KMeans fallback to cluster centroids.
        Required when ``fold_ids`` is ``None``.
    draws, tune, chains, random_seed, progressbar
        Forwarded to :meth:`SpatialModel.fit` for each per-fold refit.
        Defaults are deliberately modest to keep CV affordable.
    **fit_kwargs
        Extra keyword arguments forwarded to :meth:`SpatialModel.fit`.

    Returns
    -------
    SpatialCVResult

    Notes
    -----
    Computation is :math:`O(K \\cdot G \\cdot \\text{nnz}(W))` per fold
    plus the cost of refitting; spatial folds are typically a handful
    (e.g. ``n_blocks=5``\u201310).  For ``OLS``/``SLX`` the predictive
    collapses to the standard independent Gaussian and the per-fold
    cost is :math:`O(G \\cdot n_{\\text{test}} \\cdot k)`.
    """
    n = int(model._y.shape[0])
    if fold_ids is None:
        if geometry is None:
            raise ValueError(
                "Either fold_ids or geometry (for KMeans blocking) must be supplied."
            )
        fold_ids = _kmeans_fold_ids(geometry, n_blocks=n_blocks, seed=random_seed)
        method = "kmeans"
    else:
        fold_ids = np.asarray(fold_ids, dtype=np.int64).ravel()
        if fold_ids.shape[0] != n:
            raise ValueError(
                f"fold_ids has length {fold_ids.shape[0]}, expected n={n}."
            )
        method = "explicit"

    unique_folds = np.unique(fold_ids)
    n_folds = unique_folds.shape[0]
    if n_folds < 2:
        raise ValueError(
            f"spatial_kfold requires at least 2 folds (got {n_folds})."
        )

    kind = _model_kind(model)
    y_full = np.asarray(model._y, dtype=np.float64)
    design_full = _full_design(model)
    W_full = model._W_sparse
    if W_full is not None and not sp.isspmatrix_csr(W_full):
        W_full = W_full.tocsr()

    base_fit_kwargs = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        progressbar=progressbar,
    )
    base_fit_kwargs.update(fit_kwargs)

    elpd_per_fold = np.empty(n_folds, dtype=np.float64)
    n_per_fold = np.empty(n_folds, dtype=np.int64)
    per_obs_elpd = np.empty(n, dtype=np.float64)

    for f, fold_label in enumerate(unique_folds):
        test_mask = fold_ids == fold_label
        test_idx = np.flatnonzero(test_mask).astype(np.int64)
        train_idx = np.flatnonzero(~test_mask).astype(np.int64)
        if test_idx.size == 0 or train_idx.size == 0:
            raise ValueError(
                f"Fold {fold_label} produces an empty test or training partition."
            )
        refit = _refit_on_train(model, train_idx, base_fit_kwargs)
        fold_elpd = _fold_elpd(
            refit,
            y_full=y_full,
            design_full=design_full,
            W_full=W_full,
            test_idx=test_idx,
            kind=kind,
        )
        elpd_per_fold[f] = fold_elpd
        n_per_fold[f] = test_idx.size
        per_obs_elpd[test_idx] = fold_elpd / test_idx.size

    elpd_total = float(elpd_per_fold.sum())
    se = float(np.sqrt(n * np.var(per_obs_elpd, ddof=1))) if n > 1 else 0.0

    return SpatialCVResult(
        elpd=elpd_total,
        se=se,
        elpd_per_fold=elpd_per_fold,
        n_per_fold=n_per_fold,
        fold_ids=fold_ids,
        n_folds=n_folds,
        method=method,
    )
