"""Shared helper functions for Bayesian spatial regression models.

These utilities are used by multiple base classes (SpatialModel,
SpatialPanelModel, FlowModel, FlowPanelModel) and are collected here
to avoid circular imports and code duplication.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from libpysal.graph import Graph

from ..._lazy_deps import az, pm


def gelman_default_beta_prior(
    y: np.ndarray,
    design: np.ndarray,
    feature_names: list[str],
    scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Weakly-informative default prior on regression coefficients.

    Follows Gelman, Jakulin, Pittau & Su (2008) by setting per-column
    prior scales from ``sd(y)`` and ``sd(x_j)``.  For each column ``j``
    of ``design``:

    * **Intercept-like** (named ``"intercept"`` or numerically constant):
      ``mu_j = mean(y)``, ``sigma_j = scale * sd(y)``.
    * **Slope**:
      ``mu_j = 0``, ``sigma_j = scale * sd(y) / sd(x_j)``.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    design : ndarray, shape (n, p)
        Effective design matrix used by ``beta`` in the model
        (i.e. ``X`` for SAR/SEM/OLS; ``[X, WX]`` for SDM/SDEM/SLX).
    feature_names : list[str]
        Column labels aligned with ``design``.  Used to detect
        intercept-like columns named ``"intercept"``.
    scale : float, default 2.5
        Multiplier on the standardised prior scale.

    Returns
    -------
    beta_mu : ndarray, shape (p,)
    beta_sigma : ndarray, shape (p,)

    References
    ----------
    Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008).
    *A weakly informative default prior distribution for logistic and
    other regression models.* Annals of Applied Statistics, 2(4),
    1360-1383.
    """
    sd_y = float(np.std(y))
    if sd_y <= 0.0:
        sd_y = 1.0
    mean_y = float(np.mean(y))
    p = design.shape[1]
    beta_mu = np.zeros(p, dtype=np.float64)
    beta_sigma = np.empty(p, dtype=np.float64)
    for j in range(p):
        col = design[:, j]
        name = feature_names[j] if j < len(feature_names) else ""
        is_named_intercept = name.lower() == "intercept"
        is_constant = np.allclose(col, col[0])
        if is_named_intercept or is_constant:
            beta_mu[j] = mean_y
            beta_sigma[j] = scale * sd_y
        else:
            sd_col = float(np.std(col))
            beta_sigma[j] = scale * sd_y / sd_col if sd_col > 0.0 else scale * sd_y
    return beta_mu, beta_sigma


def _is_row_standardized_csr(W_csr: sp.csr_matrix) -> bool:
    """Return True when each row sum is numerically close to one."""
    row_sums = np.asarray(W_csr.sum(axis=1)).ravel()
    return bool(np.allclose(row_sums, 1.0, atol=1e-6))


def resolve_W(
    W: Union[Graph, sp.spmatrix],
    n: int,
    T: int = 1,
) -> tuple[sp.csr_matrix, bool]:
    """Validate and normalise a spatial weights argument to CSR.

    Unified W parser for cross-section (T=1) and panel (T>1) models.
    Accepts a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
    matrix.

    Parameters
    ----------
    W :
        Either a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
        matrix.
    n :
        Expected number of cross-sectional units.
    T :
        Number of time periods.  When ``T=1`` (default), W must be ``n×n``.
        When ``T>1``, W may be ``n×n`` (broadcast over time) or
        ``(n*T)×(n*T)`` (full block-diagonal panel matrix).

    Returns
    -------
    W_csr : scipy.sparse.csr_matrix
        Row-compressed version of W.
    row_std : bool
        Whether W appears to be row-standardised.

    Raises
    ------
    TypeError
        If *W* is not a Graph or scipy sparse matrix.
    ValueError
        If *W* is not square or its size does not match *n* (or *n*T*).

    Warns
    -----
    UserWarning
        If *W* does not appear to be row-standardised.
    """
    if isinstance(W, Graph):
        W_csr = W.sparse.tocsr().astype(np.float64)
        transform = getattr(W, "transformation", None)
        row_std = transform in ("r", "R") or _is_row_standardized_csr(W_csr)
    elif sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
        row_std = _is_row_standardized_csr(W_csr)
    elif hasattr(W, "sparse") and hasattr(W, "transform"):
        raise TypeError(
            "W appears to be a legacy libpysal.weights.W object. "
            "Convert it to a libpysal.graph.Graph first: "
            "Graph.from_W(w), or pass w.sparse (the scipy sparse matrix) directly."
        )
    else:
        raise TypeError(
            f"W must be a libpysal.graph.Graph or a scipy sparse matrix, "
            f"got {type(W).__name__}."
        )

    if W_csr.ndim != 2 or W_csr.shape[0] != W_csr.shape[1]:
        raise ValueError(f"W must be a square matrix, got shape {W_csr.shape}.")

    if T > 1:
        # Panel mode: accept n×n or (n*T)×(n*T)
        if W_csr.shape[0] == n:
            pass  # n×n — will be Kronecker-expanded by caller
        elif W_csr.shape[0] == n * T:
            pass  # full block-diagonal panel matrix
        else:
            raise ValueError(
                f"W has shape {W_csr.shape} but data has N={n} units (T={T} periods). "
                f"W must be ({n},{n}) or ({n * T},{n * T})."
            )
    else:
        # Cross-section mode: must be n×n
        if W_csr.shape[0] != n:
            raise ValueError(
                f"W has shape {W_csr.shape} but data has {n} observations. "
                "W must be an n\u00d7n matrix."
            )

    if not row_std:
        warnings.warn(
            "W does not appear to be row-standardised (row sums \u2260 1). "
            "Most spatial models assume W is row-standardised; results may be "
            "unreliable otherwise. For a scipy sparse matrix normalise rows "
            "manually (divide each row by its sum). To use a libpysal.graph.Graph "
            "set its transformation attribute: "
            "graph = graph.transform('r').",
            UserWarning,
            stacklevel=3,
        )
    return W_csr, row_std


def _parse_W(
    W: Union[Graph, sp.spmatrix],
    n: int,
) -> tuple[sp.csr_matrix, bool]:
    """Backward-compatible alias for :func:`resolve_W` with ``T=1``."""
    return resolve_W(W, n, T=1)


def _pointwise_gaussian_loglik(
    eps: np.ndarray,
    sigma_draws: np.ndarray,
    nu_draws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute pointwise Gaussian or Student-t log-likelihood.

    Parameters
    ----------
    eps : np.ndarray
        Residual matrix of shape ``(n_draws, n_obs)``.
    sigma_draws : np.ndarray
        Posterior scale draws of shape ``(n_draws,)``.
    nu_draws : np.ndarray, optional
        Student-t degrees-of-freedom draws of shape ``(n_draws,)``.
        When ``None``, computes Gaussian log-likelihood.

    Returns
    -------
    np.ndarray
        Pointwise log-likelihood with shape ``(n_draws, n_obs)``.
    """
    eps = np.asarray(eps, dtype=np.float64)
    sigma = np.asarray(sigma_draws, dtype=np.float64).reshape(-1)
    if eps.ndim != 2:
        raise ValueError(f"eps must be 2D (n_draws, n_obs), got shape {eps.shape}.")
    if sigma.shape[0] != eps.shape[0]:
        raise ValueError(
            "sigma_draws length must equal eps first dimension; "
            f"got {sigma.shape[0]} and {eps.shape[0]}."
        )

    sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    sigma_2d = sigma[:, None]

    if nu_draws is None:
        return (
            -0.5 * (eps / sigma_2d) ** 2 - np.log(sigma_2d) - 0.5 * np.log(2.0 * np.pi)
        )

    nu = np.asarray(nu_draws, dtype=np.float64).reshape(-1)
    if nu.shape[0] != eps.shape[0]:
        raise ValueError(
            "nu_draws length must equal eps first dimension; "
            f"got {nu.shape[0]} and {eps.shape[0]}."
        )
    nu = np.maximum(nu, np.finfo(np.float64).tiny)
    from scipy import stats

    return stats.t.logpdf(eps, df=nu[:, None], loc=0.0, scale=sigma_2d)


def _write_log_likelihood_to_idata(
    idata: az.InferenceData,
    ll_array: np.ndarray,
) -> None:
    """Write a complete pointwise log-likelihood array to InferenceData.

    Parameters
    ----------
    idata : az.InferenceData
        Target inference data object to mutate in place.
    ll_array : np.ndarray
        Array with shape ``(chain, draw, obs)``.
    """
    import xarray as xr

    ll = np.asarray(ll_array, dtype=np.float64)
    if ll.ndim != 3:
        raise ValueError(
            f"ll_array must be 3D (chain, draw, obs), got shape {ll.shape}."
        )

    ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
    idata["log_likelihood"] = xr.Dataset({"obs": ll_da})


def _tobit_pointwise_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    sigma_f: np.ndarray,
    censored: np.ndarray,
    censoring: float,
    nu_f: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pointwise Tobit log-likelihood (density part, no Jacobian).

    Uncensored observations contribute the log-density of the (Normal, or
    Student-t when ``nu_f`` is given) innovation evaluated at ``y``; censored
    observations contribute the corresponding left-tail log-CDF at the
    censoring point ``c``.  The caller adds the ``log|I - ρW|`` Jacobian.

    Parameters
    ----------
    y : np.ndarray
        Observed (censored) response, shape ``(n,)``.
    mu : np.ndarray
        Latent mean per flattened posterior draw, shape ``(s, n)``.
    sigma_f : np.ndarray
        Innovation scale per draw, shape ``(s,)``.
    censored : np.ndarray
        Boolean mask of censored observations, shape ``(n,)``.
    censoring : float
        Left-censoring threshold ``c``.
    nu_f : np.ndarray, optional
        Student-t degrees of freedom per draw, shape ``(s,)``.  ``None``
        (default) selects the Gaussian innovation.

    Returns
    -------
    np.ndarray
        Pointwise log-likelihood, shape ``(s, n)``.
    """
    s, n = mu.shape
    ll = np.empty((s, n), dtype=np.float64)
    uncens = ~censored
    sig = sigma_f[:, None]
    resid = (y[uncens][None, :] - mu[:, uncens]) / sig
    if nu_f is not None:
        from scipy.special import gammaln
        from scipy.stats import t as t_dist

        nu = nu_f[:, None]
        ll[:, uncens] = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - np.log(sig)
            - (nu + 1) / 2 * np.log1p(resid**2 / nu)
        )
        ll[:, censored] = t_dist.logcdf((censoring - mu[:, censored]) / sig, df=nu)
    else:
        from scipy.stats import norm

        ll[:, uncens] = -0.5 * (resid**2 + np.log(2.0 * np.pi) + 2.0 * np.log(sig))
        ll[:, censored] = norm.logcdf((censoring - mu[:, censored]) / sig)
    return ll


class SharedSpatialMethods:
    """Behaviour-preserving mixin of methods shared verbatim by the two base
    classes (:class:`SpatialModel` and :class:`SpatialPanelModel`).

    Phase 5c converges the ~39 duplicated methods into this single home before
    the two hierarchies are collapsed.  Only methods whose bodies are *already
    identical* across the two classes live here; anything that differs (N vs NT
    sizing, cross-section vs panel Jacobians, ...) stays on the subclasses and
    overrides the mixin via normal MRO.  All methods rely only on attributes
    both classes provide (``self._X``, ``self._feature_names``, ``self.priors``,
    ``self._idata``).
    """

    @property
    def _nonintercept_indices(self) -> list[int]:
        """Return indices of non-constant (non-intercept) columns in X.

        This is used to exclude the intercept from impact measures, since
        the intercept has no meaningful spatial effect interpretation.

        Returns
        -------
        list[int]
            Column indices of X that are not constant/intercept columns.
        """
        indices: list[int] = []
        for j, name in enumerate(self._feature_names):
            column = self._X[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    @property
    def _nonintercept_feature_names(self) -> list[str]:
        """Return feature names for non-intercept columns.

        Returns
        -------
        list[str]
            Feature names excluding intercept/constant columns.
        """
        return [self._feature_names[i] for i in self._nonintercept_indices]

    def _add_nu_prior(self, model: pm.Model) -> pm.Model:
        """Add the degrees-of-freedom prior for robust (Student-t) models.

        Called inside ``_build_pymc_model`` when ``self.robust`` is True.
        Uses an :math:`\\mathrm{Exp}(\\lambda_\\nu)` prior on ``nu`` with rate ``nu_lam`` (default
        1/30, giving mean ≈ 30, favouring near-Normal tails). A lower
        bound of 2 is enforced so that the variance exists.

        Parameters
        ----------
        model : pymc.Model
            The model context in which to add the ``nu`` prior.

        Returns
        -------
        pymc.Model
            The same model context (``nu`` is added as a side effect).
        """
        nu_lam = self.priors.get("nu_lam", 1.0 / 30.0)
        pm.Truncated("nu", pm.Exponential.dist(lam=nu_lam), lower=2.0)
        return model

    def _beta_names(self) -> list[str]:
        """Return coefficient labels used for posterior summaries.

        Returns
        -------
        list[str]
            Coefficient labels aligned with the ``beta`` parameter.
        """
        return list(self._feature_names)

    def _model_coords(self) -> dict[str, list[str]]:
        """Return PyMC coordinate labels for named dimensions.

        Returns
        -------
        dict[str, list[str]]
            Coordinates passed to :class:`pymc.Model`.
        """
        return {"coefficient": self._beta_names()}

    @staticmethod
    def _rename_summary_index(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Strip the ``beta[...]`` wrapper from coefficient row labels.

        Parameters
        ----------
        summary_df : pandas.DataFrame
            ArviZ summary output.

        Returns
        -------
        pandas.DataFrame
            Summary with human-readable coefficient row labels.
        """
        renamed = []
        for label in summary_df.index.astype(str):
            if label.startswith("beta[") and label.endswith("]"):
                renamed.append(label[5:-1])
            else:
                renamed.append(label)
        out = summary_df.copy()
        out.index = renamed
        return out

    @property
    def inference_data(self) -> Optional[az.InferenceData]:
        """Return the ArviZ InferenceData from the most recent fit.

        Returns
        -------
        arviz.InferenceData or None
            The inference data object, or ``None`` if the model has not
            been fit yet.
        """
        return self._idata

    def _require_fit(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

    def _posterior_mean(self, var: str) -> np.ndarray:
        return self._idata.posterior[var].mean(("chain", "draw")).to_numpy()

    def _postprocess_idata(self, idata: az.InferenceData) -> az.InferenceData:
        """Hook to augment ``idata`` after sampling, before it is returned.

        The default is a no-op.  Subclasses whose likelihood is expressed via
        ``pm.Potential`` (so PyMC captures no pointwise ``log_likelihood``),
        e.g. the (panel) Tobit families, override this to attach a complete
        Jacobian-corrected pointwise log-likelihood for information criteria.
        """
        return idata

    @staticmethod
    def _spatial_lag_column_indices(
        X: np.ndarray, feature_names: list[str]
    ) -> list[int]:
        """Return indices of regressors that should receive spatial lags.

        Constant columns are treated as intercept-like and excluded, which
        avoids adding redundant ``W * intercept`` terms to SLX/Durbin models.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (raw, before any panel fixed-effects transformation).
        feature_names : list[str]
            Column labels aligned with ``X``.

        Returns
        -------
        list[int]
            Column indices eligible for spatial lags.
        """
        indices: list[int] = []
        for j, name in enumerate(feature_names):
            column = X[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    def summary(self, var_names: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """Return posterior summary table.

        Parameters
        ----------
        var_names : list, optional
            Variable names to include in the summary.
        **kwargs
            Additional arguments passed to :func:`arviz.summary`.

        Returns
        -------
        pandas.DataFrame
            Posterior summary statistics.
        """
        self._require_fit()
        summary_df = az.summary(self._idata, var_names=var_names, **kwargs)
        return self._rename_summary_index(summary_df)

    def fitted_values(self) -> np.ndarray:
        """Return fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values (on the model's native scale;
            fixed-effects-transformed for panel models).
        """
        self._require_fit()
        return self._fitted_mean_from_posterior()

    def residuals(self) -> np.ndarray:
        """Return residuals ``y - fitted_values``.

        Returns
        -------
        np.ndarray
            Residual vector ``y - fitted_values`` on the same scale as
            :meth:`fitted_values`.
        """
        self._require_fit()
        return self._y - self.fitted_values()
