"""Base classes and helpers for spatial panel models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from formulaic import model_matrix
from libpysal.graph import Graph

from ..diagnostics import (
    arch_test,
    bpagan_test,
    ljung_box_q,
    outlier_candidates,
    panel_residual_structure,
    pesaran_cd_test,
    rdiagnose_like,
)


def _demean_panel(y: np.ndarray, X: np.ndarray, N: int, T: int, model: int):
    """Apply panel demeaning transformation.

    Parameters
    ----------
    y : np.ndarray
        Stacked dependent variable of shape ``(N*T,)``.
    X : np.ndarray
        Stacked regressor matrix of shape ``(N*T, k)``.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    model : int
        Fixed-effects mode: 0 pooled, 1 unit FE, 2 time FE, 3 two-way FE.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Demeaned ``(y, X)`` arrays in stacked format.
    """
    y2 = y.reshape(T, N)
    X3 = X.reshape(T, N, X.shape[1])

    if model == 0:
        y_with = y2
        X_with = X3
    elif model == 1:
        y_with = y2 - y2.mean(axis=0, keepdims=True)
        X_with = X3 - X3.mean(axis=0, keepdims=True)
    elif model == 2:
        y_with = y2 - y2.mean(axis=1, keepdims=True)
        X_with = X3 - X3.mean(axis=1, keepdims=True)
    elif model == 3:
        y_i = y2.mean(axis=0, keepdims=True)
        y_t = y2.mean(axis=1, keepdims=True)
        y_g = y2.mean()
        y_with = y2 - y_i - y_t + y_g

        X_i = X3.mean(axis=0, keepdims=True)
        X_t = X3.mean(axis=1, keepdims=True)
        X_g = X3.mean(axis=(0, 1), keepdims=True)
        X_with = X3 - X_i - X_t + X_g
    else:
        raise ValueError("model must be one of {0,1,2,3}")

    return y_with.reshape(-1), X_with.reshape(-1, X.shape[1])


def _as_dense_W(W: Union[Graph, np.ndarray], N: int, T: int) -> np.ndarray:
    """Convert graph/array weights into dense panel-compatible matrix.

    Parameters
    ----------
    W : Graph or np.ndarray
        Either ``N x N`` cross-sectional weights or ``(N*T) x (N*T)`` panel
        weights.
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    np.ndarray
        Dense panel weights matrix.
    """
    if isinstance(W, Graph):
        Wn = W.sparse.toarray().astype(float)
    else:
        Wn = np.asarray(W, dtype=float)

    if Wn.shape == (N, N):
        return np.kron(np.eye(T), Wn)
    if Wn.shape == (N * T, N * T):
        return Wn

    raise ValueError(
        f"W has shape {Wn.shape}; expected (N,N)=({N},{N}) or (N*T,N*T)=({N*T},{N*T})."
    )


class SpatialPanelModel(ABC):
    """Base class for static spatial panel models with FE transforms.

    Parameters
    ----------
    formula, data, y, X
        Either formula mode (formula + data) or matrix mode (y + X).
    W
        Spatial weights matrix for N units (preferred) or block matrix for N*T.
    unit_col, time_col
        Required in formula mode for robust panel sorting and N/T inference.
    N, T
        Required in matrix mode if not inferable.
    model
        0 pooled, 1 unit FE, 2 time FE, 3 two-way FE.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, np.ndarray]] = None,
        unit_col: Optional[str] = None,
        time_col: Optional[str] = None,
        N: Optional[int] = None,
        T: Optional[int] = None,
        model: int = 0,
        priors: Optional[dict] = None,
        logdet_method: str = "auto",
    ):
        if W is None:
            raise ValueError("W is required.")

        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.model = int(model)
        self._idata: Optional[az.InferenceData] = None

        if formula is not None:
            if data is None:
                raise ValueError("data is required with formula mode.")
            if unit_col is None or time_col is None:
                raise ValueError("unit_col and time_col are required in formula mode.")

            d = data.sort_values([time_col, unit_col]).reset_index(drop=True)
            lhs, rhs = formula.split("~", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()

            X_mm = model_matrix(rhs, d)
            self._feature_names = list(X_mm.columns)
            y_arr = np.asarray(d[lhs], dtype=float)
            X_arr = np.asarray(X_mm, dtype=float)

            units = d[unit_col].nunique()
            times = d[time_col].nunique()
            if units * times != len(d):
                raise ValueError("Data are not a balanced panel after sorting by time/unit.")
            self._N = units
            self._T = times
            self._panel_index = d[[time_col, unit_col]].copy()
        elif y is not None and X is not None:
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            if isinstance(X, pd.DataFrame):
                self._feature_names = list(X.columns)
                X_arr = X.to_numpy(dtype=float)
            else:
                X_arr = np.asarray(X, dtype=float)
                self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

            if N is None or T is None:
                raise ValueError("N and T are required in matrix mode.")
            self._N = int(N)
            self._T = int(T)
            self._panel_index = None
            if self._N * self._T != len(y_arr):
                raise ValueError("N*T must equal number of stacked observations.")
        else:
            raise ValueError("Provide either (formula,data,unit_col,time_col) or (y,X,N,T).")

        self._y_raw = y_arr
        self._X_raw = X_arr

        self._wx_column_indices = self._spatial_lag_column_indices(self._X_raw, self._feature_names)
        self._wx_feature_names = [self._feature_names[i] for i in self._wx_column_indices]

        self._W_dense = _as_dense_W(W, self._N, self._T)

        self._y, self._X = _demean_panel(self._y_raw, self._X_raw, self._N, self._T, self.model)
        self._Wy = self._W_dense @ self._y
        if self._wx_column_indices:
            self._WX = self._W_dense @ self._X[:, self._wx_column_indices]
        else:
            self._WX = np.empty((self._X.shape[0], 0), dtype=float)

    @staticmethod
    def _spatial_lag_column_indices(X: np.ndarray, feature_names: list[str]) -> list[int]:
        """Return indices of regressors that should receive spatial lags.

        Constant columns are treated as intercept-like and excluded, which
        avoids adding redundant ``W * intercept`` terms to SLX/Durbin models.

        Parameters
        ----------
        X : np.ndarray
            Raw panel design matrix before FE transformation.
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

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model."""

    @abstractmethod
    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute direct/indirect/total effects."""

    @abstractmethod
    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values on transformed scale."""

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

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Sample the posterior for the panel model.

        Parameters
        ----------
        draws : int, default=2000
            Number of post-tuning draws per chain.
        tune : int, default=1000
            Number of tuning draws per chain.
        chains : int, default=4
            Number of chains.
        target_accept : float, default=0.9
            NUTS target acceptance probability.
        random_seed : int, optional
            Random seed used by PyMC.
        **sample_kwargs
            Extra keyword arguments forwarded to :func:`pymc.sample`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples and diagnostics.
        """
        model = self._build_pymc_model()
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                **sample_kwargs,
            )
        return self._idata

    def _require_fit(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

    def _posterior_mean(self, var: str) -> np.ndarray:
        return self._idata.posterior[var].mean(("chain", "draw")).to_numpy()

    def summary(self, var_names: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """Return posterior summary table.

        Parameters
        ----------
        var_names : list, optional
            Variable names to include.
        **kwargs
            Additional arguments passed to :func:`arviz.summary`.

        Returns
        -------
        pandas.DataFrame
            Posterior summary table.
        """
        self._require_fit()
        summary_df = az.summary(self._idata, var_names=var_names, **kwargs)
        return self._rename_summary_index(summary_df)

    def fitted_values(self) -> np.ndarray:
        """Return fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Fitted values on transformed panel scale.
        """
        self._require_fit()
        return self._fitted_mean_from_posterior()

    def residuals(self) -> np.ndarray:
        """Return transformed residuals ``y - fitted``.

        Returns
        -------
        np.ndarray
            Residual vector on transformed panel scale.
        """
        self._require_fit()
        return self._y - self.fitted_values()

    def spatial_effects(self) -> dict[str, np.ndarray]:
        """Return direct, indirect, and total effects.

        Returns
        -------
        dict
            Dictionary containing effects and feature names.
        """
        self._require_fit()
        return self._compute_spatial_effects()

    def regression_diagnostics(self) -> dict:
        """Compute rdiagnose-style influence diagnostics.

        Returns
        -------
        dict
            Influence diagnostics dictionary.
        """
        self._require_fit()
        return rdiagnose_like(self._y, self._X, self.residuals())

    def heteroskedasticity_diagnostics(self, arch_lags: int | list[int] = 5) -> dict:
        """Compute BPagan and ARCH diagnostics.

        Parameters
        ----------
        arch_lags : int or list[int], default=5
            Lag order(s) for ARCH test.

        Returns
        -------
        dict
            Dictionary with keys ``bpagan`` and ``arch``.
        """
        self._require_fit()
        resid = self.residuals()
        return {
            "bpagan": bpagan_test(resid, self._X),
            "arch": arch_test(resid, arch_lags),
        }

    def autocorrelation_diagnostics(self, lags: int | list[int] = 10) -> dict:
        """Compute Ljung-Box residual autocorrelation diagnostics.

        Parameters
        ----------
        lags : int or list[int], default=10
            Lag order(s) for Ljung-Box test.

        Returns
        -------
        dict
            Ljung-Box test output dictionary.
        """
        self._require_fit()
        return ljung_box_q(self.residuals(), lags)

    def outlier_diagnostics(self) -> dict:
        """Compute outlier candidate indices.

        Returns
        -------
        dict
            Indices flagged by leverage, r-student, DFFIT, and DFBETA rules.
        """
        self._require_fit()
        n, k = self._X.shape
        return outlier_candidates(self.regression_diagnostics(), n=n, k=k)

    def panel_diagnostics(self) -> dict:
        """Panel-specific diagnostics on transformed residuals."""
        self._require_fit()
        resid = self.residuals()
        return {
            "structure": panel_residual_structure(resid, N=self._N, T=self._T),
            "pesaran_cd": pesaran_cd_test(resid, N=self._N, T=self._T),
        }

    def diagnostics(
        self,
        arch_lags: int | list[int] = 5,
        autocorr_lags: int | list[int] = 10,
    ) -> dict:
        """Return a bundled set of panel model diagnostics.

        Parameters
        ----------
        arch_lags : int or list[int], default=5
            Lag order(s) for ARCH diagnostics.
        autocorr_lags : int or list[int], default=10
            Lag order(s) for Ljung-Box diagnostics.

        Returns
        -------
        dict
            Dictionary containing regression, heteroskedasticity,
            autocorrelation, outlier, and panel-specific diagnostics.
        """
        return {
            "regression": self.regression_diagnostics(),
            "heteroskedasticity": self.heteroskedasticity_diagnostics(arch_lags=arch_lags),
            "autocorrelation": self.autocorrelation_diagnostics(lags=autocorr_lags),
            "outliers": self.outlier_diagnostics(),
            "panel": self.panel_diagnostics(),
        }

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(N={self._N}, T={self._T}, n={n}, "
            f"k={k}, model={self.model}, features={self._feature_names})"
        )


