"""Base class for Bayesian spatial regression models."""

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
    rdiagnose_like,
)


class SpatialModel(ABC):
    """Base class for Bayesian spatial regression models.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula string, e.g. ``"price ~ poverty + rev_rating"``.
        If provided, ``data`` must also be supplied. An intercept is included
        by default; suppress with ``"y ~ x - 1"``.
    data : DataFrame or GeoDataFrame, optional
        Data source when using formula mode.
    y : array-like, optional
        Dependent variable. Required in matrix mode.
    X : array-like, optional
        Predictor matrix. Required in matrix mode. If a DataFrame, column
        names are preserved for labelling.
    W : libpysal.graph.Graph
        Spatial weights graph.
    priors : dict, optional
        Override default priors. Keys depend on the model subclass; see
        each model's docstring for supported keys.
    logdet_method : str
        How to compute ``log|I - rho*W|``. ``"auto"`` / ``"eigenvalue"``
        (default) pre-computes W's eigenvalues once and evaluates O(n) per
        step; ``"exact"`` uses symbolic pytensor det (slow for n > 500);
        ``"grid"`` uses spline interpolation (approximate).
    w_vars : list of str, optional
        Names of X columns to spatially lag. Only relevant for models that
        include ``WX`` terms (SLX, SDM, SDEM and their panel/Tobit variants).
        By default all non-constant columns are lagged. Pass a subset to
        restrict which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Graph] = None,
        priors: Optional[dict] = None,
        logdet_method: str = "auto",
        w_vars: Optional[list] = None,
    ):
        if W is None:
            raise ValueError("W (spatial weights Graph) is required.")

        self.W = W
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self._idata: Optional[az.InferenceData] = None

        if formula is not None:
            if data is None:
                raise ValueError("data must be provided when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
        elif y is not None and X is not None:
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
        else:
            raise ValueError(
                "Provide either (formula, data) or (y, X)."
            )

        # Dense weight matrix for log-det computation
        self._W_dense: np.ndarray = W.sparse.toarray().astype(np.float64)

        self._wx_column_indices = self._spatial_lag_column_indices(self._X, self._feature_names)
        if w_vars is not None:
            unknown = [v for v in w_vars if v not in self._feature_names]
            if unknown:
                raise ValueError(
                    f"w_vars contains names not found in X columns: {unknown}. "
                    f"Available: {self._feature_names}"
                )
            self._wx_column_indices = [
                i for i in self._wx_column_indices if self._feature_names[i] in w_vars
            ]
        self._wx_feature_names = [self._feature_names[i] for i in self._wx_column_indices]

        # Pre-compute spatial lags that subclasses may need
        self._Wy: np.ndarray = np.asarray(W.lag(pd.Series(self._y)), dtype=np.float64)
        if self._wx_column_indices:
            self._WX = np.column_stack([
                np.asarray(W.lag(pd.Series(self._X[:, j])), dtype=np.float64)
                for j in self._wx_column_indices
            ])
        else:
            self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)

    # ------------------------------------------------------------------
    # Input parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_formula(formula: str, data: pd.DataFrame):
        """Parse formula/data inputs into ``y`` and ``X`` arrays.

        Parameters
        ----------
        formula : str
            Wilkinson-style formula containing ``lhs ~ rhs``.
        data : pandas.DataFrame
            Tabular data source referenced by the formula.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Dependent variable, design matrix, and feature names.
        """
        lhs_name, rhs = formula.split("~", 1)
        lhs_name = lhs_name.strip()
        rhs = rhs.strip()

        # Build RHS model matrix (handles intercept, interactions, transforms)
        X_mm = model_matrix(rhs, data)
        feature_names = list(X_mm.columns)
        X_arr = np.asarray(X_mm, dtype=np.float64)

        y_arr = np.asarray(data[lhs_name], dtype=np.float64)
        return y_arr, X_arr, feature_names

    @staticmethod
    def _parse_matrices(y, X):
        """Parse matrix-mode inputs and infer feature names.

        Parameters
        ----------
        y : array-like
            Dependent variable.
        X : array-like or pandas.DataFrame
            Predictor matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Dependent variable, design matrix, and feature names.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        return y_arr, X_arr, feature_names

    @staticmethod
    def _spatial_lag_column_indices(X: np.ndarray, feature_names: list[str]) -> list[int]:
        """Return indices of regressors that should receive spatial lags.

        Constant columns are treated as intercept-like and excluded, which
        avoids adding redundant ``W * intercept`` terms to SLX/Durbin models.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.
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

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model. Subclasses implement this."""

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws : int
            Number of posterior samples per chain (after tuning).
        tune : int
            Number of tuning (burn-in) steps per chain.
        chains : int
            Number of parallel chains.
        target_accept : float
            Target acceptance rate for NUTS.
        random_seed : int, optional
            Seed for reproducibility.
        **sample_kwargs
            Additional keyword arguments forwarded to ``pm.sample``.

        Returns
        -------
        arviz.InferenceData
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

    def spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute posterior direct, indirect, and total impacts.

        For models without a spatial lag on y (SLX, SDEM), the
        derivative matrix is diagonal, so indirect = 0.

        Returns
        -------
        dict with keys "direct", "indirect", "total", each shaped
        (n_features,) containing posterior mean impacts.
        """
        self._require_fit()
        return self._compute_spatial_effects()

    @abstractmethod
    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute model-specific impact measures.

        Returns
        -------
        dict
            Dictionary with direct, indirect, and total effects.
        """

    @abstractmethod
    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Fitted mean vector.
        """

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fit(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

    def _posterior_mean(self, var: str) -> np.ndarray:
        return self._idata.posterior[var].mean(("chain", "draw")).to_numpy()

    def fitted_values(self) -> np.ndarray:
        """Return fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        self._require_fit()
        return self._fitted_mean_from_posterior()

    def residuals(self) -> np.ndarray:
        """Return residuals on the observed scale.

        Returns
        -------
        np.ndarray
            Residual vector ``y - fitted_values``.
        """
        self._require_fit()
        return self._y - self.fitted_values()

    def regression_diagnostics(self) -> dict:
        """Compute OLS-style influence diagnostics.

        Returns
        -------
        dict
            Diagnostics from :func:`bayespreg.diagnostics.rdiagnose_like`.
        """
        self._require_fit()
        resid = self.residuals()
        return rdiagnose_like(self._y, self._X, resid)

    def heteroskedasticity_diagnostics(self, arch_lags: int | list[int] = 5) -> dict:
        """Compute heteroskedasticity diagnostics.

        Parameters
        ----------
        arch_lags : int or list[int], default=5
            Lag order(s) for ARCH test.

        Returns
        -------
        dict
            Dictionary with Breusch-Pagan and ARCH test outputs.
        """
        self._require_fit()
        resid = self.residuals()
        return {
            "bpagan": bpagan_test(resid, self._X),
            "arch": arch_test(resid, arch_lags),
        }

    def autocorrelation_diagnostics(self, lags: int | list[int] = 10) -> dict:
        """Compute residual autocorrelation diagnostics.

        Parameters
        ----------
        lags : int or list[int], default=10
            Lag order(s) for Ljung-Box test.

        Returns
        -------
        dict
            Ljung-Box statistics and p-values.
        """
        self._require_fit()
        resid = self.residuals()
        return ljung_box_q(resid, lags)

    def outlier_diagnostics(self) -> dict:
        """Compute outlier candidate indices.

        Returns
        -------
        dict
            Candidate outlier indices based on leverage/r-student/DFFIT/DFBETA.
        """
        self._require_fit()
        d = self.regression_diagnostics()
        n, k = self._X.shape
        return outlier_candidates(d, n=n, k=k)

    def diagnostics(
        self,
        arch_lags: int | list[int] = 5,
        autocorr_lags: int | list[int] = 10,
    ) -> dict:
        """Return a bundled set of standard model diagnostics.

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
            autocorrelation, and outlier diagnostics.
        """
        return {
            "regression": self.regression_diagnostics(),
            "heteroskedasticity": self.heteroskedasticity_diagnostics(arch_lags=arch_lags),
            "autocorrelation": self.autocorrelation_diagnostics(lags=autocorr_lags),
            "outliers": self.outlier_diagnostics(),
        }

    def __repr__(self) -> str:
        n, k = self._X.shape
        return f"{self.__class__.__name__}(n={n}, k={k}, features={self._feature_names})"


