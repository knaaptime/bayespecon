"""Spatial probit model with spatially dependent regional effects.

Implements a Bayesian binary-response model analogous to MATLAB ``semip_g``:

.. math::
    y_{ij} = 1[z_{ij} > 0],\\quad z = X\\beta + \\Delta a + \\varepsilon,

where regional effects follow

.. math::
    a = \\rho W a + u,\\quad u \\sim \\mathcal{N}(0, \\sigma_a^2 I).

The probit link is used directly via ``P(y=1) = Phi(X\\beta + \\Delta a)``.
"""

from __future__ import annotations

from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from formulaic import model_matrix
from libpysal.graph import Graph


class SpatialProbit:
    """Bayesian spatial probit with regional random effects.

    Parameters
    ----------
    formula : str, optional
        Formula for the binary response model, e.g. ``"y ~ x1 + x2"``.
        Requires ``data`` and ``region_col``.
    data : pandas.DataFrame, optional
        Data source used with ``formula`` mode.
    y : array-like, optional
        Binary dependent variable (0/1), required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Covariate matrix, required in matrix mode.
    W : libpysal.graph.Graph or np.ndarray
        Region-level ``m x m`` spatial weights matrix.
    region_col : str, optional
        Region identifier column in ``data`` (formula mode).
    region_ids : array-like, optional
        Region identifier per observation (matrix mode).
    mobs : array-like, optional
        Region observation counts ``(m,)`` in sorted region order
        (matrix mode alternative to ``region_ids``).
    priors : dict, optional
        Prior overrides.

    Notes
    -----
    This class follows the core ``semip_g`` structure (binary response with
    spatially dependent regional effects). It uses a standard probit link with
    unit observation-level variance and does not currently sample the ``v_i``/
    ``r`` heteroskedastic hierarchy from MATLAB ``semip_g``.

    Priors (``priors`` keys):

    - ``rho_lower``, ``rho_upper``: bounds for ``rho`` (default ``-0.95, 0.95``)
    - ``beta_mu``, ``beta_sigma``: Normal prior for ``beta``
    - ``sigma_a_sigma``: HalfNormal scale for ``sigma_a``
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, np.ndarray]] = None,
        region_col: Optional[str] = None,
        region_ids: Optional[Union[np.ndarray, pd.Series]] = None,
        mobs: Optional[Union[np.ndarray, list[int]]] = None,
        priors: Optional[dict] = None,
    ):
        if W is None:
            raise ValueError("W is required.")

        self.priors = priors or {}
        self._idata: Optional[az.InferenceData] = None

        self._W_dense = self._as_dense_region_W(W)
        self._m = self._W_dense.shape[0]

        if formula is not None:
            if data is None:
                raise ValueError("data is required when using formula mode.")
            if region_col is None:
                raise ValueError("region_col is required when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
            region_series = data[region_col]
            # pd.factorize returns (codes, uniques)
            codes, uniques = pd.factorize(region_series, sort=False)
            self._region_codes = codes.astype(int)
            self._region_names = [str(v) for v in uniques.tolist()]
        elif y is not None and X is not None:
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
            self._region_codes, self._region_names = self._parse_regions(
                nobs=self._X.shape[0],
                region_ids=region_ids,
                mobs=mobs,
            )
        else:
            raise ValueError("Provide either (formula, data, region_col) or (y, X, region_ids/mobs).")

        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")

        if not np.isin(self._y, [0.0, 1.0]).all():
            raise ValueError("y must be binary with values in {0, 1}.")

        if len(np.unique(self._region_codes)) != self._m:
            raise ValueError(
                f"Number of observed regions ({len(np.unique(self._region_codes))}) "
                f"must match W dimension ({self._m})."
            )

    @staticmethod
    def _as_dense_region_W(W: Union[Graph, np.ndarray]) -> np.ndarray:
        if isinstance(W, Graph):
            W_dense = W.sparse.toarray().astype(float)
        else:
            W_dense = np.asarray(W, dtype=float)
        if W_dense.ndim != 2 or W_dense.shape[0] != W_dense.shape[1]:
            raise ValueError("W must be a square region-level matrix.")
        return W_dense

    @staticmethod
    def _parse_formula(formula: str, data: pd.DataFrame):
        lhs_name, rhs = formula.split("~", 1)
        lhs_name = lhs_name.strip()
        rhs = rhs.strip()
        X_mm = model_matrix(rhs, data)
        feature_names = list(X_mm.columns)
        X_arr = np.asarray(X_mm, dtype=np.float64)
        y_arr = np.asarray(data[lhs_name], dtype=np.float64).reshape(-1)
        return y_arr, X_arr, feature_names

    @staticmethod
    def _parse_matrices(y, X):
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        return y_arr, X_arr, feature_names

    @staticmethod
    def _parse_regions(
        nobs: int,
        region_ids: Optional[Union[np.ndarray, pd.Series]],
        mobs: Optional[Union[np.ndarray, list[int]]],
    ) -> tuple[np.ndarray, list[str]]:
        if region_ids is not None:
            ids = np.asarray(region_ids)
            if ids.shape[0] != nobs:
                raise ValueError("region_ids must have one entry per observation.")
            codes, uniques = pd.factorize(ids, sort=False)
            return codes.astype(int), [str(v) for v in uniques.tolist()]

        if mobs is not None:
            counts = np.asarray(mobs, dtype=int).reshape(-1)
            if counts.sum() != nobs:
                raise ValueError("sum(mobs) must equal number of observations.")
            codes = np.repeat(np.arange(len(counts), dtype=int), counts)
            names = [f"region_{i}" for i in range(len(counts))]
            return codes, names

        raise ValueError("Provide either region_ids or mobs in matrix mode.")

    def _model_coords(self) -> dict[str, list[str]]:
        return {
            "coefficient": list(self._feature_names),
            "region": list(self._region_names),
        }

    def _build_pymc_model(self) -> pm.Model:
        k = self._X.shape[1]
        if k == 0:
            raise ValueError("X must contain at least one predictor.")

        rho_lower = self.priors.get("rho_lower", -0.95)
        rho_upper = self.priors.get("rho_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        sigma_a_sigma = self.priors.get("sigma_a_sigma", 2.0)

        W_pt = pt.as_tensor_variable(self._W_dense)
        I_pt = pt.eye(self._m)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma_a = pm.HalfNormal("sigma_a", sigma=sigma_a_sigma)

            a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, dims="region")
            a = pm.Deterministic(
                "a",
                pt.linalg.solve(I_pt - rho * W_pt, sigma_a * a_raw),
                dims="region",
            )

            eta = pt.dot(self._X, beta) + a[self._region_codes]
            p = pm.Deterministic("p", 0.5 * (1.0 + pt.erf(eta / np.sqrt(2.0))))
            pm.Bernoulli("obs", p=p, observed=self._y)

        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior."""
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

    @staticmethod
    def _rename_summary_index(summary_df: pd.DataFrame) -> pd.DataFrame:
        renamed = []
        for label in summary_df.index.astype(str):
            if label.startswith("beta[") and label.endswith("]"):
                renamed.append(label[5:-1])
            elif label.startswith("a[") and label.endswith("]"):
                renamed.append(f"a:{label[2:-1]}")
            else:
                renamed.append(label)
        out = summary_df.copy()
        out.index = renamed
        return out

    def summary(self, var_names: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """Return posterior summary table."""
        self._require_fit()
        summary_df = az.summary(self._idata, var_names=var_names, **kwargs)
        return self._rename_summary_index(summary_df)

    def random_effects_mean(self) -> pd.Series:
        """Return posterior mean regional effects."""
        self._require_fit()
        amean = self._idata.posterior["a"].mean(("chain", "draw")).to_numpy()
        return pd.Series(amean, index=self._region_names, name="a_mean")

    def fitted_probabilities(self) -> np.ndarray:
        """Return posterior mean fitted probabilities for observed data."""
        self._require_fit()
        p = self._idata.posterior["p"].mean(("chain", "draw")).to_numpy()
        return np.asarray(p, dtype=float)
