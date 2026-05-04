"""Bayesian spatial flow (origin-destination) models.

Implements fully Bayesian SAR-type flow models following
:cite:t:`lesage2008SpatialEconometric`.  The observed variable is an
:math:`n \\times n` flow matrix (or its vectorised form), and the weight
structure uses three Kronecker-product matrices:

.. math::

    W_d = I_n \\otimes W, \\quad W_o = W \\otimes I_n, \\quad W_w = W \\otimes W

so that the model is:

.. math::

    y = \\rho_d W_d y + \\rho_o W_o y + \\rho_w W_w y + X\\beta + \\varepsilon,
    \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

where :math:`N = n^2`.

Two variants are provided:

* :class:`SARFlow` — three free ρ parameters with a Dirichlet stability
  constraint (default) or a quadratic-wall potential when competitive effects
  are needed (``restrict_positive=False``).
* :class:`SARFlowSeparable` — constrained :math:`\\rho_w = -\\rho_d \\rho_o`,
  enabling exact eigenvalue-based log-determinant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp
from libpysal.graph import Graph

from ..graph import _validate_graph, flow_trace_blocks, flow_weight_matrices
from ..logdet import (
    _flow_logdet_poly_coeffs,
    compute_flow_traces,
    flow_logdet_numpy,
    flow_logdet_pytensor,
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)
from ..ops import kron_solve_matrix, kron_solve_vec
from ._sampler import (
    enforce_c_backend,
    prepare_compile_kwargs,
    prepare_idata_kwargs,
)


def _build_flow_effect_masks(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (N, n) boolean masks for LeSage origin/destination/intra shocks.

    For each region ``j``, column ``j`` flags the flow indices receiving the
    region-specific shock under the LeSage (2008) effects decomposition:

    - ``dmask[:, j]``: flows whose destination = j and origin ≠ j (β_d shock).
    - ``omask[:, j]``: flows whose origin = j and destination ≠ j (β_o shock).
    - ``imask[:, j]``: the intra flow (j, j) (β_d + β_o shock).

    Flow vec ordering is row-major ``arr[o, d].ravel()`` so flat index
    ``i = o * n + d``.
    """
    N = n * n
    flat = np.arange(N)
    o_idx = flat // n
    d_idx = flat % n
    j = np.arange(n)
    dmask = (d_idx[:, None] == j[None, :]) & (o_idx[:, None] != j[None, :])
    omask = (o_idx[:, None] == j[None, :]) & (d_idx[:, None] != j[None, :])
    imask = (o_idx[:, None] == j[None, :]) & (d_idx[:, None] == j[None, :])
    return dmask, omask, imask


_EFFECT_KEYS = ("origin", "destination", "intra", "network", "total")


def _compute_flow_effects_lesage(
    A_solve,
    dmask: np.ndarray,
    omask: np.ndarray,
    imask: np.ndarray,
    beta_d: np.ndarray,
    beta_o: np.ndarray,
    n: int,
    k_d: int,
    k_o: int | None = None,
    beta_intra: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Compute scalar LeSage / Thomas-Agnan effects for one posterior draw.

    Implements the decomposition of Thomas-Agnan & LeSage (2014, §83.5).  For
    each predictor *p*, two independent shocks are propagated through the
    spatial filter so that origin-side and destination-side effects can be
    reported separately when the design uses different attributes for the
    origin and destination blocks.

    The destination shock places ``β_d^{(p)}`` on every flow whose destination
    equals region ``j`` (off-diagonal) and ``β_d^{(p)} + β_intra^{(p)}`` on the
    intraregional flow ``(j, j)``.  The origin shock places ``β_o^{(p)}`` on
    every flow whose origin equals region ``j``, including ``(j, j)``.  The
    intra block is tied to the destination side because
    :func:`bayespecon.graph.flow_design_matrix` constructs
    ``X_intra = intra_indicator * X_dest``.

    When ``k_d != k_o``, the destination and origin predictors are different
    variables.  Destination-side effects have length ``k_d``, origin-side
    effects have length ``k_o``, and combined effects have length
    ``k_d + k_o`` (concatenated).

    Parameters
    ----------
    A_solve : callable
        Function ``A_solve(rhs)`` that solves ``A x = rhs`` for ``rhs`` of shape
        ``(N, n)`` where ``N = n * n``.  Must accept a 2-D right-hand side.
    dmask, omask, imask : np.ndarray, shape (N, n), dtype bool
        Region-shock masks from :func:`_build_flow_effect_masks`.
    beta_d : np.ndarray, shape (k_d,)
        Destination coefficient vector for one posterior draw.
    beta_o : np.ndarray, shape (k_o,)
        Origin coefficient vector for one posterior draw.
    n : int
        Number of regions.
    k_d : int
        Number of destination-side regional attribute predictors.
    k_o : int or None
        Number of origin-side regional attribute predictors.  If ``None``,
        defaults to ``k_d`` (symmetric case).
    beta_intra : np.ndarray, optional, shape (k_d,)
        Coefficients on the ``intra_*`` design block.  If ``None``, treated as
        zero (legacy behaviour).

    Returns
    -------
    dict
        Combined keys ``"origin"``, ``"destination"``, ``"intra"``,
        ``"network"``, ``"total"`` (each length-``k_d + k_o``) plus the per-side keys
        ``"dest_<eff>"`` (length ``k_d``) and ``"orig_<eff>"`` (length ``k_o``)
        for the same five effects.  The combined values equal the concatenation
        of the corresponding ``dest_*`` and ``orig_*`` arrays.
    """
    if k_o is None:
        k_o = k_d
    N = n * n
    bi = (
        np.zeros(k_d, dtype=np.float64)
        if beta_intra is None
        else np.asarray(beta_intra, dtype=np.float64)
    )

    out: dict[str, np.ndarray] = {}
    for side in ("dest", "orig"):
        k_side = k_d if side == "dest" else k_o
        for eff in _EFFECT_KEYS:
            out[f"{side}_{eff}"] = np.empty(k_side, dtype=np.float64)

    for p in range(k_d):
        bd = float(beta_d[p])
        bint = float(bi[p])

        # Destination-side shock: β_d on flows with destination=j, plus β_intra
        # at (j, j) since X_intra is built from X_dest.
        shock_d = np.zeros((N, n), dtype=np.float64)
        shock_d[dmask] = bd
        shock_d[imask] = bd + bint
        T_d = A_solve(shock_d)
        total_d = T_d.sum() / N
        intra_d = T_d[imask].sum() / N
        origin_d = T_d[omask].sum() / N
        dest_d = T_d[dmask].sum() / N
        out["dest_total"][p] = total_d
        out["dest_intra"][p] = intra_d
        out["dest_origin"][p] = origin_d
        out["dest_destination"][p] = dest_d
        out["dest_network"][p] = total_d - origin_d - dest_d - intra_d

    for p in range(k_o):
        bo = float(beta_o[p])

        # Origin-side shock: β_o on flows with origin=j, including (j, j).
        shock_o = np.zeros((N, n), dtype=np.float64)
        shock_o[omask] = bo
        shock_o[imask] = bo
        T_o = A_solve(shock_o)
        total_o = T_o.sum() / N
        intra_o = T_o[imask].sum() / N
        origin_o = T_o[omask].sum() / N
        dest_o = T_o[dmask].sum() / N
        out["orig_total"][p] = total_o
        out["orig_intra"][p] = intra_o
        out["orig_origin"][p] = origin_o
        out["orig_destination"][p] = dest_o
        out["orig_network"][p] = total_o - origin_o - dest_o - intra_o

    # Combined effects: concatenation of dest and orig (different variables
    # when k_d != k_o, same variables summed when k_d == k_o).
    if k_d == k_o:
        # Symmetric case: sum dest and orig effects (same variables)
        for eff in _EFFECT_KEYS:
            out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]
    else:
        # Asymmetric case: concatenate dest and orig effects (different variables)
        for eff in _EFFECT_KEYS:
            out[eff] = np.concatenate([out[f"dest_{eff}"], out[f"orig_{eff}"]])

    return out


class FlowModel(ABC):
    """Abstract base class for Bayesian spatial flow regression models.

    Unlike :class:`~bayespecon.models.base.SpatialModel`, this class works
    with an :math:`N = n^2` vectorised response and three Kronecker-product
    weight matrices constructed from a single n×n graph.  The API mirrors
    :class:`~bayespecon.models.base.SpatialModel` (``fit``, ``summary``,
    ``inference_data``) but the internals are tailored to the flow structure.

    The model accepts a full O-D design matrix *X* of shape ``(n², p)``,
    typically produced by :func:`~bayespecon.graph.flow_design_matrix` or
    :func:`~bayespecon.graph.flow_design_matrix_with_orig`.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed O-D flow matrix (or its vec-form).  Must be a square
        matrix or a flat vector of length :math:`N = n^2`.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.  Validated by
        :func:`~bayespecon.graph._validate_graph`.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        This is typically produced by
        :func:`~bayespecon.graph.flow_design_matrix` or
        :func:`~bayespecon.graph.flow_design_matrix_with_orig`.
        If a DataFrame, column names are inferred automatically.
    col_names : list[str], optional
        Column labels for *X*.  If *X* is a DataFrame, column names are
        inferred automatically.  Defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns in the design matrix (i.e.,
        the number of destination/origin variable pairs).  When the design
        matrix follows the standard LeSage layout
        ``[intercept, intra_indicator, dest_*, orig_*, intra_*, (dist)]``,
        *k* can be inferred from the column names.  Provide *k* explicitly
        if column names do not follow the ``dest_*``/``orig_*`` convention.
    priors : dict, optional
        Override default priors.  Supported keys vary by subclass.
    logdet_method : str, default "traces"
        How to compute :math:`\\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|`.
        ``"traces"`` uses Barry-Pace stochastic traces with the multinomial
        Kronecker identity (the default and recommended method).
        ``"eigenvalue"``, ``"chebyshev"``, ``"mc_poly"`` (separable
        flow models only) use the Kronecker eigenvalue factorisation.
    restrict_positive : bool, default True
        If True, use a ``pm.Dirichlet`` prior that restricts :math:`\\rho_d,
        \\rho_o, \\rho_w \\geq 0` with :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`.
        This is NUTS-safe and appropriate for most flow applications.
        If False, use three independent ``pm.Uniform(-1, 1)`` priors with a
        differentiable quadratic-wall stability potential.
    miter : int, default 30
        Trace polynomial order for the log-determinant (only used when
        ``logdet_method="traces"``).  Higher values improve accuracy at
        the cost of more precomputation.
    titer : int, default 800
        Geometric tail cutoff for the log-determinant series.
    trace_riter : int, default 50
        Number of Monte Carlo probes for trace estimation.
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), the destination and origin design blocks are
        compared and symmetry is auto-detected.  Set explicitly to override
        the heuristic — for example, when using
        :func:`~bayespecon.graph.flow_design_matrix_with_orig` with distinct
        attributes for the origin and destination sides.  Controls the default
        behaviour of :meth:`spatial_effects` when ``mode="auto"``.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series],
        G: Graph,
        X: Union[np.ndarray, pd.DataFrame],
        col_names: Optional[list] = None,
        k: Optional[int] = None,
        priors: Optional[dict] = None,
        logdet_method: str = "traces",
        restrict_positive: bool = True,
        miter: int = 30,
        titer: int = 800,
        trace_riter: int = 50,
        trace_seed: Optional[int] = None,
        symmetric_xo_xd: Optional[bool] = None,
    ):
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.restrict_positive = restrict_positive
        self.miter = miter
        self.titer = titer
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None
        self._approximation = None

        # Validate and extract the n×n weight matrix
        self._W_sparse: sp.csr_matrix = _validate_graph(G)
        self._n: int = self._W_sparse.shape[0]
        self._N: int = self._n * self._n

        # Validate y
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 2:
            if y_arr.shape != (self._n, self._n):
                raise ValueError(
                    f"y matrix must be ({self._n}, {self._n}), got {y_arr.shape}."
                )
            self._y_vec = y_arr.ravel()
        elif y_arr.ndim == 1:
            if len(y_arr) != self._N:
                raise ValueError(
                    f"y vector must have length N={self._N} (= n²), got {len(y_arr)}."
                )
            self._y_vec = y_arr
        else:
            raise ValueError("y must be a 1-D or 2-D array.")

        # Validate X and build design matrix
        if isinstance(X, pd.DataFrame):
            if col_names is None:
                col_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[0] != self._N:
            raise ValueError(
                f"X must have {self._N} rows (= n² = {self._n}²), got {X_arr.shape[0]}."
            )

        self._X_design: np.ndarray = X_arr  # (N, p)
        if col_names is not None:
            self._feature_names: list[str] = list(col_names)
        elif X_arr.shape[1] == 0:
            self._feature_names = []
        else:
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        # Infer k_d and k_o (number of regional attribute columns) for effects computation.
        # Standard LeSage layout: [intercept, intra_indicator, dest_*, orig_*, intra_*, (dist)]
        if k is not None:
            self._k: int = k
            self._k_d: int = k
            self._k_o: int = k
        else:
            dest_cols = [
                name for name in self._feature_names if name.startswith("dest_")
            ]
            orig_cols = [
                name for name in self._feature_names if name.startswith("orig_")
            ]
            self._k_d = len(dest_cols)
            self._k_o = len(orig_cols)
            self._k = self._k_d  # backward compat alias
            if self._k_d == 0 and self._k_o == 0:
                # Fallback: cannot infer k from column names; effects decomposition
                # will not be available.  Set k=0 as a sentinel.
                self._k = 0

        # Locate β_intra slice (Thomas-Agnan & LeSage 2014, §83.4): coefficients
        # on the `intra_*` block contribute to the intraregional shock.  When
        # the design lacks these columns the intra contribution is zero.
        if self._k_d > 0:
            intra_cols = [
                i
                for i, name in enumerate(self._feature_names)
                if name.startswith("intra_")
            ]
            self._intra_idx: Optional[np.ndarray] = (
                np.asarray(intra_cols, dtype=np.int64) if intra_cols else None
            )
        else:
            self._intra_idx = None

        # Detect whether the destination and origin design blocks are
        # identical (the symmetric Xo = Xd case).  When asymmetric the
        # Thomas-Agnan & LeSage (2014, §83.5.2) shortcut of summing dest and
        # orig effects is not appropriate, and `spatial_effects(mode="auto")`
        # falls back to reporting both sides separately.
        if (
            symmetric_xo_xd is None
            and self._k_d > 0
            and self._k_d == self._k_o
            and X_arr.shape[1] >= 2 + self._k_d + self._k_o
        ):
            dest_block = X_arr[:, 2 : 2 + self._k_d]
            orig_block = X_arr[:, 2 + self._k_d : 2 + self._k_d + self._k_o]
            self._symmetric_xo_xd: bool = bool(np.array_equal(dest_block, orig_block))
        else:
            self._symmetric_xo_xd = (
                bool(symmetric_xo_xd)
                if symmetric_xo_xd is not None
                else (self._k_d == self._k_o)
            )

        # Pre-compute logdet data for separable constraint: log|Lo⊗Ld| = n*f(ρ_d) + n*f(ρ_o).
        # Also keep _W_eigs for backward compatibility.
        self._W_eigs: Optional[np.ndarray] = None
        self._separable_logdet_fn = None
        self._separable_logdet_numpy_fn = None
        _SEPARABLE_METHODS = {"eigenvalue", "chebyshev", "mc_poly"}
        if logdet_method in _SEPARABLE_METHODS:
            self._separable_logdet_fn = make_flow_separable_logdet(
                self._W_sparse,
                self._n,
                method=logdet_method,
                miter=miter,
                riter=trace_riter,
                random_state=trace_seed,
            )
            self._separable_logdet_numpy_fn = make_flow_separable_logdet_numpy(
                self._W_sparse,
                self._n,
                method=logdet_method,
                miter=miter,
                riter=trace_riter,
                random_state=trace_seed,
            )
            if logdet_method == "eigenvalue":
                self._W_eigs = np.linalg.eigvals(
                    self._W_sparse.toarray().astype(np.float64)
                ).real

        # Pre-compute spatial lags: Wd_y, Wo_y, Ww_y
        wms = flow_weight_matrices(G)
        self._Wd_y: np.ndarray = wms["destination"] @ self._y_vec
        self._Wo_y: np.ndarray = wms["origin"] @ self._y_vec
        self._Ww_y: np.ndarray = wms["network"] @ self._y_vec

        # Keep N×N sparse weight matrices for effects computation
        self._Wd: sp.csr_matrix = wms["destination"]
        self._Wo: sp.csr_matrix = wms["origin"]
        self._Ww: sp.csr_matrix = wms["network"]

        # Cache region-shock masks for LeSage effects decomposition.
        self._dmask, self._omask, self._imask = _build_flow_effect_masks(self._n)

        # Cache the symmetric 3x3 Kronecker trace matrix used by Bayesian
        # LM diagnostics: T[i,j] = tr(W_i' W_j) + tr(W_i W_j) for
        # (W_d, W_o, W_w).  Computed in O(nnz) from the base n x n graph.
        self._T_flow_traces: np.ndarray = flow_trace_blocks(self._W_sparse)

        # Pre-compute flow log-det traces (only for "traces" method)
        if logdet_method == "traces":
            self._traces: np.ndarray = compute_flow_traces(
                self._W_sparse, miter=miter, riter=trace_riter, random_state=trace_seed
            )
            result = _flow_logdet_poly_coeffs(self._traces, self._n, miter)
            (
                self._poly_a,
                self._poly_b,
                self._poly_c,
                self._poly_coeffs,
                self._miter_a,
                self._miter_b,
                self._miter_c,
                self._miter_coeffs,
            ) = result
        else:
            self._traces = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return the PyMC model.  Implemented by subclasses."""

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior spatial effects.  Implemented by subclasses."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _posterior_var_names(
        self,
        model: pm.Model,
        *,
        store_lambda: bool,
    ) -> list[str]:
        names = [rv.name for rv in model.free_RVs]
        names.extend(
            var.name
            for var in model.deterministics
            if store_lambda or var.name != "lambda"
        )
        return list(dict.fromkeys(name for name in names if name is not None))

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws : int, default 2000
            Number of posterior samples per chain (after tuning).
        tune : int, default 1000
            Number of tuning (warm-up) steps per chain.
        chains : int, default 4
            Number of parallel chains.
        target_accept : float, default 0.9
            Target acceptance rate for NUTS.
        random_seed : int, optional
            Seed for reproducibility.
        store_lambda : bool, default False
            If True, include the high-dimensional fitted mean ``lambda`` in the
            stored posterior. Leaving this False reduces memory and conversion
            overhead for Poisson flow models.
        idata_kwargs : dict, optional
            Forwarded to ``pm.sample``.  Defaults to
            ``{"log_likelihood": True}`` so that ``az.loo`` / ``az.waic`` /
            ``az.compare`` work out of the box; for SAR flow variants the
            captured Gaussian log-likelihood is post-processed to add the
            Jacobian contribution from ``log|I_N - rho_d W_d - rho_o W_o
            - rho_w W_w|``.
        **sample_kwargs
            Additional keyword arguments forwarded to ``pm.sample``.

        Returns
        -------
        arviz.InferenceData
        """
        idata_kwargs = dict(idata_kwargs) if idata_kwargs else {}
        idata_kwargs.setdefault("log_likelihood", True)
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
        nuts_sampler = enforce_c_backend(
            nuts_sampler,
            requires_c_backend=getattr(self, "_requires_c_backend", False),
            model_name=type(self).__name__,
        )

        model = self._build_pymc_model()
        self._pymc_model = model
        self._approximation = None
        if "var_names" not in sample_kwargs and not store_lambda:
            sample_kwargs["var_names"] = self._posterior_var_names(
                model,
                store_lambda=False,
            )
        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        sample_kwargs = prepare_compile_kwargs(sample_kwargs, nuts_sampler)
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs=idata_kwargs,
                nuts_sampler=nuts_sampler,
                **sample_kwargs,
            )
        if compute_log_likelihood:
            self._attach_complete_log_likelihood(self._idata)
        return self._idata

    def fit_approx(
        self,
        draws: int = 2000,
        n: int = 10000,
        method: str = "advi",
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        compute_log_likelihood: bool = True,
        **fit_kwargs,
    ) -> az.InferenceData:
        """Fit a variational approximation and return posterior draws.

        Parameters
        ----------
        draws : int, default 2000
            Number of samples to draw from the fitted approximation.
        n : int, default 10000
            Number of optimisation iterations for ``pm.fit``.
        method : {"advi", "fullrank_advi"}, default "advi"
            Variational inference family to fit.
        random_seed : int, optional
            Seed for optimisation and posterior sampling.
        store_lambda : bool, default False
            If True, keep the high-dimensional fitted mean ``lambda`` in the
            posterior draws.
        compute_log_likelihood : bool, default True
            If True, compute pointwise log-likelihood after sampling and
            attach to the InferenceData (with Jacobian correction for SAR
            flow variants), enabling ``az.loo`` / ``az.waic``.
        **fit_kwargs
            Additional keyword arguments forwarded to ``pm.fit``.
        """
        method = method.lower()
        if method not in {"advi", "fullrank_advi"}:
            raise ValueError("fit_approx method must be 'advi' or 'fullrank_advi'.")

        model = self._build_pymc_model()
        self._pymc_model = model
        with model:
            self._approximation = pm.fit(
                n=n,
                method=method,
                random_seed=random_seed,
                **fit_kwargs,
            )
            self._idata = self._approximation.sample(
                draws=draws,
                random_seed=random_seed,
                return_inferencedata=True,
            )
            if compute_log_likelihood:
                pm.compute_log_likelihood(
                    self._idata,
                    extend_inferencedata=True,
                    progressbar=False,
                )

        if (
            not store_lambda
            and self._idata is not None
            and hasattr(self._idata, "posterior")
            and "lambda" in self._idata.posterior.data_vars
        ):
            self._idata.posterior = self._idata.posterior.drop_vars("lambda")

        if compute_log_likelihood:
            self._attach_complete_log_likelihood(self._idata)

        return self._idata

    @property
    def inference_data(self) -> Optional[az.InferenceData]:
        """Return ArviZ InferenceData from the most recent fit, or None."""
        return self._idata

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model used for the most recent fit, or None."""
        return self._pymc_model

    @property
    def approximation(self):
        """Return the most recent PyMC variational approximation, if any."""
        return self._approximation

    def summary(
        self,
        var_names: Optional[list] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Return posterior summary table via ArviZ.

        Parameters
        ----------
        var_names : list, optional
            Variable names to include.  Defaults to all parameters.
        **kwargs
            Additional keyword arguments forwarded to ``az.summary``.

        Returns
        -------
        pandas.DataFrame
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        return az.summary(self._idata, var_names=var_names, **kwargs)

    # ------------------------------------------------------------------
    # Class-level registry of applicable Bayesian LM specification tests.
    # Mirrors the SpatialModel pattern; subclasses populate this list.
    # ------------------------------------------------------------------
    _spatial_diagnostics_tests: list[tuple] = []

    def spatial_diagnostics(self) -> pd.DataFrame:
        """Run Bayesian LM specification tests for flow models.

        Iterates over the class-level ``_spatial_diagnostics_tests`` registry
        and returns a tidy DataFrame with one row per test.  See
        :meth:`bayespecon.models.base.SpatialModel.spatial_diagnostics` for
        the column schema.

        Raises
        ------
        RuntimeError
            If the model has not been fit yet.
        """
        from .base import SpatialModel

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        return SpatialModel._run_lm_diagnostics(self, self._spatial_diagnostics_tests)

    def spatial_diagnostics_decision(
        self, alpha: float = 0.05, format: str = "graphviz"
    ) -> Any:
        """Return a model-selection decision from Bayesian LM test results.

        Walks the flow decision tree using Bayesian p-values from
        :meth:`spatial_diagnostics` and recommends either ``OLSFlow`` (no
        spatial dependence detected) or ``SARFlow`` (at least one direction
        is significant).

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the Bayesian p-values.
        format : {"graphviz", "ascii", "model"}, default "graphviz"
            Output format.  ``"model"`` returns the recommended model name
            string.  ``"ascii"`` returns an indented box-drawing tree.
            ``"graphviz"`` returns a :class:`graphviz.Digraph` (with ASCII
            fallback if graphviz is not installed).

        Returns
        -------
        str or graphviz.Digraph
        """
        from ..diagnostics import _decision_trees as _dt

        diag = self.spatial_diagnostics()
        model_type = self.__class__.__name__

        def _sig(test_name: str) -> bool:
            if test_name not in diag.index:
                return False
            pval = diag.loc[test_name, "p_value"]
            return not np.isnan(pval) and pval < alpha

        spec = _dt.get_flow_spec(model_type)
        decision, path = _dt.evaluate(spec, sig_lookup=_sig)

        p_values: dict[str, float] = {}
        for test_name in diag.index:
            pv = diag.loc[test_name, "p_value"]
            if not np.isnan(pv):
                p_values[str(test_name)] = float(pv)

        return _dt.render(
            spec,
            path,
            decision,
            p_values=p_values,
            alpha=alpha,
            fmt=format,
            title=f"{model_type} decision tree (alpha={alpha})",
        )

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        """Return PyMC coordinate labels for named dimensions."""
        coords: dict = {"coefficient": self._feature_names}
        if extra:
            coords.update(extra)
        return coords

    # ------------------------------------------------------------------
    # Pointwise log-likelihood (with Jacobian correction for SAR variants)
    # ------------------------------------------------------------------

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        """Per-draw :math:`\\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|`.

        Returns ``None`` (the default) when no Jacobian correction is
        required — for example, OLS / Poisson flow baselines (``A = I_N``)
        and the Poisson SAR variants (the ``pm.Poisson("obs", ...)``
        log-likelihood already captured by PyMC is the appropriate
        pointwise density on observed counts).

        Subclasses with a Gaussian observation model and a
        ``pm.Potential("jacobian", ...)`` term must override this to return
        an array of shape ``(n_draws,)`` with the per-draw log-determinant.
        """
        return None

    def _attach_complete_log_likelihood(self, idata) -> None:
        """Add Jacobian contribution to the pointwise log-likelihood.

        ``pm.sample(idata_kwargs={"log_likelihood": True})`` only captures
        observed-RV log densities, so the ``pm.Potential("jacobian", ...)``
        contribution from ``log|I_N - rho_d W_d - rho_o W_o - rho_w W_w|``
        is missing for SAR flow variants.  This helper recomputes the
        complete pointwise log-likelihood and replaces the captured group.
        """
        if idata is None or not hasattr(idata, "log_likelihood"):
            return
        if "obs" not in idata.log_likelihood.data_vars:
            return

        jacobian_draws = self._compute_jacobian_log_det(idata.posterior)
        if jacobian_draws is None:
            return

        import xarray as xr

        ll_da = idata.log_likelihood["obs"]
        n_chains = ll_da.sizes["chain"]
        n_draws_per_chain = ll_da.sizes["draw"]
        n_obs = int(np.prod(ll_da.shape[2:]))

        ll_array = ll_da.values.reshape(n_chains * n_draws_per_chain, n_obs)
        jacobian_draws = np.asarray(jacobian_draws, dtype=np.float64).reshape(-1)
        if jacobian_draws.shape[0] != ll_array.shape[0]:
            raise RuntimeError(
                "Posterior draw count does not match log-likelihood shape: "
                f"{jacobian_draws.shape[0]} vs {ll_array.shape[0]}."
            )

        ll_array = ll_array + jacobian_draws[:, None] / n_obs
        ll_array = ll_array.reshape(n_chains, n_draws_per_chain, n_obs)

        new_da = xr.DataArray(ll_array, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": new_da})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_A(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
    ) -> sp.csr_matrix:
        """Assemble A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww (sparse N×N)."""
        I_N = sp.eye(self._N, format="csr", dtype=np.float64)
        return I_N - rho_d * self._Wd - rho_o * self._Wo - rho_w * self._Ww

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    def spatial_effects(
        self,
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
        mode: str = "auto",
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Summarise posterior origin/destination/intra/network/total effects.

        Wraps :meth:`_compute_spatial_effects_posterior` to produce a tidy
        DataFrame indexed by predictor with posterior means, credible-interval
        bounds, and Bayesian p-values for each effect type (origin,
        destination, intra, network, total).  Following Thomas-Agnan & LeSage
        (2014, §83.5.2), when destination and origin design blocks differ the
        decomposition is reported separately for shocks applied to each side.

        Parameters
        ----------
        draws : int, optional
            Maximum number of posterior draws to use.  Defaults to all.
        return_posterior_samples : bool, default False
            If True, also return the underlying posterior-draw arrays.
        ci : float, default 0.95
            Credible-interval coverage.
        mode : {"auto", "combined", "separate"}, default "auto"
            Controls whether destination- and origin-side effects are summed
            or reported separately.  ``"auto"`` collapses to combined when
            the destination and origin design blocks are identical
            (``self._symmetric_xo_xd``) and reports both sides otherwise.
            ``"combined"`` always sums; ``"separate"`` always reports both.

        Returns
        -------
        pandas.DataFrame, or (DataFrame, dict)
            Long-format summary indexed by ``(predictor, side, effect)`` where
            ``side`` is one of ``"combined"``, ``"dest"``, ``"orig"``.
        """
        from ..diagnostics.spatial_effects import _compute_bayesian_pvalue

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        if self._k == 0:
            raise RuntimeError(
                "Cannot compute spatial effects: no `dest_*` columns detected "
                "in the design matrix.  Pass `k=` explicitly when constructing "
                "the model."
            )
        if mode not in {"auto", "combined", "separate"}:
            raise ValueError(
                f"mode must be 'auto', 'combined', or 'separate'; got {mode!r}."
            )

        posterior = self._compute_spatial_effects_posterior(draws=draws)

        if mode == "auto":
            effective_mode = "combined" if self._symmetric_xo_xd else "separate"
        else:
            effective_mode = mode

        if effective_mode == "combined":
            display = [("combined", eff) for eff in _EFFECT_KEYS]
        else:
            display = [(side, eff) for side in ("dest", "orig") for eff in _EFFECT_KEYS]

        # Predictor names: prefer dest_* and orig_* labels stripped of the prefix.
        dest_feature_names = [
            name[len("dest_") :] if name.startswith("dest_") else name
            for name in self._feature_names
            if name.startswith("dest_")
        ][: self._k_d]
        if len(dest_feature_names) != self._k_d:
            dest_feature_names = [f"x{i}" for i in range(self._k_d)]

        orig_feature_names = [
            name[len("orig_") :] if name.startswith("orig_") else name
            for name in self._feature_names
            if name.startswith("orig_")
        ][: self._k_o]
        if len(orig_feature_names) != self._k_o:
            orig_feature_names = [f"y{i}" for i in range(self._k_o)]

        # For combined mode: when k_d == k_o, combined effects are the sum
        # of dest and orig (same variables), so use dest names.
        # When k_d != k_o, combined effects are concatenated (different variables).
        if self._k_d == self._k_o:
            feature_names = dest_feature_names
        else:
            feature_names = dest_feature_names + orig_feature_names

        alpha = (1.0 - ci) / 2.0
        rows = []
        for side, effect_name in display:
            key = effect_name if side == "combined" else f"{side}_{effect_name}"
            samples = posterior[key]
            means = samples.mean(axis=0)
            lower = np.quantile(samples, alpha, axis=0)
            upper = np.quantile(samples, 1.0 - alpha, axis=0)
            pvals = _compute_bayesian_pvalue(samples)
            if side == "combined":
                fnames = feature_names
            elif side == "dest":
                fnames = dest_feature_names
            else:
                fnames = orig_feature_names
            for j, fname in enumerate(fnames):
                rows.append(
                    {
                        "predictor": fname,
                        "side": side,
                        "effect": effect_name,
                        "mean": float(means[j]),
                        "ci_lower": float(lower[j]),
                        "ci_upper": float(upper[j]),
                        "bayes_pvalue": float(pvals[j]),
                    }
                )

        df = pd.DataFrame(rows).set_index(["predictor", "side", "effect"])
        if return_posterior_samples:
            return df, posterior
        return df

    def _simulate_y_rep(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw a single posterior-predictive replicate.

        Default implementation: Gaussian SAR flow,
        ``y_rep = A^{-1} (X β + σ ε)`` with ``ε ~ N(0, I_N)``.
        Subclasses (PoissonSARFlow, PoissonSARFlowSeparable) override this.
        """
        A = self._assemble_A(rho_d, rho_o, rho_w)
        Xb = self._X_design @ beta
        eps = rng.normal(scale=float(sigma), size=self._N) if sigma is not None else 0.0
        rhs = Xb + eps
        return sp.linalg.spsolve(A, rhs)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive samples ``y_rep``.

        For each (subsampled) posterior draw, simulates a new flow vector
        ``y_rep`` from the implied data-generating process by solving the
        sparse system ``A(rho) y_rep = X β + ε`` (Gaussian) or
        ``y_rep ~ Poisson(exp(A^{-1} X β))`` (Poisson variants).

        Parameters
        ----------
        n_draws : int, optional
            Number of posterior draws to use.  Defaults to all available.
        random_seed : int, optional
            Seed for the noise/Poisson sampler.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_draws, N)`` with posterior-predictive flows.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        rho_d = post["rho_d"].values.reshape(-1)
        rho_o = post["rho_o"].values.reshape(-1)
        rho_w = post["rho_w"].values.reshape(-1)
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = len(rho_d)
        if n_draws is not None:
            total = min(int(n_draws), total)
            rho_d = rho_d[:total]
            rho_o = rho_o[:total]
            rho_w = rho_w[:total]
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep(
                float(rho_d[g]),
                float(rho_o[g]),
                float(rho_w[g]),
                beta_draws[g],
                sigma_g,
                rng,
            )
        return out


# ---------------------------------------------------------------------------
# Model 1: SARFlow — unrestricted 3-ρ
# ---------------------------------------------------------------------------


class SARFlow(FlowModel):
    """Bayesian SAR flow model with three free spatial autoregressive parameters.

    .. math::

        y = \\rho_d W_d y + \\rho_o W_o y + \\rho_w W_w y + X\\beta + \\varepsilon,
        \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

    where :math:`W_d = I_n \\otimes W`, :math:`W_o = W \\otimes I_n`,
    :math:`W_w = W \\otimes W`.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form. Must be
        a square matrix or a flat vector of length :math:`N = n^2`.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        Typically produced by :func:`~bayespecon.graph.flow_design_matrix`
        or :func:`~bayespecon.graph.flow_design_matrix_with_orig`.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted;
        otherwise defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns (destination/origin variable
        pairs). Inferred from ``dest_*``/``orig_*`` column names when the
        standard LeSage layout is used.
    logdet_method : str, default "traces"
        Log-determinant method. Only ``"traces"`` (Barry-Pace stochastic
        traces with the Kronecker identity) is supported for this model.
    restrict_positive : bool, default True
        If True, use ``pm.Dirichlet("rho_simplex", a=ones(4))`` to enforce
        :math:`\\rho_d, \\rho_o, \\rho_w \\geq 0` and
        :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`. NUTS-safe via the
        stick-breaking bijection and appropriate when competitive
        (negative) spillovers are not expected. If False, three
        independent ``pm.Uniform(rho_lower, rho_upper)`` priors are used
        together with a differentiable quadratic-wall stability potential.
    miter : int, default 30
        Trace polynomial order for the log-determinant.
    titer : int, default 800
        Geometric tail cutoff for the log-determinant series.
    trace_riter : int, default 50
        Number of Monte Carlo probes for trace estimation.
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected. Set explicitly to override
        the heuristic.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``rho_lower`` : float, default -1.0 — Lower bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
        - ``rho_upper`` : float, default 1.0 — Upper bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
    """

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_flow_dest_test"],
            ).bayesian_robust_lm_flow_dest_test(m),
            "Robust-LM-Flow-Dest",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_flow_orig_test"],
            ).bayesian_robust_lm_flow_orig_test(m),
            "Robust-LM-Flow-Orig",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_robust_lm_flow_network_test"],
            ).bayesian_robust_lm_flow_network_test(m),
            "Robust-LM-Flow-Network",
        ),
    ]

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y_vec.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                # Dirichlet encodes ρ_d, ρ_o, ρ_w ≥ 0 and ρ_d+ρ_o+ρ_w ≤ 1 exactly.
                # rho_simplex[3] is the slack variable (not a spatial parameter).
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                # Quadratic-wall stability potential (differentiable everywhere)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            # Jacobian: log|I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww|
            pm.Potential(
                "jacobian",
                flow_logdet_pytensor(
                    rho_d,
                    rho_o,
                    rho_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        rho_d = posterior["rho_d"].values.reshape(-1)
        rho_o = posterior["rho_o"].values.reshape(-1)
        rho_w = posterior["rho_w"].values.reshape(-1)
        return flow_logdet_numpy(
            rho_d,
            rho_o,
            rho_w,
            self._poly_a,
            self._poly_b,
            self._poly_c,
            self._poly_coeffs,
            self._miter_a,
            self._miter_b,
            self._miter_c,
            self._miter_coeffs,
            self.miter,
            self.titer,
        )

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior origin / destination / intra / network / total effects.

        Implements the LeSage (2008) effects decomposition: for each posterior
        draw of :math:`(\\rho_d, \\rho_o, \\rho_w, \\beta_d, \\beta_o)` and each
        regional predictor *p*, builds an :math:`N\\times n` shock matrix whose
        column ``j`` contains :math:`\\beta_d^{(p)}` at flows with destination
        ``j``, :math:`\\beta_o^{(p)}` at flows with origin ``j``, and
        :math:`\\beta_d^{(p)} + \\beta_o^{(p)}` at the intra flow ``(j, j)``.
        The system :math:`A\\,T = \\text{shock}` is solved with one sparse
        :math:`LU` factorisation per draw (re-used for all ``n`` columns and all
        ``k`` predictors), and scalar effects are obtained by averaging
        :math:`T` over the appropriate masks.  Mirrors LeSage's
        ``calc_effects.m`` reference implementation.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws to use.  Defaults to all draws.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``"origin"``, ``"destination"``, ``"intra"``, ``"network"``,
            ``"total"``.  Each value has shape ``(draws, k)``.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        n = self._n
        k_d = self._k_d
        k_o = self._k_o

        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        rho_w_draws = idata.posterior["rho_w"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            rho_w_draws = rho_w_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        out: dict[str, np.ndarray] = {}
        for side in ("dest", "orig"):
            k_side = k_d if side == "dest" else k_o
            for eff in _EFFECT_KEYS:
                out[f"{side}_{eff}"] = np.zeros(
                    (n_draws_total, k_side), dtype=np.float64
                )
        k_combined = k_d + k_o if k_d != k_o else k_d
        for eff in _EFFECT_KEYS:
            out[eff] = np.zeros((n_draws_total, k_combined), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            rw = float(rho_w_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k_d]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k_o]
            beta_intra_vec = (
                beta_draws[idx, intra_start : intra_start + k_d] if has_intra else None
            )

            A = self._assemble_A(rd, ro, rw).tocsc()
            lu = sp.linalg.splu(A)

            def _solve(rhs: np.ndarray, _lu=lu) -> np.ndarray:
                return _lu.solve(rhs)

            res = _compute_flow_effects_lesage(
                _solve,
                self._dmask,
                self._omask,
                self._imask,
                beta_d_vec,
                beta_o_vec,
                n,
                k_d,
                k_o=k_o,
                beta_intra=beta_intra_vec,
            )
            for key, arr in res.items():
                out[key][idx, : len(arr)] = arr

        return out


# ---------------------------------------------------------------------------
# Model 2: SARFlowSeparable — constrained ρ_w = −ρ_d·ρ_o
# ---------------------------------------------------------------------------


class SARFlowSeparable(FlowModel):
    """Bayesian separable SAR flow model with ρ_w = −ρ_d · ρ_o.

    The separability constraint :math:`\\rho_w = -\\rho_d \\rho_o` reduces
    the flow weight matrix to a Kronecker structure whose log-determinant
    factors as:

    .. math::

        \\log|I_N - \\rho_d W_d - \\rho_o W_o + \\rho_d \\rho_o W_w|
        = n \\log|I_n - \\rho_d W| + n \\log|I_n - \\rho_o W|

    enabling exact O(n) log-det evaluation via eigenvalues — no trace
    estimation required.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form. Must be
        a square matrix or a flat vector of length :math:`N = n^2`.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        Typically produced by :func:`~bayespecon.graph.flow_design_matrix`
        or :func:`~bayespecon.graph.flow_design_matrix_with_orig`.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted;
        otherwise defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns (destination/origin variable
        pairs). Inferred from ``dest_*``/``orig_*`` column names when the
        standard LeSage layout is used.
    logdet_method : {"eigenvalue", "chebyshev", "mc_poly"}, default "eigenvalue"
        Method for the Kronecker-factored log-determinant
        :math:`\\log|I_n - \\rho_d W| + \\log|I_n - \\rho_o W|`.
        ``"eigenvalue"`` is exact; ``"chebyshev"`` and ``"mc_poly"`` are
        polynomial approximations for large *n*.
    miter : int, default 30
        Polynomial / approximation order (used by ``"chebyshev"`` /
        ``"mc_poly"`` methods).
    titer : int, default 800
        Geometric tail cutoff for series-based log-determinant variants.
    trace_riter : int, default 50
        Number of Monte Carlo probes (used by ``"mc_poly"``).
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected. Set explicitly to override
        the heuristic.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``rho_lower`` : float, default -0.999 — Lower bound of Uniform prior on ``rho_d`` and ``rho_o``.
        - ``rho_upper`` : float, default 0.999 — Upper bound of Uniform prior on ``rho_d`` and ``rho_o``.

    Notes
    -----
    The ``restrict_positive`` argument inherited from :class:`FlowModel`
    has no effect on this class — separable variants always use Uniform
    priors on the individual :math:`\rho` components.
    """

    def __init__(self, y, G, X, **kwargs):
        method = kwargs.pop("logdet_method", "eigenvalue")
        _VALID = {"eigenvalue", "chebyshev", "mc_poly"}
        if method not in _VALID:
            raise ValueError(
                f"SARFlowSeparable logdet_method must be one of {sorted(_VALID)}; "
                f"got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, G, X, **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SARFlowSeparable requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y_vec.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            # rho_w is deterministic: -rho_d * rho_o  (must appear in posterior)
            rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            # Jacobian: n*log|I_n - rho_d*W| + n*log|I_n - rho_o*W|
            # factorisation holds exactly for the separable constraint.
            pm.Potential(
                "jacobian",
                self._separable_logdet_fn(rho_d, rho_o),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        rho_d = np.asarray(posterior["rho_d"].values.reshape(-1), dtype=np.float64)
        rho_o = np.asarray(posterior["rho_o"].values.reshape(-1), dtype=np.float64)
        if self._separable_logdet_numpy_fn is None:
            raise RuntimeError(
                "Missing separable numeric logdet evaluator. "
                "Initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        return self._separable_logdet_numpy_fn(rho_d, rho_o)

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior effects using Kronecker-factored solve.

        Implements the LeSage (2008) effects decomposition (see
        :meth:`SARFlow._compute_spatial_effects_posterior`) but exploits
        :math:`A = L_o \\otimes L_d` to replace the
        :math:`N\\times N` sparse factorisation with two :math:`n\\times n`
        sparse solves per predictor via
        :func:`~bayespecon.ops.kron_solve_matrix`.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws to use.  Defaults to all.

        Returns
        -------
        dict[str, np.ndarray]
            Same keys and shapes as :meth:`SARFlow._compute_spatial_effects_posterior`.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        n = self._n
        k_d = self._k_d
        k_o = self._k_o
        W = self._W_sparse.tocsr()
        I_n = sp.eye(n, format="csr", dtype=np.float64)

        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        out: dict[str, np.ndarray] = {}
        for side in ("dest", "orig"):
            k_side = k_d if side == "dest" else k_o
            for eff in _EFFECT_KEYS:
                out[f"{side}_{eff}"] = np.zeros(
                    (n_draws_total, k_side), dtype=np.float64
                )
        k_combined = k_d + k_o if k_d != k_o else k_d
        for eff in _EFFECT_KEYS:
            out[eff] = np.zeros((n_draws_total, k_combined), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k_d]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k_o]
            beta_intra_vec = (
                beta_draws[idx, intra_start : intra_start + k_d] if has_intra else None
            )

            Ld = (I_n - rd * W).tocsr()
            Lo = (I_n - ro * W).tocsr()

            def _solve(rhs: np.ndarray, _Lo=Lo, _Ld=Ld, _n=n) -> np.ndarray:
                return kron_solve_matrix(_Lo, _Ld, rhs, _n)

            res = _compute_flow_effects_lesage(
                _solve,
                self._dmask,
                self._omask,
                self._imask,
                beta_d_vec,
                beta_o_vec,
                n,
                k_d,
                k_o=k_o,
                beta_intra=beta_intra_vec,
            )
            for key, arr in res.items():
                out[key][idx, : len(arr)] = arr

        return out


# ---------------------------------------------------------------------------
# Model 3: PoissonSARFlow — Poisson response, unrestricted 3-rho
# ---------------------------------------------------------------------------


class PoissonSARFlow(SARFlow):
    """Bayesian SAR flow model with a Poisson observation distribution.

    Models origin-destination flow **counts**
    :math:`y_{ij} \\in \\mathbb{N}_0` as:

    .. math::

        y_{ij} \\sim \\operatorname{Poisson}(\\lambda_{ij}), \\qquad
        \\log \\boldsymbol{\\lambda} = A(\\boldsymbol{\\rho})^{-1} X\\beta

    where the system matrix :math:`A(\\rho_d, \\rho_o, \\rho_w) =
    I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w`.

    The spatial filter is on the **log-mean** scale. The implicit solve
    :math:`\\eta = A^{-1} X\\beta` is embedded in the PyMC graph via
    :class:`~bayespecon.ops.SparseFlowSolveOp`, which provides NUTS
    gradients via the adjoint method.

    Parameters
    ----------
    y : array-like of int, shape (n, n) or (N,)
        Observed non-negative integer flow counts. Float arrays whose
        values round exactly to integers are silently coerced.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted;
        otherwise defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns (destination/origin variable
        pairs). Inferred from ``dest_*``/``orig_*`` column names when the
        standard LeSage layout is used.
    logdet_method : str, default "traces"
        Log-determinant method. Only ``"traces"`` is supported here.
    restrict_positive : bool, default True
        If True, use ``pm.Dirichlet("rho_simplex", a=ones(4))`` to enforce
        :math:`\\rho_d, \\rho_o, \\rho_w \\geq 0` and
        :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`. If False, three
        independent ``pm.Uniform(rho_lower, rho_upper)`` priors are used
        with a differentiable quadratic-wall stability potential.
    miter : int, default 30
        Trace polynomial order for the log-determinant.
    titer : int, default 800
        Geometric tail cutoff for the log-determinant series.
    trace_riter : int, default 50
        Number of Monte Carlo probes for trace estimation.
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 10.0 — Normal prior std for ``beta``.
        - ``rho_lower`` : float, default -1.0 — Lower bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
        - ``rho_upper`` : float, default 1.0 — Upper bound of Uniform prior on each ρ (only when ``restrict_positive=False``).

    Notes
    -----
    There is no ``sigma`` parameter; the Poisson variance equals the mean.
    Robust (Student-t) likelihoods are not supported for Poisson counts.
    """

    def __init__(self, y, G, X, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonSARFlow requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "PoissonSARFlow requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), G, X, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        # Poisson observation density on observed counts is captured
        # directly by PyMC; the pm.Potential("jacobian", ...) acts on the
        # latent linear predictor, not on the observed-data density.
        return None

    def _simulate_y_rep(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
    ) -> np.ndarray:
        A = self._assemble_A(rho_d, rho_o, rho_w)
        eta = sp.linalg.spsolve(A, self._X_design @ beta)
        # Clip to avoid overflow on extreme draws.
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        return rng.poisson(lam).astype(np.float64)

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import SparseFlowSolveOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")

            Xb = pt.dot(X_t, beta)
            solve_op = SparseFlowSolveOp(self._Wd, self._Wo, self._Ww)
            eta = solve_op(rho_d, rho_o, rho_w, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                flow_logdet_pytensor(
                    rho_d,
                    rho_o,
                    rho_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model


# ---------------------------------------------------------------------------
# Model 4: PoissonSARFlowSeparable — Poisson response, rho_w = -rho_d * rho_o
# ---------------------------------------------------------------------------


class PoissonSARFlowSeparable(SARFlowSeparable):
    """Bayesian separable SAR flow model with a Poisson observation distribution.

    Combines the Poisson observation model of :class:`PoissonSARFlow` with
    the separability constraint :math:`\\rho_w = -\\rho_d \\rho_o`,
    enabling the exact eigenvalue-based log-determinant:

    .. math::

        \\log|A| = n\\,\\log|I_n - \\rho_d W| + n\\,\\log|I_n - \\rho_o W|

    Parameters
    ----------
    y : array-like of int, shape (n, n) or (N,)
        Observed non-negative integer flow counts. Float arrays whose
        values round exactly to integers are silently coerced.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of regional attribute columns. Inferred from
        ``dest_*``/``orig_*`` column names when the standard LeSage
        layout is used.
    logdet_method : {"eigenvalue", "chebyshev", "mc_poly"}, default "eigenvalue"
        Method for the Kronecker-factored log-determinant.
    miter : int, default 30
        Polynomial / approximation order (used by ``"chebyshev"`` /
        ``"mc_poly"``).
    titer : int, default 800
        Geometric tail cutoff for series-based variants.
    trace_riter : int, default 50
        Number of Monte Carlo probes (used by ``"mc_poly"``).
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 10.0 — Normal prior std for ``beta``.
        - ``rho_lower`` : float, default -0.999 — Lower bound of Uniform prior on ``rho_d`` and ``rho_o``.
        - ``rho_upper`` : float, default 0.999 — Upper bound of Uniform prior on ``rho_d`` and ``rho_o``.

    Notes
    -----
    There is no ``sigma`` parameter; the Poisson variance equals the mean.
    Robust (Student-t) likelihoods are not supported for Poisson counts.
    The ``restrict_positive`` argument has no effect on this class.
    """

    def __init__(self, y, G, X, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonSARFlowSeparable requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "PoissonSARFlowSeparable requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), G, X, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        # Poisson observation density on observed counts is captured
        # directly by PyMC; the pm.Potential("jacobian", ...) acts on the
        # latent linear predictor, not on the observed-data density.
        return None

    def _simulate_y_rep(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,  # ignored: separable constraint enforces rho_w = -rho_d*rho_o
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
    ) -> np.ndarray:
        I_n = sp.eye(self._n, format="csr", dtype=np.float64)
        Ld = I_n - rho_d * self._W_sparse
        Lo = I_n - rho_o * self._W_sparse
        eta = kron_solve_vec(Lo, Ld, self._X_design @ beta, self._n)
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        return rng.poisson(lam).astype(np.float64)

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import KroneckerFlowSolveOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        n = self._n
        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "PoissonSARFlowSeparable requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")

            # KroneckerFlowSolveOp exploits A = L_d ⊗ L_o (two n×n solves
            # instead of one n²×n² solve — O(n³) vs O(n⁶)).
            # rho_w is kept as a Deterministic for trace reporting only.
            Xb = pt.dot(X_t, beta)
            solve_op = KroneckerFlowSolveOp(self._W_sparse, n)
            eta = solve_op(rho_d, rho_o, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                self._separable_logdet_fn(rho_d, rho_o),
            )

        return model


# ---------------------------------------------------------------------------
# Model 5: OLSFlow — non-spatial gravity baseline (Thomas-Agnan & LeSage 2014)
# ---------------------------------------------------------------------------


class OLSFlow(FlowModel):
    r"""Non-spatial Bayesian OD-flow gravity model (independence baseline).

    Implements the conventional log-linear gravity model from
    :cite:t:`thomas-agnan2014SpatialEconometric` (eq. 83.2):

    .. math::

        y = \alpha \iota_{N} + X_o \beta_o + X_d \beta_d + g\gamma + \varepsilon,
        \quad \varepsilon \sim \mathcal{N}(0, \sigma^{2} I_{N})

    with no spatial-lag terms.  Provided as a baseline for comparison with
    :class:`SARFlow` / :class:`SARFlowSeparable` and to reproduce Table 83.1
    of the chapter.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed O-D flow matrix (or its vec-form). Must be a square
        matrix or a flat vector of length :math:`N = n^2`.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units. Required for API
        symmetry with the spatial flow models, but the graph weights
        are not used in estimation.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix.
    col_names : list[str], optional
        Column labels for *X*. Defaults to ``["x0", "x1", ...]`` when
        not provided and *X* is not a DataFrame.
    k : int, optional
        Number of regional attribute columns. Inferred from column names
        when they follow the ``dest_*``/``orig_*`` convention.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.

        Spatial keys (``rho_*``) are ignored.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin/destination design symmetry is
        auto-detected. Set explicitly to override.

    Notes
    -----
    No spatial-lag term enters the likelihood, so no log-determinant
    is required and ``logdet_method`` is ignored if passed.
    """

    _spatial_diagnostics_tests = [
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_flow_dest_test"],
            ).bayesian_lm_flow_dest_test(m),
            "LM-Flow-Dest",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_flow_orig_test"],
            ).bayesian_lm_flow_orig_test(m),
            "LM-Flow-Orig",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_flow_network_test"],
            ).bayesian_lm_flow_network_test(m),
            "LM-Flow-Network",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_flow_joint_test"],
            ).bayesian_lm_flow_joint_test(m),
            "LM-Flow-Joint",
        ),
        (
            lambda m: __import__(
                "bayespecon.diagnostics.bayesian_lmtests",
                fromlist=["bayesian_lm_flow_intra_test"],
            ).bayesian_lm_flow_intra_test(m),
            "LM-Flow-Intra",
        ),
    ]

    def __init__(self, y, G, X, **kwargs):
        # Skip log-determinant precomputation: A = I_N has |A| = 1.
        kwargs.pop("logdet_method", None)
        kwargs.pop("restrict_positive", None)
        super().__init__(y, G, X, logdet_method="none", **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y_vec.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = pt.dot(X_t, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

        return model

    def _simulate_y_rep(
        self,
        rho_d: float,  # unused
        rho_o: float,  # unused
        rho_w: float,  # unused
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        Xb = self._X_design @ beta
        if sigma is None:
            return Xb
        return Xb + rng.normal(scale=float(sigma), size=self._N)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flows for the OLS gravity model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep(0.0, 0.0, 0.0, beta_draws[g], sigma_g, rng)
        return out

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        r"""Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

        With :math:`A = I_N` the response to any shock equals the shock
        itself, so the Thomas-Agnan decomposition simplifies analytically to:

        .. math::

            \mathrm{TE} = \beta_d + \beta_o, \qquad
            \mathrm{NE} = 0, \qquad
            \mathrm{IE} = (\beta_d + \beta_o + \beta_{\text{intra}}) / n,

        with :math:`\mathrm{OE} = \beta_o (n-1)/n` and
        :math:`\mathrm{DE} = \beta_d (n-1)/n`, and the symmetric
        contributions ``β_intra / n`` distributed to the destination side.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        n = self._n
        k_d = self._k_d
        k_o = self._k_o
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = beta_draws.shape[0]
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            beta_draws = beta_draws[:n_draws_total]

        bd = beta_draws[:, dest_start : dest_start + k_d]
        bo = beta_draws[:, orig_start : orig_start + k_o]
        bi = (
            beta_draws[:, intra_start : intra_start + k_d]
            if has_intra
            else np.zeros((n_draws_total, k_d), dtype=np.float64)
        )

        # Per-region totals (averaged across all n perturbation regions).
        # Destination shock contributes bd to one whole row plus bi at the
        # diagonal cell; averaged over n regions: total_d = bd + bi / n.
        zeros_d = np.zeros_like(bd)
        zeros_o = np.zeros_like(bo)
        out: dict[str, np.ndarray] = {}
        out["dest_total"] = bd + bi / n
        out["dest_destination"] = bd * (n - 1) / n
        out["dest_intra"] = (bd + bi) / n
        out["dest_origin"] = zeros_d.copy()
        out["dest_network"] = zeros_d.copy()

        out["orig_total"] = bo
        out["orig_origin"] = bo * (n - 1) / n
        out["orig_intra"] = bo / n
        out["orig_destination"] = zeros_o.copy()
        out["orig_network"] = zeros_o.copy()

        if k_d == k_o:
            # Symmetric case: sum dest and orig effects (same variables)
            for eff in _EFFECT_KEYS:
                out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]
        else:
            # Asymmetric case: concatenate dest and orig effects (different variables)
            for eff in _EFFECT_KEYS:
                out[eff] = np.concatenate(
                    [out[f"dest_{eff}"], out[f"orig_{eff}"]], axis=1
                )

        return out


# ---------------------------------------------------------------------------
# Model 6: PoissonFlow — non-spatial Poisson gravity baseline
# ---------------------------------------------------------------------------


class PoissonFlow(OLSFlow):
    r"""Non-spatial Bayesian OD-flow Poisson gravity model (count baseline).

    Aspatial count analogue of :class:`OLSFlow`: models origin-destination
    flow **counts** :math:`y_{ij} \in \mathbb{N}_0` with a log-linear
    gravity mean and no spatial-lag terms,

    .. math::

        y_{ij} \sim \operatorname{Poisson}(\lambda_{ij}), \qquad
        \log \boldsymbol{\lambda} = X\beta.

    Provided as the count-data baseline analogue of :class:`OLSFlow`,
    matching the role :class:`PoissonSARFlow` plays relative to
    :class:`SARFlow`.  No spatial filter is applied (``A = I_N``), so there
    is no log-determinant precomputation, no ``rho_*`` parameters, and no
    ``sigma`` (the Poisson variance equals the mean).

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed non-negative integer flow counts.  Float arrays whose
        values are close to integers are silently rounded.
    G : libpysal.graph.Graph
        Graph on *n* units.  Required for API symmetry but the spatial
        weights are not used in estimation.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix.
    col_names : list[str], optional
        Column labels for *X*. Defaults to ``["x0", "x1", ...]`` when
        not provided and *X* is not a DataFrame.
    k : int, optional
        Number of regional attribute columns. Inferred from column names
        when they follow the ``dest_*``/``orig_*`` convention.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 10.0 — Normal prior std for ``beta``.

        Spatial keys (``rho_*``) and ``sigma_sigma`` are ignored.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin/destination design symmetry is
        auto-detected. Set explicitly to override.

    Notes
    -----
    No spatial filter is applied (``A = I_N``); the closed-form
    Thomas-Agnan & LeSage (2014, Table 83.1) effects from :class:`OLSFlow`
    are reused unchanged on the (log-mean) linear-predictor scale.
    """

    def __init__(self, y, G, X, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonFlow requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError("PoissonFlow requires non-negative integer observations.")
        super().__init__(y_arr.astype(np.float64), G, X, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            eta = pt.dot(X_t, beta)
            lam = pm.Deterministic("lambda", pt.exp(eta))
            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

        return model

    def _simulate_y_rep(
        self,
        rho_d: float,  # unused
        rho_o: float,  # unused
        rho_w: float,  # unused
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
    ) -> np.ndarray:
        eta = self._X_design @ beta
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        return rng.poisson(lam).astype(np.float64)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for the Poisson gravity model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            out[g] = self._simulate_y_rep(0.0, 0.0, 0.0, beta_draws[g], None, rng)
        return out


# ---------------------------------------------------------------------------
# Model 7: NegativeBinomial SAR/OLS flow variants
# ---------------------------------------------------------------------------


class NegativeBinomialSARFlow(PoissonSARFlow):
    r"""Bayesian SAR flow model with NB2 observation noise.

    This class mirrors :class:`PoissonSARFlow` but replaces the Poisson
    likelihood with:

    .. math::

        y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha),

    where ``alpha`` is an overdispersion parameter sampled from a
    HalfNormal prior.
    """

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import SparseFlowSolveOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)

            Xb = pt.dot(X_t, beta)
            solve_op = SparseFlowSolveOp(self._Wd, self._Wo, self._Ww)
            eta = solve_op(rho_d, rho_o, rho_w, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                flow_logdet_pytensor(
                    rho_d,
                    rho_o,
                    rho_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for NB SAR flow model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)
        rho_w_draws = post["rho_w"].values.reshape(-1)
        alpha_draws = post["alpha"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]
            rho_w_draws = rho_w_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            A = self._assemble_A(rho_d_draws[g], rho_o_draws[g], rho_w_draws[g])
            eta = sp.linalg.spsolve(A, self._X_design @ beta_draws[g])
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            alpha = float(alpha_draws[g])
            p = alpha / (alpha + lam)
            out[g] = rng.negative_binomial(alpha, p).astype(np.float64)
        return out


class NegativeBinomialSARFlowSeparable(PoissonSARFlowSeparable):
    """Separable SAR flow model with NB2 observation noise."""

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import KroneckerFlowSolveOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        n = self._n
        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "NegativeBinomialSARFlowSeparable requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)

            Xb = pt.dot(X_t, beta)
            solve_op = KroneckerFlowSolveOp(self._W_sparse, n)
            eta = solve_op(rho_d, rho_o, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                self._separable_logdet_fn(rho_d, rho_o),
            )

        return model

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for separable NB SAR flow."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)
        alpha_draws = post["alpha"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        n = self._n
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        for g in range(total):
            Ld = I_n - float(rho_d_draws[g]) * self._W_sparse
            Lo = I_n - float(rho_o_draws[g]) * self._W_sparse
            eta = kron_solve_vec(Lo, Ld, self._X_design @ beta_draws[g], n)
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            alpha = float(alpha_draws[g])
            p = alpha / (alpha + lam)
            out[g] = rng.negative_binomial(alpha, p).astype(np.float64)
        return out


class NegativeBinomialFlow(PoissonFlow):
    """Aspatial OD-flow Negative Binomial gravity baseline."""

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)
            eta = pt.dot(X_t, beta)
            lam = pm.Deterministic("lambda", pt.exp(eta))
            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

        return model

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for NB gravity baseline."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        alpha_draws = post["alpha"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            eta = self._X_design @ beta_draws[g]
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            alpha = float(alpha_draws[g])
            p = alpha / (alpha + lam)
            out[g] = rng.negative_binomial(alpha, p).astype(np.float64)
        return out


# ---------------------------------------------------------------------------
# Model 7: SEMFlow — Spatial-error analogue of SARFlow (unrestricted 3-rho)
# ---------------------------------------------------------------------------


class SEMFlow(FlowModel):
    r"""Bayesian spatial-error flow model with three free spatial parameters.

    .. math::

        y = X\beta + u, \qquad
        B u = \varepsilon, \qquad
        B = I_N - \lambda_d W_d - \lambda_o W_o - \lambda_w W_w,
        \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I_N)

    where :math:`W_d = I_n \otimes W`, :math:`W_o = W \otimes I_n`,
    :math:`W_w = W \otimes W`.  The Kronecker spatial structure is identical
    to :class:`SARFlow`, but the spatial filter acts on the *disturbance*
    rather than the dependent variable.  Equivalently the model implies a
    Gaussian likelihood with covariance :math:`\sigma^2 (B^\top B)^{-1}`.

    Marginal mean is :math:`\mathbb{E}[y] = X\beta`, so there are no
    :math:`X`-mediated spatial spillovers — the LeSage / Thomas-Agnan
    decomposition reduces to the closed-form expressions used by
    :class:`OLSFlow` (direct effect equals :math:`\beta`, network effect
    equals zero).  Use :class:`SARFlow` if spillovers from observed
    covariates are of interest.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted;
        otherwise defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns (destination/origin variable
        pairs). Inferred from ``dest_*``/``orig_*`` column names when the
        standard LeSage layout is used.
    logdet_method : str, default "traces"
        Log-determinant method. Only ``"traces"`` is supported here.
    restrict_positive : bool, default True
        If True, use ``pm.Dirichlet("lam_simplex", a=ones(4))`` to enforce
        :math:`\\lambda_d, \\lambda_o, \\lambda_w \\geq 0` and
        :math:`\\lambda_d + \\lambda_o + \\lambda_w \\leq 1`. If False,
        three independent ``pm.Uniform(lam_lower, lam_upper)`` priors are
        used with a differentiable quadratic-wall stability potential.
    miter : int, default 30
        Trace polynomial order for the log-determinant.
    titer : int, default 800
        Geometric tail cutoff for the log-determinant series.
    trace_riter : int, default 50
        Number of Monte Carlo probes for trace estimation.
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``lam_lower`` : float, default -1.0 — Lower bound of Uniform prior on each λ (only when ``restrict_positive=False``).
        - ``lam_upper`` : float, default 1.0 — Upper bound of Uniform prior on each λ (only when ``restrict_positive=False``).

    Notes
    -----
    Implementation: PyMC body uses precomputed lags of both ``y`` and
    ``X`` (``self._Wd``, ``self._Wo``, ``self._Ww`` applied to
    ``self._X_design``) so that the residual
    :math:`B u = B y - B X \\beta` is expressible as a linear combination
    of fixed quantities — no symbolic sparse mat-vec is required. The
    Jacobian :math:`\\log|B|` reuses the same trace-based polynomial as
    :class:`SARFlow`.
    """

    def __init__(self, y, G, X, **kwargs):
        super().__init__(y, G, X, **kwargs)
        # Precompute lags of the design matrix (constant — no parameter dependence).
        self._Wd_X: np.ndarray = np.asarray(self._Wd @ self._X_design, dtype=np.float64)
        self._Wo_X: np.ndarray = np.asarray(self._Wo @ self._X_design, dtype=np.float64)
        self._Ww_X: np.ndarray = np.asarray(self._Ww @ self._X_design, dtype=np.float64)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        Wd_X_t = pt.as_tensor_variable(self._Wd_X.astype(np.float64))
        Wo_X_t = pt.as_tensor_variable(self._Wo_X.astype(np.float64))
        Ww_X_t = pt.as_tensor_variable(self._Ww_X.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y_vec.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                lam_simplex = pm.Dirichlet("lam_simplex", a=np.ones(4))
                lam_d = pm.Deterministic("lam_d", lam_simplex[0])
                lam_o = pm.Deterministic("lam_o", lam_simplex[1])
                lam_w = pm.Deterministic("lam_w", lam_simplex[2])
            else:
                lam_lower = self.priors.get("lam_lower", -1.0)
                lam_upper = self.priors.get("lam_upper", 1.0)
                lam_d = pm.Uniform("lam_d", lower=lam_lower, upper=lam_upper)
                lam_o = pm.Uniform("lam_o", lower=lam_lower, upper=lam_upper)
                lam_w = pm.Uniform("lam_w", lower=lam_lower, upper=lam_upper)
                slack = 1.0 - lam_d - lam_o - lam_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # mu chosen so that y - mu = By - BXβ = Bu (the whitened residual).
            mu = (
                lam_d * Wd_y_t
                + lam_o * Wo_y_t
                + lam_w * Ww_y_t
                + pt.dot(X_t, beta)
                - lam_d * pt.dot(Wd_X_t, beta)
                - lam_o * pt.dot(Wo_X_t, beta)
                - lam_w * pt.dot(Ww_X_t, beta)
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            # Jacobian of the change of variables from epsilon = B*u to y:
            # log|det B| has the same Kronecker structure as |det A|.
            pm.Potential(
                "jacobian",
                flow_logdet_pytensor(
                    lam_d,
                    lam_o,
                    lam_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        lam_d = posterior["lam_d"].values.reshape(-1)
        lam_o = posterior["lam_o"].values.reshape(-1)
        lam_w = posterior["lam_w"].values.reshape(-1)
        return flow_logdet_numpy(
            lam_d,
            lam_o,
            lam_w,
            self._poly_a,
            self._poly_b,
            self._poly_c,
            self._poly_coeffs,
            self._miter_a,
            self._miter_b,
            self._miter_c,
            self._miter_coeffs,
            self.miter,
            self.titer,
        )

    def _simulate_y_rep(
        self,
        lam_d: float,
        lam_o: float,
        lam_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """SEM posterior-predictive: ``y_rep = X β + B^{-1} ε``."""
        Xb = self._X_design @ beta
        if sigma is None:
            return Xb
        B = self._assemble_A(lam_d, lam_o, lam_w)
        eps = rng.normal(scale=float(sigma), size=self._N)
        u = sp.linalg.spsolve(B, eps)
        return Xb + u

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        r"""Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

        The marginal mean :math:`\mathbb{E}[y] = X\beta` is unaffected by the
        spatial-error filter, so the LeSage decomposition collapses to the
        same closed form used by :class:`OLSFlow`: direct effect equals
        :math:`\beta`, network effect equals zero, intra/origin/destination
        contributions split :math:`\beta` per Table 83.1.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        n = self._n
        k_d = self._k_d
        k_o = self._k_o
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = beta_draws.shape[0]
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            beta_draws = beta_draws[:n_draws_total]

        bd = beta_draws[:, dest_start : dest_start + k_d]
        bo = beta_draws[:, orig_start : orig_start + k_o]
        bi = (
            beta_draws[:, intra_start : intra_start + k_d]
            if has_intra
            else np.zeros((n_draws_total, k_d), dtype=np.float64)
        )

        zeros_d = np.zeros_like(bd)
        zeros_o = np.zeros_like(bo)
        out: dict[str, np.ndarray] = {}
        out["dest_total"] = bd + bi / n
        out["dest_destination"] = bd * (n - 1) / n
        out["dest_intra"] = (bd + bi) / n
        out["dest_origin"] = zeros_d.copy()
        out["dest_network"] = zeros_d.copy()

        out["orig_total"] = bo
        out["orig_origin"] = bo * (n - 1) / n
        out["orig_intra"] = bo / n
        out["orig_destination"] = zeros_o.copy()
        out["orig_network"] = zeros_o.copy()

        if k_d == k_o:
            for eff in _EFFECT_KEYS:
                out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]
        else:
            for eff in _EFFECT_KEYS:
                out[eff] = np.concatenate(
                    [out[f"dest_{eff}"], out[f"orig_{eff}"]], axis=1
                )

        return out


# ---------------------------------------------------------------------------
# Model 8: SEMFlowSeparable — separable SEM with lam_w = -lam_d * lam_o
# ---------------------------------------------------------------------------


class SEMFlowSeparable(SEMFlow):
    r"""Bayesian separable spatial-error flow model with :math:`\lambda_w = -\lambda_d \lambda_o`.

    Spatial-error analogue of :class:`SARFlowSeparable`.  The separability
    constraint reduces :math:`\log|B|` to the eigenvalue / Chebyshev factored
    form

    .. math::

        \log|B| = n \log|I_n - \lambda_d W| + n \log|I_n - \lambda_o W|

    enabling :math:`O(n)` log-determinant evaluation per draw.  All other
    properties (no :math:`X`-mediated spillovers, closed-form effects, etc.)
    are identical to :class:`SEMFlow`.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of regional attribute columns. Inferred from
        ``dest_*``/``orig_*`` column names when the standard LeSage
        layout is used.
    logdet_method : {"eigenvalue", "chebyshev", "mc_poly"}, default "eigenvalue"
        Method for the Kronecker-factored log-determinant.
    miter : int, default 30
        Polynomial / approximation order (used by ``"chebyshev"`` /
        ``"mc_poly"``).
    titer : int, default 800
        Geometric tail cutoff for series-based variants.
    trace_riter : int, default 50
        Number of Monte Carlo probes (used by ``"mc_poly"``).
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``lam_lower`` : float, default -0.999 — Lower bound of Uniform prior on ``lam_d`` and ``lam_o``.
        - ``lam_upper`` : float, default 0.999 — Upper bound of Uniform prior on ``lam_d`` and ``lam_o``.

    Notes
    -----
    The ``restrict_positive`` argument inherited from :class:`FlowModel`
    has no effect on this class — separable variants always use Uniform
    priors on the individual :math:`\\lambda` components.
    """

    def __init__(self, y, G, X, **kwargs):
        method = kwargs.pop("logdet_method", "eigenvalue")
        _VALID = {"eigenvalue", "chebyshev", "mc_poly"}
        if method not in _VALID:
            raise ValueError(
                f"SEMFlowSeparable logdet_method must be one of {sorted(_VALID)}; "
                f"got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, G, X, **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        lam_lower = self.priors.get("lam_lower", -0.999)
        lam_upper = self.priors.get("lam_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SEMFlowSeparable requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        Wd_X_t = pt.as_tensor_variable(self._Wd_X.astype(np.float64))
        Wo_X_t = pt.as_tensor_variable(self._Wo_X.astype(np.float64))
        Ww_X_t = pt.as_tensor_variable(self._Ww_X.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y_vec.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            lam_d = pm.Uniform("lam_d", lower=lam_lower, upper=lam_upper)
            lam_o = pm.Uniform("lam_o", lower=lam_lower, upper=lam_upper)
            lam_w = pm.Deterministic("lam_w", -lam_d * lam_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = (
                lam_d * Wd_y_t
                + lam_o * Wo_y_t
                + lam_w * Ww_y_t
                + pt.dot(X_t, beta)
                - lam_d * pt.dot(Wd_X_t, beta)
                - lam_o * pt.dot(Wo_X_t, beta)
                - lam_w * pt.dot(Ww_X_t, beta)
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._separable_logdet_fn(lam_d, lam_o),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        lam_d = np.asarray(posterior["lam_d"].values.reshape(-1), dtype=np.float64)
        lam_o = np.asarray(posterior["lam_o"].values.reshape(-1), dtype=np.float64)
        if self._separable_logdet_numpy_fn is None:
            raise RuntimeError(
                "Missing separable numeric logdet evaluator. "
                "Initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        return self._separable_logdet_numpy_fn(lam_d, lam_o)
