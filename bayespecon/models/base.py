"""Base class for Bayesian spatial regression models."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from libpysal.graph import Graph

from .._backends.sampler_helpers import (
    jax_available,
    prepare_compile_kwargs,
    prepare_idata_kwargs,
    use_jax_likelihood,
)
from .._lazy_deps import az, pm
from .._logdet import (
    resolve_logdet_bounds,
)
from .._logdet._config import _auto_logdet_method
from ._base._shared import (
    SharedSpatialMethods,
    _parse_W,
    _pointwise_gaussian_loglik,
    _write_log_likelihood_to_idata,
)
from ._base._structure import CrossSectionStructure


class SpatialModel(SharedSpatialMethods, ABC):
    """Base class for Bayesian spatial regression models. Models follow the notation
    of :cite:p:`anselin1988SpatialEconometrics` and :cite:p:`lesage2009IntroductionSpatial`.
    The API supports both formula and matrix input modes.

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
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``. Accepts a
        :class:`libpysal.graph.Graph` (the modern libpysal graph API) or any
        :class:`scipy.sparse` matrix.  The legacy :class:`libpysal.weights.W`
        object is **not** accepted directly; pass ``w.sparse`` to use the
        underlying sparse matrix, or convert with
        ``libpysal.graph.Graph.from_W(w)``.
        W should be row-standardised; a :class:`UserWarning` is raised if not.
    priors : dict, optional
        Override default priors. Keys depend on the model subclass; see
        each model's docstring for supported keys.
    logdet_method : str
        How to compute ``log|I - rho*W|``. ``"eigenvalue"`` (default for
        ``n <= 500``) pre-computes W's eigenvalues once and evaluates
        O(n) per step; ``"exact"`` uses symbolic pytensor det (slow for
        ``n > 500``); ``"grid_dense"`` uses dense eigenvalue grid +
        cubic-spline interpolation (MATLAB-style ``lndetfull`` for dense
        W); ``"grid_sparse"`` uses sparse-LU grid + cubic-spline
        interpolation (``lndetfull`` style for large sparse W);
        ``"sparse_spline"`` uses sparse-LU + spline on
        ``[max(rho_min, 0), rho_max]`` (``lndetint`` style); ``"grid_mc"``
        uses Monte Carlo trace approximation (``lndetmc``); ``"grid_ilu"``
        uses ILU-based approximation (``lndetichol`` analog);
        ``"chebyshev"`` (default for ``n > 500``) uses a Chebyshev
        polynomial approximation evaluated via Clenshaw's algorithm.
        For large ``n`` the Chebyshev coefficients are built from
        Barry-Pace Hutchinson stochastic trace estimates.
    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal,
        yielding a model that is robust to heavy-tailed outliers. When
        ``robust=True``, a ``nu`` (degrees of freedom) parameter is added
        to the model with an :math:`\\mathrm{Exp}(\\lambda_\\nu)` prior (default
        ``nu_lam = 1/30``, mean ≈ 30). The ``nu`` prior can be controlled
        via the ``priors`` dict with key ``nu_lam``.
    w_vars : list of str, optional
        Names of X columns to spatially lag. Only relevant for models that
        include ``WX`` terms (SLX, SDM, SDEM and their panel/Tobit variants).
        By default all non-constant columns are lagged. Pass a subset to
        restrict which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``.

    Attributes
    ----------
    _spatial_params : tuple[str, ...]
        Spatial autoregressive parameters in the model (e.g. ``("rho",)``
        for SAR, ``("lam",)`` for SEM).  Empty for OLS and SLX.
    _lag_terms : tuple[str, ...]
        Lagged terms present in the model specification (e.g. ``("Wy",)``
        for SAR, ``("WX",)`` for SLX, ``("Wy", "WX")`` for SDM).
    _jacobian_param : str or None
        Name of the parameter that appears in the Jacobian determinant
        ``log|I - param * W|``.  ``"rho"`` for SAR/SDM, ``"lam"`` for
        SEM/SDEM, ``None`` for OLS/SLX (no Jacobian).
    _has_wx_in_beta : bool
        Whether the ``beta`` coefficient vector includes WX coefficients
        (i.e. whether the design matrix is ``[X, WX]`` rather than just
        ``X``).  True for SLX, SDM, SDEM.
    _gibbs_class : str or None
        Fully-qualified class name of the Gibbs sampler for this model
        (e.g. ``"GaussianSARGibbs"``), or ``None`` if no Gibbs sampler
        exists.  Used to look up the sampler at runtime to avoid circular
        imports.
    _model_type : str
        Short lowercase model name used as the ``model_type`` argument to
        the Gibbs sampler (e.g. ``"sar"``, ``"sdm"``).  Also used for
        InferenceData coordinate labels.
    """

    # --- Declarative model metadata ----------------------------------------
    # Subclasses override these to declare their spatial structure.
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = ""

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        priors: Optional[Union[dict, Any]] = None,
        logdet_method: str | None = None,
        robust: bool = False,
        w_vars: Optional[list] = None,
    ):
        # Resolve typed priors (dataclass) and dict view.
        from .priors import BasePriors, priors_as_dict, resolve_priors

        _priors_cls = getattr(self.__class__, "_priors_cls", BasePriors)
        self.priors_obj = resolve_priors(priors, _priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        self.logdet_method = logdet_method
        self.robust = robust

        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None

        if formula is not None:
            if data is None:
                raise ValueError("data must be provided when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
        elif y is not None and X is not None:
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
        else:
            raise ValueError("Provide either (formula, data) or (y, X).")

        if W is not None:
            # Validate W and store as CSR sparse matrix.
            # Dense conversion is deferred to _W_dense (lazy property).
            self._W_sparse, self._is_row_std = _parse_W(W, len(self._y))
            self._structure = CrossSectionStructure(self._W_sparse)
            # Eigenvalues are computed lazily via the _W_eigs cached property
            # to avoid the O(n³) eigendecomposition for large n where trace
            # or Chebyshev methods are used instead.
            # Resolve the logdet method up-front so the lazy property
            # accessors know whether eigenvalues are required.
            self._resolved_logdet_method = (
                self.logdet_method
                if self.logdet_method is not None
                else _auto_logdet_method(self._W_sparse.shape[0], W=self._W_sparse)
            )
            # Resolve rho/lambda bounds from method and priors.
            # For row-standardised W the spectral stability interval is
            # always approximately (-1, 1), so no eigenvalue computation
            # is needed here.
            self._logdet_bounds = resolve_logdet_bounds(
                self.logdet_method,
                n=len(self._y),
                priors=self.priors,
                W=self._W_sparse,
            )
            self._wx_column_indices = self._spatial_lag_column_indices(
                self._X, self._feature_names
            )
            if w_vars is not None:
                unknown = [v for v in w_vars if v not in self._feature_names]
                if unknown:
                    raise ValueError(
                        f"w_vars contains names not found in X columns: {unknown}. "
                        f"Available: {self._feature_names}"
                    )
                self._wx_column_indices = [
                    i
                    for i in self._wx_column_indices
                    if self._feature_names[i] in w_vars
                ]
            self._wx_feature_names = [
                self._feature_names[i] for i in self._wx_column_indices
            ]
            # Logdet builders are constructed lazily on first access — see
            # the _logdet_numpy_fn, _logdet_numpy_vec_fn and
            # _logdet_pytensor_fn properties.  Caches are seeded as None so
            # that init never triggers the underlying eigendecomposition for
            # chebyshev / trace methods.
            self._logdet_numpy_fn_cache = None
            self._logdet_numpy_vec_fn_cache = None
            self._logdet_grad_numpy_vec_fn_cache = None
            self._logdet_pytensor_fn_cache = None
            self._W_for_logdet_cache = None
            self._Wy: np.ndarray = np.asarray(
                self._W_sparse @ self._y, dtype=np.float64
            )
            if self._wx_column_indices:
                self._WX = np.asarray(
                    self._W_sparse @ self._X[:, self._wx_column_indices],
                    dtype=np.float64,
                )
            else:
                self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
        else:
            # W-free mode: no spatial structure; spec tests require W to be supplied.
            self._W_sparse = None
            self._structure = None
            self._is_row_std = False
            self._wx_column_indices: list[int] = []
            self._wx_feature_names: list[str] = []
            self._Wy = np.zeros(len(self._y), dtype=np.float64)
            self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
            if w_vars is not None:
                raise ValueError("w_vars requires a spatial weights matrix W.")

    @cached_property
    def _W_dense(self) -> np.ndarray:
        """Dense weight matrix, materialised lazily on first access."""
        return np.asarray(self._W_sparse.toarray(), dtype=np.float64)

    @cached_property
    def _W_pt_sparse(self):
        """PyTensor sparse variable wrapping :attr:`_W_sparse`.

        Cached so repeated PyMC model builds reuse the same symbolic sparse
        operator and avoid the ``O(n²)`` dense materialisation that
        ``pt.as_tensor_variable(self._W_dense)`` performs each time.

        Use with :func:`pytensor.sparse.structured_dot` (vector inputs must
        first be reshaped to ``(n, 1)`` because the vector overload's
        backward pass is broken in PyTensor).
        """
        import scipy.sparse as _sp
        from pytensor import sparse as _pts

        return _pts.as_sparse_variable(_sp.csc_matrix(self._W_sparse))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model. Subclasses implement this."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float | None = None,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        sampler: str | None = None,
        gibbs_backend: str = "auto",
        thin: int = 1,
        n_jobs: int = -1,
        idata_kwargs: dict[str, Any] | None = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior.

        Dispatches to this model's Gibbs sampler (``sampler="gibbs"``) or NUTS
        (``sampler="nuts"``).  When ``sampler`` is ``None`` (default), Gibbs is
        used if the model has a registered Gibbs sampler, otherwise NUTS.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup steps, and number of chains.
        target_accept : float, optional
            Target acceptance rate for NUTS.  NUTS-only: passing it with the
            Gibbs sampler raises ``TypeError``.  Defaults to ``0.9`` for NUTS.
        random_seed : int, optional
            Seed for reproducibility.
        progressbar : bool, default True
            Show progress bar(s) during sampling.
        sampler : {"gibbs", "nuts", None}, default None
            Sampling method.  ``None`` auto-selects Gibbs when this model has
            one, else NUTS.
        gibbs_backend : {"auto", "jax", "numpy"}, default "auto"
            Execution backend for the Gibbs sampler.  ``"auto"`` uses JAX when
            installed and supported by the family, else NumPy.  Ignored for NUTS.
        thin : int, default 1
            Keep every ``thin``-th post-warmup Gibbs draw (Gibbs only).
        n_jobs : int, default -1
            Parallel workers for the NumPy Gibbs path (Gibbs only).
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` (NUTS only).  ``{"log_likelihood": True}``
            reconstructs the complete Jacobian-corrected pointwise
            log-likelihood.
        **sample_kwargs
            For NUTS, forwarded to ``pm.sample`` (``nuts_sampler=...``); for
            Gibbs, the family's declared options (an unsupported key raises).

        Returns
        -------
        arviz.InferenceData
        """
        from ..samplers._registry import pop_options, resolve, resolve_backend

        gibbs_key = getattr(self, "_gibbs_key", None)
        entry = resolve(*gibbs_key) if gibbs_key is not None else None
        if sampler is None:
            sampler = "gibbs" if entry is not None else "nuts"

        if sampler == "gibbs":
            if entry is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no Gibbs sampler. "
                    "Use sampler='nuts' (the default)."
                )
            if self.robust and not entry.supports_robust:
                raise NotImplementedError(
                    "Gibbs sampling is not supported for robust (Student-t) "
                    "models. Use sampler='nuts'."
                )
            if target_accept is not None:
                raise TypeError(
                    "target_accept is a NUTS-only argument and is not valid for "
                    "the Gibbs sampler (sampler='gibbs'). Remove it, or use "
                    "sampler='nuts'."
                )
            backend = resolve_backend(gibbs_backend, entry, jax_ok=jax_available())
            family_opts = pop_options(sample_kwargs, entry)
            self._idata = entry.run(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                thin=thin,
                n_jobs=n_jobs,
                progressbar=progressbar,
                backend=backend,
                **family_opts,
            )
            self._idata = self._postprocess_idata(self._idata)
            return self._idata

        if sampler != "nuts":
            raise ValueError(
                f"sampler must be 'gibbs', 'nuts', or None, got {sampler!r}"
            )

        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
        compute_log_likelihood = bool((idata_kwargs or {}).get("log_likelihood", False))
        _, compute_log_likelihood = self._fit_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9 if target_accept is None else target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
            idata_kwargs=idata_kwargs,
            compute_log_likelihood=compute_log_likelihood,
            sample_kwargs=sample_kwargs,
        )
        # Gaussian spatial models capture only the Normal part natively; add
        # the Jacobian-corrected pointwise log-likelihood (Pattern A+J).
        if compute_log_likelihood and getattr(self, "_likelihood", None) == "gaussian":
            self._reconstruct_cross_sectional_log_likelihood(nuts_sampler=nuts_sampler)
        self._idata = self._postprocess_idata(self._idata)
        return self._idata

    def _fit_nuts(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        target_accept: float,
        random_seed: Optional[int],
        progressbar: bool,
        nuts_sampler: str = "pymc",
        idata_kwargs: dict[str, Any] | None = None,
        compute_log_likelihood: bool = False,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> tuple[az.InferenceData, bool]:
        """Shared NUTS sampling path used by model-specific ``fit`` methods.

        Returns
        -------
        tuple[arviz.InferenceData, bool]
            ``(idata, compute_log_likelihood)`` where the boolean reflects the
            post-policy value after :func:`prepare_idata_kwargs`.
        """
        sample_kwargs = dict(sample_kwargs or {})
        idata_kwargs = dict(idata_kwargs or {})

        build_kwargs: dict[str, Any] = {}
        build_sig = inspect.signature(self._build_pymc_model)
        if "compute_log_likelihood" in build_sig.parameters:
            build_kwargs["compute_log_likelihood"] = compute_log_likelihood
        if "nuts_sampler" in build_sig.parameters:
            build_kwargs["nuts_sampler"] = nuts_sampler

        model = self._build_pymc_model(**build_kwargs)
        self._pymc_model = model

        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
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
                progressbar=progressbar,
                **sample_kwargs,
            )
        return self._idata, compute_log_likelihood

    def _reconstruct_cross_sectional_log_likelihood(
        self,
        *,
        nuts_sampler: str,
    ) -> None:
        """Rebuild complete pointwise log-likelihood for cross-sectional models.

        Dispatches by spatial term type (``rho`` vs ``lam``) and whether the
        model's beta vector includes WX terms.
        """
        if not hasattr(self, "_idata"):
            return

        spatial_param = self._jacobian_param
        if spatial_param not in {"rho", "lam"}:
            return

        # SEM/SDEM on JAX backends build an observed CustomDist and already
        # have complete log_likelihood from PyMC.
        if spatial_param == "lam" and use_jax_likelihood(nuts_sampler):
            return

        idata = self._idata
        n = int(self._y.shape[0])
        Z = np.hstack([self._X, self._WX]) if self._has_wx_in_beta else self._X

        spatial_draws = idata.posterior[spatial_param].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(-1, Z.shape[1])
        sigma_draws = idata.posterior["sigma"].values.reshape(-1)
        nu_draws = idata.posterior["nu"].values.reshape(-1) if self.robust else None

        if spatial_param == "rho":
            mu = spatial_draws[:, None] * self._Wy[None, :] + (beta_draws @ Z.T)
            eps = self._y[None, :] - mu
        else:
            resid = self._y[None, :] - (beta_draws @ Z.T)
            W_resid = (self._W_sparse @ resid.T).T
            eps = resid - spatial_draws[:, None] * W_resid

        ll_data = _pointwise_gaussian_loglik(eps, sigma_draws, nu_draws)
        jacobian = self._logdet_numpy_vec_fn(spatial_draws)
        ll_total = ll_data + jacobian[:, None] / n

        n_chains = idata.posterior.sizes["chain"]
        n_draws_per_chain = idata.posterior.sizes["draw"]
        _write_log_likelihood_to_idata(
            idata,
            ll_total.reshape(n_chains, n_draws_per_chain, n),
        )

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        gibbs_method: str = "numpy",
        mala_step_size: float = 0.05,
        use_mala: bool = True,
        use_slice: bool = True,
        slice_width: float | None = None,
        chain_method: str | None = None,
    ) -> az.InferenceData:
        """Sample posterior via 3-block Gaussian Gibbs.

        Uses the model's :attr:`_gibbs_class` attribute to resolve the
        appropriate Gibbs sampler class at runtime.  Only models with
        ``_gibbs_class is not None`` support Gibbs sampling; calling this
        method on OLS or SLX raises :exc:`NotImplementedError`.

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup (burn-in) draws per chain.
        chains : int, default 4
            Number of independent chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup.
        n_jobs : int, default -1
            Number of parallel workers for the NumPy path.  ``-1`` uses
            all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "numpy"
            Execution backend: ``"numpy"`` for Python-loop Gibbs with
            adaptive slice sampling, or ``"jax"`` for full-JIT Gibbs
            with MALA for ρ/λ.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
        use_mala : bool, default True
            If True, use MALA for the ρ/λ update in the JAX path.
        use_slice : bool, default True
            If True, use slice sampling for the ρ/λ update.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.

        Returns
        -------
        arviz.InferenceData
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.

        Raises
        ------
        NotImplementedError
            If the model has no Gibbs sampler (``_gibbs_class is None``)
            or uses a robust (Student-t) likelihood.
        """
        if self._gibbs_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support Gibbs sampling. "
                f"Use sampler='nuts' (the default)."
            )
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        # --- Resolve Gibbs class (lazy import to avoid circular deps) ---
        import importlib

        from ..samplers.gaussian import GaussianGibbsPriors

        gibbs_module = importlib.import_module(
            "..samplers.gaussian", package=__package__
        )
        GibbsClass = getattr(gibbs_module, self._gibbs_class)

        # --- Build design matrix and feature names ---
        if self._has_wx_in_beta:
            Z = np.hstack([self._X, self._WX])  # (n, 2k)
            feature_names = list(self._feature_names) + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        else:
            Z = self._X
            feature_names = list(self._feature_names)

        # --- Build priors ---
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            Z, feature_names
        )
        priors = GaussianGibbsPriors(
            beta_mu=self.priors.get("beta_mu", default_beta_mu),
            beta_sigma=self.priors.get("beta_sigma", default_beta_sigma),
            sigma2_alpha=self.priors.get("sigma2_alpha", 2.0),
            sigma2_beta=self.priors.get("sigma2_beta", float(np.var(self._y))),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        # --- Build Gibbs sampler kwargs ---
        gibbs_kwargs: dict[str, Any] = dict(
            y=self._y,
            X=Z,
            W_sparse=self._W_sparse,
            priors=priors,
            logdet_fn=self._logdet_numpy_fn,
            logdet_vec_fn=self._logdet_numpy_vec_fn,
            feature_names=feature_names,
            model_type=self._model_type,
            W_eigs=self._W_eigs
            if self._resolved_logdet_method == "eigenvalue"
            else None,
            logdet_method=self.logdet_method,
        )
        # SAR/SDM need Wy; SEM/SDEM do not
        if self._jacobian_param == "rho":
            gibbs_kwargs["Wy"] = self._Wy

        gibbs = GibbsClass(**gibbs_kwargs)

        self._idata = gibbs.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            n_jobs=n_jobs,
            progressbar=progressbar,
            gibbs_method=gibbs_method,
            slice_width=slice_width,
            chain_method=chain_method,
        )
        return self._idata

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model object built for the most recent fit.

        For Gibbs-fitted models the PyMC model is not constructed during
        sampling; it is built lazily on first access so that downstream
        consumers (e.g. bridge sampling for marginal likelihoods) can
        evaluate ``logp`` and the prior under the same model definition
        used by the NUTS path.

        Returns
        -------
        pymc.Model or None
            The model object used by :meth:`fit`, or ``None`` if the instance
            has not been fit yet.
        """
        if self._pymc_model is None and self._idata is not None:
            try:
                self._pymc_model = self._build_pymc_model()
            except TypeError:
                self._pymc_model = self._build_pymc_model(nuts_sampler="pymc")
        return self._pymc_model

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)`` where each
            array has shape ``(G, k)`` or ``(G, k_wx)``, with *G* being the
            total number of posterior draws and *k* / *k_wx* being the
            number of covariates for which effects are reported.
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

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(n={n}, k={k}, features={self._feature_names})"
        )
