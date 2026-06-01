r"""Structural-form SAR-logit with Pólya–Gamma Gibbs sampler.

.. math::

    y_i \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_i)), \quad
    \eta = \rho W \eta + X\beta + \nu, \quad
    \nu \sim N(0, I)

The logit link absorbs the error scale, so σ² is fixed at 1 and does
not appear as a free parameter.  The Pólya–Gamma augmentation yields
fully conjugate Gibbs updates for η, β, and ρ (via collapsed slice
sampling).

Use this model when:
- The response is binary (0/1).
- You need spatial autocorrelation in the latent log-odds.
- NUTS is slow or unreliable for the spatial parameter ρ.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import scipy.sparse as sp

from ..._logdet import make_logdet_numpy_fn
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers._utils._slice import SliceWidthState
from ...samplers._utils._spatial_normal import CholmodFactor, has_cholmod
from ...samplers.gaussian._chain_runner import run_chains
from ...samplers.logit import (
    LogitGibbsCache,
    LogitGibbsPriors,
    LogitGibbsState,
    run_chain,
)
from ...samplers.logit._jax import run_chains_jax_vectorized
from ..base import SpatialModel
from ..priors import SpatialLogitPriors, resolve_priors


class SARSpatialLogit(SpatialModel):
    """Bayesian structural-form SAR-logit with Pólya–Gamma Gibbs sampler.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. An intercept is included by default; suppress with
        ``"y ~ x - 1"``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Binary dependent variable of shape ``(n,)``. Required in matrix
        mode. Must contain only 0 and 1.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``.
    priors : dict or SpatialLogitPriors, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -0.999): Lower bound of the
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 0.999): Upper bound of the
          Uniform prior on :math:`\\rho`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 10.0): Normal prior std for
          :math:`\\beta`.

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects based on ``n``.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.

    Notes
    -----
    The structural form parameterises the latent log-odds as
    ``eta = rho * W @ eta + X @ beta + nu`` with ``nu ~ N(0, I)``,
    and augments the logistic likelihood with Pólya–Gamma auxiliary
    variables to obtain fully conjugate Gibbs updates for η and β.

    The sampler bypasses PyMC's NUTS entirely. It produces an
    ``arviz.InferenceData`` object compatible with all downstream
    diagnostics (``spatial_diagnostics()``, ``spatial_effects()``,
    ``summary()``).

    The ``fit()`` method does **not** accept ``nuts_sampler`` or
    ``target_accept`` kwargs — these are NUTS-specific and will raise
    ``TypeError`` if passed.

    Because the logit link absorbs the error scale, σ² is fixed at 1
    and does not appear in the posterior.  The PG shape parameter is
    always h = 1 (one trial per observation), so the Devroye method
    is valid and typically fastest.
    """

    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy",)
    _jacobian_param: str | None = "rho"
    _gibbs_class: str | None = None  # Gibbs-only, no NUTS
    _model_type: str = "sar_logit"
    _priors_cls = SpatialLogitPriors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError(
                "robust=True is not supported for SARSpatialLogit."
            )

        # Validate y is binary
        if not np.isin(self._y, [0.0, 1.0]).all():
            raise ValueError("y must be binary with values in {0, 1}.")

        # Precompute logdet callable for the ρ slice sampler.
        self._logdet_fn = self._make_logdet_fn()

    def _make_logdet_fn(self):
        """Build a callable logdet(rho) -> float for the ρ slice sampler."""
        bounds = self._logdet_bounds
        return make_logdet_numpy_fn(
            self._W_sparse,
            eigs=self._W_eigs,
            method=bounds.method,
            rho_min=bounds.rho_min,
            rho_max=bounds.rho_max,
            trace_estimator=self.trace_estimator,
            trace_k=self.trace_k,
        )

    # ------------------------------------------------------------------
    # Auto-selection for Gibbs path
    # ------------------------------------------------------------------

    _JAX_DENSE_THRESHOLD: int = 10000

    def _initialize_from_ols(self, rng):
        """Warm-start the Gibbs sampler from a linear probability model.

        Fits OLS of y on X (treating y as continuous), then uses the
        fitted values as initial η.  This is a standard approach for
        logit/probit models — the linear probability model provides
        reasonable starting values even though it can predict outside
        [0, 1].

        Returns a LogitGibbsState with reasonable starting values.
        """
        y = self._y
        X = self._X
        n, k = X.shape

        # β₀: OLS on y (linear probability model)
        try:
            beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_init = np.zeros(k)

        # η₀: X @ β₀ (no spatial structure)
        eta_init = X @ beta_init

        # ρ₀: start at 0 (no spatial dependence)
        rho_init = 0.0

        # ω₀: draw from PG(1, η)
        from ...samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        return LogitGibbsState(
            eta=eta_init,
            beta=beta_init,
            rho=rho_init,
            omega=omega_init,
        )

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        return_eta: bool = False,
        n_jobs: int = -1,
        progressbar: bool = True,
        gibbs_method: str = "auto",
        pg_n_terms: int = 25,
        n_probes: int = 5,
        lanczos_deg: int = 15,
        **kwargs,
    ) -> az.InferenceData:
        """Sample posterior via Pólya–Gamma block Gibbs.

        Parameters
        ----------
        draws : int
            Number of post-warmup draws per chain.
        tune : int
            Number of warmup (burn-in) draws per chain.
        chains : int
            Number of independent chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int
            Keep every ``thin``-th draw. Default 1 (no thinning).
        return_eta : bool
            If True, store the full latent field η in the posterior.
            Default False — η is n × draws × chains, which can be large.
        n_jobs : int
            Number of parallel chains. -1 = all CPUs.
        progressbar : bool
            Show per-chain progress bars.
        gibbs_method : str, default "auto"
            Which Gibbs sampler path to use:

            - ``"auto"``: select based on JAX availability and CHOLMOD.
            - ``"factorize"``: force factorisation-based path (CHOLMOD if
              available, else ``scipy.sparse.linalg.splu``).
            - ``"jax_dense"``: force JAX-accelerated path.  Requires
              JAX with float64 enabled.  Viable for n ≤ ~10 000.
        pg_n_terms : int, default 25
            Number of alternating-series terms for the JAX Pólya–Gamma
            sampler.  Only used when ``gibbs_method="jax_dense"``.
        n_probes : int, default 5
            Number of Lanczos probe vectors for stochastic log|P|
            estimation.  Only used when ``gibbs_method="jax_dense"``.
        lanczos_deg : int, default 15
            Lanczos iteration depth for log|P| estimation.  Only used
            when ``gibbs_method="jax_dense"``.

        Returns
        -------
        az.InferenceData
            With posterior, log_likelihood, and observed_data groups.

        Raises
        ------
        TypeError
            If NUTS-specific kwargs (nuts_sampler, target_accept) are passed.
        """
        # Reject NUTS-specific kwargs
        for bad_kwarg in ("nuts_sampler", "target_accept", "idata_kwargs"):
            if bad_kwarg in kwargs:
                raise TypeError(
                    f"SARSpatialLogit.fit() does not accept '{bad_kwarg}'. "
                    f"This model uses a Gibbs sampler, not NUTS."
                )

        y = self._y
        X = self._X
        W_sparse = self._W_sparse
        n, k = X.shape

        # Build priors from the typed priors object
        priors_obj = resolve_priors(
            self.priors if isinstance(self.priors, dict) else None,
            SpatialLogitPriors,
        )
        if isinstance(self.priors, SpatialLogitPriors):
            priors_obj = self.priors

        priors = LogitGibbsPriors(
            beta_mu=priors_obj.beta_mu,
            beta_sigma=priors_obj.beta_sigma,
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        # Build cache
        XtX = X.T @ X

        # Precompute matrix pieces for the precision expansion:
        # P = I + diag(ω) - ρ*(W+W^T) + ρ²*W^T W  (σ² = 1)
        W_sym = W_sparse + W_sparse.T
        WtW = W_sparse.T @ W_sparse

        # Create CHOLMOD factor for the precision matrix sparsity pattern.
        if has_cholmod():
            _P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
            cholmod_factor = CholmodFactor(_P0)
        else:
            cholmod_factor = None

        # Resolve Gibbs method
        _valid_methods = {"auto", "factorize", "jax_dense"}
        if gibbs_method not in _valid_methods:
            raise ValueError(
                f"gibbs_method must be one of {_valid_methods}, got '{gibbs_method}'"
            )

        import importlib.util

        _jax_available = importlib.util.find_spec("jax") is not None

        if gibbs_method == "jax_dense" and not _jax_available:
            raise ImportError(
                "gibbs_method='jax_dense' requires JAX. Install with: pip install jax"
            )

        if gibbs_method == "factorize":
            solve_method = "cholmod" if cholmod_factor is not None else "splu"
            logdet_P_method = "cholmod"
            sample_method = "cholmod" if cholmod_factor is not None else "splu"
        elif gibbs_method == "jax_dense":
            solve_method = "jax_dense"
            logdet_P_method = "jax_dense"
            sample_method = "jax_dense"
        else:  # "auto"
            if cholmod_factor is not None:
                solve_method = "cholmod"
                logdet_P_method = "cholmod"
                sample_method = "cholmod"
            elif _jax_available and n <= self._JAX_DENSE_THRESHOLD:
                solve_method = "jax_dense"
                logdet_P_method = "jax_dense"
                sample_method = "jax_dense"
            else:
                solve_method = "splu"
                logdet_P_method = "cholmod"
                sample_method = "splu"

        # Precompute JAX dense components if using jax_dense path
        W_sym_dense = None
        WtW_dense = None
        logdet_jax = None
        if solve_method == "jax_dense":
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
            WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)

            from ..._logdet import make_logdet_jax_fn

            bounds = self._logdet_bounds
            logdet_jax = make_logdet_jax_fn(
                W_sparse,
                method=bounds.method,
                rho_min=bounds.rho_min,
                rho_max=bounds.rho_max,
                trace_estimator=self.trace_estimator,
                trace_k=self.trace_k,
            )

        cache = LogitGibbsCache(
            W_sparse=W_sparse,
            XtX=XtX,
            logdet_fn=self._logdet_fn,
            rho_lower=priors.rho_lower,
            rho_upper=priors.rho_upper,
            cholmod_factor=cholmod_factor,
            W_sym=W_sym,
            WtW=WtW,
            solve_method=solve_method,
            logdet_P_method=logdet_P_method,
            sample_method=sample_method,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            rho_adaptive_width=True,
            rho_slice_width_state=SliceWidthState(w=0.2),
        )

        # Derive per-chain seeds
        if random_seed is not None:
            parent_ss = np.random.SeedSequence(random_seed)
        else:
            parent_ss = np.random.SeedSequence()
        child_seeds = parent_ss.spawn(chains)
        seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

        # Define the per-chain function
        _use_jax_full = sample_method == "jax_dense"

        # JAX dense path: run all chains in parallel via jax.vmap.  This
        # JITs the Gibbs step once and executes every chain as a single
        # fused XLA program, which is strictly faster than driving the
        # per-chain Python loop ``chains`` times.
        if _use_jax_full:
            if return_eta:
                raise NotImplementedError(
                    "return_eta=True is not supported with gibbs_method='jax_dense'. "
                    "Use gibbs_method='factorize' (or 'auto' on systems without "
                    "CHOLMOD-only data) if you need the full latent field stored."
                )
            chain_inits = [
                self._initialize_from_ols(np.random.default_rng(seed)) for seed in seeds
            ]
            chain_results = run_chains_jax_vectorized(
                y=y,
                X=X,
                W_sparse=W_sparse,
                W_sym_dense=W_sym_dense,
                WtW_dense=WtW_dense,
                logdet_jax=logdet_jax,
                priors=priors,
                inits=chain_inits,
                draws=draws,
                tune=tune,
                thin=thin,
                jax_seeds=seeds,
                pg_n_terms=pg_n_terms,
                n_probes=n_probes,
                lanczos_deg=lanczos_deg,
                progressbar=progressbar,
            )
        else:

            def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
                rng = np.random.default_rng(seed)
                init = self._initialize_from_ols(rng)
                progress_chain_id = chain_id if chain_id_kw is None else chain_id_kw
                return run_chain(
                    y=y,
                    X=X,
                    W_sparse=W_sparse,
                    priors=priors,
                    cache=cache,
                    init=init,
                    draws=draws,
                    tune=tune,
                    thin=thin,
                    return_eta=return_eta,
                    rng=rng,
                    progress_manager=progress_manager,
                    chain_id=progress_chain_id,
                )

            # Run chains.  Non-JAX paths parallelise across chains when
            # the user requests multiple workers.
            parallel = n_jobs != 1
            chain_results = run_chains(
                chain_fn=_run_one_chain,
                n_chains=chains,
                seeds=seeds,
                n_jobs=n_jobs,
                progressbar=progressbar,
                parallel=parallel,
                draws=draws,
                tune=tune,
                model_type="sar_logit",
            )

        # Assemble InferenceData
        param_keys = ["rho"]
        if return_eta:
            param_keys.append("eta")

        posterior_samples = {}
        for key in param_keys:
            arrays = [c[key] for c in chain_results]
            posterior_samples[key] = np.stack(arrays, axis=0)

        # beta has shape (n_keep, k) per chain
        posterior_samples["beta"] = np.stack([c["beta"] for c in chain_results], axis=0)

        # Feature names for coords
        feature_names = list(self._feature_names)
        coords = {
            "coefficient": feature_names,
        }
        dims = {
            "beta": ["coefficient"],
        }
        if return_eta:
            coords["obs_id"] = list(range(n))
            dims["eta"] = ["obs_id"]

        # Log-likelihood: shape (chains, n_keep, n)
        log_lik = np.stack([c["log_lik"] for c in chain_results], axis=0)

        idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood={"obs": log_lik},
            observed_data={"obs": y},
            coords=coords,
            dims=dims,
        )

        self._idata = idata
        return idata

    def _build_pymc_model(self):
        """Not supported — SARSpatialLogit uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            "SARSpatialLogit does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling."
        )

    def fitted_probabilities(self) -> np.ndarray:
        """Compute fitted probabilities at posterior mean parameters.

        Returns the probability P(y=1) = logit⁻¹(η) where η is computed
        from the posterior mean of ρ and β via the reduced form:
        η = (I - ρW)⁻¹ Xβ.

        Returns
        -------
        probs : ndarray of shape (n,)
            Fitted probabilities at posterior mean.
        """
        self._require_fit()
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        A_rho_inv = sp.linalg.spsolve(
            sp.eye(self._X.shape[0], format="csr") - rho * self._W_sparse,
            self._X @ beta,
        )
        return 1.0 / (1.0 + np.exp(-A_rho_inv))

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        For the logit model, the fitted mean is the fitted probability.
        """
        return self.fitted_probabilities()

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute average direct/indirect/total impacts on the log-odds scale."""
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        mean_row_sum = float(self._batch_mean_row_sum(np.array([rho]))[0])
        ni = self._nonintercept_indices
        direct = mean_diag * beta[ni]
        total = mean_row_sum * beta[ni]
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._nonintercept_feature_names,
        }

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior impacts on the log-odds scale for each draw."""
        from ...diagnostics.lmtests import _get_posterior_draws
        from ...diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        eigs = self._W_eigs
        mean_diag = _chunked_eig_means(rho_draws, eigs)
        mean_row_sum = self._batch_mean_row_sum(rho_draws)

        ni = self._nonintercept_indices
        direct_samples = mean_diag[:, None] * beta_draws[:, ni]
        total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
        indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    #: Threshold above which probability-scale spatial effects use the
    #: sparse Hutchinson path instead of the eigendecomposition path.
    #: See :attr:`SARNegativeBinomial._COUNT_EFFECTS_EIGEN_MAX_N` for the
    #: cost model — the logit case is identical structurally.
    _PROBABILITY_EFFECTS_EIGEN_MAX_N: int = 2000

    def _compute_probability_scale_spatial_effects_posterior(
        self,
        method: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute posterior impacts on the probability scale for each draw.

        Notes
        -----
        For the SAR-logit model with

        .. math::

            \eta = (I - \rho W)^{-1} X\beta, \qquad p = \sigma(\eta),

        the average partial-effect matrix for covariate :math:`x_r` on the
        response (probability) scale is

        .. math::

            \frac{\partial p}{\partial x_r'} =
            \operatorname{diag}\bigl(p \odot (1 - p)\bigr)
            (I - \rho W)^{-1} \beta_r,

        which matches LeSage & Pace (2009) and §3.6 of the spatial PG paper.
        Direct, indirect, and total effects are the average diagonal, the
        average row sum minus diagonal, and the average row sum of this
        matrix respectively.

        For ``n ≤ _PROBABILITY_EFFECTS_EIGEN_MAX_N`` (default 2000) this
        uses the shared eigendecomposition cache; otherwise it falls back
        to per-draw sparse LU + Hutchinson diagonal estimation.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        n = self._X.shape[0]
        ni = self._nonintercept_indices
        n_draws = rho_draws.shape[0]
        n_effects = len(ni)

        if method not in {"auto", "eigen", "sparse"}:
            raise ValueError(
                f"method must be one of {{'auto', 'eigen', 'sparse'}}, got {method!r}."
            )

        use_sparse = method == "sparse" or (
            method == "auto" and n > self._PROBABILITY_EFFECTS_EIGEN_MAX_N
        )
        if use_sparse:
            return self._compute_probability_scale_spatial_effects_posterior_sparse(
                rho_draws=rho_draws,
                beta_draws=beta_draws,
                n=n,
                ni=ni,
                n_draws=n_draws,
                n_effects=n_effects,
            )

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        eigs_c = decomp[0]
        V_c = decomp[1]
        Vinv_c = decomp[2]

        VinvX = Vinv_c @ self._X.astype(np.complex128)
        ones_c = np.ones(n, dtype=np.complex128)
        Vinv_ones = Vinv_c @ ones_c

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            inv_eigs_c = 1.0 / (1.0 - float(rho) * eigs_c)

            coeff = inv_eigs_c * (VinvX @ beta.astype(np.complex128))
            eta = (V_c @ coeff).real.astype(np.float64)
            # Stable sigmoid; clip to avoid overflow in the rare extreme draw.
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -50.0, 50.0)))
            w = p * (1.0 - p)

            multiplier_diag = ((V_c * Vinv_c.T) @ inv_eigs_c).real.astype(np.float64)
            if self._is_row_std:
                multiplier_row_sums = np.full(
                    n, 1.0 / (1.0 - float(rho)), dtype=np.float64
                )
            else:
                multiplier_row_sums = (V_c @ (inv_eigs_c * Vinv_ones)).real.astype(
                    np.float64
                )

            direct_base = float(np.mean(w * multiplier_diag))
            total_base = float(np.mean(w * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    def _compute_probability_scale_spatial_effects_posterior_sparse(
        self,
        rho_draws: np.ndarray,
        beta_draws: np.ndarray,
        n: int,
        ni: list[int],
        n_draws: int,
        n_effects: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Probability-scale spatial effects via sparse solves + Hutchinson.

        Direct port of
        :meth:`SARNegativeBinomial._compute_count_scale_spatial_effects_posterior_sparse`
        with the count-mean weight :math:`\mu` replaced by the Bernoulli
        variance :math:`p(1-p)`.
        """
        from ..._ops import _make_cached_umfpack_solver

        W = self._W_sparse
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        rng = np.random.default_rng(42)
        n_probes = 20

        Z = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=(n, n_probes),
        )

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            rho_f = float(rho)
            A = (I_n - rho_f * W).tocsc()

            solver = _make_cached_umfpack_solver(A)
            if solver is None:
                solver = sp.linalg.splu(A)

            Xbeta = self._X @ beta
            if self._is_row_std:
                rhs = np.empty((n, 1 + n_probes), dtype=np.float64)
                rhs[:, 0] = Xbeta
                rhs[:, 1:] = Z
                sol = np.asarray(solver.solve(rhs), dtype=np.float64)
                eta = sol[:, 0]
                AinvZ = sol[:, 1:]
                multiplier_row_sums = np.full(n, 1.0 / (1.0 - rho_f), dtype=np.float64)
            else:
                rhs = np.empty((n, 2 + n_probes), dtype=np.float64)
                rhs[:, 0] = Xbeta
                rhs[:, 1] = ones
                rhs[:, 2:] = Z
                sol = np.asarray(solver.solve(rhs), dtype=np.float64)
                eta = sol[:, 0]
                multiplier_row_sums = sol[:, 1]
                AinvZ = sol[:, 2:]

            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -50.0, 50.0)))
            w = p * (1.0 - p)

            multiplier_diag = np.mean(Z * AinvZ, axis=1)

            direct_base = float(np.mean(w * multiplier_diag))
            total_base = float(np.mean(w * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    def spatial_effects(
        self,
        return_posterior_samples: bool = False,
        scale: str = "logodds",
        method: str = "auto",
    ):
        r"""Compute Bayesian inference for direct, indirect, and total impacts.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, also return the posterior draws for each effect type.
        scale : {"logodds", "probability"}, default "logodds"
            Scale on which impacts are reported.

            ``"logodds"`` returns impacts on the linear-predictor (log-odds)
            scale :math:`\eta = (I - \rho W)^{-1} X\beta`. This is the
            cheap default; effects are linear in :math:`\beta` and do not
            require evaluating the link.

            ``"probability"`` returns response-scale impacts

            .. math::

                \partial p / \partial x_r =
                \operatorname{diag}(p(1-p))\,(I - \rho W)^{-1}\beta_r,

            which match the LeSage-Pace (2009) and PG-paper formulas. This
            is more expensive because it requires the diagonal of the
            spatial multiplier and the fitted probabilities for each draw.
        method : {"auto", "eigen", "sparse"}, default "auto"
            Only used when ``scale="probability"``. ``"eigen"`` uses the
            shared eigendecomposition cache (O(n²) per draw); ``"sparse"``
            uses one sparse LU per draw plus a Hutchinson diagonal
            estimator (O(nnz) per draw); ``"auto"`` picks sparse when
            :math:`n` exceeds :attr:`_PROBABILITY_EFFECTS_EIGEN_MAX_N`
            (default 2000).
        """
        from ...diagnostics.spatial_effects import _build_effects_dataframe

        if scale == "logodds":
            return super().spatial_effects(
                return_posterior_samples=return_posterior_samples
            )
        if scale != "probability":
            raise ValueError("scale must be either 'logodds' or 'probability'.")

        self._require_fit()
        direct_samples, indirect_samples, total_samples = (
            self._compute_probability_scale_spatial_effects_posterior(method=method)
        )

        k_effects = direct_samples.shape[1]
        if (
            hasattr(self, "_wx_feature_names")
            and len(self._wx_feature_names) == k_effects
        ):
            feature_names = list(self._wx_feature_names)
        elif (
            hasattr(self, "_nonintercept_feature_names")
            and len(self._nonintercept_feature_names) == k_effects
        ):
            feature_names = list(self._nonintercept_feature_names)
        else:
            feature_names = list(self._feature_names[:k_effects])

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type=self.__class__.__name__,
        )
        df.attrs["scale"] = scale

        if return_posterior_samples:
            posterior_samples = {
                "direct": direct_samples,
                "indirect": indirect_samples,
                "total": total_samples,
            }
            return df, posterior_samples
        return df
