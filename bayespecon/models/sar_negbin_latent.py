r"""Structural-form SAR Negative Binomial with Pólya–Gamma Gibbs sampler.

.. math::

    y_i \sim \mathrm{NegBin}(\exp(\eta_i), \alpha), \quad
    \eta = \rho W \eta + X\beta + \nu, \quad
    \nu \sim N(0, \sigma^2 I)

Same observable likelihood as :class:`SARNegativeBinomial` (reduced form)
but sampled via Pólya–Gamma data augmentation and block Gibbs, yielding
substantially higher ESS/s for n > ~1000.

Use this model when:
- n is large (> 1000) and NUTS ESS/s is poor.
- You need reliable ρ and α posteriors without long tuning.
- You want to compare with the NUTS path for validation.

Use :class:`SARNegativeBinomial` when:
- n is small (< 500) and NUTS works fine.
- You need the full PyMC model graph for custom inference.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import scipy.sparse as sp

from .._samplers._chain_runner import run_chains
from .._samplers._idata import gibbs_to_inference_data
from .._samplers._jax_gibbs import run_chain_jax
from .._samplers._spatial_normal import CholmodFactor, has_cholmod
from .._samplers.pg_gibbs import GibbsCache, GibbsPriors, GibbsState, run_chain
from ..diagnostics.lmtests import SAR_NEGBIN_SUITE
from ..logdet import make_logdet_numpy_fn
from .base import SpatialModel


class SARNegBinLatent(SpatialModel):
    """Bayesian structural-form SAR-NB with Pólya–Gamma Gibbs sampler.

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method
        Same interface as :class:`SARNegativeBinomial`.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.

    Notes
    -----
    The structural form parameterises the latent log-mean as
    ``eta = rho * W @ eta + X @ beta + nu`` with ``nu ~ N(0, sigma^2 I)``,
    and augments the NB likelihood with Pólya–Gamma auxiliary variables
    to obtain fully conjugate Gibbs updates for eta, beta, and sigma^2.

    The sampler bypasses PyMC's NUTS entirely. It produces an
    ``arviz.InferenceData`` object compatible with all downstream
    diagnostics (``spatial_diagnostics()``, ``spatial_effects()``,
    ``summary()``).

    The ``fit()`` method does **not** accept ``nuts_sampler`` or
    ``target_accept`` kwargs — these are NUTS-specific and will raise
    ``TypeError`` if passed.

    α (NB dispersion) mixing can be slower than ρ or β. Monitor ESS
    for α specifically and use longer runs if needed.
    """

    _spatial_diagnostics_tests = SAR_NEGBIN_SUITE.tests

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError(
                "robust=True is not supported for SARNegBinLatent."
            )

        # Validate y is integer and non-negative
        y_round = np.round(self._y).astype(np.int64)
        if not np.allclose(self._y, y_round):
            raise ValueError("SARNegBinLatent requires integer-valued observations.")
        if np.any(y_round < 0):
            raise ValueError(
                "SARNegBinLatent requires non-negative integer observations."
            )
        self._y_int = y_round
        self._y = y_round.astype(np.float64)

        # Precompute logdet callable for the ρ slice sampler.
        # Small n: eigenvalue method (one-time O(n³), then O(n) per eval).
        # Large n: Chebyshev approximation (one-time O(nnz * order), then
        # O(order) per eval via Clenshaw recurrence).
        self._logdet_fn = self._make_logdet_fn()

    def _make_logdet_fn(self):
        """Build a callable logdet(rho) -> float for the ρ slice sampler."""
        bounds = self._logdet_bounds
        return make_logdet_numpy_fn(
            self._W_sparse,
            eigs=self._W_eigs.real if self._W_eigs is not None else None,
            method=self._resolved_logdet_method,
            rho_min=bounds.rho_min,
            rho_max=bounds.rho_max,
        )

    # ------------------------------------------------------------------
    # Auto-selection for decoupled Gibbs path
    # ------------------------------------------------------------------

    # Thresholds for switching from factorisation to iterative methods.
    _CG_SOLVE_THRESHOLD: int = 5000
    _LANCZOS_LOGDET_THRESHOLD: int = 5000
    _CHEBYSHEV_SAMPLE_THRESHOLD: int = 5000
    # Threshold for JAX dense backend (O(n²) memory, competitive matvec)
    _JAX_DENSE_THRESHOLD: int = 5000

    def _resolve_solve_method(self, n: int, cholmod_factor) -> str:
        """Choose the solve method for P_η in the ρ slice sampler.

        Parameters
        ----------
        n : int
            Number of spatial units.
        cholmod_factor : CholmodFactor or None
            CHOLMOD factor if available.

        Returns
        -------
        method : str
            One of "cholmod", "splu", "cg".
        """
        if n < self._CG_SOLVE_THRESHOLD:
            # Small n: factorisation is fast and exact.
            if cholmod_factor is not None:
                return "cholmod"
            return "splu"
        else:
            # Large n: CG avoids O(nnz^{1.5}) factorisation cost.
            return "cg"

    def _resolve_logdet_P_method(self, n: int, cholmod_factor) -> str:
        """Choose the log|P_η| method for the ρ slice sampler.

        Parameters
        ----------
        n : int
            Number of spatial units.
        cholmod_factor : CholmodFactor or None
            CHOLMOD factor if available.

        Returns
        -------
        method : str
            One of "cholmod", "lanczos".
        """
        if n < self._LANCZOS_LOGDET_THRESHOLD:
            # Small n: factorisation-based logdet is exact.
            if cholmod_factor is not None:
                return "cholmod"
            # splu path: logdet from LU diagonal
            return "cholmod"  # same code path, just via splu
        else:
            # Large n: Lanczos avoids factorisation entirely.
            return "lanczos"

    def _resolve_sample_method(self, n: int, cholmod_factor) -> str:
        """Choose the η draw method for the Gibbs sampler.

        Parameters
        ----------
        n : int
            Number of spatial units.
        cholmod_factor : CholmodFactor or None
            CHOLMOD factor if available.

        Returns
        -------
        method : str
            One of "cholmod", "splu", "chebyshev".
        """
        if n < self._CHEBYSHEV_SAMPLE_THRESHOLD:
            # Small n: factorisation is fast and exact.
            if cholmod_factor is not None:
                return "cholmod"
            return "splu"
        else:
            # Large n: Chebyshev polynomial avoids factorisation.
            return "chebyshev"

    def _initialize_from_glm(self, rng):
        """Warm-start the Gibbs sampler from a non-spatial NB GLM fit.

        Returns a GibbsState with reasonable starting values for all
        parameters.
        """
        y = self._y
        X = self._X
        n, k = X.shape

        # Fit a simple NB GLM via method of moments for warm start
        # β₀: OLS on log(y + 0.5) as a rough estimate
        y_offset = np.log(np.maximum(y, 0.5))
        try:
            beta_init = np.linalg.lstsq(X, y_offset, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_init = np.zeros(k)

        # η₀: X @ β₀ (no spatial structure)
        eta_init = X @ beta_init

        # σ²₀: residual variance from OLS
        residuals = y_offset - eta_init
        sigma2_init = max(float(residuals @ residuals) / n, 0.01)

        # ρ₀: start at 0 (no spatial dependence)
        rho_init = 0.0

        # α₀: method of moments estimate
        mu_init = np.exp(eta_init)
        np.mean(mu_init)
        y_mean = np.mean(y)
        y_var = np.var(y)
        if y_var > y_mean and y_mean > 0:
            alpha_init = y_mean**2 / (y_var - y_mean)
            alpha_init = max(alpha_init, 0.1)
        else:
            alpha_init = 1.0

        # ω₀: draw from PG(y + α, η)
        from .._samplers._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(y + alpha_init, eta_init, rng=rng)

        return GibbsState(
            eta=eta_init,
            beta=beta_init,
            sigma2=sigma2_init,
            rho=rho_init,
            alpha=alpha_init,
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
            Thinning is for memory management, not statistical efficiency.
        return_eta : bool
            If True, store the full latent field η in the posterior.
            Default False — η is n × draws × chains, which can be large.
            A scalar summary ||η||² is always stored.
        n_jobs : int
            Number of parallel chains. -1 = all CPUs.
        progressbar : bool
            Show per-chain progress bars.
        gibbs_method : str, default "auto"
            Which Gibbs sampler path to use:

            - ``"auto"``: select based on problem size and JAX
              availability.  When JAX is installed and n ≤ 5000, uses
              ``"jax_dense"`` for maximum speed.  Otherwise falls back
              to factorisation (n ≤ 5000) or iterative (n > 5000).
            - ``"factorize"``: force factorisation-based path (CHOLMOD if
              available, else ``scipy.sparse.linalg.splu``). Exact but
              O(nnz^{1.5}) for the factorisation step.
            - ``"iterative"``: force iterative path (CG solve + Lanczos
              logdet + Chebyshev polynomial draw). Approximate but
              avoids factorisation entirely — preferred for large n.
            - ``"jax_dense"``: force JAX-accelerated path (dense matvec
              + vmap over Lanczos probes and Chebyshev draws).  3–4×
              faster for single draws, 20–27× per-draw when batching
              Chebyshev draws.  Requires JAX with float64 enabled.
              Only viable for n ≤ ~5000 due to O(n²) memory.

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
                    f"SARNegBinLatent.fit() does not accept '{bad_kwarg}'. "
                    f"This model uses a Gibbs sampler, not NUTS. "
                    f"Use SARNegativeBinomial for NUTS-based sampling."
                )

        y = self._y
        X = self._X
        W_sparse = self._W_sparse
        n, k = X.shape

        # Build priors
        priors = GibbsPriors(
            beta_mu=self.priors.get("beta_mu", 0.0),
            beta_sigma=self.priors.get("beta_sigma", 1e6),
            sigma_sigma=self.priors.get("sigma_sigma", 10.0),
            alpha_sigma=self.priors.get("alpha_sigma", 10.0),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        # Build cache
        XtX = X.T @ X

        # Create CHOLMOD factor for the precision matrix sparsity pattern.
        # P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
        # The sparsity pattern is the union of diag, W+W^T, and W^T W,
        # which is the same for any ρ≠0.  Use a representative ρ to build
        # the pattern matrix (ρ=0 gives a diagonal matrix, which has the
        # wrong pattern and forces CHOLMOD to redo symbolic analysis each
        # call — very expensive).
        # Precompute matrix pieces for the precision expansion:
        # P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
        # These are constant across the entire sampling run.
        # Division by σ² happens at runtime since σ² changes each iteration.
        W_sym = W_sparse + W_sparse.T
        WtW = W_sparse.T @ W_sparse

        # Create CHOLMOD factor for the precision matrix sparsity pattern.
        # P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
        # The sparsity pattern is the union of diag, W+W^T, and W^T W,
        # which is the same for any ρ≠0.  Use a representative ρ to build
        # the pattern matrix (ρ=0 gives a diagonal matrix, which has the
        # wrong pattern and forces CHOLMOD to redo symbolic analysis each
        # call — very expensive).
        if has_cholmod():
            # Any ρ≠0 gives the correct pattern; ρ=0.5 is arbitrary.
            _P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
            cholmod_factor = CholmodFactor(_P0)
        else:
            cholmod_factor = None

        # Resolve Gibbs method based on user choice or auto-selection
        _valid_methods = {"auto", "factorize", "iterative", "jax_dense"}
        if gibbs_method not in _valid_methods:
            raise ValueError(
                f"gibbs_method must be one of {_valid_methods}, got '{gibbs_method}'"
            )

        # Check JAX availability for jax_dense path
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
        elif gibbs_method == "iterative":
            solve_method = "cg"
            logdet_P_method = "lanczos"
            sample_method = "chebyshev"
        elif gibbs_method == "jax_dense":
            solve_method = "jax_dense"
            logdet_P_method = "jax_dense"
            sample_method = "jax_dense"
        else:  # "auto"
            # Prefer JAX dense when available and n is small enough
            if _jax_available and n <= self._JAX_DENSE_THRESHOLD:
                solve_method = "jax_dense"
                logdet_P_method = "jax_dense"
                sample_method = "jax_dense"
            else:
                solve_method = self._resolve_solve_method(n, cholmod_factor)
                logdet_P_method = self._resolve_logdet_P_method(n, cholmod_factor)
                sample_method = self._resolve_sample_method(n, cholmod_factor)

        # Precompute JAX dense components if using jax_dense path
        W_sym_dense = None
        WtW_dense = None
        W_eigs_jax = None
        if solve_method == "jax_dense":
            import jax
            import jax.numpy as jnp

            # Enable float64 for numerical stability
            jax.config.update("jax_enable_x64", True)
            W_sym_dense = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
            WtW_dense = jnp.asarray(WtW.toarray(), dtype=jnp.float64)
            W_eigs_jax = jnp.asarray(self._W_eigs.real, dtype=jnp.float64)

        cache = GibbsCache(
            W_sparse=W_sparse,
            XtX=XtX,
            logdet_fn=self._logdet_fn,
            rho_lower=priors.rho_lower,
            rho_upper=priors.rho_upper,
            cholmod_factor=cholmod_factor,
            W_sym_over_s2=W_sym,
            WtW_over_s2=WtW,
            solve_method=solve_method,
            logdet_P_method=logdet_P_method,
            sample_method=sample_method,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            W_eigs=W_eigs_jax,
        )

        # Derive per-chain seeds
        if random_seed is not None:
            parent_ss = np.random.SeedSequence(random_seed)
        else:
            parent_ss = np.random.SeedSequence()
        child_seeds = parent_ss.spawn(chains)
        seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

        # Define the per-chain function
        # When using the full-JIT JAX backend, delegate to run_chain_jax()
        # which composes all Gibbs blocks into a single @jax.jit function,
        # eliminating Python→JAX dispatch overhead (~30ms/call × 6 calls).
        _use_jax_full = sample_method == "jax_dense"

        def _run_one_chain(chain_id, seed):
            rng = np.random.default_rng(seed)
            init = self._initialize_from_glm(rng)
            if _use_jax_full:
                return run_chain_jax(
                    y=y,
                    X=X,
                    W_sparse=W_sparse,
                    W_sym_dense=W_sym_dense,
                    WtW_dense=WtW_dense,
                    W_eigs=W_eigs_jax,
                    priors=priors,
                    init=init,
                    draws=draws,
                    tune=tune,
                    thin=thin,
                    return_eta=return_eta,
                    rng=rng,
                )
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
            )

        # Run chains
        chain_results = run_chains(
            chain_fn=_run_one_chain,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
        )

        # Assemble InferenceData
        # Stack chain results: each key has shape (n_keep, ...) per chain
        param_keys = ["rho", "sigma", "alpha"]
        if return_eta:
            param_keys.append("eta")

        posterior_samples = {}
        for key in param_keys:
            arrays = [c[key] for c in chain_results]
            # Shape: (chains, n_keep, ...) for scalar params
            # Shape: (chains, n_keep, k) for beta
            # Shape: (chains, n_keep, n) for eta
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
        """Not supported — SARNegBinLatent uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            "SARNegBinLatent does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling, or use "
            "SARNegativeBinomial for NUTS-based inference."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns the expected count E[y] = exp(eta) where eta is computed
        from the posterior mean of rho and beta via the reduced form:
        eta = (I - rho * W)^{-1} X @ beta.
        """
        self._require_fit()
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        A_rho_inv = sp.linalg.spsolve(
            sp.eye(self._X.shape[0], format="csr") - rho * self._W_sparse,
            self._X @ beta,
        )
        return np.exp(A_rho_inv)

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute average direct/indirect/total impacts on the log-mean scale."""
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
        """Compute posterior impacts on the log-mean scale for each draw."""
        from ..diagnostics.lmtests import _get_posterior_draws
        from ..diagnostics.spatial_effects import _chunked_eig_means

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        eigs = self._W_eigs.real.astype(np.float64)
        mean_diag = _chunked_eig_means(rho_draws, eigs)
        mean_row_sum = self._batch_mean_row_sum(rho_draws)

        ni = self._nonintercept_indices
        direct_samples = mean_diag[:, None] * beta_draws[:, ni]
        total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
        indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples
