"""GibbsEstimation base class for Gaussian spatial Gibbs samplers.

Orchestrates chain running, InferenceData assembly, and method
dispatch for the 3-block Gaussian Gibbs sampler (β, σ², ρ/λ).

Two execution backends are supported:

- ``gibbs_method="numpy"`` (default): Python-loop Gibbs with adaptive
  slice sampling for ρ/λ.  No JAX dependency.
- ``gibbs_method="jax"``: Full-JIT Gibbs with MALA for ρ/λ.  Requires
  JAX and equinox.  Faster per-iteration but has JIT compilation
  overhead on the first call.

Subclasses implement model-specific logic:
- ``_spatial_param_name()``: "rho" or "lam"
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod

import arviz as az
import numpy as np
import scipy.sparse as sp

_log = logging.getLogger(__name__)

from .._utils._idata import gibbs_to_inference_data
from ._chain_runner import run_chains
from ._core import (
    GaussianGibbsCache,
    GaussianGibbsPriors,
    _initialize_gaussian_gibbs,
    run_gaussian_chain,
)


class GibbsEstimation:
    """Base class for Gaussian spatial Gibbs sampler configuration and execution.

    Encapsulates the data, priors, cache, and chain-running logic for
    the 3-block Gibbs sampler (β, σ², ρ/λ).  Subclasses provide
    model-specific details (SAR vs SEM, collapsed vs un-collapsed ρ/λ).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix (for SDM/SDEM, this is [X, WX]).
    W_sparse : csr_matrix of shape (n, n)
        Row-standardised spatial weights matrix.
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed, for SAR/SDM).
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable for arrays of rho values.
    feature_names : list of str
        Names for the columns of X (for InferenceData coords).
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        Wy: np.ndarray | None,
        priors: GaussianGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        model_type: str,
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        T: int = 1,
    ):
        self.y = y
        self.X = X
        self.W_sparse = W_sparse
        self.Wy = Wy
        self.priors = priors
        self.logdet_fn = logdet_fn
        self.logdet_vec_fn = logdet_vec_fn
        self.feature_names = feature_names
        self.model_type = model_type
        self.W_eigs = W_eigs
        self.logdet_method = logdet_method
        self.T = int(T)
        self.n, self.k = X.shape

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
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
        """Run Gibbs chains and assemble InferenceData.

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
            Number of parallel workers for the NumPy path.
            ``-1`` uses all CPUs.  When ``n_jobs=1``, chains run
            sequentially with progress bars.  When ``n_jobs>1``
            (or ``-1``), chains run in parallel via ``joblib``.
            Ignored for the JAX path (use ``chain_method`` instead).
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "numpy"
            Execution backend: ``"numpy"`` for Python-loop Gibbs with
            adaptive slice sampling, or ``"jax"`` for full-JIT Gibbs
            with MALA for ρ/λ.  The JAX path requires JAX and equinox.
        mala_step_size : float, default 0.05
            Initial MALA step size (or RW-MH proposal sd) for the
            JAX path.  Ignored when ``gibbs_method="numpy"``.
        use_mala : bool, default True
            If True, use MALA (gradient-guided proposals) for the
            ρ/λ update in the JAX path.  If False, use random-walk
            Metropolis–Hastings.  Ignored when ``gibbs_method="numpy"``
            or ``use_slice=True``.
        use_slice : bool, default False
            If True, use slice sampling for the ρ/λ update in the
            JAX path.  Slice sampling gives much better ESS per sample
            than MALA.  Takes priority over ``use_mala``.
            Ignored when ``gibbs_method="numpy"``.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.  If None, defaults
            to ``(rho_upper - rho_lower) * 0.1``.  Ignored when
            ``use_slice=False`` or ``gibbs_method="numpy"``.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.
            ``"vectorized"`` uses ``jax.vmap`` for JAX-native
            parallelism (all chains on one device).  ``"sequential"``
            runs chains one after another with progress bars.
            ``"parallel"`` is not supported for the JAX path.
            If None, defaults to ``"vectorized"`` when
            ``gibbs_method="jax"``.  Ignored for the NumPy path
            (use ``n_jobs`` to control parallelism instead).

        Returns
        -------
        az.InferenceData
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.
        """
        # Default chain_method for JAX path
        if chain_method is None:
            chain_method = "vectorized" if gibbs_method == "jax" else None

        if gibbs_method == "jax":
            return self._fit_jax(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                thin=thin,
                n_jobs=n_jobs,
                progressbar=progressbar,
                mala_step_size=mala_step_size,
                use_mala=use_mala,
                use_slice=use_slice,
                slice_width=slice_width,
                chain_method=chain_method,
            )

        # ── NumPy path (default) ──
        # Build cache
        cache = self._build_cache()

        spatial_param = self._spatial_param_name()
        _log.info(f"Gibbs sampling ({chains} chains, 3-block: β, σ², {spatial_param})")
        t_start = time.time()

        # Derive per-chain seeds
        if random_seed is not None:
            parent_ss = np.random.SeedSequence(random_seed)
        else:
            parent_ss = np.random.SeedSequence()
        child_seeds = parent_ss.spawn(chains)
        seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

        # Define per-chain function
        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            rng = np.random.default_rng(seed)
            init = _initialize_gaussian_gibbs(
                self.y,
                self.X,
                cache.XtX_cho,
                self.priors,
                rng,
            )
            return run_gaussian_chain(
                y=self.y,
                X=self.X,
                cache=cache,
                priors=self.priors,
                init=init,
                draws=draws,
                tune=tune,
                thin=thin,
                rng=rng,
                progressbar=progressbar,
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
            )

        # Run chains: n_jobs=1 → sequential, n_jobs≠1 → parallel via joblib
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
            model_type=self.model_type,
        )

        # Assemble InferenceData
        idata = self._assemble_idata(chain_results)
        elapsed = time.time() - t_start
        _log.info(
            f"Sampling {chains} chains for {tune} tune and {draws} draw "
            f"iterations ({chains * tune:,} + {chains * draws:,} draws total) "
            f"took {elapsed:.0f} seconds."
        )
        return idata

    def _fit_jax(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        thin: int = 1,
        n_jobs: int = 1,
        progressbar: bool = True,
        mala_step_size: float = 0.05,
        use_mala: bool = True,
        use_slice: bool = True,
        slice_width: float | None = None,
        chain_method: str = "vectorized",
    ) -> az.InferenceData:
        """Run JAX JIT Gibbs chains and assemble InferenceData.

        Uses MALA, RW-MH, or slice sampling for the ρ/λ update,
        enabling full JIT compilation of the Gibbs step.

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
        n_jobs : int, default 1
            Number of parallel workers. Default is ``1`` (sequential)
            because JAX multithreading is incompatible with process
            forking. Use ``chain_method='vectorized'`` for JAX-native
            parallelism instead.
        progressbar : bool, default True
            Show per-chain progress bars.
        mala_step_size : float, default 0.05
            Initial MALA step size.  Ignored when ``use_slice=True``.
        use_mala : bool, default True
            If True, use MALA for the ρ/λ update.  Ignored when
            ``use_slice=True``.
        use_slice : bool, default False
            If True, use slice sampling for the ρ/λ update.  Gives
            much better ESS per sample than MALA.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.  If None,
            defaults to ``(rho_upper - rho_lower) * 0.1``.
        chain_method : str, default "vectorized"
            How to run multiple chains. ``"sequential"`` runs chains
            one after another with progress bars. ``"vectorized"``
            uses ``jax.vmap`` for JAX-native parallelism (all chains
            on one device). ``"parallel"`` is not supported for the
            JAX path.

        Returns
        -------
        az.InferenceData
        """
        from ._jax import (
            run_chain_jax_gaussian,
            run_chains_jax_gibbs_vectorized,
        )

        # Build JAX-native logdet function
        logdet_jax = self._build_logdet_jax()

        spatial_param = self._spatial_param_name()
        sampler_name = "MALA" if use_mala else "RW-MH"
        method_str = f" ({chain_method})" if chain_method != "sequential" else ""
        _log.info(
            f"JAX Gibbs sampling{method_str} ({chains} chains, {sampler_name}, "
            f"3-block: β, σ², {spatial_param})"
        )
        t_start = time.time()

        # ── Vectorized path: jax.vmap ──
        if chain_method == "vectorized":
            # Derive per-chain seeds
            if random_seed is not None:
                parent_ss = np.random.SeedSequence(random_seed)
            else:
                parent_ss = np.random.SeedSequence()
            child_seeds = parent_ss.spawn(chains)
            seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

            # Build cache for initialization
            cache = self._build_cache()

            # Initialize per-chain states
            inits = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                init = _initialize_gaussian_gibbs(
                    self.y,
                    self.X,
                    cache.XtX_cho,
                    self.priors,
                    rng,
                )
                inits.append(init)

            chain_results = run_chains_jax_gibbs_vectorized(
                y=self.y,
                X=self.X,
                W_sparse=self.W_sparse,
                Wy=self.Wy,
                logdet_jax=logdet_jax,
                logdet_vec_fn=self.logdet_vec_fn,
                priors=self.priors,
                inits=inits,
                draws=draws,
                tune=tune,
                thin=thin,
                jax_seeds=seeds,
                model_type=self.model_type,
                mala_step_size=mala_step_size,
                use_mala=use_mala,
                use_slice=use_slice,
                slice_width=slice_width,
                progressbar=progressbar,
            )

            # Assemble InferenceData
            idata = self._assemble_idata(chain_results)
            elapsed = time.time() - t_start
            mean_accept = np.mean([r["mh_accept_rate"] for r in chain_results])
            _log.info(
                f"Sampling {chains} chains for {tune} tune and {draws} draw "
                f"iterations ({chains * tune:,} + {chains * draws:,} draws total) "
                f"took {elapsed:.0f} seconds."
            )
            _log.info(
                f"{sampler_name} acceptance rate: {mean_accept:.3f} (target: 0.574)"
            )
            return idata

        if chain_method == "parallel":
            raise NotImplementedError(
                "chain_method='parallel' is not supported for the JAX path. "
                "Use chain_method='vectorized' for JAX-native parallelism."
            )

        # Derive per-chain seeds
        if random_seed is not None:
            parent_ss = np.random.SeedSequence(random_seed)
        else:
            parent_ss = np.random.SeedSequence()
        child_seeds = parent_ss.spawn(chains)
        seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

        # Build cache for initialization
        cache = self._build_cache()

        # Define per-chain function
        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            rng = np.random.default_rng(seed)
            init = _initialize_gaussian_gibbs(
                self.y,
                self.X,
                cache.XtX_cho,
                self.priors,
                rng,
            )
            return run_chain_jax_gaussian(
                y=self.y,
                X=self.X,
                W_sparse=self.W_sparse,
                Wy=self.Wy,
                logdet_jax=logdet_jax,
                logdet_vec_fn=self.logdet_vec_fn,
                priors=self.priors,
                init=init,
                draws=draws,
                tune=tune,
                thin=thin,
                rng=rng,
                model_type=self.model_type,
                mala_step_size=mala_step_size,
                use_mala=use_mala,
                use_slice=use_slice,
                slice_width=slice_width,
                progressbar=progressbar,
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
            )

        # Run chains sequentially (JAX sequential path)
        chain_results = run_chains(
            chain_fn=_run_one_chain,
            n_chains=chains,
            seeds=seeds,
            n_jobs=1,
            progressbar=progressbar,
            parallel=False,
            draws=draws,
            tune=tune,
            model_type=self.model_type,
        )

        # Assemble InferenceData
        idata = self._assemble_idata(chain_results)
        elapsed = time.time() - t_start
        mean_accept = np.mean([r["mh_accept_rate"] for r in chain_results])
        _log.info(
            f"Sampling {chains} chains for {tune} tune and {draws} draw "
            f"iterations ({chains * tune:,} + {chains * draws:,} draws total) "
            f"took {elapsed:.0f} seconds."
        )
        _log.info(f"{sampler_name} acceptance rate: {mean_accept:.3f} (target: 0.574)")
        return idata

    def _build_logdet_jax(self) -> callable:
        """Build a JAX-native logdet callable for the JAX Gibbs path.

        Uses ``make_logdet_jax_fn`` from ``bayespecon.logdet`` with the
        model's eigenvalues (if available) or sparse W matrix.

        Returns
        -------
        callable
            JAX-native logdet function ``(rho) -> jax.numpy.ndarray``.
        """
        from ..._logdet import make_logdet_jax_fn

        # Use eigenvalues if available (fastest for JAX path)
        W_input = self.W_eigs if self.W_eigs is not None else self.W_sparse

        return make_logdet_jax_fn(
            W=W_input,
            method=self.logdet_method,
            rho_min=self.priors.rho_lower,
            rho_max=self.priors.rho_upper,
            T=self.T,
        )

    def _build_cache(self) -> GaussianGibbsCache:
        """Build the GibbsCache from model data."""
        from scipy.linalg import cho_factor

        XtX = self.X.T @ self.X
        XtX_cho = cho_factor(XtX)

        return GaussianGibbsCache(
            XtX=XtX,
            XtX_cho=XtX_cho,
            logdet_fn=self.logdet_fn,
            logdet_vec_fn=self.logdet_vec_fn,
            rho_lower=self.priors.rho_lower,
            rho_upper=self.priors.rho_upper,
            model_type=self.model_type,
            Wy=self.Wy,
            W_sparse=self.W_sparse,
        )

    def _assemble_idata(
        self,
        chain_results: list[dict],
    ) -> az.InferenceData:
        """Convert chain output dicts to InferenceData.

        Parameters
        ----------
        chain_results : list of dict
            One dict per chain, each containing parameter trace arrays.

        Returns
        -------
        az.InferenceData
        """
        spatial_param = self._spatial_param_name()

        # Stack chain results
        posterior_samples = {}
        for key in [spatial_param, "sigma"]:
            arrays = [c[key] for c in chain_results]
            posterior_samples[key] = np.stack(arrays, axis=0)  # (chains, n_keep)

        # Also expose sigma² so downstream consumers (e.g. bridge sampling) can
        # evaluate the PyMC model logp, which treats sigma² as the free RV.
        posterior_samples["sigma2"] = posterior_samples["sigma"] ** 2

        # beta has shape (n_keep, k) per chain
        posterior_samples["beta"] = np.stack(
            [c["beta"] for c in chain_results], axis=0
        )  # (chains, n_keep, k)

        # Feature names for coords
        coords = {"coefficient": self.feature_names}
        dims = {"beta": ["coefficient"]}

        # Log-likelihood: shape (chains, n_keep, n)
        log_lik = np.stack(
            [c["log_lik"] for c in chain_results], axis=0
        )  # (chains, n_keep, n)

        # Sample stats: per-draw joint log-likelihood and acceptance rate.
        # ``lp`` is the sum of the pointwise log-likelihood (which already
        # includes the Jacobian correction for SAR/SEM); broadcasting the
        # per-chain ``mh_accept_rate`` scalar across draws gives ArviZ a
        # uniform ``(chain, draw)``-shaped stat without per-step tracking.
        n_keep = log_lik.shape[1]
        lp = log_lik.sum(axis=-1)  # (chains, n_keep)
        accept_per_chain = np.array(
            [c.get("mh_accept_rate", 1.0) for c in chain_results],
            dtype=np.float64,
        )
        acceptance_rate = np.broadcast_to(
            accept_per_chain[:, None], (len(chain_results), n_keep)
        ).copy()
        sample_stats = {"lp": lp, "acceptance_rate": acceptance_rate}

        idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood={"obs": log_lik},
            observed_data={"obs": self.y},
            coords=coords,
            dims=dims,
            sample_stats=sample_stats,
        )

        return idata

    @abstractmethod
    def _spatial_param_name(self) -> str:
        """Return the name of the spatial parameter ('rho' or 'lam')."""
        ...


class GaussianSARGibbs(GibbsEstimation):
    """Gibbs sampler for SAR/SDM Gaussian models.

    3-block sampler: β (conjugate normal), σ² (conjugate Inv-Γ),
    ρ (collapsed slice sampling or MALA).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix (for SDM, this is [X, WX]).
    W_sparse : csr_matrix of shape (n, n)
        Row-standardised spatial weights matrix.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable.
    feature_names : list of str
        Names for the columns of X.
    model_type : str
        "sar" or "sdm".
    W_eigs : ndarray or None
        Real eigenvalues of W (for JAX logdet).
    logdet_method : str or None
        Logdet method for JAX path (auto-selected when None).
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        Wy: np.ndarray,
        priors: GaussianGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        model_type: str = "sar",
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        T: int = 1,
    ):
        super().__init__(
            y=y,
            X=X,
            W_sparse=W_sparse,
            Wy=Wy,
            priors=priors,
            logdet_fn=logdet_fn,
            logdet_vec_fn=logdet_vec_fn,
            feature_names=feature_names,
            model_type=model_type,
            W_eigs=W_eigs,
            logdet_method=logdet_method,
            T=T,
        )

    def _spatial_param_name(self) -> str:
        return "rho"


class GaussianSEMGibbs(GibbsEstimation):
    """Gibbs sampler for SEM/SDEM Gaussian models.

    3-block sampler: β (conjugate normal), σ² (conjugate Inv-Γ),
    λ (conditional slice sampling or MALA).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix (for SDEM, this is [X, WX]).
    W_sparse : csr_matrix of shape (n, n)
        Row-standardised spatial weights matrix.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - lam*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable.
    feature_names : list of str
        Names for the columns of X.
    model_type : str
        "sem" or "sdem".
    W_eigs : ndarray or None
        Real eigenvalues of W (for JAX logdet).
    logdet_method : str or None
        Logdet method for JAX path (auto-selected when None).
    T : int, default 1
        Panel time-period count.
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        priors: GaussianGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        model_type: str = "sem",
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        T: int = 1,
    ):
        super().__init__(
            y=y,
            X=X,
            W_sparse=W_sparse,
            Wy=None,  # SEM doesn't use Wy
            priors=priors,
            logdet_fn=logdet_fn,
            logdet_vec_fn=logdet_vec_fn,
            feature_names=feature_names,
            model_type=model_type,
            W_eigs=W_eigs,
            logdet_method=logdet_method,
            T=T,
        )

    def _spatial_param_name(self) -> str:
        return "lam"
