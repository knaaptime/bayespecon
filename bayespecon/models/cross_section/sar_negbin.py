r"""Reduced-form spatial autoregressive Negative Binomial (SAR-NB) model.

This is the **canonical** Bayesian SAR-NB model in
:mod:`bayespecon`: spatial dependence enters purely through the mean
propagator :math:`(I - \rho W)^{-1}` and overdispersion is captured by
the Negative Binomial dispersion parameter :math:`\alpha`.  There is no
latent spatial random effect (no :math:`\sigma`).

.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
    \mu = \exp(\eta), \qquad
    \eta = (I - \rho W)^{-1} X \beta.

This is the reduced form that the spatial-econometrics literature
(LeSage & Pace 2009) writes down by default and is the preferred
specification when there is no substantive reason to model an
additional unobserved spatially-smoothed latent field.

Sampling
--------
The default sampler is a P\u00f3lya-Gamma Gibbs sampler over
:math:`(\beta, \rho, \alpha)` (see
:mod:`bayespecon.samplers.negbin_reduced`).  Each sweep performs

* one Pólya-Gamma augmentation step :math:`\omega \mid \cdot`;
* one conjugate-Gaussian update for :math:`\beta` after building
  :math:`\tilde X = A_\rho^{-1} X`;
* one 1-D adaptive slice sample for :math:`\rho`;
* one 1-D slice sample for :math:`\log\alpha`.

See also
--------
SARNegativeBinomialNUTS
    NUTS-sampled structural-form variant (with :math:`\sigma`).  Kept
    primarily for validation and sampler comparison.
SARNegBinLatent
    P\u00f3lya-Gamma Gibbs sampler for the structural form (with
    :math:`\sigma`).  Also exported as ``SARNegativeBinomialLatent``.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import scipy.sparse as sp

from ...samplers._utils._slice import SliceWidthState
from ...samplers._utils._spatial_normal import has_cholmod
from .sar_negbin_nuts import SARNegativeBinomialNUTS


class SARNegativeBinomial(SARNegativeBinomialNUTS):
    r"""Reduced-form Bayesian SAR Negative Binomial sampled with PG-Gibbs.

    Target posterior:

    .. math::

        y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
        \mu = \exp\{(I - \rho W)^{-1} X\beta\}.

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method
        Same interface and semantics as :class:`bayespecon.models.sar.SAR`.
    robust : bool, default False
        Not supported for count outcomes. If True, ``NotImplementedError``
        is raised.

    Notes
    -----
    Inherits the post-fit machinery (spatial effects on the log-mean and
    count scales, count-scale Hutchinson estimator, …) from
    :class:`SARNegativeBinomialNUTS` because those methods depend only
    on the posteriors of :math:`(\rho, \beta)`, which both models share.
    The :math:`\sigma`-specific posterior-mean fitting and the
    PyMC-NUTS build are overridden because the reduced form has no
    :math:`\sigma` parameter and no PyMC graph.
    """

    # ------------------------------------------------------------------
    # PyMC graph
    # ------------------------------------------------------------------
    # The reduced form is fit with a custom PG-Gibbs sampler — no PyMC
    # model is constructed.  Calling ``_build_pymc_model`` would be a
    # programmer error.
    def _build_pymc_model(self):  # pragma: no cover - guard
        raise NotImplementedError(
            "SARNegativeBinomial uses a custom Pólya–Gamma Gibbs sampler "
            "and does not construct a PyMC model.  Use .fit() directly. "
            "If you want the NUTS variant, use SARNegativeBinomialNUTS."
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        init_jitter: float = 0.1,
        progressbar: bool = True,
        n_jobs: int = 1,
        timeout: float | None = None,
        **_unused,
    ) -> "az.InferenceData":
        r"""Sample the posterior via Pólya-Gamma Gibbs.

        Parameters
        ----------
        draws, tune : int
            Post-warmup draws and warmup sweeps per chain.
        chains : int, default 4
            Number of independent chains.
        random_seed : int, optional
            Seed for the per-chain RNGs.  Each chain receives a distinct
            child seed.
        thin : int, default 1
            Keep every ``thin``-th post-warmup draw.
        init_jitter : float, default 0.1
            Std-dev of the Gaussian noise applied to the default
            :math:`(\beta=0, \rho=0, \alpha=1)` initial state.
        progressbar : bool, default True
            Show per-chain progress bars (sequential or parallel,
            depending on ``n_jobs``).
        n_jobs : int, default 1
            Number of parallel worker processes.  ``1`` runs chains
            sequentially in this process (recommended for small problems
            where the per-chain runtime is < a few seconds).  ``-1``
            uses all available CPUs.
        timeout : float or None, default None
            Maximum wall-clock seconds to wait for all chains to finish
            when ``n_jobs != 1``.  If any worker has not returned by this
            deadline, a :class:`TimeoutError` is raised.  Set to ``None``
            to wait indefinitely.  Ignored when ``n_jobs == 1``.
        krylov_degree : int, default 8
            Krylov basis degree for the shift-invert polynomial
            approximation of :math:`(I - \rho W)^{-1} X` inside the
            ρ-slice density.  Higher degree → more accurate approximation
            but more ``lu.solve`` calls per basis build.  Set to 0 to
            disable Krylov acceleration (CG iterative solve per
            candidate).
        krylov_dmax : float, default 0.15
            Maximum :math:`|\Delta\rho|` for which the Krylov basis is
            used.  Candidates outside this radius get a CG iterative
            solve (no factorisation needed).
        n_rho_omega_cycles : int, default 1
            Number of :math:`(\omega, \rho, \beta)` Gibbs cycles per
            sweep.  At high :math:`\rho` with large :math:`\beta_0`,
            the :math:`\rho` conditional mode shifts by ~2 posterior
            stds when :math:`\omega` is redrawn.  A single
            :math:`\omega \to \rho` update leaves the chain lagging
            behind the mode, giving ESS ≈ 6.  Interleaving multiple
            :math:`\omega \to \rho \to \beta` cycles allows
            :math:`\rho` to track the conditional mode, dramatically
            improving ESS.  Each cycle is a valid Gibbs update.
            Default 1 (single cycle, original behaviour).  Set to
            3–10 for data with high :math:`\rho` and large
            :math:`\beta_0`.
        **_unused
            Other arguments accepted for API parity with the PyMC-based
            siblings (e.g. ``target_accept``, ``idata_kwargs``); silently
            ignored.

        Returns
        -------
        arviz.InferenceData
            Posterior draws of ``rho``, ``beta``, ``alpha`` and pointwise
            ``log_likelihood`` for the observed counts.
        """
        # Pop Krylov kwargs from _unused before they're silently swallowed.
        krylov_degree = _unused.pop("krylov_degree", 8)
        krylov_dmax = _unused.pop("krylov_dmax", 0.15)
        n_rho_omega_cycles = _unused.pop("n_rho_omega_cycles", 1)
        from ...samplers._utils._idata import gibbs_to_inference_data
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.negbin_reduced import (
            ReducedGibbsCache,
            ReducedGibbsPriors,
            ReducedGibbsState,
            run_chain,
        )

        n, k = self._X.shape
        bounds = self._logdet_bounds
        rho_lower = float(bounds.rho_min)
        rho_upper = float(bounds.rho_max)

        priors = ReducedGibbsPriors(
            beta_mu=self.priors.get("beta_mu", 0.0),
            beta_sigma=self.priors.get("beta_sigma", 1e6),
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
            rho_lower=rho_lower,
            rho_upper=rho_upper,
        )

        W_csr = self._W_sparse.tocsr()
        W_csc = self._W_sparse.tocsc()
        X = np.ascontiguousarray(self._X, dtype=np.float64)

        # Eigenvalue bounds for CG iterative solver.
        # For A_ρ = I − ρW: λ_min(A_ρ) = 1 − ρ·λ_max(W), λ_max(A_ρ) = 1 − ρ·λ_min(W).
        if self._W_eigs is not None:
            W_eig_max = float(np.max(np.abs(self._W_eigs)))
            W_eig_min = float(np.min(np.real(self._W_eigs)))
        else:
            W_eig_max = 1.0
            W_eig_min = -1.0

        # Precompute CHOLMOD pattern for the normal-equations matrix
        # A^T A = I − ρ(W+W^T) + ρ² W^T W  (SPD for any valid ρ).
        # When CHOLMOD is available, the sampler uses this instead of
        # ``splu`` (UMFPACK) to avoid Apple Accelerate BLAS deadlocks
        # on macOS under concurrent process access.
        #
        # IMPORTANT: We pass the *pattern matrix* (a sparse CSC matrix)
        # to the worker, NOT a CholmodFactor object.  Creating a
        # CholmodFactor in the parent process calls CHOLMOD/BLAS, which
        # accumulates state across many fit() calls and eventually
        # causes deadlocks on macOS.  The worker creates the
        # CholmodFactor from the pattern matrix on its side.
        from ...samplers.negbin_reduced._core import _make_cholmod_pattern

        if has_cholmod():
            W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, n)
            cholmod_pattern = pattern
        else:
            W_sym, WtW, cholmod_pattern = None, None, None

        rng = np.random.default_rng(random_seed)
        chain_seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]

        # --- Smart initialization ---
        # At high ρ, β and ρ are strongly correlated: the intercept
        # β₀ at ρ = 0.8 is much smaller than at ρ = 0 because the
        # spatial multiplier (I − ρW)⁻¹ amplifies Xβ.  Starting the
        # chain at (ρ ≈ 0, β ≈ 0) places it in a completely wrong
        # mode that the Gibbs sampler cannot escape.
        #
        # We use a profile-log-likelihood initialisation on log(y+0.5):
        #   1. For each ρ on a coarse grid, compute
        #      X̃ = (I − ρW)⁻¹ X and OLS β̂ = (X̃ᵀX̃)⁻¹ X̃ᵀ log(y)
        #   2. Pick the (ρ, β) that maximises the Gaussian log-lik
        #   3. Estimate α from method-of-moments on Pearson residuals
        _log_y = np.log(self._y + 0.5)
        _rho_grid = np.arange(0.05, 0.96, 0.05)
        _best_rho, _best_beta, _best_ll = 0.0, np.zeros(k), -np.inf
        for _rho_g in _rho_grid:
            try:
                _A_g = sp.eye(n, format="csc") - _rho_g * W_csc
                _Xtilde_g = sp.linalg.spsolve(_A_g, X)
                _beta_g = np.linalg.lstsq(_Xtilde_g, _log_y, rcond=None)[0]
                _eta_g = _Xtilde_g @ _beta_g
                _sig2_g = float(np.mean((_log_y - _eta_g) ** 2))
                _ll_g = -0.5 * n * np.log(_sig2_g) - 0.5 * n
                if _ll_g > _best_ll:
                    _best_ll = _ll_g
                    _best_rho = _rho_g
                    _best_beta = _beta_g.copy()
            except Exception:
                pass
        _rho_init_mle = float(np.clip(_best_rho, rho_lower + 0.05, rho_upper - 0.05))
        _beta_init_mle = _best_beta
        # Estimate α from the Pearson dispersion of the Gaussian fit
        # on log(y).  The residual variance σ² from the profile
        # log-likelihood gives the dispersion on the log scale.
        # For NB data: Var(log y) ≈ 1/α + 1/(2μ) (delta method), so
        # α ≈ 1/σ² when μ is large.  Cap at a reasonable range.
        try:
            _A_init = sp.eye(n, format="csc") - _rho_init_mle * W_csc
            _Xtilde_init = sp.linalg.spsolve(_A_init, X)
            _eta_init = _Xtilde_init @ _beta_init_mle
            _resid2 = float(np.mean((_log_y - _eta_init) ** 2))
            _alpha_init_mle = float(np.clip(1.0 / max(_resid2, 0.01), 0.5, 50.0))
        except Exception:
            _alpha_init_mle = 1.0

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            chain_rng = np.random.default_rng(seed)
            # Jitter around the profile-loglik estimates.
            # Use a smaller jitter for ρ than for β because the
            # posterior is extremely peaked in ρ at high spatial
            # autocorrelation, and a large ρ jitter can push the
            # chain into a wrong mode that the Gibbs sampler
            # cannot escape.
            _rho_jitter = min(init_jitter, 0.02)
            beta_init = _beta_init_mle + init_jitter * chain_rng.standard_normal(k)
            rho_init = float(
                np.clip(
                    _rho_init_mle + _rho_jitter * chain_rng.standard_normal(),
                    rho_lower + 0.01,
                    rho_upper - 0.01,
                )
            )
            alpha_init = float(
                np.clip(
                    _alpha_init_mle * np.exp(init_jitter * chain_rng.standard_normal()),
                    0.05,
                    50.0,
                )
            )
            omega_init = 0.25 * np.ones(n, dtype=np.float64)
            init = ReducedGibbsState(
                beta=beta_init,
                rho=rho_init,
                alpha=alpha_init,
                omega=omega_init,
            )
            cache = ReducedGibbsCache(
                W_sparse=W_csr,
                W_csc=W_csc,
                rho_lower=rho_lower,
                rho_upper=rho_upper,
                rho_adaptive_width=True,
                rho_slice_width_state=SliceWidthState(w=0.2),
                krylov_degree=krylov_degree,
                krylov_dmax=krylov_dmax,
                cholmod_pattern=cholmod_pattern,
                W_sym=W_sym,
                WtW=WtW,
                W_eig_max=W_eig_max,
                W_eig_min=W_eig_min,
                n_rho_omega_cycles=n_rho_omega_cycles,
            )
            return run_chain(
                y=self._y,
                X=X,
                W_sparse=W_csr,
                priors=priors,
                cache=cache,
                init=init,
                draws=draws,
                tune=tune,
                thin=thin,
                rng=chain_rng,
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
            )

        chain_samples = run_chains(
            chain_fn=_run_one_chain,
            n_chains=chains,
            seeds=chain_seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=(n_jobs != 1),
            draws=draws,
            tune=tune,
            model_type="sar_negbin",
            timeout=timeout,
        )

        # Stack chains: each entry → (chains, draws, ...)
        stacked = {
            "rho": np.stack([s["rho"] for s in chain_samples], axis=0),
            "beta": np.stack([s["beta"] for s in chain_samples], axis=0),
            "alpha": np.stack([s["alpha"] for s in chain_samples], axis=0),
        }
        log_lik = np.stack([s["log_lik"] for s in chain_samples], axis=0)

        idata = gibbs_to_inference_data(
            posterior_samples=stacked,
            log_likelihood={"obs": log_lik},
            observed_data={"obs": self._y_int},
            coords={
                "coefficient": list(self._feature_names),
                "obs_id": list(range(n)),
            },
            dims={
                "beta": ["coefficient"],
                "obs": ["obs_id"],
            },
        )
        self._idata = idata
        return idata

    # ------------------------------------------------------------------
    # Posterior-mean fitted values
    # ------------------------------------------------------------------
    def _fitted_mean_from_posterior(self) -> np.ndarray:
        r"""Posterior-mean fitted expected counts.

        Computes :math:`\exp(\eta)` where
        :math:`\eta = (I - \rho W)^{-1} X\beta` at the posterior means of
        :math:`\rho` and :math:`\beta`.  There is no :math:`\sigma z`
        term because the reduced form has no latent random effect.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        n = self._X.shape[0]
        A = sp.eye(n, format="csr", dtype=np.float64) - rho * self._W_sparse
        eta = sp.linalg.spsolve(A, self._X @ beta)
        return np.exp(eta)
