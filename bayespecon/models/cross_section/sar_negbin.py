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
from ...samplers._utils._spatial_normal import CholmodFactor, has_cholmod
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
        krylov_degree : int, default 8
            Krylov basis degree for the shift-invert polynomial
            approximation of :math:`(I - \rho W)^{-1} X` inside the
            ρ-slice density.  Higher degree → more accurate approximation
            but more ``lu.solve`` calls per basis build.  Set to 0 to
            disable Krylov acceleration (exact LU per candidate).
        krylov_dmax : float, default 0.15
            Maximum :math:`|\Delta\rho|` for which the Krylov basis is
            used.  Candidates outside this radius get a fresh LU.
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

        # Precompute CHOLMOD pattern for the normal-equations matrix
        # A^T A = I − ρ(W+W^T) + ρ² W^T W  (SPD for any valid ρ).
        # When CHOLMOD is available, the sampler uses this instead of
        # ``splu`` (UMFPACK) to avoid Apple Accelerate BLAS deadlocks
        # on macOS under concurrent process access.
        from ...samplers.negbin_reduced._core import _make_cholmod_pattern

        if has_cholmod():
            W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, n)
            cholmod_factor = CholmodFactor(pattern)
        else:
            W_sym, WtW, cholmod_factor = None, None, None

        rng = np.random.default_rng(random_seed)
        chain_seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            chain_rng = np.random.default_rng(seed)
            beta_init = init_jitter * chain_rng.standard_normal(k)
            rho_init = float(
                np.clip(init_jitter * chain_rng.standard_normal(), -0.5, 0.5)
            )
            alpha_init = float(np.exp(init_jitter * chain_rng.standard_normal()))
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
                cholmod_factor=cholmod_factor,
                W_sym=W_sym,
                WtW=WtW,
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
