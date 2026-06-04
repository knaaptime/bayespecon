.. _api_ref:

.. currentmodule:: bayespecon

API reference
=============

The public API of ``bayespecon`` is everything reachable from the
top-level ``bayespecon`` namespace (model classes, samplers, DGPs,
diagnostics, graph helpers).  Internal subpackages are prefixed with
an underscore (``_logdet``, ``_ops``, ``_backends``, ``_jax_dispatch``,
``_numba_dispatch``); they are documented here for reference but their
APIs are not covered by the package's stability guarantees and may
change without deprecation.


Base Classes
------------

.. currentmodule:: bayespecon.models.base

.. autosummary::
   :toctree: generated/

   SpatialModel

.. currentmodule:: bayespecon.models.panel_base

.. autosummary::
   :toctree: generated/

   SpatialPanelModel


Cross-Sectional Spatial Models
------------------------------

.. currentmodule:: bayespecon

.. autosummary::
   :toctree: generated/

   OLS
   SAR
   SEM
   SLX
   SDM
   SDEM


Panel Spatial Models (Fixed Effects)
------------------------------------

.. autosummary::
   :toctree: generated/

   OLSPanelFE
   SARPanelFE
   SEMPanelFE
   SDMPanelFE
   SDEMPanelFE
   SLXPanelFE


Panel Spatial Models (Random Effects)
-------------------------------------

.. autosummary::
   :toctree: generated/

   OLSPanelRE
   SARPanelRE
   SEMPanelRE
   SDEMPanelRE


Dynamic Panel Spatial Models
----------------------------

.. autosummary::
   :toctree: generated/

   OLSPanelDynamic
   SDMRPanelDynamic
   SDMUPanelDynamic
   SARPanelDynamic
   SEMPanelDynamic
   SDEMPanelDynamic
   SLXPanelDynamic


Non-Linear Spatial Models
-------------------------

.. autosummary::
   :toctree: generated/

   SpatialProbit
   SARTobit
   SEMTobit
   SDMTobit
   SARNegativeBinomial
   SARNegativeBinomialNUTS
   SARNegBinLatent


Panel Spatial Models (Tobit)
----------------------------

.. autosummary::
   :toctree: generated/

   SARPanelTobit
   SEMPanelTobit


Flow Models
-----------

.. currentmodule:: bayespecon.models.flow

.. autosummary::
   :toctree: generated/

   FlowModel
   OLSFlow
   PoissonFlow
   SARFlow
   SARFlowSeparable
   PoissonSARFlow
   PoissonSARFlowSeparable
   NegativeBinomialFlow
   NegativeBinomialSARFlow
   NegativeBinomialSARFlowSeparable
   SARNegBinFlowLatent
   SARNegBinFlowSeparableLatent
   SEMFlow
   SEMFlowSeparable


Panel Flow Models
^^^^^^^^^^^^^^^^^

.. currentmodule:: bayespecon.models.flow_panel

.. autosummary::
   :toctree: generated/

   FlowPanelModel
   OLSFlowPanel
   PoissonFlowPanel
   SARFlowPanel
   SARFlowSeparablePanel
   PoissonSARFlowPanel
   PoissonSARFlowSeparablePanel
   NegativeBinomialFlowPanel
   NegativeBinomialSARFlowPanel
   NegativeBinomialSARFlowSeparablePanel
   SEMFlowPanel
   SEMFlowSeparablePanel


Bayesian Diagnostics
--------------------

.. currentmodule:: bayespecon.diagnostics.lmtests

.. autosummary::
   :toctree: generated/

   BayesianLMTestResult
   bayesian_lm_lag_test
   bayesian_lm_error_test
   bayesian_lm_wx_test
   bayesian_lm_sdm_joint_test
   bayesian_lm_slx_error_joint_test
   bayesian_lm_error_from_sar_test
   bayesian_lm_error_sdm_test
   bayesian_lm_lag_sdem_test
   bayesian_lm_wx_sem_test
   bayesian_robust_lm_lag_test
   bayesian_robust_lm_error_test
   bayesian_robust_lm_lag_sdm_test
   bayesian_robust_lm_wx_test
   bayesian_robust_lm_error_sdem_test
   bayesian_robust_lm_error_sar_test
   bayesian_robust_lm_error_sdm_test
   bayesian_robust_lm_lag_sdem_test
   bayesian_robust_lm_lag_sem_test
   bayesian_robust_lm_wx_sem_test


Panel Bayesian LM Tests
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   bayesian_panel_lm_lag_test
   bayesian_panel_lm_error_test
   bayesian_panel_robust_lm_lag_test
   bayesian_panel_robust_lm_error_test
   bayesian_panel_lm_wx_test
   bayesian_panel_lm_sdm_joint_test
   bayesian_panel_lm_slx_error_joint_test
   bayesian_panel_lm_error_sdm_test
   bayesian_panel_lm_lag_sdem_test
   bayesian_panel_lm_wx_sem_test
   bayesian_panel_robust_lm_lag_sdm_test
   bayesian_panel_robust_lm_wx_test
   bayesian_panel_robust_lm_error_sdem_test


Flow Bayesian LM Tests
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   bayesian_lm_flow_dest_test
   bayesian_lm_flow_orig_test
   bayesian_lm_flow_network_test
   bayesian_lm_flow_intra_test
   bayesian_lm_flow_joint_test
   bayesian_robust_lm_flow_dest_test
   bayesian_robust_lm_flow_orig_test
   bayesian_robust_lm_flow_network_test
   bayesian_panel_lm_flow_dest_test
   bayesian_panel_lm_flow_orig_test
   bayesian_panel_lm_flow_network_test
   bayesian_panel_lm_flow_intra_test
   bayesian_panel_lm_flow_joint_test


Diagnostic Test Suites
^^^^^^^^^^^^^^^^^^^^^^

Pre-bundled collections of LM tests used by ``model.spatial_diagnostics()``
and the decision-tree renderers.

.. autosummary::
   :toctree: generated/

   DiagnosticSuite
   get_diagnostic_suite
   OLS_SUITE
   OLS_PANEL_SUITE
   SAR_SUITE
   SAR_PANEL_SUITE
   SAR_NEGBIN_SUITE
   SAR_TOBIT_SUITE
   SEM_SUITE
   SEM_PANEL_SUITE
   SEM_PANEL_DYNAMIC_SUITE
   SLX_SUITE
   SLX_PANEL_SUITE
   SLX_PANEL_DYNAMIC_SUITE
   SDM_SUITE
   SDM_PANEL_SUITE
   SDM_TOBIT_SUITE
   SDEM_SUITE
   SDEM_PANEL_SUITE
   FLOW_SUITE
   FLOW_INTRA_SUITE
   FLOW_PANEL_SUITE


Bayesian Model Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: bayespecon.diagnostics

.. autosummary::
   :toctree: generated/

   ModelComparison
   bayes_factor_compare_models
   bic_to_bf
   compile_log_posterior
   post_prob


MCMC Efficiency
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   SpatialMCMCReport
   spatial_mcmc_diagnostic


Spatial Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   SpatialCVResult
   spatial_kfold


Data Generating Processes
-------------------------

.. note::

   The cross-sectional and (scalar) panel DGP simulators accept ``W``
   (Graph/sparse/dense) and ``gdf`` inputs.  You may provide both
   together; in that case ``W`` is used for simulation and is checked
   against ``gdf`` for dimensional compatibility (a ``ValueError`` is
   raised when they do not describe the same number of spatial units).

   The flow DGPs below take ``G`` (libpysal Graph), ``gdf``, ``n``, and
   ``knn_k`` instead.  All four are optional: when none is supplied the
   DGP synthesises a point grid via
   :func:`~bayespecon.dgp.utils.synth_point_geodataframe` and builds a
   row-standardised KNN graph automatically.

.. currentmodule:: bayespecon.dgp

.. autosummary::
   :toctree: generated/

   simulate_ols
   simulate_sar
   simulate_sem
   simulate_slx
   simulate_sdm
   simulate_sdem
   simulate_sar_negbin
   simulate_spatial_probit
   simulate_sar_tobit
   simulate_sem_tobit
   simulate_sdm_tobit
   simulate_panel_ols_fe
   simulate_panel_sar_fe
   simulate_panel_sem_fe
   simulate_panel_sdm_fe
   simulate_panel_sdem_fe
   simulate_panel_slx_fe
   simulate_panel_ols_re
   simulate_panel_sar_re
   simulate_panel_sem_re
   simulate_panel_dlm_fe
   simulate_panel_sdmr_fe
   simulate_panel_sdmu_fe
   simulate_panel_sar_dynamic_fe
   simulate_panel_sem_dynamic_fe
   simulate_panel_sdem_dynamic_fe
   simulate_panel_slx_dynamic_fe
   simulate_panel_sar_tobit_fe
   simulate_panel_sem_tobit_fe


Flow Data Generating Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each flow DGP accepts ``n``, ``G``, ``gdf``, and ``knn_k`` as optional
arguments and (unless ``gamma_dist=0.0``) appends a ``log_distance``
column ``log(1 + d_{ij})`` with default coefficient ``gamma_dist=-0.5``.

The Gaussian flow DGPs (``generate_flow_data``,
``generate_panel_flow_data`` and their separable variants) default to
``distribution="lognormal"``, returning strictly-positive flows
``y = exp(eta)`` where ``eta`` is the latent SAR-filtered linear
predictor (also exposed in the result dict as ``"eta_vec"`` /
``"eta"``).  Pass ``distribution="normal"`` to recover the legacy
Gaussian-on-y behaviour.  The Gaussian-likelihood flow models in
``bayespecon.models.flow`` operate on the latent scale, so fit on
``np.log(y)`` to recover the SAR parameters.  The Poisson and Negative
Binomial DGPs are unchanged.

.. autosummary::
   :toctree: generated/

   generate_flow_data
   generate_flow_data_separable
   generate_poisson_flow_data
   generate_poisson_flow_data_separable
   generate_negbin_flow_data
   generate_negbin_flow_data_separable
   generate_sem_flow_data
   generate_sem_flow_data_separable
   generate_panel_flow_data
   generate_panel_flow_data_separable
   generate_panel_poisson_flow_data
   generate_panel_poisson_flow_data_separable
   generate_panel_negbin_flow_data
   generate_panel_negbin_flow_data_separable
   generate_panel_sem_flow_data
   generate_panel_sem_flow_data_separable


Graph Utilities
---------------

.. currentmodule:: bayespecon.graph

.. autosummary::
   :toctree: generated/

   FlowDesignMatrix
   flow_design_matrix
   flow_design_matrix_asymmetric
   flow_design_matrix_with_orig
   flow_weight_matrices
   destination_weights
   origin_weights
   network_weights


Gibbs Samplers
--------------

Block-Gibbs samplers for Gaussian spatial models.  These bypass NUTS
entirely and exploit conditional conjugacy for faster sampling.  All
four symbols are re-exported from the top-level ``bayespecon``
namespace.

.. currentmodule:: bayespecon

.. autosummary::
   :toctree: generated/

   GibbsEstimation
   GaussianSARGibbs
   GaussianSEMGibbs
   GaussianGibbsPriors


Configuration
-------------

.. currentmodule:: bayespecon

.. autosummary::
   :toctree: generated/

   enable_compile_cache


Internal Modules
----------------

The following subpackages are private (underscore-prefixed).  They
back the public model classes and samplers and may change without
deprecation; they are documented here for reference and for users
writing custom extensions.

Custom PyTensor Ops (``bayespecon._ops``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Differentiable sparse and Kronecker-structured linear solves with
adjoint-method gradients, used inside the flow-model PyMC graphs.

.. currentmodule:: bayespecon._ops

.. autosummary::
   :toctree: generated/

   SparseFlowSolveOp
   SparseFlowSolveMatrixOp
   KroneckerFlowSolveOp
   KroneckerFlowSolveMatrixOp
   kron_solve_vec
   kron_solve_matrix


Log-Determinant Methods (``bayespecon._logdet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: bayespecon._logdet

Positive-only methods (``sparse_spline``, ``grid_mc``) auto-restrict
the rho/lambda support to ``[1e-5, 1.0]`` when the prior or method
default would otherwise admit negative values.  Explicit
``rho_min``/``rho_max`` overrides still raise.

Method resolution and bound handling
""""""""""""""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   LogDetMethod
   LogdetBounds
   resolve_logdet_method
   resolve_logdet_bounds

Factories
"""""""""

.. autosummary::
   :toctree: generated/

   make_logdet_fn
   make_logdet_numpy_fn
   make_logdet_numpy_vec_fn
   make_logdet_jax_fn
   get_cached_logdet_fn
   clear_logdet_fn_cache

PyTensor kernels
""""""""""""""""

.. autosummary::
   :toctree: generated/

   logdet_eigenvalue
   logdet_exact
   logdet_chebyshev
   logdet_interpolated
   logdet_mc_poly_pytensor

JAX kernels
"""""""""""

.. autosummary::
   :toctree: generated/

   jax_logdet_chebyshev
   jax_logdet_trace_poly

Grid / polynomial primitives
""""""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   mc
   chebyshev
   spline

Flow log-determinants
"""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   flow_logdet_pytensor
   flow_logdet_numpy
   compute_flow_traces
   make_flow_separable_logdet
   make_flow_separable_logdet_numpy
