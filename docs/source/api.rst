.. _api_ref:

.. currentmodule:: bayespecon

API reference
=============



Base Classes
------------

.. currentmodule:: bayespecon.models.base

.. autosummary::
   :toctree: generated/

   SpatialModel :no-index:


.. currentmodule:: bayespecon.models.panel_base

.. autosummary::
   :toctree: generated/

   SpatialPanelModel :no-index:


Cross Sectional Spatial Models
------------------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   OLS
   SAR
   SEM
   SLX
   SDM
   SDEM


Panel Spatial Models
-----------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   OLSPanelFE
   SARPanelFE
   SEMPanelFE
   SDMPanelFE
   SDEMPanelFE
   SLXPanelFE

Panel Spatial Models (Random Effects)
--------------------------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   OLSPanelRE
   SARPanelRE
   SEMPanelRE
   SDEMPanelRE

Dynamic Panel Spatial Models
----------------------------

.. currentmodule:: bayespecon.models

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

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   SpatialProbit :no-index:
   SARTobit
   SEMTobit
   SDMTobit


Panel Spatial Models (Tobit)
-----------------------------

.. currentmodule:: bayespecon.models

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
   SEMFlowPanel
   SEMFlowSeparablePanel



Bayesian Diagnostics
---------------------

.. currentmodule:: bayespecon.diagnostics.bayesian_lmtests

.. autosummary::
   :toctree: generated/

   BayesianLMTestResult
   bayesian_lm_lag_test
   bayesian_lm_error_test
   bayesian_lm_wx_test
   bayesian_lm_sdm_joint_test
   bayesian_lm_slx_error_joint_test
   bayesian_robust_lm_lag_sdm_test
   bayesian_robust_lm_wx_test
   bayesian_robust_lm_error_sdem_test

Panel Bayesian LM Tests
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: bayespecon.diagnostics.bayesian_lmtests

.. autosummary::
   :toctree: generated/

   bayesian_panel_lm_lag_test
   bayesian_panel_lm_error_test
   bayesian_panel_robust_lm_lag_test
   bayesian_panel_robust_lm_error_test
   bayesian_panel_lm_wx_test
   bayesian_panel_lm_sdm_joint_test
   bayesian_panel_lm_slx_error_joint_test
   bayesian_panel_robust_lm_lag_sdm_test
   bayesian_panel_robust_lm_wx_test
   bayesian_panel_robust_lm_error_sdem_test
   bayesian_panel_lm_wx_sem_test


Bayesian Model Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: bayespecon.diagnostics.bayesfactor

.. autosummary::
   :toctree: generated/

   bayes_factor_compare_models
   bic_to_bf
   compile_log_posterior
   post_prob


Log-Determinant Methods
-----------------------

.. currentmodule:: bayespecon.logdet

.. autosummary::
   :toctree: generated/

   make_logdet_fn
   logdet_eigenvalue
   logdet_exact
   logdet_chebyshev
   logdet_interpolated
   mc
   chebyshev
   ilu
   sparse_grid
   spline


Data Generating Processes
-------------------------

.. note::

   The cross-sectional and (scalar) panel DGP simulators accept ``W``
   (Graph/sparse/dense) and ``gdf`` inputs.  You may provide both together;
   in that case ``W`` is used for simulation and is checked against ``gdf``
   for dimensional compatibility (a ``ValueError`` is raised when they do
   not describe the same number of spatial units).

   The flow DGPs below take ``G`` (libpysal Graph), ``gdf``, ``n``, and
   ``knn_k`` instead.  All four are optional: when none is supplied the
   DGP synthesises a point grid via
   :func:`~bayespecon.dgp.utils.synth_point_geodataframe` and builds a
   row-standardised KNN graph automatically.

.. currentmodule:: bayespecon.dgp

.. autosummary::
   :toctree: generated/

   simulate_sar
   simulate_ols
   simulate_sem
   simulate_slx
   simulate_sdm
   simulate_sdem
   simulate_sar_tobit
   simulate_sem_tobit
   simulate_sdm_tobit
   simulate_spatial_probit
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
``np.log(y)`` to recover the SAR parameters.  The Poisson DGPs are
unchanged.

.. currentmodule:: bayespecon.dgp

.. autosummary::
   :toctree: generated/

   generate_flow_data
   generate_flow_data_separable
   generate_poisson_flow_data
   generate_poisson_flow_data_separable
   generate_panel_flow_data
   generate_panel_flow_data_separable
   generate_panel_poisson_flow_data
   generate_panel_poisson_flow_data_separable


Graph Utilities
---------------

.. currentmodule:: bayespecon.graph

.. autosummary::
   :toctree: generated/

   FlowDesignMatrix
   flow_design_matrix
   flow_design_matrix_with_orig
   flow_weight_matrices
   destination_weights
   origin_weights
   network_weights
