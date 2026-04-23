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

    SAR
      OLS
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

Panel Spatial Models (Random Effects)
--------------------------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   OLSPanelRE
   SARPanelRE
   SEMPanelRE

Dynamic Panel Spatial Models
----------------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   DLMPanelFE
   SDMRPanelFE
   SDMUPanelFE


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
   summarize_bayesian_lm_test

.. currentmodule:: bayespecon.diagnostics.bayesfactor

.. autosummary::
   :toctree: generated/

   bayes_factor_compare_models
   bic_to_bf
   compile_log_posterior
   post_prob


Data Generating Processes
-------------------------

.. note::

   All DGP simulators accept ``W`` (Graph/sparse/dense) and ``gdf`` inputs.
   You may provide both together. In that case, ``W`` is used for simulation
   and checked against ``gdf`` for dimensional compatibility; a ``ValueError``
   is raised when they do not describe the same number of spatial units.

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
   simulate_panel_ols_re
   simulate_panel_sar_re
   simulate_panel_sem_re
   simulate_panel_dlm_fe
   simulate_panel_sdmr_fe
   simulate_panel_sdmu_fe
   simulate_panel_sar_tobit_fe
   simulate_panel_sem_tobit_fe
