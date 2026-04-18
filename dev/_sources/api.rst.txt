.. _api_ref:

.. currentmodule:: bayespecon

API reference
=============



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


Cross Sectional Spatial Models
--------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

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


Dynamic Panel Spatial Models
----------------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   DLMPanelFE
   SDMRPanelFE
   SDMUPanelFE


Panel Spatial Models (Random Effects)
--------------------------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   OLSPanelRE
   SARPanelRE
   SEMPanelRE


Panel Spatial Models (Tobit)
-----------------------------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   SARPanelTobit
   SEMPanelTobit


Non-Linear Spatial Models
---------

.. currentmodule:: bayespecon.models

.. autosummary::
   :toctree: generated/

   SpatialProbit
   SARTobit
   SEMTobit
   SDMTobit


Regression Diagnostics
----------------------

.. currentmodule:: bayespecon.diagnostics

.. autosummary::
   :toctree: generated/

   rdiagnose_like
   bpagan_test
   arch_test
   ljung_box_q
   outlier_candidates
   panel_residual_structure
   pesaran_cd_test


Spatial Stats Helpers
---------------------

.. currentmodule:: bayespecon.stats

.. autosummary::
   :toctree: generated/

   lmerror
   lmlag
   lmrho
   lmrhorob
   lmsar
   lratios
   moran
   walds
   trans_tslow
   f_sarpanel
   f2_sarpanel
   f_sempanel
   f2_sempanel
   f_sarar_panel
   f2_sarar_panel
   sar_panel_FE_LY
   sem_panel_FE_LY
   sarar_panel_FE_LY
   lm_f_err
   lm_f_sar
   lm_f_joint
   lm_f_err_c
   lm_f_sar_c
   lr_f_err
   lr_f_sar
   lr_f_joint
   lr_f_err_c
   lr_f_sar_c
   prt_fe
   prt_back
   prt_test_fe


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
