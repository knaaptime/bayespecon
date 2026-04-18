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
