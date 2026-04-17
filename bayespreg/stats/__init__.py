"""MATLAB-style spatial statistics and panel test helpers.

This namespace collects cross-sectional spatial diagnostics from the MATLAB
``spatial/stats`` toolbox alongside panel fixed-effects tests inspired by
``panel_g`` and ``FESPD_Est_Tests`` routines.

Cross-sectional helpers
-----------------------
- ``lmerror``
- ``lmlag``
- ``lmrho``
- ``lmrhorob``
- ``lmsar``
- ``lratios``
- ``model_probs``
- ``moran``
- ``walds``
- ``prt_spat``
- ``prt_spat2``
- ``prt_old``
- ``plt_spat``

Panel helpers
-------------
- ``trans_tslow``
- ``f_sarpanel``
- ``f2_sarpanel``
- ``f_sempanel``
- ``f2_sempanel``
- ``f_sarar_panel``
- ``f2_sarar_panel``
- ``sar_panel_FE_LY``
- ``sem_panel_FE_LY``
- ``sarar_panel_FE_LY``
- ``lm_f_err``
- ``lm_f_sar``
- ``lm_f_joint``
- ``lm_f_err_c``
- ``lm_f_sar_c``
- ``lr_f_err``
- ``lr_f_sar``
- ``lr_f_joint``
- ``lr_f_err_c``
- ``lr_f_sar_c``
- ``prt_fe``
- ``prt_back``
- ``prt_test_fe``
"""

from .core import (
    lmerror,
    lmlag,
    lmrho,
    lmrhorob,
    lmsar,
    lratios,
    model_probs,
    moran,
    plt_spat,
    prt_old,
    prt_spat,
    prt_spat2,
    walds,
)
from .panel import (
    f2_sarar_panel,
    f2_sarpanel,
    f2_sempanel,
    f_sarar_panel,
    f_sarpanel,
    f_sempanel,
    lm_f_err,
    lm_f_err_c,
    lm_f_joint,
    lm_f_sar,
    lm_f_sar_c,
    lr_f_err,
    lr_f_err_c,
    lr_f_joint,
    lr_f_sar,
    lr_f_sar_c,
    prt_back,
    prt_fe,
    prt_test_fe,
    sar_panel_FE_LY,
    sarar_panel_FE_LY,
    sem_panel_FE_LY,
    trans_tslow,
)

__all__ = [
    "lmerror",
    "lmlag",
    "lmrho",
    "lmrhorob",
    "lmsar",
    "lratios",
    "model_probs",
    "moran",
    "walds",
    "prt_spat",
    "prt_spat2",
    "prt_old",
    "plt_spat",
    "trans_tslow",
    "f_sarpanel",
    "f2_sarpanel",
    "f_sempanel",
    "f2_sempanel",
    "f_sarar_panel",
    "f2_sarar_panel",
    "sar_panel_FE_LY",
    "sem_panel_FE_LY",
    "sarar_panel_FE_LY",
    "lm_f_err",
    "lm_f_sar",
    "lm_f_joint",
    "lm_f_err_c",
    "lm_f_sar_c",
    "lr_f_err",
    "lr_f_sar",
    "lr_f_joint",
    "lr_f_err_c",
    "lr_f_sar_c",
    "prt_fe",
    "prt_back",
    "prt_test_fe",
]
