"""Centralized environment-variable configuration for :mod:`bayespecon`.

All ``BAYESPECON_*`` env vars are read in one place so defaults and
documentation live here.  Modules read from :data:`settings` instead of
calling ``os.environ.get`` directly.

Env vars
--------
``BAYESPECON_OP_INSTRUMENT`` (default ``"0"``)
    Set to ``"1"`` to enable per-Op callback timing instrumentation.
    Off by default for zero overhead.

``BAYESPECON_KRON_DENSE_MAX`` (default ``512``)
    Largest *n* for which Kronecker Ops use dense LAPACK instead of
    SuperLU.

``BAYESPECON_SPARSE_BACKEND`` (default ``"auto"``)
    Sparse solve backend: ``"auto"``, ``"scipy"``, or ``"umfpack"``.

``BAYESPECON_SPARSE_STRICT`` (default ``"0"``)
    If truthy, missing requested optional backends raise instead of
    falling back.

``BAYESPECON_LOGDET_EIGEN_MAX_N`` (default ``500``)
    Largest *n* for which the eigenvalue logdet method is auto-selected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip().lower()


@dataclass(frozen=True)
class Settings:
    """Snapshot of all ``BAYESPECON_*`` env vars read at import time.

    Use :func:`set_settings` to override values programmatically (e.g. in
    tests).  Changes to env vars after import do **not** affect this
    snapshot unless :func:`reload_settings` is called.
    """

    # _ops / _instrument
    op_instrument: bool
    # _ops / _backend
    kron_dense_max: int
    sparse_backend: str
    sparse_strict: bool
    # _logdet
    logdet_eigen_max_n: int


def _read_settings() -> Settings:
    return Settings(
        op_instrument=_env_bool("BAYESPECON_OP_INSTRUMENT", "0"),
        kron_dense_max=_env_int("BAYESPECON_KRON_DENSE_MAX", 512),
        sparse_backend=_env_str("BAYESPECON_SPARSE_BACKEND", "auto"),
        sparse_strict=_env_bool("BAYESPECON_SPARSE_STRICT", "0"),
        logdet_eigen_max_n=_env_int("BAYESPECON_LOGDET_EIGEN_MAX_N", 500),
    )


settings: Settings = _read_settings()


def reload_settings() -> Settings:
    """Re-read env vars and update :data:`settings` in place.

    Useful in tests that set env vars via ``monkeypatch.setenv`` and need
    the new values to take effect.
    """
    global settings
    settings = _read_settings()
    return settings


def set_settings(**kwargs) -> Settings:
    """Override one or more settings programmatically.

    Example
    -------
    >>> set_settings(kron_dense_max=0)  # force sparse path in tests
    """
    global settings
    current = {
        f.name: getattr(settings, f.name)
        for f in settings.__dataclass_fields__.values()
    }
    current.update(kwargs)
    settings = Settings(**current)  # type: ignore[arg-type]
    return settings
