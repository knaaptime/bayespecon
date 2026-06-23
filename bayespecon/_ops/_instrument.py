"""Callback instrumentation for Op perform calls.

Gated behind the ``BAYESPECON_OP_INSTRUMENT`` env var (default: off).
"""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager

from .._config import settings as _settings

# Module-level counter ensures every Op instance gets a unique id so that
# pytensor does not incorrectly merge two distinct Op instances during graph
# optimisation (relevant when multiple flow models exist in one Python session).
_op_id_counter = itertools.count()


# Instrumentation is gated behind the ``BAYESPECON_OP_INSTRUMENT`` env var
# (default: off).  When disabled, ``_measure_callback`` is a zero-overhead
# no-op and ``_record_callback`` returns immediately, so Op ``perform``
# calls incur no timing cost.
_INSTRUMENT_ENABLED = _settings.op_instrument

_callback_stats_lock = threading.Lock()
_callback_count = 0
_callback_seconds = 0.0
_callback_by_op: dict[str, dict[str, float]] = {}


def reset_callback_stats() -> None:
    """Reset in-process callback counters used by benchmark instrumentation."""
    global _callback_count, _callback_seconds, _callback_by_op
    with _callback_stats_lock:
        _callback_count = 0
        _callback_seconds = 0.0
        _callback_by_op = {}


def get_callback_stats() -> dict[str, object]:
    """Return callback counter snapshot.

    Returns
    -------
    dict
        Keys:
        - ``count`` : total number of instrumented Op ``perform`` calls.
        - ``seconds`` : total wall-clock time spent in those calls.
        - ``by_op`` : per-op breakdown with ``count`` and ``seconds``.
    """
    with _callback_stats_lock:
        by_op_copy = {
            k: {"count": int(v["count"]), "seconds": float(v["seconds"])}
            for k, v in _callback_by_op.items()
        }
        return {
            "count": int(_callback_count),
            "seconds": float(_callback_seconds),
            "by_op": by_op_copy,
        }


def _record_callback(op_name: str, elapsed_seconds: float) -> None:
    if not _INSTRUMENT_ENABLED:
        return
    global _callback_count, _callback_seconds
    with _callback_stats_lock:
        _callback_count += 1
        _callback_seconds += float(elapsed_seconds)
        bucket = _callback_by_op.setdefault(op_name, {"count": 0, "seconds": 0.0})
        bucket["count"] += 1
        bucket["seconds"] += float(elapsed_seconds)


@contextmanager
def _measure_callback(op_name: str):
    if not _INSTRUMENT_ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _record_callback(op_name, time.perf_counter() - t0)
