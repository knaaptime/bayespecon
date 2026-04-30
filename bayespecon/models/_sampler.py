"""Helpers for tweaking ``pm.sample`` keyword arguments per backend.

PyMC supports several NUTS samplers via the ``nuts_sampler`` keyword:

* ``"pymc"``    — built-in C/PyTensor implementation (always available; default).
* ``"blackjax"`` — JAX-backed sampler.
* ``"numpyro"`` — also JAX-backed; uses NumPyro's NUTS.
* ``"nutpie"``  — Rust-backed.

This module provides three small helpers used by ``fit()`` methods:

* :func:`enforce_c_backend` — for models whose custom :class:`pytensor.graph.op.Op`
  has no JAX dispatch (e.g. Poisson sparse-flow models that wrap
  :class:`scipy.sparse.linalg.splu`), downgrade a JAX-backed request to
  ``"pymc"`` with a one-time ``UserWarning``.
* :func:`prepare_idata_kwargs` — strip ``log_likelihood=True`` for JAX backends
  on potential-only models where PyMC's JAX path would crash.
* :func:`prepare_compile_kwargs` — auto-inject ``compile_kwargs={"mode": "NUMBA"}``
  when the resolved sampler is ``"pymc"`` and ``numba`` is importable.
"""

from __future__ import annotations

import importlib.util
import warnings
from functools import lru_cache


@lru_cache(maxsize=None)
def _has_module(name: str) -> bool:
    """Return ``True`` if ``name`` is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _jax_dispatches_available() -> bool:
    """Return ``True`` if both JAX and PyTensor's JAX dispatch are present.

    When this is true, the custom Ops in :mod:`bayespecon.ops` register their
    own ``jax_funcify`` implementations on import, so models that previously
    required the C backend can sample under ``"blackjax"`` or ``"numpyro"``.
    """
    return _has_module("jax") and _has_module("pytensor.link.jax.dispatch")


def enforce_c_backend(
    nuts_sampler: str,
    *,
    requires_c_backend: bool,
    model_name: str,
) -> str:
    """Downgrade a JAX-backed NUTS request to ``"pymc"`` when JAX dispatch is missing.

    Parameters
    ----------
    nuts_sampler :
        The user-requested ``nuts_sampler`` value (already resolved from
        ``sample_kwargs``).
    requires_c_backend :
        If ``True``, the calling model relies on a custom :class:`pytensor.graph.op.Op`
        that has no JAX dispatch.  Any non-``"pymc"`` request is downgraded to
        ``"pymc"`` with a one-time ``UserWarning``.
    model_name :
        Class name used in the warning message.

    Returns
    -------
    str
        Either the original ``nuts_sampler`` value or ``"pymc"`` if a downgrade
        was forced.
    """
    if not requires_c_backend:
        return nuts_sampler
    if nuts_sampler == "pymc":
        return nuts_sampler
    if _jax_dispatches_available():
        return nuts_sampler
    _warn_c_backend_once(nuts_sampler, model_name)
    return "pymc"


def prepare_idata_kwargs(
    idata_kwargs: dict | None,
    model,
    nuts_sampler: str,
) -> dict:
    """Strip ``log_likelihood=True`` for JAX backends on potential-only models.

    PyMC's JAX sampling path (``pm.sampling.jax._get_log_likelihood``) iterates
    ``model.observed_RVs``; when a model defines its likelihood with
    ``pm.Potential`` (e.g. SEM, SDEM, Tobit, panel SEM), that list is empty
    and the helper raises ``TypeError: 'NoneType' object is not iterable``.
    These models recompute the log-likelihood manually after sampling, so it
    is safe to drop the request before calling ``pm.sample``.
    """
    idata_kwargs = dict(idata_kwargs or {})
    if not idata_kwargs.get("log_likelihood"):
        return idata_kwargs
    if nuts_sampler not in ("blackjax", "numpyro"):
        return idata_kwargs
    if getattr(model, "observed_RVs", None):
        return idata_kwargs
    idata_kwargs.pop("log_likelihood", None)
    return idata_kwargs


def prepare_compile_kwargs(
    sample_kwargs: dict | None,
    nuts_sampler: str,
) -> dict:
    """Inject ``compile_kwargs={"mode": "NUMBA"}`` for the PyMC sampler.

    Numba is a soft dependency; when it is importable the C/PyTensor
    backend used by ``nuts_sampler="pymc"`` is materially faster under
    the NUMBA mode.  This helper sets that compile mode by default while
    leaving JAX-backed (``"blackjax"``, ``"numpyro"``) and Rust-backed
    (``"nutpie"``) samplers untouched — they ignore ``compile_kwargs``.

    Behaviour:

    * Non-``"pymc"`` sampler → returns ``sample_kwargs`` unchanged.
    * ``"compile_kwargs"`` already present (caller override, including the
      empty dict ``{}``) → returns ``sample_kwargs`` unchanged.
    * ``numba`` importable → returns a copy with
      ``compile_kwargs={"mode": "NUMBA"}`` inserted.
    * ``numba`` missing → returns ``sample_kwargs`` unchanged and emits a
      one-time ``UserWarning``.

    Parameters
    ----------
    sample_kwargs :
        The keyword-argument dict eventually splatted into ``pm.sample``.
        ``None`` is treated as an empty dict.
    nuts_sampler :
        The resolved sampler name.

    Returns
    -------
    dict
        A new dict that may have ``compile_kwargs`` added; never mutates
        the input.
    """
    out = dict(sample_kwargs or {})
    if nuts_sampler != "pymc":
        return out
    if "compile_kwargs" in out:
        return out
    if not _has_module("numba"):
        _warn_numba_missing_once()
        return out
    out["compile_kwargs"] = {"mode": "NUMBA"}
    return out


@lru_cache(maxsize=None)
def _warn_numba_missing_once() -> None:
    warnings.warn(
        "numba is not installed; the PyMC NUTS sampler will use PyTensor's "
        "default C compile mode.  Install 'numba' to enable the faster "
        "NUMBA backend.",
        UserWarning,
        stacklevel=3,
    )


@lru_cache(maxsize=None)
def _warn_c_backend_once(sampler: str, model_name: str) -> None:
    warnings.warn(
        f"nuts_sampler={sampler!r} requested but {model_name} uses a custom "
        "PyTensor Op without a JAX dispatch; falling back to "
        "PyMC's default NUTS sampler.",
        UserWarning,
        stacklevel=3,
    )
