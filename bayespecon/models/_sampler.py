"""Helpers for selecting the NUTS backend used by ``pm.sample``.

PyMC supports several NUTS samplers via the ``nuts_sampler`` keyword:

* ``"pymc"``    — built-in C/PyTensor implementation (always available).
* ``"blackjax"`` — JAX-backed sampler (typically the fastest on CPU for the
  models in this package).
* ``"numpyro"`` — also JAX-backed; uses NumPyro's NUTS.
* ``"nutpie"``  — Rust-backed.

``bayespecon`` defaults to ``"blackjax"``.  If the requested optional
backend is not importable, we fall back to PyMC's default with a one-time
``UserWarning`` so notebooks and scripts keep working in environments
where the optional dependency is missing.

When the resolved sampler is ``"pymc"``, this module also injects
``compile_kwargs={"mode": "NUMBA"}`` into ``pm.sample`` automatically
(see :func:`prepare_compile_kwargs`) so the C/PyTensor path benefits from
numba's JIT.  Falls back silently to PyTensor's default mode with a
one-time warning when ``numba`` is not installed.  An explicit
``compile_kwargs`` supplied by the caller always wins.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from functools import lru_cache

_DEFAULT_SAMPLER = "blackjax"
_ENV_OVERRIDE = "BAYESPECON_SAMPLER"

_KNOWN_OPTIONAL_SAMPLERS = ("blackjax", "numpyro", "nutpie")


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


def _default_sampler() -> str:
    """Return the package-wide default sampler.

    Honors the ``BAYESPECON_SAMPLER`` environment variable so the test
    suite (and downstream users) can pin a deterministic backend without
    editing every ``fit()`` call.
    """
    return os.environ.get(_ENV_OVERRIDE, _DEFAULT_SAMPLER)


def resolve_sampler(
    sampler: str | None,
    *,
    requires_c_backend: bool = False,
    model_name: str | None = None,
) -> str:
    """Resolve a user-facing sampler name to a value for ``pm.sample(nuts_sampler=...)``.

    Parameters
    ----------
    sampler :
        One of ``"pymc"``, ``"blackjax"``, ``"numpyro"``, ``"nutpie"``,
        ``"default"``, or ``None``.  ``None`` and ``"default"`` mean
        "use the package-wide default" (typically ``"blackjax"``; can be
        overridden via the ``BAYESPECON_SAMPLER`` environment variable).
    requires_c_backend :
        If ``True``, the calling model relies on a custom :class:`pytensor.graph.op.Op`
        that has no JAX dispatch (e.g. the Poisson sparse-flow models that
        wrap :class:`scipy.sparse.linalg.splu`).  Any non-``"pymc"`` request
        is downgraded to ``"pymc"`` with a one-time ``UserWarning``.
    model_name :
        Class name used in the C-backend warning message; ignored when
        ``requires_c_backend`` is ``False``.

    Returns
    -------
    str
        A string suitable for ``pm.sample(nuts_sampler=...)``.  When the
        requested optional backend is not installed, this returns
        ``"pymc"`` and emits a ``UserWarning`` once per backend per process.

    Notes
    -----
    Availability is probed via :func:`importlib.util.find_spec` so the
    optional package is never actually imported just to check.  Results
    are cached.
    """
    if sampler is None or sampler == "default":
        sampler = _default_sampler()
    if sampler == "pymc":
        return "pymc"
    if requires_c_backend and not _jax_dispatches_available():
        _warn_c_backend_once(sampler, model_name or "this model")
        return "pymc"
    if sampler in _KNOWN_OPTIONAL_SAMPLERS:
        if _has_module(sampler):
            return sampler
        _warn_missing_once(sampler)
        return "pymc"
    # Unknown name — pass through and let pm.sample raise a clear error.
    return sampler


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
        The resolved sampler name (output of :func:`resolve_sampler`).

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
def _warn_missing_once(sampler: str) -> None:
    warnings.warn(
        f"sampler={sampler!r} was requested but the {sampler!r} package is "
        "not importable; falling back to PyMC's default NUTS sampler. "
        f"Install {sampler!r} to enable this backend.",
        UserWarning,
        stacklevel=3,
    )


@lru_cache(maxsize=None)
def _warn_c_backend_once(sampler: str, model_name: str) -> None:
    warnings.warn(
        f"sampler={sampler!r} requested but {model_name} uses a custom "
        "PyTensor Op without a JAX dispatch; falling back to "
        "PyMC's default NUTS sampler.",
        UserWarning,
        stacklevel=3,
    )
