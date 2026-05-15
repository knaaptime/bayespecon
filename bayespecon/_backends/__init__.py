"""Probabilistic-programming backend protocol.

Phase 3 of the package refactor introduces an internal abstraction over the
probabilistic-programming library that performs MCMC.  The user-facing API
exposes a small set of validated strings (``"pymc"``, ``"numpyro"``,
``"blackjax"``); internally those strings resolve to objects implementing the
:class:`ProbabilisticBackend` protocol.

All three backends construct the model in PyMC.  The PyMC backend runs PyMC's
own C/PyTensor NUTS sampler, while the NumPyro and BlackJAX backends route the
NUTS step through PyMC's JAX-sampling adapter (``pm.sample(nuts_sampler=...)``)
backed by the requested JAX library.  Selecting a JAX backend forces the
JAX-likelihood code path in ``_build_pymc_model`` and validates that the
required runtime packages are importable, so misconfiguration fails at
construction time instead of mid-sample.

This module is intentionally internal — model classes import it directly
rather than re-exporting backend objects through :mod:`bayespecon.models`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models._sampler import (
    _has_module,
    _jax_dispatches_available,
)
from ..models._sampler import (
    enforce_c_backend as _enforce_c_backend,
)
from ..models._sampler import (
    prepare_compile_kwargs as _prepare_compile_kwargs,
)
from ..models._sampler import (
    prepare_idata_kwargs as _prepare_idata_kwargs,
)
from ..models._sampler import (
    use_jax_likelihood as _use_jax_likelihood,
)


@runtime_checkable
class ProbabilisticBackend(Protocol):
    """Minimal protocol for the MCMC backend layer."""

    name: str

    def resolve_nuts_sampler(self, nuts_sampler: str | None) -> str:
        """Resolve the effective ``nuts_sampler`` string for ``pm.sample``.

        Implementations may override the user-provided value to force a
        particular runtime (e.g. the NumPyro backend always returns
        ``"numpyro"``).
        """
        ...

    def use_jax_likelihood(self, nuts_sampler: str) -> bool:
        """Return ``True`` when ``nuts_sampler`` requires a JAX likelihood path."""
        ...

    def enforce_c_backend(
        self,
        nuts_sampler: str,
        *,
        requires_c_backend: bool,
        model_name: str,
    ) -> str:
        """Downgrade a JAX-backed request to ``"pymc"`` when JAX dispatch is missing."""
        ...

    def prepare_sample_kwargs(
        self,
        sample_kwargs: dict | None,
        nuts_sampler: str,
    ) -> dict:
        """Normalise ``pm.sample`` keyword arguments (e.g. inject compile mode)."""
        ...

    def prepare_idata_kwargs(
        self,
        idata_kwargs: dict | None,
        pymc_model: Any,
        nuts_sampler: str,
    ) -> dict:
        """Normalise ``idata_kwargs`` (e.g. strip log-likelihood on potential-only JAX models)."""
        ...


class PyMCBackend:
    """PyMC implementation of :class:`ProbabilisticBackend`.

    Thin façade over the existing helpers in
    :mod:`bayespecon.models._sampler`.  Honors the user's ``nuts_sampler``
    keyword (defaulting to ``"pymc"``) so existing call sites that expose
    ``nuts_sampler="numpyro"`` etc. via ``sample_kwargs`` continue to work.
    """

    name = "pymc"

    def resolve_nuts_sampler(self, nuts_sampler: str | None) -> str:
        return nuts_sampler or "pymc"

    def use_jax_likelihood(self, nuts_sampler: str) -> bool:
        return _use_jax_likelihood(nuts_sampler)

    def enforce_c_backend(
        self,
        nuts_sampler: str,
        *,
        requires_c_backend: bool,
        model_name: str,
    ) -> str:
        return _enforce_c_backend(
            nuts_sampler,
            requires_c_backend=requires_c_backend,
            model_name=model_name,
        )

    def prepare_sample_kwargs(
        self,
        sample_kwargs: dict | None,
        nuts_sampler: str,
    ) -> dict:
        return _prepare_compile_kwargs(sample_kwargs, nuts_sampler)

    def prepare_idata_kwargs(
        self,
        idata_kwargs: dict | None,
        pymc_model: Any,
        nuts_sampler: str,
    ) -> dict:
        return _prepare_idata_kwargs(idata_kwargs, pymc_model, nuts_sampler)


class _JaxBackend(PyMCBackend):
    """Base for JAX-backed PyMC sampling backends.

    JAX backends build the model in PyMC and call ``pm.sample`` with
    ``nuts_sampler=<self.name>``; PyMC's JAX adapter then drives NUTS via the
    chosen JAX library.  This subclass:

    * fails fast at construction time if the JAX runtime or backend package
      is not importable;
    * forces the resolved ``nuts_sampler`` to match the backend name,
      raising :class:`ValueError` if the caller passed a conflicting
      ``nuts_sampler`` keyword in ``sample_kwargs``;
    * upgrades :func:`enforce_c_backend` from a silent downgrade to a hard
      :class:`NotImplementedError` — the user explicitly asked for a JAX
      runtime, so masking that with a PyMC fallback would be surprising.
    """

    _required_packages: tuple[str, ...] = ()

    def __init__(self) -> None:
        missing = [pkg for pkg in self._required_packages if not _has_module(pkg)]
        if missing:
            joined = ", ".join(missing)
            raise ImportError(
                f"backend={self.name!r} requires {joined} to be installed."
            )

    def resolve_nuts_sampler(self, nuts_sampler: str | None) -> str:
        if nuts_sampler not in (None, self.name):
            raise ValueError(
                f"backend={self.name!r} fixes nuts_sampler to {self.name!r}, "
                f"but received nuts_sampler={nuts_sampler!r}. Either drop the "
                f"nuts_sampler keyword or switch backend."
            )
        return self.name

    def enforce_c_backend(
        self,
        nuts_sampler: str,
        *,
        requires_c_backend: bool,
        model_name: str,
    ) -> str:
        if not requires_c_backend:
            return nuts_sampler
        if _jax_dispatches_available():
            return nuts_sampler
        raise NotImplementedError(
            f"{model_name} uses a custom PyTensor Op without a JAX dispatch, "
            f"so it cannot run under backend={self.name!r}. Use backend='pymc'."
        )


class NumPyroBackend(_JaxBackend):
    """Run NUTS via NumPyro through PyMC's JAX sampling adapter."""

    name = "numpyro"
    _required_packages = ("jax", "numpyro")


class BlackjaxBackend(_JaxBackend):
    """Run NUTS via BlackJAX through PyMC's JAX sampling adapter."""

    name = "blackjax"
    _required_packages = ("jax", "blackjax")


_BACKENDS: dict[str, type[ProbabilisticBackend]] = {
    "pymc": PyMCBackend,
    "numpyro": NumPyroBackend,
    "blackjax": BlackjaxBackend,
}


def available_backends() -> list[str]:
    """Return the sorted list of known backend names."""
    return sorted(_BACKENDS)


def resolve_backend(name: str | ProbabilisticBackend | None) -> ProbabilisticBackend:
    """Resolve a backend identifier to a concrete :class:`ProbabilisticBackend`.

    Parameters
    ----------
    name :
        ``None`` (resolves to ``"pymc"``), a string in :func:`available_backends`,
        or an already-constructed backend instance (returned unchanged).

    Returns
    -------
    ProbabilisticBackend
        A fresh backend instance; cheap to construct.

    Raises
    ------
    TypeError
        If ``name`` is not a string, ``None``, or a backend instance.
    ValueError
        If ``name`` is a string not in :func:`available_backends`.
    ImportError
        If a JAX backend is requested but its required runtime packages are
        not importable.
    """
    if name is None:
        return PyMCBackend()
    if isinstance(name, ProbabilisticBackend):
        return name
    if not isinstance(name, str):
        raise TypeError(
            f"backend must be None, a string, or a ProbabilisticBackend "
            f"instance; got {type(name).__name__}."
        )
    key = name.lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available backends: {available_backends()}."
        )
    return _BACKENDS[key]()


__all__ = [
    "BlackjaxBackend",
    "NumPyroBackend",
    "ProbabilisticBackend",
    "PyMCBackend",
    "available_backends",
    "resolve_backend",
]
