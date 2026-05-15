"""Probabilistic-programming backend protocol.

Phase 3 of the package refactor introduces an internal abstraction over the
probabilistic-programming library that performs MCMC.  The user-facing API
exposes a small set of validated strings (``"pymc"``, ``"numpyro"``,
``"blackjax"``); internally those strings resolve to objects implementing the
:class:`ProbabilisticBackend` protocol.

For now only the PyMC backend is wired up; the others are explicit stubs that
raise :class:`NotImplementedError`.  Their presence in the registry lets us
validate user input up front (rather than at sampling time) and gives future
phases a stable seam to hang real implementations off of.

This module is intentionally internal — model classes import it directly
rather than re-exporting backend objects through :mod:`bayespecon.models`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

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
    """Minimal protocol for the MCMC backend layer.

    The protocol intentionally mirrors the helpers in
    :mod:`bayespecon.models._sampler` so that the existing ``fit()`` paths
    can be migrated incrementally (Phase 6) without changing call sites.
    Implementations should be cheap to construct and stateless.
    """

    name: str

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
    :mod:`bayespecon.models._sampler`; it changes no behaviour.  The
    methods exist purely so that future phases can swap implementations.
    """

    name = "pymc"

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


class _NotImplementedBackend:
    """Base for backend stubs that are recognised but not yet implemented."""

    name = ""

    def _raise(self) -> None:
        raise NotImplementedError(
            f"The {self.name!r} backend is recognised but not yet implemented. "
            f"Use backend='pymc'."
        )

    def use_jax_likelihood(self, nuts_sampler: str) -> bool:  # pragma: no cover - stub
        self._raise()
        return False  # unreachable

    def enforce_c_backend(
        self,
        nuts_sampler: str,
        *,
        requires_c_backend: bool,
        model_name: str,
    ) -> str:  # pragma: no cover - stub
        self._raise()
        return nuts_sampler  # unreachable

    def prepare_sample_kwargs(
        self,
        sample_kwargs: dict | None,
        nuts_sampler: str,
    ) -> dict:  # pragma: no cover - stub
        self._raise()
        return {}  # unreachable

    def prepare_idata_kwargs(
        self,
        idata_kwargs: dict | None,
        pymc_model: Any,
        nuts_sampler: str,
    ) -> dict:  # pragma: no cover - stub
        self._raise()
        return {}  # unreachable


class NumPyroBackend(_NotImplementedBackend):
    """Placeholder for a future NumPyro-native backend."""

    name = "numpyro"


class BlackjaxBackend(_NotImplementedBackend):
    """Placeholder for a future BlackJAX-native backend."""

    name = "blackjax"


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
