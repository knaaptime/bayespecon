"""Runtime configuration helpers for :mod:`bayespecon`.

This module exposes opt-in helpers that tune global runtime behaviour
(compile caches, etc.) for the JAX and PyTensor toolchains used by the
package. Nothing in this module is invoked at import time — users must
call the helpers explicitly.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["enable_compile_cache"]


def enable_compile_cache(path: str | Path | None = None) -> Path:
    """Enable persistent on-disk compile caches for JAX and PyTensor.

    JAX-backed NUTS samplers (``"numpyro"``, ``"blackjax"``) and the
    PyTensor-compiled PyMC sampler can both reuse compiled artefacts
    across processes when given a stable cache directory. This helper
    sets:

    * ``jax.config.update("jax_compilation_cache_dir", <path>/"jax")``
    * ``pytensor.config.compiledir = <path>/"pytensor"``

    Both subdirectories are created if they do not already exist. The
    helper is idempotent and safe to call from notebooks or scripts at
    startup.

    Parameters
    ----------
    path :
        Root directory for the caches. When omitted, defaults to the
        platform-appropriate user cache directory
        (``platformdirs.user_cache_dir("bayespecon")``).

    Returns
    -------
    pathlib.Path
        The resolved root cache path.

    Notes
    -----
    This is an opt-in helper. The package does not enable caching
    automatically because tests and CI generally prefer ephemeral
    directories.
    """
    if path is None:
        from platformdirs import user_cache_dir

        root = Path(user_cache_dir("bayespecon")) / "compile_cache"
    else:
        root = Path(path)
    jax_dir = root / "jax"
    pytensor_dir = root / "pytensor"
    jax_dir.mkdir(parents=True, exist_ok=True)
    pytensor_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import pytensor

    jax.config.update("jax_compilation_cache_dir", str(jax_dir))
    pytensor.config.compiledir = str(pytensor_dir)
    return root
