"""Parallel chain dispatch for Gibbs samplers.

Supports sequential execution with rich progress bars, process-based
parallelism via ``joblib.Parallel`` (with per-chain progress bars
fed via a shared-memory counter block), and JAX vectorized chains
via ``jax.vmap``.
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing
import platform
import threading
import warnings
from multiprocessing import shared_memory
from typing import Callable

import numpy as np

_log = logging.getLogger(__name__)


def _initialize_multiprocessing_context(
    mp_ctx: str | None = None,
    *,
    quiet: bool = False,
) -> multiprocessing.context.BaseContext:
    """Pick a safe multiprocessing start method.

    Detects JAX and auto-switches from ``'fork'`` to
    ``'forkserver'``/``'spawn'`` to avoid deadlocks from forking
    a multithreaded JAX runtime.

    Mimics PyMC's ``_initialize_multiprocessing_context`` pattern.

    Parameters
    ----------
    mp_ctx : str or None
        Requested start method. If None, a platform-appropriate
        default is chosen.
    quiet : bool
        Suppress auto-switch log messages.

    Returns
    -------
    multiprocessing.context.BaseContext
        A multiprocessing context with a safe start method.
    """
    user_specified = mp_ctx is not None
    jax_available = importlib.util.find_spec("jax") is not None

    if mp_ctx is None:
        if platform.system() == "Darwin" and platform.processor() == "arm":
            mp_ctx = "fork"  # fastest on macOS ARM
        else:
            mp_ctx = "forkserver"

    ctx = multiprocessing.get_context(mp_ctx)

    if jax_available and ctx.get_start_method() == "fork":
        if user_specified:
            warnings.warn(
                "Using multiprocessing start method 'fork' with JAX installed "
                "is unsafe and may deadlock. Consider passing mp_ctx='forkserver' "
                "or mp_ctx='spawn'.",
                UserWarning,
                stacklevel=2,
            )
        else:
            new_method = (
                "forkserver"
                if "forkserver" in multiprocessing.get_all_start_methods()
                else "spawn"
            )
            ctx = multiprocessing.get_context(new_method)
            if not quiet:
                _log.debug(
                    f"Auto-switched multiprocessing from 'fork' to '{new_method}' "
                    "because JAX is installed."
                )

    return ctx


def run_chains(
    chain_fn: Callable,
    n_chains: int,
    seeds: list[int] | None = None,
    n_jobs: int = -1,
    progressbar: bool = True,
    parallel: bool = False,
    draws: int = 2000,
    tune: int = 1000,
    model_type: str = "sar",
) -> list[dict]:
    """Run n_chains independent Gibbs chains.

    Parameters
    ----------
    chain_fn : callable
        Function with signature
        ``(chain_id, seed, progress_manager=None, chain_id=0) -> dict``
        that runs a single chain and returns a dict of arrays.
    n_chains : int
        Number of independent chains.
    seeds : list of int, optional
        Per-chain seeds. If None, derived from a parent
        ``numpy.random.SeedSequence``.
    n_jobs : int, default -1
        Number of parallel workers when ``parallel=True``.
        ``-1`` uses all CPUs. Ignored when ``parallel=False``.
    progressbar : bool, default True
        Show per-chain progress bars.  In sequential mode
        (``parallel=False``), uses ``rich`` progress bars directly.
        In parallel mode (``parallel=True``), uses ``rich`` progress
        bars updated via a multiprocessing Queue.
    parallel : bool, default False
        If True, run chains in parallel via ``joblib.Parallel``.
        If False, run chains sequentially with progress bars.
    draws : int, default 2000
        Post-warmup draws per chain (used for progress bar setup).
    tune : int, default 1000
        Warmup draws per chain (used for progress bar setup).
    model_type : str, default "sar"
        Model type (used for progress bar display).

    Returns
    -------
    list of dict
        One dict per chain, each containing parameter trace arrays.
    """
    if seeds is None:
        parent_ss = np.random.SeedSequence()
        child_seeds = parent_ss.spawn(n_chains)
        seeds = [int(s.generate_state(1)[0]) for s in child_seeds]
    elif len(seeds) != n_chains:
        raise ValueError(
            f"len(seeds) must equal n_chains, got {len(seeds)} != {n_chains}."
        )

    if parallel:
        # Process-based parallelism via joblib (handles closures)
        import os

        from joblib import Parallel, delayed

        # Cap workers at n_chains: launching more processes than chains
        # wastes spawn/import cost (the extra workers sit idle) and risks
        # BLAS oversubscription.
        if n_jobs is None or n_jobs < 0:
            n_workers = min(n_chains, os.cpu_count() or 1)
        else:
            n_workers = min(n_jobs, n_chains)

        if progressbar:
            # Per-chain progress bars via shared-memory counters + rich.
            # Workers do two int64 stores per iteration (no IPC); a
            # daemon thread on the main process polls the buffer at
            # 10 Hz and updates the rich tasks.
            from .._utils._progress import (
                _ParallelProgressRenderer,
                _SharedCounterReporter,
            )

            # Layout: (n_chains, 2) int64 — [iteration, tuning_flag].
            nbytes = n_chains * 2 * 8
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            try:
                # Initialise: iteration=0 (not started), tuning_flag=1.
                init_buf = np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)
                init_buf[:, 0] = 0
                init_buf[:, 1] = 1
                del init_buf

                renderer = _ParallelProgressRenderer(
                    n_chains=n_chains,
                    draws=draws,
                    tune=tune,
                    model_type=model_type,
                )
                reporters = [
                    _SharedCounterReporter(shm.name, c, n_chains)
                    for c in range(n_chains)
                ]

                stop_event = threading.Event()
                poll_thread = threading.Thread(
                    target=renderer.poll,
                    args=(shm.name, stop_event),
                    kwargs={"interval": 0.1},
                    daemon=True,
                )

                with renderer:
                    poll_thread.start()
                    try:
                        results = Parallel(n_jobs=n_workers)(
                            delayed(chain_fn)(
                                chain_id,
                                seed,
                                progress_manager=reporters[chain_id],
                                chain_id_kw=chain_id,
                            )
                            for chain_id, seed in enumerate(seeds)
                        )
                    finally:
                        stop_event.set()
                        poll_thread.join(timeout=5.0)
                return list(results)
            finally:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        else:
            # No progress bar — silent parallel execution
            results = Parallel(n_jobs=n_workers)(
                delayed(chain_fn)(chain_id, seed) for chain_id, seed in enumerate(seeds)
            )
            return list(results)

    # Sequential execution with progress bars
    from .._utils._progress import GibbsProgressBarManager

    with GibbsProgressBarManager(
        chains=n_chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type=model_type,
    ) as pm:
        results = []
        for chain_id, seed in enumerate(seeds):
            if pm is not None:
                pm.start_chain(chain_id)
            result = chain_fn(
                chain_id,
                seed,
                progress_manager=pm,
                chain_id_kw=chain_id,
            )
            results.append(result)
        return results
