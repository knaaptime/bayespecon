"""Parallel chain dispatch for Gibbs samplers.

Uses joblib for process-based parallelism with SeedSequence-derived RNGs.
Each chain runs independently in its own process, with its own
``numpy.random.Generator`` seeded from a parent ``SeedSequence``.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from tqdm.auto import tqdm


def run_chains(
    chain_fn: Callable,
    n_chains: int,
    seeds: list[int] | None = None,
    n_jobs: int = -1,
    progressbar: bool = True,
) -> list[dict]:
    """Run n_chains independent Gibbs chains in parallel.

    Parameters
    ----------
    chain_fn : callable
        Function with signature ``(chain_id, seed) -> dict`` that
        runs a single chain and returns a dict of arrays.
    n_chains : int
        Number of independent chains.
    seeds : list of int, optional
        Per-chain seeds. If None, derived from a parent
        ``numpy.random.SeedSequence``.
    n_jobs : int, default -1
        Number of joblib workers. ``-1`` uses all CPUs. ``1`` runs
        sequentially (useful for debugging).
    progressbar : bool, default True
        Show per-chain progress bars.

    Returns
    -------
    list of dict
        One dict per chain, each containing parameter trace arrays.
    """
    if seeds is None:
        # Derive per-chain seeds from a parent SeedSequence
        parent_ss = np.random.SeedSequence()
        child_seeds = parent_ss.spawn(n_chains)
        seeds = [int(s.generate_state(1)[0]) for s in child_seeds]
    elif len(seeds) != n_chains:
        raise ValueError(
            f"len(seeds) must equal n_chains, got {len(seeds)} != {n_chains}."
        )

    if n_jobs == 1:
        # Sequential execution for debugging
        results = []
        for chain_id, seed in enumerate(seeds):
            if progressbar:
                print(f"Chain {chain_id + 1}/{n_chains} (seed={seed})")
            result = chain_fn(chain_id, seed)
            results.append(result)
        return results
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(chain_fn)(chain_id, seed)
            for chain_id, seed in enumerate(seeds)
        )
        return list(results)