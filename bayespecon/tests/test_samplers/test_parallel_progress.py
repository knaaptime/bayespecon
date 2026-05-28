"""Tests for parallel progress bar support.

Tests for ``_SharedCounterReporter`` and ``_ParallelProgressRenderer``
— the shared-memory-based progress bar mechanism used when running
Gibbs chains in parallel via ``joblib.Parallel``.
"""

from __future__ import annotations

import pickle
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from bayespecon.samplers._utils._progress import (
    GibbsProgressBarManager,
    _ParallelProgressRenderer,
    _SharedCounterReporter,
)

# ---------------------------------------------------------------------------
# Module-level helpers (need to be picklable for loky worker tests).
# ---------------------------------------------------------------------------


def _silent_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
    return {"chain": chain_id}


def _record_reporter_type(chain_id, seed, progress_manager=None, chain_id_kw=None):
    return {"chain": chain_id, "type": type(progress_manager).__name__}


def _reporting_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
    # Simulate a short chain that writes a few iteration updates
    # to SHM.  Tuning first, then sampling.
    if progress_manager is not None:
        progress_manager.start_chain(chain_id)
        for i in range(3):
            progress_manager.update(chain_id, i, tuning=True)
        for i in range(3, 8):
            progress_manager.update(chain_id, i, tuning=False)
    return {"chain": chain_id}


# ---------------------------------------------------------------------------
# _SharedCounterReporter
# ---------------------------------------------------------------------------


@pytest.fixture
def shm_block():
    """Allocate a (2, 2) int64 SHM block for two chains and clean up."""
    n_chains = 2
    nbytes = n_chains * 2 * np.dtype(np.int64).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    # Zero-initialize.
    np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)[:] = 0
    try:
        yield shm, n_chains
    finally:
        shm.close()
        shm.unlink()


class TestSharedCounterReporter:
    def test_update_writes_iteration_and_tuning_flag(self, shm_block):
        shm, n_chains = shm_block
        reporter = _SharedCounterReporter(shm.name, chain_id=0, n_chains=n_chains)

        reporter.update(0, 5, tuning=False)

        buf = np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)
        # iteration is stored 1-based, tuning flag is 0 for sampling
        assert buf[0, 0] == 6
        assert buf[0, 1] == 0

    def test_update_tuning_flag(self, shm_block):
        shm, n_chains = shm_block
        reporter = _SharedCounterReporter(shm.name, chain_id=1, n_chains=n_chains)

        reporter.update(1, 2, tuning=True)

        buf = np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)
        assert buf[1, 0] == 3
        assert buf[1, 1] == 1

    def test_multiple_reporters_share_block(self, shm_block):
        shm, n_chains = shm_block
        r0 = _SharedCounterReporter(shm.name, chain_id=0, n_chains=n_chains)
        r1 = _SharedCounterReporter(shm.name, chain_id=1, n_chains=n_chains)

        r0.update(0, 7, tuning=False)
        r1.update(1, 9, tuning=True)

        buf = np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)
        assert buf[0, 0] == 8 and buf[0, 1] == 0
        assert buf[1, 0] == 10 and buf[1, 1] == 1

    def test_set_accept_rate_is_noop(self, shm_block):
        shm, n_chains = shm_block
        reporter = _SharedCounterReporter(shm.name, chain_id=0, n_chains=n_chains)
        # Must not raise; accept rate is intentionally not surfaced.
        reporter.set_accept_rate(0, 0.5)

    def test_pickle_roundtrip(self, shm_block):
        shm, n_chains = shm_block
        reporter = _SharedCounterReporter(shm.name, chain_id=0, n_chains=n_chains)

        revived = pickle.loads(pickle.dumps(reporter))
        revived.update(0, 4, tuning=False)

        buf = np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)
        assert buf[0, 0] == 5
        assert buf[0, 1] == 0


# ---------------------------------------------------------------------------
# _ParallelProgressRenderer
# ---------------------------------------------------------------------------


class TestParallelProgressRenderer:
    def test_context_manager_creates_one_task_per_chain(self):
        renderer = _ParallelProgressRenderer(
            n_chains=3, draws=10, tune=5, model_type="sar"
        )
        with renderer:
            assert len(renderer._tasks) == 3

    def test_poll_thread_reads_shm_until_stop(self, shm_block):
        shm, n_chains = shm_block
        renderer = _ParallelProgressRenderer(
            n_chains=n_chains, draws=10, tune=5, model_type="sar"
        )
        stop_event = threading.Event()

        buf = np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)

        with renderer:
            t = threading.Thread(
                target=renderer.poll,
                args=(shm.name, stop_event),
                kwargs={"interval": 0.02},
                daemon=True,
            )
            t.start()

            # Simulate worker writes: tune then draw phase.
            buf[0, :] = [3, 1]  # iter=3, tuning
            buf[1, :] = [2, 1]
            time.sleep(0.1)
            buf[0, :] = [8, 0]  # iter=8 (3 draw iters past tune=5)
            buf[1, :] = [12, 0]
            time.sleep(0.1)

            stop_event.set()
            t.join(timeout=2.0)
            assert not t.is_alive()

            # Renderer should have recorded the final draw iteration.
            task0 = renderer._progress.tasks[renderer._tasks[0]]
            task1 = renderer._progress.tasks[renderer._tasks[1]]
            assert task0.completed == 8
            assert task1.completed == 12

    def test_exit_forces_final_refresh(self):
        """__exit__ forces one final refresh for notebook rendering."""
        renderer = _ParallelProgressRenderer(
            n_chains=1, draws=10, tune=5, model_type="sar"
        )
        renderer._progress.live.auto_refresh = False
        refresh_calls = {"count": 0}

        with renderer:
            renderer._progress.refresh = lambda: refresh_calls.__setitem__(
                "count", refresh_calls["count"] + 1
            )

        assert refresh_calls["count"] == 1


# ---------------------------------------------------------------------------
# Integration: run_chains with the parallel + progressbar path
# ---------------------------------------------------------------------------


class TestRunChainsParallelProgress:
    def test_parallel_progressbar_false_is_silent(self):
        from bayespecon.samplers.gaussian._chain_runner import run_chains

        results = run_chains(
            chain_fn=_silent_chain,
            n_chains=2,
            seeds=[42, 43],
            n_jobs=1,
            progressbar=False,
            parallel=True,
            draws=10,
            tune=5,
        )
        assert len(results) == 2

    def test_parallel_progressbar_true_uses_shm_reporter(self):
        from bayespecon.samplers.gaussian._chain_runner import run_chains

        results = run_chains(
            chain_fn=_record_reporter_type,
            n_chains=2,
            seeds=[42, 43],
            n_jobs=2,
            progressbar=True,
            parallel=True,
            draws=10,
            tune=5,
        )
        assert len(results) == 2
        assert all(r["type"] == "_SharedCounterReporter" for r in results)

    def test_parallel_progressbar_true_end_to_end_with_writes(self):
        from bayespecon.samplers.gaussian._chain_runner import run_chains

        # Just verify the path completes without error when workers
        # actually call progress_manager.update().
        results = run_chains(
            chain_fn=_reporting_chain,
            n_chains=2,
            seeds=[1, 2],
            n_jobs=2,
            progressbar=True,
            parallel=True,
            draws=10,
            tune=5,
        )
        assert len(results) == 2

    def test_sequential_path_uses_gibbs_manager(self):
        from bayespecon.samplers.gaussian._chain_runner import run_chains

        results = run_chains(
            chain_fn=_record_reporter_type,
            n_chains=2,
            seeds=[42, 43],
            n_jobs=1,
            progressbar=False,
            parallel=False,
            draws=10,
            tune=5,
        )
        assert len(results) == 2
        assert all(r["type"] == GibbsProgressBarManager.__name__ for r in results)
