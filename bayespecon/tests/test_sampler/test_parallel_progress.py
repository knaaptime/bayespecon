"""Tests for parallel progress bar support.

Tests for ``_ParallelProgressReporter`` and
``_ParallelProgressRenderer`` — the IPC-based progress bar
mechanism used when running Gibbs chains in parallel via
``joblib.Parallel``.
"""

from __future__ import annotations

import multiprocessing
import threading
import time

import pytest


class TestParallelProgressReporter:
    """Tests for _ParallelProgressReporter."""

    def test_update_puts_message_on_queue(self):
        """update() puts a correctly-structured dict on the queue."""
        from bayespecon._samplers._progress import _ParallelProgressReporter

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        reporter = _ParallelProgressReporter(queue, chain_id=0)

        reporter.update(0, 5, tuning=False, accept=True)

        msg = queue.get(timeout=1.0)
        assert msg["type"] == "update"
        assert msg["chain"] == 0
        assert msg["iteration"] == 5
        assert msg["tuning"] is False
        assert msg["accept"] is True

    def test_start_chain_puts_message_on_queue(self):
        """start_chain() puts a start message on the queue."""
        from bayespecon._samplers._progress import _ParallelProgressReporter

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        reporter = _ParallelProgressReporter(queue, chain_id=2)

        reporter.start_chain(2)

        msg = queue.get(timeout=1.0)
        assert msg["type"] == "start"
        assert msg["chain"] == 2

    def test_set_accept_rate_is_noop(self):
        """set_accept_rate() does not put messages on the queue."""
        from bayespecon._samplers._progress import _ParallelProgressReporter

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        reporter = _ParallelProgressReporter(queue, chain_id=0)

        reporter.set_accept_rate(0, 0.75)

        assert queue.empty()

    def test_multiple_chains_independent_queues(self):
        """Different reporters on the same queue send distinct chain IDs."""
        from bayespecon._samplers._progress import _ParallelProgressReporter

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        r0 = _ParallelProgressReporter(queue, chain_id=0)
        r1 = _ParallelProgressReporter(queue, chain_id=1)

        r0.update(0, 0, tuning=True, accept=None)
        r1.update(1, 0, tuning=True, accept=None)

        msgs = [queue.get(timeout=1.0), queue.get(timeout=1.0)]
        chains = {m["chain"] for m in msgs}
        assert chains == {0, 1}


class TestParallelProgressRenderer:
    """Tests for _ParallelProgressRenderer."""

    def test_context_manager_creates_tasks(self):
        """Renderer creates one rich task per chain on enter."""
        from bayespecon._samplers._progress import _ParallelProgressRenderer

        renderer = _ParallelProgressRenderer(
            n_chains=3, draws=10, tune=5, model_type="sar"
        )
        with renderer:
            assert len(renderer._tasks) == 3

    def test_process_update_message_tuning(self):
        """process_message() handles a tuning-phase update."""
        from bayespecon._samplers._progress import _ParallelProgressRenderer

        renderer = _ParallelProgressRenderer(
            n_chains=1, draws=10, tune=5, model_type="sar"
        )
        with renderer:
            renderer.process_message(
                {"type": "update", "chain": 0, "iteration": 2, "tuning": True, "accept": None}
            )
            # No assertion on rich task state — just verify no errors

    def test_process_update_message_draw(self):
        """process_message() handles a draw-phase update."""
        from bayespecon._samplers._progress import _ParallelProgressRenderer

        renderer = _ParallelProgressRenderer(
            n_chains=1, draws=10, tune=5, model_type="sar"
        )
        with renderer:
            renderer.process_message(
                {"type": "start", "chain": 0}
            )
            renderer.process_message(
                {"type": "update", "chain": 0, "iteration": 5, "tuning": False, "accept": True}
            )
            # Verify accept tracking
            assert renderer._accept_counts[0] == 1
            assert renderer._accept_totals[0] == 1

    def test_process_start_message_records_time(self):
        """process_message() records chain start time on 'start'."""
        from bayespecon._samplers._progress import _ParallelProgressRenderer

        renderer = _ParallelProgressRenderer(
            n_chains=2, draws=10, tune=5, model_type="sar"
        )
        with renderer:
            renderer.process_message({"type": "start", "chain": 0})
            assert 0 in renderer._chain_start_times
            assert renderer._chain_start_times[0] > 0

    def test_drain_thread_processes_messages(self):
        """drain() processes messages from the queue in a thread."""
        from bayespecon._samplers._progress import (
            _ParallelProgressRenderer,
            _ParallelProgressReporter,
        )

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        stop_event = threading.Event()

        renderer = _ParallelProgressRenderer(
            n_chains=1, draws=10, tune=5, model_type="sar"
        )
        reporter = _ParallelProgressReporter(queue, chain_id=0)

        with renderer:
            drain_thread = threading.Thread(
                target=renderer.drain, args=(queue, stop_event), daemon=True
            )
            drain_thread.start()

            # Send a start and an update
            reporter.start_chain(0)
            reporter.update(0, 0, tuning=True, accept=None)

            # Give the drain thread time to process
            time.sleep(0.3)
            stop_event.set()
            drain_thread.join(timeout=2.0)

        # If we get here without hanging, the drain thread worked


class TestRunChainsParallelProgress:
    """Integration tests for run_chains with parallel progress bars."""

    def test_parallel_progressbar_false_is_silent(self):
        """parallel=True with progressbar=False runs silently."""
        from bayespecon._samplers._chain_runner import run_chains

        def mock_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            return {"chain": chain_id}

        results = run_chains(
            chain_fn=mock_chain,
            n_chains=2,
            seeds=[42, 43],
            n_jobs=1,
            progressbar=False,
            parallel=True,
            draws=10,
            tune=5,
        )
        assert len(results) == 2

    def test_parallel_progressbar_true_uses_reporter(self):
        """parallel=True with progressbar=True passes reporters to chain_fn."""
        from bayespecon._samplers._chain_runner import run_chains

        received_managers = []

        def mock_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            received_managers.append(progress_manager)
            # Send a start and update message
            if progress_manager is not None:
                progress_manager.start_chain(chain_id)
                progress_manager.update(chain_id, 0, tuning=True, accept=None)
            return {"chain": chain_id}

        results = run_chains(
            chain_fn=mock_chain,
            n_chains=2,
            seeds=[42, 43],
            n_jobs=1,
            progressbar=True,
            parallel=True,
            draws=10,
            tune=5,
        )
        assert len(results) == 2
        # Each chain should have received a _ParallelProgressReporter
        from bayespecon._samplers._progress import _ParallelProgressReporter

        for pm in received_managers:
            assert isinstance(pm, _ParallelProgressReporter)

    def test_sequential_path_unchanged(self):
        """parallel=False still uses GibbsProgressBarManager."""
        from bayespecon._samplers._chain_runner import run_chains
        from bayespecon._samplers._progress import GibbsProgressBarManager

        received_managers = []

        def mock_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            received_managers.append(type(progress_manager).__name__)
            return {"chain": chain_id}

        results = run_chains(
            chain_fn=mock_chain,
            n_chains=2,
            seeds=[42, 43],
            n_jobs=1,
            progressbar=False,
            parallel=False,
            draws=10,
            tune=5,
        )
        assert len(results) == 2
        # Sequential path uses GibbsProgressBarManager (even if disabled)
        assert all(name == "GibbsProgressBarManager" for name in received_managers)