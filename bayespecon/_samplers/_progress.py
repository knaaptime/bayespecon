"""Progress bar manager for Gibbs samplers.

Uses rich.progress for terminal rendering with per-chain bars
showing iteration count, MALA accept rate, speed, and timing.

Like PyMC's progress bar, only the draw phase is tracked in the
progress bar.  The tune phase is indicated by a ``tune`` label
but does not advance the bar.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table
from rich.theme import Theme

default_gibbs_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "#1764f4",
        "progress.remaining": "none",
        "progress.elapsed": "none",
    }
)


class _GibbsProgress(Progress):
    """Custom Progress subclass that renders column headers.

    Overrides ``make_tasks_table`` to return a proper ``Table`` with
    ``show_header=True`` instead of the default ``Table.grid()`` which
    hides headers.  Mimics PyMC's ``CustomProgress`` pattern.
    """

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        table_columns = (
            Column(no_wrap=True)
            if isinstance(_column, str)
            else _column.get_table_column().copy()
            for _column in self.columns
        )
        table = Table(
            *table_columns,
            padding=(0, 1),
            expand=self.expand,
            show_header=True,
            show_edge=True,
            box=SIMPLE_HEAD,
        )
        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        column.format(task=task)
                        if isinstance(column, str)
                        else (
                            column(task)
                            if hasattr(column, "callbacks")
                            else column(task)
                        )
                        for column in self.columns
                    )
                )
        return table


class GibbsProgressBarManager:
    """Progress bar manager for Gibbs sampling.

    Shows per-chain rich progress bars with iteration count,
    MALA accept rate, speed, and timing.

    Like PyMC's progress bar, the bar only advances during the
    draw phase.  The tune phase is indicated by a ``tune`` label
    but does not advance the progress bar.

    Parameters
    ----------
    chains : int
        Number of chains.
    draws : int
        Post-warmup draws per chain.
    tune : int
        Warmup draws per chain.
    progressbar : bool, default True
        Show progress bars.
    model_type : str, default "sar"
        Model type (for display).
    """

    def __init__(
        self,
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool = True,
        model_type: str = "sar",
    ):
        self.chains = chains
        self.draws = draws
        self.tune = tune
        self.total = draws + tune
        self._show = progressbar
        self.model_type = model_type

        # Accept-rate tracking state (populated on first update)
        self._accept_counts: list[int] | None = None
        self._accept_totals: list[int] | None = None

        # Per-chain start times for accurate speed reporting.
        # Without this, sequential chains inherit elapsed time from
        # task creation (not chain start), making later chains appear
        # slower than they really are.
        self._chain_start_times: dict[int, float] = {}

        if self._show:
            self._progress = _GibbsProgress(
                BarColumn(bar_width=None, table_column=Column("Progress", ratio=2)),
                TextColumn(
                    "{task.fields[draw_iter]:>4d}/{task.fields[total_draws]}",
                    table_column=Column("Draws", ratio=2),
                ),
                TextColumn(
                    "{task.fields[accept_rate]}",
                    table_column=Column("Accept", ratio=1),
                ),
                TextColumn(
                    "{task.fields[speed]}",
                    table_column=Column("Speed", ratio=2),
                ),
                TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
                TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
                console=Console(theme=default_gibbs_theme),
                expand=True,
            )
            self._tasks: list[Any] = []
        else:
            self._progress = nullcontext()
            self._tasks = []

    def __enter__(self):
        if self._show:
            self._progress.__enter__()
            # Add one task per chain
            for c in range(self.chains):
                task_id = self._progress.add_task(
                    f"Chain {c + 1}",
                    total=self.draws,
                    phase="tune",
                    draw_iter=0,
                    total_draws=self.draws,
                    accept_rate="--",
                    speed="--",
                )
                self._tasks.append(task_id)
        return self

    def __exit__(self, *args):
        if self._show:
            # Compute final summary line
            total_draws = self.chains * (self.draws + self.tune)
            if self._tasks:
                elapsed = self._progress.tasks[self._tasks[0]].elapsed or 0.0
            else:
                elapsed = 0.0
            speed = total_draws / elapsed if elapsed > 0 else 0.0
            summary = (
                f"Sampling {self.chains} chain{'s' if self.chains > 1 else ''} "
                f"for {self.tune} tune and {self.draws} draw iterations, "
                f"{self.chains} x {self.tune + self.draws:,} draws total "
                f"took {elapsed:.0f}s ({speed:.0f} draws/s)"
            )
            # In Jupyter, print inside the ipy_widget before exiting
            # so the summary appears in the same output block as the
            # progress bar (avoids creating a separate rich block).
            ipy_widget = getattr(self._progress.live, "ipy_widget", None)
            if ipy_widget is not None:
                with ipy_widget:
                    self._progress.console.print(summary)
                self._progress.__exit__(*args)
            else:
                self._progress.__exit__(*args)
                self._progress.console.print(summary)

    def start_chain(self, chain_idx: int) -> None:
        """Record the start time for a chain.

        Must be called just before the chain begins sampling so that
        the speed column reflects actual per-chain sampling time rather
        than wall-clock time since task creation.

        Parameters
        ----------
        chain_idx : int
            Chain index (0-based).
        """
        self._chain_start_times[chain_idx] = time.monotonic()

    def update(
        self,
        chain_idx: int,
        iteration: int,
        tuning: bool,
        accept: bool | None = None,
    ):
        """Update progress for a chain.

        Parameters
        ----------
        chain_idx : int
            Chain index (0-based).
        iteration : int
            Current iteration (0-based, counting both tune and draw).
        tuning : bool
            Whether in warmup phase.
        accept : bool or None
            Whether MALA/MH step was accepted (None for NumPy slice path).
        """
        if not self._show:
            return

        task_id = self._tasks[chain_idx]
        phase = "tune" if tuning else "draw"

        # Compute running accept rate
        if accept is not None:
            # Lazy-init accept tracking
            if self._accept_counts is None:
                self._accept_counts = [0] * self.chains
                self._accept_totals = [0] * self.chains
            self._accept_totals[chain_idx] += 1
            if accept:
                self._accept_counts[chain_idx] += 1
            rate = self._accept_counts[chain_idx] / self._accept_totals[chain_idx]
            accept_str = f"{rate:.0%}"
        else:
            accept_str = "--"

        # Compute speed using per-chain start time when available.
        # Falls back to task.elapsed for backward compatibility.
        # Using task.elapsed directly is inaccurate for sequential chains
        # because all tasks are created at once — later chains inherit
        # elapsed time from before they started.
        chain_start = self._chain_start_times.get(chain_idx)
        if chain_start is not None:
            elapsed = time.monotonic() - chain_start
        else:
            elapsed = self._progress.tasks[task_id].elapsed or 0.0
        if elapsed > 0:
            speed_str = f"{(iteration + 1) / elapsed:.1f} draws/s"
        else:
            speed_str = "--"

        # Only advance the bar during the draw phase (like PyMC)
        if tuning:
            self._progress.update(
                task_id,
                phase=phase,
                draw_iter=0,
                accept_rate=accept_str,
                speed=speed_str,
            )
        else:
            draw_iter = iteration - self.tune + 1
            self._progress.update(
                task_id,
                advance=1,
                phase=phase,
                draw_iter=draw_iter,
                accept_rate=accept_str,
                speed=speed_str,
            )

    def set_accept_rate(self, chain_idx: int, rate: float) -> None:
        """Set the aggregate accept rate for a chain.

        Used by JAX-based Gibbs samplers (MALA/MH) to report the
        overall accept rate after the draw phase completes, since
        per-iteration accept booleans are not available from JIT-compiled
        scans.  For NumPy slice sampling, this is not called (the
        Accept column shows ``"--"``).

        Parameters
        ----------
        chain_idx : int
            Chain index (0-based).
        rate : float
            Accept rate between 0 and 1 (e.g., 0.574 for 57.4%).
        """
        if not self._show:
            return
        # Lazy-init accept tracking
        if self._accept_counts is None:
            self._accept_counts = [0] * self.chains
            self._accept_totals = [0] * self.chains
        # Use a large denominator so the percentage is precise
        self._accept_totals[chain_idx] = 1000
        self._accept_counts[chain_idx] = round(rate * 1000)
        # Update the progress bar display
        task_id = self._tasks[chain_idx]
        accept_str = f"{rate:.0%}"
        self._progress.update(task_id, accept_rate=accept_str)

    def refresh(self):
        """Force a refresh of the progress bar display."""
        if self._show:
            self._progress.refresh()


# ---------------------------------------------------------------------------
# Parallel progress bar support
# ---------------------------------------------------------------------------


class _ParallelProgressReporter:
    """Picklable progress reporter for worker processes.

    Implements the same interface as :class:`GibbsProgressBarManager`
    (``update``, ``start_chain``, ``set_accept_rate``) but serialises
    updates as dicts on a :class:`multiprocessing.Queue` so that a
    main-process renderer can display them.

    This allows ``joblib.Parallel`` worker processes to report progress
    back to the main process without holding any unpicklable ``rich``
    objects.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue to post update messages on.
    chain_id : int
        0-based chain index that this reporter represents.
    """

    def __init__(self, queue: multiprocessing.Queue, chain_id: int):
        # Store the *context* queue so it survives pickling across
        # process boundaries (Manager queues are already proxy objects).
        self._queue = queue
        self._chain_id = chain_id

    # -- same interface as GibbsProgressBarManager -----------------------

    def update(
        self,
        chain_idx: int,
        iteration: int,
        tuning: bool,
        accept: bool | None = None,
    ):
        self._queue.put(
            {
                "type": "update",
                "chain": self._chain_id,
                "iteration": iteration,
                "tuning": tuning,
                "accept": accept,
            }
        )

    def start_chain(self, chain_idx: int):
        self._queue.put({"type": "start", "chain": self._chain_id})

    def set_accept_rate(self, chain_idx: int, rate: float):
        # Aggregate accept rate is not useful per-iteration in
        # parallel mode; the main-process renderer tracks running
        # rates from individual accept booleans.
        pass

    def refresh(self):
        # No-op: the daemon drain thread continuously refreshes
        # the rich progress display.
        pass


class _ParallelProgressRenderer:
    """Main-process renderer that reads queue messages and updates rich bars.

    Owns a :class:`_GibbsProgress` instance with one task per chain.
    A daemon thread calls :meth:`drain` in a loop to consume messages
    from the worker queue and update the corresponding tasks.

    Parameters
    ----------
    n_chains : int
        Number of chains.
    draws : int
        Post-warmup draws per chain.
    tune : int
        Warmup draws per chain.
    model_type : str
        Model type (for display).
    """

    def __init__(
        self,
        n_chains: int,
        draws: int,
        tune: int,
        model_type: str = "sar",
    ):
        self.n_chains = n_chains
        self.draws = draws
        self.tune = tune
        self.model_type = model_type

        # Accept-rate tracking (mirrors GibbsProgressBarManager logic)
        self._accept_counts: list[int] = [0] * n_chains
        self._accept_totals: list[int] = [0] * n_chains

        # Per-chain start times for speed reporting
        self._chain_start_times: dict[int, float] = {}

        # Create the rich Progress instance (same layout as sequential)
        self._progress = _GibbsProgress(
            BarColumn(bar_width=None, table_column=Column("Progress", ratio=2)),
            TextColumn(
                "{task.fields[draw_iter]:>4d}/{task.fields[total_draws]}",
                table_column=Column("Draws", ratio=2),
            ),
            TextColumn(
                "{task.fields[accept_rate]}",
                table_column=Column("Accept", ratio=1),
            ),
            TextColumn(
                "{task.fields[speed]}",
                table_column=Column("Speed", ratio=2),
            ),
            TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
            TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
            console=Console(theme=default_gibbs_theme),
            expand=True,
        )
        self._tasks: list[Any] = []

    def __enter__(self):
        self._progress.__enter__()
        for c in range(self.n_chains):
            task_id = self._progress.add_task(
                f"Chain {c + 1}",
                total=self.draws,
                phase="tune",
                draw_iter=0,
                total_draws=self.draws,
                accept_rate="--",
                speed="--",
            )
            self._tasks.append(task_id)
        return self

    def __exit__(self, *args):
        # Compute final summary line
        total_draws = self.n_chains * (self.draws + self.tune)
        if self._tasks:
            elapsed = self._progress.tasks[self._tasks[0]].elapsed or 0.0
        else:
            elapsed = 0.0
        speed = total_draws / elapsed if elapsed > 0 else 0.0
        summary = (
            f"Sampling {self.n_chains} chain{'s' if self.n_chains > 1 else ''} "
            f"for {self.tune} tune and {self.draws} draw iterations, "
            f"{self.n_chains} x {self.tune + self.draws:,} draws total "
            f"took {elapsed:.0f}s ({speed:.0f} draws/s)"
        )
        # In Jupyter, print inside the ipy_widget before exiting
        # so the summary appears in the same output block as the
        # progress bar (avoids creating a separate rich block).
        ipy_widget = getattr(self._progress.live, "ipy_widget", None)
        if ipy_widget is not None:
            with ipy_widget:
                self._progress.console.print(summary)
            self._progress.__exit__(*args)
        else:
            self._progress.__exit__(*args)
            self._progress.console.print(summary)

    def process_message(self, msg: dict) -> None:
        """Process a single message from the worker queue.

        Parameters
        ----------
        msg : dict
            Message dict with ``"type"`` key (``"start"`` or ``"update"``).
        """
        msg_type = msg["type"]
        chain = msg["chain"]

        if msg_type == "start":
            self._chain_start_times[chain] = time.monotonic()

        elif msg_type == "update":
            iteration = msg["iteration"]
            tuning = msg["tuning"]
            accept = msg.get("accept")

            task_id = self._tasks[chain]
            phase = "tune" if tuning else "draw"

            # Accept rate
            if accept is not None:
                self._accept_totals[chain] += 1
                if accept:
                    self._accept_counts[chain] += 1
                rate = self._accept_counts[chain] / self._accept_totals[chain]
                accept_str = f"{rate:.0%}"
            else:
                accept_str = "--"

            # Speed
            chain_start = self._chain_start_times.get(chain)
            if chain_start is not None:
                elapsed = time.monotonic() - chain_start
            else:
                elapsed = self._progress.tasks[task_id].elapsed or 0.0
            if elapsed > 0:
                speed_str = f"{(iteration + 1) / elapsed:.1f} draws/s"
            else:
                speed_str = "--"

            # Advance bar only during draw phase (like GibbsProgressBarManager)
            if tuning:
                self._progress.update(
                    task_id,
                    phase=phase,
                    draw_iter=0,
                    accept_rate=accept_str,
                    speed=speed_str,
                )
            else:
                draw_iter = iteration - self.tune + 1
                self._progress.update(
                    task_id,
                    advance=1,
                    phase=phase,
                    draw_iter=draw_iter,
                    accept_rate=accept_str,
                    speed=speed_str,
                )

    def drain(self, queue: multiprocessing.Queue, stop_event: threading.Event) -> None:
        """Drain messages from *queue* until *stop_event* is set.

        Intended to be run in a daemon thread.

        Parameters
        ----------
        queue : multiprocessing.Queue
            The shared queue that workers post messages on.
        stop_event : threading.Event
            When set, the loop exits after draining remaining messages.
        """
        while not stop_event.is_set():
            try:
                msg = queue.get(timeout=0.1)
            except Exception:
                # queue.get raises Empty on timeout — just retry
                continue
            self.process_message(msg)
        # Drain any remaining messages after stop signal
        while not queue.empty():
            try:
                msg = queue.get_nowait()
            except Exception:
                break
            self.process_message(msg)
