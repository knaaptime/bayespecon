"""Shared pytest fixtures for parameter recovery tests.

Helper functions and constants live in tests/helpers.py so they can be
imported from test files directly without conftest injection.

All tests are marked slow because they run MCMC. Run them with::

    pytest -m slow
    pytest -m recovery   # parameter recovery tests only
    pytest               # skips slow by default (see pyproject.toml)
"""

from __future__ import annotations

import numpy as np
import pytest
from libpysal.graph import Graph

from .helpers import PANEL_N, W_to_graph, make_line_W, make_rook_W

SIDE = 6  # 36 cross-sectional units


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def W_dense() -> np.ndarray:
    return make_rook_W(SIDE)


@pytest.fixture(scope="session")
def W_graph(W_dense: np.ndarray) -> Graph:
    return W_to_graph(W_dense)


@pytest.fixture(scope="session")
def W_panel_dense() -> np.ndarray:
    return make_line_W(PANEL_N)


@pytest.fixture(scope="session")
def W_panel_graph(W_panel_dense: np.ndarray) -> Graph:
    return W_to_graph(W_panel_dense)
