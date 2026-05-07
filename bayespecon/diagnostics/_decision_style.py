"""Style system for decision-tree visualisations.

Provides dataclasses that describe Graphviz node / edge / graph attributes,
plus a small registry of built-in themes.  Themes are intentionally plain
Python objects so users can construct their own without learning a new DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "NodeStyle",
    "EdgeStyle",
    "GraphTheme",
    "DEFAULT_THEME",
    "MINIMAL_THEME",
    "DARK_THEME",
    "get_theme",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeStyle:
    """Graphviz node attributes."""

    shape: str = "ellipse"
    fontname: str = "Helvetica"
    fontsize: str = "11"
    fontcolor: str = "#1a1a1a"
    fillcolor: str = "white"
    color: str = "#495057"
    penwidth: str = "1.0"
    style: str = "filled,rounded"
    margin: str = "0.15,0.08"

    def as_dict(self) -> dict[str, str]:
        return {
            "shape": self.shape,
            "fontname": self.fontname,
            "fontsize": self.fontsize,
            "fontcolor": self.fontcolor,
            "fillcolor": self.fillcolor,
            "color": self.color,
            "penwidth": self.penwidth,
            "style": self.style,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class EdgeStyle:
    """Graphviz edge attributes."""

    color: str = "#888888"
    penwidth: str = "1.0"
    fontcolor: str = "#495057"
    fontsize: str = "10"
    arrowhead: str = "normal"
    style: str = "solid"

    def as_dict(self) -> dict[str, str]:
        return {
            "color": self.color,
            "penwidth": self.penwidth,
            "fontcolor": self.fontcolor,
            "fontsize": self.fontsize,
            "arrowhead": self.arrowhead,
            "style": self.style,
        }


@dataclass(frozen=True)
class GraphTheme:
    """Complete theme for a decision-tree graph.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``"default"``).
    rankdir : str
        Graphviz layout direction (``"TB"``, ``"LR"``, …).
    bgcolor : str
        Background colour of the canvas.
    margin : str
        Graph margin in inches.
    nodesep : str
        Horizontal separation between nodes.
    ranksep : str
        Vertical separation between ranks.
    fontname : str
        Default typeface for graph-level text.
    title_fontsize : str
        Font size for the graph title (if any).
    title_fontcolor : str
        Font colour for the graph title.
    test_node, predicate_node, leaf_chosen, leaf_other : NodeStyle
        Styles for the four node categories.
    node_on_path : NodeStyle | None
        Overrides applied when a node lies on the traversed decision path.
        ``None`` means no override.
    edge_on_path, edge_off_path : EdgeStyle
        Styles for edges that are / are not on the traversed path.
    """

    name: str
    rankdir: str = "TB"
    bgcolor: str = "white"
    margin: str = "0"
    nodesep: str = "0.45"
    ranksep: str = "0.65"
    fontname: str = "Helvetica"
    title_fontsize: str = "14"
    title_fontcolor: str = "#1a1a1a"

    test_node: NodeStyle = field(default_factory=lambda: NodeStyle())
    predicate_node: NodeStyle = field(default_factory=lambda: NodeStyle())
    leaf_chosen: NodeStyle = field(default_factory=lambda: NodeStyle())
    leaf_other: NodeStyle = field(default_factory=lambda: NodeStyle())

    node_on_path: dict[str, str] | None = None
    edge_on_path: EdgeStyle = field(default_factory=lambda: EdgeStyle())
    edge_off_path: EdgeStyle = field(default_factory=lambda: EdgeStyle())

    def graph_attrs(self) -> dict[str, str]:
        return {
            "rankdir": self.rankdir,
            "bgcolor": self.bgcolor,
            "margin": self.margin,
            "nodesep": self.nodesep,
            "ranksep": self.ranksep,
            "fontname": self.fontname,
        }


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------

# --- Default (enhanced, colour-blind-friendly) --------------------------------

_DEFAULT_TEST = NodeStyle(
    shape="ellipse",
    fillcolor="#e8f4fd",
    color="#4c72b0",
    fontcolor="#1a1a1a",
    penwidth="1.2",
)

_DEFAULT_PREDICATE = NodeStyle(
    shape="diamond",
    fillcolor="#fff3cd",
    color="#dd8452",
    fontcolor="#1a1a1a",
    penwidth="1.2",
    margin="0.18,0.10",
)

_DEFAULT_LEAF_CHOSEN = NodeStyle(
    shape="box",
    fillcolor="#d4edda",
    color="#55a868",
    fontcolor="#1a1a1a",
    penwidth="2.0",
    style="filled,rounded,bold",
)

_DEFAULT_LEAF_OTHER = NodeStyle(
    shape="box",
    fillcolor="#f8f9fa",
    color="#adb5bd",
    fontcolor="#6c757d",
    penwidth="1.0",
)

_DEFAULT_NODE_ON_PATH = NodeStyle(
    fillcolor="#ffd966",
    color="#b58900",
    penwidth="2.0",
    style="filled,rounded,bold",
)

_DEFAULT_EDGE_ON_PATH = EdgeStyle(
    color="#4c72b0",
    penwidth="2.5",
    fontcolor="#1a1a1a",
    arrowhead="vee",
)

_DEFAULT_EDGE_OFF_PATH = EdgeStyle(
    color="#ced4da",
    penwidth="1.0",
    fontcolor="#adb5bd",
    style="dashed",
    arrowhead="normal",
)

DEFAULT_THEME = GraphTheme(
    name="default",
    rankdir="TB",
    bgcolor="white",
    nodesep="0.45",
    ranksep="0.65",
    test_node=_DEFAULT_TEST,
    predicate_node=_DEFAULT_PREDICATE,
    leaf_chosen=_DEFAULT_LEAF_CHOSEN,
    leaf_other=_DEFAULT_LEAF_OTHER,
    node_on_path=_DEFAULT_NODE_ON_PATH,
    edge_on_path=_DEFAULT_EDGE_ON_PATH,
    edge_off_path=_DEFAULT_EDGE_OFF_PATH,
)

# --- Minimal (grayscale, print-friendly) --------------------------------------

_MINIMAL_TEST = NodeStyle(
    shape="ellipse",
    fillcolor="white",
    color="#333333",
    fontcolor="#1a1a1a",
    penwidth="1.0",
    style="rounded",
)

_MINIMAL_PREDICATE = NodeStyle(
    shape="diamond",
    fillcolor="white",
    color="#333333",
    fontcolor="#1a1a1a",
    penwidth="1.0",
    style="rounded",
    margin="0.18,0.10",
)

_MINIMAL_LEAF_CHOSEN = NodeStyle(
    shape="box",
    fillcolor="#e9ecef",
    color="#000000",
    fontcolor="#000000",
    penwidth="2.0",
    style="filled,rounded,bold",
)

_MINIMAL_LEAF_OTHER = NodeStyle(
    shape="box",
    fillcolor="white",
    color="#adb5bd",
    fontcolor="#6c757d",
    penwidth="1.0",
    style="rounded",
)

_MINIMAL_NODE_ON_PATH = NodeStyle(
    fillcolor="#dee2e6",
    color="#000000",
    penwidth="2.0",
    style="filled,rounded,bold",
)

_MINIMAL_EDGE_ON_PATH = EdgeStyle(
    color="#000000",
    penwidth="2.5",
    fontcolor="#1a1a1a",
    arrowhead="vee",
)

_MINIMAL_EDGE_OFF_PATH = EdgeStyle(
    color="#ced4da",
    penwidth="1.0",
    fontcolor="#adb5bd",
    style="dashed",
    arrowhead="normal",
)

MINIMAL_THEME = GraphTheme(
    name="minimal",
    rankdir="TB",
    bgcolor="white",
    nodesep="0.45",
    ranksep="0.65",
    test_node=_MINIMAL_TEST,
    predicate_node=_MINIMAL_PREDICATE,
    leaf_chosen=_MINIMAL_LEAF_CHOSEN,
    leaf_other=_MINIMAL_LEAF_OTHER,
    node_on_path=_MINIMAL_NODE_ON_PATH,
    edge_on_path=_MINIMAL_EDGE_ON_PATH,
    edge_off_path=_MINIMAL_EDGE_OFF_PATH,
)

# --- Dark (for dark-mode notebooks) -------------------------------------------

_DARK_TEST = NodeStyle(
    shape="ellipse",
    fillcolor="#2a3f5f",
    color="#8ab4f8",
    fontcolor="#e8eaed",
    penwidth="1.2",
)

_DARK_PREDICATE = NodeStyle(
    shape="diamond",
    fillcolor="#5c4033",
    color="#f9ab00",
    fontcolor="#e8eaed",
    penwidth="1.2",
    margin="0.18,0.10",
)

_DARK_LEAF_CHOSEN = NodeStyle(
    shape="box",
    fillcolor="#1e4620",
    color="#81c995",
    fontcolor="#e8eaed",
    penwidth="2.0",
    style="filled,rounded,bold",
)

_DARK_LEAF_OTHER = NodeStyle(
    shape="box",
    fillcolor="#3c4043",
    color="#9aa0a6",
    fontcolor="#9aa0a6",
    penwidth="1.0",
)

_DARK_NODE_ON_PATH = NodeStyle(
    fillcolor="#b45f06",
    color="#f9ab00",
    penwidth="2.0",
    style="filled,rounded,bold",
)

_DARK_EDGE_ON_PATH = EdgeStyle(
    color="#8ab4f8",
    penwidth="2.5",
    fontcolor="#e8eaed",
    arrowhead="vee",
)

_DARK_EDGE_OFF_PATH = EdgeStyle(
    color="#5f6368",
    penwidth="1.0",
    fontcolor="#9aa0a6",
    style="dashed",
    arrowhead="normal",
)

DARK_THEME = GraphTheme(
    name="dark",
    rankdir="TB",
    bgcolor="#1e1e1e",
    nodesep="0.45",
    ranksep="0.65",
    title_fontcolor="#e8eaed",
    test_node=_DARK_TEST,
    predicate_node=_DARK_PREDICATE,
    leaf_chosen=_DARK_LEAF_CHOSEN,
    leaf_other=_DARK_LEAF_OTHER,
    node_on_path=_DARK_NODE_ON_PATH,
    edge_on_path=_DARK_EDGE_ON_PATH,
    edge_off_path=_DARK_EDGE_OFF_PATH,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_THEME_REGISTRY: dict[str, GraphTheme] = {
    "default": DEFAULT_THEME,
    "minimal": MINIMAL_THEME,
    "dark": DARK_THEME,
}


def get_theme(name: str | GraphTheme) -> GraphTheme:
    """Return a :class:`GraphTheme` by name or pass through a custom one.

    Parameters
    ----------
    name : str or GraphTheme
        Built-in theme name (``"default"``, ``"minimal"``, ``"dark"``) or an
        already-constructed :class:`GraphTheme` instance.

    Returns
    -------
    GraphTheme

    Raises
    ------
    ValueError
        If ``name`` is a string not present in the built-in registry.
    """
    if isinstance(name, GraphTheme):
        return name
    try:
        return _THEME_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(_THEME_REGISTRY)
        raise ValueError(f"unknown theme {name!r}; choose from: {available}") from exc
