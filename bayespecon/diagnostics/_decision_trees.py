"""Declarative decision-tree specs and renderers for spatial model selection.

This module provides:

- A small ``TreeNode`` dataclass that represents the Koley & Bera / stge_kb
  decision trees as data, parameterised per starting model type.
- ``evaluate(node, sig_lookup, predicate_lookup)`` which walks the tree and
  returns the recommended model name plus the traversed path.
- ``render_ascii`` and ``render_graphviz`` which render the tree (with the
  traversed path highlighted) as either an indented string or a
  ``graphviz.Digraph``.

The same specs back both the cross-sectional (``models/base.py``) and the
panel (``models/panel_base.py``) decision methods.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Union

__all__ = [
    "TreeNode",
    "evaluate",
    "render_ascii",
    "render_graphviz",
    "graphviz_available",
    "get_spec",
    "get_panel_spec",
]


# ---------------------------------------------------------------------------
# Tree primitives
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A single node in a decision tree.

    Parameters
    ----------
    kind : {"test", "predicate"}
        ``"test"`` means evaluate the named Bayesian LM test against alpha.
        ``"predicate"`` means evaluate a custom callable (used for the
        OLS robust-pair tie-break).
    name : str
        For ``kind="test"``, the test name as it appears in the diagnostics
        DataFrame (e.g. ``"LM-Lag"``). For ``kind="predicate"``, a human
        readable label (used as the node label and edge labels).
    if_true : TreeNode or str
        Subtree to follow when the test is significant (or predicate is
        true). A ``str`` is treated as a leaf with that model name.
    if_false : TreeNode or str
        Subtree to follow otherwise.
    predicate_id : str, optional
        For ``kind="predicate"``, an identifier looked up in the
        ``predicate_lookup`` dict passed to :func:`evaluate`.
    """

    kind: str  # "test" | "predicate"
    name: str
    if_true: Union["TreeNode", str]
    if_false: Union["TreeNode", str]
    predicate_id: str | None = None
    # Auto-assigned during ``evaluate`` for stable rendering identifiers.
    _node_id: str = field(default="", repr=False)


def _assign_ids(node: Union[TreeNode, str], counter: list[int]) -> None:
    if isinstance(node, str):
        return
    node._node_id = f"n{counter[0]}"
    counter[0] += 1
    _assign_ids(node.if_true, counter)
    _assign_ids(node.if_false, counter)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    root: Union[TreeNode, str],
    sig_lookup: Callable[[str], bool],
    predicate_lookup: dict[str, Callable[[], bool]] | None = None,
) -> tuple[str, list[tuple[str, bool]]]:
    """Walk the tree and return the recommended model and traversed path.

    Parameters
    ----------
    root : TreeNode or str
        Root of the decision tree (or a bare leaf string).
    sig_lookup : callable
        Maps a test name to a boolean indicating significance.
    predicate_lookup : dict, optional
        Maps a ``predicate_id`` to a zero-argument callable returning bool.

    Returns
    -------
    decision : str
        Recommended model name.
    path : list of (node_id, branch_taken)
        Sequence of decisions made. ``branch_taken`` is ``True`` if the
        ``if_true`` branch was followed, else ``False``.
    """
    _assign_ids(root, [0])
    predicate_lookup = predicate_lookup or {}
    path: list[tuple[str, bool]] = []
    node: Union[TreeNode, str] = root
    while not isinstance(node, str):
        if node.kind == "test":
            branch = bool(sig_lookup(node.name))
        elif node.kind == "predicate":
            if node.predicate_id is None:
                raise ValueError("predicate node missing predicate_id")
            branch = bool(predicate_lookup[node.predicate_id]())
        else:
            raise ValueError(f"unknown node kind: {node.kind}")
        path.append((node._node_id, branch))
        node = node.if_true if branch else node.if_false
    return node, path


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------


def _node_label(node: TreeNode) -> str:
    if node.kind == "predicate":
        return node.name
    return node.name


def render_ascii(
    root: Union[TreeNode, str],
    path: list[tuple[str, bool]],
    decision: str,
    p_values: dict[str, float] | None = None,
    alpha: float = 0.05,
) -> str:
    """Render the decision tree as an indented ASCII string.

    The traversed path is highlighted (``*`` marker on the chosen edge),
    and the chosen leaf is annotated with ``← SELECTED``.
    """
    p_values = p_values or {}
    path_dict = dict(path)

    lines: list[str] = []

    def render(
        node: Union[TreeNode, str],
        prefix: str,
        is_last: bool,
        edge_label: str | None,
        on_path: bool,
    ) -> None:
        connector = "" if edge_label is None else ("└── " if is_last else "├── ")
        marker = " *" if on_path else ""
        if isinstance(node, str):
            tag = " ← SELECTED" if on_path and node == decision else ""
            lines.append(f"{prefix}{connector}[{node}]{marker}{tag}")
            return

        # Internal node
        label = _node_label(node)
        annot = ""
        if on_path and node.kind == "test" and node.name in p_values:
            pv = p_values[node.name]
            annot = f"  (p={pv:.4f}, alpha={alpha})"
        edge_str = f"<{edge_label}> " if edge_label else ""
        lines.append(f"{prefix}{connector}{edge_str}{label}{marker}{annot}")

        # Determine child path membership
        new_prefix = prefix + ("    " if is_last else "│   ") if edge_label else ""
        # Children:
        children: list[tuple[str, Union[TreeNode, str]]] = [
            ("sig" if node.kind == "test" else "true", node.if_true),
            ("not sig" if node.kind == "test" else "false", node.if_false),
        ]
        # Determine which child is on path
        taken = path_dict.get(node._node_id)
        for i, (lbl, child) in enumerate(children):
            child_on_path = on_path and (
                (taken is True and lbl in ("sig", "true"))
                or (taken is False and lbl in ("not sig", "false"))
            )
            render(child, new_prefix, i == len(children) - 1, lbl, child_on_path)

    if isinstance(root, str):
        return f"[{root}] ← SELECTED"
    render(root, "", True, None, on_path=True)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphviz renderer
# ---------------------------------------------------------------------------


def graphviz_available() -> bool:
    """Return True if the optional ``graphviz`` package is importable."""
    return importlib.util.find_spec("graphviz") is not None


def render_graphviz(
    root: Union[TreeNode, str],
    path: list[tuple[str, bool]],
    decision: str,
    p_values: dict[str, float] | None = None,
    alpha: float = 0.05,
    title: str | None = None,
) -> Any:
    """Render the decision tree as a ``graphviz.Digraph``.

    The traversed path is drawn with thicker, colored edges and filled
    nodes; the chosen leaf is highlighted in green.

    Requires the optional ``graphviz`` Python package.
    """
    graphviz = importlib.import_module("graphviz")
    p_values = p_values or {}
    path_dict = dict(path)

    dot = graphviz.Digraph()
    dot.attr("graph", rankdir="TB")
    if title:
        dot.attr(label=title, labelloc="t", fontsize="14")
    dot.attr("node", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    if isinstance(root, str):
        dot.node(
            "leaf",
            label=root,
            shape="box",
            style="filled,bold",
            fillcolor="#9be29b",
        )
        return dot

    leaf_counter = [0]

    def add(
        node: Union[TreeNode, str],
        parent_id: str | None,
        edge_label: str | None,
        on_path: bool,
    ) -> None:
        if isinstance(node, str):
            leaf_id = f"leaf{leaf_counter[0]}"
            leaf_counter[0] += 1
            is_chosen = on_path and node == decision
            dot.node(
                leaf_id,
                label=node,
                shape="box",
                style="filled" + (",bold" if is_chosen else ""),
                fillcolor="#9be29b" if is_chosen else "#f0f0f0",
            )
            if parent_id is not None:
                dot.edge(
                    parent_id,
                    leaf_id,
                    label=edge_label or "",
                    color="#cc3333" if on_path else "#888888",
                    penwidth="2.0" if on_path else "1.0",
                )
            return

        label = _node_label(node)
        if on_path and node.kind == "test" and node.name in p_values:
            label = f"{label}\np={p_values[node.name]:.3f}"
        dot.node(
            node._node_id,
            label=label,
            shape="ellipse",
            style="filled" + (",bold" if on_path else ""),
            fillcolor="#ffd966" if on_path else "white",
        )
        if parent_id is not None:
            dot.edge(
                parent_id,
                node._node_id,
                label=edge_label or "",
                color="#cc3333" if on_path else "#888888",
                penwidth="2.0" if on_path else "1.0",
            )

        taken = path_dict.get(node._node_id)
        true_label = "sig" if node.kind == "test" else "true"
        false_label = "not sig" if node.kind == "test" else "false"
        add(node.if_true, node._node_id, true_label, on_path and taken is True)
        add(node.if_false, node._node_id, false_label, on_path and taken is False)

    add(root, None, None, on_path=True)
    return dot


# ---------------------------------------------------------------------------
# Public dispatch helper
# ---------------------------------------------------------------------------


def render(
    root: Union[TreeNode, str],
    path: list[tuple[str, bool]],
    decision: str,
    p_values: dict[str, float] | None = None,
    alpha: float = 0.05,
    fmt: str = "graphviz",
    title: str | None = None,
) -> Any:
    """Dispatch to the requested renderer, with graphviz availability check.

    If ``fmt == "graphviz"`` and the ``graphviz`` package is not installed,
    a :class:`UserWarning` is issued and the ASCII rendering is returned.
    """
    if fmt == "model":
        return decision
    if fmt == "ascii":
        return render_ascii(root, path, decision, p_values=p_values, alpha=alpha)
    if fmt == "graphviz":
        if not graphviz_available():
            warnings.warn(
                "graphviz package is not installed; falling back to ASCII "
                "rendering. Install with `pip install graphviz` (and the "
                "system 'graphviz' binary) to enable graph output.",
                UserWarning,
                stacklevel=3,
            )
            return render_ascii(root, path, decision, p_values=p_values, alpha=alpha)
        return render_graphviz(
            root, path, decision, p_values=p_values, alpha=alpha, title=title
        )
    raise ValueError(
        f"unknown format {fmt!r}; expected 'graphviz', 'ascii', or 'model'"
    )


# ---------------------------------------------------------------------------
# Cross-sectional decision tree specs
# ---------------------------------------------------------------------------


def _ols_spec() -> TreeNode:
    """OLS decision tree (Koley & Bera 2024, stge_kb)."""
    # Both naive sig: robust pair branch
    both_sig = TreeNode(
        kind="test",
        name="Robust-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-Error",
            if_true="SARAR",
            if_false="SAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-Error",
            if_true="SEM",
            if_false=TreeNode(
                kind="predicate",
                name="LM-Lag p <= LM-Error p",
                if_true="SAR",
                if_false="SEM",
                predicate_id="lag_pval_le_error_pval",
            ),
        ),
    )
    return TreeNode(
        kind="test",
        name="LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="LM-Error",
            if_true=both_sig,
            if_false="SAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="LM-Error",
            if_true="SEM",
            if_false="OLS",
        ),
    )


def _sar_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="LM-Error",
        if_true="SARAR",
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true="SDM",
            if_false=TreeNode(
                kind="test",
                name="LM-WX",
                if_true="SDM",
                if_false="SAR",
            ),
        ),
    )


def _sem_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="LM-Lag",
        if_true="SARAR",
        if_false=TreeNode(
            kind="test",
            name="LM-WX",
            if_true="SDEM",
            if_false="SEM",
        ),
    )


def _slx_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="Robust-LM-Lag-SDM",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-Error-SDEM",
            if_true="MANSAR",
            if_false="SDM",
        ),
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-Error-SDEM",
            if_true="SDEM",
            if_false="SLX",
        ),
    )


def _sdm_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="LM-Error-SDM",
        if_true="MANSAR",
        if_false="SDM",
    )


def _sdem_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="LM-Lag-SDEM",
        if_true="MANSAR",
        if_false="SDEM",
    )


# Tobit variants: same logic with -Tobit suffix on leaves.
def _sar_tobit_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="LM-Error",
        if_true="SARAR-Tobit",
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true="SDM-Tobit",
            if_false=TreeNode(
                kind="test",
                name="LM-WX",
                if_true="SDM-Tobit",
                if_false="SAR-Tobit",
            ),
        ),
    )


def _sem_tobit_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="LM-Lag",
        if_true="SARAR-Tobit",
        if_false=TreeNode(
            kind="test",
            name="LM-WX",
            if_true="SDEM-Tobit",
            if_false="SEM-Tobit",
        ),
    )


def _sdm_tobit_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="LM-Error",
        if_true="MANSAR-Tobit",
        if_false="SDM-Tobit",
    )


_CROSS_SECTIONAL_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLS": _ols_spec,
    "SAR": _sar_spec,
    "SEM": _sem_spec,
    "SLX": _slx_spec,
    "SDM": _sdm_spec,
    "SDEM": _sdem_spec,
    "SARTobit": _sar_tobit_spec,
    "SEMTobit": _sem_tobit_spec,
    "SDMTobit": _sdm_tobit_spec,
}


def get_spec(model_type: str) -> TreeNode | str:
    """Return the cross-sectional decision tree for ``model_type``.

    Falls back to a single leaf with the model name if no spec exists
    (matches the current ``return model_type`` fallback behavior).
    """
    factory = _CROSS_SECTIONAL_SPECS.get(model_type)
    if factory is None:
        return model_type
    return factory()


# ---------------------------------------------------------------------------
# Panel decision tree specs
# ---------------------------------------------------------------------------


def _panel_ols_spec(suffix: str) -> TreeNode:
    both_sig = TreeNode(
        kind="test",
        name="Panel-Robust-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error",
            if_true=f"SARARPanel{suffix}",
            if_false=f"SARPanel{suffix}",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error",
            if_true=f"SEMPanel{suffix}",
            if_false=TreeNode(
                kind="predicate",
                name="Panel-LM-Lag p <= Panel-LM-Error p",
                if_true=f"SARPanel{suffix}",
                if_false=f"SEMPanel{suffix}",
                predicate_id="panel_lag_pval_le_error_pval",
            ),
        ),
    )
    # OLSPanel{RE} doesn't expose robust tests; the evaluator will simply
    # follow `if_false` when the test isn't present (sig_lookup returns
    # False for missing tests).
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Panel-LM-Error",
            if_true=both_sig,
            if_false=f"SARPanel{suffix}",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-LM-Error",
            if_true=f"SEMPanel{suffix}",
            if_false=f"OLSPanel{suffix}",
        ),
    )


def _panel_sar_spec(suffix: str) -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Error",
        if_true=f"SARARPanel{suffix}",
        if_false=TreeNode(
            kind="test",
            name="Panel-Robust-LM-WX",
            if_true=f"SDMPanel{suffix}",
            if_false=TreeNode(
                kind="test",
                name="Panel-LM-WX",
                if_true=f"SDMPanel{suffix}",
                if_false=f"SARPanel{suffix}",
            ),
        ),
    )


def _panel_sem_spec(suffix: str) -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag",
        if_true=f"SARARPanel{suffix}",
        if_false=TreeNode(
            kind="test",
            name="Panel-LM-WX",
            if_true=f"SDEMPanel{suffix}",
            if_false=f"SEMPanel{suffix}",
        ),
    )


def _panel_slx_spec(suffix: str) -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-Robust-LM-Lag-SDM",
        if_true=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error-SDEM",
            if_true=f"MANSARPanel{suffix}",
            if_false=f"SDMPanel{suffix}",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error-SDEM",
            if_true=f"SDEMPanel{suffix}",
            if_false=f"SLXPanel{suffix}",
        ),
    )


def _panel_sdm_spec(suffix: str) -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Error-SDM",
        if_true=f"MANSARPanel{suffix}",
        if_false=f"SDMPanel{suffix}",
    )


def _panel_sdem_spec(suffix: str) -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag-SDEM",
        if_true=f"MANSARPanel{suffix}",
        if_false=f"SDEMPanel{suffix}",
    )


_PANEL_SPECS: dict[str, Callable[[str], TreeNode]] = {
    "OLSPanel": _panel_ols_spec,
    "SARPanel": _panel_sar_spec,
    "SEMPanel": _panel_sem_spec,
    "SLXPanel": _panel_slx_spec,
    "SDMPanel": _panel_sdm_spec,
    "SDEMPanel": _panel_sdem_spec,
}


def get_panel_spec(model_type: str) -> TreeNode | str:
    """Return the panel decision tree for ``model_type``.

    The model type is expected to be ``<Family>Panel<Suffix>`` where
    ``Suffix`` is ``FE`` or ``RE``. Falls back to a leaf if not found.
    """
    suffix = "FE" if "FE" in model_type else "RE" if "RE" in model_type else ""
    for prefix, factory in _PANEL_SPECS.items():
        if model_type.startswith(prefix):
            return factory(suffix)
    return model_type
