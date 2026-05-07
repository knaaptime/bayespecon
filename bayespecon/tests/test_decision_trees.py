"""Fast unit tests for bayespecon.diagnostics._decision_trees.

Tests TreeNode, evaluate, render_ascii, render, get_spec, and get_panel_spec
covering branches not exercised by the integration tests.
"""

from __future__ import annotations

import pytest

from bayespecon.diagnostics._decision_trees import (
    TreeNode,
    evaluate,
    get_panel_spec,
    get_spec,
    render,
    render_ascii,
    render_graphviz,
)

# ---------------------------------------------------------------------------
# TreeNode basics
# ---------------------------------------------------------------------------


class TestTreeNode:
    def test_simple_tree(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        assert tree.kind == "test"
        assert tree.name == "LM-Lag"
        assert tree.if_true == "SAR"
        assert tree.if_false == "OLS"

    def test_predicate_node(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="lag_le_error",
        )
        assert tree.predicate_id == "lag_le_error"


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_leaf_string(self):
        result, path = evaluate("OLS", sig_lookup=lambda _: False)
        assert result == "OLS"
        assert path == []

    def test_single_test_sig(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda n: n == "LM-Lag")
        assert result == "SAR"
        assert len(path) == 1
        assert path[0][1] is True

    def test_single_test_not_sig(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "OLS"
        assert path[0][1] is False

    def test_nested_tree(self):
        inner = TreeNode(kind="test", name="LM-Error", if_true="SEM", if_false="OLS")
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false=inner)
        result, path = evaluate(tree, sig_lookup=lambda n: n == "LM-Error")
        assert result == "SEM"
        assert len(path) == 2

    def test_predicate_node(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="lag_le_error",
        )
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: False,
            predicate_lookup={"lag_le_error": lambda: True},
        )
        assert result == "SAR"

    def test_predicate_false(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="lag_le_error",
        )
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: False,
            predicate_lookup={"lag_le_error": lambda: False},
        )
        assert result == "SEM"

    def test_predicate_missing_id_raises(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id=None,
        )
        with pytest.raises(ValueError, match="predicate node missing predicate_id"):
            evaluate(tree, sig_lookup=lambda _: False)

    def test_predicate_missing_lookup_raises(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="missing_key",
        )
        with pytest.raises(KeyError):
            evaluate(tree, sig_lookup=lambda _: False, predicate_lookup={})

    def test_unknown_node_kind_raises(self):
        tree = TreeNode(kind="bad", name="x", if_true="A", if_false="B")
        with pytest.raises(ValueError, match="unknown node kind"):
            evaluate(tree, sig_lookup=lambda _: False)


# ---------------------------------------------------------------------------
# render_ascii
# ---------------------------------------------------------------------------


class TestRenderAscii:
    def test_leaf(self):
        result = render_ascii("OLS", [], "OLS")
        assert "OLS" in result
        assert "SELECTED" in result

    def test_simple_tree(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda n: n == "LM-Lag")
        text = render_ascii(tree, path, result)
        assert "SAR" in text
        assert "SELECTED" in text


# ---------------------------------------------------------------------------
# render dispatch
# ---------------------------------------------------------------------------


class TestRenderDispatch:
    def test_model_format(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        output = render(tree, path, result, fmt="model")
        assert output == "SAR"

    def test_ascii_format(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        output = render(tree, path, result, fmt="ascii")
        assert isinstance(output, str)
        assert "SAR" in output

    def test_unknown_format_raises(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        with pytest.raises(ValueError, match="unknown format"):
            render(tree, path, result, fmt="bad_format")

    def test_graphviz_fallback_warns(self, monkeypatch):
        """When graphviz is not installed, should warn and fall back to ASCII."""
        from bayespecon.diagnostics import _decision_trees as _dt

        monkeypatch.setattr(_dt, "graphviz_available", lambda: False)
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        with pytest.warns(UserWarning, match="graphviz package is not installed"):
            output = render(tree, path, result, fmt="graphviz")
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# get_spec / get_panel_spec
# ---------------------------------------------------------------------------


class TestGetSpec:
    def test_ols_spec(self):
        tree = get_spec("OLS")
        assert isinstance(tree, TreeNode)

    def test_sar_spec(self):
        tree = get_spec("SAR")
        assert isinstance(tree, TreeNode)

    def test_sem_spec(self):
        tree = get_spec("SEM")
        assert isinstance(tree, TreeNode)

    def test_slx_spec(self):
        tree = get_spec("SLX")
        assert isinstance(tree, TreeNode)

    def test_sdm_spec(self):
        tree = get_spec("SDM")
        assert isinstance(tree, TreeNode)

    def test_sdem_spec(self):
        tree = get_spec("SDEM")
        assert isinstance(tree, TreeNode)

    def test_unknown_returns_string(self):
        result = get_spec("UnknownModel")
        assert result == "UnknownModel"

    def test_ols_tree_evaluate_all_sig(self):
        """All naive + robust significant → SAR via robust p-value tie-break.

        ``SARAR`` is intentionally unreachable from the OLS tree because
        its proper null is a fitted SAR (or SEM) model.
        """
        tree = get_spec("OLS")
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: True,
            predicate_lookup={
                "robust_lag_pval_le_error_pval": lambda: True,
                "lag_pval_le_error_pval": lambda: True,
            },
        )
        assert result == "SAR"

    def test_ols_tree_evaluate_none_sig(self):
        """Walk the OLS tree with no tests significant → OLS."""
        tree = get_spec("OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "OLS"

    def test_ols_tree_only_lag_sig(self):
        """Only LM-Lag significant → SAR."""
        tree = get_spec("OLS")

        def lookup(name):
            return name == "LM-Lag"

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SAR"

    def test_ols_tree_only_error_sig(self):
        """Only LM-Error significant → SEM."""
        tree = get_spec("OLS")

        def lookup(name):
            return name == "LM-Error"

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SEM"

    def test_ols_tree_robust_lag_only(self):
        """Both naive sig, robust lag only → SAR."""
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error", "Robust-LM-Lag")

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SAR"

    def test_ols_tree_robust_error_only(self):
        """Both naive sig, robust error only → SEM."""
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error", "Robust-LM-Error")

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SEM"

    def test_ols_tree_robust_both(self):
        """Both naive and both robust sig → robust p-value tie-break (SAR/SEM).

        The OLS tree never reaches SARAR because SARAR's correct null is
        a fitted SAR (or SEM); the user must escalate by fitting that
        intermediate model and re-running diagnostics.
        """
        tree = get_spec("OLS")

        def lookup(name):
            return name in (
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag",
                "Robust-LM-Error",
            )

        # Robust-LM-Lag wins the tie-break -> SAR.
        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"robust_lag_pval_le_error_pval": lambda: True},
        )
        assert result == "SAR"

        # Robust-LM-Error wins the tie-break -> SEM.
        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"robust_lag_pval_le_error_pval": lambda: False},
        )
        assert result == "SEM"

    def test_ols_tree_robust_neither_predicate(self):
        """Both naive sig, neither robust → predicate fallback."""
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error")

        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"lag_pval_le_error_pval": lambda: True},
        )
        assert result == "SAR"

    def test_ols_tree_robust_neither_predicate_false(self):
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error")

        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"lag_pval_le_error_pval": lambda: False},
        )
        assert result == "SEM"


class TestGetPanelSpec:
    def test_ols_fe_spec(self):
        tree = get_panel_spec("OLSPanelFE")
        assert isinstance(tree, TreeNode)

    def test_sar_fe_spec(self):
        tree = get_panel_spec("SARPanelFE")
        assert isinstance(tree, TreeNode)

    def test_sem_fe_spec(self):
        tree = get_panel_spec("SEMPanelFE")
        assert isinstance(tree, TreeNode)

    def test_slx_fe_spec(self):
        tree = get_panel_spec("SLXPanelFE")
        assert isinstance(tree, TreeNode)

    def test_sdm_fe_spec(self):
        tree = get_panel_spec("SDMPanelFE")
        assert isinstance(tree, TreeNode)

    def test_sdem_fe_spec(self):
        tree = get_panel_spec("SDEMPanelFE")
        assert isinstance(tree, TreeNode)

    def test_ols_re_spec(self):
        tree = get_panel_spec("OLSPanelRE")
        assert isinstance(tree, TreeNode)

    def test_unknown_returns_string(self):
        result = get_panel_spec("UnknownPanel")
        assert result == "UnknownPanel"

    def test_panel_ols_all_sig(self):
        # All naive + robust significant: route via the robust p-value
        # tie-break to the dominant single-channel panel model.  SARAR is
        # intentionally unreachable from a panel-OLS fit.
        tree = get_panel_spec("OLSPanelFE")
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: True,
            predicate_lookup={
                "panel_robust_lag_pval_le_error_pval": lambda: True,
                "panel_lag_pval_le_error_pval": lambda: True,
            },
        )
        assert result == "SARPanelFE"

    def test_panel_ols_none_sig(self):
        tree = get_panel_spec("OLSPanelFE")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert "OLS" in result


class TestGetPanelDynamicSpec:
    """Dynamic panel decision trees reuse static panel structure with 'Dynamic' suffix."""

    def test_ols_dynamic_spec(self):
        tree = get_panel_spec("OLSPanelDynamic")
        assert isinstance(tree, TreeNode)

    def test_sar_dynamic_spec(self):
        tree = get_panel_spec("SARPanelDynamic")
        assert isinstance(tree, TreeNode)

    def test_sem_dynamic_spec(self):
        tree = get_panel_spec("SEMPanelDynamic")
        assert isinstance(tree, TreeNode)

    def test_slx_dynamic_spec(self):
        tree = get_panel_spec("SLXPanelDynamic")
        assert isinstance(tree, TreeNode)

    def test_sdm_dynamic_spec(self):
        tree = get_panel_spec("SDMPanelDynamic")
        assert isinstance(tree, TreeNode)

    def test_sdem_dynamic_spec(self):
        tree = get_panel_spec("SDEMPanelDynamic")
        assert isinstance(tree, TreeNode)

    def test_dynamic_ols_all_sig(self):
        tree = get_panel_spec("OLSPanelDynamic")
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: True,
            predicate_lookup={
                "panel_robust_lag_pval_le_error_pval": lambda: True,
                "panel_lag_pval_le_error_pval": lambda: True,
            },
        )
        assert result == "SARPanelDynamic"

    def test_dynamic_ols_none_sig(self):
        tree = get_panel_spec("OLSPanelDynamic")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert "OLS" in result and "Dynamic" in result


class TestGetPanelTobitSpec:
    """Panel Tobit decision trees mirror cross-sectional Tobit specs."""

    def test_sar_tobit_spec(self):
        tree = get_panel_spec("SARPanelTobit")
        assert isinstance(tree, TreeNode)

    def test_sem_tobit_spec(self):
        tree = get_panel_spec("SEMPanelTobit")
        assert isinstance(tree, TreeNode)

    def test_sar_tobit_all_sig(self):
        tree = get_panel_spec("SARPanelTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        assert result == "SDMPanelTobit"

    def test_sar_tobit_none_sig(self):
        tree = get_panel_spec("SARPanelTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "SARPanelTobit"

    def test_sem_tobit_all_sig(self):
        tree = get_panel_spec("SEMPanelTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        assert result == "SDEMPanelTobit"

    def test_sem_tobit_none_sig(self):
        tree = get_panel_spec("SEMPanelTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "SEMPanelTobit"


# ---------------------------------------------------------------------------
# Theme / styling tests
# ---------------------------------------------------------------------------


class TestGraphvizThemes:
    """Tests for the new style system in render_graphviz."""

    @pytest.fixture(scope="class")
    def simple_tree(self):
        return TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")

    @pytest.fixture(scope="class")
    def evaluated(self, simple_tree):
        return evaluate(simple_tree, sig_lookup=lambda _: True)

    def _assert_digraph(self, obj):
        """Helper: assert that obj is a graphviz.Digraph."""
        # graphviz may not be installed in all CI jobs, so skip if missing.
        graphviz = pytest.importorskip("graphviz")
        assert isinstance(obj, graphviz.Digraph)

    def test_default_theme_renders(self, simple_tree, evaluated):
        result, path = evaluated
        dot = render_graphviz(simple_tree, path, result, theme="default")
        self._assert_digraph(dot)

    def test_minimal_theme_renders(self, simple_tree, evaluated):
        result, path = evaluated
        dot = render_graphviz(simple_tree, path, result, theme="minimal")
        self._assert_digraph(dot)

    def test_dark_theme_renders(self, simple_tree, evaluated):
        result, path = evaluated
        dot = render_graphviz(simple_tree, path, result, theme="dark")
        self._assert_digraph(dot)

    def test_custom_graph_theme(self, simple_tree, evaluated):
        """Pass a custom GraphTheme object directly."""
        from bayespecon.diagnostics._decision_style import GraphTheme, NodeStyle

        custom = GraphTheme(
            name="custom",
            test_node=NodeStyle(fillcolor="#ff0000", color="#000000"),
            leaf_chosen=NodeStyle(fillcolor="#00ff00", color="#000000"),
            leaf_other=NodeStyle(fillcolor="#cccccc", color="#000000"),
        )
        result, path = evaluated
        dot = render_graphviz(simple_tree, path, result, theme=custom)
        self._assert_digraph(dot)

    def test_unknown_theme_raises(self, simple_tree, evaluated):
        result, path = evaluated
        with pytest.raises(ValueError, match="unknown theme"):
            render_graphviz(simple_tree, path, result, theme="nonexistent")

    def test_predicate_node_is_diamond_default(self):
        """In the default theme, predicate nodes should be diamonds."""
        graphviz = pytest.importorskip("graphviz")
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="lag_le_error",
        )
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: False,
            predicate_lookup={"lag_le_error": lambda: True},
        )
        dot = render_graphviz(tree, path, result, theme="default")
        assert isinstance(dot, graphviz.Digraph)
        # The predicate node is on the path (True branch taken).
        # Check that the node with id "n0" has shape diamond.
        # graphviz.Digraph stores nodes in dot.body as strings, so we
        # inspect the generated source.
        src = dot.source
        assert "shape=diamond" in src

    def test_pruned_edges_dashed_default(self):
        """Edges not on the active path should be dashed in default theme."""
        graphviz = pytest.importorskip("graphviz")
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        dot = render_graphviz(tree, path, result, theme="default")
        src = dot.source
        # The active edge (to SAR) should be solid (no style=dashed).
        # The pruned edge (to OLS) should be dashed.
        assert "style=dashed" in src

    def test_render_dispatch_passes_theme(self, simple_tree, evaluated):
        """render(..., fmt='graphviz', theme=...) should forward the theme."""
        graphviz = pytest.importorskip("graphviz")
        result, path = evaluated
        dot = render(simple_tree, path, result, fmt="graphviz", theme="dark")
        assert isinstance(dot, graphviz.Digraph)
        # Dark theme has a dark background.
        assert 'bgcolor="#1e1e1e"' in dot.source

    def test_leaf_string_with_theme(self):
        """A bare string root should still render correctly with a theme."""
        graphviz = pytest.importorskip("graphviz")
        dot = render_graphviz("OLS", [], "OLS", theme="dark")
        assert isinstance(dot, graphviz.Digraph)
        assert "OLS" in dot.source

    def test_p_value_annotation_present(self):
        """When p-values are provided, they should appear in the node label."""
        graphviz = pytest.importorskip("graphviz")
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        dot = render_graphviz(
            tree, path, result, p_values={"LM-Lag": 0.0123}, theme="default"
        )
        src = dot.source
        assert "p=0.012" in src
