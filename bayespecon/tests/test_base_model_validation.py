"""Fast unit tests for bayespecon.models.base input validation and decision tree.

Tests _parse_W error paths, SpatialModel.__init__ formula mode,
w_vars validation, and spatial_diagnostics_decision branches
without running MCMC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from bayespecon.models.base import _parse_W

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rook_W(n: int = 4) -> np.ndarray:
    """Row-standardized rook W on a line of n units."""
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def _W_to_graph(W_dense: np.ndarray) -> Graph:
    n = W_dense.shape[0]
    focal, neighbor, weight = [], [], []
    for i in range(n):
        for j in range(n):
            if W_dense[i, j] != 0:
                focal.append(i)
                neighbor.append(j)
                weight.append(W_dense[i, j])
    return Graph.from_arrays(
        np.array(focal, int), np.array(neighbor, int), np.array(weight, float)
    ).transform("r")


# ---------------------------------------------------------------------------
# _parse_W
# ---------------------------------------------------------------------------


class TestParseW:
    """Tests for _parse_W validation."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    def test_accepts_graph(self, W_graph):
        W_csr, row_std = _parse_W(W_graph, n=4)
        assert W_csr.shape == (4, 4)
        assert row_std is True

    def test_accepts_sparse_matrix(self):
        W_sp = sp.csr_matrix(_rook_W(4))
        W_csr, row_std = _parse_W(W_sp, n=4)
        assert W_csr.shape == (4, 4)

    def test_rejects_legacy_libpysal_W(self):
        """A mock object with .sparse and .transform should raise TypeError."""

        class FakeLegacyW:
            sparse = sp.csr_matrix(_rook_W(4))
            transform = "r"

        with pytest.raises(TypeError, match="legacy libpysal.weights.W"):
            _parse_W(FakeLegacyW(), n=4)

    def test_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="W must be a libpysal.graph.Graph"):
            _parse_W(np.ones((4, 4)), n=4)

    def test_rejects_non_square(self):
        W_rect = sp.csr_matrix(np.ones((3, 4)))
        with pytest.raises(ValueError, match="W must be a square matrix"):
            _parse_W(W_rect, n=4)

    def test_rejects_wrong_size(self):
        W_sp = sp.csr_matrix(_rook_W(4))
        with pytest.raises(ValueError, match="W has shape"):
            _parse_W(W_sp, n=5)

    def test_warns_non_row_standardized(self):
        W = sp.csr_matrix(np.ones((4, 4)))  # Not row-standardized
        with pytest.warns(UserWarning, match="row-standardised"):
            _parse_W(W, n=4)


# ---------------------------------------------------------------------------
# SpatialModel.__init__ error paths
# ---------------------------------------------------------------------------


class TestSpatialModelInit:
    """Test SpatialModel constructor validation (no MCMC)."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    def test_formula_mode_requires_data(self, W_graph):
        from bayespecon.models.ols import OLS

        with pytest.raises(ValueError, match="data must be provided"):
            OLS(formula="y ~ x1", data=None, W=W_graph)

    def test_formula_mode_creates_model(self, W_graph):
        from bayespecon.models.ols import OLS

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "y": rng.standard_normal(4),
                "x1": rng.standard_normal(4),
            }
        )
        model = OLS(formula="y ~ x1", data=df, W=W_graph)
        assert model._y.shape == (4,)
        assert "Intercept" in model._feature_names or "x1" in model._feature_names

    def test_matrix_mode_creates_model(self, W_graph):
        from bayespecon.models.ols import OLS

        rng = np.random.default_rng(0)
        model = OLS(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        assert model._y.shape == (4,)

    def test_no_formula_no_matrices_raises(self, W_graph):
        from bayespecon.models.ols import OLS

        with pytest.raises(ValueError, match="Provide either"):
            OLS(W=W_graph)

    def test_w_vars_unknown_raises(self, W_graph):
        from bayespecon.models.slx import SLX

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="w_vars contains names not found"):
            SLX(
                y=rng.standard_normal(4),
                X=rng.standard_normal((4, 2)),
                W=W_graph,
                w_vars=["nonexistent"],
            )


# ---------------------------------------------------------------------------
# spatial_diagnostics_decision for cross-sectional models
# ---------------------------------------------------------------------------


class TestCrossSectionalDiagnosticsDecision:
    """Test spatial_diagnostics_decision branches for cross-sectional models."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    @pytest.fixture
    def ols_model(self, W_graph):
        from bayespecon.models.ols import OLS

        rng = np.random.default_rng(0)
        return OLS(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)

    def test_ols_only_lag(self, ols_model, monkeypatch):
        df = pd.DataFrame({"p_value": [0.001, 0.9]}, index=["LM-Lag", "LM-Error"])
        monkeypatch.setattr(ols_model, "spatial_diagnostics", lambda: df)
        assert (
            ols_model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"
        )

    def test_ols_only_error(self, ols_model, monkeypatch):
        df = pd.DataFrame({"p_value": [0.9, 0.001]}, index=["LM-Lag", "LM-Error"])
        monkeypatch.setattr(ols_model, "spatial_diagnostics", lambda: df)
        assert (
            ols_model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SEM"
        )

    def test_ols_neither(self, ols_model, monkeypatch):
        df = pd.DataFrame({"p_value": [0.9, 0.9]}, index=["LM-Lag", "LM-Error"])
        monkeypatch.setattr(ols_model, "spatial_diagnostics", lambda: df)
        assert (
            ols_model.spatial_diagnostics_decision(alpha=0.05, format="model") == "OLS"
        )

    def test_ols_both_robust_lag(self, ols_model, monkeypatch):
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.001, 0.9]},
            index=["LM-Lag", "LM-Error", "Robust-LM-Lag", "Robust-LM-Error"],
        )
        monkeypatch.setattr(ols_model, "spatial_diagnostics", lambda: df)
        assert (
            ols_model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"
        )

    def test_ols_both_robust_error(self, ols_model, monkeypatch):
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.9, 0.001]},
            index=["LM-Lag", "LM-Error", "Robust-LM-Lag", "Robust-LM-Error"],
        )
        monkeypatch.setattr(ols_model, "spatial_diagnostics", lambda: df)
        assert (
            ols_model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SEM"
        )

    def test_ols_both_robust_both(self, ols_model, monkeypatch):
        # Both naive AND both robust significant: route to the dominant
        # single-channel model via the robust p-value tie-break.  Equal
        # robust p-values resolve in favour of SAR (lag <= error).
        # SARAR is intentionally unreachable from OLS because its proper
        # null is a fitted SAR (or SEM) — the user must escalate.
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.001, 0.001]},
            index=["LM-Lag", "LM-Error", "Robust-LM-Lag", "Robust-LM-Error"],
        )
        monkeypatch.setattr(ols_model, "spatial_diagnostics", lambda: df)
        assert (
            ols_model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SAR"
        )

    def test_ols_both_no_robust_fallback(self, ols_model, monkeypatch):
        """When both naive tests fire but no robust tests available, fall back to p-value comparison."""
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Lag", "LM-Error"],
        )
        monkeypatch.setattr(ols_model, "spatial_diagnostics", lambda: df)
        result = ols_model.spatial_diagnostics_decision(alpha=0.05, format="model")
        # Should fall back to comparing p-values; both equal so LM-Lag wins (<=)
        assert result == "SAR"

    def test_sar_error_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sar import SAR

        rng = np.random.default_rng(0)
        model = SAR(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        # SARAR is reachable only when naive ``LM-Error`` AND its robust
        # refinement ``Robust-LM-Error`` (Schur-purged for ρ) both fire.
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Error", "Robust-LM-Error"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SARAR"

    def test_sar_error_naive_only_falls_through(self, W_graph, monkeypatch):
        # If only naive ``LM-Error`` fires but the robust refinement
        # clears it, the SAR fit already absorbs the dependence; keep SAR.
        from bayespecon.models.sar import SAR

        rng = np.random.default_rng(0)
        model = SAR(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.9]},
            index=["LM-Error", "Robust-LM-Error"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"

    def test_sar_wx_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sar import SAR

        rng = np.random.default_rng(0)
        model = SAR(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        # SDM is reachable only when naive ``LM-WX`` AND its robust
        # refinement both fire (precondition principle).
        df = pd.DataFrame(
            {"p_value": [0.9, 0.001, 0.001]},
            index=["LM-Error", "LM-WX", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_sar_wx_naive_only_falls_through(self, W_graph, monkeypatch):
        # If naive ``LM-WX`` fires but the robust refinement clears it, the
        # tree must fall through to the LM-Error branch — not commit to SDM.
        from bayespecon.models.sar import SAR

        rng = np.random.default_rng(0)
        model = SAR(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.9, 0.001, 0.9]},
            index=["LM-Error", "LM-WX", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"

    def test_sar_robust_only_does_not_route_sdm(self, W_graph, monkeypatch):
        # Robust-LM-WX firing without the naive precursor must NOT commit
        # the tree to SDM.
        from bayespecon.models.sar import SAR

        rng = np.random.default_rng(0)
        model = SAR(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.9, 0.9, 0.001]},
            index=["LM-Error", "LM-WX", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"

    def test_sar_neither_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sar import SAR

        rng = np.random.default_rng(0)
        model = SAR(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.9, 0.9, 0.9]},
            index=["LM-Error", "LM-WX", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"

    def test_sem_lag_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sem import SEM

        rng = np.random.default_rng(0)
        model = SEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        # SARAR requires both naive ``LM-Lag`` AND its robust refinement
        # ``Robust-LM-Lag`` (Schur-purged for the WX block) to fire.
        df = pd.DataFrame(
            {"p_value": [0.001, 0.9, 0.001, 0.9]},
            index=["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SARAR"

    def test_sem_lag_naive_only_keeps_sem(self, W_graph, monkeypatch):
        from bayespecon.models.sem import SEM

        rng = np.random.default_rng(0)
        model = SEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.9, 0.9, 0.9]},
            index=["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SEM"

    def test_sem_wx_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sem import SEM

        rng = np.random.default_rng(0)
        model = SEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        # SDEM requires both naive ``LM-WX`` AND ``Robust-LM-WX`` to fire.
        df = pd.DataFrame(
            {"p_value": [0.9, 0.001, 0.9, 0.001]},
            index=["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_sem_all_robust_lag_wins(self, W_graph, monkeypatch):
        from bayespecon.models.sem import SEM

        rng = np.random.default_rng(0)
        model = SEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        # Both naive precursors fire; both robust survive; smaller-p
        # (= larger statistic) side wins.  Lag side has smaller p → SARAR.
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.0001, 0.01]},
            index=["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SARAR"

    def test_sem_all_robust_wx_wins(self, W_graph, monkeypatch):
        from bayespecon.models.sem import SEM

        rng = np.random.default_rng(0)
        model = SEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.01, 0.0001]},
            index=["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_sdm_robust_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sdm import SDM

        rng = np.random.default_rng(0)
        model = SDM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Error-SDM", "Robust-LM-Error-SDM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "MANSAR"

    def test_sdm_naive_only_keeps_sdm(self, W_graph, monkeypatch):
        from bayespecon.models.sdm import SDM

        rng = np.random.default_rng(0)
        model = SDM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.9]},
            index=["LM-Error-SDM", "Robust-LM-Error-SDM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_sdem_robust_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sdem import SDEM

        rng = np.random.default_rng(0)
        model = SDEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Lag-SDEM", "Robust-LM-Lag-SDEM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "MANSAR"

    def test_sdem_naive_only_keeps_sdem(self, W_graph, monkeypatch):
        from bayespecon.models.sdem import SDEM

        rng = np.random.default_rng(0)
        model = SDEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.9]},
            index=["LM-Lag-SDEM", "Robust-LM-Lag-SDEM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_slx_robust_lag_sdm(self, W_graph, monkeypatch):
        from bayespecon.models.slx import SLX

        rng = np.random.default_rng(0)
        model = SLX(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        # Naive ``LM-Lag`` must fire to reach the robust refinement.
        df = pd.DataFrame(
            {"p_value": [0.001, 0.9, 0.001, 0.9]},
            index=[
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag-SDM",
                "Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_slx_robust_error_sdem(self, W_graph, monkeypatch):
        from bayespecon.models.slx import SLX

        rng = np.random.default_rng(0)
        model = SLX(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        # Naive ``LM-Error`` must fire to reach the robust refinement.
        df = pd.DataFrame(
            {"p_value": [0.9, 0.001, 0.9, 0.001]},
            index=[
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag-SDM",
                "Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_slx_robust_both_lag_wins(self, W_graph, monkeypatch):
        # When both robust tests fire (with both naive precursors firing),
        # the smaller-p (= larger statistic) side wins.  Lag-SDM has the
        # smaller p-value here, so SDM is preferred.
        from bayespecon.models.slx import SLX

        rng = np.random.default_rng(0)
        model = SLX(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.0001, 0.01]},
            index=[
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag-SDM",
                "Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_slx_robust_both_error_wins(self, W_graph, monkeypatch):
        from bayespecon.models.slx import SLX

        rng = np.random.default_rng(0)
        model = SLX(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.01, 0.0001]},
            index=[
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag-SDM",
                "Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_slx_neither_robust(self, W_graph, monkeypatch):
        from bayespecon.models.slx import SLX

        rng = np.random.default_rng(0)
        model = SLX(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.9, 0.9, 0.9, 0.9]},
            index=[
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag-SDM",
                "Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SLX"

    def test_slx_robust_only_no_naive_routes_slx(self, W_graph, monkeypatch):
        # Robust tests firing without naive precursors must NOT escalate.
        from bayespecon.models.slx import SLX

        rng = np.random.default_rng(0)
        model = SLX(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.9, 0.9, 0.001, 0.001]},
            index=[
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag-SDM",
                "Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SLX"

    def test_sdm_error_sdm_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sdm import SDM

        rng = np.random.default_rng(0)
        model = SDM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Error-SDM", "Robust-LM-Error-SDM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model") == "MANSAR"
        )

    def test_sdem_lag_sdem_significant(self, W_graph, monkeypatch):
        from bayespecon.models.sdem import SDEM

        rng = np.random.default_rng(0)
        model = SDEM(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Lag-SDEM", "Robust-LM-Lag-SDEM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model") == "MANSAR"
        )

    def test_sartobit_error_significant(self, W_graph, monkeypatch):
        from bayespecon.models.tobit import SARTobit

        rng = np.random.default_rng(0)
        model = SARTobit(
            y=rng.standard_normal(4),
            X=rng.standard_normal((4, 2)),
            W=W_graph,
            censoring=0.0,
        )
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Error", "Robust-LM-Error"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SARAR-Tobit"
        )

    def test_semtobit_lag_significant(self, W_graph, monkeypatch):
        from bayespecon.models.tobit import SEMTobit

        rng = np.random.default_rng(0)
        model = SEMTobit(
            y=rng.standard_normal(4),
            X=rng.standard_normal((4, 2)),
            W=W_graph,
            censoring=0.0,
        )
        df = pd.DataFrame(
            {"p_value": [0.001, 0.9, 0.001, 0.9]},
            index=["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SARAR-Tobit"
        )

    def test_sdmtobit_error_significant(self, W_graph, monkeypatch):
        from bayespecon.models.tobit import SDMTobit

        rng = np.random.default_rng(0)
        model = SDMTobit(
            y=rng.standard_normal(4),
            X=rng.standard_normal((4, 2)),
            W=W_graph,
            censoring=0.0,
        )
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Error", "Robust-LM-Error"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "MANSAR-Tobit"
        )


# ---------------------------------------------------------------------------
# Output format tests for spatial_diagnostics_decision
# ---------------------------------------------------------------------------


class TestDecisionOutputFormats:
    """format= kwarg for spatial_diagnostics_decision (ascii, graphviz, model)."""

    @pytest.fixture
    def fitted_ols(self):
        from bayespecon.models.ols import OLS

        rng = np.random.default_rng(0)
        return OLS(
            y=rng.standard_normal(4),
            X=rng.standard_normal((4, 2)),
            W=_W_to_graph(_rook_W(4)),
        )

    @pytest.fixture
    def diag_df(self):
        return pd.DataFrame(
            {"p_value": [0.001, 0.9, 0.001, 0.9]},
            index=["LM-Lag", "LM-Error", "Robust-LM-Lag", "Robust-LM-Error"],
        )

    def test_format_model_returns_string(self, fitted_ols, diag_df, monkeypatch):
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision(format="model")
        assert isinstance(result, str)
        assert result == "SAR"

    def test_format_ascii_returns_tree_string(self, fitted_ols, diag_df, monkeypatch):
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision(format="ascii")
        assert isinstance(result, str)
        assert "← SELECTED" in result
        assert "SAR" in result
        # The chosen leaf line must contain the marker
        chosen_lines = [ln for ln in result.splitlines() if "SELECTED" in ln]
        assert len(chosen_lines) == 1
        assert "[SAR]" in chosen_lines[0]

    def test_format_graphviz_returns_digraph(self, fitted_ols, diag_df, monkeypatch):
        graphviz = pytest.importorskip("graphviz")
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision(format="graphviz")
        assert isinstance(result, graphviz.Digraph)
        src = result.source
        assert "SAR" in src

    def test_format_graphviz_default(self, fitted_ols, diag_df, monkeypatch):
        """graphviz is the default when the package is available."""
        pytest.importorskip("graphviz")
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision()
        # graphviz.Digraph has a .source attribute
        assert hasattr(result, "source")

    def test_graphviz_fallback_when_unavailable(self, fitted_ols, diag_df, monkeypatch):
        """If graphviz is not installed, warn and fall back to ASCII."""
        from bayespecon.diagnostics import _decision_trees as _dt

        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        # Force `find_spec("graphviz")` to return None
        monkeypatch.setattr(
            _dt.importlib.util,
            "find_spec",
            lambda name: (
                None
                if name == "graphviz"
                else __import__("importlib").util.find_spec(name)
            ),
        )
        with pytest.warns(UserWarning, match="graphviz package is not installed"):
            result = fitted_ols.spatial_diagnostics_decision(format="graphviz")
        assert isinstance(result, str)
        assert "← SELECTED" in result

    def test_invalid_format_raises(self, fitted_ols, diag_df, monkeypatch):
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        with pytest.raises(ValueError, match="unknown format"):
            fitted_ols.spatial_diagnostics_decision(format="bogus")

    def test_sar_ascii_error_branch_drawn_per_channel(self):
        """The SAR error_branch is intentionally built twice (not shared)
        so that the graphviz/ASCII renderer draws naive→robust strictly
        within each channel — i.e. the reader sees a separate ``LM-Error``
        subtree under both the ``LM-WX`` "not sig" edge and the
        ``Robust-LM-WX`` "not sig" edge, never a shared back-reference
        that visually places ``LM-Error`` ahead of ``Robust-LM-WX``.
        """
        from bayespecon.diagnostics import _decision_trees as _dt

        spec = _dt.get_spec("SAR")
        sig = lambda name: {"LM-WX": False, "LM-Error": True, "Robust-LM-WX": False}.get(
            name, False
        )
        d, p = _dt.evaluate(spec, sig)
        result = _dt.render_ascii(
            spec,
            p,
            d,
            p_values={"LM-WX": 0.5, "LM-Error": 0.01, "Robust-LM-WX": 0.5},
            alpha=0.05,
        )
        lines = result.splitlines()
        # Full node expansions of LM-Error: lines that contain " LM-Error"
        # (preceded by space — to exclude the substring match in
        # "Robust-LM-Error", which is itself an inner gate inside each
        # error_branch subtree).
        full_expansions = [
            ln for ln in lines
            if " LM-Error" in ln and "see above" not in ln and "→" not in ln
        ]
        assert len(full_expansions) == 2, (
            f"Expected two full LM-Error expansions (one per channel), "
            f"got {len(full_expansions)}:\n{result}"
        )
        # No back-references should remain — the subtrees are independent.
        back_refs = [ln for ln in lines if "→" in ln and "LM-Error" in ln]
        assert len(back_refs) == 0, (
            f"Expected no back-references to LM-Error, got {len(back_refs)}:\n{result}"
        )
