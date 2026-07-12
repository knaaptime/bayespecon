"""Spatial-structure strategy objects (Phase 5c).

These encapsulate the structure-specific operations that differ between
cross-section and panel models — the spatial lag and the non-eigenvalue ``W``
operand handed to the log-determinant factory — behind one small interface so
the two model base classes (:class:`SpatialModel`, :class:`SpatialPanelModel`)
can share a single implementation of the surrounding methods and, eventually,
collapse into one class parameterised by its structure.

Only the concerns that are genuinely structure-dependent live here; everything
behaviour-identical across structures stays in
:class:`bayespecon.models._base._shared.SharedSpatialMethods`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SpatialStructure(ABC):
    """Strategy encapsulating structure-specific spatial operations."""

    @abstractmethod
    def spatial_lag(self, x: np.ndarray) -> np.ndarray:
        """Return the spatial lag of ``x`` (a vector or a column-stacked matrix)."""

    @abstractmethod
    def logdet_W_operand(self):
        """Return the non-eigenvalue ``W`` operand for the logdet factory."""


class CrossSectionStructure(SpatialStructure):
    """Cross-section: a single ``N×N`` weights matrix ``W``.

    The lag is the plain ``W @ x`` product and the logdet factory receives the
    dense ``W`` (mirrors the historical cross-section behaviour).
    """

    def __init__(self, W_sparse):
        self._W_sparse = W_sparse

    def spatial_lag(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._W_sparse @ x, dtype=np.float64)

    def logdet_W_operand(self):
        return self._W_sparse.toarray().astype(np.float64)


class PanelStructure(SpatialStructure):
    """Panel: ``W`` acts per period through the Kronecker structure ``W ⊗ I_T``.

    Accepts either an ``N×N`` ``W`` (applied per period in one batched matmul)
    or a full ``(N*T)×(N*T)`` block matrix.  The logdet factory receives the
    sparse ``W`` (the panel logdet path stays sparse).
    """

    def __init__(self, W_sparse, N: int, T: int):
        self._W_sparse = W_sparse
        self._N = int(N)
        self._T = int(T)

    def spatial_lag(self, v: np.ndarray) -> np.ndarray:
        W = self._W_sparse
        N, T = self._N, self._T
        v = np.asarray(v, dtype=float)
        if W.shape[0] == N:
            if v.ndim == 1:
                # Stack ordered (T, N); apply W per period in one matmul.
                chunks = v.reshape(T, N)  # (T, N)
                return np.asarray((W @ chunks.T).T, dtype=float).ravel()
            # 2-D path: (N*T, k) → reshape so all periods/columns become a
            # single dense block, perform ONE sparse matmul, then reshape back.
            k = v.shape[1]
            chunks = v.reshape(T, N, k)  # (T, N, k)
            mat = chunks.transpose(1, 0, 2).reshape(N, T * k)
            out = np.asarray(W @ mat, dtype=float)  # (N, T*k)
            return out.reshape(N, T, k).transpose(1, 0, 2).reshape(T * N, k)
        # Full (N*T)×(N*T) block matrix provided.
        return np.asarray(W @ v, dtype=float)

    def logdet_W_operand(self):
        return self._W_sparse
