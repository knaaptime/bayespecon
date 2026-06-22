"""Standalone Kronecker solve utilities (no PyMC dependency)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def kron_solve_vec(
    Lo: sp.csr_matrix,
    Ld: sp.csr_matrix,
    b: np.ndarray,
    n: int,
) -> np.ndarray:
    r"""Solve :math:`(L_o \otimes L_d)\,\eta = b` via two :math:`n \times n` sparse solves.

    Uses the vec-permutation identity
    :math:`(L_o \otimes L_d)\operatorname{vec}(H) = \operatorname{vec}(L_d H L_o^\top)`:

    1. :math:`H' = L_d^{-1} H_b`
    2. :math:`Z  = L_o^{-1} H'^{\,\top}` (i.e. solve :math:`L_o Z = H'^\top`)
    3. :math:`\eta = \operatorname{vec}(Z^\top)`

    Parameters
    ----------
    Lo, Ld : scipy.sparse.csr_matrix, shape (n, n)
        Factor matrices :math:`L_o = I_n - \rho_o W` and
        :math:`L_d = I_n - \rho_d W`.
    b : ndarray, shape (N,) where :math:`N = n^2`
    n : int
        Number of spatial units.

    Returns
    -------
    eta : ndarray, shape (N,)
    """
    Hb = b.reshape(n, n, order="F")
    Hp = sp.linalg.spsolve(Ld, Hb)
    Z = sp.linalg.spsolve(Lo, Hp.T)
    return np.asarray(Z, dtype=np.float64).T.ravel(order="F")


def kron_solve_matrix(
    Lo: sp.csr_matrix,
    Ld: sp.csr_matrix,
    B: np.ndarray,
    n: int,
) -> np.ndarray:
    r"""Solve :math:`(L_o \otimes L_d)\,H = B` for a matrix RHS via batched two-step solve.

    Applies the same Kronecker algorithm as :func:`kron_solve_vec` to all
    *k* columns of *B* simultaneously using a single :math:`L_d` factorisation
    and a single :math:`L_o^\top` factorisation (both of size :math:`n \times n`).

    Parameters
    ----------
    Lo, Ld : scipy.sparse.csr_matrix, shape (n, n)
    B : ndarray, shape (N, k) where :math:`N = n^2`
    n : int

    Returns
    -------
    H : ndarray, shape (N, k)
    """
    k = B.shape[1]
    R = B.reshape(n, n * k, order="F")
    Hp = sp.linalg.spsolve(Ld, R)
    Hp3 = Hp.reshape(n, n, k, order="F")
    RHS2 = Hp3.transpose(2, 0, 1).reshape(k * n, n).T
    Z_h = sp.linalg.spsolve(Lo, RHS2)
    Z3 = Z_h.reshape(n, n, k, order="F")
    return np.asarray(
        Z3.transpose(1, 0, 2).reshape(n * n, k, order="F"), dtype=np.float64
    )
