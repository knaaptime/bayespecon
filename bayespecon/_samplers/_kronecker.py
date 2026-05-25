r"""Kronecker-product matvec primitives for separable flow models.

Provides O(n³) matrix-vector products for the N×N precision matrix
:math:`P = (L_d^\top L_d \otimes L_o^\top L_o) / \sigma^2 + \mathrm{diag}(\omega)`
that arises in the Pólya–Gamma Gibbs sampler for separable SAR flow models,
where :math:`N = n^2` and :math:`L_k = I_n - \rho_k W`.

The key identity is the vec-permutation (Kronecker) product rule:

.. math::

    (A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B\,X\,A^\top),

which reduces an :math:`N \times N` matvec to two :math:`n \times n` dense
products — :math:`O(n^3)` instead of :math:`O(n^6)`.

References
----------
LeSage, J. P., & Pace, R. K. (2008). *Spatial Econometric Modeling of
Origin-Destination Flows*. Journal of Regional Science, 48(5), 941–967.
"""

from __future__ import annotations

import numpy as np


def kron_matvec(
    v: np.ndarray,
    Ld: np.ndarray,
    Lo: np.ndarray,
) -> np.ndarray:
    r"""Compute :math:`(L_o \otimes L_d)\,v` via the vec-permutation identity.

    Uses :math:`(L_o \otimes L_d)\,\mathrm{vec}(X) = \mathrm{vec}(L_d\,X\,L_o^\top)`
    where :math:`X` is the :math:`n \times n` column-major reshape of :math:`v`.

    Parameters
    ----------
    v : ndarray of shape (N,) where :math:`N = n^2`
        Vector to multiply.
    Ld : ndarray of shape (n, n)
        Destination factor matrix, typically :math:`L_d = I_n - \rho_d W`.
    Lo : ndarray of shape (n, n)
        Origin factor matrix, typically :math:`L_o = I_n - \rho_o W`.

    Returns
    -------
    ndarray of shape (N,)
        The product :math:`(L_o \otimes L_d)\,v`.

    Notes
    -----
    Cost: two :math:`n \times n` dense matrix products, :math:`O(n^3)`.
    """
    n = Ld.shape[0]
    X = v.reshape(n, n, order="F")
    return (Ld @ X @ Lo.T).ravel(order="F")


def kron_At_matvec(
    v: np.ndarray,
    Ld_T: np.ndarray,
    Lo_T: np.ndarray,
) -> np.ndarray:
    r"""Compute :math:`(L_o^\top \otimes L_d^\top)\,v` (adjoint Kronecker matvec).

    Uses :math:`(L_o^\top \otimes L_d^\top)\,\mathrm{vec}(X)
    = \mathrm{vec}(L_d^\top\,X\,L_o)` where :math:`X` is the
    :math:`n \times n` column-major reshape of :math:`v`.

    This is the transpose of :func:`kron_matvec` and is used to
    construct the right-hand side of the η draw:

    .. math::

        \mathit{rhs} = A^\top X\beta / \sigma^2 + \kappa,
        \quad A^\top = L_o^\top \otimes L_d^\top.

    Parameters
    ----------
    v : ndarray of shape (N,) where :math:`N = n^2`
        Vector to multiply.
    Ld_T : ndarray of shape (n, n)
        Transpose of the destination factor, typically :math:`L_d^\top`.
    Lo_T : ndarray of shape (n, n)
        Transpose of the origin factor, typically :math:`L_o^\top`.

    Returns
    -------
    ndarray of shape (N,)
        The product :math:`(L_o^\top \otimes L_d^\top)\,v`.

    Notes
    -----
    Cost: two :math:`n \times n` dense matrix products, :math:`O(n^3)`.
    """
    n = Ld_T.shape[0]
    X = v.reshape(n, n, order="F")
    return (Ld_T @ X @ Lo_T.T).ravel(order="F")


def kron_P_matvec(
    v: np.ndarray,
    LdtLd: np.ndarray,
    LotLo: np.ndarray,
    omega: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    r"""Matvec with the Kronecker-structured precision matrix.

    Computes :math:`P\,v` where

    .. math::

        P = (L_d^\top L_d \otimes L_o^\top L_o) / \sigma^2
            + \mathrm{diag}(\omega),

    using the vec-permutation identity for the Kronecker term and
    elementwise multiplication for the diagonal term.

    This is the central primitive for the separable flow Gibbs sampler.
    The Chebyshev η-draw and Lanczos log-determinant estimation both
    only require a matvec callback, so they plug in directly.

    Parameters
    ----------
    v : ndarray of shape (N,) where :math:`N = n^2`
        Vector to multiply.
    LdtLd : ndarray of shape (n, n)
        Product :math:`L_d^\top L_d`, precomputed once per ρ-draw.
    LotLo : ndarray of shape (n, n)
        Product :math:`L_o^\top L_o`, precomputed once per ρ-draw.
    omega : ndarray of shape (N,)
        Pólya–Gamma auxiliary variables.
    sigma2 : float
        Residual variance :math:`\sigma^2`.

    Returns
    -------
    ndarray of shape (N,)
        The product :math:`P\,v`.

    Notes
    -----
    Cost: two :math:`n \times n` dense matrix products plus one
    elementwise multiply, :math:`O(n^3)`.
    """
    n = LdtLd.shape[0]
    X = v.reshape(n, n, order="F")
    kron_part = (LdtLd @ X @ LotLo.T / sigma2).ravel(order="F")
    return kron_part + omega * v


def kron_logdet_A(
    rho_d: float,
    rho_o: float,
    n: int,
    logdet_fn: callable,
) -> float:
    r"""Compute :math:`\log|A|` for the separable flow model.

    Under the Kronecker factorisation
    :math:`A = L_o \otimes L_d` where :math:`L_k = I_n - \rho_k W`:

    .. math::

        \log|A| = n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    This is a trivial wrapper that calls the existing ``logdet_fn``
    twice and scales by :math:`n`.

    Parameters
    ----------
    rho_d : float
        Destination autoregressive parameter.
    rho_o : float
        Origin autoregressive parameter.
    n : int
        Number of spatial units (not :math:`N = n^2`).
    logdet_fn : callable
        Function ``logdet_fn(rho) -> float`` that computes
        :math:`\log|I_n - \rho W|` for a single scalar :math:`\rho`.

    Returns
    -------
    float
        The log-determinant :math:`\log|A|`.

    Notes
    -----
    Cost: two calls to ``logdet_fn``, typically :math:`O(n)` each
    (eigenvalue or Chebyshev method).
    """
    return n * logdet_fn(rho_d) + n * logdet_fn(rho_o)


def kron_eigenvalue_bounds(
    LdtLd: np.ndarray,
    LotLo: np.ndarray,
    omega: np.ndarray,
    sigma2: float,
) -> tuple[float, float]:
    r"""Compute Gershgorin eigenvalue bounds for the Kronecker precision.

    For the precision matrix
    :math:`P = (L_d^\top L_d \otimes L_o^\top L_o) / \sigma^2 + \mathrm{diag}(\omega)`,
    the Gershgorin bounds are computed from the diagonal elements
    without constructing the :math:`N \times N` matrix.

    The diagonal of :math:`P` is:
    :math:`P_{ii} = (L_d^\top L_d)_{i_d, i_d} \cdot (L_o^\top L_o)_{i_o, i_o} / \sigma^2 + \omega_i`

    where :math:`i = i_d + n \cdot i_o` (column-major ordering).

    The off-diagonal row sums are bounded by:
    :math:`R_i = \sum_{j \neq i} |P_{ij}|`

    Since :math:`P` is Kronecker-structured, the off-diagonal sums can
    be computed from the :math:`n \times n` factors without forming the
    :math:`N \times N` matrix.

    Parameters
    ----------
    LdtLd : ndarray of shape (n, n)
        Product :math:`L_d^\top L_d`.
    LotLo : ndarray of shape (n, n)
        Product :math:`L_o^\top L_o`.
    omega : ndarray of shape (N,) where :math:`N = n^2`
        Pólya–Gamma auxiliary variables.
    sigma2 : float
        Residual variance :math:`\sigma^2`.

    Returns
    -------
    lambda_min : float
        Lower bound on the smallest eigenvalue of :math:`P`.
    lambda_max : float
        Upper bound on the largest eigenvalue of :math:`P`.

    Notes
    -----
    Cost: :math:`O(n^2)` for the diagonal computation, avoiding the
    :math:`O(n^4)` cost of constructing the full :math:`N \times N` matrix.

    The bounds are tight for diagonally-dominant matrices, which is
    typical for spatial precision matrices with small :math:`\rho`.
    """
    n = LdtLd.shape[0]
    N = n * n

    # Diagonal of P: diag(LdtLd) ⊗ diag(LotLo) / sigma2 + omega
    diag_LdtLd = np.diag(LdtLd)  # (n,)
    diag_LotLo = np.diag(LotLo)  # (n,)
    # Kronecker product of diagonals: (n^2,) in column-major order
    kron_diag = np.outer(diag_LotLo, diag_LdtLd).ravel(order="F") / sigma2
    P_diag = kron_diag + omega

    # Off-diagonal row sums: for each row i, sum |P_{ij}| for j != i
    # Row sums of |LdtLd| and |LotLo| (excluding diagonal)
    abs_LdtLd = np.abs(LdtLd)
    abs_LotLo = np.abs(LotLo)
    row_sums_LdtLd = abs_LdtLd.sum(axis=1) - np.abs(diag_LdtLd)  # (n,)
    row_sums_LotLo = abs_LotLo.sum(axis=1) - np.abs(diag_LotLo)  # (n,)

    # Gershgorin R_i for Kronecker product:
    # R_i = |diag_LdtLd[id]| * row_sums_LotLo[io] / sigma2
    #     + row_sums_LdtLd[id] * |diag_LotLo[io]| / sigma2
    #     + row_sums_LdtLd[id] * row_sums_LotLo[io] / sigma2
    # where i = id + n * io (column-major)
    # This simplifies to:
    # R_i = (|diag_LdtLd[id]| + row_sums_LdtLd[id]) * row_sums_LotLo[io] / sigma2
    #     + row_sums_LdtLd[id] * |diag_LotLo[io]| / sigma2
    # But more directly:
    # R_i = row_sums_kron[i] where row_sums_kron = row_sums_LotLo[io] * (|diag_LdtLd[id]| + row_sums_LdtLd[id]) / sigma2
    #      + row_sums_LdtLd[id] * |diag_LotLo[io]| / sigma2
    # Actually, the Gershgorin radius for a Kronecker product A ⊗ B is:
    # R_{(i,j)} = |B_{jj}| * R_A_i + |A_{ii}| * R_B_j + R_A_i * R_B_j
    # where R_A_i = sum_{k!=i} |A_{ik}|, R_B_j = sum_{k!=j} |B_{jk}|
    # But for our case P = (LdtLd ⊗ LotLo)/sigma2 + diag(omega),
    # the off-diagonal part is just (LdtLd ⊗ LotLo)/sigma2 minus its diagonal.
    # So R_i = row_sum_kron[i] where:
    # row_sum_kron[(id, io)] = total_row_LdtLd[id] * total_row_LotLo[io] / sigma2
    #                        - |diag_LdtLd[id]| * |diag_LotLo[io]| / sigma2
    # where total_row = sum of |entries| in that row
    total_row_LdtLd = abs_LdtLd.sum(axis=1)  # (n,)
    total_row_LotLo = abs_LotLo.sum(axis=1)  # (n,)

    # R_{(id, io)} = total_row_LdtLd[id] * total_row_LotLo[io] / sigma2
    #              - |diag_LdtLd[id]| * |diag_LotLo[io]| / sigma2
    # (subtract the diagonal contribution which is not off-diagonal)
    R_kron = (
        np.outer(total_row_LotLo, total_row_LdtLd).ravel(order="F")
        - np.outer(np.abs(diag_LotLo), np.abs(diag_LdtLd)).ravel(order="F")
    ) / sigma2

    # Total Gershgorin radius: R_i = R_kron[i] (omega is diagonal, no off-diagonal)
    R = R_kron

    lambda_min = float(np.min(P_diag - R))
    lambda_max = float(np.max(P_diag + R))
    return lambda_min, lambda_max