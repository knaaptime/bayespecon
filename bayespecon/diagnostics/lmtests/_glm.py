r"""GLM (PG-augmented) helpers for Bayesian LM diagnostic tests.

The Pólya-Gamma augmentation of Polson, Scott & Windle (2013) makes the
logistic and negative-binomial likelihoods conditionally Gaussian in the
linear predictor:

.. math::
    p(y_i \mid \psi_i, \cdot) \propto \exp\!\left\{-\tfrac{\omega_i}{2}
        \bigl(\psi_i - \kappa_i/\omega_i\bigr)^2\right\} \cdot p(\omega_i),
    \qquad \omega_i \sim \mathrm{PG}(b_i, \psi_i).

Conditional on :math:`\omega`, this is exactly a weighted Gaussian
regression with pseudo-outcome :math:`\tilde z_i = \kappa_i/\omega_i`
and weights :math:`\Omega = \mathrm{diag}(\omega_i)`.  All Gaussian SAR
LM-test algebra (Anselin 1996; Doğan, Taşpınar & Bera 2021) therefore
carries over with the substitutions

* residual :math:`e = y - X\beta \;\longrightarrow\; \tilde e = \tilde z - X\beta`
* response lag :math:`Wy \;\longrightarrow\; W\tilde z`
* scale :math:`\sigma^2 I \;\longrightarrow\; \Omega^{-1}`
* projection :math:`M_X \;\longrightarrow\; M_X^\Omega
  = I - X(X^\top \Omega X)^{-1} X^\top \Omega`.

We marginalise :math:`\omega` analytically by replacing it with its
conditional mean given the current draw of :math:`\beta` (and
:math:`\alpha` for NB):

.. math::
    E[\omega_i \mid \psi_i] = \frac{b_i}{2\psi_i}\,\tanh(\psi_i/2),
    \qquad \lim_{\psi_i\to 0} = b_i/4.

This is exact for the conditional expectation of the score (the score
is linear in :math:`\omega`) and avoids any auxiliary PG sampling.

Parameter conventions
---------------------
* Logit:  :math:`b_i = 1`,  :math:`\kappa_i = y_i - 1/2`.
* NegBin: :math:`b_i = y_i + \alpha`,  :math:`\kappa_i = (y_i - \alpha)/2`,
  with :math:`\alpha` the dispersion of the NB-2 logistic parameterisation.
"""

from __future__ import annotations

import numpy as np

from .core import _get_posterior_draws

# ---------------------------------------------------------------------------
# PG conditional-mean weight
# ---------------------------------------------------------------------------


def _pg_mean_weight(psi: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    r"""Conditional mean :math:`E[\omega \mid \psi]` for :math:`\omega \sim \mathrm{PG}(b,\psi)`.

    .. math::
        E[\omega \mid \psi] = \frac{b}{2\psi}\,\tanh(\psi/2),
        \qquad \lim_{\psi \to 0} = b/4.

    Computed via ``0.5 * b * tanh(psi/2) / psi`` with a numerically safe
    branch at :math:`\psi \approx 0` using the Taylor expansion
    :math:`\tanh(x)/x \approx 1 - x^2/3 + 2x^4/15`.
    """
    psi = np.asarray(psi, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    b_arr = np.broadcast_to(b_arr, psi.shape)
    half = 0.5 * psi
    small = np.abs(psi) < 1e-6
    # Default branch
    with np.errstate(divide="ignore", invalid="ignore"):
        w = b_arr * np.tanh(half) / (2.0 * psi)
    # Taylor: tanh(x)/x ≈ 1 - x²/3 + 2x⁴/15  →  b/(4) * (1 - psi²/12 + ...)
    if np.any(small):
        x2 = half[small] ** 2
        w_small = 0.25 * b_arr[small] * (1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0)
        w[small] = w_small
    return w


# ---------------------------------------------------------------------------
# Per-model working response builders
# ---------------------------------------------------------------------------


def _glm_eta_draws(model, beta_draws: np.ndarray) -> np.ndarray:
    """Linear predictor draws ``eta = X @ beta``.

    Shape: ``(draws, n)``.
    """
    X = model._X
    return beta_draws @ X.T


def _logit_working_response(
    model, beta_draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Working response, working residual and PG weights for a logit model.

    Returns
    -------
    z_tilde : (draws, n) array
        Pseudo-outcome :math:`\kappa_i/\omega_i = (y_i - 1/2)/\omega_i`.
    e_tilde : (draws, n) array
        Working residual :math:`\tilde z - X\beta`.
    omega : (draws, n) array
        Conditional-mean PG weight :math:`E[\omega_i\mid\psi_i]` with
        :math:`b_i = 1`.
    """
    y = np.asarray(model._y, dtype=np.float64)
    psi = _glm_eta_draws(model, beta_draws)  # (draws, n)
    omega = _pg_mean_weight(psi, b=1.0)  # (draws, n)
    kappa = y - 0.5  # (n,)
    z_tilde = kappa[None, :] / omega
    e_tilde = z_tilde - psi
    return z_tilde, e_tilde, omega


def _negbin_working_response(
    model, beta_draws: np.ndarray, alpha_draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Working response, working residual and PG weights for an NB-2 model.

    Uses the logistic parameterisation :math:`\psi_i = \log(\mu_i/\alpha)`
    so :math:`\psi_i = x_i'\beta - \log\alpha`.  PG shape is
    :math:`b_i = y_i + \alpha` and :math:`\kappa_i = (y_i - \alpha)/2`.

    Parameters
    ----------
    model : NegBin-like
        Must expose ``_X``, ``_y`` (raw counts).
    beta_draws : (draws, k) array
    alpha_draws : (draws,) array

    Returns
    -------
    z_tilde, e_tilde, omega : each (draws, n)
    """
    y = np.asarray(model._y, dtype=np.float64)
    eta = _glm_eta_draws(model, beta_draws)  # (draws, n) — log-mu
    alpha = alpha_draws[:, None]  # (draws, 1)
    psi = eta - np.log(alpha)  # logistic-parameterised linear predictor
    b = y[None, :] + alpha  # (draws, n)
    kappa = 0.5 * (y[None, :] - alpha)  # (draws, n)
    omega = _pg_mean_weight(psi, b=b)
    z_tilde = kappa / omega
    e_tilde = z_tilde - psi
    return z_tilde, e_tilde, omega


def glm_working_response(
    model, beta_draws: np.ndarray, idata
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the correct working-response builder based on ``model._model_type``."""
    mt = getattr(model, "_model_type", None)
    if mt == "logit":
        return _logit_working_response(model, beta_draws)
    if mt == "negbin":
        alpha_draws = _get_posterior_draws(idata, "alpha").ravel()
        return _negbin_working_response(model, beta_draws, alpha_draws)
    raise NotImplementedError(
        f"GLM LM tests are not implemented for model_type={mt!r}. "
        "Supported: 'logit', 'negbin'."
    )


# ---------------------------------------------------------------------------
# Weighted M_X projection helpers
# ---------------------------------------------------------------------------


def _weighted_mx_quadratic(X: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    r"""Compute :math:`v^\top \Omega M_X^\Omega v` where
    :math:`M_X^\Omega = I - X(X^\top \Omega X)^{-1} X^\top \Omega`.

    Equivalently :math:`v^\top \Omega v - (X^\top \Omega v)^\top
    (X^\top \Omega X)^{-1} (X^\top \Omega v)` — the residual sum of
    squares from a weighted least-squares regression of ``v`` on ``X``.
    """
    Wv = w * v
    XtWv = X.T @ Wv
    XtWX = X.T @ (w[:, None] * X)
    sol, *_ = np.linalg.lstsq(XtWX, XtWv, rcond=None)
    return float(v @ Wv) - float(XtWv @ sol)


def _weighted_mx_cross(
    X: np.ndarray, U: np.ndarray, V: np.ndarray, w: np.ndarray
) -> np.ndarray:
    r"""Generalises :func:`_weighted_mx_quadratic` to arbitrary left/right factors.

    Returns :math:`U^\top \Omega V - U^\top \Omega X (X^\top \Omega X)^{-1}
    X^\top \Omega V`.
    """
    Wv = w[:, None] * V if V.ndim == 2 else w * V
    Wu = w[:, None] * U if U.ndim == 2 else w * U
    XtWX = X.T @ (w[:, None] * X)
    XtWU = X.T @ Wu
    XtWV = X.T @ Wv
    UtWV = U.T @ Wv
    sol, *_ = np.linalg.lstsq(XtWX, XtWV, rcond=None)
    return UtWV - XtWU.T @ sol


def _weighted_T_ww(W_sp, w: np.ndarray, *, tr_W2: float | None = None) -> float:
    r"""Weighted analog of :math:`T_{WW} = \mathrm{tr}(W^2 + W^\top W)`
    for PG-augmented GLM LM-Error tests.

    Derivation: for the augmented working model :math:`\tilde z = X\beta + u`
    with :math:`u \sim \mathcal{N}(0, \Omega^{-1})`, the LM-Error score is
    :math:`S = u^\top \Omega W u` and its variance under H₀ is

    .. math::
        \mathrm{Var}(S) = \mathrm{tr}\!\bigl((\Omega W \Omega^{-1})^2\bigr)
            + \mathrm{tr}\!\bigl(\Omega W \Omega^{-1} W^\top\bigr)
            = \mathrm{tr}(W^2) + \sum_{i,j}\!\frac{\omega_i}{\omega_j}\, W_{ij}^2.

    The first term reduces to :math:`\mathrm{tr}(W^2) = \sum_{ij} W_{ij} W_{ji}`
    because the diagonal-similarity weights cancel; only the
    :math:`\mathrm{tr}(W^\top W)` analog acquires the weight ratio.

    Parameters
    ----------
    W_sp : scipy.sparse matrix
    w : (n,) array
        Per-observation working weights :math:`\omega_i`.
    tr_W2 : float, optional
        Pre-computed :math:`\mathrm{tr}(W^2) = \sum_{ij} W_{ij} W_{ji}`.
        If ``None`` it is computed from ``W_sp``.
    """
    import scipy.sparse as sp

    W = sp.csr_matrix(W_sp)
    if tr_W2 is None:
        tr_W2 = float(W.multiply(W.T).sum())
    # Sum_{ij} (w_i / w_j) W_ij^2  via COO iteration
    Wsq = W.multiply(W).tocoo()
    omega_ratio = w[Wsq.row] / w[Wsq.col]
    weighted_F2 = float((Wsq.data * omega_ratio).sum())
    return tr_W2 + weighted_F2
