"""Parameter recovery tests for SpatialProbit.

These tests generate binary outcomes from known parameters and verify that
posterior means recover the spatial and regression parameters within tolerance.

Run with::

    pytest tests/test_spatial_probit.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import SpatialProbit
from .helpers  import (
    SAMPLE_KWARGS,
    make_line_W,
    make_spatial_probit_data,
    W_to_graph,
)

pytestmark = pytest.mark.slow

# Data generating process parameters
M_REGIONS = 10
N_PER_REGION = 25
RHO_TRUE = 0.35
BETA_TRUE = np.array([0.3, 1.0])
SIGMA_A_TRUE = 0.8

ABS_TOL_RHO = 0.30
ABS_TOL_BETA = 0.45
ABS_TOL_SIGMA_A = 0.40


def _fit_spatial_probit(rng):
    W_dense = make_line_W(M_REGIONS)
    W_graph = W_to_graph(W_dense)
    y, X, region_ids = make_spatial_probit_data(
        rng,
        W_dense,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        sigma_a=SIGMA_A_TRUE,
        n_per_region=N_PER_REGION,
    )
    model = SpatialProbit(y=y, X=X, W=W_graph, region_ids=region_ids)
    return model.fit(**SAMPLE_KWARGS), model


@pytest.mark.slow
def test_spatialprobit_recovers_rho(rng):
    """Posterior mean rho should be close to the true rho."""
    idata, _ = _fit_spatial_probit(rng)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_RHO, (
        f"SpatialProbit rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


@pytest.mark.slow
def test_spatialprobit_recovers_beta(rng):
    """Posterior means of beta should be close to true values."""
    idata, _ = _fit_spatial_probit(rng)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SpatialProbit beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


@pytest.mark.slow
def test_spatialprobit_recovers_sigma_a(rng):
    """Posterior mean sigma_a should be close to true value."""
    idata, _ = _fit_spatial_probit(rng)
    sa_hat = float(idata.posterior["sigma_a"].mean())
    assert abs(sa_hat - SIGMA_A_TRUE) < ABS_TOL_SIGMA_A, (
        f"SpatialProbit sigma_a: expected ≈{SIGMA_A_TRUE}, got {sa_hat:.3f}"
    )


@pytest.mark.slow
def test_spatialprobit_fitted_probabilities_in_unit_interval(rng):
    """Posterior mean fitted probabilities should lie in [0, 1]."""
    _, model = _fit_spatial_probit(rng)
    p_hat = model.fitted_probabilities()
    assert np.all((p_hat >= 0.0) & (p_hat <= 1.0))
