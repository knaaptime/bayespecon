r"""JAX-compatible state classes for the Gaussian panel flow Gibbs sampler.

Provides :class:`JAXPanelGaussianState` as an ``equinox.Module`` that
holds JAX arrays and is automatically registered as a PyTree, so it
can be passed through ``@jax.jit`` and ``@eqx.filter_jit`` boundaries
without manual registration.

Mirrors :class:`~bayespecon.samplers.panel_flow._state.PanelGaussianState`
but with JAX arrays instead of numpy arrays.

See Also
--------
bayespecon.samplers.panel_flow._state
    Numpy-based state classes.
bayespecon.samplers.negbin._core
    ``JAXGibbsState`` — the same eqx.Module pattern for the NB sampler.
"""

from __future__ import annotations

import importlib.util

import numpy as np


def _eqx_available() -> bool:
    """Check whether equinox is importable."""
    return importlib.util.find_spec("equinox") is not None


def _jax_available() -> bool:
    """Check whether JAX is importable."""
    return importlib.util.find_spec("jax") is not None


if _eqx_available() and _jax_available():
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    class JAXPanelGaussianState(eqx.Module):
        """JAX-compatible panel Gaussian Gibbs state (equinox Module).

        An ``equinox.Module`` that holds JAX arrays and is automatically
        registered as a PyTree, so it can be passed through
        ``@jax.jit`` and ``@eqx.filter_jit`` boundaries.

        For the Python-loop path, use
        :class:`~bayespecon.samplers.panel_flow._state.PanelGaussianState`
        instead.

        Attributes
        ----------
        eta : jax.Array of shape (n², T)
            Latent field in the original (spatial) basis.
        beta : jax.Array of shape (k,)
            Regression coefficients.
        sigma2_u : jax.Array (scalar)
            Innovation variance.
        sigma2_y : jax.Array (scalar)
            Observation variance.
        rho_d : jax.Array (scalar)
            Destination autoregressive parameter.
        rho_o : jax.Array (scalar)
            Origin autoregressive parameter.
        gamma : jax.Array (scalar)
            Temporal AR(1) parameter.
        """

        eta: jax.Array
        beta: jax.Array
        sigma2_u: jax.Array
        sigma2_y: jax.Array
        rho_d: jax.Array
        rho_o: jax.Array
        gamma: jax.Array

        def to_numpy(self):
            """Convert to a numpy-based PanelGaussianState."""
            from ._state import PanelGaussianState

            return PanelGaussianState(
                eta=np.asarray(self.eta),
                beta=np.asarray(self.beta),
                sigma2_u=float(self.sigma2_u),
                sigma2_y=float(self.sigma2_y),
                rho_d=float(self.rho_d),
                rho_o=float(self.rho_o),
                gamma=float(self.gamma),
            )

        @classmethod
        def from_numpy(cls, state):
            """Create from a numpy-based PanelGaussianState."""
            return cls(
                eta=jnp.asarray(state.eta, dtype=jnp.float64),
                beta=jnp.asarray(state.beta, dtype=jnp.float64),
                sigma2_u=jnp.float64(state.sigma2_u),
                sigma2_y=jnp.float64(state.sigma2_y),
                rho_d=jnp.float64(state.rho_d),
                rho_o=jnp.float64(state.rho_o),
                gamma=jnp.float64(state.gamma),
            )

else:

    class JAXPanelGaussianState:  # type: ignore[no-redef]
        """Stub when equinox/JAX is not installed — should never be instantiated."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "JAX and equinox are required for the JAX panel flow "
                "Gibbs sampler. Install with: pip install jax equinox"
            )
