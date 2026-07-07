"""Public API type stub for :mod:`bayespecon`.

The top-level namespace intentionally exposes **only submodules**.  All
public symbols are reached through them, e.g. ``bayespecon.models.SAR``,
``bayespecon.diagnostics.bayes_factor_compare_models``,
``bayespecon.dgp.generate_flow_data``.  Nothing is flattened onto the
top-level ``bayespecon`` namespace.
"""

from . import (
    _ops as _ops,
)
from . import (
    config as config,
)
from . import (
    dgp as dgp,
)
from . import (
    diagnostics as diagnostics,
)
from . import (
    graph as graph,
)
from . import (
    models as models,
)
from . import (
    samplers as samplers,
)
