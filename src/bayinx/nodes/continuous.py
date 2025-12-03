from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
from jaxtyping import Array, PyTree

from bayinx.constraints import Identity
from bayinx.core.constraint import Constraint
from bayinx.core.types import T
from bayinx.nodes.stochastic import Stochastic


def is_float_like(element: Any) -> bool:
    """
    Check if `element` is float-like.

    The structure of this function is borrowed from the `Equinox` library.
    """
    if hasattr(element, "__jax_array__"):
        element = element.__jax_array__()
    if isinstance(element, (np.ndarray, np.generic)):
        return bool(np.issubdtype(element.dtype, np.floating))
    elif isinstance(element, Array):
        return jnp.issubdtype(element.dtype, jnp.floating)
    else:
        return isinstance(element, float)


class Continuous(Stochastic[T]):
    """
    A container for continuous stochastic nodes of a probabilistic model.


    # Attributes
    - `obj`: An internal realization of the node.
    - `_filter_spec`: An internal filter specification for `obj`.
    - `_constraint`: A constraining transformation.
    """

    _constraint: Constraint


    def __init__(
        self,
        obj: T,
        constraint: Constraint = Identity(),
        filter_spec: Optional[PyTree] = None
    ):
        if filter_spec is None: # Default filter specification
            # Generate empty specification
            filter_spec = jt.map(lambda _: False, obj)

            # Specify float-like leaves
            filter_spec = eqx.tree_at(
                where=lambda obj: obj,
                pytree=filter_spec,
                replace=jt.map(is_float_like, obj),
            )

        self.obj = obj
        self._filter_spec = filter_spec
        self._constraint = constraint
