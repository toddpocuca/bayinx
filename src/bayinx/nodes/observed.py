from typing import Optional

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

from bayinx.core.node import Node
from bayinx.core.types import T


class Observed(Node[T]):
    """
    A container for observed nodes of a probabilistic model.


    # Attributes
    - `obj`: An internal realization of the node.
    - `_filter_spec`: An internal filter specification for `obj`.
    """

    def __init__(
        self, obj: T, filter_spec: Optional[PyTree] = None
    ):
        if filter_spec is None: # Default filter specification
            # Generate empty specification
            filter_spec = jt.map(lambda _: False, obj)

            # Specify array-like leaves
            filter_spec = eqx.tree_at(
                where=lambda obj: obj,
                pytree=filter_spec,
                replace=jt.map(eqx.is_array_like, obj),
            )

        self.obj = obj
        self._filter_spec = filter_spec
