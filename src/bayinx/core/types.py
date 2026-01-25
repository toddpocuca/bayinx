from typing import Protocol, runtime_checkable

from jaxtyping import Array, ArrayLike, PyTree

from bayinx.core.constraint import Constraint
from bayinx.core.node import Node

ArrayObject = ArrayLike | Node[Array] | Node[ArrayLike]

@runtime_checkable
class HasConstraint[T: PyTree](Protocol):
    """
    Protocol for probabilistic nodes that have constraints.
    """

    _byx__obj: T
    _byx__filter_spec: PyTree
    _byx__constraint: Constraint
