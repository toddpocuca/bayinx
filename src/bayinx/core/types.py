from typing import Generic, Protocol, TypeVar, runtime_checkable

from jaxtyping import PyTree

from bayinx.core.constraint import Constraint

T = TypeVar("T", bound=PyTree)

@runtime_checkable
class HasConstraint(Protocol, Generic[T]):
    """
    Protocol for probabilistic nodes that have constraints.
    """

    obj: T
    _filter_spec: PyTree
    _constraint: Constraint
