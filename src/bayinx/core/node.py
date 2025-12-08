from typing import Any, Generic, Iterator, Self

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

from bayinx.core.types import T
from bayinx.core.utils import _extract_obj, _merge_filter_specs


class Node(eqx.Module, Generic[T]):
    """
    A thin wrapper for nodes of a probabilistic model.


    # Attributes
    - `obj`: An internal realization of the node.
    - `_filter_spec`: An internal filter specification for `obj`.
    """

    obj: T
    _filter_spec: PyTree

    def __init__(self, obj, filter_spec):
        self.obj = obj
        self._filter_spec = filter_spec

    @property
    def filter_spec(self) -> Self:
        """
        An outer filter specification for the full node.
        """
        # Generate empty specification
        node_filter_spec: Self = jt.map(lambda _: False, self)

        # Filter based on inner filter specification for 'obj'
        node_filter_spec = eqx.tree_at(
            lambda node: node.obj,
            node_filter_spec,
            replace=self._filter_spec,
        )

        return node_filter_spec

    # Wrappers around internal dunder methods ----
    def __getitem__(self, key: Any) -> "Node":
        if isinstance(key, Node):
            raise TypeError("Subsetting nodes with nodes is not yet supported.")

        # Subset internally
        new_obj = self.obj[key]
        if type(self.obj) is type(self._filter_spec):
            new_filter_spec = self._filter_spec[key]
        else:
            new_filter_spec = self._filter_spec

        # Create new subsetted node
        return type(self)(new_obj, new_filter_spec)

    def __iter__(self) -> Iterator["Node"]:
        for obj_i, spec_i in zip(self.obj, self._filter_spec):
            # Create a new Node for the current element
            yield Node(obj_i, spec_i)

    ## Arithmetic ----
    def __add__(self, other: Any) -> "Node":
        # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform addition
        new_obj = lhs_obj + rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __sub__(self, other: Any) -> "Node":
        # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform subtraction
        new_obj = lhs_obj - rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __mul__(self, other: Any) -> "Node":
        # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform multiplication
        new_obj = lhs_obj * rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __matmul__(self, other: Any) -> "Node":
        # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform matrix multiplication
        new_obj = lhs_obj @ rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __truediv__(self, other: Any) -> "Node":
        # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform true division
        new_obj = lhs_obj / rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __floordiv__(self, other: Any) -> "Node":
        # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform floor division
        new_obj = lhs_obj // rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __pow__(self, other: Any) -> "Node":
         # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform floor division
        new_obj = lhs_obj ** rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __mod__(self, other: Any) -> "Node":
        # Extract internal objects and their filter specifications
        lhs_obj, lhs_filter_spec = _extract_obj(self)
        rhs_obj, rhs_filter_spec = _extract_obj(other)

        # Perform modulus
        new_obj = lhs_obj % rhs_obj

        # Merge filter specifications
        new_filter_spec = _merge_filter_specs(
            [lhs_filter_spec, rhs_filter_spec],
            [lhs_obj, rhs_obj],
            new_obj
        )

        return Node(new_obj, new_filter_spec)

    def __call__(self, *args, **kwargs) -> "Node":
        # Unpack positional arguments using a list comprehension
        args = [_extract_obj(arg)[0] for arg in args]

        # Unpack keyword arguments using a dictionary comprehension
        kwargs = {k: _extract_obj(v)[0] for k, v in kwargs.items()}

        # Call the internal object's __call__ method
        new_obj = self.obj(*args, **kwargs)

        # Return the result wrapped in a new Node
        return Node(new_obj, True)
