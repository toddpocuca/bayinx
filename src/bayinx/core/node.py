import itertools
from typing import Any, Iterable, Iterator, Self

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

from bayinx.core.utils import _extract_obj, _merge_filter_specs


class Node[T: PyTree](eqx.Module):
    """
    A thin wrapper for nodes of a probabilistic model.


    # Attributes
    - `obj`: An internal realization of the node.
    - `_filter_spec`: An internal filter specification for `obj`.
    """

    _byx__obj: T
    _byx__filter_spec: PyTree

    def __init__(self, obj, filter_spec):
        self._byx__obj = obj
        self._byx__filter_spec = filter_spec

    @property
    def filter_spec(self) -> Self:
        """
        An outer filter specification for the full node.
        """
        # Generate empty specification
        node_filter_spec: Self = jt.map(lambda _: False, self)

        # Filter based on inner filter specification for 'obj'
        node_filter_spec = eqx.tree_at(
            lambda node: node._byx__obj,
            node_filter_spec,
            replace=self._byx__filter_spec,
        )

        return node_filter_spec

    # Wrappers around internal dunder methods ----
    def __getattribute__(self, name: str) -> Any:
        # Look up attribute in node
        try:
            return super().__getattribute__(name)
        except AttributeError:
            import bayinx.ops as byo

            # Extract internal object from node
            obj = byo.obj(self)

            # Look up attribute in internal object
            try:
                attr = getattr(obj, name)
            except AttributeError:
                raise AttributeError(
                    f"'{type(self).__name__}' object and its internal "
                    f"'{type(obj).__name__}' have no attribute '{name}'"
                ) from None

            # Wrap method to return nodes
            if callable(attr):
                def node_wrapper(*args, **kwargs):
                    # Unwrap any Node arguments
                    unwrapped_args = [byo.obj(arg) if isinstance(arg, Node) else arg for arg in args]
                    unwrapped_kwargs = {k: (byo.obj(v) if isinstance(v, Node) else v) for k, v in kwargs.items()}

                    result = attr(*unwrapped_args, **unwrapped_kwargs)

                    # Wrap result as a node
                    return Node(result, True)

                return node_wrapper

            # Wrap attribute as a node
            return Node(attr, True)

    def __getitem__(self, key: Any) -> "Node":
        if isinstance(key, Node):
            raise TypeError("Subsetting nodes with nodes is not yet supported.")

        # Extract object and filter_spec
        obj, filter_spec = _extract_obj(self)

        # Subset internally
        new_obj = obj[key]
        if type(obj) is type(filter_spec):
            new_filter_spec = filter_spec[key]
        else:
            new_filter_spec = filter_spec

        # Create new subsetted node
        return type(self)(new_obj, new_filter_spec)

    def __iter__(self) -> Iterator["Node"]:
        # Extract object and filter_spec
        obj, filter_spec = _extract_obj(self)

        # Handle cases where _filter_spec is a scalar (e.g., bool) or 0-d array
        if isinstance(filter_spec, Iterable):
            specs = filter_spec
        else:
            # Broadcast scalar specification
            specs = itertools.repeat(filter_spec)

        for obj_i, spec_i in zip(obj, specs):
            # Create a new node for the current element
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

    def __radd__(self, other: Any) -> "Node":
        return self.__add__(other)

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

    def __rmul__(self, other: Any) -> "Node":
        return self.__mul__(other)

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

    def __rmatmul__(self, other: Any) -> "Node":
        return self.__matmul__(other)

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
        new_obj = self._byx__obj(*args, **kwargs)

        # Return the result wrapped in a new Node
        return Node(new_obj, True)
