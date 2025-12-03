
from typing import Any, Dict, List, Optional, Tuple

import jax.tree as jt
from jaxtyping import PyTree


def _extract_shape_params(shape_spec: int | str | Tuple[int | str, ...]) -> set[str]:
    """
    Extract parameter names from shape specification.
    """
    params = set()
    if isinstance(shape_spec, str):
        params.add(shape_spec)
    elif isinstance(shape_spec, tuple):
        for item in shape_spec:
            if isinstance(item, str):
                params.add(item)
    #
    return params

def _resolve_shape_spec(
    shape_spec: None | int | str | Tuple[int | str, ...],
    shape_values: Dict[str, int]
) -> None | Tuple[int, ...]:
    """
    Replaces named dimensions in a shape specification with their integer or tuple values.

    # Example
    For `shape_values = {'k': 5, 's': (3,2,1)}`:
        `('k', 4, 's')` --> `(5, 4, 3, 2, 1)`
    """
    if shape_spec is None:
        return None

    # Coerce to tuple for uniform processing
    if isinstance(shape_spec, (int, str)):
        shape_spec = (shape_spec,)

    resolved_spec: List[int] = []
    for dim in shape_spec:
        if isinstance(dim, str):
            if dim in shape_values:
                resolved_value = shape_values[dim] # Grab initialized value

                if isinstance(resolved_value, int):
                    # Scalar shape dimension (e.g., 'k' -> 3)
                    resolved_spec.append(resolved_value)
                elif isinstance(resolved_value, tuple):
                    # Packed shape dimension (e.g., 'shape' -> (3, 2, 1))
                    resolved_spec.extend(resolved_value)
                else:
                    raise TypeError(f"Shape parameter '{dim}' resolved to an unsupported type: {type(resolved_value).__name__}")
            else: # dim not in shape_values
                raise TypeError(f"Shape parameter '{dim}' was not initialized with a value.")
        elif isinstance(dim, int):
            # Literal integer (e.g., 3 -> 3)
            resolved_spec.append(dim)
        else:
            raise TypeError(f"Shape parameter {dim} was incorrectly specified (must be 'int' or 'str', got '{type(dim).__name__}').")

    return tuple(resolved_spec)

def _extract_obj(x: Any) -> Tuple[Any, Any]:
    """
    Extract the object and its (potentially implicit) filter specification.
    """
    from bayinx.core.node import Node

    if isinstance(x, Node):
        obj = x.obj
        filter_spec = x._filter_spec
    else:
        obj: Any = x # type: ignore
        filter_spec = True # implicit filter specification

    return (obj, filter_spec)


def _merge_filter_specs(
    filter_specs: List[PyTree],
    objs: Optional[List[PyTree]] = None,
    obj: Optional[PyTree] = None
) -> PyTree:
    """
    Merge filter specifications that share the type of `obj`.

    If `obj` and `objs` are not provided then `filter_specs` will be merged as is.
    """

    def _merge(*args):
        return all(args)

    if objs is None and obj is None:
        filter_spec: Any = jt.map(_merge, *filter_specs) # type: ignore
    else:
        selected_specs: List[PyTree] = []

        # Include filter specs whose objects share the correct type
        for cur_obj, cur_spec in zip(objs, filter_specs): # type: ignore
            if type(obj) is type(cur_obj):
                selected_specs.append(cur_spec)

        if len(selected_specs) != 0:
            filter_spec = jt.map(_merge, *selected_specs)
        else:
            filter_spec = True

    return filter_spec
