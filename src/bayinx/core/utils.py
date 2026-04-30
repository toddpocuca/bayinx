from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

def _extract_shape_params(shape_spec: int | str | tuple[int | str, ...]) -> set[str]:
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
    shape_spec: None | int | str | tuple[int | str, ...],
    shape_values: dict[str, int]
) -> None | tuple[int, ...]:
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

    resolved_spec: list[int] = []
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
