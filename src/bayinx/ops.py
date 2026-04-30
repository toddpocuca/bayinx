from typing import Any, Callable

import jax.lax as lax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree

from bayinx.core.context import Target

# Public

def map(
    f: Callable[..., PyTree | None],
    *xs: Any
) -> PyTree | None:
    """
    Map a function over the leading axis of the arguments.

    Parameters:
        f: A user-defined function that accepts slices of the input positional arguments.
        xs: Additional positional arguments that are sliced and passed to `f`.

    Returns:
        A `PyTree` whose leaves are stacked with the evaluations of `f` (which reduces to `None` if nothing is returned).
    """
    from bayinx.core.context import _model_context

    # Check for existing model context
    within_context = hasattr(_model_context, "target")

    if within_context:
        # Reference the outer model context
        outer_target = _model_context.target

    # Scan to accumulate log-probabilities
    def scanner(carry, x):
        if within_context:
            # Shadow the outer model context in the local scope
            _model_context.target = Target(jnp.array(0.0))

        try:
            # Evaluate user callable (which uses the local scope's model context)
            user_out = f(*x)

            if within_context:
                # Extract the local target density
                local_lp = _model_context.target.value
        finally:
            if within_context:
                # Restore model context to outer scope
                _model_context.target = outer_target

        if within_context:
            # Accumulate local target density
            carry = carry + local_lp

        return carry, user_out

    # Run scan
    total_log_prob, user_results = lax.scan(scanner, 0.0, xs)

    if within_context:
        # Update outer target with accumulated local target
        outer_target += total_log_prob

    # Return unwrapped none when there are no user outputs
    if all(x is None for x in jt.flatten(user_results)):
        return None

    return user_results


def fori_loop(
    lower: int,
    upper: int,
    f: Callable[[int], PyTree | None]
) -> PyTree | None:
    """
    Loop from `lower` to `upper` evaluating a function `f`.

    Parameters:
        lower: The starting index (inclusive).
        upper: The ending index (exclusive).
        f: A function accepting an integer index `i`.

    Returns:
        A `PyTree` whose leaves are stacked with the evaluations of `f` (which reduces to `None` if nothing is returned).
    """
    from bayinx.core.context import _model_context

    # Create the sequence of indices to iterate over
    idxs = jnp.arange(lower, upper)

    # Check for existing model context
    within_context = hasattr(_model_context, "target")

    if within_context:
        # Reference the outer model context
        outer_target = _model_context.target

    # Scan to accumulate log-probabilities
    def scanner(carry, i):
        # Shadow the outer model context in the local scope
        if within_context:
            _model_context.target = Target(jnp.array(0.0))

        try:
            # Evaluate user callable with loop index
            user_out = f(i)

            if within_context:
                # Extract the local target density
                local_lp = _model_context.target.value
            else:
                local_lp = 0.0

        finally:
            if within_context:
                # Restore model context to outer scope
                _model_context.target = outer_target

        if within_context:
            # Accumulate local target density into carry
            carry = carry + local_lp

        return carry, user_out

    # Run scan
    total_log_prob, user_results = lax.scan(scanner, 0.0, idxs)

    if within_context:
        # Update outer target with accumulated local target from the loop
        outer_target += total_log_prob

    # Return unwrapped none when there are no user outputs
    if all(x is None for x in jt.flatten(user_results)):
        return None

    return user_results
