from typing import Any, Callable

import equinox as eqx
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, ArrayLike, PyTree

from bayinx.core.context import Target
from bayinx.core.node import Node
from bayinx.core.utils import _extract_obj

# Public
__all__ = ["exp", "log", "sin", "cos", "tanh", "sigmoid"]

def exp[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node:
    """
    Apply the exponential transformation (jnp.exp) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply exponential
    new_obj = jt.map(lambda x: jnp.exp(x), obj)

    return Node(new_obj, filter_spec)


def log[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node:
    """
    Apply the natural logarithm transformation (jnp.log) to an object.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply logarithm
    new_obj = jt.map(lambda x: jnp.log(x), obj)

    return Node(new_obj, filter_spec)


def sin[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the sine transformation (jnp.sin) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sine
    new_obj = jt.map(lambda x: jnp.sin(x), obj)

    return Node(new_obj, filter_spec)


def cos[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the cosine transformation (jnp.cos) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply cosine
    new_obj = jt.map(lambda x: jnp.cos(x), obj)

    return Node(new_obj, filter_spec)


def tanh[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the hyperbolic tangent transformation (jnp.tanh) to a node.
    """
    obj, filter_spec = _extract_obj(node)


    # Apply tanh
    new_obj = jt.map(lambda x: jnp.tanh(x), obj)

    return Node(new_obj, filter_spec)

def sigmoid[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the sigmoid transformation to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sigmoid
    new_obj = jt.map(jnn.sigmoid, obj)

    return Node(new_obj, filter_spec)

def ilr_inv[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the inverse isometric log-ratio transformation (map to unit simplex).
    """
    obj, filter_spec = _extract_obj(node)

    def leaf_ilr_inv(leaf: Any):
        # Apply constraining transformation ----
        N = leaf.shape[-1]

        # Construct centred basis
        idxs = jnp.arange(1, N + 1)
        scaled_leaf = leaf * jnp.reciprocal(jnp.sqrt(idxs * (idxs + 1)))

        # Compute reverse cumulative sum
        s = jnp.flip(
            jnp.cumsum(
                jnp.flip(
                    scaled_leaf, axis=-1
                ), axis=-1
            ), axis=-1
        )
        s = jnp.pad(s, ((0, 0),) * (s.ndim - 1) + ((0, 1),))

        # Construct zero-sum vector
        z = jnp.concatenate([s[..., 0:1], s[..., 1:] - (idxs * scaled_leaf)], axis=-1)

        # Compute constrained leaf
        constrained = jnn.softmax(z, axis=-1)

        return constrained

    obj = jt.map(leaf_ilr_inv, obj, filter_spec)

    return Node(obj, filter_spec)

def asarray(node: Node[ArrayLike]) -> Node[Array]:
    """
    Cast a 'Node[ArrayLike]' object to 'Node[Array]'.

    Equivalent to 'jax.numpy.asarray' but with nodes.
    """
    # Extract inner object
    node_obj = obj(node)

    # Coerce to array
    node_obj = jnp.asarray(node_obj)

    # Slot in array
    node = eqx.tree_at(
        lambda node: node._byx__obj,
        node,
        node_obj
    )

    return node

def obj[T: PyTree](node: Node[T]) -> T:
    """
    Extract internal object from a node.
    """
    return node._byx__obj

def map(
    f: Callable[..., PyTree | None],
    *args: Node[Any] | Any
) -> Node[PyTree] | None:
    """
    Map a function over the leading axis of the arguments.

    Parameters:
        f: A user-defined function that accepts slices of the input positional arguments.
        args: Additional positional arguments that are sliced and passed to `f`.

    Returns:
        A `Node[PyTree]` whose leaves are stacked with the evaluations of `f` (which reduces to `None` if nothing is returned).
    """
    from bayinx.core.context import _model_context

    # Unwrap any node arguments
    xs = tuple(obj(arg) if isinstance(arg, Node) else arg for arg in args)

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

    return Node(user_results, True)


def fori_loop(
    lower: int | Node[int],
    upper: int | Node[int],
    f: Callable[[int], Node[PyTree] | PyTree | None]
) -> Node[PyTree] | None:
    """
    Loop from `lower` to `upper` with a function `f`.

    Parameters:
        lower: The starting index (inclusive).
        upper: The ending index (exclusive).
        f: A function accepting an integer index `i`.

    Returns:
        A `Node[PyTree]` containing the stacked results of `f` (if any), or None.
    """
    from bayinx.core.context import _model_context

    # Unwrap nodes to get raw integer bounds
    if isinstance(lower, Node):
        lower: int = obj(lower)
    if isinstance(upper, Node):
        upper: int = obj(upper)

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

    return Node(user_results, True)
