import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, ArrayLike, Real

from bayinx.core.node import Node
from bayinx.core.utils import _extract_obj


def exp(node: Node) -> Node:
    """
    Apply the exponential transformation (jnp.exp) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Helper function for the single-leaf exponential transform
    def leaf_exp(x: Real[ArrayLike, "..."]) -> Array:
        return jnp.exp(x)

    # Apply exponential
    new_obj = jt.map(leaf_exp, obj)

    return type(node)(new_obj, filter_spec)


def log(node: Node) -> Node:
    """
    Apply the natural logarithm transformation (jnp.log) to a node.
    Handles input value restrictions (must be positive).
    """
    obj, filter_spec = _extract_obj(node)

    # Helper function for the single-leaf log transform
    def leaf_log(x: Real[ArrayLike, "..."]) -> Array:
        return jnp.log(x)

    # Apply logarithm
    new_obj = jt.map(leaf_log, obj)

    return type(node)(new_obj, filter_spec)


def sin(node: Node) -> Node:
    """
    Apply the sine transformation (jnp.sin) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Helper function for the single-leaf sine transform
    def leaf_sin(x: Real[ArrayLike, "..."]) -> Array:
        return jnp.sin(x)

    # Apply sine
    new_obj = jt.map(leaf_sin, obj)

    return type(node)(new_obj, filter_spec)


def cos(node: Node) -> Node:
    """
    Apply the cosine transformation (jnp.cos) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Helper function for the single-leaf cosine transform
    def leaf_cos(x: Real[ArrayLike, "..."]) -> Array:
        return jnp.cos(x)

    # Apply cosine
    new_obj = jt.map(leaf_cos, obj)

    return type(node)(new_obj, filter_spec)


def tanh(node: Node) -> Node:
    """
    Apply the hyperbolic tangent transformation (jnp.tanh) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Helper function for the single-leaf tanh transform
    def leaf_tanh(x: Real[ArrayLike, "..."]) -> Array:
        return jnp.tanh(x)

    # Apply tanh
    new_obj = jt.map(leaf_tanh, obj)

    return type(node)(new_obj, filter_spec)


def sigmoid(node: Node) -> Node:
    """
    Apply the sigmoid (logistic) transformation to a node.
    Sigmoid formula: 1 / (1 + exp(-x))
    """
    obj, filter_spec = _extract_obj(node)

    # Helper function for the single-leaf sigmoid transform
    def leaf_sigmoid(x: Real[ArrayLike, "..."]) -> Array:
        return 1.0 / (1.0 + jnp.exp(-x)) # type: ignore

    # Apply sigmoid
    new_obj = jt.map(leaf_sigmoid, obj)

    return type(node)(new_obj, filter_spec)
